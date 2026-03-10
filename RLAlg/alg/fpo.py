from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from RLAlg.ode_solver.euler_ode_solver import EulerODESolver

from ..nn.steps import FPOStep, ValueStep

NNMODEL = nn.Module

class SuperviseTarget(Enum):
    Velocity = 0
    Noise = 1
    
class TimeStepSamplerStrategy(Enum):
    Discrete = 0
    Continuous = 1


class TimeStepSampler:
    @staticmethod
    def sample(
        strategy: TimeStepSamplerStrategy,
        sample_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        flow_steps: int | None = None,
    ) -> torch.Tensor:
        time_shape = (*sample_shape, 1)

        if strategy == TimeStepSamplerStrategy.Continuous:
            return torch.rand(time_shape, dtype=dtype, device=device)

        if strategy == TimeStepSamplerStrategy.Discrete:
            if flow_steps is None or flow_steps <= 0:
                raise ValueError(
                    "flow_steps must be > 0 when using discrete timestep sampling, "
                    f"got {flow_steps}."
                )
            step_ids = torch.randint(1, flow_steps + 1, time_shape, device=device)
            return step_ids.to(dtype=dtype) / float(flow_steps)

        raise ValueError(f"Unsupported timestep sampler strategy: {strategy}.")


class FPO:
    supervise_target: SuperviseTarget = SuperviseTarget.Noise
    time_step_sampler: TimeStepSamplerStrategy = TimeStepSamplerStrategy.Continuous

    @staticmethod
    def _validate_observation_batch_dims(
        obs: torch.Tensor | dict[str, torch.Tensor],
        batch_dims: tuple[int, ...],
    ) -> None:
        if torch.is_tensor(obs):
            if obs.shape[: len(batch_dims)] != batch_dims:
                raise ValueError(
                    "obs and action batch dims must match, "
                    f"got {tuple(obs.shape[: len(batch_dims)])} and {tuple(batch_dims)}."
                )
            return

        if isinstance(obs, dict):
            for key, value in obs.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"obs[{key!r}] must be a torch.Tensor, got {type(value)}."
                    )
                if value.shape[: len(batch_dims)] != batch_dims:
                    raise ValueError(
                        "obs and action batch dims must match, "
                        f"got {tuple(value.shape[: len(batch_dims)])} and {tuple(batch_dims)} for key {key!r}."
                    )
            return

        raise TypeError(f"obs must be a torch.Tensor or dict[str, torch.Tensor], got {type(obs)}.")

    @staticmethod
    def _expand_observations(
        obs: torch.Tensor | dict[str, torch.Tensor],
        batch_dims: tuple[int, ...],
        n_samples_per_action: int,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        insert_dim = len(batch_dims)

        if torch.is_tensor(obs):
            feature_shape = obs.shape[insert_dim:]
            expand_shape = (*batch_dims, n_samples_per_action, *feature_shape)
            return obs.unsqueeze(insert_dim).expand(expand_shape)

        expanded_obs: dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            feature_shape = value.shape[insert_dim:]
            expand_shape = (*batch_dims, n_samples_per_action, *feature_shape)
            expanded_obs[key] = value.unsqueeze(insert_dim).expand(expand_shape)
        return expanded_obs

    @staticmethod
    def _expand_advantages(
        advantages: torch.Tensor,
        ratio_shape: torch.Size,
    ) -> torch.Tensor:
        if advantages.shape == ratio_shape:
            return advantages

        if advantages.shape == ratio_shape[:-1]:
            return advantages.unsqueeze(-1)

        if advantages.shape == (*ratio_shape[:-1], 1):
            return advantages

        raise ValueError(
            "advantages must match the ratio shape or the per-action batch shape, "
            f"got {tuple(advantages.shape)} and {tuple(ratio_shape)}."
        )

    @staticmethod
    def _extract_prediction(model_output: ValueStep | torch.Tensor | tuple[Any, ...]) -> torch.Tensor:
        if isinstance(model_output, tuple):
            if len(model_output) == 0:
                raise ValueError("policy output tuple must be non-empty.")
            model_output = model_output[0]

        if isinstance(model_output, ValueStep):
            prediction = model_output.value
        elif torch.is_tensor(model_output):
            prediction = model_output
        else:
            raise TypeError(
                "policy output must be ValueStep or Tensor, "
                f"got {type(model_output)}."
            )
        return prediction

    @staticmethod
    def _call_policy(
        policy: NNMODEL,
        obs: torch.Tensor | dict[str, torch.Tensor],
        current_action: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        call_errors: list[TypeError] = []
        call_attempts = (
            lambda: policy(obs, current_action, t=t),
            lambda: policy(obs, current_action, t),
            lambda: policy(obs, current_action),
            lambda: policy(obs),
        )

        for call_fn in call_attempts:
            try:
                return FPO._extract_prediction(call_fn())
            except TypeError as error:
                call_errors.append(error)

        raise TypeError(
            "Could not call policy with supported signatures: "
            "(obs, action, t=...), (obs, action, t), (obs, action), or (obs)."
        ) from call_errors[-1]

    @staticmethod
    def compute_cmf_loss(
        policy: NNMODEL,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if action.ndim < 1:
            raise ValueError(
                f"action must have at least 1 dim with trailing action_dim, got {tuple(action.shape)}."
            )

        action_dim = action.shape[-1]
        batch_dims = action.shape[:-1]
        FPO._validate_observation_batch_dims(obs, batch_dims)
        sample_shape = eps.shape[:-1]
        flow_shape = (*sample_shape, action_dim)
        time_shape = (*sample_shape, 1)

        if sample_shape[:-1] != batch_dims:
            raise ValueError(
                "eps batch dims must match action batch dims, "
                f"got {tuple(sample_shape[:-1])} and {tuple(batch_dims)}."
            )
        if eps.shape != flow_shape:
            raise ValueError(f"eps must have shape {flow_shape}, got {tuple(eps.shape)}.")
        eps = eps.to(dtype=action.dtype, device=action.device)

        if t.shape != time_shape:
            raise ValueError(f"t must have shape {time_shape}, got {tuple(t.shape)}.")
        t = t.to(dtype=action.dtype, device=action.device)

        action_samples = action.unsqueeze(-2).expand(flow_shape)
        obs_samples = FPO._expand_observations(obs, batch_dims, sample_shape[-1])
        x_t = t * eps + (1.0 - t) * action_samples

        network_pred = FPO._call_policy(policy, obs_samples, x_t, t)
        if network_pred.shape != flow_shape:
            raise ValueError(
                "policy prediction shape must match x_t shape, "
                f"got {tuple(network_pred.shape)} and {flow_shape}."
            )

        if FPO.supervise_target == SuperviseTarget.Velocity:
            velocity_gt = eps - action_samples
            loss_per_sample = ((network_pred - velocity_gt) ** 2).mean(dim=-1)
        elif FPO.supervise_target == SuperviseTarget.Noise:
            x0_pred = x_t - t * network_pred
            x1_pred = x0_pred + network_pred
            loss_per_sample = ((eps - x1_pred) ** 2).mean(dim=-1)
        else:
            raise ValueError(
                "supervise_target must be one of {SuperviseTarget.Velocity, SuperviseTarget.Noise}, "
                f"got {FPO.supervise_target}."
            )

        return loss_per_sample

    @staticmethod
    @torch.no_grad()
    def sample_actions(
        policy: NNMODEL,
        obs: torch.Tensor | dict[str, torch.Tensor],
        init_noise: torch.Tensor,
        flow_steps: int,
        deterministic: bool = False,
    ) -> FPOStep:
        action, action_path = EulerODESolver.denoise(policy, flow_steps,
                                                     obs, init_noise, deterministic)
        
        return FPOStep(action=action, action_path=action_path)
    
    @staticmethod
    @torch.no_grad()
    def sample_actions_with_cmf_info(
        policy: NNMODEL,
        obs: torch.Tensor | dict[str, torch.Tensor],
        init_noise: torch.Tensor,
        flow_steps: int,
        deterministic: bool = False,
        n_samples_per_action: int = 10
    ) -> FPOStep:
        action, action_path = EulerODESolver.denoise(
            policy,
            flow_steps,
            obs,
            init_noise,
            deterministic,
        )
        if n_samples_per_action <= 0:
            raise ValueError(f"n_samples_per_action must be > 0, got {n_samples_per_action}.")

        action_dim = action.shape[-1]
        batch_dims = action.shape[:-1]
        sample_shape = (*batch_dims, n_samples_per_action)
        flow_shape = (*sample_shape, action_dim)

        eps = torch.randn(flow_shape, dtype=action.dtype, device=action.device)
        t = TimeStepSampler.sample(
            strategy=FPO.time_step_sampler,
            sample_shape=sample_shape,
            dtype=action.dtype,
            device=action.device,
            flow_steps=flow_steps,
        )
        cmf_loss = FPO.compute_cmf_loss(
            policy=policy,
            obs=obs,
            action=action,
            eps=eps,
            t=t,
        )

        return FPOStep(
            action=action,
            action_path=action_path,
            eps=eps,
            time_step=t,
            init_cmf_loss=cmf_loss,
        )
        
    @staticmethod
    def compute_policy_loss(policy: NNMODEL,
                            observations: torch.Tensor|dict[str, torch.Tensor],
                            actions: torch.Tensor,
                            eps: torch.Tensor,
                            t: torch.Tensor,
                            init_cmf_loss: torch.Tensor,
                            advantages: torch.Tensor,
                            clip_ratio: float,
                            average_losses_before_exp: bool = False,
                            ratio_clip_range: tuple[float, float] | None = (-3.0, 3.0),
    ) -> dict[str, torch.Tensor]:
        current_cmf_loss = FPO.compute_cmf_loss(
            policy=policy,
            obs=observations,
            action=actions,
            eps=eps,
            t=t,
        )

        if current_cmf_loss.shape != init_cmf_loss.shape:
            raise ValueError(
                "current_cmf_loss and init_cmf_loss must have the same shape, "
                f"got {tuple(current_cmf_loss.shape)} and {tuple(init_cmf_loss.shape)}."
            )

        current_cmf_loss = current_cmf_loss.to(dtype=actions.dtype, device=actions.device)
        init_cmf_loss = init_cmf_loss.to(dtype=actions.dtype, device=actions.device)
        advantages = advantages.to(dtype=actions.dtype, device=actions.device)

        log_ratio = init_cmf_loss - current_cmf_loss
        if average_losses_before_exp:
            ratio = torch.exp(log_ratio.mean(dim=-1, keepdim=True))
        else:
            if ratio_clip_range is not None:
                min_log_ratio, max_log_ratio = ratio_clip_range
                log_ratio = torch.clamp(log_ratio, min=min_log_ratio, max=max_log_ratio)
            ratio = torch.exp(log_ratio)

        advantages = FPO._expand_advantages(advantages, ratio.shape)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        surrogate_loss1 = ratio * advantages
        surrogate_loss2 = clipped_ratio * advantages
        loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()

        with torch.no_grad():
            kl_divergence = (current_cmf_loss - init_cmf_loss).mean()
            clip_fraction = (torch.abs(ratio - 1.0) > clip_ratio).to(dtype=actions.dtype).mean()

        return {
            "loss": loss,
            "policy_loss": loss,
            "ratio": ratio.mean(),
            "ratio_min": ratio.min(),
            "ratio_max": ratio.max(),
            "clip_fraction": clip_fraction,
            "kl_divergence": kl_divergence,
            "cmf_loss": current_cmf_loss.mean(),
            "surrogate_loss1": surrogate_loss1.mean(),
            "surrogate_loss2": surrogate_loss2.mean(),
        }

    @staticmethod
    def _validate_same_shape(name_a: str, tensor_a: torch.Tensor, name_b: str, tensor_b: torch.Tensor) -> None:
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"{name_a} and {name_b} must have the same shape, got {tuple(tensor_a.shape)} and {tuple(tensor_b.shape)}."
            )
            
    @staticmethod
    def compute_value_loss(
        value_model: NNMODEL,
        observations: torch.Tensor|dict[str, torch.Tensor],
        returns: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        step: ValueStep = value_model(observations)
        values = step.value
        FPO._validate_same_shape("values", values, "returns", returns)
        loss = 0.5 * ((returns - values) ** 2).mean()
        return {
            "loss": loss
        }
            
    @staticmethod
    def compute_clipped_value_loss(
        value_model: NNMODEL,
        observations: torch.Tensor|dict[str, torch.Tensor],
        values_hat: torch.Tensor,
        returns: torch.Tensor,
        clip_ratio: float
    ) -> dict[str, torch.Tensor]:
        step: ValueStep = value_model(observations)
        values = step.value
        FPO._validate_same_shape("values", values, "values_hat", values_hat)
        FPO._validate_same_shape("values", values, "returns", returns)

        loss_unclipped = 0.5 * (returns - values) ** 2

        values_clipped = values_hat + torch.clamp(values - values_hat, -clip_ratio, clip_ratio)
        loss_clipped = 0.5 * (returns - values_clipped) ** 2

        loss = torch.max(loss_unclipped, loss_clipped)
        loss = loss.mean()

        return {
            "loss": loss
        }
