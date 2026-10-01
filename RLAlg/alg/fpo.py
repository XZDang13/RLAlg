from enum import Enum
import math
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
    """Flow policy optimization with t=0 at the action and t=1 at the noise.

    The model always predicts dx/dt = noise - action. ``Noise`` selects an
    epsilon reconstruction loss on that velocity, not a noise-output model.
    """
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
        # Loss evaluation and action integration must use identical calling
        # conventions, including keyword-only `time` and `t` arguments.
        return EulerODESolver._call_model(policy, obs, current_action, t)

    @staticmethod
    def compute_cmf_loss(
        policy: NNMODEL,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        *,
        supervise_target: SuperviseTarget | None = None,
    ) -> torch.Tensor:
        if action.ndim < 1 or action.numel() == 0 or not action.is_floating_point():
            raise ValueError(
                f"action must have at least 1 dim with trailing action_dim, got {tuple(action.shape)}."
            )

        action_dim = action.shape[-1]
        batch_dims = action.shape[:-1]
        FPO._validate_observation_batch_dims(obs, batch_dims)
        if eps.ndim != action.ndim + 1 or eps.shape[-2] <= 0:
            raise ValueError("eps must include a non-empty Monte Carlo sample dimension.")
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
        if not torch.isfinite(t).all() or torch.any((t < 0) | (t > 1)):
            raise ValueError("Flow timesteps must be finite and in [0, 1].")

        action_samples = action.unsqueeze(-2).expand(flow_shape)
        obs_samples = FPO._expand_observations(obs, batch_dims, sample_shape[-1])
        x_t = t * eps + (1.0 - t) * action_samples

        network_pred = FPO._call_policy(policy, obs_samples, x_t, t)
        if network_pred.shape != flow_shape:
            raise ValueError(
                "policy prediction shape must match x_t shape, "
                f"got {tuple(network_pred.shape)} and {flow_shape}."
            )

        target = FPO.supervise_target if supervise_target is None else supervise_target
        if target == SuperviseTarget.Velocity:
            velocity_gt = eps - action_samples
            loss_per_sample = ((network_pred - velocity_gt) ** 2).mean(dim=-1)
        elif target == SuperviseTarget.Noise:
            noise_pred = x_t + (1.0 - t) * network_pred
            loss_per_sample = ((eps - noise_pred) ** 2).mean(dim=-1)
        else:
            raise ValueError(
                "supervise_target must be one of {SuperviseTarget.Velocity, SuperviseTarget.Noise}, "
                f"got {target}."
            )

        return loss_per_sample

    @staticmethod
    @torch.no_grad()
    def sample_actions(
        policy: NNMODEL,
        obs: torch.Tensor | dict[str, torch.Tensor],
        init_noise: torch.Tensor,
        flow_steps: int,
        deterministic: bool = True,
    ) -> FPOStep:
        """Integrate the supplied noise; fresh initial noise provides exploration.

        By default the resulting action is the endpoint of the Euler path.
        ``deterministic=False`` explicitly enables the solver's extra jitter.
        """
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
        deterministic: bool = True,
        n_samples_per_action: int = 10,
        *,
        supervise_target: SuperviseTarget | None = None,
        time_step_sampler: TimeStepSamplerStrategy | None = None,
    ) -> FPOStep:
        if n_samples_per_action <= 0:
            raise ValueError(f"n_samples_per_action must be > 0, got {n_samples_per_action}.")
        action, action_path = EulerODESolver.denoise(
            policy,
            flow_steps,
            obs,
            init_noise,
            deterministic,
        )
        action_dim = action.shape[-1]
        batch_dims = action.shape[:-1]
        sample_shape = (*batch_dims, n_samples_per_action)
        flow_shape = (*sample_shape, action_dim)

        eps = torch.randn(flow_shape, dtype=action.dtype, device=action.device)
        t = TimeStepSampler.sample(
            strategy=FPO.time_step_sampler if time_step_sampler is None else time_step_sampler,
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
            supervise_target=supervise_target,
        )

        return FPOStep(
            action=action,
            action_path=action_path,
            eps=eps,
            time_step=t,
            init_cmf_loss=cmf_loss,
        )

    @staticmethod
    def compute_policy_loss(
        policy: NNMODEL,
        observations: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        init_cmf_loss: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float,
        average_losses_before_exp: bool = True,
        ratio_clip_range: tuple[float, float] | None = (-3.0, 3.0),
        *,
        supervise_target: SuperviseTarget | None = None,
    ) -> dict[str, torch.Tensor]:
        """Use one ratio per action: exp(mean_i(old_loss_i - new_loss_i)).

        See Algorithm 1 of https://arxiv.org/abs/2507.21053. The same stored
        actions, noise/time pairs and loss target must be used for both losses.
        Setting average_losses_before_exp=False retains the legacy ablation.
        """
        if not math.isfinite(clip_ratio) or clip_ratio < 0:
            raise ValueError("clip_ratio must be finite and non-negative.")
        if ratio_clip_range is not None:
            lower, upper = ratio_clip_range
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper and lower <= 0 <= upper):
                raise ValueError("ratio_clip_range must be finite, ordered, and include zero.")

        # Rollout targets remain fixed throughout all optimization epochs.
        current_cmf_loss = FPO.compute_cmf_loss(
            policy, observations, actions.detach(), eps.detach(), t.detach(),
            supervise_target=supervise_target,
        )
        if current_cmf_loss.shape != init_cmf_loss.shape:
            raise ValueError(
                "current_cmf_loss and init_cmf_loss must have the same shape, "
                f"got {tuple(current_cmf_loss.shape)} and {tuple(init_cmf_loss.shape)}."
            )
        # Accumulate/exponentiate in at least float32 when models use AMP.
        loss_dtype = torch.float64 if current_cmf_loss.dtype == torch.float64 else torch.float32
        current_cmf_loss = current_cmf_loss.to(dtype=loss_dtype)
        init_cmf_loss = init_cmf_loss.detach().to(dtype=loss_dtype, device=actions.device)
        advantages = advantages.detach().to(dtype=loss_dtype, device=actions.device)
        if not torch.isfinite(current_cmf_loss).all() or not torch.isfinite(init_cmf_loss).all():
            raise ValueError("Current and rollout CMF losses must be finite.")
        if not torch.isfinite(advantages).all():
            raise ValueError("advantages must be finite.")

        log_ratio = init_cmf_loss - current_cmf_loss
        if average_losses_before_exp:
            log_ratio = log_ratio.mean(dim=-1, keepdim=True)
        # Guard the aggregated log ratio as well as the per-sample ablation.
        # Clipping before the mean would change the Monte Carlo estimator.
        if ratio_clip_range is not None:
            log_ratio = log_ratio.clamp(lower, upper)
        ratio = log_ratio.exp()

        advantages = FPO._expand_advantages(advantages, ratio.shape)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        surrogate_loss1 = ratio * advantages
        surrogate_loss2 = clipped_ratio * advantages
        loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()

        with torch.no_grad():
            # This is a nonnegative drift proxy on a surrogate ratio, not an
            # exact KL divergence of the flow policies. Retain the old key.
            approx_kl = (torch.expm1(log_ratio) - log_ratio).mean()
            clip_fraction = (torch.abs(ratio - 1.0) > clip_ratio).to(loss_dtype).mean()

        return {
            "loss": loss,
            "policy_loss": loss,
            "ratio": ratio.mean(),
            "ratio_min": ratio.min(),
            "ratio_max": ratio.max(),
            "log_ratio": log_ratio.detach().mean(),
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
            "kl_divergence": approx_kl,
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
