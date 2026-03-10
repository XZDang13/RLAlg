import torch
import torch.nn as nn
from typing import Any
from RLAlg.ode_solver.euler_ode_solver import EulerODESolver
from ..nn.steps import ValueStep, FPOStep

NNMODEL = nn.Module

class FPO:
    @staticmethod
    def _extract_prediction(model_output: ValueStep | torch.Tensor | tuple[Any, ...]) -> torch.Tensor:
        if isinstance(model_output, tuple):
            if len(model_output) == 0:
                raise ValueError("Actor output tuple must be non-empty.")
            model_output = model_output[0]

        if isinstance(model_output, ValueStep):
            prediction = model_output.value
        elif torch.is_tensor(model_output):
            prediction = model_output
        else:
            raise TypeError(
                "Actor output must be ValueStep or Tensor, "
                f"got {type(model_output)}."
            )
        return prediction

    @staticmethod
    def _call_actor(
        actor: NNMODEL,
        obs: torch.Tensor,
        current_action: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        call_errors: list[TypeError] = []
        call_attempts = (
            lambda: actor(obs, current_action, t=t),
            lambda: actor(obs, current_action, t),
            lambda: actor(obs, current_action),
            lambda: actor(obs),
        )

        for call_fn in call_attempts:
            try:
                return FPO._extract_prediction(call_fn())
            except TypeError as error:
                call_errors.append(error)

        raise TypeError(
            "Could not call actor with supported signatures: "
            "(obs, action, t=...), (obs, action, t), (obs, action), or (obs)."
        ) from call_errors[-1]

    @staticmethod
    def compute_cmf_loss(
        actor: NNMODEL,
        obs: torch.Tensor,
        action: torch.Tensor,
        eps: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        n_samples_per_action: int = 1,
        output_mode: str = "u",
    ) -> dict[str, torch.Tensor]:
        if n_samples_per_action <= 0:
            raise ValueError(f"n_samples_per_action must be > 0, got {n_samples_per_action}.")
        if action.ndim < 1:
            raise ValueError(
                f"action must have at least 1 dim with trailing action_dim, got {tuple(action.shape)}."
            )
        if obs.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "obs and action batch dims must match, "
                f"got {tuple(obs.shape[:-1])} and {tuple(action.shape[:-1])}."
            )

        action_dim = action.shape[-1]
        batch_dims = action.shape[:-1]
        sample_shape = (*batch_dims, n_samples_per_action)
        flow_shape = (*sample_shape, action_dim)
        time_shape = (*sample_shape, 1)

        if eps is None:
            eps = torch.randn(flow_shape, dtype=action.dtype, device=action.device)
        elif eps.shape != flow_shape:
            raise ValueError(f"eps must have shape {flow_shape}, got {tuple(eps.shape)}.")
        else:
            eps = eps.to(dtype=action.dtype, device=action.device)

        if t is None:
            t = torch.rand(time_shape, dtype=action.dtype, device=action.device)
        elif t.shape != time_shape:
            raise ValueError(f"t must have shape {time_shape}, got {tuple(t.shape)}.")
        else:
            t = t.to(dtype=action.dtype, device=action.device)

        action_samples = action.unsqueeze(-2).expand(flow_shape)
        obs_samples = obs.unsqueeze(-2).expand(*sample_shape, obs.shape[-1])
        x_t = t * eps + (1.0 - t) * action_samples

        network_pred = FPO._call_actor(actor, obs_samples, x_t, t)
        if network_pred.shape != flow_shape:
            raise ValueError(
                "Actor prediction shape must match x_t shape, "
                f"got {tuple(network_pred.shape)} and {flow_shape}."
            )

        if output_mode == "u":
            velocity_gt = eps - action_samples
            loss_per_sample = ((network_pred - velocity_gt) ** 2).mean(dim=-1)
        elif output_mode == "u_but_supervise_as_eps":
            x0_pred = x_t - t * network_pred
            x1_pred = x0_pred + network_pred
            loss_per_sample = ((eps - x1_pred) ** 2).mean(dim=-1)
        else:
            raise ValueError(
                "output_mode must be one of {'u', 'u_but_supervise_as_eps'}, "
                f"got {output_mode}."
            )

        loss_per_action = loss_per_sample.mean(dim=-1)
        loss = loss_per_action.mean()

        return {"eps": eps, "t":t, "cmf_loss": loss}
    
    @staticmethod
    def sample_actions(
        actor: NNMODEL,
        obs: torch.Tensor,
        init_noise: torch.Tensor,
        t: torch.Tensor,
        deterministic: bool = False
    ) -> FPOStep:
        pass
    
    
