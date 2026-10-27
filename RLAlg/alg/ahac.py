import torch
import torch.nn as nn
from ..nn.steps import ValueStep
from ._common import mse
from .shac import SHAC

NNMODEL = nn.Module


class AHAC(SHAC):
    """Adaptive Horizon Actor-Critic losses (Georgiev et al., ICML 2024).

    AHAC extends SHAC with three changes: a rollout horizon adapted by dual
    variables on a contact-stiffness constraint, a double critic in place of
    the delayed target critic, and critic training until convergence. Rollout
    tensors follow SHAC: time-major ``[H, N]``. The current horizon is a float
    ``horizon``; each rollout runs ``H = len(dual) = round(horizon)`` steps.

    ``compute_policy_loss`` and ``compute_td_lambda_targets`` are inherited from
    SHAC unchanged: as in the reference implementation, the contact constraint
    only adapts the horizon and adds no term to the actor loss.
    """

    @staticmethod
    def compute_contact_norms(contact_forces: torch.Tensor, accelerations: torch.Tensor) -> torch.Tensor:
        """Normalized contact stiffness ``||contact_forces / max(accelerations, 1)||`` per environment.

        Both inputs have a leading environment dimension ``N`` and the same
        shape; the norm is taken over all other dimensions and returns ``[N]``.
        Stack one result per rollout step into the ``[H, N]`` ``contact_norms``.
        """
        if contact_forces.shape != accelerations.shape:
            raise ValueError(
                "contact_forces and accelerations must have the same shape, "
                f"got {tuple(contact_forces.shape)} and {tuple(accelerations.shape)}."
            )
        normalized = contact_forces / accelerations.clamp(min=1.0)
        return torch.linalg.vector_norm(normalized, dim=tuple(range(1, normalized.ndim)))

    @staticmethod
    @torch.no_grad()
    def update_horizon(
        horizon: float,
        *,
        dual: torch.Tensor,
        contact_norms: torch.Tensor,
        contact_threshold: float,
        lr: float,
        min_horizon: float,
        max_horizon: float,
    ) -> tuple[float, torch.Tensor]:
        """Dual step on the contact constraint, then the horizon step.

        ``dual <- dual - lr * (C - mean_n contact_norms)`` and
        ``horizon <- clip(horizon + lr * sum(dual), min_horizon, max_horizon)``,
        as in the reference implementation and Algorithm 1 of the paper.
        Returns the new horizon and the dual variables for the next rollout,
        ``dual[0]`` repeated ``round(horizon)`` times. Start from
        ``horizon = min_horizon`` and ``dual = zeros(round(min_horizon))``.
        """
        if contact_norms.ndim != 2:
            raise ValueError(f"contact_norms must have shape [H, N], got {tuple(contact_norms.shape)}.")
        if dual.shape != contact_norms.shape[:1]:
            raise ValueError(
                f"dual must have shape [H] = {tuple(contact_norms.shape[:1])}, got {tuple(dual.shape)}."
            )
        if not 1.0 <= min_horizon <= max_horizon:
            raise ValueError(f"need 1 <= min_horizon <= max_horizon, got {min_horizon} and {max_horizon}.")

        contact_norms = contact_norms.detach().to(dtype=dual.dtype, device=dual.device)
        dual = dual.detach() - lr * (contact_threshold - contact_norms.mean(dim=-1))
        horizon = horizon + lr * dual.sum().item()
        horizon = float(min(max(horizon, min_horizon), max_horizon))
        return horizon, dual[0].repeat(round(horizon))

    @staticmethod
    def compute_next_values(value_model: NNMODEL, observations: torch.Tensor|dict[str, torch.Tensor]) -> torch.Tensor:
        """Pessimistic double-critic value ``min(V1, V2)``, keeping the autograd graph."""
        value_steps: tuple[ValueStep, ValueStep] = value_model(observations)
        return torch.min(value_steps[0].value, value_steps[1].value)

    @staticmethod
    def compute_value_loss(
        value_model: NNMODEL,
        *,
        observations: torch.Tensor|dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Double-critic regression ``0.5 * mean((V1 - target)^2) + 0.5 * mean((V2 - target)^2)``.

        ``value_model`` returns two ``ValueStep`` objects, as SAC critics do.
        This equals the reference loss, the squared error averaged over both critics.
        """
        value_steps: tuple[ValueStep, ValueStep] = value_model(observations)
        value1 = value_steps[0].value
        value2 = value_steps[1].value
        targets = targets.detach()
        loss = mse(value1, targets, "value1") + mse(value2, targets, "value2")

        return {
            "loss": loss,
            "value1": value1.detach().mean(),
            "value2": value2.detach().mean(),
            "target": targets.mean(),
        }

    @staticmethod
    def critic_converged(losses: list[float], *, window: int = 5, tolerance: float = 0.2) -> bool:
        """Early stop ``|mean(diff(losses[-window:]))| < tolerance`` for critic training.

        ``losses`` holds one mean mini-batch loss per critic epoch. Returns
        ``False`` until ``window`` losses are available. The reference checks
        before recording the current epoch's loss and caps training at 64 epochs.
        """
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}.")
        if len(losses) < window:
            return False
        recent = losses[-window:]
        mean_change = sum(current - previous for previous, current in zip(recent[:-1], recent[1:])) / (window - 1)
        return abs(mean_change) < tolerance
