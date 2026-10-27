import torch
import torch.nn as nn
from ..nn.steps import ValueStep
from ._common import mse, validate_same_shape
from .shac import SHAC

NNMODEL = nn.Module


class AHAC(SHAC):
    """Adaptive Horizon Actor-Critic losses (Georgiev et al., ICML 2024).

    AHAC extends SHAC with three changes: a constrained actor objective whose
    dual variables adapt the rollout horizon, a double critic in place of the
    delayed target critic, and critic training until convergence. Rollout
    tensors follow SHAC: time-major ``[H, N]``, with ``rewards``,
    ``next_values`` and ``contact_norms`` keeping their autograd graph for the
    policy loss. ``H`` is the current integer horizon, ``int(horizon)``; the
    dual variables are a ``[H_max]`` tensor that covers the longest horizon.

    ``compute_td_lambda_targets`` is inherited from SHAC unchanged.
    """

    @staticmethod
    def _validate_dual(name: str, dual: torch.Tensor, horizon: int) -> None:
        if dual.ndim != 1 or dual.shape[0] < horizon:
            raise ValueError(
                f"{name} must have shape [H_max] with H_max >= {horizon}, got {tuple(dual.shape)}."
            )

    @staticmethod
    def compute_policy_loss(
        *,
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
        gamma: float,
        truncated: torch.Tensor | None = None,
        contact_norms: torch.Tensor | None = None,
        dual: torch.Tensor | None = None,
        contact_threshold: float = 500.0,
    ) -> dict[str, torch.Tensor]:
        """Negative Lagrangian ``J + sum_h phi_h (C - ||grad f_h||)``, divided by ``H * N``.

        ``J`` is the SHAC short-horizon return. ``contact_norms[h, n]`` is the
        normalized contact stiffness of environment ``n`` at step ``h`` and
        ``dual`` holds the Lagrange multipliers ``phi``; only ``dual[:H]`` is
        used. Without them the loss equals ``SHAC.compute_policy_loss``.
        """
        result = SHAC.compute_policy_loss(
            rewards=rewards, next_values=next_values, terminated=terminated, gamma=gamma, truncated=truncated
        )
        if (contact_norms is None) != (dual is None):
            raise ValueError("contact_norms and dual must be passed together.")
        if contact_norms is None:
            return result

        validate_same_shape("rewards", rewards, "contact_norms", contact_norms)
        horizon = rewards.shape[0]
        AHAC._validate_dual("dual", dual, horizon)
        phi = dual[:horizon].detach().to(dtype=contact_norms.dtype, device=contact_norms.device)
        # The constraint slack is summed per step like the rewards, so it shares the 1 / (H * N) scale.
        constraint = (phi.unsqueeze(-1) * (contact_threshold - contact_norms)).sum() / contact_norms.numel()

        result["loss"] = result["loss"] - constraint
        result["contact_norm"] = contact_norms.detach().mean()
        result["constraint_violation"] = (contact_norms.detach() > contact_threshold).float().mean()
        return result

    @staticmethod
    @torch.no_grad()
    def update_dual(
        dual: torch.Tensor,
        *,
        contact_norms: torch.Tensor,
        contact_threshold: float,
        lr: float,
    ) -> torch.Tensor:
        """Dual descent ``phi_h <- phi_h - lr * (C - mean_n ||grad f_h||)``, returned as a new tensor.

        Multipliers grow at steps that violate the contact threshold and shrink
        where it holds. Steps beyond the rollout length keep their value.
        They are not clipped at zero, so satisfied constraints can grow the
        horizon again through ``update_horizon``.
        """
        if contact_norms.ndim != 2:
            raise ValueError(f"contact_norms must have shape [H, N], got {tuple(contact_norms.shape)}.")
        horizon = contact_norms.shape[0]
        AHAC._validate_dual("dual", dual, horizon)
        slack = contact_threshold - contact_norms.detach().to(dtype=dual.dtype, device=dual.device).mean(dim=1)
        new_dual = dual.detach().clone()
        new_dual[:horizon] -= lr * slack
        return new_dual

    @staticmethod
    def update_horizon(
        horizon: float,
        *,
        dual: torch.Tensor,
        lr: float,
        min_horizon: float = 1.0,
        max_horizon: float | None = None,
    ) -> float:
        """``H <- clip(H - lr * sum_{h < H} phi_h, min_horizon, max_horizon)``.

        Positive multipliers (stiff contact) shorten the horizon, negative ones
        lengthen it. ``max_horizon`` defaults to ``len(dual)``. The paper's
        Algorithm 1 prints ``H + lr * sum(phi)``; the minus sign here is the one
        that shortens rollouts when the contact constraint is violated.
        """
        max_horizon = float(dual.shape[0]) if max_horizon is None else max_horizon
        if not 1.0 <= min_horizon <= max_horizon:
            raise ValueError(f"need 1 <= min_horizon <= max_horizon, got {min_horizon} and {max_horizon}.")
        current = int(horizon)
        AHAC._validate_dual("dual", dual, current)
        horizon = horizon - lr * dual[:current].sum().item()
        return float(min(max(horizon, min_horizon), max_horizon))

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
    def critic_converged(losses: list[float], *, window: int = 5, tolerance: float = 0.5) -> bool:
        """Stop criterion ``sum_{i=n-window}^{n} |L_i - L_{i-1}| < tolerance`` over critic mini-batch losses.

        Returns ``False`` until ``window + 1`` losses are available. The paper
        pairs it with a cap of 64 critic iterations, which the caller enforces.
        """
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}.")
        if len(losses) <= window:
            return False
        recent = losses[-(window + 1):]
        change = sum(abs(current - previous) for previous, current in zip(recent[:-1], recent[1:]))
        return change < tolerance
