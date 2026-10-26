import torch
import torch.nn as nn
from ..nn.steps import ValueStep
from ._common import mse, soft_update, validate_same_shape

NNMODEL = nn.Module


class SHAC:
    """Short-Horizon Actor-Critic losses for differentiable simulators.

    The caller rolls out a differentiable environment for ``H`` steps over ``N``
    parallel environments and passes time-major ``[H, N]`` tensors. ``rewards``
    and ``next_values`` must keep their autograd graph for the policy loss;
    ``next_values[t]`` is ``V(s_{t+1})`` evaluated on the actual next observation
    (the final observation before an automatic reset). Terminated steps do not
    bootstrap, truncated steps and the last rollout step do.
    """

    @staticmethod
    def _prepare_rollout_masks(
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rewards.ndim != 2:
            raise ValueError(f"rewards must have shape [H, N], got {tuple(rewards.shape)}.")
        validate_same_shape("rewards", rewards, "next_values", next_values)
        validate_same_shape("rewards", rewards, "terminated", terminated)
        if truncated is None:
            truncated = torch.zeros_like(terminated, dtype=torch.bool)
        validate_same_shape("rewards", rewards, "truncated", truncated)

        terminated = terminated.to(dtype=torch.bool, device=rewards.device)
        truncated = truncated.to(dtype=torch.bool, device=rewards.device)
        boundaries = terminated | truncated
        # The rollout horizon always ends a segment and bootstraps from the critic.
        boundaries[-1] = True
        return terminated, boundaries

    @staticmethod
    def _bootstrap_values(next_values: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        # torch.where keeps non-finite values at terminal observations out of the loss.
        return torch.where(terminated, torch.zeros_like(next_values), next_values)

    @staticmethod
    def compute_policy_loss(
        *,
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
        gamma: float,
        truncated: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Negative short-horizon return ``sum_t gamma^t r_t + gamma^h V(s_h)``.

        Episodes that end inside the rollout contribute their own discounted
        segment return, then the discount and accumulator restart. The total is
        divided by ``H * N`` as in the reference implementation.
        """
        terminated, boundaries = SHAC._prepare_rollout_masks(rewards, next_values, terminated, truncated)
        bootstrap = SHAC._bootstrap_values(next_values, terminated)
        horizon, num_envs = rewards.shape

        discount = torch.ones(num_envs, dtype=rewards.dtype, device=rewards.device)
        accumulated = torch.zeros(num_envs, dtype=rewards.dtype, device=rewards.device)
        total_return = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for t in range(horizon):
            accumulated = accumulated + discount * rewards[t]
            discount = discount * gamma
            end = boundaries[t].to(dtype=rewards.dtype)
            total_return = total_return + (end * (accumulated + discount * bootstrap[t])).sum()
            accumulated = accumulated * (1.0 - end)
            discount = torch.where(boundaries[t], torch.ones_like(discount), discount)

        loss = -total_return / (horizon * num_envs)

        return {
            "loss": loss,
            "reward": rewards.detach().mean(),
            "next_value": bootstrap.detach().mean(),
        }

    @staticmethod
    @torch.no_grad()
    def compute_td_lambda_targets(
        *,
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
        gamma: float,
        lambda_: float,
        truncated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """TD(lambda) value targets over one rollout, returned as ``[H, N]``.

        ``G_t = r_t + gamma * ((1 - lambda) V(s_{t+1}) + lambda G_{t+1})`` inside
        a segment and ``G_t = r_t + gamma * V(s_{t+1})`` at a boundary (zero value
        when terminated). ``lambda_=0`` gives one-step targets.
        """
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}.")
        rewards = rewards.detach()
        next_values = next_values.detach()
        terminated, boundaries = SHAC._prepare_rollout_masks(rewards, next_values, terminated, truncated)
        bootstrap = SHAC._bootstrap_values(next_values, terminated)

        targets = torch.zeros_like(rewards)
        next_return = torch.zeros_like(rewards[0])
        for t in reversed(range(rewards.shape[0])):
            blended = (1.0 - lambda_) * bootstrap[t] + lambda_ * next_return
            next_target = torch.where(boundaries[t], bootstrap[t], blended)
            targets[t] = rewards[t] + gamma * next_target
            next_return = targets[t]
        return targets

    @staticmethod
    def compute_value_loss(
        value_model: NNMODEL,
        *,
        observations: torch.Tensor|dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """``0.5 * mean((V - target)^2)`` to detached targets.

        The reference SHAC uses plain MSE without the 0.5; with Adam the scale
        only matters through gradient-norm clipping.
        """
        step: ValueStep = value_model(observations)
        values = step.value
        targets = targets.detach()
        loss = mse(values, targets, "values")

        return {
            "loss": loss,
            "value": values.detach().mean(),
            "target": targets.mean(),
        }

    update_target_param = staticmethod(soft_update)
