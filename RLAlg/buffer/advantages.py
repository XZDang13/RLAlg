"""Return and advantage estimators for on-policy rollouts stored as ``[T, N]`` tensors.

``terminated`` marks true terminations, which never bootstrap. ``truncated``
marks timeouts; they end the recursion like a termination. To bootstrap a
timeout from the real final observation, add ``gamma * V(final_obs)`` to that
step's reward before calling, or pass ``bootstrap_timeouts=True`` to use the
current-value convention ``gamma * values[t]`` instead (never both).
"""

from typing import Optional

import torch


def _boundaries(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: Optional[torch.Tensor],
    last_values: torch.Tensor,
    values: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [T, N], got {tuple(rewards.shape)}.")
    tensors = {"terminated": terminated} if values is None else {"values": values, "terminated": terminated}
    if truncated is not None:
        tensors["truncated"] = truncated
    for name, tensor in tensors.items():
        if tensor.shape != rewards.shape:
            raise ValueError(f"{name} must have shape {tuple(rewards.shape)}, got {tuple(tensor.shape)}.")
    if last_values.shape != rewards.shape[1:]:
        raise ValueError(
            f"last_values must have shape {tuple(rewards.shape[1:])}, got {tuple(last_values.shape)}."
        )
    terminated = terminated.to(dtype=torch.bool, device=rewards.device)
    if truncated is None:
        truncated = torch.zeros_like(terminated)
    else:
        truncated = truncated.to(dtype=torch.bool, device=rewards.device)
    return terminated | truncated, truncated


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-std advantages over the whole batch (needs at least two elements)."""
    if advantages.numel() < 2:
        raise ValueError("normalize_advantages needs at least two advantages.")
    return (advantages - advantages.mean()) / (advantages.std() + eps)


@torch.no_grad()
def compute_returns(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float = 0.99,
    truncated: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Discounted returns, bootstrapped from ``last_values`` after the final step."""
    boundaries, _ = _boundaries(rewards, terminated, truncated, last_values)
    not_boundary = 1.0 - boundaries.to(dtype=rewards.dtype)
    returns = torch.zeros_like(rewards)
    next_return = last_values.to(dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        next_return = rewards[t] + gamma * next_return * not_boundary[t]
        returns[t] = next_return
    return returns


@torch.no_grad()
def compute_advantage_mc(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float = 0.99,
    truncated: Optional[torch.Tensor] = None,
    bootstrap_timeouts: bool = False,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte Carlo returns and advantages ``returns - values``."""
    _, truncated_mask = _boundaries(rewards, terminated, truncated, last_values, values)
    if bootstrap_timeouts:
        rewards = rewards + gamma * values * truncated_mask.to(dtype=rewards.dtype)
    returns = compute_returns(
        rewards=rewards, terminated=terminated, last_values=last_values, gamma=gamma, truncated=truncated,
    )
    advantages = returns - values
    return returns, normalize_advantages(advantages) if normalize else advantages


@torch.no_grad()
def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    truncated: Optional[torch.Tensor] = None,
    bootstrap_timeouts: bool = False,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(λ) returns and advantages; ``normalize`` standardises the advantages.

    Returns are always ``advantages + values`` computed before normalization.
    """
    boundaries, truncated_mask = _boundaries(rewards, terminated, truncated, last_values, values)
    not_boundary = 1.0 - boundaries.to(dtype=rewards.dtype)
    gae_rewards = rewards
    if bootstrap_timeouts:
        gae_rewards = rewards + gamma * values * truncated_mask.to(dtype=rewards.dtype)
    next_value = last_values.to(dtype=rewards.dtype, device=rewards.device)
    next_adv = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=rewards.device)

    advantages = torch.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        delta = gae_rewards[t] + gamma * next_value * not_boundary[t] - values[t]
        next_adv = delta + gamma * lambda_ * next_adv * not_boundary[t]
        advantages[t] = next_adv
        next_value = values[t]

    returns = advantages + values
    return returns, normalize_advantages(advantages) if normalize else advantages
