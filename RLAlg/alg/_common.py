"""Helpers shared by the algorithm loss functions."""

from typing import Sequence

import torch
import torch.nn as nn


def validate_same_shape(name_a: str, tensor_a: torch.Tensor, name_b: str, tensor_b: torch.Tensor) -> None:
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, got {tuple(tensor_a.shape)} and {tuple(tensor_b.shape)}."
        )


def mse(prediction: torch.Tensor, target: torch.Tensor, name: str = "prediction") -> torch.Tensor:
    """``0.5 * mean((prediction - target) ** 2)``, the regression loss used by every value/Q loss."""
    validate_same_shape(name, prediction, "target", target)
    return 0.5 * (prediction - target).pow(2).mean()


def weighted_models(
    models: nn.Module | Sequence[nn.Module],
    weights: Sequence[float] | None,
    name: str = "critic_model",
) -> list[tuple[nn.Module, float]]:
    """Pair one model with weight 1, or a sequence of models with their weights."""
    if isinstance(models, nn.Module):
        if weights is not None:
            raise ValueError(f"{name}_weights requires a sequence of {name}s.")
        return [(models, 1.0)]
    models = list(models)
    if not models:
        raise ValueError(f"{name} must be a module or a non-empty sequence of modules.")
    if weights is None or len(weights) != len(models):
        raise ValueError(f"{name}_weights must give one weight per {name}.")
    return list(zip(models, weights))


@torch.no_grad()
def soft_update(model: nn.Module, model_target: nn.Module, tau: float) -> None:
    """Polyak-average parameters into the target and copy buffers (e.g. normalizer statistics)."""
    params, target_params = list(model.parameters()), list(model_target.parameters())
    buffers, target_buffers = list(model.buffers()), list(model_target.buffers())
    if len(params) != len(target_params) or len(buffers) != len(target_buffers):
        raise ValueError("model and model_target must have the same parameters and buffers.")
    for param, target in zip(params, target_params):
        target.lerp_(param, tau)
    for buffer, target in zip(buffers, target_buffers):
        target.copy_(buffer)


def alpha_loss(log_alpha: torch.Tensor, log_probs: torch.Tensor, target_entropy: float) -> dict[str, torch.Tensor]:
    loss = -(log_alpha.exp() * (log_probs + target_entropy).detach()).mean()
    return {"loss": loss, "alpha": log_alpha.exp().detach()}
