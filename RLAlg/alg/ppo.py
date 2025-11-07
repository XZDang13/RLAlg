import torch
import torch.nn as nn
from typing import Union
from ..nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep

NNMODEL = nn.Module


class PPO:
    @staticmethod
    def compute_kl_divergence(
        log_probs: torch.Tensor,
        log_probs_hat: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            ratio = (log_probs - log_probs_hat)
            kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
        return kl_divergence

    @staticmethod
    def compute_policy_loss_with_multi_critic(
        policy_model: NNMODEL,
        log_probs_hat: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        advantages_list: list[torch.Tensor],
        weights: list[float],
        clip_ratio: float,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor]:

        step: Union[StochasticContinuousPolicyStep, DiscretePolicyStep] = policy_model(observations, actions)
        log_probs = step.log_prob
        entropy = step.entropy.mean()

        ratio = (log_probs - log_probs_hat).exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        loss = 0.0
        for weight, advantages in zip(weights, advantages_list):
            loss += -torch.min(ratio * advantages, clipped_ratio * advantages).mean() * weight

        if isinstance(step, StochasticContinuousPolicyStep):
            loss += step.mean.pow(2).mean() * regularization_weight

        kl_divergence = PPO.compute_kl_divergence(log_probs, log_probs_hat)

        return {
            "loss": loss,
            "entropy": entropy,
            "kl_divergence": kl_divergence
        }

    @staticmethod
    def compute_policy_loss(
        policy_model: NNMODEL,
        log_probs_hat: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor]:

        step: Union[StochasticContinuousPolicyStep, DiscretePolicyStep] = policy_model(observations, actions)
        log_probs = step.log_prob
        entropy = step.entropy.mean()

        ratio = (log_probs - log_probs_hat).exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        if isinstance(step, StochasticContinuousPolicyStep):
            loss += step.mean.pow(2).mean() * regularization_weight

        kl_divergence = PPO.compute_kl_divergence(log_probs, log_probs_hat)

        return {
            "loss": loss,
            "entropy": entropy,
            "kl_divergence": kl_divergence
        }

    @staticmethod
    def compute_value_loss(
        value_model: NNMODEL,
        observations: torch.Tensor,
        returns: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        step: ValueStep = value_model(observations)
        values = step.value
        loss = 0.5 * ((returns - values) ** 2).mean()
        return {
            "loss": loss
        }

    @staticmethod
    def compute_clipped_value_loss(
        value_model: NNMODEL,
        observations: torch.Tensor,
        values_hat: torch.Tensor,
        returns: torch.Tensor,
        clip_ratio: float
    ) -> dict[str, torch.Tensor]:
        step: ValueStep = value_model(observations)
        values = step.value

        loss_unclipped = 0.5 * (returns - values) ** 2

        values_clipped = values_hat + torch.clamp(values - values_hat, -clip_ratio, clip_ratio)
        loss_clipped = 0.5 * (returns - values_clipped) ** 2

        loss = torch.max(loss_unclipped, loss_clipped)
        loss = loss.mean()

        return {
            "loss": loss
        }