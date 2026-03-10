import torch
import torch.nn as nn
from typing import Union, Any
from ..nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep

NNMODEL = nn.Module

class PPO:
    @staticmethod
    def _validate_same_shape(name_a: str, tensor_a: torch.Tensor, name_b: str, tensor_b: torch.Tensor) -> None:
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"{name_a} and {name_b} must have the same shape, got {tuple(tensor_a.shape)} and {tuple(tensor_b.shape)}."
            )

    @staticmethod
    def _validate_multi_critic_inputs(weights: list[float], advantages_list: list[torch.Tensor]) -> None:
        if len(weights) == 0:
            raise ValueError("weights must be non-empty.")
        if len(advantages_list) == 0:
            raise ValueError("advantages_list must be non-empty.")
        if len(weights) != len(advantages_list):
            raise ValueError(
                f"weights and advantages_list must have the same length, got {len(weights)} and {len(advantages_list)}."
            )

    @staticmethod
    def _prepare_valid_mask(values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor | None:
        if valid_mask is None:
            return None
        if values.shape != valid_mask.shape:
            raise ValueError(
                f"valid_mask must have the same shape as values, got {tuple(valid_mask.shape)} and {tuple(values.shape)}."
            )
        return valid_mask.to(dtype=values.dtype, device=values.device)

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if valid_mask is None:
            return values.mean()
        denominator = torch.clamp(valid_mask.sum(), min=1.0)
        return (values * valid_mask).sum() / denominator

    @staticmethod
    def _unpack_recurrent_output(model_output: Any) -> tuple[Any, Any | None]:
        if isinstance(model_output, tuple):
            if len(model_output) != 2:
                raise ValueError(
                    f"Recurrent model output tuple must be (step, next_state), got tuple of length {len(model_output)}."
                )
            return model_output[0], model_output[1]
        return model_output, None

    @staticmethod
    def compute_kl_divergence(
        log_probs: torch.Tensor,
        log_probs_hat: torch.Tensor
    ) -> torch.Tensor:
        PPO._validate_same_shape("log_probs", log_probs, "log_probs_hat", log_probs_hat)
        with torch.no_grad():
            ratio = (log_probs - log_probs_hat)
            kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
        return kl_divergence

    @staticmethod
    def compute_policy_loss_with_multi_critic(
        policy_model: NNMODEL,
        log_probs_hat: torch.Tensor,
        observations: torch.Tensor|dict[str, torch.Tensor],
        actions: torch.Tensor,
        advantages_list: list[torch.Tensor],
        weights: list[float],
        clip_ratio: float,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor]:
        PPO._validate_multi_critic_inputs(weights, advantages_list)
        step: Union[StochasticContinuousPolicyStep, DiscretePolicyStep] = policy_model(observations, actions)
        log_probs = step.log_prob
        PPO._validate_same_shape("log_probs", log_probs, "log_probs_hat", log_probs_hat)
        entropy = step.entropy.mean()

        ratio = (log_probs - log_probs_hat).exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        loss = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
        for weight, advantages in zip(weights, advantages_list):
            PPO._validate_same_shape("log_probs", log_probs, "advantages", advantages)
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
        observations: torch.Tensor|dict[str, torch.Tensor],
        actions: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor]:
        step: Union[StochasticContinuousPolicyStep, DiscretePolicyStep] = policy_model(observations, actions)
        log_probs = step.log_prob
        PPO._validate_same_shape("log_probs", log_probs, "log_probs_hat", log_probs_hat)
        PPO._validate_same_shape("log_probs", log_probs, "advantages", advantages)
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
        observations: torch.Tensor|dict[str, torch.Tensor],
        returns: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        step: ValueStep = value_model(observations)
        values = step.value
        PPO._validate_same_shape("values", values, "returns", returns)
        loss = 0.5 * ((returns - values) ** 2).mean()
        return {
            "loss": loss
        }

    @staticmethod
    def compute_policy_loss_recurrent(
        policy_model: NNMODEL,
        log_probs_hat: torch.Tensor,
        observations: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float,
        episode_starts: torch.Tensor,
        initial_state: Any | None = None,
        valid_mask: torch.Tensor | None = None,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor | Any]:
        model_output = policy_model(observations, actions, initial_state, episode_starts)
        step, next_state = PPO._unpack_recurrent_output(model_output)
        log_probs = step.log_prob

        PPO._validate_same_shape("log_probs", log_probs, "log_probs_hat", log_probs_hat)
        PPO._validate_same_shape("log_probs", log_probs, "advantages", advantages)
        PPO._validate_same_shape("log_probs", log_probs, "episode_starts", episode_starts)
        valid_mask = PPO._prepare_valid_mask(log_probs, valid_mask)

        ratio = (log_probs - log_probs_hat).exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_objective = torch.min(ratio * advantages, clipped_ratio * advantages)

        loss = -PPO._masked_mean(policy_objective, valid_mask)
        entropy = PPO._masked_mean(step.entropy, valid_mask)

        if isinstance(step, StochasticContinuousPolicyStep):
            reg_term = step.mean.pow(2).reshape(*log_probs.shape, -1).mean(dim=-1)
            loss += PPO._masked_mean(reg_term, valid_mask) * regularization_weight

        kl_values = (torch.exp(log_probs - log_probs_hat) - 1) - (log_probs - log_probs_hat)
        kl_divergence = PPO._masked_mean(kl_values, valid_mask).detach()

        return {
            "loss": loss,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
            "next_state": next_state,
        }

    @staticmethod
    def compute_policy_loss_with_multi_critic_recurrent(
        policy_model: NNMODEL,
        log_probs_hat: torch.Tensor,
        observations: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
        advantages_list: list[torch.Tensor],
        weights: list[float],
        clip_ratio: float,
        episode_starts: torch.Tensor,
        initial_state: Any | None = None,
        valid_mask: torch.Tensor | None = None,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor | Any]:
        PPO._validate_multi_critic_inputs(weights, advantages_list)
        model_output = policy_model(observations, actions, initial_state, episode_starts)
        step, next_state = PPO._unpack_recurrent_output(model_output)
        log_probs = step.log_prob

        PPO._validate_same_shape("log_probs", log_probs, "log_probs_hat", log_probs_hat)
        PPO._validate_same_shape("log_probs", log_probs, "episode_starts", episode_starts)
        valid_mask = PPO._prepare_valid_mask(log_probs, valid_mask)

        ratio = (log_probs - log_probs_hat).exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        loss = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
        for weight, advantages in zip(weights, advantages_list):
            PPO._validate_same_shape("log_probs", log_probs, "advantages", advantages)
            objective = torch.min(ratio * advantages, clipped_ratio * advantages)
            loss += -PPO._masked_mean(objective, valid_mask) * weight

        entropy = PPO._masked_mean(step.entropy, valid_mask)

        if isinstance(step, StochasticContinuousPolicyStep):
            reg_term = step.mean.pow(2).reshape(*log_probs.shape, -1).mean(dim=-1)
            loss += PPO._masked_mean(reg_term, valid_mask) * regularization_weight

        kl_values = (torch.exp(log_probs - log_probs_hat) - 1) - (log_probs - log_probs_hat)
        kl_divergence = PPO._masked_mean(kl_values, valid_mask).detach()

        return {
            "loss": loss,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
            "next_state": next_state,
        }

    @staticmethod
    def compute_value_loss_recurrent(
        value_model: NNMODEL,
        observations: torch.Tensor | dict[str, torch.Tensor],
        returns: torch.Tensor,
        episode_starts: torch.Tensor,
        initial_state: Any | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | Any]:
        model_output = value_model(observations, initial_state, episode_starts)
        step, next_state = PPO._unpack_recurrent_output(model_output)
        values = step.value

        PPO._validate_same_shape("values", values, "returns", returns)
        PPO._validate_same_shape("values", values, "episode_starts", episode_starts)
        valid_mask = PPO._prepare_valid_mask(values, valid_mask)

        loss = 0.5 * ((returns - values) ** 2)
        loss = PPO._masked_mean(loss, valid_mask)
        return {
            "loss": loss,
            "next_state": next_state,
        }

    @staticmethod
    def compute_clipped_value_loss_recurrent(
        value_model: NNMODEL,
        observations: torch.Tensor | dict[str, torch.Tensor],
        values_hat: torch.Tensor,
        returns: torch.Tensor,
        clip_ratio: float,
        episode_starts: torch.Tensor,
        initial_state: Any | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | Any]:
        model_output = value_model(observations, initial_state, episode_starts)
        step, next_state = PPO._unpack_recurrent_output(model_output)
        values = step.value

        PPO._validate_same_shape("values", values, "values_hat", values_hat)
        PPO._validate_same_shape("values", values, "returns", returns)
        PPO._validate_same_shape("values", values, "episode_starts", episode_starts)
        valid_mask = PPO._prepare_valid_mask(values, valid_mask)

        loss_unclipped = 0.5 * (returns - values) ** 2
        values_clipped = values_hat + torch.clamp(values - values_hat, -clip_ratio, clip_ratio)
        loss_clipped = 0.5 * (returns - values_clipped) ** 2
        loss = torch.max(loss_unclipped, loss_clipped)
        loss = PPO._masked_mean(loss, valid_mask)

        return {
            "loss": loss,
            "next_state": next_state,
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
        PPO._validate_same_shape("values", values, "values_hat", values_hat)
        PPO._validate_same_shape("values", values, "returns", returns)

        loss_unclipped = 0.5 * (returns - values) ** 2

        values_clipped = values_hat + torch.clamp(values - values_hat, -clip_ratio, clip_ratio)
        loss_clipped = 0.5 * (returns - values_clipped) ** 2

        loss = torch.max(loss_unclipped, loss_clipped)
        loss = loss.mean()

        return {
            "loss": loss
        }
