import torch
import torch.nn as nn
from typing import Union, Any, Sequence
from ..nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep
from ._common import mse, validate_same_shape

NNMODEL = nn.Module

class PPO:
    @staticmethod
    def _weighted_advantages(
        advantages: torch.Tensor | Sequence[torch.Tensor],
        advantage_weights: Sequence[float] | None,
    ) -> list[tuple[torch.Tensor, float]]:
        """Pair one advantage tensor with weight 1, or one weight per critic's advantages."""
        if torch.is_tensor(advantages):
            if advantage_weights is not None:
                raise ValueError("advantage_weights requires a sequence of advantages.")
            return [(advantages, 1.0)]
        advantages = list(advantages)
        if not advantages:
            raise ValueError("advantages must be a tensor or a non-empty sequence of tensors.")
        if advantage_weights is None or len(advantage_weights) != len(advantages):
            raise ValueError("advantage_weights must give one weight per advantages tensor.")
        return list(zip(advantages, advantage_weights))

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
    def _reduce_to_reference_shape(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if values.shape == reference.shape:
            return values
        if values.ndim > reference.ndim and values.shape[:reference.ndim] == reference.shape:
            return values.reshape(*reference.shape, -1).mean(dim=-1)
        return values

    @staticmethod
    def _masked_std(values: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        mean = PPO._masked_mean(values, valid_mask)
        variance = PPO._masked_mean((values - mean) ** 2, valid_mask)
        return torch.sqrt(torch.clamp(variance, min=0.0))

    @staticmethod
    def _policy_clip_fraction(
        ratio: torch.Tensor,
        clip_ratio: float,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clipped = (torch.abs(ratio - 1.0) > clip_ratio).to(dtype=ratio.dtype)
        return PPO._masked_mean(clipped, valid_mask).detach()

    @staticmethod
    def _policy_action_stats(
        step: Union[StochasticContinuousPolicyStep, DiscretePolicyStep],
        reference: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), dtype=reference.dtype, device=reference.device)
        if not isinstance(step, StochasticContinuousPolicyStep):
            return {
                "action_log_std": zero,
                "action_std": zero,
            }

        log_std = PPO._reduce_to_reference_shape(step.log_std, reference)
        action_std = PPO._reduce_to_reference_shape(torch.exp(step.log_std), reference)
        if log_std.shape == reference.shape:
            action_log_std = PPO._masked_mean(log_std, valid_mask)
        else:
            action_log_std = log_std.mean()
        if action_std.shape == reference.shape:
            action_std_mean = PPO._masked_mean(action_std, valid_mask)
        else:
            action_std_mean = action_std.mean()

        return {
            "action_log_std": action_log_std.detach(),
            "action_std": action_std_mean.detach(),
        }

    @staticmethod
    def _policy_advantage_stats(
        advantages: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "advantage_mean": PPO._masked_mean(advantages, valid_mask).detach(),
            "advantage_std": PPO._masked_std(advantages, valid_mask).detach(),
        }

    @staticmethod
    def _policy_metric_payload(
        step: Union[StochasticContinuousPolicyStep, DiscretePolicyStep],
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "policy_clip_fraction": PPO._policy_clip_fraction(ratio, clip_ratio, valid_mask),
            **PPO._policy_action_stats(step, ratio, valid_mask),
            **PPO._policy_advantage_stats(advantages, valid_mask),
        }

    @staticmethod
    def _explained_variance(
        values: torch.Tensor,
        returns: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = values.float()
        returns = returns.float()
        mask = None if valid_mask is None else valid_mask.to(dtype=returns.dtype, device=returns.device)

        residual = returns - values
        returns_mean = PPO._masked_mean(returns, mask)
        residual_mean = PPO._masked_mean(residual, mask)
        returns_var = PPO._masked_mean((returns - returns_mean) ** 2, mask)
        residual_var = PPO._masked_mean((residual - residual_mean) ** 2, mask)
        eps = torch.finfo(returns.dtype).eps
        explained = torch.where(
            returns_var > eps,
            1.0 - residual_var / torch.clamp(returns_var, min=eps),
            torch.zeros_like(returns_var),
        )
        return explained.detach()

    @staticmethod
    def _value_clip_fraction(
        values: torch.Tensor,
        values_hat: torch.Tensor,
        clip_ratio: float,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clipped = (torch.abs(values - values_hat) > clip_ratio).to(dtype=values.dtype)
        return PPO._masked_mean(clipped, valid_mask).detach()

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
        validate_same_shape("log_probs", log_probs, "log_probs_hat", log_probs_hat)
        with torch.no_grad():
            ratio = (log_probs - log_probs_hat)
            kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
        return kl_divergence

    @staticmethod
    def _call_model(
        model: NNMODEL,
        inputs: tuple[Any, ...],
        episode_starts: torch.Tensor | None,
        initial_state: Any | None,
    ) -> tuple[Any, Any | None]:
        # Recurrent models take (inputs..., initial_state, episode_starts) and return (step, next_state).
        if episode_starts is None:
            if initial_state is not None:
                raise ValueError("initial_state requires episode_starts.")
            return model(*inputs), None
        return PPO._unpack_recurrent_output(model(*inputs, initial_state, episode_starts))

    @staticmethod
    def compute_policy_loss(
        policy_model: NNMODEL,
        *,
        log_probs_hat: torch.Tensor,
        observations: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
        advantages: torch.Tensor | Sequence[torch.Tensor],
        clip_ratio: float,
        entropy_coef: float = 0.0,
        regularization_weight: float = 0.0,
        advantage_weights: Sequence[float] | None = None,
        episode_starts: torch.Tensor | None = None,
        initial_state: Any | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | Any]:
        """Clipped PPO surrogate.

        A sequence of ``advantages`` (one per critic) with ``advantage_weights`` sums the
        weighted per-critic objectives. ``episode_starts`` switches to the recurrent calling
        convention and adds ``"next_state"``; ``valid_mask`` averages over valid steps only.
        """
        weighted_advantages_list = PPO._weighted_advantages(advantages, advantage_weights)
        step, next_state = PPO._call_model(policy_model, (observations, actions), episode_starts, initial_state)
        step: Union[StochasticContinuousPolicyStep, DiscretePolicyStep]
        log_probs = step.log_prob

        validate_same_shape("log_probs", log_probs, "log_probs_hat", log_probs_hat)
        if episode_starts is not None:
            validate_same_shape("log_probs", log_probs, "episode_starts", episode_starts)
        validate_same_shape("entropy", step.entropy, "log_probs", log_probs)
        valid_mask = PPO._prepare_valid_mask(log_probs, valid_mask)

        ratio = (log_probs - log_probs_hat).exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        loss = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
        weighted_advantages = torch.zeros_like(log_probs)
        for advantage, weight in weighted_advantages_list:
            validate_same_shape("log_probs", log_probs, "advantages", advantage)
            objective = torch.min(ratio * advantage, clipped_ratio * advantage)
            loss += -PPO._masked_mean(objective, valid_mask) * weight
            weighted_advantages += advantage * weight

        entropy = PPO._masked_mean(step.entropy, valid_mask)
        loss -= entropy_coef * entropy

        if isinstance(step, StochasticContinuousPolicyStep):
            if valid_mask is None:
                reg_term = step.mean.pow(2).mean()
            else:
                reg_term = PPO._masked_mean(step.mean.pow(2).reshape(*log_probs.shape, -1).mean(dim=-1), valid_mask)
            loss += reg_term * regularization_weight

        if valid_mask is None:
            kl_divergence = PPO.compute_kl_divergence(log_probs, log_probs_hat)
        else:
            kl_values = (torch.exp(log_probs - log_probs_hat) - 1) - (log_probs - log_probs_hat)
            kl_divergence = PPO._masked_mean(kl_values, valid_mask).detach()

        result = {
            "loss": loss,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
            **PPO._policy_metric_payload(step, ratio, weighted_advantages, clip_ratio, valid_mask),
        }
        if episode_starts is not None:
            result["next_state"] = next_state
        return result

    @staticmethod
    def compute_value_loss(
        value_model: NNMODEL,
        *,
        observations: torch.Tensor | dict[str, torch.Tensor],
        returns: torch.Tensor,
        episode_starts: torch.Tensor | None = None,
        initial_state: Any | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | Any]:
        step, next_state = PPO._call_model(value_model, (observations,), episode_starts, initial_state)
        step: ValueStep
        values = step.value

        validate_same_shape("values", values, "returns", returns)
        if episode_starts is not None:
            validate_same_shape("values", values, "episode_starts", episode_starts)
        valid_mask = PPO._prepare_valid_mask(values, valid_mask)

        if valid_mask is None:
            loss = mse(values, returns, "values")
        else:
            loss = PPO._masked_mean(0.5 * (returns - values) ** 2, valid_mask)

        result = {
            "loss": loss,
            "value_explained_variance": PPO._explained_variance(values, returns, valid_mask),
            "value_clip_fraction": torch.zeros((), dtype=values.dtype, device=values.device),
        }
        if episode_starts is not None:
            result["next_state"] = next_state
        return result

    @staticmethod
    def compute_clipped_value_loss(
        value_model: NNMODEL,
        *,
        observations: torch.Tensor | dict[str, torch.Tensor],
        values_hat: torch.Tensor,
        returns: torch.Tensor,
        clip_ratio: float,
        episode_starts: torch.Tensor | None = None,
        initial_state: Any | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | Any]:
        step, next_state = PPO._call_model(value_model, (observations,), episode_starts, initial_state)
        step: ValueStep
        values = step.value

        validate_same_shape("values", values, "values_hat", values_hat)
        validate_same_shape("values", values, "returns", returns)
        if episode_starts is not None:
            validate_same_shape("values", values, "episode_starts", episode_starts)
        valid_mask = PPO._prepare_valid_mask(values, valid_mask)

        loss_unclipped = 0.5 * (returns - values) ** 2
        values_clipped = values_hat + torch.clamp(values - values_hat, -clip_ratio, clip_ratio)
        loss_clipped = 0.5 * (returns - values_clipped) ** 2
        loss = PPO._masked_mean(torch.max(loss_unclipped, loss_clipped), valid_mask)

        result = {
            "loss": loss,
            "value_explained_variance": PPO._explained_variance(values, returns, valid_mask),
            "value_clip_fraction": PPO._value_clip_fraction(values, values_hat, clip_ratio, valid_mask),
        }
        if episode_starts is not None:
            result["next_state"] = next_state
        return result
