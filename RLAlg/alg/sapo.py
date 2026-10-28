import torch
import torch.nn as nn
from ..nn.steps import ValueStep
from ._common import validate_same_shape
from .shac import SHAC

NNMODEL = nn.Module


class SAPO(SHAC):
    """Soft Analytic Policy Optimization losses (Xing et al., ICLR 2025).

    SAPO is maximum-entropy SHAC: every reward becomes the soft reward
    ``r_t + alpha * normalized_entropy_t`` in both the analytic policy return and
    the TD(lambda) value targets, and the temperature ``alpha`` is tuned toward a
    target entropy. The value function is an ensemble of critics without a
    target network: the policy bootstraps from the critics' mean and the value
    targets from their minimum. Rollout tensors follow SHAC: time-major
    ``[H, N]``. ``entropies[t]`` is the policy entropy at step ``t`` summed over
    action dimensions; for a tanh-squashed Gaussian use the one-sample estimate
    ``-log_prob`` of a reparameterized action, keeping its autograd graph.

    Entropies are normalized as in the reference implementation,
    ``(entropy + |target_entropy|) / (2 * |target_entropy|)``, which maps the
    target entropy to 0 and ``entropy = |target_entropy|`` to 1.
    """

    @staticmethod
    def normalize_entropy(entropies: torch.Tensor, target_entropy: float) -> torch.Tensor:
        """``(entropy + |target_entropy|) * 0.5 / |target_entropy|``, keeping the autograd graph."""
        scale = abs(target_entropy)
        if scale == 0.0:
            raise ValueError("target_entropy must be non-zero.")
        return (entropies + scale) * (0.5 / scale)

    @staticmethod
    def _soft_rewards(
        rewards: torch.Tensor, entropies: torch.Tensor, alpha: float, target_entropy: float
    ) -> torch.Tensor:
        validate_same_shape("rewards", rewards, "entropies", entropies)
        return rewards + alpha * SAPO.normalize_entropy(entropies, target_entropy)

    @staticmethod
    def compute_policy_loss(
        *,
        rewards: torch.Tensor,
        entropies: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
        gamma: float,
        alpha: float,
        target_entropy: float,
        truncated: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Negative short-horizon soft return, divided by ``H * N``.

        This is the SHAC loss on the soft rewards, so the entropy bonus is
        discounted with the rewards. ``alpha`` is the detached temperature at
        rollout time. ``next_values`` should be the critics' mean
        (``compute_next_values(...)[1]``).
        """
        soft_rewards = SAPO._soft_rewards(rewards, entropies, alpha, target_entropy)
        result = SHAC.compute_policy_loss(
            rewards=soft_rewards, next_values=next_values, terminated=terminated, gamma=gamma,
            truncated=truncated,
        )
        result["reward"] = rewards.detach().mean()
        result["entropy"] = entropies.detach().mean()
        return result

    @staticmethod
    @torch.no_grad()
    def compute_td_lambda_targets(
        *,
        rewards: torch.Tensor,
        entropies: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
        gamma: float,
        lambda_: float,
        alpha: float,
        target_entropy: float,
        truncated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SHAC TD(lambda) targets on the soft rewards, returned as ``[H, N]``.

        ``next_values`` should be the critics' minimum
        (``compute_next_values(...)[0]``). The reference computes the targets
        after the temperature update, so pass the updated ``alpha``.
        """
        soft_rewards = SAPO._soft_rewards(rewards.detach(), entropies.detach(), alpha, target_entropy)
        return SHAC.compute_td_lambda_targets(
            rewards=soft_rewards, next_values=next_values, terminated=terminated, gamma=gamma,
            lambda_=lambda_, truncated=truncated,
        )

    @staticmethod
    def _critic_values(value_model: NNMODEL, observations: torch.Tensor|dict[str, torch.Tensor]) -> torch.Tensor:
        value_steps: tuple[ValueStep, ...] = value_model(observations)
        if len(value_steps) < 1:
            raise ValueError("value_model must return at least one ValueStep.")
        return torch.stack([step.value for step in value_steps])

    @staticmethod
    def compute_next_values(
        value_model: NNMODEL, observations: torch.Tensor|dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(min_k V_k, mean_k V_k)`` over the critic ensemble, keeping the autograd graph.

        ``value_model`` returns one ``ValueStep`` per critic. The minimum goes to
        the value targets and the mean to the policy loss, as in the reference.
        """
        values = SAPO._critic_values(value_model, observations)
        return values.min(dim=0).values, values.mean(dim=0)

    @staticmethod
    def compute_value_loss(
        value_model: NNMODEL,
        *,
        observations: torch.Tensor|dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """``mean_k mean((V_k - target)^2)`` over the critic ensemble, as in the reference.

        Unlike the library's other value losses this has no ``0.5`` factor, so
        gradient-norm clipping matches the reference.
        """
        values = SAPO._critic_values(value_model, observations)
        targets = targets.detach()
        validate_same_shape("values", values[0], "targets", targets)
        loss = (values - targets.unsqueeze(0)).pow(2).mean()

        return {
            "loss": loss,
            "value": values.detach().mean(),
            "target": targets.mean(),
        }

    @staticmethod
    def compute_alpha_loss(
        log_alpha: torch.Tensor, entropies: torch.Tensor, target_entropy: float
    ) -> dict[str, torch.Tensor]:
        """``|target_entropy| * alpha * mean(entropy - target_entropy)`` with ``alpha = exp(log_alpha)``.

        ``entropies`` are the raw (unnormalized) rollout entropies. Scaling
        ``alpha`` by ``|target_entropy|`` undoes the entropy normalization, as in
        the reference. The temperature falls when the entropy is above target.
        """
        alpha = log_alpha.exp()
        loss = abs(target_entropy) * alpha * (entropies.detach() - target_entropy).mean()
        return {"loss": loss, "alpha": alpha.detach()}
