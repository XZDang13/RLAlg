from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import ComposeTransform, Normal, TanhTransform, Transform, TransformedDistribution
from torch.distributions.kl import kl_divergence
from ..nn.steps import StochasticContinuousPolicyStep
from ._common import validate_same_shape
from .sapo import SAPO

NNMODEL = nn.Module


class RPO(SAPO):
    """Reparameterization Proximal Policy Optimization losses (Zhong et al., ICML 2026).

    RPO reuses one batch of BPTT gradients for several policy epochs. After a
    SAPO-style rollout (soft rewards, critic-mean bootstrap) the gradient of the
    rollout loss with respect to every action is computed once and cached
    (``compute_action_gradients``). Each policy epoch then regenerates the stored
    actions under the current policy, ``a = mu + sigma * eps_reg`` with
    ``eps_reg = (a - mu) / sigma`` detached, and backpropagates the cached
    gradients through this new path, weighted by the importance ratio
    ``rho = pi(a|s) / pi_old(a|s)`` and zeroed outside
    ``[1 - clip_low, 1 + clip_high]``. A KL penalty to the behavior policy and an
    optional entropy bonus regularize the update.

    The value side is SAPO's: TD(lambda) targets on the soft rewards and the mean
    squared error averaged over a critic ensemble. The paper says the targets use
    the critics' mean (``compute_next_values(...)[1]``), whereas SAPO's reference
    uses the minimum.
    """

    @staticmethod
    def compute_action_gradients(
        *,
        actions: torch.Tensor | Sequence[torch.Tensor],
        rewards: torch.Tensor,
        entropies: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
        gamma: float,
        alpha: float,
        target_entropy: float,
        truncated: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Cache ``d loss / d a_t`` of the SAPO rollout loss, returned as ``[H, N, A]``.

        ``actions`` are the per-step action tensors that were fed to the
        environment (a list of ``[N, A]``, or the ``[H, N, A]`` tensor if the
        rollout indexed into it), still in the autograd graph. Each gradient is
        the total derivative through all later dynamics, rewards, entropies and
        bootstrap values, so backpropagating it through the local action path
        ``a_t(theta; s_t)`` recovers the full BPTT policy gradient. The graph is
        freed. ``alpha=0`` gives the SHAC objective. ``"loss"`` is the detached
        rollout loss for logging.
        """
        result = SAPO.compute_policy_loss(
            rewards=rewards, entropies=entropies, next_values=next_values, terminated=terminated,
            gamma=gamma, alpha=alpha, target_entropy=target_entropy, truncated=truncated,
        )
        action_list = [actions] if torch.is_tensor(actions) else list(actions)
        if not action_list:
            raise ValueError("actions must be a tensor or a non-empty sequence of tensors.")
        if torch.is_tensor(actions):
            leading = tuple(actions.shape[:2])
        else:
            same = all(action.shape == action_list[0].shape for action in action_list)
            leading = (len(action_list), action_list[0].shape[0]) if same and action_list[0].ndim else None
        # Checked before autograd.grad, which frees the rollout graph.
        if leading != tuple(rewards.shape):
            raise ValueError(f"actions must have shape [H, N, ...] matching rewards {tuple(rewards.shape)}.")
        gradients = torch.autograd.grad(result["loss"], action_list, allow_unused=True)
        gradients = [
            torch.zeros_like(action) if gradient is None else gradient
            for action, gradient in zip(action_list, gradients)
        ]
        action_gradients = gradients[0] if torch.is_tensor(actions) else torch.stack(gradients)

        result["loss"] = result["loss"].detach()
        result["action_gradients"] = action_gradients
        return result

    @staticmethod
    def _inverse(transform: Transform, value: torch.Tensor) -> torch.Tensor:
        if isinstance(transform, ComposeTransform):
            for part in reversed(transform.parts):
                value = RPO._inverse(part, value)
            return value
        if isinstance(transform, TanhTransform):
            # Same clamp as GaussianHead.log_prob, so saturated actions stay finite.
            eps = 1e-6
            return torch.atanh(value.clamp(-1 + eps, 1 - eps))
        return transform.inv(value)

    @staticmethod
    def _base_normal(pi: TransformedDistribution | Normal) -> tuple[Normal, list[Transform]]:
        if isinstance(pi, Normal):
            return pi, []
        if isinstance(pi, TransformedDistribution) and isinstance(pi.base_dist, Normal):
            return pi.base_dist, list(pi.transforms)
        raise ValueError("RPO requires a Gaussian policy, optionally with invertible transforms (e.g. tanh).")

    @staticmethod
    def regenerate_actions(pi: TransformedDistribution | Normal, actions: torch.Tensor) -> torch.Tensor:
        """``f_theta(eps_reg; s)`` with ``eps_reg = f_theta^{-1}(a; s)`` detached.

        The result equals the stored ``actions`` (up to the tanh clamp) but is a
        differentiable function of the current mean and standard deviation.
        """
        base, transforms = RPO._base_normal(pi)
        pre_transform = actions.detach()
        for transform in reversed(transforms):
            pre_transform = RPO._inverse(transform, pre_transform)
        noise = ((pre_transform - base.loc) / base.scale).detach()
        regenerated = base.loc + base.scale * noise
        for transform in transforms:
            regenerated = transform(regenerated)
        return regenerated

    @staticmethod
    def compute_policy_loss(
        policy_model: NNMODEL,
        *,
        observations: torch.Tensor|dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_gradients: torch.Tensor,
        log_probs_hat: torch.Tensor,
        means_hat: torch.Tensor,
        stds_hat: torch.Tensor,
        clip_low: float,
        clip_high: float,
        kl_coef: float,
        entropy_coef: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """One RPO policy epoch on the cached rollout.

        ``observations``, ``actions`` and ``action_gradients`` hold the detached
        rollout, e.g. ``[H, N, ...]``; ``policy_model(observations, actions)``
        returns a ``StochasticContinuousPolicyStep``. ``log_probs_hat`` are the
        behavior log-probabilities of ``actions`` and ``means_hat`` / ``stds_hat``
        the behavior Gaussian before any squashing (``step.pi.base_dist.loc`` and
        ``.scale``). The loss is

        ``sum(mask * sg(rho) * <a_reg, g>) + kl_coef * KL(pi_old || pi) - entropy_coef * H(pi)``

        with ``mask = 1 - clip_low <= rho <= 1 + clip_high``. The surrogate is a
        sum, not a mean, because ``g`` already carries SAPO's ``1 / (H * N)``;
        in the first epoch (``rho = 1``) its gradient is the SAPO/SHAC policy
        gradient through the actions. The KL is the analytic Gaussian KL summed
        over action dimensions (unchanged by tanh) and, like the entropy, is
        averaged over states.
        """
        if not (0.0 <= clip_low <= 1.0 and clip_high >= 0.0):
            raise ValueError("clip_low must be in [0, 1] and clip_high non-negative.")
        actions = actions.detach()
        action_gradients = action_gradients.detach()
        validate_same_shape("actions", actions, "action_gradients", action_gradients)
        validate_same_shape("actions", actions, "means_hat", means_hat)
        validate_same_shape("actions", actions, "stds_hat", stds_hat)

        step: StochasticContinuousPolicyStep = policy_model(observations, actions)
        log_probs = step.log_prob
        validate_same_shape("log_probs", log_probs, "log_probs_hat", log_probs_hat)
        validate_same_shape("actions", actions[..., 0], "log_probs", log_probs)

        ratio = (log_probs - log_probs_hat.detach()).exp().detach()
        in_range = (ratio >= 1.0 - clip_low) & (ratio <= 1.0 + clip_high)
        weight = torch.where(in_range, ratio, torch.zeros_like(ratio))

        regenerated = RPO.regenerate_actions(step.pi, actions)
        surrogate = (weight * (regenerated * action_gradients).sum(dim=-1)).sum()

        base, _ = RPO._base_normal(step.pi)
        behavior = Normal(means_hat.detach(), stds_hat.detach())
        kl = kl_divergence(behavior, base).sum(dim=-1).mean()
        entropy = step.entropy.mean()

        loss = surrogate + kl_coef * kl - entropy_coef * entropy

        return {
            "loss": loss,
            "surrogate": surrogate.detach(),
            "kl_divergence": kl.detach(),
            "entropy": entropy.detach(),
            "ratio": ratio.mean(),
            "policy_clip_fraction": 1.0 - in_range.to(dtype=ratio.dtype).mean(),
        }
