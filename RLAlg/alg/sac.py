import torch
import torch.nn as nn
from typing import Sequence, Tuple
from ..nn.steps import StochasticContinuousPolicyStep, ValueStep
from ._common import validate_same_shape, mse, weighted_models, soft_update, alpha_loss

NNMODEL = nn.Module
Q_STEPS = Tuple[ValueStep, ValueStep]
OBSERVATION = torch.Tensor | dict[str, torch.Tensor]


class SAC:
    @staticmethod
    def compute_policy_loss(
        policy: NNMODEL,
        critic_model: NNMODEL | Sequence[NNMODEL],
        *,
        observations: OBSERVATION,
        alpha: float,
        critic_observations: OBSERVATION | None = None,
        critic_weights: Sequence[float] | None = None,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor]:
        # The actor sees ``observations``; the critics see ``critic_observations`` (same by default).
        critic_observations = observations if critic_observations is None else critic_observations
        critics = weighted_models(critic_model, critic_weights)
        step: StochasticContinuousPolicyStep = policy(observations)
        policy_loss = torch.zeros((), dtype=step.log_prob.dtype, device=step.log_prob.device)

        for critic, weight in critics:
            q_steps: Q_STEPS = critic(critic_observations, step.action)
            q1, q2 = q_steps[0].value, q_steps[1].value
            validate_same_shape("q1", q1, "q2", q2)
            q = torch.min(q1, q2)
            validate_same_shape("log_prob", step.log_prob, "q", q)

            policy_loss += (alpha * step.log_prob - q).mean() * weight

        policy_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

        return {
            "loss": policy_loss
        }

    @staticmethod
    def compute_critic_loss(
        policy: NNMODEL,
        critic: NNMODEL,
        critic_target: NNMODEL,
        *,
        observations: OBSERVATION,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: OBSERVATION,
        terminated: torch.Tensor,
        alpha: float,
        gamma: float,
        critic_observations: OBSERVATION | None = None,
        next_critic_observations: OBSERVATION | None = None
    ) -> dict[str, torch.Tensor]:
        # The actor only evaluates the next state, so ``observations`` is used by the critic
        # unless ``critic_observations`` is given.
        critic_observations = observations if critic_observations is None else critic_observations
        next_critic_observations = next_observations if next_critic_observations is None else next_critic_observations

        q_steps: Q_STEPS = critic(critic_observations, actions)
        q1, q2 = q_steps[0].value, q_steps[1].value
        validate_same_shape("q1", q1, "q2", q2)

        with torch.no_grad():
            terminated = terminated.to(dtype=rewards.dtype, device=rewards.device)
            next_step: StochasticContinuousPolicyStep = policy(next_observations)
            next_q_steps: Q_STEPS = critic_target(next_critic_observations, next_step.action)
            next_q1, next_q2 = next_q_steps[0].value, next_q_steps[1].value
            validate_same_shape("next_q1", next_q1, "next_q2", next_q2)
            q_next = torch.min(next_q1, next_q2)
            validate_same_shape("rewards", rewards, "terminated", terminated)
            validate_same_shape("rewards", rewards, "q_next", q_next)
            validate_same_shape("q_next", q_next, "next_log_prob", next_step.log_prob)

            q_targ = rewards + gamma * (1 - terminated) * (q_next - alpha * next_step.log_prob)

        critic_loss = mse(q1, q_targ, "q1") + mse(q2, q_targ, "q2")

        return {
            "loss": critic_loss,
            "q1": q1.mean().detach(),
            "q2": q2.mean().detach(),
            "q_target": q_targ.mean()
        }

    @staticmethod
    def compute_alpha_loss(
        policy: NNMODEL,
        *,
        log_alpha: torch.Tensor,
        observations: OBSERVATION,
        target_entropy: float
    ) -> dict[str, torch.Tensor]:
        step: StochasticContinuousPolicyStep = policy(observations)
        return alpha_loss(log_alpha, step.log_prob, target_entropy)

    update_target_param = staticmethod(soft_update)
