from typing import Sequence

import torch
import torch.nn as nn
from ..nn.steps import DeterministicContinuousPolicyStep, ValueStep
from ._common import mse, soft_update, validate_same_shape, weighted_models

NNMODEL = nn.Module
OBSERVATION = torch.Tensor | dict[str, torch.Tensor]
Q_STEPS = tuple[ValueStep, ValueStep]


class DDPGDoubleQ:
    @staticmethod
    def perturb_action(step: DeterministicContinuousPolicyStep, std: torch.Tensor | float) -> torch.Tensor:
        """Reparameterized Gaussian noise on the policy action, added before the tanh when squashed."""
        if step.pre_tanh_mean is None:
            return step.mean + std * torch.randn_like(step.mean)
        noisy = step.pre_tanh_mean + std * torch.randn_like(step.pre_tanh_mean)
        return step.max_action * torch.tanh(noisy)

    @staticmethod
    def compute_critic_loss(
        policy_model: NNMODEL,
        critic_model: NNMODEL,
        critic_target_model: NNMODEL,
        *,
        observations: OBSERVATION,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: OBSERVATION,
        terminated: torch.Tensor,
        std: torch.Tensor | float,
        gamma: float = 0.99,
        critic_observations: OBSERVATION | None = None,
        next_critic_observations: OBSERVATION | None = None,
    ) -> dict[str, torch.Tensor]:
        # The policy sees next_observations; the critics see the (optional) privileged critic observations.
        if critic_observations is None:
            critic_observations = observations
        if next_critic_observations is None:
            next_critic_observations = next_observations

        with torch.no_grad():
            terminated = terminated.to(dtype=rewards.dtype, device=rewards.device)
            next_action = DDPGDoubleQ.perturb_action(policy_model(next_observations), std)
            q_target_steps: Q_STEPS = critic_target_model(next_critic_observations, next_action)
            q1_target_step, q2_target_step = q_target_steps
            q1_target = q1_target_step.value
            q2_target = q2_target_step.value
            next_q = torch.min(q1_target, q2_target)
            validate_same_shape("rewards", rewards, "terminated", terminated)
            validate_same_shape("rewards", rewards, "next_q", next_q)
            q_target = rewards + gamma * (1 - terminated) * next_q

        q_steps: Q_STEPS = critic_model(critic_observations, actions)
        q1_step, q2_step = q_steps
        q1 = q1_step.value
        q2 = q2_step.value

        critic_loss = mse(q1, q_target, "q1") + mse(q2, q_target, "q2")

        return {
            "loss": critic_loss,
            "q1": q1.detach().mean(),
            "q2": q2.detach().mean(),
            "q_target": q_target.mean()
        }

    @staticmethod
    def compute_policy_loss(
        policy_model: NNMODEL,
        critic_model: NNMODEL | Sequence[NNMODEL],
        *,
        observations: OBSERVATION,
        std: torch.Tensor | float,
        regularization_weight: float = 0.0,
        critic_observations: OBSERVATION | None = None,
        critic_weights: Sequence[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        critics = weighted_models(critic_model, critic_weights)
        if critic_observations is None:
            critic_observations = observations

        dist: DeterministicContinuousPolicyStep = policy_model(observations)
        action = DDPGDoubleQ.perturb_action(dist, std)

        policy_loss = torch.zeros((), dtype=action.dtype, device=action.device)
        for critic, weight in critics:
            q_steps: Q_STEPS = critic(critic_observations, action)
            q1_step, q2_step = q_steps
            q = torch.min(q1_step.value, q2_step.value)
            policy_loss = policy_loss - q.mean() * weight

        policy_loss = policy_loss + dist.mean.pow(2).mean() * regularization_weight

        return {
            "loss": policy_loss
        }

    update_target_param = staticmethod(soft_update)
