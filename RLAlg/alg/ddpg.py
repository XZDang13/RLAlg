from typing import Sequence

import torch
import torch.nn as nn
from ..nn.steps import DeterministicContinuousPolicyStep, ValueStep
from ._common import mse, soft_update, validate_same_shape, weighted_models

NNMODEL = nn.Module
OBSERVATION = torch.Tensor | dict[str, torch.Tensor]


class DDPG:
    @staticmethod
    def compute_critic_loss(
        actor_target_model: NNMODEL,
        critic_model: NNMODEL,
        critic_target_model: NNMODEL,
        *,
        observations: OBSERVATION,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: OBSERVATION,
        terminated: torch.Tensor,
        gamma: float = 0.99,
        critic_observations: OBSERVATION | None = None,
        next_critic_observations: OBSERVATION | None = None,
    ) -> dict[str, torch.Tensor]:
        # The target actor sees next_observations; the critics see the (optional) privileged critic observations.
        if critic_observations is None:
            critic_observations = observations
        if next_critic_observations is None:
            next_critic_observations = next_observations

        q_step: ValueStep = critic_model(critic_observations, actions)
        q = q_step.value
        with torch.no_grad():
            terminated = terminated.to(dtype=rewards.dtype, device=rewards.device)
            next_action_step: DeterministicContinuousPolicyStep = actor_target_model(next_observations)
            next_action = next_action_step.mean
            next_q_step: ValueStep = critic_target_model(next_critic_observations, next_action)
            next_q = next_q_step.value
            validate_same_shape("rewards", rewards, "terminated", terminated)
            validate_same_shape("rewards", rewards, "next_q", next_q)
            q_targ = rewards + gamma * (1 - terminated) * next_q

        critic_loss = mse(q, q_targ, "q")

        return {
            "loss": critic_loss,
            "q": q.detach().mean(),
            "q_target": q_targ.mean()
        }

    @staticmethod
    def compute_policy_loss(
        policy_model: NNMODEL,
        critic_model: NNMODEL | Sequence[NNMODEL],
        *,
        observations: OBSERVATION,
        regularization_weight: float = 0.0,
        critic_observations: OBSERVATION | None = None,
        critic_weights: Sequence[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        critics = weighted_models(critic_model, critic_weights)
        if critic_observations is None:
            critic_observations = observations

        action_step: DeterministicContinuousPolicyStep = policy_model(observations)
        action = action_step.mean

        policy_loss = torch.zeros((), dtype=action.dtype, device=action.device)
        for critic, weight in critics:
            q_value: ValueStep = critic(critic_observations, action)
            policy_loss = policy_loss - q_value.value.mean() * weight

        policy_loss = policy_loss + (action ** 2).mean() * regularization_weight

        return {
            "loss": policy_loss
        }

    update_target_param = staticmethod(soft_update)
