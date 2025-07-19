import torch
import torch.nn as nn
from nn.steps import DeterministicContinuousPolicyStep, ValueStep

NNMODEL = nn.Module

class DDPG:
    @staticmethod
    def compute_critic_loss(
        actor_target_model: NNMODEL,
        critic_model: NNMODEL,
        critic_target_model: NNMODEL,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        qvalue: ValueStep = critic_model(torch.cat([observation, action], dim=-1))
        with torch.no_grad():
            next_action_step: DeterministicContinuousPolicyStep = actor_target_model(next_observation)
            next_action = next_action_step.mean
            next_qvalue_target: ValueStep = critic_target_model(
                torch.cat([next_observation, next_action], dim=-1)
            )
            q_targ = reward + gamma * (1 - done) * next_qvalue_target.value

        critic_loss = ((qvalue.value - q_targ) ** 2).mean()
        return critic_loss

    @staticmethod
    def compute_actor_loss(
        actor_model: NNMODEL,
        critic_model: NNMODEL,
        observation: torch.Tensor,
        regularization_weight: float = 0.0,
    ) -> torch.Tensor:
        action_step: DeterministicContinuousPolicyStep = actor_model(observation)
        action = action_step.mean

        q_value: ValueStep = critic_model(torch.cat([observation, action], dim=-1))
        actor_loss = -q_value.value.mean()

        actor_loss += (action ** 2).mean() * regularization_weight
        return actor_loss

    @staticmethod
    def compute_actor_loss_with_multi_critic(
        actor_model: NNMODEL,
        critic_models: list[NNMODEL],
        weights: list[float],
        observation: torch.Tensor,
        regularization_weight: float = 0.0,
    ) -> torch.Tensor:
        action_step: DeterministicContinuousPolicyStep = actor_model(observation)
        action = action_step.mean

        actor_loss = 0
        for weight, critic_model in zip(weights, critic_models):
            q_value: ValueStep = critic_model(torch.cat([observation, action], dim=-1))
            actor_loss += -q_value.value.mean() * weight

        actor_loss += (action ** 2).mean() * regularization_weight
        return actor_loss

    @staticmethod
    def compute_critic_loss_asymmetric(
        actor_target_model: NNMODEL,
        critic_model: NNMODEL,
        critic_target_model: NNMODEL,
        critic_observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_actor_observation: torch.Tensor,
        next_critic_observation: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        qvalue: ValueStep = critic_model(torch.cat([critic_observation, action], dim=-1))
        with torch.no_grad():
            next_action_step: DeterministicContinuousPolicyStep = actor_target_model(next_actor_observation)
            next_action = next_action_step.mean
            next_qvalue_target: ValueStep = critic_target_model(
                torch.cat([next_critic_observation, next_action], dim=-1)
            )
            q_targ = reward + gamma * (1 - done) * next_qvalue_target.value

        critic_loss = ((qvalue.value - q_targ) ** 2).mean()
        return critic_loss

    @staticmethod
    def compute_actor_loss_asymmetric(
        actor_model: NNMODEL,
        critic_model: NNMODEL,
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
        regularization_weight: float = 0.0,
    ) -> torch.Tensor:
        action_step: DeterministicContinuousPolicyStep = actor_model(actor_observation)
        action = action_step.mean

        q_value: ValueStep = critic_model(torch.cat([critic_observation, action], dim=-1))
        actor_loss = -q_value.value.mean()

        actor_loss += (action ** 2).mean() * regularization_weight
        return actor_loss

    @staticmethod
    def compute_actor_loss_asymmetric_with_multi_critic(
        actor_model: NNMODEL,
        critic_models: list[NNMODEL],
        weights: list[float],
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
        regularization_weight: float = 0.0,
    ) -> torch.Tensor:
        action_step: DeterministicContinuousPolicyStep = actor_model(actor_observation)
        action = action_step.mean

        actor_loss = 0
        for weight, critic_model in zip(weights, critic_models):
            q_value: ValueStep = critic_model(torch.cat([critic_observation, action], dim=-1))
            actor_loss += -q_value.value.mean() * weight

        actor_loss += (action ** 2).mean() * regularization_weight
        return actor_loss

    @staticmethod
    @torch.no_grad()
    def update_target_param(model: NNMODEL, model_target: NNMODEL, tau: float) -> None:
        for param, param_target in zip(model.parameters(), model_target.parameters()):
            param_target.data.mul_(1 - tau)
            param_target.data.add_(tau * param.data)