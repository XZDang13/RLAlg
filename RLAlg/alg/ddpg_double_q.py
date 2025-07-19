import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.steps import DeterministicContinuousPolicyStep, ValueStep

NNMODEL = nn.Module
Q_STEPS = tuple[ValueStep, ValueStep]

class DDPGDoubleQ:
    @staticmethod
    def compute_critic_loss(
        actor_model: NNMODEL,
        critic_model: NNMODEL,
        critic_target_model: NNMODEL,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
        std: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:

        with torch.no_grad():
            dist: DeterministicContinuousPolicyStep = actor_model(next_observation, std)
            next_action = dist.pi.rsample()
            q_target_steps:Q_STEPS = critic_target_model(torch.cat([next_observation, next_action], dim=-1))
            q1_target_step, q2_target_step = q_target_steps
            q1_target = q1_target_step.value
            q2_target = q2_target_step.value
            next_q = torch.min(q1_target, q2_target)
            q_target = reward + gamma * (1 - done) * next_q

        q_steps:Q_STEPS = critic_model(torch.cat([observation, action], dim=-1))
        q1_step, q2_step = q_steps
        q1 = q1_step.value
        q2 = q2_step.value

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        return critic_loss

    @staticmethod
    def compute_actor_loss(
        actor_model: NNMODEL,
        critic_model: NNMODEL,
        observation: torch.Tensor,
        std: torch.Tensor,
        regularization_weight: float = 0.0,
    ) -> torch.Tensor:
        dist: DeterministicContinuousPolicyStep = actor_model(observation, std)
        action = dist.pi.rsample()
        q_steps:Q_STEPS = critic_model(torch.cat([observation, action], dim=-1))
        q1_step, q2_step = q_steps
        q1 = q1_step.value
        q2 = q2_step.value
        q = torch.min(q1, q2)

        actor_loss = -q.mean()
        actor_loss += dist.mean.pow(2).mean() * regularization_weight

        return actor_loss

    @staticmethod
    def compute_actor_loss_with_multi_critic(
        actor_model: NNMODEL,
        critic_models: list[NNMODEL],
        weights: list[float],
        observation: torch.Tensor,
        std: torch.Tensor,
        regularization_weight: float = 0.0,
    ) -> torch.Tensor:
        dist: DeterministicContinuousPolicyStep = actor_model(observation, std)
        action = dist.pi.rsample()
        actor_loss = 0

        for weight, critic_model in zip(weights, critic_models):
            q_steps:Q_STEPS = critic_model(torch.cat([observation, action], dim=-1))
            q1_step, q2_step = q_steps
            q1 = q1_step.value
            q2 = q2_step.value
            q = torch.min(q1, q2)
            actor_loss += -q.mean() * weight

        actor_loss += dist.mean.pow(2).mean() * regularization_weight

        return actor_loss

    @staticmethod
    def compute_critic_loss_asymmetric(
        actor_model: NNMODEL,
        critic_model: NNMODEL,
        critic_target_model: NNMODEL,
        critic_observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_actor_observation: torch.Tensor,
        next_critic_observation: torch.Tensor,
        done: torch.Tensor,
        std: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        with torch.no_grad():
            dist: DeterministicContinuousPolicyStep = actor_model(next_actor_observation, std)
            next_action = dist.pi.rsample()
            q_target_steps:Q_STEPS = critic_target_model(torch.cat([next_critic_observation, next_action], dim=-1))
            q1_target_step, q2_target_step = q_target_steps
            q1_target = q1_target_step.value
            q2_target = q2_target_step.value
            next_q = torch.min(q1_target, q2_target)
            q_target = reward + gamma * (1 - done) * next_q

        q_steps:Q_STEPS = critic_model(torch.cat([critic_observation, action], dim=-1))
        q1_step, q2_step = q_steps
        q1 = q1_step.value
        q2 = q2_step.value

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        return critic_loss

    @staticmethod
    def compute_actor_loss_asymmetric(
        actor_model: NNMODEL,
        critic_model: NNMODEL,
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
        std: torch.Tensor,
        regularization_weight: float = 0.0,
    ) -> torch.Tensor:
        dist: DeterministicContinuousPolicyStep = actor_model(actor_observation, std)
        action = dist.pi.rsample()
        q_steps:Q_STEPS = critic_model(torch.cat([critic_observation, action], dim=-1))
        q1_step, q2_step = q_steps
        q1 = q1_step.value
        q2 = q2_step.value
        q = torch.min(q1, q2)

        actor_loss = -q.mean()
        actor_loss += dist.mean.pow(2).mean() * regularization_weight

        return actor_loss

    @staticmethod
    def compute_actor_loss_asymmetric_with_multi_critic(
        actor_model: NNMODEL,
        critic_models: list[NNMODEL],
        weights: list[float],
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
        std: torch.Tensor,
        regularization_weight: float = 0.0,
    ) -> torch.Tensor:
        dist: DeterministicContinuousPolicyStep = actor_model(actor_observation, std)
        action = dist.pi.rsample()
        actor_loss = 0

        for weight, critic_model in zip(weights, critic_models):
            q_steps:Q_STEPS = critic_model(torch.cat([critic_observation, action], dim=-1))
            q1_step, q2_step = q_steps
            q1 = q1_step.value
            q2 = q2_step.value
            q = torch.min(q1, q2)
            actor_loss += -q.mean() * weight

        actor_loss += dist.mean.pow(2).mean() * regularization_weight

        return actor_loss

    @staticmethod
    @torch.no_grad()
    def update_target_param(model: NNMODEL, model_target: NNMODEL, tau: float) -> None:
        for param, param_target in zip(model.parameters(), model_target.parameters()):
            param_target.data.mul_(1 - tau)
            param_target.data.add_(tau * param.data)