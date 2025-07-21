import torch
import torch.nn as nn
from typing import Tuple, List
from ..nn.steps import StochasticContinuousPolicyStep, ValueStep

NNMODEL = nn.Module
Q_STEPS = Tuple[ValueStep, ValueStep]


class SAC:
    @staticmethod
    def compute_actor_loss_with_multi_critic(
        actor_model: NNMODEL,
        critic_models: List[NNMODEL],
        weights: List[float],
        observation: torch.Tensor,
        alpha: float,
        regularization_weight: float = 0.0
    ) -> torch.Tensor:
        step: StochasticContinuousPolicyStep = actor_model(observation)
        actor_loss = 0.0

        for weight, critic_model in zip(weights, critic_models):
            q_steps: Q_STEPS = critic_model(observation, step.action)
            q1, q2 = q_steps[0].value, q_steps[1].value
            q = torch.min(q1, q2)

            actor_loss += (alpha * step.log_prob - q).mean() * weight

        actor_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

        return actor_loss

    @staticmethod
    def compute_actor_loss(
        actor_model: NNMODEL,
        critic_model: NNMODEL,
        observation: torch.Tensor,
        alpha: float,
        regularization_weight: float = 0.0
    ) -> torch.Tensor:
        step: StochasticContinuousPolicyStep = actor_model(observation)
        q_steps: Q_STEPS = critic_model(observation, step.action)
        q1, q2 = q_steps[0].value, q_steps[1].value
        q = torch.min(q1, q2)

        actor_loss = (alpha * step.log_prob - q).mean()
        actor_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

        return actor_loss

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
        alpha: float,
        gamma: float
    ) -> torch.Tensor:
        q_steps: Q_STEPS = critic_model(observation, action)
        q1, q2 = q_steps[0].value, q_steps[1].value

        with torch.no_grad():
            next_step: StochasticContinuousPolicyStep = actor_model(next_observation)
            next_q_steps: Q_STEPS = critic_target_model(next_observation, next_step.action)
            next_q1, next_q2 = next_q_steps[0].value, next_q_steps[1].value
            q_next = torch.min(next_q1, next_q2)

            q_targ = reward + gamma * (1 - done) * (q_next - alpha * next_step.log_prob)

        critic_1_loss = 0.5 * ((q_targ - q1) ** 2).mean()
        critic_2_loss = 0.5 * ((q_targ - q2) ** 2).mean()
        critic_loss = critic_1_loss + critic_2_loss

        return critic_loss

    @staticmethod
    def compute_actor_loss_asymmetric_with_multi_critic(
        actor_model: NNMODEL,
        critic_models: List[NNMODEL],
        weights: List[float],
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
        alpha: float,
        regularization_weight: float = 0.0
    ) -> torch.Tensor:
        step: StochasticContinuousPolicyStep = actor_model(actor_observation)
        actor_loss = 0.0

        for weight, critic_model in zip(weights, critic_models):
            q_steps: Q_STEPS = critic_model(critic_observation, step.action)
            q1, q2 = q_steps[0].value, q_steps[1].value
            q = torch.min(q1, q2)

            actor_loss += (alpha * step.log_prob - q).mean() * weight

        actor_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

        return actor_loss

    @staticmethod
    def compute_actor_loss_asymmetric(
        actor_model: NNMODEL,
        critic_model: NNMODEL,
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
        alpha: float,
        regularization_weight: float = 0.0
    ) -> torch.Tensor:
        step: StochasticContinuousPolicyStep = actor_model(actor_observation)
        q_steps: Q_STEPS = critic_model(critic_observation, step.action)
        q1, q2 = q_steps[0].value, q_steps[1].value
        q = torch.min(q1, q2)

        actor_loss = (alpha * step.log_prob - q).mean()
        actor_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

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
        alpha: float,
        gamma: float
    ) -> torch.Tensor:
        q_steps: Q_STEPS = critic_model(critic_observation, action)
        q1, q2 = q_steps[0].value, q_steps[1].value

        with torch.no_grad():
            next_step: StochasticContinuousPolicyStep = actor_model(next_actor_observation)
            next_q_steps: Q_STEPS = critic_target_model(next_critic_observation, next_step.action)
            next_q1, next_q2 = next_q_steps[0].value, next_q_steps[1].value
            q_next = torch.min(next_q1, next_q2)

            q_targ = reward + gamma * (1 - done) * (q_next - alpha * next_step.log_prob)

        critic_1_loss = 0.5 * ((q_targ - q1) ** 2).mean()
        critic_2_loss = 0.5 * ((q_targ - q2) ** 2).mean()
        critic_loss = critic_1_loss + critic_2_loss

        return critic_loss

    @staticmethod
    def compute_alpha_loss(
        actor_model: NNMODEL,
        log_alpha: torch.Tensor,
        observation: torch.Tensor,
        target_entropy: float
    ) -> torch.Tensor:
        step: StochasticContinuousPolicyStep = actor_model(observation)
        alpha_loss = -(log_alpha.exp() * (step.log_prob + target_entropy).detach()).mean()
        return alpha_loss

    @staticmethod
    @torch.no_grad()
    def update_target_param(model: NNMODEL, model_target: NNMODEL, tau: float):
        for param, param_target in zip(model.parameters(), model_target.parameters()):
            param_target.data.mul_(1 - tau)
            param_target.data.add_(tau * param.data)