import torch
import torch.nn as nn
from typing import Tuple, List
from ..nn.steps import StochasticContinuousPolicyStep, ValueStep

NNMODEL = nn.Module
Q_STEPS = Tuple[ValueStep, ValueStep]


class SAC:
    @staticmethod
    def _validate_same_shape(name_a: str, tensor_a: torch.Tensor, name_b: str, tensor_b: torch.Tensor) -> None:
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"{name_a} and {name_b} must have the same shape, got {tuple(tensor_a.shape)} and {tuple(tensor_b.shape)}."
            )

    @staticmethod
    def _validate_multi_critic_inputs(critic_models: List[NNMODEL], weights: List[float]) -> None:
        if len(critic_models) == 0:
            raise ValueError("critic_models must be non-empty.")
        if len(weights) == 0:
            raise ValueError("weights must be non-empty.")
        if len(critic_models) != len(weights):
            raise ValueError(
                f"critic_models and weights must have the same length, got {len(critic_models)} and {len(weights)}."
            )

    @staticmethod
    def compute_policy_loss_with_multi_critic(
        policy_model: NNMODEL,
        critic_models: List[NNMODEL],
        weights: List[float],
        observation: torch.Tensor|dict[str, torch.Tensor],
        alpha: float,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor]:
        SAC._validate_multi_critic_inputs(critic_models, weights)
        step: StochasticContinuousPolicyStep = policy_model(observation)
        policy_loss = torch.zeros((), dtype=step.log_prob.dtype, device=step.log_prob.device)

        for weight, critic_model in zip(weights, critic_models):
            q_steps: Q_STEPS = critic_model(observation, step.action)
            q1, q2 = q_steps[0].value, q_steps[1].value
            SAC._validate_same_shape("q1", q1, "q2", q2)
            q = torch.min(q1, q2)
            SAC._validate_same_shape("log_prob", step.log_prob, "q", q)

            policy_loss += (alpha * step.log_prob - q).mean() * weight

        policy_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

        return {
            "loss": policy_loss
        }

    @staticmethod
    def compute_policy_loss(
        policy_model: NNMODEL,
        critic_model: NNMODEL,
        observation: torch.Tensor|dict[str, torch.Tensor],
        alpha: float,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor]:
        step: StochasticContinuousPolicyStep = policy_model(observation)
        q_steps: Q_STEPS = critic_model(observation, step.action)
        q1, q2 = q_steps[0].value, q_steps[1].value
        SAC._validate_same_shape("q1", q1, "q2", q2)
        q = torch.min(q1, q2)
        SAC._validate_same_shape("log_prob", step.log_prob, "q", q)

        policy_loss = (alpha * step.log_prob - q).mean()
        policy_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

        return {
            "loss": policy_loss
        }

    @staticmethod
    def compute_critic_loss(
        policy_model: NNMODEL,
        critic_model: NNMODEL,
        critic_target_model: NNMODEL,
        observation: torch.Tensor|dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor|dict[str, torch.Tensor],
        done: torch.Tensor,
        alpha: float,
        gamma: float
    ) -> dict[str, torch.Tensor]:
        q_steps: Q_STEPS = critic_model(observation, action)
        q1, q2 = q_steps[0].value, q_steps[1].value
        SAC._validate_same_shape("q1", q1, "q2", q2)

        with torch.no_grad():
            done = done.to(dtype=reward.dtype, device=reward.device)
            next_step: StochasticContinuousPolicyStep = policy_model(next_observation)
            next_q_steps: Q_STEPS = critic_target_model(next_observation, next_step.action)
            next_q1, next_q2 = next_q_steps[0].value, next_q_steps[1].value
            SAC._validate_same_shape("next_q1", next_q1, "next_q2", next_q2)
            q_next = torch.min(next_q1, next_q2)
            SAC._validate_same_shape("reward", reward, "done", done)
            SAC._validate_same_shape("reward", reward, "q_next", q_next)
            SAC._validate_same_shape("q_next", q_next, "next_log_prob", next_step.log_prob)

            q_targ = reward + gamma * (1 - done) * (q_next - alpha * next_step.log_prob)
        SAC._validate_same_shape("q1", q1, "q_target", q_targ)
        SAC._validate_same_shape("q2", q2, "q_target", q_targ)

        critic_1_loss = 0.5 * ((q_targ - q1) ** 2).mean()
        critic_2_loss = 0.5 * ((q_targ - q2) ** 2).mean()
        critic_loss = critic_1_loss + critic_2_loss

        return {
            "loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
            "q_target": q_targ.mean()
        }

    @staticmethod
    def compute_policy_loss_asymmetric_with_multi_critic(
        policy_model: NNMODEL,
        critic_models: List[NNMODEL],
        weights: List[float],
        actor_observation: torch.Tensor|dict[str, torch.Tensor],
        critic_observation: torch.Tensor|dict[str, torch.Tensor],
        alpha: float,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor]:
        SAC._validate_multi_critic_inputs(critic_models, weights)
        step: StochasticContinuousPolicyStep = policy_model(actor_observation)
        policy_loss = torch.zeros((), dtype=step.log_prob.dtype, device=step.log_prob.device)

        for weight, critic_model in zip(weights, critic_models):
            q_steps: Q_STEPS = critic_model(critic_observation, step.action)
            q1, q2 = q_steps[0].value, q_steps[1].value
            SAC._validate_same_shape("q1", q1, "q2", q2)
            q = torch.min(q1, q2)
            SAC._validate_same_shape("log_prob", step.log_prob, "q", q)

            policy_loss += (alpha * step.log_prob - q).mean() * weight

        policy_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

        return {
            "loss": policy_loss
        }

    @staticmethod
    def compute_policy_loss_asymmetric(
        policy_model: NNMODEL,
        critic_model: NNMODEL,
        actor_observation: torch.Tensor|dict[str, torch.Tensor],
        critic_observation: torch.Tensor|dict[str, torch.Tensor],
        alpha: float,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor]:
        step: StochasticContinuousPolicyStep = policy_model(actor_observation)
        q_steps: Q_STEPS = critic_model(critic_observation, step.action)
        q1, q2 = q_steps[0].value, q_steps[1].value
        SAC._validate_same_shape("q1", q1, "q2", q2)
        q = torch.min(q1, q2)
        SAC._validate_same_shape("log_prob", step.log_prob, "q", q)

        policy_loss = (alpha * step.log_prob - q).mean()
        policy_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

        return {
            "loss": policy_loss
        }

    @staticmethod
    def compute_critic_loss_asymmetric(
        policy_model: NNMODEL,
        critic_model: NNMODEL,
        critic_target_model: NNMODEL,
        critic_observation: torch.Tensor|dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        next_actor_observation: torch.Tensor|dict[str, torch.Tensor],
        next_critic_observation: torch.Tensor|dict[str, torch.Tensor],
        done: torch.Tensor,
        alpha: float,
        gamma: float
    ) -> dict[str, torch.Tensor]:
        q_steps: Q_STEPS = critic_model(critic_observation, action)
        q1, q2 = q_steps[0].value, q_steps[1].value
        SAC._validate_same_shape("q1", q1, "q2", q2)

        with torch.no_grad():
            done = done.to(dtype=reward.dtype, device=reward.device)
            next_step: StochasticContinuousPolicyStep = policy_model(next_actor_observation)
            next_q_steps: Q_STEPS = critic_target_model(next_critic_observation, next_step.action)
            next_q1, next_q2 = next_q_steps[0].value, next_q_steps[1].value
            SAC._validate_same_shape("next_q1", next_q1, "next_q2", next_q2)
            q_next = torch.min(next_q1, next_q2)
            SAC._validate_same_shape("reward", reward, "done", done)
            SAC._validate_same_shape("reward", reward, "q_next", q_next)
            SAC._validate_same_shape("q_next", q_next, "next_log_prob", next_step.log_prob)

            q_targ = reward + gamma * (1 - done) * (q_next - alpha * next_step.log_prob)
        SAC._validate_same_shape("q1", q1, "q_target", q_targ)
        SAC._validate_same_shape("q2", q2, "q_target", q_targ)

        critic_1_loss = 0.5 * ((q_targ - q1) ** 2).mean()
        critic_2_loss = 0.5 * ((q_targ - q2) ** 2).mean()
        critic_loss = critic_1_loss + critic_2_loss

        return {
            "loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
            "q_target": q_targ.mean()
        }

    @staticmethod
    def compute_alpha_loss(
        policy_model: NNMODEL,
        log_alpha: torch.Tensor,
        observation: torch.Tensor|dict[str, torch.Tensor],
        target_entropy: float
    ) -> dict[str, torch.Tensor]:
        step: StochasticContinuousPolicyStep = policy_model(observation)
        alpha_loss = -(log_alpha.exp() * (step.log_prob + target_entropy).detach()).mean()
        return {
            "alpha_loss": alpha_loss
        }

    @staticmethod
    @torch.no_grad()
    def update_target_param(model: NNMODEL, model_target: NNMODEL, tau: float):
        for param, param_target in zip(model.parameters(), model_target.parameters()):
            param_target.data.mul_(1 - tau)
            param_target.data.add_(tau * param.data)
