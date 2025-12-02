import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn.steps import StochasticContinuousPolicyStep, DistributionStep

NNMODEL = nn.Module
Q_STEPS = tuple[DistributionStep, DistributionStep]


class DSAC:

    @staticmethod
    def compute_policy_loss(
        policy_model: NNMODEL,
        critic_model: NNMODEL,
        observation: torch.Tensor|dict[str, torch.Tensor],
        alpha: float,
        regularization_weight: float = 0.0
    ) -> dict[str, torch.Tensor]:
        step: StochasticContinuousPolicyStep = policy_model(observation)

        q1_step, q2_step = critic_model(observation, step.action)
        q1 = q1_step.mean
        q2 = q2_step.mean
        q = torch.min(q1, q2)

        policy_loss = (alpha * step.log_prob - q).mean()
        policy_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

        return {
            "loss": policy_loss
        }

    @staticmethod
    def compute_q_targ(
        reward: torch.Tensor,
        done: torch.Tensor,
        q: torch.Tensor,
        q_next: torch.Tensor,
        next_log_prob: torch.Tensor,
        alpha: float,
        gamma: float,
        td_bound: float
    ) -> tuple[torch.Tensor, torch.Tensor]:

        q_target = reward + (1 - done) * gamma * (q_next - alpha * next_log_prob)
        difference = torch.clamp(q_target - q, -td_bound, td_bound)
        q_target_bound = q + difference

        return q_target, q_target_bound

    @staticmethod
    def compute_critic_loss(
        actor_target_model: NNMODEL,
        critic_model: NNMODEL,
        critic_target_model: NNMODEL,
        observation: torch.Tensor|dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor|dict[str, torch.Tensor],
        done: torch.Tensor,
        alpha: float,
        gamma: float,
        td_bound: float
    ) -> dict[str, torch.Tensor]:

        q1_step, q2_step = critic_model(observation, action)
        q1 = q1_step.mean
        q2 = q2_step.mean
        q1_std = q1_step.std
        q2_std = q2_step.std

        with torch.no_grad():
            next_step: StochasticContinuousPolicyStep = actor_target_model(next_observation)

            next_q1_step, next_q2_step = critic_target_model(next_observation, next_step.action)
            next_q1_sample = next_q1_step.sample
            next_q2_sample = next_q2_step.sample
            next_q_sample = torch.min(next_q1_sample, next_q2_sample)

            q_targ, q_targ_bound = DSAC.compute_q_targ(
                reward,
                done,
                torch.min(q1.detach(), q2.detach()),
                next_q_sample.detach(),
                next_step.log_prob.detach(),
                alpha,
                gamma,
                td_bound
            )

        bias = 1e-5
        q1_std_detach = torch.clamp(q1_std, min=0).detach()
        q2_std_detach = torch.clamp(q2_std, min=0).detach()

        # loss for q1
        loss_q1 = (
            -((q_targ - q1).detach()) / (q1_std_detach**2 + bias) * q1
            - (((q1.detach() - q_targ_bound)**2 - q1_std_detach**2) / (q1_std_detach**3 + bias)) * q1_std
        )

        # loss for q2
        loss_q2 = (
            -((q_targ - q2).detach()) / (q2_std_detach**2 + bias) * q2
            - (((q2.detach() - q_targ_bound)**2 - q2_std_detach**2) / (q2_std_detach**3 + bias)) * q2_std
        )

        critic_loss = (loss_q1 + loss_q2).mean()

        return {
            "loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
            "q_target": q_targ.mean(),
            "q_target_bound": q_targ_bound.mean()
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