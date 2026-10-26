import torch
import torch.nn as nn
from typing import Sequence
from ..nn.steps import StochasticContinuousPolicyStep, DistributionStep
from ._common import validate_same_shape, weighted_models, soft_update, alpha_loss

NNMODEL = nn.Module
Q_STEPS = tuple[DistributionStep, DistributionStep]
OBSERVATION = torch.Tensor | dict[str, torch.Tensor]


class DSAC:

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
        critic_observations = observations if critic_observations is None else critic_observations
        critics = weighted_models(critic_model, critic_weights)
        step: StochasticContinuousPolicyStep = policy(observations)
        policy_loss = torch.zeros((), dtype=step.log_prob.dtype, device=step.log_prob.device)

        for critic, weight in critics:
            q1_step, q2_step = critic(critic_observations, step.action)
            q1 = q1_step.mean
            q2 = q2_step.mean
            validate_same_shape("q1", q1, "q2", q2)
            q = torch.min(q1, q2)
            validate_same_shape("log_prob", step.log_prob, "q", q)

            policy_loss += (alpha * step.log_prob - q).mean() * weight

        policy_loss += (step.mean.pow(2).mean() + step.log_std.pow(2).mean()) * regularization_weight

        return {
            "loss": policy_loss
        }

    @staticmethod
    def compute_q_targ(
        *,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        q: torch.Tensor,
        q_next: torch.Tensor,
        next_log_prob: torch.Tensor,
        alpha: float,
        gamma: float,
        td_bound: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = terminated.to(dtype=rewards.dtype, device=rewards.device)
        validate_same_shape("rewards", rewards, "terminated", terminated)
        validate_same_shape("rewards", rewards, "q_next", q_next)
        validate_same_shape("q_next", q_next, "next_log_prob", next_log_prob)
        q_target = rewards + (1 - terminated) * gamma * (q_next - alpha * next_log_prob)
        validate_same_shape("q", q, "q_target", q_target)
        difference = torch.clamp(q_target - q, -td_bound, td_bound)
        q_target_bound = q + difference

        return q_target, q_target_bound

    @staticmethod
    def compute_critic_loss(
        policy_target: NNMODEL,
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
        td_bound: float,
        critic_observations: OBSERVATION | None = None,
        next_critic_observations: OBSERVATION | None = None
    ) -> dict[str, torch.Tensor]:
        critic_observations = observations if critic_observations is None else critic_observations
        next_critic_observations = next_observations if next_critic_observations is None else next_critic_observations

        q1_step, q2_step = critic(critic_observations, actions)
        q1 = q1_step.mean
        q2 = q2_step.mean
        q1_std = q1_step.std
        q2_std = q2_step.std
        validate_same_shape("q1", q1, "q2", q2)

        with torch.no_grad():
            next_step: StochasticContinuousPolicyStep = policy_target(next_observations)

            next_q1_step, next_q2_step = critic_target(next_critic_observations, next_step.action)
            next_q1_sample = next_q1_step.sample
            next_q2_sample = next_q2_step.sample
            validate_same_shape("next_q1_sample", next_q1_sample, "next_q2_sample", next_q2_sample)
            next_q_sample = torch.min(next_q1_sample, next_q2_sample)

            q_targ, q_targ_bound = DSAC.compute_q_targ(
                rewards=rewards,
                terminated=terminated,
                q=torch.min(q1.detach(), q2.detach()),
                q_next=next_q_sample.detach(),
                next_log_prob=next_step.log_prob.detach(),
                alpha=alpha,
                gamma=gamma,
                td_bound=td_bound
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
            "q1": q1.mean().detach(),
            "q2": q2.mean().detach(),
            "q_target": q_targ.mean(),
            "q_target_bound": q_targ_bound.mean()
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
