import torch
import torch.nn as nn
from typing import Sequence
from ..nn.steps import StochasticContinuousPolicyStep, DistributionStep
from ._common import validate_same_shape, weighted_models, soft_update, alpha_loss

NNMODEL = nn.Module
Q_STEPS = tuple[DistributionStep, DistributionStep]
OBSERVATION = torch.Tensor | dict[str, torch.Tensor]


class DSACT:

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
            q_steps: Q_STEPS = critic(critic_observations, step.action)
            q1 = q_steps[0].mean
            q2 = q_steps[1].mean
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
        q_std: torch.Tensor,
        q_next: torch.Tensor,
        q_next_sample: torch.Tensor,
        next_log_prob: torch.Tensor,
        alpha: float,
        gamma: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = terminated.to(dtype=rewards.dtype, device=rewards.device)
        validate_same_shape("rewards", rewards, "terminated", terminated)
        validate_same_shape("rewards", rewards, "q_next", q_next)
        validate_same_shape("q_next", q_next, "q_next_sample", q_next_sample)
        validate_same_shape("q_next", q_next, "next_log_prob", next_log_prob)
        q_targ = rewards + (1 - terminated) * gamma * (q_next - alpha * next_log_prob)
        q_targ_sample = rewards + (1 - terminated) * gamma * (q_next_sample - alpha * next_log_prob)
        validate_same_shape("q", q, "q_target", q_targ)

        td_bound = 3 * q_std
        difference = torch.clamp(q_targ_sample - q, -td_bound, td_bound)
        q_targ_bound = q + difference

        return q_targ, q_targ_bound

    @staticmethod
    def update_mean_std(
        *,
        q_mean_std: torch.Tensor | None,
        q1_std: torch.Tensor,
        q2_std: torch.Tensor,
        tau_b: float
    ) -> torch.Tensor:
        """Return the running means of the q1/q2 std (shape ``[2]``) after one EMA step.

        ``None`` (first call) initialises the statistics to the current batch means.
        """
        current = torch.stack((q1_std.detach().mean(), q2_std.detach().mean()))
        if q_mean_std is None:
            return current
        q_mean_std = torch.as_tensor(q_mean_std, dtype=current.dtype, device=current.device)
        if q_mean_std.shape != (2,):
            raise ValueError(f"q_mean_std must have shape (2,), got {tuple(q_mean_std.shape)}.")
        return (1 - tau_b) * q_mean_std.detach() + tau_b * current

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
        tau_b: float,
        q_mean_std: torch.Tensor | None,
        critic_observations: OBSERVATION | None = None,
        next_critic_observations: OBSERVATION | None = None
    ) -> dict[str, torch.Tensor]:
        # ``q_mean_std`` holds the running means of the q1/q2 std; pass ``None`` on the first call
        # and feed the returned ``"q_mean_std"`` back on the next one.
        critic_observations = observations if critic_observations is None else critic_observations
        next_critic_observations = next_observations if next_critic_observations is None else next_critic_observations

        q_steps: Q_STEPS = critic(critic_observations, actions)
        q1, q2 = q_steps[0].mean, q_steps[1].mean
        q1_std, q2_std = q_steps[0].std, q_steps[1].std
        validate_same_shape("q1", q1, "q2", q2)

        q_mean_std = DSACT.update_mean_std(q_mean_std=q_mean_std, q1_std=q1_std, q2_std=q2_std, tau_b=tau_b)
        q1_mean_std, q2_mean_std = q_mean_std[0], q_mean_std[1]

        with torch.no_grad():
            next_step: StochasticContinuousPolicyStep = policy(next_observations)

            next_q_steps: Q_STEPS = critic_target(next_critic_observations, next_step.action)
            next_q1, next_q2 = next_q_steps[0].mean, next_q_steps[1].mean
            next_q1_sample, next_q2_sample = next_q_steps[0].sample, next_q_steps[1].sample
            validate_same_shape("next_q1", next_q1, "next_q2", next_q2)

            q_next = torch.min(next_q1, next_q2)
            q_next_sample = torch.where(next_q1 < next_q2, next_q1_sample, next_q2_sample)

            q1_targ, q1_targ_bound = DSACT.compute_q_targ(
                rewards=rewards, terminated=terminated, q=q1, q_std=q1_mean_std, q_next=q_next,
                q_next_sample=q_next_sample, next_log_prob=next_step.log_prob, alpha=alpha, gamma=gamma
            )
            q2_targ, q2_targ_bound = DSACT.compute_q_targ(
                rewards=rewards, terminated=terminated, q=q2, q_std=q2_mean_std, q_next=q_next,
                q_next_sample=q_next_sample, next_log_prob=next_step.log_prob, alpha=alpha, gamma=gamma
            )

        q1_std_detach = torch.clamp(q1_std, min=0.).detach()
        q2_std_detach = torch.clamp(q2_std, min=0.).detach()
        bias = 1e-5

        critic_1_loss = (q1_mean_std**2 + bias) * torch.mean(
            -((q1_targ - q1).detach()) / (q1_std_detach**2 + bias) * q1
            - (((q1.detach() - q1_targ_bound)**2 - q1_std_detach**2) / (q1_std_detach**3 + bias)) * q1_std
        )

        critic_2_loss = (q2_mean_std**2 + bias) * torch.mean(
            -((q2_targ - q2).detach()) / (q2_std_detach**2 + bias) * q2
            - (((q2.detach() - q2_targ_bound)**2 - q2_std_detach**2) / (q2_std_detach**3 + bias)) * q2_std
        )

        critic_loss = critic_1_loss + critic_2_loss

        return {
            "loss": critic_loss,
            "q1": q1.mean().detach(),
            "q2": q2.mean().detach(),
            "q1_target": q1_targ.mean(),
            "q1_target_bound": q1_targ_bound.mean(),
            "q2_target": q2_targ.mean(),
            "q2_target_bound": q2_targ_bound.mean(),
            "q_mean_std": q_mean_std.detach()
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
