import torch
import torch.nn as nn
from ..nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep
from ._common import mse, soft_update, validate_same_shape
from typing import Tuple, Union

OBSERVATION = torch.Tensor | dict[str, torch.Tensor]
Q_STEPS = Tuple[ValueStep, ValueStep]


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class IQL:
    @staticmethod
    def _validate_expectile(expectile: float) -> None:
        if not 0.0 <= expectile <= 1.0:
            raise ValueError(f"expectile must be in [0, 1], got {expectile}.")

    @staticmethod
    def _validate_temperature(temperature: float) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

    @staticmethod
    def compute_value_loss(
        value_model: nn.Module,
        critic_target_model: nn.Module,
        *,
        observations: OBSERVATION,
        actions: torch.Tensor,
        expectile: float,
    ) -> dict[str, torch.Tensor]:
        # Only value/critic models are evaluated here, so observations are whatever the critics see.
        IQL._validate_expectile(expectile)
        with torch.no_grad():
            q_steps: Q_STEPS = critic_target_model(observations, actions)
            q1 = q_steps[0].value
            q2 = q_steps[1].value
            q = torch.min(q1, q2).detach()

        value_step: ValueStep = value_model(observations)
        value = value_step.value
        validate_same_shape("q", q, "value", value)

        diff = q - value
        value_loss = asymmetric_l2_loss(diff, expectile)

        return {
            "loss": value_loss
        }

    @staticmethod
    def compute_policy_loss(
        policy_model: nn.Module,
        value_model: nn.Module,
        critic_target_model: nn.Module,
        *,
        observations: OBSERVATION,
        actions: torch.Tensor,
        temperature: float,
        critic_observations: OBSERVATION | None = None,
    ) -> dict[str, torch.Tensor]:
        IQL._validate_temperature(temperature)
        if critic_observations is None:
            critic_observations = observations
        with torch.no_grad():
            value_step: ValueStep = value_model(critic_observations)
            value = value_step.value

            q_steps: Q_STEPS = critic_target_model(critic_observations, actions)
            q1 = q_steps[0].value
            q2 = q_steps[1].value
            q = torch.min(q1, q2)
            validate_same_shape("q", q, "value", value)

            exp_a = torch.exp((q - value) * temperature)
            exp_a = torch.clamp(exp_a, max=100.0)

        step: Union[StochasticContinuousPolicyStep, DiscretePolicyStep] = policy_model(observations, actions)

        log_prob = step.log_prob
        validate_same_shape("exp_a", exp_a, "log_prob", log_prob)

        policy_loss = -(exp_a * log_prob).mean()

        return {
            "loss": policy_loss
        }

    @staticmethod
    def compute_critic_loss(
        value_model: nn.Module,
        critic_model: nn.Module,
        *,
        observations: OBSERVATION,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: OBSERVATION,
        terminated: torch.Tensor,
        gamma: float = 0.99,
    ) -> dict[str, torch.Tensor]:
        # Only value/critic models are evaluated here, so observations are whatever the critics see.
        with torch.no_grad():
            next_value_step: ValueStep = value_model(next_observations)
            next_value = next_value_step.value
            terminated = terminated.to(dtype=rewards.dtype, device=rewards.device)
            validate_same_shape("rewards", rewards, "terminated", terminated)
            validate_same_shape("rewards", rewards, "next_value", next_value)
            q_targ = rewards + gamma * (1 - terminated) * next_value

        q_steps: Q_STEPS = critic_model(observations, actions)
        q1 = q_steps[0].value
        q2 = q_steps[1].value

        critic_loss = mse(q1, q_targ, "q1") + mse(q2, q_targ, "q2")

        return {
            "loss": critic_loss,
            "q1": q1.detach().mean(),
            "q2": q2.detach().mean(),
            "q_target": q_targ.mean()
        }

    update_target_param = staticmethod(soft_update)
