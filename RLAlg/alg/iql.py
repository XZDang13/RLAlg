import torch
import torch.nn as nn
from ..nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep
from typing import Tuple, Union

Q_STEPS = Tuple[ValueStep, ValueStep]


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class IQL:
    @staticmethod
    def _validate_same_shape(name_a: str, tensor_a: torch.Tensor, name_b: str, tensor_b: torch.Tensor) -> None:
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"{name_a} and {name_b} must have the same shape, got {tuple(tensor_a.shape)} and {tuple(tensor_b.shape)}."
            )

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
        observation: torch.Tensor|dict[str, torch.Tensor],
        action: torch.Tensor,
        expectile: float
    ) -> dict[str, torch.Tensor]:
        IQL._validate_expectile(expectile)
        with torch.no_grad():
            q_steps: Q_STEPS = critic_target_model(observation, action)
            q1 = q_steps[0].value
            q2 = q_steps[1].value
            q = torch.min(q1, q2).detach()

        value_step: ValueStep = value_model(observation)
        value = value_step.value
        IQL._validate_same_shape("q", q, "value", value)

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
        observation: torch.Tensor|dict[str, torch.Tensor],
        action: torch.Tensor,
        temperature: float
    ) -> dict[str, torch.Tensor]:
        IQL._validate_temperature(temperature)
        with torch.no_grad():
            value_step: ValueStep = value_model(observation)
            value = value_step.value

            q_steps: Q_STEPS = critic_target_model(observation, action)
            q1 = q_steps[0].value
            q2 = q_steps[1].value
            q = torch.min(q1, q2)
            IQL._validate_same_shape("q", q, "value", value)

            exp_a = torch.exp((q - value) * temperature)
            exp_a = torch.clamp(exp_a, max=100.0)

        step:Union[StochasticContinuousPolicyStep, DiscretePolicyStep] = policy_model(observation, action)

        log_prob = step.log_prob
        IQL._validate_same_shape("exp_a", exp_a, "log_prob", log_prob)

        policy_loss = -(exp_a * log_prob).mean()

        return {
            "loss": policy_loss
        }

    @staticmethod
    def compute_critic_loss(
        value_model: nn.Module,
        critic_model: nn.Module,
        observation: torch.Tensor|dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_observation: torch.Tensor|dict[str, torch.Tensor],
        gamma: float
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            next_value_step: ValueStep = value_model(next_observation)
            next_value = next_value_step.value
            done = done.to(dtype=reward.dtype, device=reward.device)
            IQL._validate_same_shape("reward", reward, "done", done)
            IQL._validate_same_shape("reward", reward, "next_value", next_value)
            q_targ = reward + gamma * (1 - done) * next_value

        q_steps: Q_STEPS = critic_model(observation, action)
        q1 = q_steps[0].value
        q2 = q_steps[1].value
        IQL._validate_same_shape("q1", q1, "q_target", q_targ)
        IQL._validate_same_shape("q2", q2, "q_target", q_targ)

        critic_1_loss = ((q1 - q_targ) ** 2).mean()
        critic_2_loss = ((q2 - q_targ) ** 2).mean()
        critic_loss = critic_1_loss + critic_2_loss

        return {
            "loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
            "q_target": q_targ.mean()
        }

    @staticmethod
    @torch.no_grad()
    def update_target_param(model: nn.Module, model_target: nn.Module, tau: float) -> None:
        for param, param_target in zip(model.parameters(), model_target.parameters()):
            param_target.data.mul_(1 - tau)
            param_target.data.add_(tau * param.data)
