import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical

from RLAlg.alg.iql import IQL
from RLAlg.nn.steps import DiscretePolicyStep, ValueStep


class DummyValueModel(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.values = values

    def forward(self, observation):
        return ValueStep(value=self.values)


class DummyCritic(nn.Module):
    def __init__(self, q1: torch.Tensor, q2: torch.Tensor):
        super().__init__()
        self.q1 = q1
        self.q2 = q2

    def forward(self, observation, action):
        return ValueStep(value=self.q1), ValueStep(value=self.q2)


class DummyDiscretePolicy(nn.Module):
    def __init__(self, log_prob: torch.Tensor, entropy: torch.Tensor):
        super().__init__()
        self.log_prob = log_prob
        self.entropy = entropy

    def forward(self, observation, action):
        batch = self.log_prob.shape[0]
        pi = Categorical(logits=torch.zeros(batch, 2, dtype=self.log_prob.dtype, device=self.log_prob.device))
        return DiscretePolicyStep(
            pi=pi,
            action=action,
            log_prob=self.log_prob,
            entropy=self.entropy,
        )


def test_compute_critic_loss_accepts_bool_done():
    batch = 4
    observation = torch.zeros(batch, 3)
    next_observation = torch.zeros(batch, 3)
    action = torch.zeros(batch, 2)
    reward = torch.ones(batch)
    done = torch.tensor([True, False, True, False])

    value_model = DummyValueModel(values=torch.full((batch,), 0.5))
    critic = DummyCritic(q1=torch.full((batch,), 1.0), q2=torch.full((batch,), 1.2))

    output = IQL.compute_critic_loss(
        value_model=value_model,
        critic_model=critic,
        observation=observation,
        action=action,
        reward=reward,
        done=done,
        next_observation=next_observation,
        gamma=0.99,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])


def test_compute_value_loss_rejects_invalid_expectile():
    batch = 4
    observation = torch.zeros(batch, 3)
    action = torch.zeros(batch, 2)
    value_model = DummyValueModel(values=torch.zeros(batch))
    critic_target = DummyCritic(q1=torch.zeros(batch), q2=torch.zeros(batch))

    with pytest.raises(ValueError, match="expectile"):
        IQL.compute_value_loss(
            value_model=value_model,
            critic_target_model=critic_target,
            observation=observation,
            action=action,
            expectile=1.5,
        )


def test_compute_policy_loss_rejects_non_positive_temperature():
    batch = 4
    observation = torch.zeros(batch, 3)
    action = torch.zeros(batch, 2)

    policy_model = DummyDiscretePolicy(
        log_prob=torch.zeros(batch),
        entropy=torch.zeros(batch),
    )
    value_model = DummyValueModel(values=torch.zeros(batch))
    critic_target = DummyCritic(q1=torch.zeros(batch), q2=torch.zeros(batch))

    with pytest.raises(ValueError, match="temperature"):
        IQL.compute_policy_loss(
            policy_model=policy_model,
            value_model=value_model,
            critic_target_model=critic_target,
            observation=observation,
            action=action,
            temperature=0.0,
        )


def test_compute_value_loss_raises_on_shape_mismatch():
    batch = 4
    observation = torch.zeros(batch, 3)
    action = torch.zeros(batch, 2)
    value_model = DummyValueModel(values=torch.zeros(batch))
    critic_target = DummyCritic(q1=torch.zeros(batch, 1), q2=torch.zeros(batch, 1))

    with pytest.raises(ValueError, match="same shape"):
        IQL.compute_value_loss(
            value_model=value_model,
            critic_target_model=critic_target,
            observation=observation,
            action=action,
            expectile=0.7,
        )
