import pytest
import torch
import torch.nn as nn
from torch.distributions import Normal

from RLAlg.alg.ddpg_double_q import DDPGDoubleQ
from RLAlg.nn.steps import DeterministicContinuousPolicyStep, ValueStep


class DummyPolicy(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, observation, std):
        if isinstance(observation, dict):
            first = next(iter(observation.values()))
            batch = first.shape[0]
            device = first.device
            dtype = first.dtype
        else:
            batch = observation.shape[0]
            device = observation.device
            dtype = observation.dtype

        mean = torch.ones(batch, self.action_dim, device=device, dtype=dtype)
        std_tensor = torch.as_tensor(std, dtype=dtype, device=device)
        if std_tensor.ndim == 0:
            scale = torch.full_like(mean, float(std_tensor))
        else:
            scale = std_tensor.expand_as(mean)
        pi = Normal(mean, scale)
        return DeterministicContinuousPolicyStep(pi=pi, mean=mean)


class DummyCritic(nn.Module):
    def forward(self, observation, action):
        if isinstance(observation, dict):
            obs_tensor = next(iter(observation.values()))
        else:
            obs_tensor = observation

        q1 = obs_tensor.sum(dim=-1, keepdim=True) + action.sum(dim=-1, keepdim=True)
        q2 = q1 + 1.0
        return ValueStep(value=q1), ValueStep(value=q2)


def test_compute_policy_loss_with_multi_critic_validates_lengths():
    observation = torch.zeros(4, 3)
    std = torch.tensor(0.1)
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    with pytest.raises(ValueError, match="same length"):
        DDPGDoubleQ.compute_policy_loss_with_multi_critic(
            policy_model=policy,
            critic_models=[critic],
            weights=[1.0, 0.5],
            observation=observation,
            std=std,
        )


def test_compute_policy_loss_asymmetric_with_multi_critic_validates_non_empty():
    actor_observation = torch.zeros(4, 3)
    critic_observation = torch.zeros(4, 3)
    std = torch.tensor(0.1)
    policy = DummyPolicy(action_dim=2)

    with pytest.raises(ValueError, match="non-empty"):
        DDPGDoubleQ.compute_policy_loss_asymmetric_with_multi_critic(
            policy_model=policy,
            critic_models=[],
            weights=[],
            actor_observation=actor_observation,
            critic_observation=critic_observation,
            std=std,
        )


def test_compute_policy_loss_with_multi_critic_returns_tensor():
    observation = torch.zeros(4, 3)
    std = torch.tensor(0.1)
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = DDPGDoubleQ.compute_policy_loss_with_multi_critic(
        policy_model=policy,
        critic_models=[critic],
        weights=[1.0],
        observation=observation,
        std=std,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])


def test_compute_critic_loss_accepts_bool_done():
    batch = 4
    observation = torch.zeros(batch, 3)
    next_observation = torch.zeros(batch, 3)
    action = torch.zeros(batch, 2)
    reward = torch.ones(batch, 1)
    done = torch.tensor([[True], [False], [True], [False]])
    std = torch.tensor(0.1)

    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = DDPGDoubleQ.compute_critic_loss(
        policy_model=policy,
        critic_model=critic,
        critic_target_model=critic,
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        done=done,
        std=std,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])


def test_compute_critic_loss_asymmetric_accepts_bool_done():
    batch = 4
    action = torch.zeros(batch, 2)
    reward = torch.ones(batch, 1)
    done = torch.tensor([[False], [True], [False], [True]])
    actor_observation = torch.zeros(batch, 3)
    critic_observation = torch.zeros(batch, 3)
    next_actor_observation = torch.zeros(batch, 3)
    next_critic_observation = torch.zeros(batch, 3)
    std = torch.tensor(0.1)

    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = DDPGDoubleQ.compute_critic_loss_asymmetric(
        policy_model=policy,
        critic_model=critic,
        critic_target_model=critic,
        critic_observation=critic_observation,
        action=action,
        reward=reward,
        next_actor_observation=next_actor_observation,
        next_critic_observation=next_critic_observation,
        done=done,
        std=std,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])
