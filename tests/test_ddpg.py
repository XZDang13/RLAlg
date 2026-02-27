import pytest
import torch
import torch.nn as nn

from RLAlg.alg.ddpg import DDPG
from RLAlg.nn.steps import DeterministicContinuousPolicyStep, ValueStep


class DummyPolicy(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, observation):
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
        return DeterministicContinuousPolicyStep(pi=None, mean=mean)


class DummyCritic(nn.Module):
    def forward(self, observation, action):
        if isinstance(observation, dict):
            obs_tensor = next(iter(observation.values()))
        else:
            obs_tensor = observation
        q = obs_tensor.sum(dim=-1, keepdim=True) + action.sum(dim=-1, keepdim=True)
        return ValueStep(value=q)


def test_compute_policy_loss_with_multi_critic_validates_lengths():
    observation = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    with pytest.raises(ValueError, match="same length"):
        DDPG.compute_policy_loss_with_multi_critic(
            policy_model=policy,
            critic_models=[critic],
            weights=[1.0, 0.5],
            observation=observation,
        )


def test_compute_policy_loss_asymmetric_with_multi_critic_validates_non_empty():
    actor_observation = torch.zeros(4, 3)
    critic_observation = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2)

    with pytest.raises(ValueError, match="non-empty"):
        DDPG.compute_policy_loss_asymmetric_with_multi_critic(
            policy_model=policy,
            critic_models=[],
            weights=[],
            actor_observation=actor_observation,
            critic_observation=critic_observation,
        )


def test_compute_policy_loss_asymmetric_with_multi_critic_uses_two_arg_critic():
    actor_observation = torch.zeros(4, 3)
    critic_observation = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = DDPG.compute_policy_loss_asymmetric_with_multi_critic(
        policy_model=policy,
        critic_models=[critic],
        weights=[1.0],
        actor_observation=actor_observation,
        critic_observation=critic_observation,
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

    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = DDPG.compute_critic_loss(
        actor_target_model=policy,
        critic_model=critic,
        critic_target_model=critic,
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        done=done,
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

    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = DDPG.compute_critic_loss_asymmetric(
        actor_target_model=policy,
        critic_model=critic,
        critic_target_model=critic,
        critic_observation=critic_observation,
        action=action,
        reward=reward,
        next_actor_observation=next_actor_observation,
        next_critic_observation=next_critic_observation,
        done=done,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])
