import pytest
import torch
import torch.nn as nn
from torch.distributions import Normal

from RLAlg.alg.sac import SAC
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep


def _batch_info(observation):
    if isinstance(observation, dict):
        first = next(iter(observation.values()))
        return first.shape[0], first.device, first.dtype
    return observation.shape[0], observation.device, observation.dtype


class DummyPolicy(nn.Module):
    def __init__(self, action_dim: int, log_prob_shape: str = "flat"):
        super().__init__()
        self.action_dim = action_dim
        self.log_prob_shape = log_prob_shape

    def forward(self, observation):
        batch, device, dtype = _batch_info(observation)
        mean = torch.zeros(batch, self.action_dim, device=device, dtype=dtype)
        std = torch.ones_like(mean)
        pi = Normal(mean, std)
        action = mean
        if self.log_prob_shape == "column":
            log_prob = torch.zeros(batch, 1, device=device, dtype=dtype)
            entropy = torch.zeros(batch, 1, device=device, dtype=dtype)
        else:
            log_prob = torch.zeros(batch, device=device, dtype=dtype)
            entropy = torch.zeros(batch, device=device, dtype=dtype)
        log_std = torch.zeros_like(mean)
        return StochasticContinuousPolicyStep(
            pi=pi,
            action=action,
            log_prob=log_prob,
            mean=mean,
            log_std=log_std,
            entropy=entropy,
        )


class DummyCritic(nn.Module):
    def __init__(self, output_shape: str = "flat"):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, observation, action):
        batch, device, dtype = _batch_info(observation)
        q1 = torch.ones(batch, device=device, dtype=dtype)
        q2 = torch.full((batch,), 2.0, device=device, dtype=dtype)
        if self.output_shape == "column":
            q1 = q1.unsqueeze(-1)
            q2 = q2.unsqueeze(-1)
        return ValueStep(value=q1), ValueStep(value=q2)


def test_compute_policy_loss_with_multi_critic_validates_lengths():
    observation = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    with pytest.raises(ValueError, match="same length"):
        SAC.compute_policy_loss_with_multi_critic(
            policy_model=policy,
            critic_models=[critic],
            weights=[1.0, 0.5],
            observation=observation,
            alpha=0.2,
        )


def test_compute_policy_loss_asymmetric_with_multi_critic_validates_non_empty():
    actor_observation = torch.zeros(4, 3)
    critic_observation = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2)

    with pytest.raises(ValueError, match="non-empty"):
        SAC.compute_policy_loss_asymmetric_with_multi_critic(
            policy_model=policy,
            critic_models=[],
            weights=[],
            actor_observation=actor_observation,
            critic_observation=critic_observation,
            alpha=0.2,
        )


def test_compute_policy_loss_raises_on_shape_mismatch():
    observation = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2, log_prob_shape="flat")
    critic = DummyCritic(output_shape="column")

    with pytest.raises(ValueError, match="same shape"):
        SAC.compute_policy_loss(
            policy_model=policy,
            critic_model=critic,
            observation=observation,
            alpha=0.2,
        )


def test_compute_critic_loss_accepts_bool_done():
    batch = 4
    observation = torch.zeros(batch, 3)
    next_observation = torch.zeros(batch, 3)
    action = torch.zeros(batch, 2)
    reward = torch.ones(batch)
    done = torch.tensor([True, False, True, False])

    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = SAC.compute_critic_loss(
        policy_model=policy,
        critic_model=critic,
        critic_target_model=critic,
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        done=done,
        alpha=0.2,
        gamma=0.99,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])


def test_compute_critic_loss_asymmetric_accepts_bool_done():
    batch = 4
    critic_observation = torch.zeros(batch, 3)
    next_actor_observation = torch.zeros(batch, 3)
    next_critic_observation = torch.zeros(batch, 3)
    action = torch.zeros(batch, 2)
    reward = torch.ones(batch)
    done = torch.tensor([False, True, False, True])

    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = SAC.compute_critic_loss_asymmetric(
        policy_model=policy,
        critic_model=critic,
        critic_target_model=critic,
        critic_observation=critic_observation,
        action=action,
        reward=reward,
        next_actor_observation=next_actor_observation,
        next_critic_observation=next_critic_observation,
        done=done,
        alpha=0.2,
        gamma=0.99,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])
