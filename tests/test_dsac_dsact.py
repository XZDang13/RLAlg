import torch
import torch.nn as nn
from torch.distributions import Normal

from RLAlg.alg.dsac import DSAC
from RLAlg.alg.dsact import DSACT
from RLAlg.nn.steps import DistributionStep, StochasticContinuousPolicyStep


def _batch_info(observation):
    if isinstance(observation, dict):
        first = next(iter(observation.values()))
        return first.shape[0], first.device, first.dtype
    return observation.shape[0], observation.device, observation.dtype


class DummyPolicy(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, observation):
        batch, device, dtype = _batch_info(observation)
        mean = torch.zeros(batch, self.action_dim, device=device, dtype=dtype)
        std = torch.ones_like(mean)
        pi = Normal(mean, std)
        action = mean
        log_prob = torch.zeros(batch, device=device, dtype=dtype)
        log_std = torch.zeros_like(mean)
        entropy = torch.zeros(batch, device=device, dtype=dtype)
        return StochasticContinuousPolicyStep(
            pi=pi,
            action=action,
            log_prob=log_prob,
            mean=mean,
            log_std=log_std,
            entropy=entropy,
        )


class DummyDistributionalCritic(nn.Module):
    def __init__(self, std_value: float):
        super().__init__()
        self.std_value = std_value

    def forward(self, observation, action):
        batch, device, dtype = _batch_info(observation)
        obs_tensor = next(iter(observation.values())) if isinstance(observation, dict) else observation
        mean = obs_tensor.sum(dim=-1) + action.sum(dim=-1)
        std = torch.full((batch,), self.std_value, device=device, dtype=dtype)
        sample = mean + 0.1
        pi = Normal(mean, std)
        step1 = DistributionStep(pi=pi, mean=mean, std=std, sample=sample)
        step2 = DistributionStep(pi=pi, mean=mean + 0.5, std=std, sample=sample + 0.5)
        return step1, step2


def test_dsac_compute_critic_loss_accepts_bool_done():
    batch = 5
    observation = torch.zeros(batch, 4)
    next_observation = torch.zeros(batch, 4)
    action = torch.zeros(batch, 2)
    reward = torch.ones(batch)
    done = torch.tensor([True, False, True, False, False])

    policy = DummyPolicy(action_dim=2)
    critic = DummyDistributionalCritic(std_value=1.0)

    output = DSAC.compute_critic_loss(
        actor_target_model=policy,
        critic_model=critic,
        critic_target_model=critic,
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        done=done,
        alpha=0.2,
        gamma=0.99,
        td_bound=1.0,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])


def test_dsact_compute_critic_loss_accepts_bool_done():
    DSACT.q1_mean_std = None
    DSACT.q2_mean_std = None

    batch = 5
    observation = torch.zeros(batch, 4)
    next_observation = torch.zeros(batch, 4)
    action = torch.zeros(batch, 2)
    reward = torch.ones(batch)
    done = torch.tensor([False, True, False, True, False])

    policy = DummyPolicy(action_dim=2)
    critic = DummyDistributionalCritic(std_value=1.0)

    output = DSACT.compute_critic_loss(
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
        tau_b=0.1,
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])


def test_dsact_shared_running_std_updates():
    DSACT.q1_mean_std = None
    DSACT.q2_mean_std = None

    batch = 5
    observation = torch.zeros(batch, 4)
    next_observation = torch.zeros(batch, 4)
    action = torch.zeros(batch, 2)
    reward = torch.ones(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    policy = DummyPolicy(action_dim=2)

    critic_first = DummyDistributionalCritic(std_value=1.0)
    DSACT.compute_critic_loss(
        policy_model=policy,
        critic_model=critic_first,
        critic_target_model=critic_first,
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        done=done,
        alpha=0.2,
        gamma=0.99,
        tau_b=0.5,
    )
    first_q1 = DSACT.q1_mean_std.clone()
    first_q2 = DSACT.q2_mean_std.clone()

    critic_second = DummyDistributionalCritic(std_value=3.0)
    DSACT.compute_critic_loss(
        policy_model=policy,
        critic_model=critic_second,
        critic_target_model=critic_second,
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        done=done,
        alpha=0.2,
        gamma=0.99,
        tau_b=0.5,
    )

    assert DSACT.q1_mean_std is not None
    assert DSACT.q2_mean_std is not None
    assert DSACT.q1_mean_std.device == first_q1.device
    assert DSACT.q2_mean_std.device == first_q2.device
    assert DSACT.q1_mean_std > first_q1
    assert DSACT.q2_mean_std > first_q2
