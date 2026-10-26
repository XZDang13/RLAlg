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
        return DeterministicContinuousPolicyStep(mean=mean)


class DummyCritic(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, observation, action):
        if isinstance(observation, dict):
            obs_tensor = next(iter(observation.values()))
        else:
            obs_tensor = observation
        q = obs_tensor.sum(dim=-1, keepdim=True) + self.scale * action.sum(dim=-1, keepdim=True)
        return ValueStep(value=q)


class LinearPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, action_dim)

    def forward(self, observation):
        return DeterministicContinuousPolicyStep(mean=torch.tanh(self.linear(observation)))


class LinearCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim + action_dim, 1)

    def forward(self, observation, action):
        return ValueStep(value=self.linear(torch.cat([observation, action], dim=-1)))


def _batch(batch: int = 4, obs_dim: int = 3, action_dim: int = 2, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "observations": torch.randn(batch, obs_dim, generator=generator),
        "actions": torch.randn(batch, action_dim, generator=generator),
        "rewards": torch.randn(batch, 1, generator=generator),
        "next_observations": torch.randn(batch, obs_dim, generator=generator),
        "terminated": torch.tensor([[True], [False], [True], [False]])[:batch],
    }


def test_compute_policy_loss_multi_critic_validates_lengths():
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    with pytest.raises(ValueError, match="one weight per"):
        DDPG.compute_policy_loss(
            policy, [critic], observations=torch.zeros(4, 3), critic_weights=[1.0, 0.5]
        )


def test_compute_policy_loss_multi_critic_validates_non_empty():
    policy = DummyPolicy(action_dim=2)

    with pytest.raises(ValueError, match="non-empty"):
        DDPG.compute_policy_loss(policy, [], observations=torch.zeros(4, 3), critic_weights=[])


def test_compute_policy_loss_single_critic_rejects_weights():
    with pytest.raises(ValueError, match="sequence"):
        DDPG.compute_policy_loss(
            DummyPolicy(action_dim=2), DummyCritic(), observations=torch.zeros(4, 3), critic_weights=[1.0]
        )


def test_compute_policy_loss_multi_critic_is_weighted_sum():
    observations = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2)
    critics = [DummyCritic(1.0), DummyCritic(3.0)]

    output = DDPG.compute_policy_loss(
        policy, critics, observations=observations, critic_weights=[0.25, 0.5], regularization_weight=0.1
    )

    # actions are all ones: q_k = scale_k * 2, regularisation = 0.1 * mean(1)
    expected = -(0.25 * 2.0 + 0.5 * 6.0) + 0.1
    torch.testing.assert_close(output["loss"], torch.tensor(expected))


def test_compute_policy_loss_one_element_critic_list_matches_single_critic():
    torch.manual_seed(0)
    policy = LinearPolicy(3, 2)
    critic = LinearCritic(3, 2)
    observations = torch.randn(5, 3)

    single = DDPG.compute_policy_loss(policy, critic, observations=observations, regularization_weight=0.2)
    listed = DDPG.compute_policy_loss(
        policy, [critic], observations=observations, critic_weights=[1.0], regularization_weight=0.2
    )

    torch.testing.assert_close(listed["loss"], single["loss"])


def test_compute_policy_loss_uses_critic_observations_for_critic_only():
    actor_observations = torch.zeros(4, 3)
    critic_observations = torch.ones(4, 5)
    policy = DummyPolicy(action_dim=2)

    output = DDPG.compute_policy_loss(
        policy, DummyCritic(), observations=actor_observations, critic_observations=critic_observations
    )

    # q = sum(critic_obs) + sum(action) = 5 + 2
    torch.testing.assert_close(output["loss"], torch.tensor(-7.0))


def test_compute_policy_loss_critic_observations_default_matches_explicit():
    torch.manual_seed(1)
    policy = LinearPolicy(3, 2)
    critic = LinearCritic(3, 2)
    observations = torch.randn(5, 3)

    default = DDPG.compute_policy_loss(policy, critic, observations=observations)
    explicit = DDPG.compute_policy_loss(policy, critic, observations=observations, critic_observations=observations)

    torch.testing.assert_close(explicit["loss"], default["loss"])


def test_compute_critic_loss_accepts_bool_terminated():
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = DDPG.compute_critic_loss(policy, critic, critic, **_batch())

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])
    assert set(output) == {"loss", "q", "q_target"}


def test_compute_critic_loss_is_half_mse():
    torch.manual_seed(2)
    policy = LinearPolicy(3, 2)
    critic = LinearCritic(3, 2)
    critic_target = LinearCritic(3, 2)
    batch = _batch(seed=3)

    output = DDPG.compute_critic_loss(policy, critic, critic_target, gamma=0.9, **batch)

    with torch.no_grad():
        next_q = critic_target(batch["next_observations"], policy(batch["next_observations"]).mean).value
        target = batch["rewards"] + 0.9 * (1 - batch["terminated"].float()) * next_q
        q = critic(batch["observations"], batch["actions"]).value
    torch.testing.assert_close(output["loss"], 0.5 * ((q - target) ** 2).mean())


def test_compute_critic_loss_critic_observations_default_matches_explicit():
    torch.manual_seed(4)
    policy = LinearPolicy(3, 2)
    critic = LinearCritic(3, 2)
    batch = _batch(seed=5)

    default = DDPG.compute_critic_loss(policy, critic, critic, **batch)
    explicit = DDPG.compute_critic_loss(
        policy,
        critic,
        critic,
        critic_observations=batch["observations"],
        next_critic_observations=batch["next_observations"],
        **batch,
    )

    torch.testing.assert_close(explicit["loss"], default["loss"])


def test_compute_critic_loss_asymmetric_observations():
    batch = _batch()
    batch["observations"] = torch.zeros(4, 3)
    batch["next_observations"] = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = DDPG.compute_critic_loss(
        policy,
        critic,
        critic,
        critic_observations=torch.ones(4, 5),
        next_critic_observations=torch.full((4, 5), 2.0),
        **batch,
    )

    torch.testing.assert_close(output["q"], (5.0 + batch["actions"].sum(dim=-1)).mean())
    expected_target = batch["rewards"] + 0.99 * (1 - batch["terminated"].float()) * (10.0 + 2.0)
    torch.testing.assert_close(output["q_target"], expected_target.mean())


def test_compute_critic_loss_rejects_broadcasting_shapes():
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    batch = _batch()
    batch["rewards"] = batch["rewards"].squeeze(-1)
    with pytest.raises(ValueError, match="same shape"):
        DDPG.compute_critic_loss(policy, critic, critic, **batch)

    batch = _batch()
    batch["terminated"] = batch["terminated"].squeeze(-1)
    with pytest.raises(ValueError, match="same shape"):
        DDPG.compute_critic_loss(policy, critic, critic, **batch)


def test_data_arguments_are_keyword_only():
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()
    batch = _batch()

    with pytest.raises(TypeError):
        DDPG.compute_critic_loss(policy, critic, critic, *batch.values())
    with pytest.raises(TypeError):
        DDPG.compute_policy_loss(policy, critic, batch["observations"])


def test_update_target_param():
    model = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.0)
        for param in target.parameters():
            param.fill_(0.0)

    DDPG.update_target_param(model, target, tau=0.25)

    for param in target.parameters():
        torch.testing.assert_close(param, torch.full_like(param, 0.25))
