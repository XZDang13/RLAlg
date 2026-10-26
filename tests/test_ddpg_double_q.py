import pytest
import torch
import torch.nn as nn

from RLAlg.alg.ddpg_double_q import DDPGDoubleQ
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

        q1 = obs_tensor.sum(dim=-1, keepdim=True) + self.scale * action.sum(dim=-1, keepdim=True)
        q2 = q1 + 1.0
        return ValueStep(value=q1), ValueStep(value=q2)


class LinearPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, action_dim)

    def forward(self, observation):
        mean = torch.tanh(self.linear(observation))
        return DeterministicContinuousPolicyStep(mean=mean)


class LinearTwinCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.q1 = nn.Linear(obs_dim + action_dim, 1)
        self.q2 = nn.Linear(obs_dim + action_dim, 1)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        return ValueStep(value=self.q1(x)), ValueStep(value=self.q2(x))


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
    with pytest.raises(ValueError, match="one weight per"):
        DDPGDoubleQ.compute_policy_loss(
            DummyPolicy(action_dim=2),
            [DummyCritic()],
            observations=torch.zeros(4, 3),
            std=torch.tensor(0.1),
            critic_weights=[1.0, 0.5],
        )


def test_compute_policy_loss_multi_critic_validates_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        DDPGDoubleQ.compute_policy_loss(
            DummyPolicy(action_dim=2), [], observations=torch.zeros(4, 3), std=torch.tensor(0.1), critic_weights=[]
        )


def test_compute_policy_loss_multi_critic_is_weighted_sum():
    observations = torch.zeros(4, 3)
    critics = [DummyCritic(1.0), DummyCritic(3.0)]

    # std=0 makes the sampled action equal to the mean (all ones).
    output = DDPGDoubleQ.compute_policy_loss(
        DummyPolicy(action_dim=2),
        critics,
        observations=observations,
        std=torch.tensor(1e-12),
        critic_weights=[0.25, 0.5],
        regularization_weight=0.1,
    )

    expected = -(0.25 * 2.0 + 0.5 * 6.0) + 0.1
    torch.testing.assert_close(output["loss"], torch.tensor(expected))


def test_compute_policy_loss_one_element_critic_list_matches_single_critic():
    torch.manual_seed(0)
    policy = LinearPolicy(3, 2)
    critic = LinearTwinCritic(3, 2)
    observations = torch.randn(5, 3)

    torch.manual_seed(1)
    single = DDPGDoubleQ.compute_policy_loss(policy, critic, observations=observations, std=0.2)
    torch.manual_seed(1)
    listed = DDPGDoubleQ.compute_policy_loss(
        policy, [critic], observations=observations, std=0.2, critic_weights=[1.0]
    )

    torch.testing.assert_close(listed["loss"], single["loss"])


def test_compute_policy_loss_critic_observations_default_matches_explicit():
    torch.manual_seed(0)
    policy = LinearPolicy(3, 2)
    critic = LinearTwinCritic(3, 2)
    observations = torch.randn(5, 3)

    torch.manual_seed(1)
    default = DDPGDoubleQ.compute_policy_loss(policy, critic, observations=observations, std=0.2)
    torch.manual_seed(1)
    explicit = DDPGDoubleQ.compute_policy_loss(
        policy, critic, observations=observations, std=0.2, critic_observations=observations
    )

    torch.testing.assert_close(explicit["loss"], default["loss"])


def test_compute_policy_loss_uses_critic_observations_for_critic_only():
    output = DDPGDoubleQ.compute_policy_loss(
        DummyPolicy(action_dim=2),
        DummyCritic(),
        observations=torch.zeros(4, 3),
        critic_observations=torch.ones(4, 5),
        std=torch.tensor(1e-12),
    )

    # min(q1, q2) = sum(critic_obs) + sum(action) = 5 + 2
    torch.testing.assert_close(output["loss"], torch.tensor(-7.0))


def test_compute_critic_loss_accepts_bool_terminated():
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    output = DDPGDoubleQ.compute_critic_loss(policy, critic, critic, std=torch.tensor(0.1), **_batch())

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])
    assert set(output) == {"loss", "q1", "q2", "q_target"}


def test_compute_critic_loss_is_half_mse_summed_over_twins():
    torch.manual_seed(2)
    policy = LinearPolicy(3, 2)
    critic = LinearTwinCritic(3, 2)
    critic_target = LinearTwinCritic(3, 2)
    batch = _batch(seed=3)

    torch.manual_seed(7)
    output = DDPGDoubleQ.compute_critic_loss(policy, critic, critic_target, std=0.1, gamma=0.9, **batch)

    torch.manual_seed(7)
    with torch.no_grad():
        next_action = DDPGDoubleQ.perturb_action(policy(batch["next_observations"]), 0.1)
        q1_t, q2_t = critic_target(batch["next_observations"], next_action)
        target = batch["rewards"] + 0.9 * (1 - batch["terminated"].float()) * torch.min(q1_t.value, q2_t.value)
        q1, q2 = critic(batch["observations"], batch["actions"])
    expected = 0.5 * ((q1.value - target) ** 2).mean() + 0.5 * ((q2.value - target) ** 2).mean()
    torch.testing.assert_close(output["loss"], expected)


def test_compute_critic_loss_critic_observations_default_matches_explicit():
    torch.manual_seed(4)
    policy = LinearPolicy(3, 2)
    critic = LinearTwinCritic(3, 2)
    batch = _batch(seed=5)

    torch.manual_seed(1)
    default = DDPGDoubleQ.compute_critic_loss(policy, critic, critic, std=0.1, **batch)
    torch.manual_seed(1)
    explicit = DDPGDoubleQ.compute_critic_loss(
        policy,
        critic,
        critic,
        std=0.1,
        critic_observations=batch["observations"],
        next_critic_observations=batch["next_observations"],
        **batch,
    )

    torch.testing.assert_close(explicit["loss"], default["loss"])


def test_compute_critic_loss_asymmetric_observations():
    batch = _batch()
    batch["observations"] = torch.zeros(4, 3)
    batch["next_observations"] = torch.zeros(4, 3)

    output = DDPGDoubleQ.compute_critic_loss(
        DummyPolicy(action_dim=2),
        DummyCritic(),
        DummyCritic(),
        std=torch.tensor(1e-12),
        critic_observations=torch.ones(4, 5),
        next_critic_observations=torch.full((4, 5), 2.0),
        **batch,
    )

    torch.testing.assert_close(output["q1"], (5.0 + batch["actions"].sum(dim=-1)).mean())
    expected_target = batch["rewards"] + 0.99 * (1 - batch["terminated"].float()) * (10.0 + 2.0)
    torch.testing.assert_close(output["q_target"], expected_target.mean())


def test_compute_critic_loss_rejects_broadcasting_shapes():
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    batch = _batch()
    batch["rewards"] = batch["rewards"].squeeze(-1)
    with pytest.raises(ValueError, match="same shape"):
        DDPGDoubleQ.compute_critic_loss(policy, critic, critic, std=0.1, **batch)

    batch = _batch()
    batch["terminated"] = batch["terminated"].squeeze(-1)
    with pytest.raises(ValueError, match="same shape"):
        DDPGDoubleQ.compute_critic_loss(policy, critic, critic, std=0.1, **batch)


def test_data_arguments_are_keyword_only():
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()
    batch = _batch()

    with pytest.raises(TypeError):
        DDPGDoubleQ.compute_critic_loss(policy, critic, critic, *batch.values(), 0.1)
    with pytest.raises(TypeError):
        DDPGDoubleQ.compute_policy_loss(policy, critic, batch["observations"], 0.1)


def test_update_target_param():
    model = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.0)
        for param in target.parameters():
            param.fill_(0.0)

    DDPGDoubleQ.update_target_param(model, target, tau=0.25)

    for param in target.parameters():
        torch.testing.assert_close(param, torch.full_like(param, 0.25))


def test_perturb_action_adds_noise_before_tanh_and_matches_the_old_distribution():
    from torch.distributions import AffineTransform, ComposeTransform, Normal, TanhTransform, TransformedDistribution

    from RLAlg.nn.layers import DeterministicHead

    head = DeterministicHead(3, 2, max_action=2.0)
    x = torch.randn(5, 3)
    step = head(x)
    torch.testing.assert_close(step.mean, 2.0 * torch.tanh(step.pre_tanh_mean))

    torch.manual_seed(0)
    noisy = DDPGDoubleQ.perturb_action(step, 0.3)
    torch.manual_seed(0)
    transform = ComposeTransform([TanhTransform(), AffineTransform(loc=0, scale=torch.tensor(2.0))])
    old = TransformedDistribution(Normal(step.pre_tanh_mean, 0.3), transform).rsample()
    torch.testing.assert_close(noisy, old)
    assert torch.all(noisy.abs() <= 2.0)
    noisy.sum().backward()
    assert head.linear.weight.grad.abs().sum() > 0

    unbounded = DeterministicHead(3, 2)(x)
    assert unbounded.pre_tanh_mean is None and unbounded.max_action is None
    torch.manual_seed(1)
    expected = unbounded.mean + 0.5 * torch.randn_like(unbounded.mean)
    torch.manual_seed(1)
    torch.testing.assert_close(DDPGDoubleQ.perturb_action(unbounded, 0.5), expected)
