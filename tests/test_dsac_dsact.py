import pytest
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


def _critic_batch(batch=5, terminated=None):
    return dict(
        observations=torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 4) / 10,
        actions=torch.zeros(batch, 2),
        rewards=torch.ones(batch),
        next_observations=torch.ones(batch, 4),
        terminated=torch.tensor([True, False, True, False, False]) if terminated is None else terminated,
    )


def _dsac_critic(policy, critic, **kwargs):
    return DSAC.compute_critic_loss(policy, critic, critic, alpha=0.2, gamma=0.99, td_bound=1.0, **kwargs)


def _dsact_critic(policy, critic, q_mean_std=None, tau_b=0.1, **kwargs):
    return DSACT.compute_critic_loss(
        policy, critic, critic, alpha=0.2, gamma=0.99, tau_b=tau_b, q_mean_std=q_mean_std, **kwargs
    )


ALGS = [DSAC, DSACT]


def test_dsac_compute_critic_loss_accepts_bool_done():
    output = _dsac_critic(DummyPolicy(action_dim=2), DummyDistributionalCritic(std_value=1.0), **_critic_batch())

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])
    assert set(output) == {"loss", "q1", "q2", "q_target", "q_target_bound"}


def test_dsact_compute_critic_loss_accepts_bool_done():
    output = _dsact_critic(DummyPolicy(action_dim=2), DummyDistributionalCritic(std_value=1.0), **_critic_batch())

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])
    assert output["q_mean_std"].shape == (2,)
    assert not output["q_mean_std"].requires_grad


def test_dsact_running_std_initialises_then_ema():
    policy = DummyPolicy(action_dim=2)
    batch = _critic_batch(terminated=torch.zeros(5, dtype=torch.bool))

    first = _dsact_critic(policy, DummyDistributionalCritic(std_value=1.0), tau_b=0.5, **batch)["q_mean_std"]
    # First call: initialised to the current batch mean of each critic's std.
    assert torch.equal(first, torch.tensor([1.0, 1.0]))

    second = _dsact_critic(
        policy, DummyDistributionalCritic(std_value=3.0), q_mean_std=first, tau_b=0.5, **batch
    )["q_mean_std"]
    # EMA: (1 - tau_b) * previous + tau_b * current.
    assert torch.allclose(second, torch.tensor([2.0, 2.0]))
    assert second.device == first.device
    assert torch.all(second > first)


def test_dsact_independent_call_sequences_do_not_share_state():
    policy = DummyPolicy(action_dim=2)
    batch = _critic_batch()

    state_a = None
    for _ in range(3):
        state_a = _dsact_critic(policy, DummyDistributionalCritic(std_value=5.0), q_mean_std=state_a, **batch)["q_mean_std"]

    # A fresh sequence starts from None and is unaffected by sequence A.
    fresh = _dsact_critic(policy, DummyDistributionalCritic(std_value=1.0), q_mean_std=None, **batch)
    assert torch.equal(fresh["q_mean_std"], torch.tensor([1.0, 1.0]))
    assert torch.equal(state_a, torch.tensor([5.0, 5.0]))
    assert not hasattr(DSACT, "q1_mean_std") and not hasattr(DSACT, "q2_mean_std")

    # The loss depends only on the state passed in.
    again = _dsact_critic(policy, DummyDistributionalCritic(std_value=1.0), q_mean_std=None, **batch)
    assert torch.equal(fresh["loss"], again["loss"])


def test_dsact_rejects_bad_q_mean_std_shape():
    with pytest.raises(ValueError, match="q_mean_std"):
        _dsact_critic(
            DummyPolicy(action_dim=2), DummyDistributionalCritic(std_value=1.0), q_mean_std=torch.ones(1), **_critic_batch()
        )


def test_dsact_critic_loss_value():
    """Check the distributional loss against a direct evaluation of the formula."""
    policy = DummyPolicy(action_dim=2)
    critic = DummyDistributionalCritic(std_value=2.0)
    batch = _critic_batch()
    output = _dsact_critic(policy, critic, q_mean_std=torch.tensor([1.0, 3.0]), tau_b=0.5, **batch)

    q1 = batch["observations"].sum(-1)
    q2 = q1 + 0.5
    std = torch.full((5,), 2.0)
    mean_std = torch.tensor([1.5, 2.5])
    not_done = 1 - batch["terminated"].float()
    next_q1 = torch.full((5,), 4.0)  # next obs sum = 4, next action = 0
    q_next, q_next_sample = next_q1, next_q1 + 0.1
    q_targ = batch["rewards"] + not_done * 0.99 * q_next
    q_targ_sample = batch["rewards"] + not_done * 0.99 * q_next_sample
    bias = 1e-5
    expected = 0.0
    for q, m in ((q1, mean_std[0]), (q2, mean_std[1])):
        bound = q + torch.clamp(q_targ_sample - q, -3 * m, 3 * m)
        term = -(q_targ - q) / (std**2 + bias) * q - (((q - bound) ** 2 - std**2) / (std**3 + bias)) * std
        expected = expected + (m**2 + bias) * term.mean()
    assert torch.allclose(output["loss"], expected)
    assert torch.allclose(output["q_mean_std"], mean_std)


@pytest.mark.parametrize("alg", ALGS)
def test_policy_loss_value_and_multi_critic(alg):
    policy = DummyPolicy(action_dim=2)
    observations = torch.ones(5, 4)
    critic = DummyDistributionalCritic(std_value=1.0)
    single = alg.compute_policy_loss(policy, critic, observations=observations, alpha=0.2)["loss"]
    one = alg.compute_policy_loss(policy, [critic], observations=observations, alpha=0.2, critic_weights=[1.0])["loss"]
    two = alg.compute_policy_loss(policy, [critic, critic], observations=observations, alpha=0.2, critic_weights=[0.25, 0.5])["loss"]
    # log_prob = 0, min(q1, q2) = obs sum = 4.
    assert torch.allclose(single, torch.tensor(-4.0))
    assert torch.equal(single, one)
    assert torch.allclose(two, torch.tensor(-3.0))
    with pytest.raises(ValueError, match="one weight per"):
        alg.compute_policy_loss(policy, [critic], observations=observations, alpha=0.2, critic_weights=[1.0, 2.0])


@pytest.mark.parametrize("alg", ALGS)
def test_policy_loss_critic_observations(alg):
    policy = DummyPolicy(action_dim=2)
    critic = DummyDistributionalCritic(std_value=1.0)
    observations = torch.zeros(5, 4)
    base = alg.compute_policy_loss(policy, critic, observations=observations, alpha=0.2)["loss"]
    same = alg.compute_policy_loss(policy, critic, observations=observations, critic_observations=observations, alpha=0.2)["loss"]
    other = alg.compute_policy_loss(policy, critic, observations=observations, critic_observations=torch.ones(5, 4), alpha=0.2)["loss"]
    assert torch.equal(base, same)
    assert not torch.equal(base, other)


@pytest.mark.parametrize("alg", ALGS)
def test_rejects_positional_data(alg):
    policy = DummyPolicy(action_dim=2)
    critic = DummyDistributionalCritic(std_value=1.0)
    with pytest.raises(TypeError):
        alg.compute_policy_loss(policy, critic, torch.zeros(5, 4), 0.2)
    with pytest.raises(TypeError):
        alg.compute_critic_loss(policy, critic, critic, *_critic_batch().values(), 0.2, 0.99, 1.0)
    with pytest.raises(TypeError):
        alg.compute_alpha_loss(policy, torch.zeros(1), torch.zeros(5, 4), -2.0)


@pytest.mark.parametrize("critic_loss", [_dsac_critic, _dsact_critic])
def test_critic_loss_critic_observations(critic_loss):
    policy = DummyPolicy(action_dim=2)
    critic = DummyDistributionalCritic(std_value=1.0)
    batch = _critic_batch()
    base = critic_loss(policy, critic, **batch)["loss"]
    same = critic_loss(
        policy, critic, critic_observations=batch["observations"], next_critic_observations=batch["next_observations"], **batch
    )["loss"]
    other = critic_loss(policy, critic, critic_observations=torch.ones(5, 4), **batch)["loss"]
    assert torch.equal(base, same)
    assert not torch.equal(base, other)


@pytest.mark.parametrize("critic_loss", [_dsac_critic, _dsact_critic])
def test_critic_loss_rejects_broadcasting(critic_loss):
    batch = _critic_batch()
    batch["rewards"] = batch["rewards"].unsqueeze(-1)
    batch["terminated"] = batch["terminated"].unsqueeze(-1)
    with pytest.raises(ValueError, match="same shape"):
        critic_loss(DummyPolicy(action_dim=2), DummyDistributionalCritic(std_value=1.0), **batch)


@pytest.mark.parametrize("alg", ALGS)
def test_alpha_loss_returns_loss_key(alg):
    log_alpha = torch.zeros(1, requires_grad=True)
    output = alg.compute_alpha_loss(DummyPolicy(action_dim=2), log_alpha=log_alpha, observations=torch.zeros(5, 4), target_entropy=-2.0)
    assert "alpha_loss" not in output
    assert torch.allclose(output["loss"], torch.tensor(2.0))


@pytest.mark.parametrize("alg", ALGS)
def test_update_target_param_soft_update(alg):
    model, target = nn.Linear(2, 2), nn.Linear(2, 2)
    expected = [0.9 * t + 0.1 * p for p, t in zip(model.parameters(), target.parameters())]
    alg.update_target_param(model, target, 0.1)
    for value, param in zip(expected, target.parameters()):
        assert torch.allclose(value, param)
