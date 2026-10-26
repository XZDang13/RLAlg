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


class LinearCritic(nn.Module):
    """Twin critic whose output depends on observations and actions, so losses are non-trivial."""

    def forward(self, observation, action):
        q1 = observation.sum(dim=-1) + action.sum(dim=-1)
        return ValueStep(value=q1), ValueStep(value=q1 * 0.5 + 1.0)


def _critic_batch(batch=4, done=None):
    return dict(
        observations=torch.arange(batch * 3, dtype=torch.float32).reshape(batch, 3) / 10,
        actions=torch.zeros(batch, 2),
        rewards=torch.ones(batch),
        next_observations=torch.ones(batch, 3),
        terminated=torch.tensor([True, False, True, False]) if done is None else done,
    )


def test_compute_policy_loss_with_multi_critic_validates_lengths():
    observations = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2)
    critic = DummyCritic()

    with pytest.raises(ValueError, match="one weight per"):
        SAC.compute_policy_loss(policy, [critic], observations=observations, alpha=0.2, critic_weights=[1.0, 0.5])


def test_compute_policy_loss_with_multi_critic_validates_non_empty():
    observations = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2)

    with pytest.raises(ValueError, match="non-empty"):
        SAC.compute_policy_loss(
            policy, [], observations=observations, critic_observations=observations, alpha=0.2, critic_weights=[]
        )


def test_compute_policy_loss_weights_require_sequence():
    with pytest.raises(ValueError, match="requires a sequence"):
        SAC.compute_policy_loss(
            DummyPolicy(action_dim=2), DummyCritic(), observations=torch.zeros(4, 3), alpha=0.2, critic_weights=[1.0]
        )


def test_compute_policy_loss_raises_on_shape_mismatch():
    observations = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2, log_prob_shape="flat")
    critic = DummyCritic(output_shape="column")

    with pytest.raises(ValueError, match="same shape"):
        SAC.compute_policy_loss(policy, critic, observations=observations, alpha=0.2)


def test_compute_policy_loss_value():
    observations = torch.zeros(4, 3)
    output = SAC.compute_policy_loss(DummyPolicy(action_dim=2), DummyCritic(), observations=observations, alpha=0.2)
    # log_prob = 0 and min(q1, q2) = 1.
    assert torch.allclose(output["loss"], torch.tensor(-1.0))


def test_compute_policy_loss_multi_critic_weighted_sum():
    observations = torch.zeros(4, 3)
    policy = DummyPolicy(action_dim=2)
    single = SAC.compute_policy_loss(policy, DummyCritic(), observations=observations, alpha=0.2)["loss"]
    one = SAC.compute_policy_loss(policy, [DummyCritic()], observations=observations, alpha=0.2, critic_weights=[1.0])["loss"]
    two = SAC.compute_policy_loss(
        policy, [DummyCritic(), LinearCritic()], observations=torch.ones(4, 3), alpha=0.2, critic_weights=[0.5, 2.0]
    )["loss"]
    assert torch.equal(single, one)
    # 0.5 * -1 + 2.0 * -(min(3, 2.5)) = -5.5
    assert torch.allclose(two, torch.tensor(-5.5))


def test_compute_policy_loss_critic_observations():
    policy = DummyPolicy(action_dim=2)
    observations = torch.zeros(4, 3)
    base = SAC.compute_policy_loss(policy, LinearCritic(), observations=observations, alpha=0.2)["loss"]
    same = SAC.compute_policy_loss(
        policy, LinearCritic(), observations=observations, critic_observations=observations, alpha=0.2
    )["loss"]
    other = SAC.compute_policy_loss(
        policy, LinearCritic(), observations=observations, critic_observations=torch.ones(4, 3), alpha=0.2
    )["loss"]
    assert torch.equal(base, same)
    assert not torch.equal(base, other)


def test_compute_policy_loss_rejects_positional_data():
    with pytest.raises(TypeError):
        SAC.compute_policy_loss(DummyPolicy(action_dim=2), DummyCritic(), torch.zeros(4, 3), 0.2)


def test_compute_critic_loss_accepts_bool_done():
    output = SAC.compute_critic_loss(
        DummyPolicy(action_dim=2), DummyCritic(), DummyCritic(), alpha=0.2, gamma=0.99, **_critic_batch()
    )

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])
    assert set(output) == {"loss", "q1", "q2", "q_target"}


def test_compute_critic_loss_value():
    batch = _critic_batch()
    output = SAC.compute_critic_loss(DummyPolicy(action_dim=2), LinearCritic(), LinearCritic(), alpha=0.2, gamma=0.9, **batch)
    q1 = batch["observations"].sum(-1)
    q2 = q1 * 0.5 + 1.0
    not_done = 1 - batch["terminated"].float()
    q_target = batch["rewards"] + 0.9 * not_done * torch.min(torch.full((4,), 3.0), torch.full((4,), 2.5))
    expected = 0.5 * (q1 - q_target).pow(2).mean() + 0.5 * (q2 - q_target).pow(2).mean()
    assert torch.allclose(output["loss"], expected)
    assert torch.allclose(output["q_target"], q_target.mean())


def test_compute_critic_loss_critic_observations():
    policy = DummyPolicy(action_dim=2)
    batch = _critic_batch()
    base = SAC.compute_critic_loss(policy, LinearCritic(), LinearCritic(), alpha=0.2, gamma=0.99, **batch)["loss"]
    same = SAC.compute_critic_loss(
        policy, LinearCritic(), LinearCritic(), alpha=0.2, gamma=0.99,
        critic_observations=batch["observations"], next_critic_observations=batch["next_observations"], **batch
    )["loss"]
    assert torch.equal(base, same)


def test_compute_critic_loss_asymmetric_inputs():
    # The critic sees critic_observations; the actor only sees next_observations.
    batch = _critic_batch()
    batch["observations"] = torch.zeros(4, 7)  # actor features of a different width, unused by the critic
    batch["next_observations"] = torch.zeros(4, 7)
    output = SAC.compute_critic_loss(
        DummyPolicy(action_dim=2), LinearCritic(), LinearCritic(), alpha=0.2, gamma=0.99,
        critic_observations=torch.ones(4, 3), next_critic_observations=torch.ones(4, 3), **batch
    )
    assert torch.allclose(output["q1"], torch.tensor(3.0))


def test_compute_critic_loss_rejects_broadcasting():
    batch = _critic_batch()
    batch["rewards"] = batch["rewards"].unsqueeze(-1)
    batch["terminated"] = batch["terminated"].unsqueeze(-1)
    with pytest.raises(ValueError, match="same shape"):
        SAC.compute_critic_loss(DummyPolicy(action_dim=2), DummyCritic(), DummyCritic(), alpha=0.2, gamma=0.99, **batch)


def test_compute_critic_loss_rejects_positional_data():
    batch = _critic_batch()
    with pytest.raises(TypeError):
        SAC.compute_critic_loss(DummyPolicy(action_dim=2), DummyCritic(), DummyCritic(), *batch.values(), 0.2, 0.99)


def test_compute_alpha_loss_returns_loss_key():
    log_alpha = torch.zeros(1, requires_grad=True)
    output = SAC.compute_alpha_loss(DummyPolicy(action_dim=2), log_alpha=log_alpha, observations=torch.zeros(4, 3), target_entropy=-2.0)
    assert "alpha_loss" not in output
    # log_prob = 0, so loss = -exp(0) * (0 - 2) = 2.
    assert torch.allclose(output["loss"], torch.tensor(2.0))
    output["loss"].backward()
    assert log_alpha.grad is not None
    with pytest.raises(TypeError):
        SAC.compute_alpha_loss(DummyPolicy(action_dim=2), log_alpha, torch.zeros(4, 3), -2.0)


def test_update_target_param_soft_update():
    model, target = nn.Linear(2, 2), nn.Linear(2, 2)
    expected = [0.9 * t + 0.1 * p for p, t in zip(model.parameters(), target.parameters())]
    SAC.update_target_param(model, target, 0.1)
    for value, param in zip(expected, target.parameters()):
        assert torch.allclose(value, param)
