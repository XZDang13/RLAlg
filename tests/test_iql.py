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


class ObsValueModel(nn.Module):
    """V(s) = sum(s), so tests can tell which observations reached the value model."""

    def forward(self, observation):
        return ValueStep(value=observation.sum(dim=-1))


class ObsCritic(nn.Module):
    def forward(self, observation, action):
        q = observation.sum(dim=-1) + action.sum(dim=-1)
        return ValueStep(value=q), ValueStep(value=q + 1.0)


class LogProbPolicy(nn.Module):
    """log pi(a|s) = -sum((a - s[:, :A])^2) so the policy sees only actor observations."""

    def forward(self, observation, action):
        log_prob = -((action - observation[:, : action.shape[-1]]) ** 2).sum(dim=-1)
        pi = Categorical(logits=torch.zeros(action.shape[0], 2))
        return DiscretePolicyStep(pi=pi, action=action, log_prob=log_prob, entropy=torch.zeros_like(log_prob))


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


def _critic_batch(batch: int = 4):
    return {
        "observations": torch.zeros(batch, 3),
        "actions": torch.zeros(batch, 2),
        "rewards": torch.ones(batch),
        "next_observations": torch.zeros(batch, 3),
        "terminated": torch.tensor([True, False, True, False])[:batch],
    }


def test_compute_critic_loss_accepts_bool_terminated():
    batch = 4
    value_model = DummyValueModel(values=torch.full((batch,), 0.5))
    critic = DummyCritic(q1=torch.full((batch,), 1.0), q2=torch.full((batch,), 1.2))

    output = IQL.compute_critic_loss(value_model, critic, gamma=0.99, **_critic_batch(batch))

    assert isinstance(output["loss"], torch.Tensor)
    assert torch.isfinite(output["loss"])
    assert set(output) == {"loss", "q1", "q2", "q_target"}


def test_compute_critic_loss_is_half_mse_summed_over_twins():
    batch = 4
    value_model = DummyValueModel(values=torch.full((batch,), 0.5))
    critic = DummyCritic(q1=torch.full((batch,), 1.0), q2=torch.full((batch,), 1.2))
    data = _critic_batch(batch)

    output = IQL.compute_critic_loss(value_model, critic, gamma=0.9, **data)

    target = data["rewards"] + 0.9 * (1 - data["terminated"].float()) * 0.5
    expected = 0.5 * ((1.0 - target) ** 2).mean() + 0.5 * ((1.2 - target) ** 2).mean()
    torch.testing.assert_close(output["loss"], expected)
    torch.testing.assert_close(output["q_target"], target.mean())


def test_compute_critic_loss_rejects_broadcasting_shapes():
    batch = 4
    value_model = DummyValueModel(values=torch.full((batch,), 0.5))
    critic = DummyCritic(q1=torch.full((batch,), 1.0), q2=torch.full((batch,), 1.2))

    data = _critic_batch(batch)
    data["rewards"] = data["rewards"].unsqueeze(-1)
    with pytest.raises(ValueError, match="same shape"):
        IQL.compute_critic_loss(value_model, critic, **data)

    data = _critic_batch(batch)
    data["terminated"] = data["terminated"].unsqueeze(-1)
    with pytest.raises(ValueError, match="same shape"):
        IQL.compute_critic_loss(value_model, critic, **data)

    # [B, 1] predictions against [B] targets.
    wide_critic = DummyCritic(q1=torch.ones(batch, 1), q2=torch.ones(batch, 1))
    with pytest.raises(ValueError, match="same shape"):
        IQL.compute_critic_loss(value_model, wide_critic, **_critic_batch(batch))


def test_compute_value_loss_rejects_invalid_expectile():
    batch = 4
    value_model = DummyValueModel(values=torch.zeros(batch))
    critic_target = DummyCritic(q1=torch.zeros(batch), q2=torch.zeros(batch))

    with pytest.raises(ValueError, match="expectile"):
        IQL.compute_value_loss(
            value_model,
            critic_target,
            observations=torch.zeros(batch, 3),
            actions=torch.zeros(batch, 2),
            expectile=1.5,
        )


def test_compute_value_loss_is_expectile_regression():
    value_model = DummyValueModel(values=torch.tensor([0.0, 0.0]))
    critic_target = DummyCritic(q1=torch.tensor([1.0, -2.0]), q2=torch.tensor([3.0, -1.0]))

    output = IQL.compute_value_loss(
        value_model, critic_target, observations=torch.zeros(2, 3), actions=torch.zeros(2, 2), expectile=0.7
    )

    # min q = [1, -2]: 0.7 * 1 for the positive residual, 0.3 * 4 for the negative one.
    torch.testing.assert_close(output["loss"], torch.tensor((0.7 * 1.0 + 0.3 * 4.0) / 2))


def test_compute_policy_loss_rejects_non_positive_temperature():
    batch = 4
    policy_model = DummyDiscretePolicy(
        log_prob=torch.zeros(batch),
        entropy=torch.zeros(batch),
    )
    value_model = DummyValueModel(values=torch.zeros(batch))
    critic_target = DummyCritic(q1=torch.zeros(batch), q2=torch.zeros(batch))

    with pytest.raises(ValueError, match="temperature"):
        IQL.compute_policy_loss(
            policy_model,
            value_model,
            critic_target,
            observations=torch.zeros(batch, 3),
            actions=torch.zeros(batch, 2),
            temperature=0.0,
        )


def test_compute_policy_loss_is_advantage_weighted_log_likelihood():
    log_prob = torch.tensor([-1.0, -2.0])
    policy_model = DummyDiscretePolicy(log_prob=log_prob, entropy=torch.zeros(2))
    value_model = DummyValueModel(values=torch.tensor([0.0, 1.0]))
    critic_target = DummyCritic(q1=torch.tensor([1.0, 0.0]), q2=torch.tensor([2.0, 5.0]))

    output = IQL.compute_policy_loss(
        policy_model,
        value_model,
        critic_target,
        observations=torch.zeros(2, 3),
        actions=torch.zeros(2, 2),
        temperature=3.0,
    )

    weights = torch.exp(torch.tensor([1.0, -1.0]) * 3.0)
    torch.testing.assert_close(output["loss"], -(weights * log_prob).mean())


def test_compute_policy_loss_critic_observations():
    generator = torch.Generator().manual_seed(0)
    observations = torch.randn(4, 3, generator=generator)
    actions = torch.randn(4, 2, generator=generator)
    kwargs = {"observations": observations, "actions": actions, "temperature": 0.5}

    default = IQL.compute_policy_loss(LogProbPolicy(), ObsValueModel(), ObsCritic(), **kwargs)
    explicit = IQL.compute_policy_loss(
        LogProbPolicy(), ObsValueModel(), ObsCritic(), critic_observations=observations, **kwargs
    )
    torch.testing.assert_close(explicit["loss"], default["loss"])

    # Value and critic see the privileged observations; the policy keeps the actor observations.
    critic_observations = torch.randn(4, 6, generator=generator)
    output = IQL.compute_policy_loss(
        LogProbPolicy(), ObsValueModel(), ObsCritic(), critic_observations=critic_observations, **kwargs
    )
    log_prob = -((actions - observations[:, :2]) ** 2).sum(dim=-1)
    weights = torch.clamp(torch.exp(actions.sum(dim=-1) * 0.5), max=100.0)
    torch.testing.assert_close(output["loss"], -(weights * log_prob).mean())


def test_compute_value_loss_raises_on_shape_mismatch():
    batch = 4
    value_model = DummyValueModel(values=torch.zeros(batch))
    critic_target = DummyCritic(q1=torch.zeros(batch, 1), q2=torch.zeros(batch, 1))

    with pytest.raises(ValueError, match="same shape"):
        IQL.compute_value_loss(
            value_model,
            critic_target,
            observations=torch.zeros(batch, 3),
            actions=torch.zeros(batch, 2),
            expectile=0.7,
        )


def test_data_arguments_are_keyword_only():
    batch = 4
    value_model = DummyValueModel(values=torch.zeros(batch))
    critic = DummyCritic(q1=torch.zeros(batch), q2=torch.zeros(batch))
    policy = DummyDiscretePolicy(log_prob=torch.zeros(batch), entropy=torch.zeros(batch))
    observations, actions = torch.zeros(batch, 3), torch.zeros(batch, 2)

    with pytest.raises(TypeError):
        IQL.compute_critic_loss(value_model, critic, *_critic_batch(batch).values(), 0.99)
    with pytest.raises(TypeError):
        IQL.compute_value_loss(value_model, critic, observations, actions, 0.7)
    with pytest.raises(TypeError):
        IQL.compute_policy_loss(policy, value_model, critic, observations, actions, 3.0)


def test_update_target_param():
    model = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.0)
        for param in target.parameters():
            param.fill_(0.0)

    IQL.update_target_param(model, target, tau=0.25)

    for param in target.parameters():
        torch.testing.assert_close(param, torch.full_like(param, 0.25))
