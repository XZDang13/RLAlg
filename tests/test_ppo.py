import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical

from RLAlg.alg.ppo import PPO
from RLAlg.nn.steps import DiscretePolicyStep, ValueStep


class DummyDiscretePolicy(nn.Module):
    def __init__(self, log_prob: torch.Tensor, entropy: torch.Tensor):
        super().__init__()
        self.log_prob = log_prob
        self.entropy = entropy

    def forward(self, observations, actions):
        batch = self.log_prob.shape[0]
        pi = Categorical(logits=torch.zeros(batch, 2, dtype=self.log_prob.dtype, device=self.log_prob.device))
        return DiscretePolicyStep(
            pi=pi,
            action=actions,
            log_prob=self.log_prob,
            entropy=self.entropy,
        )


class DummyValueModel(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.values = values

    def forward(self, observations):
        return ValueStep(value=self.values)


class DummyRecurrentDiscretePolicy(nn.Module):
    def __init__(self, log_prob: torch.Tensor, entropy: torch.Tensor):
        super().__init__()
        self.log_prob = log_prob
        self.entropy = entropy
        self.last_observations = None

    def forward(self, observations, actions, initial_state, episode_starts):
        self.last_observations = observations
        pi = Categorical(
            logits=torch.zeros(*self.log_prob.shape, 2, dtype=self.log_prob.dtype, device=self.log_prob.device)
        )
        step = DiscretePolicyStep(
            pi=pi,
            action=actions,
            log_prob=self.log_prob,
            entropy=self.entropy,
        )
        next_state = None if initial_state is None else initial_state + 1
        return step, next_state


class DummyRecurrentValueModel(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.values = values

    def forward(self, observations, initial_state, episode_starts):
        next_state = None if initial_state is None else initial_state + 1
        return ValueStep(value=self.values), next_state


def test_compute_policy_loss_raises_on_shape_mismatch():
    model = DummyDiscretePolicy(
        log_prob=torch.tensor([[0.1], [0.2]]),
        entropy=torch.tensor([0.3, 0.4]),
    )

    with pytest.raises(ValueError, match="same shape"):
        PPO.compute_policy_loss(
            policy_model=model,
            log_probs_hat=torch.tensor([0.1, 0.2]),
            observations=torch.zeros(2, 1),
            actions=torch.tensor([0, 1]),
            advantages=torch.tensor([[1.0], [1.0]]),
            clip_ratio=0.2,
        )


def test_compute_value_loss_raises_on_shape_mismatch():
    value_model = DummyValueModel(values=torch.tensor([[1.0], [2.0]]))

    with pytest.raises(ValueError, match="same shape"):
        PPO.compute_value_loss(
            value_model=value_model,
            observations=torch.zeros(2, 3),
            returns=torch.tensor([1.0, 2.0]),
        )


def test_compute_clipped_value_loss_raises_on_shape_mismatch():
    value_model = DummyValueModel(values=torch.tensor([[1.0], [2.0]]))

    with pytest.raises(ValueError, match="same shape"):
        PPO.compute_clipped_value_loss(
            value_model=value_model,
            observations=torch.zeros(2, 3),
            values_hat=torch.tensor([1.0, 2.0]),
            returns=torch.tensor([[1.0], [2.0]]),
            clip_ratio=0.2,
        )


def test_compute_policy_loss_with_multi_critic_raises_on_length_mismatch():
    model = DummyDiscretePolicy(
        log_prob=torch.tensor([[0.1], [0.2]]),
        entropy=torch.tensor([0.3, 0.4]),
    )

    with pytest.raises(ValueError, match="same length"):
        PPO.compute_policy_loss_with_multi_critic(
            policy_model=model,
            log_probs_hat=torch.tensor([[0.1], [0.2]]),
            observations=torch.zeros(2, 1),
            actions=torch.tensor([0, 1]),
            advantages_list=[torch.ones(2, 1)],
            weights=[1.0, 0.5],
            clip_ratio=0.2,
        )


def test_compute_policy_loss_with_multi_critic_raises_on_empty_lists():
    model = DummyDiscretePolicy(
        log_prob=torch.tensor([[0.1], [0.2]]),
        entropy=torch.tensor([0.3, 0.4]),
    )

    with pytest.raises(ValueError, match="non-empty"):
        PPO.compute_policy_loss_with_multi_critic(
            policy_model=model,
            log_probs_hat=torch.tensor([[0.1], [0.2]]),
            observations=torch.zeros(2, 1),
            actions=torch.tensor([0, 1]),
            advantages_list=[],
            weights=[],
            clip_ratio=0.2,
        )


def test_compute_policy_loss_with_multi_critic_returns_tensor_loss():
    model = DummyDiscretePolicy(
        log_prob=torch.tensor([[0.1], [0.2]]),
        entropy=torch.tensor([0.3, 0.4]),
    )

    output = PPO.compute_policy_loss_with_multi_critic(
        policy_model=model,
        log_probs_hat=torch.tensor([[0.1], [0.2]]),
        observations=torch.zeros(2, 1),
        actions=torch.tensor([0, 1]),
        advantages_list=[torch.ones(2, 1)],
        weights=[1.0],
        clip_ratio=0.2,
    )

    assert isinstance(output["loss"], torch.Tensor)


def test_compute_policy_loss_recurrent_matches_feedforward_for_t1():
    log_prob = torch.tensor([[0.1, 0.2, -0.1]])
    entropy = torch.tensor([[0.3, 0.4, 0.5]])
    log_probs_hat = torch.tensor([[0.0, 0.1, -0.2]])
    advantages = torch.tensor([[1.0, 1.0, 1.0]])

    ff_model = DummyDiscretePolicy(log_prob=log_prob, entropy=entropy)
    rec_model = DummyRecurrentDiscretePolicy(log_prob=log_prob, entropy=entropy)

    ff_output = PPO.compute_policy_loss(
        policy_model=ff_model,
        log_probs_hat=log_probs_hat,
        observations=torch.zeros(1, 3, 2),
        actions=torch.tensor([[0, 1, 0]]),
        advantages=advantages,
        clip_ratio=0.2,
    )
    initial_state = torch.zeros(1, 3, 4)
    rec_output = PPO.compute_policy_loss_recurrent(
        policy_model=rec_model,
        log_probs_hat=log_probs_hat,
        observations=torch.zeros(1, 3, 2),
        actions=torch.tensor([[0, 1, 0]]),
        advantages=advantages,
        clip_ratio=0.2,
        episode_starts=torch.zeros_like(log_prob, dtype=torch.bool),
        initial_state=initial_state,
        valid_mask=torch.ones_like(log_prob, dtype=torch.bool),
    )

    assert torch.allclose(rec_output["loss"], ff_output["loss"])
    assert torch.allclose(rec_output["entropy"], ff_output["entropy"])
    assert torch.allclose(rec_output["kl_divergence"], ff_output["kl_divergence"])
    assert torch.equal(rec_output["next_state"], initial_state + 1)


def test_compute_policy_loss_recurrent_applies_valid_mask():
    log_prob = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    entropy = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    model = DummyRecurrentDiscretePolicy(log_prob=log_prob, entropy=entropy)
    mask = torch.tensor([[True, True], [True, False]])

    output = PPO.compute_policy_loss_recurrent(
        policy_model=model,
        log_probs_hat=torch.zeros_like(log_prob),
        observations=torch.zeros(2, 2, 1),
        actions=torch.zeros(2, 2, dtype=torch.long),
        advantages=torch.ones_like(log_prob),
        clip_ratio=10.0,
        episode_starts=torch.zeros_like(log_prob, dtype=torch.bool),
        valid_mask=mask,
    )

    expected_loss = -torch.exp(log_prob)[mask].mean()
    expected_entropy = entropy[mask].mean()
    expected_kl = (((torch.exp(log_prob) - 1.0) - log_prob)[mask]).mean()

    assert torch.allclose(output["loss"], expected_loss)
    assert torch.allclose(output["entropy"], expected_entropy)
    assert torch.allclose(output["kl_divergence"], expected_kl)


def test_compute_policy_loss_recurrent_accepts_dict_observations():
    log_prob = torch.tensor([[0.1, -0.1]])
    entropy = torch.tensor([[0.2, 0.3]])
    model = DummyRecurrentDiscretePolicy(log_prob=log_prob, entropy=entropy)
    observations = {
        "state": torch.zeros(1, 2, 3),
        "goal": torch.zeros(1, 2, 1),
    }

    output = PPO.compute_policy_loss_recurrent(
        policy_model=model,
        log_probs_hat=torch.zeros_like(log_prob),
        observations=observations,
        actions=torch.zeros(1, 2, dtype=torch.long),
        advantages=torch.ones_like(log_prob),
        clip_ratio=0.2,
        episode_starts=torch.zeros_like(log_prob, dtype=torch.bool),
    )

    assert isinstance(model.last_observations, dict)
    assert torch.isfinite(output["loss"])


def test_compute_policy_loss_with_multi_critic_recurrent_raises_on_length_mismatch():
    log_prob = torch.tensor([[0.1, 0.2]])
    entropy = torch.tensor([[0.3, 0.4]])
    model = DummyRecurrentDiscretePolicy(log_prob=log_prob, entropy=entropy)

    with pytest.raises(ValueError, match="same length"):
        PPO.compute_policy_loss_with_multi_critic_recurrent(
            policy_model=model,
            log_probs_hat=torch.zeros_like(log_prob),
            observations=torch.zeros(1, 2, 1),
            actions=torch.zeros(1, 2, dtype=torch.long),
            advantages_list=[torch.ones_like(log_prob)],
            weights=[1.0, 0.5],
            clip_ratio=0.2,
            episode_starts=torch.zeros_like(log_prob, dtype=torch.bool),
        )


def test_compute_value_loss_recurrent_matches_feedforward_for_t1():
    values = torch.tensor([[1.0, 2.0, 3.0]])
    returns = torch.tensor([[1.5, 2.5, 2.0]])

    ff_model = DummyValueModel(values=values)
    rec_model = DummyRecurrentValueModel(values=values)

    ff_output = PPO.compute_value_loss(
        value_model=ff_model,
        observations=torch.zeros(1, 3, 2),
        returns=returns,
    )
    rec_output = PPO.compute_value_loss_recurrent(
        value_model=rec_model,
        observations=torch.zeros(1, 3, 2),
        returns=returns,
        episode_starts=torch.zeros_like(values, dtype=torch.bool),
        valid_mask=torch.ones_like(values, dtype=torch.bool),
    )

    assert torch.allclose(rec_output["loss"], ff_output["loss"])


def test_compute_value_loss_recurrent_applies_valid_mask():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    returns = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    mask = torch.tensor([[True, True], [False, True]])
    model = DummyRecurrentValueModel(values=values)

    output = PPO.compute_value_loss_recurrent(
        value_model=model,
        observations=torch.zeros(2, 2, 1),
        returns=returns,
        episode_starts=torch.zeros_like(values, dtype=torch.bool),
        valid_mask=mask,
    )

    expected = 0.5 * ((returns - values) ** 2)[mask].mean()
    assert torch.allclose(output["loss"], expected)
