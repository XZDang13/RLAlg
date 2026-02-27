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
