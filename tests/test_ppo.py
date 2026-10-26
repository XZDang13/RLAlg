import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical

from RLAlg.alg.ppo import PPO
from RLAlg.nn.steps import DiscretePolicyStep, StochasticContinuousPolicyStep, ValueStep


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


class DummyContinuousPolicy(nn.Module):
    def __init__(self, log_prob: torch.Tensor, log_std: torch.Tensor):
        super().__init__()
        self.log_prob = log_prob
        self.log_std = log_std

    def forward(self, observations, actions):
        return StochasticContinuousPolicyStep(
            pi=None,
            action=actions,
            log_prob=self.log_prob,
            mean=torch.zeros_like(actions, dtype=self.log_prob.dtype),
            log_std=self.log_std,
            entropy=torch.ones_like(self.log_prob),
        )


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
        entropy=torch.tensor([[0.3], [0.4]]),
    )

    with pytest.raises(ValueError, match="same shape"):
        PPO.compute_policy_loss(
            model,
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
            value_model,
            observations=torch.zeros(2, 3),
            returns=torch.tensor([1.0, 2.0]),
        )


def test_compute_clipped_value_loss_raises_on_shape_mismatch():
    value_model = DummyValueModel(values=torch.tensor([[1.0], [2.0]]))

    with pytest.raises(ValueError, match="same shape"):
        PPO.compute_clipped_value_loss(
            value_model,
            observations=torch.zeros(2, 3),
            values_hat=torch.tensor([1.0, 2.0]),
            returns=torch.tensor([[1.0], [2.0]]),
            clip_ratio=0.2,
        )


def test_compute_policy_loss_multi_critic_raises_on_length_mismatch():
    model = DummyDiscretePolicy(
        log_prob=torch.tensor([[0.1], [0.2]]),
        entropy=torch.tensor([0.3, 0.4]),
    )

    with pytest.raises(ValueError, match="one weight per"):
        PPO.compute_policy_loss(
            model,
            log_probs_hat=torch.tensor([[0.1], [0.2]]),
            observations=torch.zeros(2, 1),
            actions=torch.tensor([0, 1]),
            advantages=[torch.ones(2, 1)],
            advantage_weights=[1.0, 0.5],
            clip_ratio=0.2,
        )


def test_compute_policy_loss_multi_critic_raises_on_empty_lists():
    model = DummyDiscretePolicy(
        log_prob=torch.tensor([[0.1], [0.2]]),
        entropy=torch.tensor([0.3, 0.4]),
    )

    with pytest.raises(ValueError, match="non-empty"):
        PPO.compute_policy_loss(
            model,
            log_probs_hat=torch.tensor([[0.1], [0.2]]),
            observations=torch.zeros(2, 1),
            actions=torch.tensor([0, 1]),
            advantages=[],
            advantage_weights=[],
            clip_ratio=0.2,
        )


def test_compute_policy_loss_multi_critic_returns_tensor_loss():
    model = DummyDiscretePolicy(
        log_prob=torch.tensor([[0.1], [0.2]]),
        entropy=torch.tensor([[0.3], [0.4]]),
    )

    output = PPO.compute_policy_loss(
        model,
        log_probs_hat=torch.tensor([[0.1], [0.2]]),
        observations=torch.zeros(2, 1),
        actions=torch.tensor([0, 1]),
        advantages=[torch.ones(2, 1)],
        advantage_weights=[1.0],
        clip_ratio=0.2,
    )

    assert isinstance(output["loss"], torch.Tensor)


def test_compute_policy_loss_multi_critic_applies_entropy_bonus():
    entropy = torch.tensor([2.0, 4.0])
    model = DummyDiscretePolicy(log_prob=torch.zeros(2), entropy=entropy)

    output = PPO.compute_policy_loss(
        model,
        log_probs_hat=torch.zeros(2),
        observations=torch.zeros(2, 1),
        actions=torch.zeros(2, dtype=torch.long),
        advantages=[torch.ones(2)],
        advantage_weights=[1.0],
        clip_ratio=0.2,
        entropy_coef=0.1,
    )

    torch.testing.assert_close(output["entropy"], torch.tensor(3.0))
    torch.testing.assert_close(output["loss"], torch.tensor(-1.3))


def test_compute_policy_loss_applies_entropy_bonus():
    entropy = torch.tensor([2.0, 4.0])
    model = DummyDiscretePolicy(log_prob=torch.zeros(2), entropy=entropy)

    output = PPO.compute_policy_loss(
        model,
        log_probs_hat=torch.zeros(2),
        observations=torch.zeros(2, 1),
        actions=torch.zeros(2, dtype=torch.long),
        advantages=torch.ones(2),
        clip_ratio=0.2,
        entropy_coef=0.1,
    )

    torch.testing.assert_close(output["entropy"], torch.tensor(3.0))
    torch.testing.assert_close(output["loss"], torch.tensor(-1.3))


def test_compute_policy_loss_returns_trainer_metrics_for_continuous_policy():
    ratio = torch.tensor([1.25, 0.70])
    log_std = torch.log(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    advantages = torch.tensor([1.0, 3.0])
    model = DummyContinuousPolicy(log_prob=torch.log(ratio), log_std=log_std)

    output = PPO.compute_policy_loss(
        model,
        log_probs_hat=torch.zeros_like(ratio),
        observations=torch.zeros(2, 1),
        actions=torch.zeros(2, 2),
        advantages=advantages,
        clip_ratio=0.2,
    )

    assert torch.allclose(output["policy_clip_fraction"], torch.tensor(1.0))
    assert torch.allclose(output["action_log_std"], log_std.mean())
    assert torch.allclose(output["action_std"], torch.tensor(2.5))
    assert torch.allclose(output["advantage_mean"], advantages.mean())
    assert torch.allclose(output["advantage_std"], advantages.std(unbiased=False))


def test_compute_clipped_value_loss_returns_trainer_metrics():
    values = torch.tensor([1.30, 0.90, 1.00])
    values_hat = torch.tensor([1.00, 1.00, 1.00])
    returns = torch.tensor([1.00, 2.00, 3.00])
    value_model = DummyValueModel(values=values)

    output = PPO.compute_clipped_value_loss(
        value_model,
        observations=torch.zeros(3, 1),
        values_hat=values_hat,
        returns=returns,
        clip_ratio=0.2,
    )

    residual = returns - values
    expected_explained_variance = 1.0 - residual.var(unbiased=False) / returns.var(unbiased=False)

    assert torch.allclose(output["value_explained_variance"], expected_explained_variance)
    assert torch.allclose(output["value_clip_fraction"], torch.tensor(1.0 / 3.0))


def test_compute_policy_loss_recurrent_variant_matches_feedforward_for_t1():
    log_prob = torch.tensor([[0.1, 0.2, -0.1]])
    entropy = torch.tensor([[0.3, 0.4, 0.5]])
    log_probs_hat = torch.tensor([[0.0, 0.1, -0.2]])
    advantages = torch.tensor([[1.0, 1.0, 1.0]])

    ff_model = DummyDiscretePolicy(log_prob=log_prob, entropy=entropy)
    rec_model = DummyRecurrentDiscretePolicy(log_prob=log_prob, entropy=entropy)

    ff_output = PPO.compute_policy_loss(
        ff_model,
        log_probs_hat=log_probs_hat,
        observations=torch.zeros(1, 3, 2),
        actions=torch.tensor([[0, 1, 0]]),
        advantages=advantages,
        clip_ratio=0.2,
    )
    initial_state = torch.zeros(1, 3, 4)
    rec_output = PPO.compute_policy_loss(
        rec_model,
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


def test_compute_policy_loss_recurrent_variant_applies_valid_mask():
    log_prob = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    entropy = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    model = DummyRecurrentDiscretePolicy(log_prob=log_prob, entropy=entropy)
    mask = torch.tensor([[True, True], [True, False]])

    output = PPO.compute_policy_loss(
        model,
        log_probs_hat=torch.zeros_like(log_prob),
        observations=torch.zeros(2, 2, 1),
        actions=torch.zeros(2, 2, dtype=torch.long),
        advantages=torch.ones_like(log_prob),
        clip_ratio=10.0,
        episode_starts=torch.zeros_like(log_prob, dtype=torch.bool),
        valid_mask=mask,
        entropy_coef=0.1,
    )

    expected_entropy = entropy[mask].mean()
    expected_loss = -torch.exp(log_prob)[mask].mean() - 0.1 * expected_entropy
    expected_kl = (((torch.exp(log_prob) - 1.0) - log_prob)[mask]).mean()

    assert torch.allclose(output["loss"], expected_loss)
    assert torch.allclose(output["entropy"], expected_entropy)
    assert torch.allclose(output["kl_divergence"], expected_kl)


def test_compute_policy_loss_recurrent_variant_accepts_dict_observations():
    log_prob = torch.tensor([[0.1, -0.1]])
    entropy = torch.tensor([[0.2, 0.3]])
    model = DummyRecurrentDiscretePolicy(log_prob=log_prob, entropy=entropy)
    observations = {
        "state": torch.zeros(1, 2, 3),
        "goal": torch.zeros(1, 2, 1),
    }

    output = PPO.compute_policy_loss(
        model,
        log_probs_hat=torch.zeros_like(log_prob),
        observations=observations,
        actions=torch.zeros(1, 2, dtype=torch.long),
        advantages=torch.ones_like(log_prob),
        clip_ratio=0.2,
        episode_starts=torch.zeros_like(log_prob, dtype=torch.bool),
    )

    assert isinstance(model.last_observations, dict)
    assert torch.isfinite(output["loss"])


def test_compute_policy_loss_multi_critic_recurrent_applies_masked_entropy_bonus():
    log_prob = torch.zeros(2, 2)
    entropy = torch.tensor([[1.0, 2.0], [3.0, 100.0]])
    mask = torch.tensor([[True, True], [True, False]])
    model = DummyRecurrentDiscretePolicy(log_prob=log_prob, entropy=entropy)

    output = PPO.compute_policy_loss(
        model,
        log_probs_hat=torch.zeros_like(log_prob),
        observations=torch.zeros(2, 2, 1),
        actions=torch.zeros(2, 2, dtype=torch.long),
        advantages=[torch.ones_like(log_prob)],
        advantage_weights=[1.0],
        clip_ratio=0.2,
        episode_starts=torch.zeros_like(log_prob, dtype=torch.bool),
        valid_mask=mask,
        entropy_coef=0.1,
    )

    torch.testing.assert_close(output["entropy"], torch.tensor(2.0))
    torch.testing.assert_close(output["loss"], torch.tensor(-1.2))


def test_compute_policy_loss_multi_critic_recurrent_raises_on_length_mismatch():
    log_prob = torch.tensor([[0.1, 0.2]])
    entropy = torch.tensor([[0.3, 0.4]])
    model = DummyRecurrentDiscretePolicy(log_prob=log_prob, entropy=entropy)

    with pytest.raises(ValueError, match="one weight per"):
        PPO.compute_policy_loss(
            model,
            log_probs_hat=torch.zeros_like(log_prob),
            observations=torch.zeros(1, 2, 1),
            actions=torch.zeros(1, 2, dtype=torch.long),
            advantages=[torch.ones_like(log_prob)],
            advantage_weights=[1.0, 0.5],
            clip_ratio=0.2,
            episode_starts=torch.zeros_like(log_prob, dtype=torch.bool),
        )


def test_compute_value_loss_recurrent_matches_feedforward_for_t1():
    values = torch.tensor([[1.0, 2.0, 3.0]])
    returns = torch.tensor([[1.5, 2.5, 2.0]])

    ff_model = DummyValueModel(values=values)
    rec_model = DummyRecurrentValueModel(values=values)

    ff_output = PPO.compute_value_loss(
        ff_model,
        observations=torch.zeros(1, 3, 2),
        returns=returns,
    )
    rec_output = PPO.compute_value_loss(
        rec_model,
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

    output = PPO.compute_value_loss(
        model,
        observations=torch.zeros(2, 2, 1),
        returns=returns,
        episode_starts=torch.zeros_like(values, dtype=torch.bool),
        valid_mask=mask,
    )

    expected = 0.5 * ((returns - values) ** 2)[mask].mean()
    assert torch.allclose(output["loss"], expected)


def test_compute_policy_loss_rejects_positional_data():
    model = DummyDiscretePolicy(log_prob=torch.zeros(2), entropy=torch.ones(2))

    with pytest.raises(TypeError):
        PPO.compute_policy_loss(model, torch.zeros(2), torch.zeros(2, 1), torch.zeros(2), torch.ones(2), 0.2)


def test_value_losses_reject_positional_data():
    model = DummyValueModel(values=torch.zeros(2))

    with pytest.raises(TypeError):
        PPO.compute_value_loss(model, torch.zeros(2, 1), torch.zeros(2))
    with pytest.raises(TypeError):
        PPO.compute_clipped_value_loss(model, torch.zeros(2, 1), torch.zeros(2), torch.zeros(2), 0.2)


def test_single_advantage_list_with_unit_weight_matches_single_advantage():
    log_prob = torch.tensor([0.1, -0.3, 0.2])
    model = DummyContinuousPolicy(log_prob=log_prob, log_std=torch.zeros(3, 2))
    kwargs = dict(
        log_probs_hat=torch.zeros(3),
        observations=torch.zeros(3, 1),
        actions=torch.ones(3, 2),
        clip_ratio=0.2,
        entropy_coef=0.01,
        regularization_weight=0.1,
    )

    single = PPO.compute_policy_loss(model, advantages=torch.tensor([1.0, -2.0, 0.5]), **kwargs)
    listed = PPO.compute_policy_loss(
        model, advantages=[torch.tensor([1.0, -2.0, 0.5])], advantage_weights=[1.0], **kwargs,
    )

    assert single.keys() == listed.keys()
    for key in single:
        torch.testing.assert_close(listed[key], single[key])


def test_compute_policy_loss_sums_weighted_per_critic_objectives():
    log_prob = torch.tensor([0.1, -0.3])
    model = DummyDiscretePolicy(log_prob=log_prob, entropy=torch.zeros(2))
    advantages = [torch.tensor([1.0, -1.0]), torch.tensor([-2.0, 3.0])]
    weights = [0.7, 0.3]
    kwargs = dict(log_probs_hat=torch.zeros(2), observations=torch.zeros(2, 1), actions=torch.zeros(2), clip_ratio=0.2)

    output = PPO.compute_policy_loss(model, advantages=advantages, advantage_weights=weights, **kwargs)
    expected = sum(
        weight * PPO.compute_policy_loss(model, advantages=advantage, **kwargs)["loss"]
        for advantage, weight in zip(advantages, weights)
    )

    torch.testing.assert_close(output["loss"], expected)
    torch.testing.assert_close(output["advantage_mean"], (0.7 * advantages[0] + 0.3 * advantages[1]).mean())


def test_compute_policy_loss_advantage_weights_require_a_sequence():
    model = DummyDiscretePolicy(log_prob=torch.zeros(2), entropy=torch.ones(2))

    with pytest.raises(ValueError, match="advantage_weights"):
        PPO.compute_policy_loss(
            model,
            log_probs_hat=torch.zeros(2),
            observations=torch.zeros(2, 1),
            actions=torch.zeros(2),
            advantages=torch.ones(2),
            advantage_weights=[1.0],
            clip_ratio=0.2,
        )
    with pytest.raises(ValueError, match="advantage_weights"):
        PPO.compute_policy_loss(
            model,
            log_probs_hat=torch.zeros(2),
            observations=torch.zeros(2, 1),
            actions=torch.zeros(2),
            advantages=[torch.ones(2)],
            clip_ratio=0.2,
        )


def test_compute_policy_loss_valid_mask_without_recurrence_matches_masked_batch():
    log_prob = torch.tensor([0.1, -0.2, 0.3, 0.05])
    entropy = torch.tensor([1.0, 2.0, 3.0, 4.0])
    advantages = torch.tensor([1.0, -1.0, 2.0, 0.5])
    mask = torch.tensor([True, False, True, True])

    masked = PPO.compute_policy_loss(
        DummyDiscretePolicy(log_prob=log_prob, entropy=entropy),
        log_probs_hat=torch.zeros(4),
        observations=torch.zeros(4, 1),
        actions=torch.zeros(4),
        advantages=advantages,
        clip_ratio=0.2,
        entropy_coef=0.1,
        valid_mask=mask,
    )
    indexed = PPO.compute_policy_loss(
        DummyDiscretePolicy(log_prob=log_prob[mask], entropy=entropy[mask]),
        log_probs_hat=torch.zeros(3),
        observations=torch.zeros(3, 1),
        actions=torch.zeros(3),
        advantages=advantages[mask],
        clip_ratio=0.2,
        entropy_coef=0.1,
    )

    assert "next_state" not in masked
    for key in ("loss", "entropy", "kl_divergence", "policy_clip_fraction", "advantage_mean", "advantage_std"):
        torch.testing.assert_close(masked[key], indexed[key])


def test_value_losses_valid_mask_without_recurrence_matches_masked_batch():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    values_hat = torch.tensor([1.1, 1.5, 3.0, 3.0])
    returns = torch.tensor([2.0, 0.0, 2.5, 5.0])
    mask = torch.tensor([True, True, False, True])

    plain = PPO.compute_value_loss(DummyValueModel(values), observations=torch.zeros(4, 1), returns=returns, valid_mask=mask)
    clipped = PPO.compute_clipped_value_loss(
        DummyValueModel(values), observations=torch.zeros(4, 1), values_hat=values_hat, returns=returns,
        clip_ratio=0.2, valid_mask=mask,
    )
    plain_ref = PPO.compute_value_loss(DummyValueModel(values[mask]), observations=torch.zeros(3, 1), returns=returns[mask])
    clipped_ref = PPO.compute_clipped_value_loss(
        DummyValueModel(values[mask]), observations=torch.zeros(3, 1), values_hat=values_hat[mask],
        returns=returns[mask], clip_ratio=0.2,
    )

    for output, reference in ((plain, plain_ref), (clipped, clipped_ref)):
        assert "next_state" not in output
        for key in ("loss", "value_explained_variance", "value_clip_fraction"):
            torch.testing.assert_close(output[key], reference[key])


def test_compute_clipped_value_loss_recurrent_applies_valid_mask_and_returns_state():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    values_hat = torch.tensor([[1.5, 2.0], [2.0, 4.5]])
    returns = torch.full((2, 2), 2.0)
    mask = torch.tensor([[True, False], [True, True]])
    initial_state = torch.zeros(1, 2, 3)

    output = PPO.compute_clipped_value_loss(
        DummyRecurrentValueModel(values),
        observations=torch.zeros(2, 2, 1),
        values_hat=values_hat,
        returns=returns,
        clip_ratio=0.2,
        episode_starts=torch.zeros_like(values, dtype=torch.bool),
        initial_state=initial_state,
        valid_mask=mask,
    )

    clipped = values_hat + torch.clamp(values - values_hat, -0.2, 0.2)
    expected = torch.max(0.5 * (returns - values) ** 2, 0.5 * (returns - clipped) ** 2)[mask].mean()
    torch.testing.assert_close(output["loss"], expected)
    assert torch.equal(output["next_state"], initial_state + 1)


def test_initial_state_requires_episode_starts():
    with pytest.raises(ValueError, match="episode_starts"):
        PPO.compute_value_loss(
            DummyValueModel(torch.zeros(2)),
            observations=torch.zeros(2, 1),
            returns=torch.zeros(2),
            initial_state=torch.zeros(1, 2, 3),
        )


def test_recurrent_losses_check_episode_starts_shape():
    log_prob = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="same shape"):
        PPO.compute_policy_loss(
            DummyRecurrentDiscretePolicy(log_prob=log_prob, entropy=torch.ones(2, 2)),
            log_probs_hat=log_prob,
            observations=torch.zeros(2, 2, 1),
            actions=torch.zeros(2, 2),
            advantages=torch.ones(2, 2),
            clip_ratio=0.2,
            episode_starts=torch.zeros(2, 2, 1, dtype=torch.bool),
        )


def test_value_losses_raise_on_column_target_against_flat_values():
    model = DummyValueModel(values=torch.zeros(4))

    with pytest.raises(ValueError, match="same shape"):
        PPO.compute_value_loss(model, observations=torch.zeros(4, 1), returns=torch.zeros(4, 1))
    with pytest.raises(ValueError, match="same shape"):
        PPO.compute_clipped_value_loss(
            model, observations=torch.zeros(4, 1), values_hat=torch.zeros(4, 1), returns=torch.zeros(4), clip_ratio=0.2,
        )


def test_compute_value_loss_uses_half_mse():
    values = torch.tensor([1.0, 2.0, 4.0])
    returns = torch.tensor([0.0, 2.0, 1.0])

    output = PPO.compute_value_loss(DummyValueModel(values), observations=torch.zeros(3, 1), returns=returns)

    torch.testing.assert_close(output["loss"], 0.5 * ((values - returns) ** 2).mean())
