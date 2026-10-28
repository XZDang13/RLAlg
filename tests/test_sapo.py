import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.alg.sapo import SAPO
from RLAlg.alg.shac import SHAC
from RLAlg.nn.steps import ValueStep


def _reference_entropy(entropy, target_entropy):
    # offset_by_target_entropy and scale_by_target_entropy in the reference SAPO.
    entropy = (entropy + abs(target_entropy)) * 0.5
    return entropy * (1.0 / abs(target_entropy))


def _reference_policy_loss(rewards, entropies, next_values, done, gamma, alpha, target_entropy):
    # Port of compute_actor_loss with entropy_in_return=True in the reference SAPO.
    horizon, num_envs = rewards.shape
    rew_acc = torch.zeros(horizon + 1, num_envs, dtype=rewards.dtype)
    discount = torch.ones(num_envs, dtype=rewards.dtype)
    returns = torch.zeros(num_envs, dtype=rewards.dtype)
    for i in range(horizon):
        done_ids = done[i].nonzero(as_tuple=False).squeeze(-1)
        entropy = _reference_entropy(entropies[i], target_entropy)
        rew_acc[i + 1] = rew_acc[i] + discount * (rewards[i] + alpha * entropy)
        if i < horizon - 1:
            returns[done_ids] += rew_acc[i + 1, done_ids] + gamma * discount[done_ids] * next_values[i, done_ids]
        else:
            returns += rew_acc[i + 1] + gamma * discount * next_values[i]
        discount = discount * gamma
        discount[done_ids] = 1.0
        rew_acc[i + 1, done_ids] = 0.0
    return -(returns / horizon).mean()


def _reference_td_lambda(rewards, entropies, next_values, done, gamma, lam_param, alpha, target_entropy):
    # Port of compute_target_values_with_entropy("td-lambda") in the reference SAPO.
    horizon, num_envs = rewards.shape
    entropy = _reference_entropy(entropies, target_entropy)
    done_mask = done.to(rewards.dtype)
    done_mask[-1] = 1.0
    Ai = torch.zeros(num_envs, dtype=rewards.dtype)
    Bi = torch.zeros(num_envs, dtype=rewards.dtype)
    lam = torch.ones(num_envs, dtype=rewards.dtype)
    targets = torch.zeros_like(rewards)
    for i in reversed(range(horizon)):
        lam = lam * lam_param * (1.0 - done_mask[i]) + done_mask[i]
        rew = rewards[i] + alpha * entropy[i]
        adjusted_rew = (1.0 - lam) / (1.0 - lam_param) * rew
        Ai = (1.0 - done_mask[i]) * (lam_param * gamma * Ai + gamma * next_values[i] + adjusted_rew)
        Bi = gamma * (next_values[i] * done_mask[i] + Bi * (1.0 - done_mask[i])) + rew
        targets[i] = (1.0 - lam_param) * Ai + lam * Bi
    return targets


def _rollout(seed=0, horizon=6, num_envs=5):
    generator = torch.Generator().manual_seed(seed)
    rewards = torch.randn(horizon, num_envs, generator=generator, dtype=torch.float64)
    entropies = torch.randn(horizon, num_envs, generator=generator, dtype=torch.float64) - 2.0
    next_values = 3.0 * torch.randn(horizon, num_envs, generator=generator, dtype=torch.float64)
    truncated = torch.rand(horizon, num_envs, generator=generator) < 0.2
    return rewards, entropies, next_values, truncated


def test_normalize_entropy_maps_target_to_zero():
    target_entropy = -4.0
    normalized = SAPO.normalize_entropy(torch.tensor([-4.0, 0.0, 4.0]), target_entropy)
    torch.testing.assert_close(normalized, torch.tensor([0.0, 0.5, 1.0]))
    with pytest.raises(ValueError, match="non-zero"):
        SAPO.normalize_entropy(torch.zeros(1), 0.0)


def test_policy_loss_matches_reference_with_entropy_in_return():
    rewards, entropies, next_values, truncated = _rollout()
    rewards.requires_grad_(True)
    entropies.requires_grad_(True)
    terminated = torch.zeros_like(truncated)
    kwargs = dict(gamma=0.99, alpha=0.3, target_entropy=-4.0)

    result = SAPO.compute_policy_loss(
        rewards=rewards, entropies=entropies, next_values=next_values, terminated=terminated,
        truncated=truncated, **kwargs,
    )
    reference = _reference_policy_loss(rewards, entropies, next_values, truncated, **kwargs)
    torch.testing.assert_close(result["loss"], reference)

    result["loss"].backward()
    entropy_grad = entropies.grad.clone()
    entropies.grad = None
    reference.backward()
    torch.testing.assert_close(entropy_grad, entropies.grad)
    assert result["entropy"] == pytest.approx(entropies.detach().mean().item())


def test_policy_loss_with_zero_alpha_is_shac():
    rewards, entropies, next_values, truncated = _rollout(seed=1)
    terminated = torch.zeros_like(truncated)
    terminated[2, 1] = True
    sapo = SAPO.compute_policy_loss(
        rewards=rewards, entropies=entropies, next_values=next_values, terminated=terminated,
        truncated=truncated, gamma=0.99, alpha=0.0, target_entropy=-4.0,
    )
    shac = SHAC.compute_policy_loss(
        rewards=rewards, next_values=next_values, terminated=terminated, truncated=truncated, gamma=0.99,
    )
    torch.testing.assert_close(sapo["loss"], shac["loss"])


@pytest.mark.parametrize("lambda_", [0.5, 0.95])
def test_td_lambda_targets_match_reference_with_entropy(lambda_):
    rewards, entropies, next_values, truncated = _rollout(seed=2)
    terminated = torch.zeros_like(truncated)
    kwargs = dict(gamma=0.99, alpha=0.3, target_entropy=-4.0)

    targets = SAPO.compute_td_lambda_targets(
        rewards=rewards, entropies=entropies, next_values=next_values, terminated=terminated,
        truncated=truncated, lambda_=lambda_, **kwargs,
    )
    reference = _reference_td_lambda(rewards, entropies, next_values, truncated, lam_param=lambda_, **kwargs)
    torch.testing.assert_close(targets, reference)


def test_td_lambda_targets_detach_inputs():
    rewards, entropies, next_values, _ = _rollout(seed=3)
    entropies.requires_grad_(True)
    targets = SAPO.compute_td_lambda_targets(
        rewards=rewards, entropies=entropies, next_values=next_values,
        terminated=torch.zeros(rewards.shape, dtype=torch.bool), gamma=0.99, lambda_=0.95,
        alpha=0.3, target_entropy=-4.0,
    )
    assert not targets.requires_grad


class EnsembleCritic(nn.Module):
    def __init__(self, *values):
        super().__init__()
        self.values = nn.ParameterList([nn.Parameter(v) for v in values])

    def forward(self, observations):
        return tuple(ValueStep(value=v) for v in self.values)


def test_next_values_are_min_and_mean_with_graph():
    critic = EnsembleCritic(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))

    min_values, mean_values = SAPO.compute_next_values(critic, torch.zeros(2, 3))
    torch.testing.assert_close(min_values.detach(), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(mean_values.detach(), torch.tensor([2.0, 3.5]))

    mean_values.sum().backward()
    torch.testing.assert_close(critic.values[0].grad, torch.tensor([0.5, 0.5]))


def test_value_loss_matches_reference_ensemble_mse():
    critic = EnsembleCritic(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]))
    targets = torch.tensor([0.0, 4.0])

    result = SAPO.compute_value_loss(critic, observations=torch.zeros(2, 3), targets=targets)
    reference = torch.stack([F.mse_loss(v, targets) for v in critic.values]).mean()
    torch.testing.assert_close(result["loss"], reference)

    with pytest.raises(ValueError, match="same shape"):
        SAPO.compute_value_loss(critic, observations=torch.zeros(2, 3), targets=torch.zeros(3))


def test_alpha_loss_matches_reference():
    target_entropy = -4.0
    log_alpha = nn.Parameter(torch.tensor(0.2))
    entropies = torch.tensor([[-3.0, -5.0], [-1.0, -6.0]])

    result = SAPO.compute_alpha_loss(log_alpha, entropies, target_entropy)
    # Reference: alpha * |target| * (per-env entropy averaged over the horizon - target).
    alpha = log_alpha.exp() * abs(target_entropy)
    reference = (alpha * (entropies.mean(0) - target_entropy)).mean()
    torch.testing.assert_close(result["loss"], reference)

    result["loss"].backward()
    assert log_alpha.grad > 0  # entropy above target lowers alpha


def test_soft_return_gradient_reaches_policy_through_squashed_gaussian():
    from RLAlg.nn.layers import GaussianHead

    torch.manual_seed(0)
    policy = GaussianHead(3, 2, max_action=1.0)
    critic = EnsembleCritic(torch.zeros(4), torch.ones(4))
    state = torch.randn(4, 3)
    rewards, entropies, next_values = [], [], []
    for _ in range(5):
        step = policy(state)
        state = state + 0.1 * torch.cat([step.action, step.action[:, :1]], dim=-1)
        rewards.append(-(state ** 2).sum(-1))
        entropies.append(step.entropy)
        next_values.append(SAPO.compute_next_values(critic, state)[1])

    result = SAPO.compute_policy_loss(
        rewards=torch.stack(rewards), entropies=torch.stack(entropies), next_values=torch.stack(next_values),
        terminated=torch.zeros(5, 4, dtype=torch.bool), gamma=0.99, alpha=1.0, target_entropy=-1.0,
    )
    result["loss"].backward()
    assert torch.isfinite(policy.log_std.grad).all() and policy.log_std.grad.abs().sum() > 0
    assert policy.mu_layer.weight.grad.abs().sum() > 0
