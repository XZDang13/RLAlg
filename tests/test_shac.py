import pytest
import torch
import torch.nn as nn

from RLAlg.alg.shac import SHAC
from RLAlg.nn.steps import ValueStep


def _reference_policy_loss(rewards, next_values, terminated, truncated, gamma):
    # Port of the actor loss loop in the reference SHAC/AHAC implementation.
    horizon, num_envs = rewards.shape
    next_values = torch.where(terminated, torch.zeros_like(next_values), next_values)
    rew_acc = torch.zeros(horizon + 1, num_envs, dtype=rewards.dtype)
    discount = torch.ones(num_envs, dtype=rewards.dtype)
    actor_loss = torch.zeros((), dtype=rewards.dtype)
    for i in range(horizon):
        rew_acc[i + 1] = rew_acc[i] + discount * rewards[i]
        done_ids = (terminated[i] | truncated[i]).nonzero(as_tuple=False).squeeze(-1)
        if i < horizon - 1:
            actor_loss += (-rew_acc[i + 1, done_ids] - gamma * discount[done_ids] * next_values[i, done_ids]).sum()
        else:
            actor_loss += (-rew_acc[i + 1] - gamma * discount * next_values[i]).sum()
        discount = discount * gamma
        discount[done_ids] = 1.0
        rew_acc[i + 1, done_ids] = 0.0
    return actor_loss / (horizon * num_envs)


def _reference_td_lambda(rewards, next_values, terminated, truncated, gamma, lam_param):
    # Port of compute_target_values("td-lambda") in the reference implementation.
    horizon, num_envs = rewards.shape
    next_values = torch.where(terminated, torch.zeros_like(next_values), next_values)
    done_mask = (terminated | truncated).to(rewards.dtype)
    done_mask[-1] = 1.0
    Ai = torch.zeros(num_envs, dtype=rewards.dtype)
    Bi = torch.zeros(num_envs, dtype=rewards.dtype)
    lam = torch.ones(num_envs, dtype=rewards.dtype)
    targets = torch.zeros_like(rewards)
    for i in reversed(range(horizon)):
        lam = lam * lam_param * (1.0 - done_mask[i]) + done_mask[i]
        Ai = (1.0 - done_mask[i]) * (
            lam_param * gamma * Ai + gamma * next_values[i] + (1.0 - lam) / (1.0 - lam_param) * rewards[i]
        )
        Bi = gamma * (next_values[i] * done_mask[i] + Bi * (1.0 - done_mask[i])) + rewards[i]
        targets[i] = (1.0 - lam_param) * Ai + lam * Bi
    return targets


def _random_rollout(horizon=6, num_envs=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    rewards = torch.randn(horizon, num_envs, generator=generator, dtype=torch.float64)
    next_values = torch.randn(horizon, num_envs, generator=generator, dtype=torch.float64)
    terminated = torch.rand(horizon, num_envs, generator=generator) < 0.2
    truncated = (torch.rand(horizon, num_envs, generator=generator) < 0.2) & ~terminated
    return rewards, next_values, terminated, truncated


def test_policy_loss_matches_reference_with_episode_boundaries():
    rewards, next_values, terminated, truncated = _random_rollout()

    result = SHAC.compute_policy_loss(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.97, truncated=truncated
    )
    expected = _reference_policy_loss(rewards, next_values, terminated, truncated, gamma=0.97)

    torch.testing.assert_close(result["loss"], expected)


def test_policy_loss_without_boundaries_is_short_horizon_return():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    next_values = torch.tensor([[10.0], [20.0], [30.0]])
    terminated = torch.zeros(3, 1, dtype=torch.bool)

    result = SHAC.compute_policy_loss(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.5
    )

    # (1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 30) / (H * N)
    expected = -(1.0 + 1.0 + 0.75 + 3.75) / 3.0
    torch.testing.assert_close(result["loss"], torch.tensor(expected))


def test_terminated_steps_ignore_non_finite_next_values():
    rewards = torch.ones(2, 2)
    next_values = torch.tensor([[float("nan"), 1.0], [1.0, float("inf")]])
    terminated = torch.tensor([[True, False], [False, True]])

    loss = SHAC.compute_policy_loss(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.9
    )["loss"]
    targets = SHAC.compute_td_lambda_targets(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.9, lambda_=0.95
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(targets).all()


@pytest.mark.parametrize("lambda_", [0.0, 0.5, 0.95])
def test_td_lambda_matches_reference(lambda_):
    rewards, next_values, terminated, truncated = _random_rollout(seed=1)

    targets = SHAC.compute_td_lambda_targets(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.99, lambda_=lambda_, truncated=truncated
    )
    if lambda_ == 0.0:
        bootstrap = torch.where(terminated, torch.zeros_like(next_values), next_values)
        expected = rewards + 0.99 * bootstrap
    else:
        expected = _reference_td_lambda(rewards, next_values, terminated, truncated, 0.99, lambda_)

    torch.testing.assert_close(targets, expected)


def test_td_lambda_one_without_boundaries_is_monte_carlo_with_bootstrap():
    rewards = torch.tensor([[1.0], [2.0]])
    next_values = torch.tensor([[100.0], [4.0]])
    terminated = torch.zeros(2, 1, dtype=torch.bool)

    targets = SHAC.compute_td_lambda_targets(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.5, lambda_=1.0
    )

    torch.testing.assert_close(targets, torch.tensor([[1.0 + 0.5 * (2.0 + 0.5 * 4.0)], [2.0 + 0.5 * 4.0]]))


def test_td_lambda_targets_are_detached():
    rewards, next_values, terminated, truncated = _random_rollout()
    rewards.requires_grad_(True)

    targets = SHAC.compute_td_lambda_targets(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.99, lambda_=0.95
    )

    assert not targets.requires_grad


class LinearPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor([-0.3], dtype=torch.float64))

    def forward(self, state):
        return self.gain * state


class QuadraticCritic(nn.Module):
    def forward(self, state):
        return ValueStep(value=-(state.squeeze(-1) ** 2))


def _toy_rollout_loss(policy, critic, initial_state, horizon, gamma):
    # Differentiable point mass: s' = s + a, r = -s'^2 - 0.1 a^2.
    state = initial_state
    rewards, next_values = [], []
    for _ in range(horizon):
        action = policy(state)
        state = state + action
        rewards.append(-(state.squeeze(-1) ** 2) - 0.1 * action.squeeze(-1) ** 2)
        next_values.append(critic(state).value)
    rewards = torch.stack(rewards)
    next_values = torch.stack(next_values)
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    return SHAC.compute_policy_loss(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=gamma
    )["loss"]


def test_policy_gradient_through_differentiable_rollout_matches_finite_difference():
    policy = LinearPolicy()
    critic = QuadraticCritic()
    initial_state = torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64)

    loss = _toy_rollout_loss(policy, critic, initial_state, horizon=4, gamma=0.95)
    loss.backward()
    analytic = policy.gain.grad.clone()

    eps = 1e-6
    with torch.no_grad():
        policy.gain += eps
        loss_plus = _toy_rollout_loss(policy, critic, initial_state, horizon=4, gamma=0.95)
        policy.gain -= 2 * eps
        loss_minus = _toy_rollout_loss(policy, critic, initial_state, horizon=4, gamma=0.95)
    numeric = (loss_plus - loss_minus) / (2 * eps)

    assert analytic.abs().item() > 0
    torch.testing.assert_close(analytic.squeeze(), numeric, rtol=1e-5, atol=1e-8)


def test_policy_gradient_step_improves_toy_return():
    policy = LinearPolicy()
    critic = QuadraticCritic()
    initial_state = torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)

    initial_loss = _toy_rollout_loss(policy, critic, initial_state, horizon=4, gamma=0.95)
    for _ in range(20):
        optimizer.zero_grad()
        _toy_rollout_loss(policy, critic, initial_state, horizon=4, gamma=0.95).backward()
        optimizer.step()
    final_loss = _toy_rollout_loss(policy, critic, initial_state, horizon=4, gamma=0.95)

    assert final_loss.item() < initial_loss.item()


class TableCritic(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(values)

    def forward(self, observations):
        return ValueStep(value=self.values)


def test_value_loss_is_half_mse_and_detaches_targets():
    critic = TableCritic(torch.tensor([1.0, 2.0]))
    targets = torch.tensor([0.0, 4.0], requires_grad=True)

    result = SHAC.compute_value_loss(critic, observations=torch.zeros(2, 3), targets=targets)
    result["loss"].backward()

    # 0.5 * mean((V - target)^2), the shared regression coefficient (the reference SHAC omits the 0.5).
    torch.testing.assert_close(result["loss"], torch.tensor(1.25))
    assert targets.grad is None
    torch.testing.assert_close(critic.values.grad, torch.tensor([0.5, -1.0]))


def test_shape_validation():
    rewards = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="same shape"):
        SHAC.compute_policy_loss(
            rewards=rewards, next_values=torch.zeros(3, 1), terminated=torch.zeros(3, 2, dtype=torch.bool), gamma=0.99
        )
    with pytest.raises(ValueError, match="same shape"):
        SHAC.compute_td_lambda_targets(
            rewards=rewards,
            next_values=rewards,
            terminated=torch.zeros(3, 2, 1, dtype=torch.bool),
            gamma=0.99,
            lambda_=0.95,
        )
    with pytest.raises(ValueError, match=r"\[H, N\]"):
        SHAC.compute_policy_loss(
            rewards=torch.zeros(3), next_values=torch.zeros(3), terminated=torch.zeros(3, dtype=torch.bool), gamma=0.99
        )
    with pytest.raises(ValueError, match="lambda_"):
        SHAC.compute_td_lambda_targets(
            rewards=rewards, next_values=rewards, terminated=torch.zeros(3, 2, dtype=torch.bool), gamma=0.99, lambda_=1.5
        )
    with pytest.raises(ValueError, match="same shape"):
        SHAC.compute_value_loss(TableCritic(torch.zeros(2)), observations=torch.zeros(2, 3), targets=torch.zeros(2, 1))


def test_data_arguments_are_keyword_only():
    rewards = torch.zeros(3, 2)
    terminated = torch.zeros(3, 2, dtype=torch.bool)
    with pytest.raises(TypeError):
        SHAC.compute_policy_loss(rewards, rewards, terminated, 0.99)
    with pytest.raises(TypeError):
        SHAC.compute_td_lambda_targets(rewards, rewards, terminated, 0.99, 0.95)
    with pytest.raises(TypeError):
        SHAC.compute_value_loss(TableCritic(torch.zeros(2)), torch.zeros(2, 3), torch.zeros(2))


def test_update_target_param():
    model = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.0)
        for param in target.parameters():
            param.fill_(0.0)

    SHAC.update_target_param(model, target, tau=0.25)

    for param in target.parameters():
        torch.testing.assert_close(param, torch.full_like(param, 0.25))
