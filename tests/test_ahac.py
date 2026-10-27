import pytest
import torch
import torch.nn as nn

from RLAlg.alg.ahac import AHAC
from RLAlg.alg.shac import SHAC
from RLAlg.nn.steps import ValueStep


def _random_rollout(horizon=6, num_envs=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    rewards = torch.randn(horizon, num_envs, generator=generator, dtype=torch.float64)
    next_values = torch.randn(horizon, num_envs, generator=generator, dtype=torch.float64)
    terminated = torch.rand(horizon, num_envs, generator=generator) < 0.2
    truncated = (torch.rand(horizon, num_envs, generator=generator) < 0.2) & ~terminated
    contact_norms = 1000.0 * torch.rand(horizon, num_envs, generator=generator, dtype=torch.float64)
    return rewards, next_values, terminated, truncated, contact_norms


def test_policy_loss_without_constraint_matches_shac():
    rewards, next_values, terminated, truncated, _ = _random_rollout()
    kwargs = dict(rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.97, truncated=truncated)

    torch.testing.assert_close(AHAC.compute_policy_loss(**kwargs)["loss"], SHAC.compute_policy_loss(**kwargs)["loss"])


def test_policy_loss_adds_lagrangian_constraint_term():
    rewards, next_values, terminated, truncated, contact_norms = _random_rollout()
    horizon, num_envs = rewards.shape
    dual = torch.linspace(-1.0, 2.0, horizon + 3, dtype=torch.float64)

    result = AHAC.compute_policy_loss(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.97, truncated=truncated,
        contact_norms=contact_norms, dual=dual, contact_threshold=500.0,
    )
    shac_loss = SHAC.compute_policy_loss(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.97, truncated=truncated
    )["loss"]
    constraint = sum(
        dual[h] * (500.0 - contact_norms[h, n]) for h in range(horizon) for n in range(num_envs)
    ) / (horizon * num_envs)

    torch.testing.assert_close(result["loss"], shac_loss - constraint)
    torch.testing.assert_close(result["constraint_violation"], (contact_norms > 500.0).float().mean())


def test_policy_loss_backpropagates_through_contact_norms_but_not_dual():
    rewards, next_values, terminated, _, contact_norms = _random_rollout(horizon=3, num_envs=2)
    contact_norms.requires_grad_(True)
    dual = torch.tensor([1.0, 0.0, -2.0], dtype=torch.float64, requires_grad=True)

    AHAC.compute_policy_loss(
        rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.9,
        contact_norms=contact_norms, dual=dual,
    )["loss"].backward()

    # d(-sum phi (C - c)) / (H N) / dc = phi / (H N)
    expected = (dual.detach() / 6.0).unsqueeze(-1).expand(3, 2)
    torch.testing.assert_close(contact_norms.grad, expected)
    assert dual.grad is None


def test_update_dual_moves_against_slack_and_keeps_tail():
    contact_norms = torch.tensor([[600.0, 800.0], [100.0, 300.0]])
    dual = torch.tensor([0.0, 0.0, 5.0])

    new_dual = AHAC.update_dual(dual, contact_norms=contact_norms, contact_threshold=500.0, lr=0.01)

    torch.testing.assert_close(new_dual, torch.tensor([2.0, -3.0, 5.0]))
    torch.testing.assert_close(dual, torch.tensor([0.0, 0.0, 5.0]))


def test_update_horizon_shrinks_on_violation_and_grows_when_satisfied():
    violated = torch.full((32,), 2.0)
    satisfied = torch.full((32,), -2.0)

    assert AHAC.update_horizon(20.5, dual=violated, lr=0.1) == pytest.approx(20.5 - 0.1 * 2.0 * 20)
    assert AHAC.update_horizon(20.5, dual=satisfied, lr=0.1) == pytest.approx(20.5 + 0.1 * 2.0 * 20)
    assert AHAC.update_horizon(20.0, dual=violated, lr=10.0, min_horizon=4.0) == 4.0
    assert AHAC.update_horizon(30.0, dual=satisfied, lr=10.0) == 32.0


def test_dual_and_horizon_drive_horizon_down_under_stiff_contact():
    dual = torch.zeros(16)
    horizon = 16.0
    for _ in range(50):
        h = int(horizon)
        contact_norms = torch.full((h, 4), 900.0)
        dual = AHAC.update_dual(dual, contact_norms=contact_norms, contact_threshold=500.0, lr=1e-3)
        horizon = AHAC.update_horizon(horizon, dual=dual, lr=1e-2, min_horizon=2.0)

    assert horizon < 16.0


class DoubleCritic(nn.Module):
    def __init__(self, values1, values2):
        super().__init__()
        self.values1 = nn.Parameter(values1)
        self.values2 = nn.Parameter(values2)

    def forward(self, observations):
        return ValueStep(value=self.values1), ValueStep(value=self.values2)


def test_next_values_take_minimum_with_graph():
    critic = DoubleCritic(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))

    next_values = AHAC.compute_next_values(critic, torch.zeros(2, 3))
    next_values.sum().backward()

    torch.testing.assert_close(next_values.detach(), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(critic.values1.grad, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(critic.values2.grad, torch.tensor([0.0, 1.0]))


def test_value_loss_sums_half_mse_over_both_critics():
    critic = DoubleCritic(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]))
    targets = torch.tensor([0.0, 4.0], requires_grad=True)

    result = AHAC.compute_value_loss(critic, observations=torch.zeros(2, 3), targets=targets)
    result["loss"].backward()

    torch.testing.assert_close(result["loss"], torch.tensor(1.25 + 4.0))
    assert targets.grad is None
    torch.testing.assert_close(critic.values2.grad, torch.tensor([0.0, -2.0]))


def test_critic_converged():
    assert not AHAC.critic_converged([1.0, 1.0, 1.0])
    assert AHAC.critic_converged([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert not AHAC.critic_converged([1.0, 1.2, 1.0, 1.2, 1.0, 1.2])
    assert AHAC.critic_converged([1.0, 0.9], window=1, tolerance=0.2)


def test_td_lambda_targets_are_inherited():
    rewards, next_values, terminated, truncated, _ = _random_rollout(seed=2)
    kwargs = dict(rewards=rewards, next_values=next_values, terminated=terminated, gamma=0.99, lambda_=0.95,
                  truncated=truncated)

    torch.testing.assert_close(AHAC.compute_td_lambda_targets(**kwargs), SHAC.compute_td_lambda_targets(**kwargs))


def test_validation():
    rewards = torch.zeros(3, 2)
    terminated = torch.zeros(3, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="together"):
        AHAC.compute_policy_loss(
            rewards=rewards, next_values=rewards, terminated=terminated, gamma=0.99, contact_norms=rewards
        )
    with pytest.raises(ValueError, match="H_max"):
        AHAC.compute_policy_loss(
            rewards=rewards, next_values=rewards, terminated=terminated, gamma=0.99,
            contact_norms=rewards, dual=torch.zeros(2),
        )
    with pytest.raises(ValueError, match="same shape"):
        AHAC.compute_policy_loss(
            rewards=rewards, next_values=rewards, terminated=terminated, gamma=0.99,
            contact_norms=torch.zeros(3, 1), dual=torch.zeros(3),
        )
    with pytest.raises(ValueError, match=r"\[H, N\]"):
        AHAC.update_dual(torch.zeros(3), contact_norms=torch.zeros(3), contact_threshold=500.0, lr=0.1)
    with pytest.raises(ValueError, match="min_horizon"):
        AHAC.update_horizon(5.0, dual=torch.zeros(8), lr=0.1, min_horizon=10.0)
    with pytest.raises(TypeError):
        AHAC.update_dual(torch.zeros(3), torch.zeros(3, 2), 500.0, 0.1)


def test_policy_gradient_through_differentiable_rollout_improves_return():
    gain = nn.Parameter(torch.tensor([-0.3], dtype=torch.float64))
    critic = DoubleCritic(torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
    initial_state = torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64)
    optimizer = torch.optim.SGD([gain], lr=0.05)
    dual = torch.full((4,), 0.1, dtype=torch.float64)

    def rollout_loss():
        state, rewards, contacts = initial_state, [], []
        for _ in range(4):
            action = gain * state
            state = state + action
            rewards.append(-(state.squeeze(-1) ** 2) - 0.1 * action.squeeze(-1) ** 2)
            contacts.append(action.squeeze(-1) ** 2)
        rewards, contacts = torch.stack(rewards), torch.stack(contacts)
        next_values = torch.stack([AHAC.compute_next_values(critic, None)] * 4)
        return AHAC.compute_policy_loss(
            rewards=rewards, next_values=next_values, terminated=torch.zeros_like(rewards, dtype=torch.bool),
            gamma=0.95, contact_norms=contacts, dual=dual,
        )["loss"]

    initial_loss = rollout_loss().item()
    for _ in range(20):
        optimizer.zero_grad()
        rollout_loss().backward()
        optimizer.step()

    assert rollout_loss().item() < initial_loss
