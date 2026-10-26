import pytest
import torch

from RLAlg.buffer import compute_advantage_mc, compute_gae, compute_returns, normalize_advantages


def test_compute_returns_accepts_bool_terminated_and_stops_at_boundaries():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    terminated = torch.tensor([[True], [False], [False]])

    returns = compute_returns(rewards=rewards, terminated=terminated, last_values=torch.tensor([10.0]), gamma=0.5)

    torch.testing.assert_close(returns, torch.tensor([[1.0], [2.0 + 0.5 * 8.0], [3.0 + 0.5 * 10.0]]))


def test_compute_returns_treats_timeouts_as_boundaries():
    rewards = torch.tensor([[1.0], [2.0]])
    terminated = torch.zeros(2, 1, dtype=torch.bool)
    truncated = torch.tensor([[True], [False]])

    returns = compute_returns(
        rewards=rewards, terminated=terminated, last_values=torch.tensor([10.0]), gamma=0.5, truncated=truncated,
    )

    torch.testing.assert_close(returns, torch.tensor([[1.0], [7.0]]))


def test_compute_gae_accepts_bool_terminated():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.4]])
    terminated = torch.tensor([[False], [True]])

    returns, advantages = compute_gae(
        rewards=rewards, values=values, terminated=terminated, last_values=torch.tensor([0.0]),
        gamma=0.99, lambda_=0.95,
    )

    assert returns.shape == advantages.shape == rewards.shape
    assert torch.isfinite(returns).all() and torch.isfinite(advantages).all()


def test_compute_gae_bootstraps_timeouts_and_stops_recursion():
    kwargs = dict(
        rewards=torch.tensor([[1.0, 1.0]]), values=torch.tensor([[10.0, 10.0]]),
        terminated=torch.tensor([[False, False]]), last_values=torch.tensor([100.0, 100.0]),
        gamma=0.9, lambda_=0.95, truncated=torch.tensor([[True, True]]),
    )

    bootstrapped, _ = compute_gae(**kwargs, bootstrap_timeouts=True)
    unbootstrapped, _ = compute_gae(**kwargs, bootstrap_timeouts=False)

    torch.testing.assert_close(bootstrapped, torch.tensor([[10.0, 10.0]]))
    torch.testing.assert_close(unbootstrapped, torch.tensor([[1.0, 1.0]]))


def test_normalization_is_optional_and_returns_are_unnormalized():
    torch.manual_seed(0)
    kwargs = dict(
        rewards=torch.randn(5, 3), values=torch.randn(5, 3), terminated=torch.rand(5, 3) < 0.2,
        last_values=torch.randn(3), gamma=0.9,
    )
    for fn, extra in ((compute_gae, {"lambda_": 0.8}), (compute_advantage_mc, {})):
        returns, raw = fn(**kwargs, **extra, normalize=False)
        same_returns, normalized = fn(**kwargs, **extra)

        torch.testing.assert_close(raw, returns - kwargs["values"])
        torch.testing.assert_close(same_returns, returns)
        torch.testing.assert_close(normalized, normalize_advantages(raw))
        assert abs(normalized.mean().item()) < 1e-6 and abs(normalized.std().item() - 1.0) < 1e-5


def test_gae_with_lambda_one_matches_monte_carlo():
    torch.manual_seed(1)
    kwargs = dict(
        rewards=torch.randn(6, 2), values=torch.randn(6, 2), terminated=torch.rand(6, 2) < 0.2,
        truncated=torch.rand(6, 2) < 0.2, last_values=torch.randn(2), gamma=0.9, normalize=False,
    )
    for bootstrap in (False, True):
        gae = compute_gae(**kwargs, lambda_=1.0, bootstrap_timeouts=bootstrap)
        mc = compute_advantage_mc(**kwargs, bootstrap_timeouts=bootstrap)
        torch.testing.assert_close(gae, mc)


def test_estimators_validate_shapes_and_keywords():
    with pytest.raises(ValueError, match="truncated must have shape"):
        compute_gae(
            rewards=torch.ones(2, 1), values=torch.ones(2, 1), terminated=torch.zeros(2, 1, dtype=torch.bool),
            last_values=torch.zeros(1), truncated=torch.zeros(1, 1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="last_values"):
        compute_returns(rewards=torch.ones(2, 3), terminated=torch.zeros(2, 3), last_values=torch.zeros(2))
    with pytest.raises(TypeError):
        compute_returns(torch.ones(2, 1), torch.zeros(2, 1), torch.zeros(1))
    with pytest.raises(ValueError, match="at least two"):
        normalize_advantages(torch.ones(1))
