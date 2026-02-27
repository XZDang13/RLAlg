import torch
import pytest

from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae, compute_returns


def test_compute_returns_accepts_bool_terminated():
    rewards = torch.tensor([[1.0], [2.0]])
    terminated = torch.tensor([[True], [False]])
    last_values = torch.tensor([0.0])

    returns = compute_returns(rewards, terminated, last_values, gamma=0.99)

    assert returns.shape == rewards.shape
    assert torch.isfinite(returns).all()


def test_compute_gae_accepts_bool_terminated():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.4]])
    terminated = torch.tensor([[False], [True]])
    last_values = torch.tensor([0.0])

    returns, advantages = compute_gae(
        rewards, values, terminated, last_values, gamma=0.99, lambda_=0.95
    )

    assert returns.shape == rewards.shape
    assert advantages.shape == rewards.shape
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()


def test_sample_batch_raises_on_empty_buffer():
    buffer = ReplayBuffer(num_envs=2, steps=4)
    buffer.create_storage_space("obs", (3,))

    with pytest.raises(ValueError, match="empty buffer"):
        buffer.sample_batch(["obs"], batch_size=1)


def test_sample_tensor_raises_on_empty_buffer():
    buffer = ReplayBuffer(num_envs=2, steps=4)
    buffer.create_storage_space("obs", (3,))

    with pytest.raises(ValueError, match="empty buffer"):
        buffer.sample_tensor("obs", batch_size=1)


def test_sample_batchs_uses_current_size_only():
    buffer = ReplayBuffer(num_envs=1, steps=4)
    buffer.create_storage_space("x", ())
    buffer.add_records({"x": torch.tensor([1.0])})
    buffer.add_records({"x": torch.tensor([2.0])})

    batches = list(buffer.sample_batchs(["x"], batch_size=8))
    values = torch.cat([batch["x"].reshape(-1) for batch in batches]).tolist()

    assert len(values) == 2
    assert sorted(values) == [1.0, 2.0]


def test_add_storage_validates_leading_dimensions():
    buffer = ReplayBuffer(num_envs=2, steps=4)

    with pytest.raises(ValueError, match="shape"):
        buffer.add_storage("x", torch.zeros(3, 2, 1))
