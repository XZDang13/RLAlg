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


def test_sample_batch_returns_future_samples_within_same_episode():
    buffer = ReplayBuffer(num_envs=1, steps=6)
    buffer.create_storage_space("obs", ())
    buffer.create_storage_space("done", (), dtype=torch.bool)

    done = [False, False, True, False, False, True]
    for t in range(6):
        buffer.add_records(
            {
                "obs": torch.tensor([float(t)]),
                "done": torch.tensor([done[t]]),
            }
        )

    torch.manual_seed(0)
    batch = buffer.sample_batch(["obs"], batch_size=128, future_steps=2)

    assert "obs_future" in batch
    current = batch["obs"].to(dtype=torch.long)
    future = batch["obs_future"].to(dtype=torch.long)

    for cur, fut in zip(current.tolist(), future.tolist()):
        expected = min(cur + 2, len(done) - 1)
        for step in range(cur, expected):
            if done[step]:
                expected = step
                break
        assert fut == expected


def test_sample_batch_future_steps_requires_episode_end_key():
    buffer = ReplayBuffer(num_envs=1, steps=4)
    buffer.create_storage_space("obs", ())
    buffer.add_records({"obs": torch.tensor([1.0])})

    with pytest.raises(ValueError, match="episode end key"):
        buffer.sample_batch(["obs"], batch_size=1, future_steps=1)


def test_sample_batch_future_steps_uses_terminated_fallback():
    buffer = ReplayBuffer(num_envs=1, steps=4)
    buffer.create_storage_space("obs", ())
    buffer.create_storage_space("terminated", (), dtype=torch.bool)

    terminated = [False, True, False, True]
    for t in range(4):
        buffer.add_records(
            {
                "obs": torch.tensor([float(t)]),
                "terminated": torch.tensor([terminated[t]]),
            }
        )

    batch = buffer.sample_batch(["obs"], batch_size=32, future_steps=2)
    assert "obs_future" in batch


def test_sample_batch_her_strategies_final_episode_and_random():
    buffer = ReplayBuffer(num_envs=1, steps=6)
    buffer.create_storage_space("obs", ())
    buffer.create_storage_space("done", (), dtype=torch.bool)

    done = [False, False, True, False, False, True]
    for t in range(6):
        buffer.add_records(
            {
                "obs": torch.tensor([float(t)]),
                "done": torch.tensor([done[t]]),
            }
        )

    torch.manual_seed(0)
    batch = buffer.sample_batch(
        ["obs"],
        batch_size=256,
        her_strategies=["final", "episode", "random"],
    )

    assert "obs_final" in batch
    assert "obs_episode" in batch
    assert "obs_random" in batch

    current = batch["obs"].to(dtype=torch.long)
    final = batch["obs_final"].to(dtype=torch.long)
    episode = batch["obs_episode"].to(dtype=torch.long)
    random = batch["obs_random"].to(dtype=torch.long)

    for cur, fin, epi in zip(current.tolist(), final.tolist(), episode.tolist()):
        if cur <= 2:
            assert fin == 2
            assert 0 <= epi <= 2
        else:
            assert fin == 5
            assert 3 <= epi <= 5

    assert ((random >= 0) & (random <= 5)).all()


def test_sample_batch_future_strategy_requires_positive_future_steps():
    buffer = ReplayBuffer(num_envs=1, steps=3)
    buffer.create_storage_space("obs", ())
    buffer.create_storage_space("done", (), dtype=torch.bool)
    for t in range(3):
        buffer.add_records({"obs": torch.tensor([float(t)]), "done": torch.tensor([False])})

    with pytest.raises(ValueError, match="future_steps > 0"):
        buffer.sample_batch(["obs"], batch_size=1, her_strategies=["future"])


def test_sample_batch_her_strategies_require_episode_end_key():
    buffer = ReplayBuffer(num_envs=1, steps=3)
    buffer.create_storage_space("obs", ())
    for t in range(3):
        buffer.add_records({"obs": torch.tensor([float(t)])})

    with pytest.raises(ValueError, match="episode end key"):
        buffer.sample_batch(["obs"], batch_size=1, her_strategies=["final"])


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


def test_add_records_validates_step_shape():
    buffer = ReplayBuffer(num_envs=2, steps=4)
    buffer.create_storage_space("x", ())

    with pytest.raises(ValueError, match="shape"):
        buffer.add_records({"x": torch.tensor([1.0])})


def test_sample_sequence_batches_returns_contiguous_chunks_and_valid_mask():
    buffer = ReplayBuffer(num_envs=2, steps=5)
    buffer.create_storage_space("x", ())

    for t in range(5):
        buffer.add_records({"x": torch.tensor([10.0 * t + 0.0, 10.0 * t + 1.0])})

    batches = list(
        buffer.sample_sequence_batches(
            key_names=["x"],
            seq_len=3,
            batch_size=2,
            shuffle=False,
        )
    )

    assert len(batches) == 2

    first = batches[0]
    assert first["x"].shape == (3, 2)
    assert torch.equal(first["x"][:, 0], torch.tensor([0.0, 10.0, 20.0]))
    assert torch.equal(first["x"][:, 1], torch.tensor([30.0, 40.0, 0.0]))
    assert torch.equal(
        first["valid_mask"],
        torch.tensor(
            [
                [True, True],
                [True, True],
                [True, False],
            ]
        ),
    )

    second = batches[1]
    assert second["x"].shape == (3, 2)
    assert torch.equal(second["x"][:, 0], torch.tensor([1.0, 11.0, 21.0]))
    assert torch.equal(second["x"][:, 1], torch.tensor([31.0, 41.0, 0.0]))


def test_sample_sequence_batches_extracts_initial_states():
    buffer = ReplayBuffer(num_envs=1, steps=5)
    buffer.create_storage_space("x", ())
    buffer.create_storage_space("h", (2,))

    for t in range(5):
        buffer.add_records(
            {
                "x": torch.tensor([float(t)]),
                "h": torch.tensor([[float(t), float(100 + t)]]),
            }
        )

    batches = list(
        buffer.sample_sequence_batches(
            key_names=["x"],
            state_keys=["h"],
            seq_len=2,
            batch_size=3,
            shuffle=False,
        )
    )

    assert len(batches) == 1
    batch = batches[0]
    assert batch["h_init"].shape == (3, 2)
    assert torch.equal(
        batch["h_init"],
        torch.tensor(
            [
                [0.0, 100.0],
                [2.0, 102.0],
                [4.0, 104.0],
            ]
        ),
    )
