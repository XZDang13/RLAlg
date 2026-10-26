import torch
import pytest

from RLAlg.buffer.replay_buffer import ReplayBuffer


def test_sample_batch_raises_on_empty_buffer():
    buffer = ReplayBuffer(num_envs=2, steps=4)
    buffer.create_storage_space("obs", (3,))

    with pytest.raises(ValueError, match="empty buffer"):
        buffer.sample_batch(["obs"], batch_size=1)


def test_sample_batch_gathers_matching_rows_across_keys():
    buffer = ReplayBuffer(num_envs=2, steps=4)
    buffer.create_storage_space("x", ())
    buffer.create_storage_space("y", (2,))
    for t in range(3):
        x = torch.tensor([10.0 * t, 10.0 * t + 1])
        buffer.add_records({"x": x, "y": torch.stack([x, -x], dim=-1)})

    batch = buffer.sample_batch(["x", "y"], batch_size=64)

    assert batch["x"].shape == (64,) and batch["y"].shape == (64, 2)
    assert torch.equal(batch["y"][:, 0], batch["x"]) and torch.equal(batch["y"][:, 1], -batch["x"])
    assert set(batch["x"].tolist()) <= {0.0, 1.0, 10.0, 11.0, 20.0, 21.0}


def _filled(steps, writes, num_envs=1):
    buffer = ReplayBuffer(num_envs=num_envs, steps=steps)
    buffer.create_storage_space("x", ())
    buffer.create_storage_space("h", (1,))
    for t in range(writes):
        x = torch.full((num_envs,), float(t))
        buffer.add_records({"x": x, "h": x.unsqueeze(-1) + 100})
    return buffer


def test_time_order_lists_slots_oldest_first_before_and_after_wrap():
    assert _filled(5, 3).time_order().tolist() == [0, 1, 2]
    wrapped = _filled(5, 7)
    assert wrapped.data["x"][:, 0].tolist() == [5.0, 6.0, 2.0, 3.0, 4.0]
    assert wrapped.time_order().tolist() == [2, 3, 4, 0, 1]
    assert wrapped.data["x"][wrapped.time_order(), 0].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_sample_sequence_batches_follow_time_order_after_wrap():
    buffer = _filled(5, 7)

    batches = list(buffer.sample_sequence_batches(["x"], seq_len=3, batch_size=2, state_keys=["h"], shuffle=False))

    assert len(batches) == 1
    batch = batches[0]
    assert batch["x"][:, 0].tolist() == [2.0, 3.0, 4.0]
    assert batch["x"][:2, 1].tolist() == [5.0, 6.0]
    assert batch["valid_mask"][:, 1].tolist() == [True, True, False]
    assert batch["h_init"][:, 0].tolist() == [102.0, 105.0]


def test_save_and_load_round_trip(tmp_path):
    buffer = _filled(5, 7, num_envs=2)
    path = tmp_path / "buffer.pt"
    buffer.save(str(path))

    restored = ReplayBuffer(num_envs=1, steps=1)
    restored.load(str(path))

    assert (restored.num_envs, restored.steps, restored.step, restored.current_size) == (2, 5, 2, 5)
    assert torch.equal(restored.data["x"], buffer.data["x"])
    assert torch.equal(restored.time_order(), buffer.time_order())


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
