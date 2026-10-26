import numpy as np
import pytest
import torch

from RLAlg.buffer import HERReplayBuffer, env_reward_fn


def _reward_fn(achieved: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    return -(achieved - desired).abs().sum(dim=-1)


def _make_buffer(steps=20, num_envs=1, **options):
    options.setdefault("reward_fn", _reward_fn)
    buffer = HERReplayBuffer(num_envs, steps, **options)
    buffer.create_storage_space("desired_goal", (1,))
    buffer.create_storage_space("next_achieved_goal", (1,))
    buffer.create_storage_space("rewards", ())
    buffer.create_storage_space("step_id", (), dtype=torch.long)
    buffer.create_storage_space("episode_end", (), dtype=torch.bool)
    buffer.create_storage_space("terminated", (), dtype=torch.bool)
    return buffer


def _add_episodes(buffer, lengths, start_id=0):
    """Add episodes for every env; next_achieved_goal and step_id equal the absolute step."""
    step_id = start_id
    for length in lengths:
        for t in range(length):
            ids = torch.full((buffer.num_envs,), step_id)
            buffer.add_records({
                "desired_goal": torch.full((buffer.num_envs, 1), -1.0),
                "next_achieved_goal": ids.float().unsqueeze(-1),
                "rewards": torch.full((buffer.num_envs,), -1.0),
                "step_id": ids,
                "episode_end": torch.full((buffer.num_envs,), t == length - 1),
                "terminated": torch.zeros(buffer.num_envs, dtype=torch.bool),
            })
            step_id += 1
    return step_id


def _episode_of(step_id, lengths):
    start = 0
    for length in lengths:
        if step_id < start + length:
            return start, start + length - 1
        start += length
    return start, None


KEYS = ["desired_goal", "rewards", "step_id", "terminated"]


def _sample(buffer, batch_size=4000, seed=0):
    torch.manual_seed(seed)
    batch = buffer.sample_batch(KEYS, batch_size)
    relabeled = batch["her_mask"]
    return batch["step_id"][relabeled], batch["desired_goal"][relabeled, 0].long(), batch, relabeled


def test_future_goals_are_uniform_over_the_rest_of_the_episode():
    buffer = _make_buffer()
    lengths = [4, 6]
    _add_episodes(buffer, lengths)

    steps, goals, _, _ = _sample(buffer)

    seen = {}
    for step, goal in zip(steps.tolist(), goals.tolist()):
        seen.setdefault(step, set()).add(goal)
    for step, goal_set in seen.items():
        _, end = _episode_of(step, lengths)
        assert goal_set == set(range(step, end + 1))


def test_final_and_episode_strategies_stay_inside_the_episode():
    lengths = [4, 6]
    for strategy in ("final", "episode"):
        buffer = _make_buffer(strategy=strategy)
        _add_episodes(buffer, lengths)
        steps, goals, _, _ = _sample(buffer)
        for step, goal in zip(steps.tolist(), goals.tolist()):
            start, end = _episode_of(step, lengths)
            if strategy == "final":
                assert goal == end
            else:
                assert start <= goal <= end


def test_relabeling_never_crosses_episodes_after_the_buffer_wraps():
    # 13 steps of 4-step episodes in a 10-slot buffer: steps 0-2 are overwritten,
    # episode 3 (steps 12-) is unfinished, and episode 2 straddles the write pointer.
    lengths = [4, 4, 4, 1]
    for strategy in ("future", "final", "episode"):
        buffer = _make_buffer(steps=10, strategy=strategy)
        _add_episodes(buffer, lengths)
        assert buffer.data["step_id"][:, 0].tolist() == [10, 11, 12, 3, 4, 5, 6, 7, 8, 9]

        steps, goals, _, _ = _sample(buffer)

        assert set(steps.tolist()) == set(range(3, 13))
        for step, goal in zip(steps.tolist(), goals.tolist()):
            start, end = _episode_of(step, lengths)
            end = 12 if end is None else end
            assert max(start, 3) <= goal <= end
            if strategy == "future":
                assert goal >= step


def test_rewards_are_recomputed_only_for_relabeled_rows():
    buffer = _make_buffer()
    _add_episodes(buffer, [5, 5])

    _, _, batch, relabeled = _sample(buffer)

    assert 0.7 < relabeled.float().mean().item() < 0.9  # replay_k=4 relabels 80%.
    achieved = batch["step_id"].float()
    expected = -(achieved - batch["desired_goal"][:, 0]).abs()
    torch.testing.assert_close(batch["rewards"][relabeled], expected[relabeled])
    assert torch.all(batch["rewards"][~relabeled] == -1.0)
    assert torch.all(batch["desired_goal"][~relabeled] == -1.0)
    # The future goal t' = t is the goal just achieved, so it is a success.
    immediate = relabeled & (batch["desired_goal"][:, 0].long() == batch["step_id"])
    assert immediate.any() and torch.all(batch["rewards"][immediate] == 0.0)


def test_replay_k_zero_disables_relabeling():
    buffer = _make_buffer(replay_k=0)
    _add_episodes(buffer, [5])

    _, _, batch, relabeled = _sample(buffer, batch_size=64)

    assert not relabeled.any()
    assert torch.all(batch["desired_goal"] == -1.0)


def test_done_fn_recomputes_termination_for_relabeled_rows():
    buffer = _make_buffer(done_fn=lambda achieved, goal, reward: reward == 0, done_key="terminated")
    _add_episodes(buffer, [5, 5])

    _, _, batch, relabeled = _sample(buffer)

    assert torch.equal(batch["terminated"][relabeled], batch["rewards"][relabeled] == 0)
    assert not batch["terminated"][~relabeled].any()


def test_episodes_are_tracked_per_env():
    buffer = _make_buffer(num_envs=2)
    for t in range(6):
        ids = torch.tensor([t, 100 + t])
        buffer.add_records({
            "desired_goal": torch.full((2, 1), -1.0),
            "next_achieved_goal": ids.float().unsqueeze(-1),
            "rewards": torch.full((2,), -1.0),
            "step_id": ids,
            # env 0: episodes [0-2], [3-5]; env 1: one episode [100-105].
            "episode_end": torch.tensor([t in (2, 5), t == 5]),
            "terminated": torch.zeros(2, dtype=torch.bool),
        })

    steps, goals, _, _ = _sample(buffer, batch_size=4000)

    for step, goal in zip(steps.tolist(), goals.tolist()):
        if step >= 100:
            assert step <= goal <= 105
        else:
            assert step <= goal <= (2 if step <= 2 else 5)


def test_sample_validates_keys_and_records():
    buffer = _make_buffer()
    with pytest.raises(ValueError, match="empty buffer"):
        buffer.sample_batch(KEYS, 1)
    with pytest.raises(ValueError, match="episode end key"):
        buffer.add_records({"rewards": torch.zeros(1)})
    _add_episodes(buffer, [3])
    with pytest.raises(ValueError, match="must include 'rewards'"):
        buffer.sample_batch(["desired_goal"], 1)
    with pytest.raises(ValueError, match="strategy"):
        HERReplayBuffer(1, 4, reward_fn=_reward_fn, strategy="random")
    with pytest.raises(ValueError, match="together"):
        HERReplayBuffer(1, 4, reward_fn=_reward_fn, done_key="terminated")


def test_reset_and_state_dict_keep_episode_bookkeeping(tmp_path):
    buffer = _make_buffer(steps=10)
    _add_episodes(buffer, [4, 4, 4, 1])
    path = tmp_path / "her.pt"
    buffer.save(str(path))

    restored = _make_buffer(steps=10)
    restored.load(str(path))
    _, goals_a, _, _ = _sample(buffer, seed=3)
    _, goals_b, _, _ = _sample(restored, seed=3)
    assert torch.equal(goals_a, goals_b)

    restored.reset()
    _add_episodes(restored, [2])
    steps, goals, _, _ = _sample(restored)
    assert set(goals.tolist()) <= {0, 1} and torch.all(goals >= steps)


class _GoalEnv:
    def compute_reward(self, achieved, desired, info):
        assert info is None
        return -(np.abs(achieved - desired).sum(axis=-1) > 0.5).astype(np.float32)


class _Wrapper:
    def __init__(self, env):
        self.unwrapped = env


class _VectorEnv:
    def __init__(self, env):
        self.envs = [_Wrapper(env)]
        self.unwrapped = self


def test_env_reward_fn_resolves_vector_and_wrapped_envs():
    achieved = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    desired = torch.tensor([[0.1, 0.0], [0.0, 0.0]])
    for env in (_GoalEnv(), _Wrapper(_GoalEnv()), _VectorEnv(_GoalEnv())):
        reward = env_reward_fn(env)(achieved, desired)
        assert reward.dtype == torch.float32
        assert reward.tolist() == [0.0, -1.0]
    with pytest.raises(ValueError, match="compute_reward"):
        env_reward_fn(object())
