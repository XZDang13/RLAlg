import torch
import pytest

from RLAlg.buffer.her import HindsightExperienceReplay


def _reward_fn(achieved: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    return -((achieved - desired).abs().sum(dim=-1, keepdim=True))


def _done_fn(
    achieved: torch.Tensor,
    desired: torch.Tensor,
    reward: torch.Tensor,
) -> torch.Tensor:
    del achieved, desired
    return reward >= 0


class _GoalEnv:
    def compute_reward(self, achieved, desired, info=None):
        del info
        return -(abs(achieved - desired).sum(axis=-1, keepdims=True))


class _AltGoalEnv:
    def calc_reward(self, achieved, desired, info=None):
        del info
        return -(abs(achieved - desired).sum(axis=-1, keepdims=True))


class _SyncVectorEnvLike:
    def __init__(self, sub_env=None):
        self.envs = [sub_env or _GoalEnv()]


class _NoComputeRewardEnv:
    pass


def test_her_relabels_goal_and_recomputes_reward_done():
    batch = {
        "achieved_goal": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        "achieved_goal_future": torch.tensor([[1.0], [1.0], [3.0], [3.0]]),
        "desired_goal": torch.zeros(4, 1),
        "reward": torch.full((4, 1), -10.0),
        "done": torch.zeros(4, 1, dtype=torch.bool),
    }

    her = HindsightExperienceReplay(
        reward_fn=_reward_fn,
        done_fn=_done_fn,
        relabel_fraction=1.0,
    )
    out = her.process_batch(batch)

    assert torch.equal(out["desired_goal"], batch["achieved_goal_future"])
    assert torch.equal(out["reward"], torch.tensor([[-1.0], [0.0], [-1.0], [0.0]]))
    assert torch.equal(out["done"], torch.tensor([[False], [True], [False], [True]]))
    assert out["her_mask"].all()

    # original batch is not mutated
    assert torch.equal(batch["desired_goal"], torch.zeros(4, 1))


def test_her_partial_relabel_updates_only_masked_rows():
    torch.manual_seed(3)
    batch = {
        "achieved_goal": torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
        "achieved_goal_future": torch.tensor([[5.0], [5.0], [5.0], [5.0], [5.0], [5.0]]),
        "desired_goal": torch.zeros(6, 1),
        "reward": torch.full((6, 1), -99.0),
    }

    her = HindsightExperienceReplay(reward_fn=_reward_fn, relabel_fraction=0.5)
    out = her.process_batch(batch)

    mask = out["her_mask"]

    expected_goal = batch["desired_goal"].clone()
    expected_goal[mask] = batch["achieved_goal_future"][mask]
    assert torch.equal(out["desired_goal"], expected_goal)

    recomputed = _reward_fn(out["achieved_goal"], out["desired_goal"])
    expected_reward = batch["reward"].clone()
    expected_reward[mask] = recomputed[mask]
    assert torch.equal(out["reward"], expected_reward)


def test_her_requires_future_goal_key():
    batch = {
        "achieved_goal": torch.tensor([[0.0]]),
        "desired_goal": torch.tensor([[0.0]]),
    }
    her = HindsightExperienceReplay()

    with pytest.raises(ValueError, match="future"):
        her.process_batch(batch)


def test_her_future_p_is_derived_from_replay_k():
    torch.manual_seed(7)
    n = 2000
    batch = {
        "achieved_goal": torch.zeros(n, 1),
        "achieved_goal_future": torch.ones(n, 1),
        "desired_goal": torch.zeros(n, 1),
        "reward": torch.zeros(n, 1),
    }

    her = HindsightExperienceReplay(replay_strategy="future", replay_k=4, reward_fn=_reward_fn)
    out = her.process_batch(batch)

    assert her.future_p == pytest.approx(0.8)
    ratio = out["her_mask"].float().mean().item()
    assert abs(ratio - her.future_p) < 0.05


def test_her_final_strategy_uses_final_goal_key():
    batch = {
        "achieved_goal": torch.tensor([[0.0], [1.0], [2.0]]),
        "achieved_goal_final": torch.tensor([[2.0], [2.0], [2.0]]),
        "desired_goal": torch.zeros(3, 1),
        "reward": torch.full((3, 1), -5.0),
    }

    her = HindsightExperienceReplay(
        replay_strategy="final",
        relabel_fraction=1.0,
        reward_fn=_reward_fn,
    )
    out = her.process_batch(batch)

    assert torch.equal(out["desired_goal"], batch["achieved_goal_final"])
    assert torch.equal(out["reward"], torch.tensor([[-2.0], [-1.0], [0.0]]))
    assert out["her_mask"].all()


def test_her_none_strategy_skips_relabel_and_future_key_requirement():
    batch = {
        "achieved_goal": torch.tensor([[0.0], [1.0]]),
        "desired_goal": torch.tensor([[1.0], [1.0]]),
        "reward": torch.full((2, 1), -10.0),
    }

    her = HindsightExperienceReplay(replay_strategy="none", reward_fn=_reward_fn)
    out = her.process_batch(batch)

    assert out["her_mask"].sum().item() == 0
    assert torch.equal(out["desired_goal"], batch["desired_goal"])
    assert torch.equal(out["reward"], batch["reward"])


def test_her_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="replay_strategy"):
        HindsightExperienceReplay(replay_strategy="invalid")


def test_her_builds_reward_fn_from_env_compute_reward():
    batch = {
        "achieved_goal": torch.tensor([[0.0], [1.0]]),
        "achieved_goal_future": torch.tensor([[1.0], [1.0]]),
        "desired_goal": torch.zeros(2, 1),
        "reward": torch.full((2, 1), -9.0),
    }

    her = HindsightExperienceReplay(
        env=_GoalEnv(),
        relabel_fraction=1.0,
    )
    out = her.process_batch(batch)

    assert torch.equal(out["desired_goal"], batch["achieved_goal_future"])
    assert torch.equal(out["reward"], torch.tensor([[-1.0], [0.0]]))


def test_her_builds_reward_fn_from_sync_vector_env_subenv():
    batch = {
        "achieved_goal": torch.tensor([[0.0], [2.0]]),
        "achieved_goal_future": torch.tensor([[1.0], [1.0]]),
        "desired_goal": torch.zeros(2, 1),
        "reward": torch.full((2, 1), -9.0),
    }

    her = HindsightExperienceReplay(
        env=_SyncVectorEnvLike(),
        relabel_fraction=1.0,
    )
    out = her.process_batch(batch)

    assert torch.equal(out["desired_goal"], batch["achieved_goal_future"])
    assert torch.equal(out["reward"], torch.tensor([[-1.0], [-1.0]]))


def test_her_env_without_compute_reward_raises():
    with pytest.raises(ValueError, match="resolve reward function"):
        HindsightExperienceReplay(env=_NoComputeRewardEnv())


def test_her_uses_custom_env_reward_fn_name():
    batch = {
        "achieved_goal": torch.tensor([[0.0], [2.0]]),
        "achieved_goal_future": torch.tensor([[1.0], [1.0]]),
        "desired_goal": torch.zeros(2, 1),
        "reward": torch.full((2, 1), -9.0),
    }

    her = HindsightExperienceReplay(
        env=_AltGoalEnv(),
        env_reward_fn_name="calc_reward",
        relabel_fraction=1.0,
    )
    out = her.process_batch(batch)
    assert torch.equal(out["reward"], torch.tensor([[-1.0], [-1.0]]))


def test_her_reward_fn_has_priority_over_env_reward_resolution():
    batch = {
        "achieved_goal": torch.tensor([[0.0], [1.0]]),
        "achieved_goal_future": torch.tensor([[1.0], [1.0]]),
        "desired_goal": torch.zeros(2, 1),
        "reward": torch.full((2, 1), -9.0),
    }

    def custom_reward_fn(achieved: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        del achieved, desired
        return torch.zeros(2, 1)

    her = HindsightExperienceReplay(
        env=_GoalEnv(),
        reward_fn=custom_reward_fn,
        relabel_fraction=1.0,
    )
    out = her.process_batch(batch)
    assert torch.equal(out["reward"], torch.zeros(2, 1))


def test_her_supports_multiple_env_reward_fn_name_candidates():
    batch = {
        "achieved_goal": torch.tensor([[0.0], [2.0]]),
        "achieved_goal_future": torch.tensor([[1.0], [1.0]]),
        "desired_goal": torch.zeros(2, 1),
        "reward": torch.full((2, 1), -9.0),
    }

    her = HindsightExperienceReplay(
        env=_AltGoalEnv(),
        env_reward_fn_name=("compute_reward", "calc_reward"),
        relabel_fraction=1.0,
    )
    out = her.process_batch(batch)
    assert torch.equal(out["reward"], torch.tensor([[-1.0], [-1.0]]))
