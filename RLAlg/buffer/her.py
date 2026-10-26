from typing import Any, Callable, Optional

import torch

from .replay_buffer import ReplayBuffer

RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
DoneFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def env_reward_fn(env: Any, name: str = "compute_reward") -> RewardFn:
    """Wrap a goal-conditioned env's ``compute_reward(achieved, desired, info)``.

    For a vector env the first sub-env's function is used, so it must be
    vectorised over a batch of goals (as Gymnasium-Robotics envs are).
    """
    candidates = []
    envs = getattr(env, "envs", None)
    if isinstance(envs, (list, tuple)) and envs:
        candidates += [getattr(envs[0], "unwrapped", envs[0]), envs[0]]
    candidates += [getattr(env, "unwrapped", env), env]
    for candidate in candidates:
        compute_reward = getattr(candidate, name, None)
        if callable(compute_reward):
            break
    else:
        raise ValueError(f"env has no callable '{name}'; pass reward_fn explicitly.")

    def reward_fn(achieved: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        reward = compute_reward(achieved.detach().cpu().numpy(), desired.detach().cpu().numpy(), None)
        return torch.as_tensor(reward, dtype=torch.float32, device=achieved.device)

    return reward_fn


class HERReplayBuffer(ReplayBuffer):
    """Replay buffer with Hindsight Experience Replay goal relabeling.

    Episode boundaries are recorded as transitions are added, so relabel goals
    always come from the sampled transition's own episode, also after the
    circular buffer wraps. Steps of an unfinished episode use the latest stored
    step of that episode as its provisional end.

    A relabeled transition gets the goal achieved after step ``t'`` (the
    ``next_achieved_goal_key`` value at ``t'``), where ``t'`` is drawn from the
    same episode:

    - ``future``: uniform over ``[t, end]``, so ``t' = t`` gives an immediate success;
    - ``final``: the episode's last step;
    - ``episode``: uniform over the episode's steps still in the buffer.

    A fraction ``1 - 1 / (1 + replay_k)`` of sampled transitions is relabeled
    and their reward recomputed with ``reward_fn(next_achieved_goal_t, goal)``.
    If ``done_fn(next_achieved_goal_t, goal, reward)`` is given, ``done_key``
    is recomputed as well (for tasks that terminate on success).

    Every record passed to ``add_records`` must contain ``episode_end_key``,
    which marks the last step of an episode, including timeouts. It is not
    the Bellman termination mask; store true terminations under another key.
    """

    _STRATEGIES = ("future", "final", "episode")

    def __init__(
        self,
        num_envs: int,
        steps: int,
        device: torch.device = torch.device("cpu"),
        *,
        reward_fn: RewardFn,
        strategy: str = "future",
        replay_k: int = 4,
        desired_goal_key: str = "desired_goal",
        next_achieved_goal_key: str = "next_achieved_goal",
        reward_key: str = "rewards",
        episode_end_key: str = "episode_end",
        done_fn: Optional[DoneFn] = None,
        done_key: Optional[str] = None,
    ) -> None:
        super().__init__(num_envs, steps, device)
        if strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {self._STRATEGIES}, got {strategy!r}.")
        if replay_k < 0:
            raise ValueError(f"replay_k must be non-negative, got {replay_k}.")
        if (done_fn is None) != (done_key is None):
            raise ValueError("done_fn and done_key must be given together.")

        self.reward_fn = reward_fn
        self.strategy = strategy
        self.relabel_probability = 1.0 - 1.0 / (1.0 + replay_k)
        self.desired_goal_key = desired_goal_key
        self.next_achieved_goal_key = next_achieved_goal_key
        self.reward_key = reward_key
        self.episode_end_key = episode_end_key
        self.done_fn = done_fn
        self.done_key = done_key

        # Absolute step counter, the absolute step stored in each slot, and the
        # absolute first/last step of each slot's episode (-1: not finished yet).
        self.total_steps = 0
        self.slot_time = torch.zeros(steps, dtype=torch.long, device=device)
        self.episode_start = torch.zeros((steps, num_envs), dtype=torch.long, device=device)
        self.episode_end = torch.full((steps, num_envs), -1, dtype=torch.long, device=device)
        self.current_episode_start = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self) -> None:
        super().reset()
        self.total_steps = 0
        self.slot_time.zero_()
        self.episode_start.zero_()
        self.episode_end.fill_(-1)
        self.current_episode_start.zero_()

    def add_records(self, record: dict[str, Any]) -> None:
        if self.episode_end_key not in record:
            raise ValueError(f"Records must contain the episode end key '{self.episode_end_key}'.")
        episode_end = torch.as_tensor(record[self.episode_end_key], device=self.device).to(torch.bool)
        if episode_end.shape != (self.num_envs,):
            raise ValueError(
                f"'{self.episode_end_key}' must have shape ({self.num_envs},), got {tuple(episode_end.shape)}."
            )

        slot, now = self.step, self.total_steps
        super().add_records(record)

        self.slot_time[slot] = now
        self.episode_start[slot] = self.current_episode_start
        self.episode_end[slot] = -1
        for env in torch.nonzero(episode_end).flatten().tolist():
            # Only the episode's most recent `steps` steps can still be in the buffer.
            first = max(int(self.current_episode_start[env]), now - self.steps + 1)
            slots = torch.arange(first, now + 1, device=self.device) % self.steps
            self.episode_end[slots, env] = now
        self.current_episode_start[episode_end] = now + 1
        self.total_steps = now + 1

    def _goal_steps(self, slots: torch.Tensor, envs: torch.Tensor) -> torch.Tensor:
        """Absolute step whose achieved goal relabels each sampled transition."""
        now = self.slot_time[slots]
        end = self.episode_end[slots, envs]
        end = torch.where(end < 0, torch.full_like(end, self.total_steps - 1), end)
        if self.strategy == "final":
            return end
        if self.strategy == "future":
            low = now
        else:
            oldest = self.total_steps - self.current_size
            low = torch.clamp(self.episode_start[slots, envs], min=oldest)
        offset = torch.rand(now.shape, device=self.device) * (end - low + 1).to(torch.float32)
        return low + torch.minimum(offset.long(), end - low)

    def sample_batch(self, key_names: list[str], batch_size: int) -> dict[str, torch.Tensor]:
        """Sample transitions and relabel a fraction of their goals.

        ``key_names`` must include the desired goal and reward keys. The
        returned batch also has a boolean ``her_mask`` marking relabeled rows.
        """
        total = self.current_size * self.num_envs
        if total == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        for key in (self.desired_goal_key, self.reward_key):
            if key not in key_names:
                raise ValueError(f"key_names must include '{key}' for HER relabeling.")
        if self.done_key is not None and self.done_key not in key_names:
            raise ValueError(f"key_names must include '{self.done_key}' when done_fn is set.")

        indices = torch.randint(0, total, (batch_size,), device=self.device)
        slots, envs = indices // self.num_envs, indices % self.num_envs
        batch = self._gather(key_names, slots, envs)

        her_mask = torch.rand(batch_size, device=self.device) < self.relabel_probability
        batch["her_mask"] = her_mask
        if not her_mask.any():
            return batch

        slots, envs = slots[her_mask], envs[her_mask]
        goal_slots = self._goal_steps(slots, envs) % self.steps
        achieved_goals = self.data[self.next_achieved_goal_key]
        goals = achieved_goals[goal_slots, envs]
        achieved = achieved_goals[slots, envs]

        batch[self.desired_goal_key] = batch[self.desired_goal_key].clone()
        batch[self.desired_goal_key][her_mask] = goals.to(batch[self.desired_goal_key].dtype)
        rewards = torch.as_tensor(self.reward_fn(achieved, goals), device=self.device)
        batch[self.reward_key] = batch[self.reward_key].clone()
        batch[self.reward_key][her_mask] = rewards.reshape(-1).to(batch[self.reward_key].dtype)
        if self.done_fn is not None:
            dones = torch.as_tensor(self.done_fn(achieved, goals, rewards), device=self.device)
            batch[self.done_key] = batch[self.done_key].clone()
            batch[self.done_key][her_mask] = dones.reshape(-1).to(batch[self.done_key].dtype)
        return batch

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["her"] = {
            "total_steps": self.total_steps,
            "slot_time": self.slot_time.cpu(),
            "episode_start": self.episode_start.cpu(),
            "episode_end": self.episode_end.cpu(),
            "current_episode_start": self.current_episode_start.cpu(),
        }
        return state

    def load_state_dict(self, state: dict[str, Any], device: torch.device | None = None) -> None:
        super().load_state_dict(state, device)
        her = state["her"]
        self.total_steps = her["total_steps"]
        for name in ("slot_time", "episode_start", "episode_end", "current_episode_start"):
            setattr(self, name, her[name].to(self.device))
