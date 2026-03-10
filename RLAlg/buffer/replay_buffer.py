from typing import Generator, Any, Optional
import torch

@torch.no_grad()
def compute_returns(
    rewards: torch.Tensor,        # [T, N]
    terminated: torch.Tensor,     # [T, N] true terminal mask (not truncation)
    last_values: torch.Tensor,    # [N] = V(s_T)
    gamma: float = 0.99,
) -> torch.Tensor:
    T, N = rewards.shape
    returns = torch.zeros_like(rewards)

    terminated = terminated.to(dtype=rewards.dtype, device=rewards.device)
    not_term = 1.0 - terminated
    next_return = last_values.to(dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(T)):
        next_return = rewards[t] + gamma * next_return * not_term[t]
        returns[t] = next_return

    return returns


@torch.no_grad()
def compute_advantage_mc(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float = 0.99,
):
    returns = compute_returns(rewards, terminated, last_values, gamma)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages

@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,        # [T, N]
    values: torch.Tensor,         # [T, N]
    terminated: torch.Tensor,     # [T, N] true terminal mask (not truncation)
    last_values: torch.Tensor,    # [N]
    gamma: float = 0.99,
    lambda_: float = 0.95,
):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    terminated = terminated.to(dtype=rewards.dtype, device=rewards.device)
    not_term = 1.0 - terminated
    next_value = last_values.to(dtype=rewards.dtype, device=rewards.device)
    next_adv = torch.zeros((N,), dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value * not_term[t] - values[t]
        next_adv = delta + gamma * lambda_ * next_adv * not_term[t]
        advantages[t] = next_adv
        next_value = values[t]

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages

class ReplayBuffer:
    def __init__(self, num_envs: int, steps: int, device: torch.device = torch.device("cpu")):
        self.num_envs = num_envs
        self.steps = steps
        self.device = device
        self.data: dict[str, torch.Tensor] = {}

        self.step = 0
        self.current_size = 0

    def reset(self) -> None:
        self.step = 0
        self.current_size = 0
        for key in self.data:
            self.data[key].fill_(0)

    def create_storage_space(self, key_name: str, data_shape: tuple[int]=(), dtype: torch.dtype=torch.float32) -> None:
        self.data[key_name] = torch.zeros((self.steps, self.num_envs, *data_shape), dtype=dtype, device=self.device)

    def add_storage(self, key_name: str, values: torch.Tensor) -> None:
        if values.ndim < 2:
            raise ValueError(
                f"Storage for key '{key_name}' must have at least 2 dims [steps, num_envs, ...], got shape {tuple(values.shape)}."
            )
        if values.shape[0] != self.steps or values.shape[1] != self.num_envs:
            raise ValueError(
                f"Storage for key '{key_name}' must have shape [steps={self.steps}, num_envs={self.num_envs}, ...], got {tuple(values.shape)}."
            )
        self.data[key_name] = values.to(self.device)

    def add_records(self, record: dict[str, Any]) -> None:
        idx = self.step
        for key, value in record.items():
            if key not in self.data:
                raise ValueError(f"Key '{key}' not found in buffer.")
            value = torch.as_tensor(value).detach().to(
                device=self.device, dtype=self.data[key].dtype
            )
            expected_shape = self.data[key][idx].shape
            if value.shape != expected_shape:
                raise ValueError(
                    f"Record value for key '{key}' must have shape {tuple(expected_shape)}, got {tuple(value.shape)}."
                )
            self.data[key][idx] = value

        self.step = (self.step + 1) % self.steps
        self.current_size = min(self.current_size + 1, self.steps)

    def sample_batchs(self, key_names:list[str], batch_size: int) -> Generator[dict[str, torch.Tensor], None, None]:
        total = self.current_size * self.num_envs
        if total == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        indices = torch.randperm(total, device=self.device)

        batch = {}
        for key in key_names:
            data = self.data[key]
            batch[key] = data[:self.current_size].reshape(total, *data.shape[2:])

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield {key: value[idx] for key, value in batch.items()}

    def sample_batch(
        self,
        key_names: list[str],
        batch_size: int,
        future_steps: int = 0,
        episode_end_key: Optional[str] = None,
        her_strategies: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a flat batch and optionally attach HER relabel goal tensors.

        Args:
            key_names: Keys to sample from storage.
            batch_size: Number of sampled transitions.
            future_steps: Maximum look-ahead steps for ``future`` strategy.
            episode_end_key: Optional episode-end key override (defaults to
                ``done`` then ``terminated`` if available).
            her_strategies: Optional HER goal strategies. Supported values:
                ``future``, ``final``, ``episode``, ``random``.

        Returns:
            A dict with sampled base keys and optional strategy keys:
            ``{key}_future``, ``{key}_final``, ``{key}_episode``, ``{key}_random``.

        Notes:
            If ``her_strategies`` is ``None``, behavior is backward compatible:
            only ``future`` goals are generated when ``future_steps > 0``.
            Strategies ``future/final/episode`` require episode boundaries.
        """
        total = self.current_size * self.num_envs
        if total == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if future_steps < 0:
            raise ValueError(f"future_steps must be non-negative, got {future_steps}.")
        indices = torch.randint(0, total, (batch_size,), device=self.device)

        batch = {}
        for key in key_names:
            data = self.data[key]
            shape = data.shape[2:]
            flat = data[:self.current_size].reshape(total, *shape)
            batch[key] = flat[indices]

        valid_strategies = {"future", "final", "episode", "random"}
        if her_strategies is None:
            strategies: list[str] = ["future"] if future_steps > 0 else []
        else:
            strategies = list(dict.fromkeys(her_strategies))
            invalid = [s for s in strategies if s not in valid_strategies]
            if invalid:
                raise ValueError(
                    f"Invalid her_strategies {invalid}. "
                    f"Expected subset of {sorted(valid_strategies)}."
                )

        if not strategies:
            return batch
        if "future" in strategies and future_steps == 0:
            raise ValueError(
                "future strategy requires future_steps > 0."
            )

        needs_episode_boundary = any(s in {"future", "final", "episode"} for s in strategies)
        if not needs_episode_boundary:
            for key in key_names:
                data = self.data[key]
                shape = data.shape[2:]
                flat = data[:self.current_size].reshape(total, *shape)
                batch[f"{key}_random"] = flat[torch.randint(0, total, (batch_size,), device=self.device)]
            return batch

        if episode_end_key is None:
            if "done" in self.data:
                episode_end_key = "done"
            elif "terminated" in self.data:
                episode_end_key = "terminated"
            else:
                raise ValueError(
                    "HER relabeling requires an episode end key. "
                    "Pass episode_end_key or add a 'done'/'terminated' storage key."
                )
        if episode_end_key not in self.data:
            raise ValueError(f"Episode end key '{episode_end_key}' not found in buffer.")

        end_data = self.data[episode_end_key][:self.current_size]
        if end_data.ndim == 2:
            end_mask = end_data.to(dtype=torch.bool)
        else:
            end_mask = end_data.to(dtype=torch.bool).reshape(
                self.current_size, self.num_envs, -1
            ).any(dim=-1)

        step_indices = indices // self.num_envs
        env_indices = indices % self.num_envs
        episode_start_indices = step_indices.clone()
        episode_end_indices = step_indices.clone()

        for i in range(batch_size):
            step = int(step_indices[i].item())
            env = int(env_indices[i].item())
            future_terminated_rel = torch.where(end_mask[step:, env])[0]
            if future_terminated_rel.numel() > 0:
                episode_end = step + int(future_terminated_rel[0].item())
            else:
                episode_end = self.current_size - 1
            episode_end_indices[i] = episode_end

            if step == 0:
                episode_start = 0
            else:
                past_terminated_rel = torch.where(end_mask[:step, env])[0]
                if past_terminated_rel.numel() > 0:
                    episode_start = int(past_terminated_rel[-1].item()) + 1
                else:
                    episode_start = 0
            episode_start_indices[i] = episode_start

        strategy_indices: dict[str, torch.Tensor] = {}

        if "future" in strategies:
            future_step_indices = step_indices.clone()
            for i in range(batch_size):
                step = int(step_indices[i].item())
                episode_end = int(episode_end_indices[i].item())
                future_step_indices[i] = min(step + future_steps, episode_end)
            strategy_indices["future"] = future_step_indices * self.num_envs + env_indices

        if "final" in strategies:
            strategy_indices["final"] = episode_end_indices * self.num_envs + env_indices

        if "episode" in strategies:
            episode_step_indices = step_indices.clone()
            for i in range(batch_size):
                episode_start = int(episode_start_indices[i].item())
                episode_end = int(episode_end_indices[i].item())
                episode_step_indices[i] = torch.randint(
                    episode_start, episode_end + 1, (1,), device=self.device
                )[0]
            strategy_indices["episode"] = episode_step_indices * self.num_envs + env_indices

        if "random" in strategies:
            strategy_indices["random"] = torch.randint(0, total, (batch_size,), device=self.device)

        for key in key_names:
            data = self.data[key]
            shape = data.shape[2:]
            flat = data[:self.current_size].reshape(total, *shape)
            for strategy, strategy_idx in strategy_indices.items():
                suffix = "_future" if strategy == "future" else f"_{strategy}"
                batch[f"{key}{suffix}"] = flat[strategy_idx]
        return batch
        
    def sample_tensor(self, key_name:str, batch_size: int) -> torch.Tensor:
        data = self.data[key_name]
        total = self.current_size * self.num_envs
        if total == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        indices = torch.randint(0, total, (batch_size,), device=self.device)
        shape = data.shape[2:]
        flat = data[:self.current_size].reshape(total, *shape)
        return flat[indices]

    def sample_sequence_batches(
        self,
        key_names: list[str],
        seq_len: int,
        batch_size: int,
        state_keys: Optional[list[str]] = None,
        shuffle: bool = True,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        if self.current_size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        state_keys = state_keys or []
        for key in key_names + state_keys:
            if key not in self.data:
                raise ValueError(f"Key '{key}' not found in buffer.")
        overlap = set(key_names) & set(state_keys)
        if overlap:
            raise ValueError(
                f"state_keys must not overlap key_names, got overlapping keys: {sorted(overlap)}."
            )

        chunk_index: list[tuple[int, int]] = []
        for env_idx in range(self.num_envs):
            for start in range(0, self.current_size, seq_len):
                chunk_index.append((env_idx, start))
        if shuffle:
            order = torch.randperm(len(chunk_index), device=self.device).tolist()
            chunk_index = [chunk_index[i] for i in order]

        for batch_start in range(0, len(chunk_index), batch_size):
            chunk_batch = chunk_index[batch_start : batch_start + batch_size]
            batch_count = len(chunk_batch)
            batch: dict[str, torch.Tensor] = {}

            for key in key_names:
                data = self.data[key]
                seq_shape = (seq_len, batch_count, *data.shape[2:])
                batch[key] = torch.zeros(seq_shape, dtype=data.dtype, device=self.device)

            valid_mask = torch.zeros((seq_len, batch_count), dtype=torch.bool, device=self.device)

            for i, (env_idx, start) in enumerate(chunk_batch):
                end = min(start + seq_len, self.current_size)
                valid_steps = end - start
                valid_mask[:valid_steps, i] = True

                for key in key_names:
                    batch[key][:valid_steps, i] = self.data[key][start:end, env_idx]

            for key in state_keys:
                data = self.data[key]
                state_shape = (batch_count, *data.shape[2:])
                init_states = torch.zeros(state_shape, dtype=data.dtype, device=self.device)
                for i, (env_idx, start) in enumerate(chunk_batch):
                    init_states[i] = data[start, env_idx]
                batch[f"{key}_init"] = init_states

            batch["valid_mask"] = valid_mask
            yield batch

    # ---------------- SAVE & LOAD ---------------- #

    def save(self, path: str) -> None:
        """Save replay buffer to a file."""
        save_dict = {
            "num_envs": self.num_envs,
            "steps": self.steps,
            "step": self.step,
            "current_size": self.current_size,
            "data": {k: v.cpu() for k, v in self.data.items()},
        }
        torch.save(save_dict, path)
        print(f"[ReplayBuffer] Saved buffer to '{path}'")

    def load(self, path: str, device: torch.device | None = None) -> None:
        """Load replay buffer from a file."""
        checkpoint = torch.load(path, map_location=device or self.device)
        self.num_envs = checkpoint["num_envs"]
        self.steps = checkpoint["steps"]
        self.step = checkpoint["step"]
        self.current_size = checkpoint["current_size"]
        self.data = {k: v.to(device or self.device) for k, v in checkpoint["data"].items()}
        self.device = device or self.device
        print(f"[ReplayBuffer] Loaded buffer from '{path}' to device '{self.device}'")
