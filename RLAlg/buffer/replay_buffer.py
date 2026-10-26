from typing import Generator, Any, Optional
import torch

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

    def time_order(self) -> torch.Tensor:
        """Storage slots of the stored steps, oldest first.

        Once the circular buffer wraps, the oldest step is at the write pointer,
        so ``data[:current_size]`` is no longer in time order.
        """
        if self.current_size < self.steps:
            return torch.arange(self.current_size, device=self.device)
        return (torch.arange(self.steps, device=self.device) + self.step) % self.steps

    def sample_batch(self, key_names: list[str], batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a flat batch of transitions uniformly with replacement."""
        total = self.current_size * self.num_envs
        if total == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        indices = torch.randint(0, total, (batch_size,), device=self.device)
        return self._gather(key_names, indices // self.num_envs, indices % self.num_envs)

    def _gather(
        self, key_names: list[str], slots: torch.Tensor, envs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        for key in key_names:
            if key not in self.data:
                raise ValueError(f"Key '{key}' not found in buffer.")
        return {key: self.data[key][slots, envs] for key in key_names}

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

        # Chunks follow time order, so a sequence never jumps across the write pointer.
        order = self.time_order()
        chunk_index: list[tuple[int, int]] = []
        for env_idx in range(self.num_envs):
            for start in range(0, self.current_size, seq_len):
                chunk_index.append((env_idx, start))
        if shuffle:
            permutation = torch.randperm(len(chunk_index), device=self.device).tolist()
            chunk_index = [chunk_index[i] for i in permutation]

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
                slots = order[start : start + seq_len]
                valid_steps = slots.shape[0]
                valid_mask[:valid_steps, i] = True

                for key in key_names:
                    batch[key][:valid_steps, i] = self.data[key][slots, env_idx]

            for key in state_keys:
                data = self.data[key]
                state_shape = (batch_count, *data.shape[2:])
                init_states = torch.zeros(state_shape, dtype=data.dtype, device=self.device)
                for i, (env_idx, start) in enumerate(chunk_batch):
                    init_states[i] = data[order[start], env_idx]
                batch[f"{key}_init"] = init_states

            batch["valid_mask"] = valid_mask
            yield batch

    # ---------------- SAVE & LOAD ---------------- #

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "steps": self.steps,
            "step": self.step,
            "current_size": self.current_size,
            "data": {k: v.cpu() for k, v in self.data.items()},
        }

    def load_state_dict(self, state: dict[str, Any], device: torch.device | None = None) -> None:
        self.device = device or self.device
        self.num_envs = state["num_envs"]
        self.steps = state["steps"]
        self.step = state["step"]
        self.current_size = state["current_size"]
        self.data = {k: v.to(self.device) for k, v in state["data"].items()}

    def save(self, path: str) -> None:
        """Save replay buffer to a file."""
        torch.save(self.state_dict(), path)
        print(f"[ReplayBuffer] Saved buffer to '{path}'")

    def load(self, path: str, device: torch.device | None = None) -> None:
        """Load replay buffer from a file."""
        self.load_state_dict(torch.load(path, map_location=device or self.device), device)
        print(f"[ReplayBuffer] Loaded buffer from '{path}' to device '{self.device}'")
