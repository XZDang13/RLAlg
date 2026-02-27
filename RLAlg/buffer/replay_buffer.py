from typing import Generator, Any
import torch

@torch.no_grad()
def compute_returns(
    rewards: torch.Tensor,        # [T, N]
    terminated: torch.Tensor,     # [T, N] 1 if true terminal else 0
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
    terminated: torch.Tensor,     # [T, N]
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
            assert key in self.data, f"Key '{key}' not found in buffer."
            value = torch.as_tensor(value).detach().to(
                device=self.device, dtype=self.data[key].dtype
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

    def sample_batch(self, key_names:list[str], batch_size: int) -> dict[str, torch.Tensor]:
        total = self.current_size * self.num_envs
        if total == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        indices = torch.randint(0, total, (batch_size,), device=self.device)

        batch = {}
        for key in key_names:
            data = self.data[key]
            shape = data.shape[2:]
            flat = data[:self.current_size].reshape(total, *shape)
            batch[key] = flat[indices]
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
