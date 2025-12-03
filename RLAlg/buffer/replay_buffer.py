from typing import Generator, Any
import os
import torch

def compute_returns(rewards:torch.Tensor, dones:torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    steps, num_envs = rewards.size(0), rewards.size(1)

    returns = torch.zeros_like(rewards, dtype=rewards.dtype, device=rewards.device)
    next_return = torch.zeros(num_envs, dtype=rewards.dtype, device=rewards.device)
    for step in reversed(range(steps)):
        next_return = rewards[step] + gamma * next_return * (1.0 - dones[step])
        returns[step] = next_return
        
    return returns

def compute_advantage(rewards:torch.Tensor,
                      values:torch.Tensor,
                      dones:torch.Tensor,
                      gamma: float = 0.99) -> tuple[torch.Tensor, torch.Tensor]:
    steps, num_envs = rewards.size(0), rewards.size(1)

    returns = torch.zeros_like(rewards, dtype=rewards.dtype, device=rewards.device)
    advantages = torch.zeros_like(rewards, dtype=rewards.dtype, device=rewards.device)
    

    next_return = torch.zeros(num_envs, dtype=rewards.dtype, device=rewards.device)
    for step in reversed(range(steps)):
        next_return = rewards[step] + gamma * next_return * (1.0 - dones[step])
        returns[step] = next_return
        advantages[step] = returns[step] - values[step]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

def compute_gae(rewards:torch.Tensor,
                values:torch.Tensor,
                dones:torch.Tensor,
                last_values: torch.Tensor,
                gamma: float = 0.99, lambda_: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    size = rewards.size()

    steps = size[0]

    returns = torch.zeros_like(rewards, dtype=rewards.dtype, device=rewards.device)
    advantages = torch.zeros_like(rewards, dtype=rewards.dtype, device=rewards.device)

    next_value = torch.as_tensor(last_values, dtype=rewards.dtype, device=rewards.device)
    next_advantage = torch.zeros(size[1:], dtype=rewards.dtype, device=rewards.device)

    for step in reversed(range(steps)):
        delta = rewards[step] + gamma * next_value * (1.0 - dones[step]) - values[step]
        next_advantage = delta + gamma * lambda_ * next_advantage * (1.0 - dones[step])
        advantages[step] = next_advantage
        returns[step] = advantages[step] + values[step]
        next_value = values[step]

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
        total = self.steps * self.num_envs
        indices = torch.randperm(total, device=self.device)

        batch = {}
        for key in key_names:
            data = self.data[key]
            batch[key] = data.view(total, *data.shape[2:])

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield {key: value[idx] for key, value in batch.items()}

    def sample_batch(self, key_names:list[str], batch_size: int) -> dict[str, torch.Tensor]:
        total = self.current_size * self.num_envs
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
