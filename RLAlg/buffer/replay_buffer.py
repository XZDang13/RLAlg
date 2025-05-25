from typing import Generator
import os
import torch
from torch import Tensor

def compute_advantage(rewards, values, dones, gamma: float = 0.99) -> None:
    
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

def compute_gae(rewards, values, dones, last_values: Tensor, gamma: float = 0.99, lambda_: float = 0.95) -> None:
    steps, num_envs = rewards.size(0), rewards.size(1)

    returns = torch.zeros_like(rewards, dtype=rewards.dtype, device=rewards.device)
    advantages = torch.zeros_like(rewards, dtype=rewards.dtype, device=rewards.device)

    next_value = torch.as_tensor(last_values, dtype=rewards.dtype, device=rewards.device)
    next_advantage = torch.zeros(num_envs, dtype=rewards.dtype, device=rewards.device)

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
        self.data: dict[str, Tensor] = {}

        self.step = 0
        self.current_size = 0

    def create_storage_space(self, key_name: str, shape: tuple[int], dtype: torch.dtype=torch.float32) -> None:
        self.data[key_name] = torch.zeros((self.steps, self.num_envs, *shape), dtype=dtype, device=self.device)

    def add_storage(self, key_name: str, values: Tensor) -> None:
        self.data[key_name] = values.to(self.device)

    def add_records(self, record: dict[str, any]) -> None:
        self.step = (self.step + 1) % self.steps
        self.current_size = min(self.current_size + 1, self.steps)

        for key, value in record.items():
            assert key in self.data, f"Key '{key}' not found in buffer."
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=self.data[key].dtype)
            self.data[key][self.step] = value.to(self.device)

    def sample_batchs(self, key_names:list[str], batch_size: int) -> Generator[dict[str, Tensor], None, None]:
        total = self.steps * self.num_envs
        indices = torch.randperm(total, device=self.device)

        batch = {}
        for key in key_names:
            data = self.data[key]

            batch[key] = data.view(total, *data.shape[2:])

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield {key: value[idx] for key, value in batch.items()}

    def sample_batch(self, key_names:list[str], batch_size: int) -> dict[str, Tensor]:
        total = self.current_size * self.num_envs
        indices = torch.randint(0, total, (batch_size,), device=self.device)

        batch = {}
        for key in key_names:
            data = self.data[key]
            shape = data.shape[2:]  # drop (T, N)
            flat = data[:self.current_size].reshape(total, *shape)
            batch[key] = flat[indices]

        return batch