from typing import Generator
import torch
from torch import Tensor


class RolloutBuffer:
    def __init__(
        self,
        num_envs: int,
        steps: int,
        observation_space: dict[str, tuple[tuple[int], torch.dtype]],
        action_space: dict[str, tuple[tuple[int], torch.dtype]],
        reward_space: list[str],
        device: torch.device = torch.device("cpu")
    ):
        self.num_envs = num_envs
        self.steps = steps
        self.device = device

        self.data: dict[str, Tensor] = {}
        self.shapes: dict[str, tuple[int]] = {}
        self.dtypes: dict[str, torch.dtype] = {}

        # Observations
        for key, (shape, dtype) in observation_space.items():
            name = f"{key}_observations"
            self.data[name] = torch.zeros((steps, num_envs, *shape), dtype=dtype, device=device)
            self.shapes[name] = shape
            self.dtypes[name] = dtype

        # Actions and log probs
        for key, (shape, dtype) in action_space.items():
            a_name = f"{key}_actions"
            lp_name = f"{key}_log_probs"
            self.data[a_name] = torch.zeros((steps, num_envs, *shape), dtype=dtype, device=device)
            self.data[lp_name] = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
            self.shapes[a_name] = shape
            self.dtypes[a_name] = dtype
            self.dtypes[lp_name] = torch.float32

        # Rewards, returns, values, advantages
        for key in reward_space:
            for suffix in ["rewards", "returns", "advantages", "values"]:
                name = f"{key}_{suffix}"
                self.data[name] = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
                self.dtypes[name] = torch.float32

        # Dones
        self.data["dones"] = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.dtypes["dones"] = torch.float32

    def add_steps(self, step: int, record: dict[str, any]) -> None:
        for key, value in record.items():
            if key in self.data:
                self.data[key][step] = torch.as_tensor(value, dtype=self.dtypes[key], device=self.device)

    def compute_advantage(self, key: str, gamma: float = 0.99) -> None:
        rewards = self.data[f"{key}_rewards"]
        values = self.data[f"{key}_values"]
        returns = self.data[f"{key}_returns"]
        advantages = self.data[f"{key}_advantages"]
        dones = self.data["dones"]

        next_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.steps)):
            next_return = rewards[step] + gamma * next_return * (1.0 - dones[step])
            returns[step] = next_return
            advantages[step] = returns[step] - values[step]

        self._normalize_advantages(key)

    def compute_gae(self, key: str, last_value: Tensor, gamma: float = 0.99, lambda_: float = 0.95) -> None:
        rewards = self.data[f"{key}_rewards"]
        values = self.data[f"{key}_values"]
        returns = self.data[f"{key}_returns"]
        advantages = self.data[f"{key}_advantages"]
        dones = self.data["dones"]

        next_value = torch.as_tensor(last_value, dtype=torch.float32, device=self.device)
        next_advantage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for step in reversed(range(self.steps)):
            delta = rewards[step] + gamma * next_value * (1.0 - dones[step]) - values[step]
            next_advantage = delta + gamma * lambda_ * next_advantage * (1.0 - dones[step])
            advantages[step] = next_advantage
            returns[step] = advantages[step] + values[step]
            next_value = values[step]

        self._normalize_advantages(key)

    def _normalize_advantages(self, key: str) -> None:
        adv = self.data[f"{key}_advantages"]
        mean = adv.mean()
        std = adv.std(unbiased=False)
        self.data[f"{key}_advantages"] = (adv - mean) / (std + 1e-8)

    def batch_sample(self, batch_size: int) -> Generator[dict[str, Tensor], None, None]:
        total = self.steps * self.num_envs
        indices = torch.randperm(total, device=self.device)

        flattened_data = {}
        for key, tensor in self.data.items():
            #if key in self.shapes:
            #    flattened_data[key] = tensor.view(total, *self.shapes[key])
            #else:
            #    flattened_data[key] = tensor.view(total)

            flattened_data[key] = tensor.view(total, *tensor.shape[2:])

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield {key: value[idx] for key, value in flattened_data.items()}

    def to(self, device: torch.device) -> None:
        self.device = device
        for key in self.data:
            self.data[key] = self.data[key].to(device)

    def __len__(self) -> int:
        return self.steps * self.num_envs
