import os
import torch
from torch import Tensor


class ReplayBuffer:
    def __init__(
        self,
        num_envs: int,
        max_size: int,
        observation_space: dict[str, tuple[tuple[int], torch.dtype]],
        action_space: dict[str, tuple[tuple[int], torch.dtype]],
        reward_space: list[str],
        device: torch.device = torch.device("cpu"),
    ):
        self.num_envs = num_envs
        self.max_step = max_size // num_envs
        self.device = device

        self.step = 0
        self.current_size = 0

        self.data: dict[str, Tensor] = {}
        self.dtypes: dict[str, torch.dtype] = {}
        self.shapes: dict[str, tuple[int]] = {}

        # Observations and next observations
        for key, (shape, dtype) in observation_space.items():
            for prefix in ["", "next_"]:
                name = f"{prefix}{key}_obs"
                self.data[name] = torch.zeros((self.max_step, num_envs, *shape), dtype=dtype, device=device)
                self.dtypes[name] = dtype
                self.shapes[name] = shape

        # Actions
        for key, (shape, dtype) in action_space.items():
            name = f"{key}_actions"
            self.data[name] = torch.zeros((self.max_step, num_envs, *shape), dtype=dtype, device=device)
            self.dtypes[name] = dtype
            self.shapes[name] = shape

        # Rewards and dones
        for key in reward_space:
            self.data[f"{key}_rewards"] = torch.zeros((self.max_step, num_envs), dtype=torch.float32, device=device)
            self.dtypes[f"{key}_rewards"] = torch.float32

        self.data["dones"] = torch.zeros((self.max_step, num_envs), dtype=torch.float32, device=device)
        self.dtypes["dones"] = torch.float32

    def add_steps(self, record: dict[str, any]) -> None:
        for key, value in record.items():
            if key in self.data:
                self.data[key][self.step] = torch.as_tensor(value, dtype=self.dtypes[key], device=self.device)

        self.step = (self.step + 1) % self.max_step
        self.current_size = min(self.current_size + 1, self.max_step)

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        total = self.current_size * self.num_envs
        indices = torch.randint(0, total, (batch_size,), device=self.device)

        batch = {}
        for key, tensor in self.data.items():
            shape = tensor.shape[2:]  # drop (T, N)
            flat = tensor[:self.current_size].reshape(total, *shape)
            batch[key] = flat[indices]

        return batch

    def save(self, folder_path: str) -> None:
        os.makedirs(folder_path, exist_ok=True)
        save_data = {
            key: tensor[:self.current_size].reshape(-1, *tensor.shape[2:]).cpu()
            for key, tensor in self.data.items()
        }
        torch.save(save_data, os.path.join(folder_path, "replays.pt"))

    def to(self, device: torch.device) -> None:
        self.device = device
        for key in self.data:
            self.data[key] = self.data[key].to(device)

    def __len__(self) -> int:
        return self.current_size * self.num_envs
