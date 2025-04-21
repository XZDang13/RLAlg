from typing import Generator
import torch
from torch import Tensor


class RolloutBuffer:
    def __init__(
        self,
        num_envs: int,
        steps: int,
        state_dim: tuple[int],
        action_dim: tuple[int],
        state_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu")
    ):
        self.num_envs = num_envs
        self.steps = steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_dtype = state_dtype
        self.device = device

        self.states = torch.zeros((steps, num_envs) + state_dim, dtype=state_dtype, device=device)
        self.actions = torch.zeros((steps, num_envs) + action_dim, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)

    def add_steps(
        self,
        step: int,
        state: any,
        action: any,
        log_prob: any,
        reward: any,
        done: any,
        value: any = None
    ) -> None:
        self.states[step] = torch.as_tensor(state, dtype=self.state_dtype, device=self.device)
        self.actions[step] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.log_probs[step] = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        self.rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[step] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        if value is not None:
            self.values[step] = torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def compute_advantage(self, gamma: float = 0.99) -> None:
        next_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.steps)):
            next_return = self.rewards[step] + gamma * next_return * (1.0 - self.dones[step])
            self.returns[step] = next_return
            self.advantages[step] = self.returns[step] - self.values[step]
        self._normalize_advantages()

    def compute_gae(self, last_value: any, gamma: float = 0.99, lambda_: float = 0.95) -> None:
        next_value = torch.as_tensor(last_value, dtype=torch.float32, device=self.device)
        next_advantage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.steps)):
            delta = self.rewards[step] + gamma * next_value * (1.0 - self.dones[step]) - self.values[step]
            next_advantage = delta + gamma * lambda_ * next_advantage * (1.0 - self.dones[step])
            self.advantages[step] = next_advantage
            self.returns[step] = self.advantages[step] + self.values[step]
            next_value = self.values[step]
        self._normalize_advantages()

    def _normalize_advantages(self) -> None:
        mean = self.advantages.mean()
        std = self.advantages.std(unbiased=False)
        self.advantages = (self.advantages - mean) / (std + 1e-8)

    def batch_sample(self, batch_size: int) -> Generator[dict[str, Tensor], None, None]:
        total_steps = self.steps * self.num_envs
        indices = torch.randperm(total_steps, device=self.device)

        flat_states = self.states.view(-1, *self.state_dim)
        flat_actions = self.actions.view(-1, *self.action_dim)
        flat_log_probs = self.log_probs.view(-1)
        flat_values = self.values.view(-1)
        flat_returns = self.returns.view(-1)
        flat_advantages = self.advantages.view(-1)

        for start in range(0, total_steps, batch_size):
            batch_indices = indices[start:start + batch_size]
            yield {
                "states": flat_states[batch_indices],
                "actions": flat_actions[batch_indices],
                "log_probs": flat_log_probs[batch_indices],
                "values": flat_values[batch_indices],
                "returns": flat_returns[batch_indices],
                "advantages": flat_advantages[batch_indices],
            }

    def to(self, device: torch.device) -> None:
        self.device = device
        for attr in ["states", "actions", "log_probs", "rewards", "dones", "values", "returns", "advantages"]:
            tensor = getattr(self, attr)
            setattr(self, attr, tensor.to(device))

    def __len__(self) -> int:
        return self.steps * self.num_envs