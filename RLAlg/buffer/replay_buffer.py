import os
import torch

class ReplayBuffer:
    def __init__(
        self,
        num_envs: int,
        max_size: int,
        state_dim: tuple[int],
        action_dim: tuple[int],
        state_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu")
    ):
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_step = max_size // self.num_envs
        self.device = device
        self.state_dtype = state_dtype

        self.step = 0
        self.current_size = 0

        self.states = torch.zeros((self.max_step, num_envs) + state_dim, dtype=state_dtype, device=self.device)
        self.next_states = torch.zeros((self.max_step, num_envs) + state_dim, dtype=state_dtype, device=self.device)
        self.actions = torch.zeros((self.max_step, num_envs) + action_dim, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.max_step, num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.max_step, num_envs), dtype=torch.float32, device=self.device)

    def add_steps(self, state: any, action: any, reward: any, done: any, next_state: any):
        self.states[self.step] = torch.as_tensor(state, dtype=self.state_dtype, device=self.device)
        self.actions[self.step] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.step] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[self.step] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.next_states[self.step] = torch.as_tensor(next_state, dtype=self.state_dtype, device=self.device)

        self.step += 1
        if self.current_size < self.max_step:
            self.current_size += 1
        if self.step >= self.max_step:
            self.step = 0

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        total_steps = self.current_size * self.num_envs
        indices = torch.randint(0, total_steps, (batch_size,), device=self.device)
        return {
            "states": self.states[:self.current_size].view(-1, *self.state_dim)[indices],
            "actions": self.actions[:self.current_size].view(-1, *self.action_dim)[indices],
            "rewards": self.rewards[:self.current_size].view(-1)[indices],
            "dones": self.dones[:self.current_size].view(-1)[indices],
            "next_states": self.next_states[:self.current_size].view(-1, *self.state_dim)[indices]
        }

    def save(self, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save([
            self.states[:self.current_size].view(-1, *self.state_dim).cpu(),
            self.actions[:self.current_size].view(-1, *self.action_dim).cpu(),
            self.rewards[:self.current_size].view(-1).cpu(),
            self.dones[:self.current_size].view(-1).cpu(),
            self.next_states[:self.current_size].view(-1, *self.state_dim).cpu()
        ], f"{folder_path}/replays.pt")

    def __len__(self):
        return self.current_size * self.num_envs