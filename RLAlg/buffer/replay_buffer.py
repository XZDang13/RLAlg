import torch

class ReplayBuffer:
    def __init__(self, num_envs: int, max_size: int, state_dim: tuple[int], action_dim: tuple[int]):
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_step = max_size // self.num_envs
        
        self.step = 0
        self.current_size = 0
        
        self.states = torch.zeros((self.max_step, num_envs) + state_dim)
        self.next_states = torch.zeros((self.max_step, num_envs) + state_dim)
        self.actions = torch.zeros((self.max_step, num_envs) + action_dim)
        self.rewards = torch.zeros((self.max_step, num_envs))
        self.dones = torch.zeros((self.max_step, num_envs))

    def add_steps(self, state:torch.Tensor, action:torch.Tensor, reward:torch.Tensor, done:torch.Tensor, next_state:torch.Tensor):
        self.states[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.next_states[self.step] = next_state
        
        self.step += 1
        
        if self.current_size < self.max_step:
            self.current_size += 1
        
        if self.step >= self.max_step:
            self.step = 0
            
    def sample(self, batch_size:int) -> dict[str, torch.Tensor]:
        total_steps = self.current_size * self.num_envs
        indices = torch.randint(0, total_steps, (batch_size,))
        
        return {
                "states": self.states[:self.current_size].view(-1, *self.state_dim)[indices],
                "actions": self.actions[:self.current_size].view(-1, *self.action_dim)[indices],
                "rewards": self.rewards[:self.current_size].view(-1)[indices],
                "dones": self.dones[:self.current_size].view(-1)[indices],
                "next_states": self.next_states[:self.current_size].view(-1, *self.state_dim)[indices]
        }