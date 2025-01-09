from typing import Generator
import torch

class RolloutBuffer:
    def __init__(self, num_envs: int, steps: int, state_dim: tuple[int], action_dim: tuple[int]):
        self.num_envs = num_envs
        self.steps = steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = torch.zeros((steps, num_envs) + state_dim)
        self.actions = torch.zeros((steps, num_envs) + action_dim)
        self.log_probs = torch.zeros((steps, num_envs))
        self.rewards = torch.zeros((steps, num_envs))
        self.dones = torch.zeros((steps, num_envs))
        self.values = torch.zeros((steps, num_envs))
        self.returns = torch.zeros((steps, num_envs))
        self.advantages = torch.zeros((steps, num_envs))

    def add_steps(self, step:int, state:any, action:any, log_prob:any, reward:any, done:any, value:any|None=None):
        self.states[step] = torch.as_tensor(state)
        self.actions[step] = torch.as_tensor(action)
        self.log_probs[step] = torch.as_tensor(log_prob)
        self.rewards[step] = torch.as_tensor(reward)
        self.dones[step] = torch.as_tensor(done)
        if value is not None:
            self.values[step] = torch.as_tensor(value)

    def compute_advantage(self, gamma:float=0.99):
        next_return = torch.zeros(self.num_envs)
        for step in reversed(range(self.steps)):
            next_return = self.rewards[step] + gamma * next_return * (1.0 - self.dones[step])
            self.returns[step] = next_return
            self.advantages[step] = self.returns[step] - self.values[step]

        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def compute_gae(self, last_value:any, gamma:float=0.99, lambda_:float=0.95):
        next_value = torch.as_tensor(last_value)
        next_advantage = torch.zeros(self.num_envs)
        for step in reversed(range(self.steps)):
            delta = self.rewards[step] + gamma * next_value * (1.0 - self.dones[step]) - self.values[step]
            next_advantage = delta + gamma * lambda_ * next_advantage * (1.0 - self.dones[step])
            self.advantages[step] = next_advantage
            next_value = self.values[step]
            self.returns[step] = self.advantages[step] + self.values[step]

        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def batch_sample(self, batch_size: int) -> Generator[dict[str, torch.Tensor]]:
        total_steps = self.steps * self.num_envs
        indices = torch.randperm(total_steps)

        for start in range(0, total_steps, batch_size):
            batch_indices = indices[start:start + batch_size]
            yield {
                "states": self.states.view(-1, *self.state_dim)[batch_indices],
                "actions": self.actions.view(-1, *self.action_dim)[batch_indices],
                "log_probs": self.log_probs.view(-1)[batch_indices],
                "values": self.values.view(-1)[batch_indices],
                "returns": self.returns.view(-1)[batch_indices],
                "advantages": self.advantages.view(-1)[batch_indices],
            }