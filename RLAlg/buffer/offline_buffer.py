import torch

class OfflineReplayBuffer:
    def __init__(self, replay_file:str):
       states, actions, rewards, dones, next_states = torch.load(replay_file, weights_only=True)
       self.states = states
       self.actions = actions
       self.rewards = rewards
       self.dones = dones
       self.next_states = next_states
       
       self.size = self.states.size(0)
            
    def sample(self, batch_size:int) -> dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,))
        
        return {
                "states": self.states[indices],
                "actions": self.actions[indices],
                "rewards": self.rewards[indices],
                "dones": self.dones[indices],
                "next_states": self.next_states[indices]
        }