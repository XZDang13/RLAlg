import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

NNMODEL = nn.Module

class PPO:
    @staticmethod
    def compute_actor_loss(policy: NNMODEL,
                           old_log_probs: torch.Tensor,
                           observations: torch.Tensor,
                           actions: torch.Tensor,
                           advantages: torch.Tensor,
                           clip_ratio: float) -> torch.Tensor:
        
        logits = policy(observations)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        
        ratio = (new_log_probs - old_log_probs).exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        return loss

    @staticmethod
    def compute_critic_loss(value_function: NNMODEL,
                            observations: torch.Tensor,
                            returns: torch.Tensor) -> torch.Tensor:
        
        value_estimates = value_function(observations).squeeze()
        loss = F.mse_loss(value_estimates, returns)
        
        return loss
