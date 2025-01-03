import torch
import torch.nn as nn
import torch.nn.functional as F

NNMODEL = nn.Module

class PPO:
    @staticmethod
    def compute_policy_loss(policy_model: NNMODEL,
                           log_probs_hat: torch.Tensor,
                           observations: torch.Tensor,
                           actions: torch.Tensor,
                           advantages: torch.Tensor,
                           clip_ratio: float) -> torch.Tensor:
        
        dist, _, log_probs = policy_model(observations, actions)
        
        ratio = (log_probs - log_probs_hat).exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        
        return loss.mean(), dist.entropy().mean()

    @staticmethod
    def compute_value_loss(value_model: NNMODEL,
                            observations: torch.Tensor,
                            returns: torch.Tensor) -> torch.Tensor:
        
        values = value_model(observations)
        loss = 0.5 * (values - returns) ** 2
        
        return loss.mean()
    
    @staticmethod
    def compute_clipped_value_loss(value_model: NNMODEL,
                                   observations: torch.Tensor,
                                   values_hat: torch.Tensor,
                                   returns: torch.Tensor,
                                   clip_ratio: float) -> torch.Tensor:
        values = value_model(observations).squeeze()
        loss_unclipped = 0.5 * (values - returns) ** 2

        values_clipped = values_hat+ torch.clamp(values - values_hat, -clip_ratio, clip_ratio)
        loss_clipped = 0.5 * (values_clipped - returns) ** 2

        loss = torch.max(loss_unclipped, loss_clipped)

        return loss.mean()
