import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

BATCH = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class IQL:
    @staticmethod
    def compute_value_loss(value_model:nn.Module, critic_target_model:nn.Module,
                           observation:torch.Tensor, action:torch.Tensor, expectile:float) -> torch.Tensor:
        with torch.no_grad():
            q1, q2 = critic_target_model(observation, action)
            q = torch.min(q1, q2).detach()

        value = value_model(observation)
        diff = q-value
        value_loss = asymmetric_l2_loss(diff, expectile)

        return value_loss


    @staticmethod
    def compute_actor_loss(actor_model:nn.Module, value_model:nn.Module,
                           critic_target_model:nn.Module, observation:torch.Tensor,
                           action:torch.Tensor, temperature:float) -> torch.Tensor:
        with torch.no_grad():
            value = value_model(observation)
            q1, q2 = critic_target_model(observation, action)
            q = torch.min(q1, q2)

        exp_a = torch.exp((q - value) * temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(observation.device))

        _, _, log_prob = actor_model(observation, action)
        
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss
    
    @staticmethod
    def compute_critic_loss(value_model:nn.Module, critic_model:nn.Module,
                            observation:torch.Tensor, action:torch.Tensor,
                            reward:torch.Tensor, done:torch.Tensor,
                            next_observation:torch.Tensor, gamma:float) -> torch.Tensor:

        with torch.no_grad():
            next_value = value_model(next_observation)
            q_targ = (reward + gamma * (1-done) * next_value)

        q1, q2 = critic_model(observation, action)

        critic_1_loss = ((q1 - q_targ)**2).mean()
        critic_2_loss = ((q2 - q_targ)**2).mean()
        critic_loss = critic_1_loss + critic_2_loss

        return critic_loss
    
    @staticmethod
    @torch.no_grad()
    def update_target_param(model: nn.Module, model_target: nn.Module, tau: float) -> None:
        for param, param_target in zip(model.parameters(), model_target.parameters()):
            param_target.data.mul_(1 - tau)
            param_target.data.add_(tau * param.data)
