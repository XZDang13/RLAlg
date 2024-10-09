import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

BATCH = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class IQL:
    @staticmethod
    def compute_value_loss(value, critic_target, observation, action, expectile):
        with torch.no_grad():
            q1, q2 = critic_target(observation, action)
            q = torch.min(q1, q2).detach()

        v = value(observation)
        diff = q-v
        value_loss = asymmetric_l2_loss(diff, expectile)

        return value_loss


    @staticmethod
    def compute_actor_loss(actor, value, critic_target, observation, action, temperature):
        with torch.no_grad():
            v = value(observation)
            q1, q2 = critic_target(observation, action)
            q = torch.min(q1, q2)

        exp_a = torch.exp((q - v) * temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(observation.device))

        action, log_prob = actor.sample_action(observation)

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, log_prob.mean(), 0
    
    @staticmethod
    def compute_critic_loss(value, critic, observation, action, reward, mask, next_observation, gamma):

        with torch.no_grad():
            next_v = value(next_observation)
            backup = (reward + gamma * (1-mask) * next_v).detach()

        q1, q2 = critic(observation, action)

        critic_1_loss = ((q1 - backup)**2).mean()
        critic_2_loss = ((q2 - backup)**2).mean()
        critic_loss = critic_1_loss + critic_2_loss

        return critic_loss, q1.mean(), q2.mean(), backup.mean()
    
    @staticmethod
    @torch.no_grad()
    def update_target_param(model: NNMODEL, model_target: NNMODEL, tau: float) -> None:
        for param, param_target in zip(model.parameters(), model_target.parameters()):
            param_target.data.mul_(1 - tau)
            param_target.data.add_(tau * param.data)
