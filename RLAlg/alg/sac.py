import torch
import torch.nn as nn
import torch.nn.functional as F

NNMODEL = nn.Module

class SAC:
    @staticmethod
    def compute_actor_loss(actor_model: NNMODEL,
                           critic_model: NNMODEL,
                           observation: torch.Tensor,
                           alpha: float,
                           regularization_weight:float=0) -> torch.Tensor:
        action, log_prob, mean, log_std = actor_model(observation)
        q1, q2 = critic_model(observation, action)

        q = torch.min(q1, q2)
        actor_loss = (alpha * log_prob - q).mean()
        actor_loss += (mean.pow(2).mean() + log_std.pow(2).mean()) * regularization_weight

        return actor_loss

    @staticmethod
    def compute_critic_loss(actor_model: NNMODEL,
                            critic_model: NNMODEL,
                            critic_target_model: NNMODEL,
                            observation: torch.Tensor,
                            action: torch.Tensor,
                            reward: torch.Tensor,
                            next_observation: torch.Tensor,
                            done: torch.Tensor,
                            alpha: float,
                            gamma: float) -> torch.Tensor:

        q1, q2 = critic_model(observation, action)

        with torch.no_grad():
            next_action, next_log_prob, _, _ = actor_model(next_observation)
            q1_targ, q2_targ = critic_target_model(next_observation, next_action)

            q_targ = torch.min(q1_targ, q2_targ)
            backup = reward + gamma * (1 - done) * (q_targ - alpha * next_log_prob)

        critic_1_loss = F.mse_loss(q1, backup)
        critic_2_loss = F.mse_loss(q2, backup)
        critic_loss = critic_1_loss + critic_2_loss

        return critic_loss
    
    @staticmethod
    def compute_alpha_loss(actor_model: NNMODEL,
                           log_alpha: torch.Tensor,
                           observation: torch.Tensor,
                           target_entropy: float) -> torch.Tensor:
        _, log_prob, _, _ = actor_model(observation)
        alpha_loss = -(log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        return alpha_loss

    @staticmethod
    @torch.no_grad()
    def update_target_param(model:NNMODEL, model_target:NNMODEL, tau:float):
        for (param, param_target) in zip(model.parameters(), model_target.parameters()):
            param_target.data.mul_((1 - tau))
            param_target.data.add_(tau * param.data)