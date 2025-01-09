import torch
import torch.nn as nn
import torch.nn.functional as F

NNMODEL = nn.Module

class DSAC:
    
    @staticmethod
    def compute_actor_loss(actor_model: NNMODEL,
                           critic_model: NNMODEL,
                           observation: torch.Tensor,
                           alpha: float,
                           regularization_weight:float=0) -> torch.Tensor:
        action, log_prob, mean, log_std = actor_model(observation)
        q, _, _ = critic_model(observation, action)

        actor_loss = (alpha * log_prob - q).mean()
        actor_loss += (mean.pow(2).mean() + log_std.pow(2).mean()) * regularization_weight

        return actor_loss
    
    @staticmethod
    def compute_q_targ(reward: torch.tensor,
                       done: torch.tensor,
                       q: torch.tensor,
                       q_next: torch.tensor,
                       next_log_prob: torch.tensor,
                       alpha: float,
                       gamma: float,
                       td_bound: float
                    ):
        
        target_q = reward + (1 - done) * gamma * (
            q_next - alpha * next_log_prob
        )
        
        difference = torch.clamp(target_q - q, -td_bound, td_bound)
        
        target_q_bound = q + difference
        
        return target_q, target_q_bound

    @staticmethod
    def compute_critic_loss(actor_target_model: NNMODEL,
                            critic_model: NNMODEL,
                            critic_target_model: NNMODEL,
                            observation: torch.Tensor,
                            action: torch.Tensor,
                            reward: torch.Tensor,
                            next_observation: torch.Tensor,
                            done: torch.Tensor,
                            alpha: float,
                            gamma: float,
                            td_bound: float) -> torch.Tensor:

        q, q_std, _ = critic_model(observation, action)
        
        with torch.no_grad():
            next_action, next_log_prob, _, _ = actor_target_model(next_observation)
            
            _, _, q_next_sample = critic_target_model(next_observation, next_action)
            
            q_targ, q_targ_bound = DSAC.compute_q_targ(reward, done, q.detach(), q_next_sample.detach(), next_log_prob.detach(), alpha, gamma, td_bound)
            
        q_std_detach = torch.clamp(q_std, min=0).detach()
        bias = 1e-5
        
        critic_loss = torch.mean(
            -(q_targ - q).detach() / (torch.pow(q_std_detach, 2) + bias) * q 
            -((torch.pow(q.detach() - q_targ_bound, 2) - q_std_detach.pow(2)) / (torch.pow(q_std_detach, 3) + bias)) * q_std
        )
            
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