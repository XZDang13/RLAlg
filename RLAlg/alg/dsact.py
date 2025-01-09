import torch
import torch.nn as nn
import torch.nn.functional as F

NNMODEL = nn.Module

class DSACT:
    
    q1_mean_std = None
    q2_mean_std = None
    
    @staticmethod
    def compute_actor_loss(actor_model: NNMODEL,
                           critic_model: NNMODEL,
                           observation: torch.Tensor,
                           alpha: float,
                           regularization_weight:float=0) -> torch.Tensor:
        action, log_prob, mean, log_std = actor_model(observation)
        q1, q2, _, _, _, _ = critic_model(observation, action)

        q = torch.min(q1, q2)
        actor_loss = (alpha * log_prob - q).mean()
        actor_loss += (mean.pow(2).mean() + log_std.pow(2).mean()) * regularization_weight

        return actor_loss
    
    @staticmethod
    def compute_q_targ(reward: torch.tensor,
                       done: torch.tensor,
                       q: torch.tensor,
                       q_std: torch.tensor,
                       q_next: torch.tensor,
                       q_next_sample: torch.tensor,
                       next_log_prob: torch.tensor,
                       alpha: float,
                       gamma: float
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        q_targ = reward + (1 - done) * gamma * (
            q_next - alpha * next_log_prob
        )
        q_targ_sample = reward + (1 - done) * gamma * (
            q_next_sample - alpha * next_log_prob
        )
        
        td_boud = 3 * q_std
        difference = torch.clamp(q_targ_sample - q, -td_boud, td_boud)
        q_targ_bound = q + difference
        
        return q_targ, q_targ_bound
    
    @staticmethod
    def update_mean_std(q1_std:torch.tensor,
                        q2_std:torch.tensor,
                        tau_b: float):
        
        if DSACT.q1_mean_std is None:
            DSACT.q1_mean_std = q1_std.detach().mean()
        else:
            DSACT.q1_mean_std = (1-tau_b) * DSACT.q1_mean_std + tau_b * q1_std.detach().mean()
            
        if DSACT.q2_mean_std is None:
            DSACT.q2_mean_std = q2_std.detach().mean()
        else:
            DSACT.q2_mean_std = (1-tau_b) * DSACT.q2_mean_std + tau_b * q2_std.detach().mean()

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
                            gamma: float,
                            tau_b: float) -> torch.Tensor:

        q1, q2, q1_std, q2_std, _, _ = critic_model(observation, action)
        
        DSACT.update_mean_std(q1_std, q2_std, tau_b)

        with torch.no_grad():
            next_action, next_log_prob, _, _ = actor_model(next_observation)
            q1_next, q2_next, _, _, q1_next_sample, q2_next_sample = critic_target_model(next_observation, next_action)

            q_next = torch.min(q1_next, q2_next)
            q_next_sample = torch.where(q1_next < q2_next, q1_next_sample, q2_next_sample)
            
            q1_targ, q1_targ_bound = DSACT.compute_q_targ(reward, done, q1, DSACT.q1_mean_std, q_next, q_next_sample, next_log_prob, alpha, gamma)
            q2_targ, q2_targ_bound = DSACT.compute_q_targ(reward, done, q2, DSACT.q2_mean_std, q_next, q_next_sample, next_log_prob, alpha, gamma)
            
            q1_std_detach = torch.clamp(q1_std, min=0.).detach()
            q2_std_detach = torch.clamp(q2_std, min=0.).detach()
            bias = 1e-5
            
        critic_1_loss = (torch.pow(DSACT.q1_mean_std, 2) + bias) * torch.mean(
            -(q1_targ - q1).detach() / ( torch.pow(q1_std_detach, 2) + bias)*q1
            -((torch.pow(q1.detach() - q1_targ_bound, 2) - q1_std_detach.pow(2) )/ (torch.pow(q1_std_detach, 3) + bias)
            ) * q1_std
        )
        
        critic_2_loss = (torch.pow(DSACT.q2_mean_std, 2) + bias) * torch.mean(
            -(q2_targ - q2).detach() / ( torch.pow(q2_std_detach, 2) + bias) * q2
            -((torch.pow(q2.detach() - q2_targ_bound, 2) - q2_std_detach.pow(2) )/ (torch.pow(q2_std_detach, 3) + bias)
            ) * q2_std
        )
        
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