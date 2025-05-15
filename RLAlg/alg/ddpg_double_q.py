import torch
import torch.nn as nn
import torch.nn.functional as F

NNMODEL = nn.Module

class DDPGDoubleQ:
    @staticmethod
    def compute_critic_loss(actor_model: NNMODEL,
                            critic_model: NNMODEL,
                            critic_target_model: NNMODEL,
                            observation: torch.Tensor,
                            action: torch.Tensor,
                            reward: torch.Tensor,
                            next_observation: torch.Tensor,
                            done: torch.Tensor,
                            std: torch.Tensor,
                            gamma: float = 0.99) -> torch.Tensor:
        
        with torch.no_grad():
            dist = actor_model(next_observation, std)
            next_action = dist.sample(0.3)
            next_qvalue_target_1, next_qvalue_target_2 = critic_target_model(next_observation, next_action)
            next_qvalue_target = torch.min(next_qvalue_target_1, next_qvalue_target_2)
            q_targ = reward + gamma * (1 - done) * next_qvalue_target

        qvalue_1, qvalue_2 = critic_model(observation, action)
        critic_loss = F.mse_loss(qvalue_1, q_targ) + F.mse_loss(qvalue_2, q_targ)

        return critic_loss
    
    @staticmethod
    def compute_actor_loss(actor_model: NNMODEL,
                           critic_model: NNMODEL,
                           observation: torch.Tensor,
                           std: torch.Tensor,
                           regularization_weight:float=0) -> torch.Tensor:
        dist = actor_model(observation, std)
        action = dist.sample(0.3)
        q_value_1, q_value_2 = critic_model(observation, action)
        q_value = torch.min(q_value_1, q_value_2)
        actor_loss = -q_value.mean()
        actor_loss += dist.mean.pow(2).mean() * regularization_weight

        return actor_loss
    
    @staticmethod
    def compute_actor_loss_with_multi_critic(actor_model: NNMODEL,
                           critic_models: list[NNMODEL],
                           weights: list[float],
                           observation: torch.Tensor,
                           std: torch.Tensor,
                           regularization_weight:float=0) -> torch.Tensor:
        dist = actor_model(observation, std)
        action = dist.sample(0.3)

        actor_loss = 0
        for weight, critic_model in zip(weights, critic_models):
            q_value_1, q_value_2 = critic_model(observation, action)
            q_value = torch.min(q_value_1, q_value_2)
            actor_loss += -q_value.mean() * weight

        actor_loss += dist.mean.pow(2).mean() * regularization_weight

        return actor_loss
    
    @staticmethod
    def compute_critic_loss_asymmetric(actor_model: NNMODEL,
                            critic_model: NNMODEL,
                            critic_target_model: NNMODEL,
                            critic_observation: torch.Tensor,
                            action: torch.Tensor,
                            reward: torch.Tensor,
                            next_actor_observation:torch.Tensor,
                            next_critic_observation:torch.Tensor,
                            done: torch.Tensor,
                            std: torch.Tensor,
                            gamma: float = 0.99) -> torch.Tensor:
        
        with torch.no_grad():
            dist = actor_model(critic_observation, std)
            next_action = dist.sample(0.3)
            next_qvalue_target_1, next_qvalue_target_2 = critic_target_model(next_actor_observation, next_action)
            next_qvalue_target = torch.min(next_qvalue_target_1, next_qvalue_target_2)
            q_targ = reward + gamma * (1 - done) * next_qvalue_target

        qvalue_1, qvalue_2 = critic_model(next_critic_observation, action)
        critic_loss = F.mse_loss(qvalue_1, q_targ) + F.mse_loss(qvalue_2, q_targ)

        return critic_loss
    
    @staticmethod
    def compute_actor_loss_asymmetric(actor_model: NNMODEL,
                           critic_model: NNMODEL,
                           actor_observation: torch.Tensor,
                           critic_observation: torch.Tensor,
                           std: torch.Tensor,
                           regularization_weight:float=0) -> torch.Tensor:
        dist = actor_model(actor_observation, std)
        action = dist.sample(0.3)
        q_value_1, q_value_2 = critic_model(critic_observation, action)
        q_value = torch.min(q_value_1, q_value_2)
        actor_loss = -q_value.mean()
        actor_loss += dist.mean.pow(2).mean() * regularization_weight

        return actor_loss
    
    @staticmethod
    def compute_actor_loss_asymmetric_with_multi_critic(actor_model: NNMODEL,
                           critic_models: list[NNMODEL],
                           weights: list[float],
                           actor_observation: torch.Tensor,
                           critic_observation: torch.Tensor,
                           std: torch.Tensor,
                           regularization_weight:float=0) -> torch.Tensor:
        dist = actor_model(actor_observation, std)
        action = dist.sample(0.3)

        actor_loss = 0
        for weight, critic_model in zip(weights, critic_models):
            q_value_1, q_value_2 = critic_model(critic_observation, action)
            q_value = torch.min(q_value_1, q_value_2)
            actor_loss += -q_value.mean() * weight
            
        actor_loss += dist.mean.pow(2).mean() * regularization_weight

        return actor_loss
    
    @staticmethod
    @torch.no_grad()
    def update_target_param(model: NNMODEL, model_target: NNMODEL, tau: float) -> None:
        for param, param_target in zip(model.parameters(), model_target.parameters()):
            param_target.data.mul_(1 - tau)
            param_target.data.add_(tau * param.data)