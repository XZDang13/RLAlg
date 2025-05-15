import torch
import torch.nn as nn
import torch.nn.functional as F

NNMODEL = nn.Module

class DDPG:
    @staticmethod
    def compute_critic_loss(actor_target_model:NNMODEL,
                            critic_model:NNMODEL,
                            critic_target_model:NNMODEL,
                            observation:torch.Tensor,
                            action:torch.Tensor,
                            reward:torch.Tensor,
                            next_observation:torch.Tensor,
                            done:torch.Tensor,
                            gamma:float=0.99) -> torch.Tensor:
        
        qvalue = critic_model(observation, action)
        with torch.no_grad():
            next_action = actor_target_model(next_observation)
            next_qvalue_target = critic_target_model(next_observation, next_action).detach()
            q_targ = reward + gamma * (1 - done) * next_qvalue_target

        critic_loss = ((qvalue-q_targ)**2).mean()

        return critic_loss

    @staticmethod
    def compute_actor_loss_with_multi_critic(actor_model:NNMODEL, critic_models:list[NNMODEL], weights:list[float],
                           observation:torch.Tensor, regularization_weight:float=0):
        action = actor_model(observation)
        actor_loss = 0
        for weight, critic_model in zip(weights, critic_models):
            q_value = critic_model(observation, action) * weight
            actor_loss += -q_value.mean() * weight
        actor_loss += (action / 1).pow(2).mean() * regularization_weight

        return actor_loss

    @staticmethod
    def compute_actor_loss(actor_model:NNMODEL, critic_model:NNMODEL,
                           observation:torch.Tensor, regularization_weight:float=0):
        action = actor_model(observation)
        q_value = critic_model(observation, action)
        actor_loss = -q_value.mean()
        actor_loss += (action / 1).pow(2).mean() * regularization_weight

        return actor_loss
    
    @staticmethod
    def compute_critic_loss_asymmetric(actor_target_model:NNMODEL,
                            critic_model:NNMODEL,
                            critic_target_model:NNMODEL,
                            critic_observation:torch.Tensor,
                            action:torch.Tensor,
                            reward:torch.Tensor,
                            next_actor_observation:torch.Tensor,
                            next_critic_observation:torch.Tensor,
                            done:torch.Tensor,
                            gamma:float=0.99) -> torch.Tensor:
        
        qvalue = critic_model(critic_observation, action)
        with torch.no_grad():
            next_action = actor_target_model(next_actor_observation)
            next_qvalue_target = critic_target_model(next_critic_observation, next_action).detach()
            q_targ = reward + gamma * (1 - done) * next_qvalue_target

        critic_loss = ((qvalue-q_targ)**2).mean()

        return critic_loss
    
    @staticmethod
    def compute_actor_loss_asymmetric_with_multi_critic(actor_model:NNMODEL, critic_models:list[NNMODEL], weights:list[float],
                           actor_observation:torch.Tensor, critic_observation:torch.Tensor, regularization_weight:float=0):
        action = actor_model(actor_observation)
        actor_loss = 0
        for weight, critic_model in zip(weights, critic_models):
            q_value = critic_model(critic_observation, action) * weight
            actor_loss += -q_value.mean() * weight
        actor_loss += (action / 1).pow(2).mean() * regularization_weight

        return actor_loss

    @staticmethod
    def compute_actor_loss_asymmetric(actor_model:NNMODEL, critic_model:NNMODEL,
                           actor_observation:torch.Tensor, critic_observation:torch.Tensor, regularization_weight:float=0):
        action = actor_model(actor_observation)
        q_value = critic_model(critic_observation, action)
        actor_loss = -q_value.mean()
        actor_loss += (action / 1).pow(2).mean() * regularization_weight

        return actor_loss
    
    @staticmethod
    @torch.no_grad()
    def update_target_param(model:NNMODEL, model_target:NNMODEL, tau:float) -> None:
        for (param, param_target) in zip(model.parameters(), model_target.parameters()):
            param_target.data.mul_((1 - tau))
            param_target.data.add_(tau * param.data)