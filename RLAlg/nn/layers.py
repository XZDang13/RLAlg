import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Distribution
from ..distribution import TruncatedNormal
from ..utils import weight_init


class MLPLayer(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, activate_func:callable[[torch.Tensor], torch.Tensor]=F.relu, norm:bool=False) -> None:
        super(MLPLayer, self).__init__()
        bias = not norm
        self.linear = nn.Linear(in_dim, out_dim, bias)
        self.norm = nn.LayerNorm(out_dim) if norm else None

        self.activate_func = activate_func

        self.reset_parameters()

    def reset_parameters(self):
        weight_init(self.linear)
        if self.norm:
            self.norm.reset_parameters()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)

        if self.activate_func:
            x = self.activate_func(x)
        
        return x
    
class Conv1DLayer(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, kernel_size:int=3, stride:int=1,
                 padding:int=1, activate_func:callable[[torch.Tensor], torch.Tensor]=F.relu, norm:bool=False) -> None:
        super(Conv1DLayer, self).__init__()
        bias = not norm
        self.conv1d = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.InstanceNorm1d(out_channel) if norm else None

        self.activate_func = activate_func

        self.reset_parameters()

    def reset_parameters(self):
        weight_init(self.conv1d)
        if self.norm:
            self.norm.reset_parameters()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        if self.norm:
            x = self.norm(x)

        if self.activate_func:
            x = self.activate_func(x)
        
        return x
    
class Conv2DLayer(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, kernel_size:int=3, stride:int=1,
                 padding:int=1, activate_func:callable[[torch.Tensor], torch.Tensor]=F.relu, norm:bool=False) -> None:
        super(Conv2DLayer, self).__init__()
        bias = not norm
        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.InstanceNorm2d(out_channel) if norm else None

        self.activate_func = activate_func

        self.reset_parameters()

    def reset_parameters(self):
        weight_init(self.conv2d)
        if self.norm:
            self.norm.reset_parameters()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)

        if self.activate_func:
            x = self.activate_func(x)
        
        return x
    
class DeterminicHead(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, max_action:float=1.0) -> None:
        super().__init__()
        self.max_action = max_action
        self.linear = nn.Linear(feature_dim, action_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.linear.weight, -3e-3, 3e-3)
        if self.linear.bias is not None:
            nn.init.uniform_(self.linear.bias, -3e-3, 3e-3)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.max_action * torch.tanh(x)

        return x
    
class GuassianHead(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, max_action:float=1) -> None:
        super().__init__()

        self.mu_layer = nn.Linear(feature_dim, action_dim)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        self.max_action = max_action    
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.mu_layer.weight, -3e-3, 3e-3)
        if self.mu_layer.bias is not None:
            nn.init.uniform_(self.mu_layer.bias, -3e-3, 3e-3)

    def forward(self, x:torch.Tensor, action:torch.Tensor|None) -> tuple[Distribution, torch.Tensor|None]:
        mu = self.mu_layer(x)
        mu = self.max_action * torch.tanh(mu)
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)
        if action is None:
            action = pi.sample()

        log_prob = pi.log_prob(action).sum(axis=-1)
        
        return pi, action, log_prob

class SquashedGaussianHead(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int,
                 max_action:float=1.0, log_std_min:float=-20,
                 log_std_max:float=2) -> None:
        super().__init__()

        self.mu_layer = nn.Linear(feature_dim, action_dim)
        self.log_std_layer = nn.Linear(feature_dim, action_dim)
        self.max_action = max_action

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.mu_layer.weight, -3e-3, 3e-3)
        if self.mu_layer.bias is not None:
            nn.init.uniform_(self.mu_layer.bias, -3e-3, 3e-3)

        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        if self.log_std_layer.bias is not None:
            nn.init.uniform_(self.log_std_layer.bias, -3e-3, 3e-3)

    def forward(self, x:torch.Tensor, deterministic:bool=False,
                with_logprob:bool=True) -> tuple[torch.Tensor, torch.Tensor|None,
                                                 torch.Tensor, torch.Tensor]:
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std)
        pi = Normal(mu, std)
        if deterministic:
            x = mu
        else:
            x = pi.rsample()

        x_tanh = torch.tanh(x)

        log_prob = None
        if with_logprob:
            log_prob = pi.log_prob(x)
            log_prob -= torch.log(self.max_action * (1 - x_tanh.pow(2)) + 1e-6)
            log_prob = log_prob.sum(-1)

        action = torch.tanh(x)
        action = self.max_action * action

        return action, log_prob, mu, log_std

class CategoricalHead(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int) -> None:
        super().__init__()

        self.logit_layer = nn.Linear(feature_dim, action_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.logit_layer.weight, -3e-3, 3e-3)
        if self.logit_layer.bias is not None:
            nn.init.uniform_(self.logit_layer.bias, -3e-3, 3e-3)

    def forward(self, x:torch.Tensor, action:torch.Tensor|None) -> tuple[Distribution, torch.Tensor|None]:
        logits = self.logit_layer(x)
        pi = Categorical(logits=logits)
        
        if action is None:
            action = pi.sample()
        
        log_prob = pi.log_prob(action)
        
        return pi, action, log_prob
    
class CriticHead(nn.Module):
    def __init__(self, feature_dim:int) -> None:
        super().__init__()

        self.critic_layer = nn.Linear(feature_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.critic_layer.weight, -3e-3, 3e-3)
        if self.critic_layer.bias is not None:
            nn.init.uniform_(self.critic_layer.bias, -3e-3, 3e-3)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        value = self.critic_layer(x)

        return value.squeeze(-1)
    
class DistributeCriticHead(nn.Module):
    def __init__(self, feature_dim:int) -> None:
        super().__init__()

        self.mu_layer = nn.Linear(feature_dim, 1)
        self.log_std_layer = nn.Linear(feature_dim, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.mu_layer.weight, -3e-3, 3e-3)
        if self.mu_layer.bias is not None:
            nn.init.uniform_(self.mu_layer.bias, -3e-3, 3e-3)
            
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        if self.log_std_layer.bias is not None:
            nn.init.uniform_(self.log_std_layer.bias, -3e-3, 3e-3)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        q_mu = self.mu_layer(x).squeeze(-1)
        log_std = self.log_std_layer(x).squeeze(-1)
        log_std = torch.tanh(log_std)
        
        q_std = torch.exp(log_std)
        
        dist = Normal(q_mu, q_std)
        q_sample = dist.rsample()
        
        return q_mu, q_std, q_sample

def make_mlp_layers(in_dim:int, layer_dims:list[int], activate_function:callable[[torch.Tensor], torch.Tensor], norm:bool) -> tuple[nn.Sequential, int]:
    layers = []

    for dim in layer_dims:
        mlp = MLPLayer(in_dim, dim, activate_function, norm)
        in_dim = dim

        layers.append(mlp)

    return nn.Sequential(*layers), in_dim