from typing import Callable, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Distribution
from ..utils import weight_init


class MLPLayer(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, activate_func:Callable[[torch.Tensor], torch.Tensor]=F.relu, norm:bool=False) -> None:
        super(MLPLayer, self).__init__()
        bias = not norm
        self.linear = nn.Linear(in_dim, out_dim, bias)
        self.norm = nn.LayerNorm(out_dim) if norm else None

        self.activate_func = activate_func

        self.apply(weight_init)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)

        x = self.activate_func(x)
        
        return x
    
class DeterminicHead(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, max_action:float=1.0) -> None:
        super().__init__()
        self.max_action = max_action
        self.linear = nn.Linear(feature_dim, action_dim)

        self.apply(weight_init)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.max_action * torch.tanh(x)

        return x
    
class GuassianHead(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int) -> None:
        super().__init__()

        self.mu_layer = nn.Linear(feature_dim, action_dim)
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.apply(weight_init)

    def forward(self, x:torch.Tensor, action:Optional[torch.Tensor]) -> Tuple[Distribution, Optional[torch.Tensor]]:
        mu = self.mu_layer(x)
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)
        log_porb = None
        if action:
            log_porb = pi.log_prob(action).sum(axis=-1)

        return pi, log_porb

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

        self.apply(weight_init)

    def forward(self, x:torch.Tensor, deterministic:bool=False,
                with_logprob:bool=True) -> Tuple[torch.Tensor, Optional[torch.Tensor],
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
            log_prob = log_prob.sum(1)

        action = torch.tanh(x)
        action = self.max_action * action

        return action, log_prob, mu, log_std

class CategoricalHead(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int) -> None:
        super().__init__()

        self.logit_layer = nn.Linear(feature_dim, action_dim)

        self.apply(weight_init)

    def forward(self, x:torch.Tensor, action:Optional[torch.Tensor]) -> Tuple[Distribution, Optional[torch.Tensor]]:
        logits = self.logit_layer(x)
        pi = Categorical(logits=logits)
        log_porb = None
        if action:
            log_porb = pi.log_prob(action)

        return pi, log_porb
    
class CriticHead(nn.Module):
    def __init__(self, feature_dim:int) -> None:
        super().__init__()

        self.critic_layer = nn.Linear(feature_dim, 1)

        self.apply(weight_init)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        value = self.critic_layer(x)

        return value.squeeze(-1)
