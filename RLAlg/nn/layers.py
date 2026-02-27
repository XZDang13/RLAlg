from typing import Callable, Optional, Union
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, TanhTransform, AffineTransform, TransformedDistribution, ComposeTransform
from .steps import DiscretePolicyStep, StochasticContinuousPolicyStep, DeterministicContinuousPolicyStep, ValueStep, DistributionStep
from ..utils import weight_init


class NormPosition(Enum):
    NONE = 0
    PRE = 1
    POST = 2

class MLPLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activate_func: Optional[nn.Module] = None,
        norm_position: NormPosition = NormPosition.NONE,
    ) -> None:
        super().__init__()
        if activate_func is None:
            activate_func = nn.Identity()
        if not isinstance(norm_position, NormPosition):
            raise TypeError(f"norm_position must be NormPosition, got {type(norm_position)}.")

        use_norm = norm_position is not NormPosition.NONE
        self.pre_norm = nn.LayerNorm(in_dim) if use_norm and norm_position is NormPosition.PRE else nn.Identity()
        self.post_norm = nn.LayerNorm(out_dim) if use_norm and norm_position is NormPosition.POST else nn.Identity()
        self.linear = nn.Linear(in_dim, out_dim, bias=not use_norm)
        self.activate_func = activate_func

        self.reset_parameters()

    def reset_parameters(self):
        weight_init(self.linear)
        if not isinstance(self.pre_norm, nn.Identity):
            self.pre_norm.reset_parameters()
        if not isinstance(self.post_norm, nn.Identity):
            self.post_norm.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.linear(x)
        x = self.post_norm(x)
        x = self.activate_func(x)
        return x
    
class Conv1DLayer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activate_func: Optional[nn.Module] = None,
        norm_position: NormPosition = NormPosition.NONE,
    ) -> None:
        super().__init__()
        if activate_func is None:
            activate_func = nn.Identity()
        if not isinstance(norm_position, NormPosition):
            raise TypeError(f"norm_position must be NormPosition, got {type(norm_position)}.")

        use_norm = norm_position is not NormPosition.NONE
        self.pre_norm = nn.InstanceNorm1d(in_channel) if use_norm and norm_position is NormPosition.PRE else nn.Identity()
        self.post_norm = nn.InstanceNorm1d(out_channel) if use_norm and norm_position is NormPosition.POST else nn.Identity()
        self.conv1d = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=not use_norm)
        self.activate_func = activate_func

        self.reset_parameters()

    def reset_parameters(self):
        weight_init(self.conv1d)
        if not isinstance(self.pre_norm, nn.Identity):
            self.pre_norm.reset_parameters()
        if not isinstance(self.post_norm, nn.Identity):
            self.post_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.conv1d(x)
        x = self.post_norm(x)
        x = self.activate_func(x)
        return x
    
class Conv2DLayer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activate_func: Optional[nn.Module] = None,
        norm_position: NormPosition = NormPosition.NONE,
    ) -> None:
        super().__init__()
        if activate_func is None:
            activate_func = nn.Identity()
        if not isinstance(norm_position, NormPosition):
            raise TypeError(f"norm_position must be NormPosition, got {type(norm_position)}.")

        use_norm = norm_position is not NormPosition.NONE
        self.pre_norm = nn.InstanceNorm2d(in_channel) if use_norm and norm_position is NormPosition.PRE else nn.Identity()
        self.post_norm = nn.InstanceNorm2d(out_channel) if use_norm and norm_position is NormPosition.POST else nn.Identity()
        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=not use_norm)
        self.activate_func = activate_func

        self.reset_parameters()

    def reset_parameters(self):
        weight_init(self.conv2d)
        if not isinstance(self.pre_norm, nn.Identity):
            self.pre_norm.reset_parameters()
        if not isinstance(self.post_norm, nn.Identity):
            self.post_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.conv2d(x)
        x = self.post_norm(x)
        x = self.activate_func(x)
        return x
    
class DeterministicHead(nn.Module):
    """
    Deterministic policy head.
    """
    def __init__(self, feature_dim: int, action_dim: int, max_action: Union[float, torch.Tensor, None] = None) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, action_dim)
        self.max_action = max_action

        if self.max_action is not None:
            self.transform = ComposeTransform(
                [
                    TanhTransform(),
                    AffineTransform(loc=0, scale=max_action)
                ],
                cache_size=1
            )
        else:
            self.transform = []

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.linear.weight, -3e-3, 3e-3)
        if self.linear.bias is not None:
            nn.init.uniform_(self.linear.bias, -3e-3, 3e-3)


    def forward(self, x:torch.Tensor, std:float=1.0) -> DeterministicContinuousPolicyStep:
        """
        Args:
            x: input features

        Returns:
            deterministic action in [-max_action, max_action]
        """
        mu = self.linear(x)
        base_pi = Normal(mu, std)
        pi = TransformedDistribution(base_pi, self.transform)
        if self.max_action is not None:
            mu_squashed = self.max_action * torch.tanh(mu)
        else:
            mu_squashed = mu

        step = DeterministicContinuousPolicyStep(pi, mu_squashed)
        return step

class GaussianHead(nn.Module):
    """
    Gaussian policy head.
    Supports state-independent or state-dependent log_std.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        log_std:float = 0,
        log_std_min: float = -20,
        log_std_max: float = 2,
        learnable_log_std: bool = True,
        max_action: Union[float,torch.Tensor, None] = None,
        state_dependent_std: bool = False,
    ) -> None:
        super().__init__()

        self.mu_layer = nn.Linear(feature_dim, action_dim)

        self.state_dependent_std = state_dependent_std
        if state_dependent_std:
            self.log_std_layer = nn.Linear(feature_dim, action_dim)
        else:
            if learnable_log_std:
                self.log_std = nn.Parameter(torch.ones(action_dim)*log_std)
            else:
                self.register_buffer("log_std", torch.ones(action_dim) * log_std)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = max_action

        if self.max_action is not None:
            self.transform = ComposeTransform(
                [
                    TanhTransform(),
                    AffineTransform(loc=0, scale=max_action)
                ],
                cache_size=1
            )
        else:
            self.transform = []

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.mu_layer.weight, -3e-3, 3e-3)
        if self.mu_layer.bias is not None:
            nn.init.uniform_(self.mu_layer.bias, -3e-3, 3e-3)

        if self.state_dependent_std:
            nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
            if self.log_std_layer.bias is not None:
                nn.init.uniform_(self.log_std_layer.bias, -3e-3, 3e-3)

    def forward(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> StochasticContinuousPolicyStep:
        """
        Args:
            x: input features
            action: optional, if None sample from policy

        Returns:
            pi: TransformedDistribution
            action: [B, action_dim]
            log_prob: [B]
            mu: [B, action_dim]
            log_std: [B, action_dim]
        """
        mu = self.mu_layer(x)

        if self.state_dependent_std:
            log_std = self.log_std_layer(x)
            log_std = torch.tanh(log_std)
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        else:
            log_std = self.log_std.expand_as(mu)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = torch.exp(log_std)

        base_pi = Normal(mu, std)
        pi = TransformedDistribution(base_pi, self.transform)

        sampled_action = False
        if action is None:
            action = pi.rsample()
            sampled_action = True
            
        if self.max_action is not None:
            eps = 1e-6
            action = action.clamp(-self.max_action + eps, self.max_action - eps)

        log_prob = pi.log_prob(action).sum(axis=-1)

        # deterministic mu after squashing if needed
        if self.max_action is not None:
            mu_squashed = self.max_action * torch.tanh(mu)
            entropy_action = action if sampled_action else pi.rsample()
            entropy = -pi.log_prob(entropy_action).sum(axis=-1)
        else:
            mu_squashed = mu
            entropy = base_pi.entropy().sum(axis=-1)

        step = StochasticContinuousPolicyStep(pi, action, log_prob, mu_squashed, log_std, entropy)

        return step

class CategoricalHead(nn.Module):
    """
    Categorical (discrete) policy head.
    """
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.logit_layer = nn.Linear(feature_dim, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.logit_layer.weight, -3e-3, 3e-3)
        if self.logit_layer.bias is not None:
            nn.init.uniform_(self.logit_layer.bias, -3e-3, 3e-3)

    def forward(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> DiscretePolicyStep:
        """
        Args:
            x: input features
            action: optional, if None sample from policy

        Returns:
            distribution, action, log_prob
        """
        logits = self.logit_layer(x)
        pi = Categorical(logits=logits)
        if action is None:
            action = pi.sample()
        log_prob = pi.log_prob(action)
        entropy = pi.entropy()
        step = DiscretePolicyStep(pi, action, log_prob, entropy)

        return step


class CriticHead(nn.Module):
    """
    Value function head.
    """
    def __init__(self, feature_dim: int, value_dim: int=1, squeeze: bool=True) -> None:
        super().__init__()
        self.critic_layer = nn.Linear(feature_dim, value_dim)
        self.value_dim = value_dim
        self.squeeze = squeeze
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.critic_layer.weight, -3e-3, 3e-3)
        if self.critic_layer.bias is not None:
            nn.init.uniform_(self.critic_layer.bias, -3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> ValueStep:
        """
        Args:
            x: input features

        Returns:
            scalar value
        """
        value = self.critic_layer(x)
        if self.squeeze:
            value = value.squeeze(-1)

        step = ValueStep(value=value)
        
        return step


class DistributeCriticHead(nn.Module):
    """
    Distributional critic head: outputs mean, std, and a sample.
    """
    def __init__(self, feature_dim:int, log_std_min:float = -10, log_std_max:float = 2,) -> None:
        super().__init__()
        self.mu_layer = nn.Linear(feature_dim, 1)
        self.log_std_layer = nn.Linear(feature_dim, 1)
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

    def forward(self, x: torch.Tensor) -> DistributionStep:
        """
        Args:
            x: input features

        Returns:
            mean, std, sampled value
        """
        mu = self.mu_layer(x).squeeze(-1)
        log_std = self.log_std_layer(x).squeeze(-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std) + 1e-7
        pi = Normal(mu, std)
        sample = pi.rsample()

        step = DistributionStep(pi, mu, std, sample)
        return step

def make_mlp_layers(in_dim:int, layer_dims:list[int], activate_function:Callable[[torch.Tensor], torch.Tensor], norm_position:NormPosition) -> tuple[nn.Sequential, int]:
    layers = []

    for dim in layer_dims:
        mlp = MLPLayer(in_dim, dim, activate_function, norm_position=norm_position)
        in_dim = dim

        layers.append(mlp)

    return nn.Sequential(*layers), in_dim
