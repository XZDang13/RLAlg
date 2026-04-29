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
    
def make_mlp_layers(in_dim:int, layer_dims:list[int], activate_function:Callable[[torch.Tensor], torch.Tensor], norm_position:NormPosition) -> tuple[nn.Sequential, int]:
    layers = []

    for dim in layer_dims:
        mlp = MLPLayer(in_dim, dim, activate_function, norm_position=norm_position)
        in_dim = dim

        layers.append(mlp)

    return nn.Sequential(*layers), in_dim

class GRULayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            batch_first=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _prepare_hidden(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = x.shape[1]
        if hidden_state is None:
            return torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=x.dtype,
                device=x.device,
            )
        if hidden_state.shape != (self.num_layers, batch_size, self.hidden_size):
            raise ValueError(
                "hidden_state must have shape "
                f"({self.num_layers}, {batch_size}, {self.hidden_size}), got {tuple(hidden_state.shape)}."
            )
        return hidden_state.to(dtype=x.dtype, device=x.device)

    @staticmethod
    def _to_reset_mask(mask: torch.Tensor, batch_size: int, timestep: int) -> torch.Tensor:
        if mask.ndim != 1 or mask.shape[0] != batch_size:
            raise ValueError(
                f"episode_starts[{timestep}] must have shape ({batch_size},), got {tuple(mask.shape)}."
            )
        if mask.dtype != torch.bool:
            mask = mask != 0
        return mask

    def _forward_time_major(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor],
        episode_starts: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_steps, batch_size = x.shape[:2]
        hidden = self._prepare_hidden(x, hidden_state)
        outputs = []

        for t in range(time_steps):
            if episode_starts is not None:
                reset_mask = self._to_reset_mask(episode_starts[t], batch_size, t).to(x.device)
                if reset_mask.any():
                    keep_mask = (~reset_mask).to(dtype=hidden.dtype, device=x.device).view(1, batch_size, 1)
                    hidden = hidden * keep_mask
            out_t, hidden = self.gru(x[t : t + 1], hidden)
            outputs.append(out_t)

        return torch.cat(outputs, dim=0), hidden

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim not in (2, 3):
            raise ValueError(f"x must have shape [B, D] or [T, B, D], got {tuple(x.shape)}.")

        if x.ndim == 2:
            x_seq = x.unsqueeze(0)
            starts_seq = None
            if episode_starts is not None:
                if episode_starts.ndim != 1:
                    raise ValueError(
                        f"episode_starts must have shape [B] for single-step input, got {tuple(episode_starts.shape)}."
                    )
                starts_seq = episode_starts.unsqueeze(0)
            output_seq, next_hidden = self._forward_time_major(x_seq, hidden_state, starts_seq)
            return output_seq.squeeze(0), next_hidden

        if self.batch_first:
            x_time_major = x.transpose(0, 1)
            starts_time_major = None
            if episode_starts is not None:
                if episode_starts.ndim != 2 or episode_starts.shape[:2] != x.shape[:2]:
                    raise ValueError(
                        f"episode_starts must match [B, T] shape {tuple(x.shape[:2])}, got {tuple(episode_starts.shape)}."
                    )
                starts_time_major = episode_starts.transpose(0, 1)
            output_time_major, next_hidden = self._forward_time_major(x_time_major, hidden_state, starts_time_major)
            return output_time_major.transpose(0, 1), next_hidden

        starts_time_major = None
        if episode_starts is not None:
            if episode_starts.ndim != 2 or episode_starts.shape[:2] != x.shape[:2]:
                raise ValueError(
                    f"episode_starts must match [T, B] shape {tuple(x.shape[:2])}, got {tuple(episode_starts.shape)}."
                )
            starts_time_major = episode_starts
        return self._forward_time_major(x, hidden_state, starts_time_major)
    
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
        eps = 1e-6

        if self.state_dependent_std:
            log_std = self.log_std_layer(x)
            log_std = torch.tanh(log_std)
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        else:
            log_std = self.log_std.expand_as(mu)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = torch.exp(log_std)

        base_pi = Normal(mu, std)
        if self.max_action is not None:
            max_action = torch.as_tensor(self.max_action, dtype=mu.dtype, device=mu.device)
            max_action = torch.clamp(torch.abs(max_action), min=eps)
            transform = ComposeTransform(
                [
                    TanhTransform(),
                    AffineTransform(loc=0, scale=max_action),
                ],
                cache_size=1,
            )
        else:
            max_action = None
            transform = self.transform

        pi = TransformedDistribution(base_pi, transform)

        sampled_action = False
        if action is None:
            sampled_action = True
            if max_action is not None:
                pre_tanh_action = base_pi.rsample()
                squashed_action = torch.tanh(pre_tanh_action)
                action = squashed_action * max_action
            else:
                action = pi.rsample()
                pre_tanh_action = None
                squashed_action = None
        else:
            pre_tanh_action = None
            squashed_action = None

        if max_action is not None:
            if pre_tanh_action is None:
                scaled_action = action / max_action
                scaled_action = scaled_action.clamp(-1 + eps, 1 - eps)
                action = scaled_action * max_action
                pre_tanh_action = 0.5 * (torch.log1p(scaled_action) - torch.log1p(-scaled_action))
                squashed_action = scaled_action
            log_det = torch.log(max_action * (1 - squashed_action.pow(2)) + eps)
            log_prob = (base_pi.log_prob(pre_tanh_action) - log_det).sum(axis=-1)
        else:
            log_prob = pi.log_prob(action).sum(axis=-1)

        # deterministic mu after squashing if needed
        if max_action is not None:
            mu_squashed = max_action * torch.tanh(mu)
            if sampled_action:
                entropy = -log_prob
            else:
                entropy_pre_tanh = base_pi.rsample()
                entropy_squashed = torch.tanh(entropy_pre_tanh)
                entropy_log_det = torch.log(max_action * (1 - entropy_squashed.pow(2)) + eps)
                entropy = -(base_pi.log_prob(entropy_pre_tanh) - entropy_log_det).mean()
        else:
            mu_squashed = mu
            entropy = base_pi.entropy().mean()

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

class DiffusionHead(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, scale:float=1.0) -> None:
        super().__init__()
        
        self.predict_layer = nn.Linear(in_dim, action_dim)
        self.scale = scale
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.predict_layer.weight, -3e-3, 3e-3)
        if self.predict_layer.bias is not None:
            nn.init.uniform_(self.predict_layer.bias, -3e-3, 3e-3)
        
    def forward(self, x: torch.Tensor) -> ValueStep:
        value = self.predict_layer(x) * self.scale

        step = ValueStep(value=value)
        
        return step
