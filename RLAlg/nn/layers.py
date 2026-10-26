from typing import Callable, Optional, Union
from enum import Enum
import math
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
    Deterministic policy head, optionally squashed into ``[-max_action, max_action]`` by tanh.

    Exploration noise is left to the caller; ``DDPGDoubleQ.perturb_action`` adds it before
    the tanh using the step's ``pre_tanh_mean``.
    """
    def __init__(self, feature_dim: int, action_dim: int, max_action: Union[float, torch.Tensor, None] = None) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, action_dim)
        self.max_action = max_action

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.linear.weight, -3e-3, 3e-3)
        if self.linear.bias is not None:
            nn.init.uniform_(self.linear.bias, -3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> DeterministicContinuousPolicyStep:
        """
        Args:
            x: input features

        Returns:
            deterministic action in [-max_action, max_action]
        """
        mu = self.linear(x)
        if self.max_action is None:
            return DeterministicContinuousPolicyStep(mu)
        max_action = torch.as_tensor(self.max_action, dtype=mu.dtype, device=mu.device)
        return DeterministicContinuousPolicyStep(max_action * torch.tanh(mu), mu, max_action)

class GaussianHead(nn.Module):
    """
    Gaussian policy head, optionally squashed into ``[-max_action, max_action]`` by tanh.

    The standard deviation is configured in std units for every mode:
    ``init_std`` is the initial value and ``[std_min, std_max]`` the allowed range.

    - ``state_dependent_std=True``: a linear layer predicts ``log_std``, squashed by tanh
      into ``[log(std_min), log(std_max)]``.
    - otherwise a state-independent ``log_std`` (``std_parameterization="log_std"``) or
      ``std`` (``"std"``) vector, learnable unless ``learnable_std=False``.

    With tanh squashing the entropy has no closed form; ``entropy`` is the one-sample
    estimate ``-log_prob(a')`` with ``a'`` the sampled action or, when an action is
    supplied (as in PPO updates), a fresh reparameterized sample from the current policy.
    This has much lower variance than adding a sampled log-det term to the analytic
    base entropy, because the two terms of ``log_prob`` largely cancel.
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        *,
        max_action: Union[float, torch.Tensor, None] = None,
        state_dependent_std: bool = False,
        std_parameterization: str = "log_std",
        init_std: float = 1.0,
        std_min: float = math.exp(-20.0),
        std_max: float = math.exp(2.0),
        learnable_std: bool = True,
        validate_args: bool | None = None,
    ) -> None:
        super().__init__()
        # None preserves PyTorch's default. Trusted GPU training batches may
        # opt out of distribution checks that otherwise read scalars on CPU.
        self.validate_args = validate_args
        if std_parameterization not in {"log_std", "std"}:
            raise ValueError("std_parameterization must be 'log_std' or 'std'.")
        if state_dependent_std and std_parameterization != "log_std":
            raise ValueError("Direct std is only supported for state-independent noise.")
        if not all(math.isfinite(v) for v in (init_std, std_min, std_max)):
            raise ValueError("init_std, std_min and std_max must be finite.")
        if not 0.0 < std_min <= init_std <= std_max:
            raise ValueError("The std range requires 0 < std_min <= init_std <= std_max.")
        self.std_parameterization = std_parameterization
        self.state_dependent_std = state_dependent_std
        self.std_min = float(std_min)
        self.std_max = float(std_max)
        self.log_std_min = math.log(std_min)
        self.log_std_max = math.log(std_max)
        self.max_action = max_action

        self.mu_layer = nn.Linear(feature_dim, action_dim)
        if state_dependent_std:
            self.log_std_layer = nn.Linear(feature_dim, action_dim)
        else:
            name = std_parameterization
            value = torch.full((action_dim,), float(init_std if name == "std" else math.log(init_std)))
            if learnable_std:
                setattr(self, name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.mu_layer.weight, -3e-3, 3e-3)
        if self.mu_layer.bias is not None:
            nn.init.uniform_(self.mu_layer.bias, -3e-3, 3e-3)

        if self.state_dependent_std:
            nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
            if self.log_std_layer.bias is not None:
                nn.init.uniform_(self.log_std_layer.bias, -3e-3, 3e-3)

    def _std(self, x: torch.Tensor, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state_dependent_std:
            log_std = torch.tanh(self.log_std_layer(x))
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
            return log_std, torch.exp(log_std)
        if self.std_parameterization == "std":
            std = torch.clamp(self.std, min=self.std_min, max=self.std_max).expand_as(mu)
            return torch.log(std), std
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max).expand_as(mu)
        return log_std, torch.exp(log_std)

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
            mu: [B, action_dim] (squashed when max_action is set)
            log_std: [B, action_dim]
            entropy: [B]
        """
        mu = self.mu_layer(x)
        log_std, std = self._std(x, mu)
        base_pi = Normal(mu, std, validate_args=self.validate_args)

        if self.max_action is None:
            pi = TransformedDistribution(base_pi, [], validate_args=self.validate_args)
            if action is None:
                action = base_pi.rsample()
            log_prob = base_pi.log_prob(action).sum(dim=-1)
            entropy = base_pi.entropy().sum(dim=-1)
            return StochasticContinuousPolicyStep(pi, action, log_prob, mu, log_std, entropy)

        eps = 1e-6
        max_action = torch.as_tensor(self.max_action, dtype=mu.dtype, device=mu.device)
        max_action = torch.clamp(torch.abs(max_action), min=eps)
        transform = ComposeTransform([TanhTransform(), AffineTransform(loc=0, scale=max_action)], cache_size=1)
        pi = TransformedDistribution(base_pi, transform, validate_args=self.validate_args)

        def squashed_log_prob(pre_tanh: torch.Tensor, squashed: torch.Tensor) -> torch.Tensor:
            log_det = torch.log(max_action * (1 - squashed.pow(2)) + eps)
            return (base_pi.log_prob(pre_tanh) - log_det).sum(dim=-1)

        if action is None:
            pre_tanh_action = base_pi.rsample()
            squashed_action = torch.tanh(pre_tanh_action)
            action = squashed_action * max_action
            log_prob = squashed_log_prob(pre_tanh_action, squashed_action)
            entropy = -log_prob
        else:
            squashed_action = (action / max_action).clamp(-1 + eps, 1 - eps)
            action = squashed_action * max_action
            pre_tanh_action = 0.5 * (torch.log1p(squashed_action) - torch.log1p(-squashed_action))
            log_prob = squashed_log_prob(pre_tanh_action, squashed_action)
            entropy_sample = base_pi.rsample()
            entropy = -squashed_log_prob(entropy_sample, torch.tanh(entropy_sample))
        mu_squashed = max_action * torch.tanh(mu)
        return StochasticContinuousPolicyStep(pi, action, log_prob, mu_squashed, log_std, entropy)

class CategoricalHead(nn.Module):
    """
    Categorical (discrete) policy head.
    """
    def __init__(self, feature_dim: int, action_dim: int, *, validate_args: bool | None = None) -> None:
        super().__init__()
        self.validate_args = validate_args
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
        pi = Categorical(logits=logits, validate_args=self.validate_args)
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
