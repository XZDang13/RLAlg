from dataclasses import dataclass

import torch
from torch.distributions import Categorical, TransformedDistribution, Normal

@dataclass
class DiscretePolicyStep:
    pi: Categorical
    action: torch.Tensor
    log_prob: torch.Tensor

@dataclass
class StochasticContinuousPolicyStep:
    pi: TransformedDistribution
    action: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor

@dataclass
class DeterministicContinuousPolicyStep:
    pi: TransformedDistribution
    mean: torch.Tensor

@dataclass
class ValueStep:
    value: torch.Tensor

@dataclass
class DistributionStep:
    pi: TransformedDistribution | Normal
    mean: torch.Tensor
    std: torch.Tensor | float
    sample: torch.Tensor