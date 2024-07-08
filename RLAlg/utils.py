import random
import numpy as np
import torch
import torch.nn as nn

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def weight_init(m):
    if isinstance(m, nn.Linear):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
