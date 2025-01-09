import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def weight_init(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.xavier_uniform_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)
