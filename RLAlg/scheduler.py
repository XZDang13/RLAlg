import torch
from torch.optim.lr_scheduler import _LRScheduler

class KLAdaptiveLR(_LRScheduler):
    def __init__(self, optimizer, kl_threshold: float, min_lr: float = 1e-6, max_lr: float = 1e-2, kl_factor:float = 2,
                 alpha: float = 1.5, beta: float = 0.5, last_epoch: int = -1):
        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.kl_factor = kl_factor
        self.alpha = alpha
        self.beta = beta    
        self.kl_divergence = None
        super().__init__(optimizer, last_epoch)

    def set_kl(self, kl_value: float):
        self.kl_divergence = kl_value

    def get_lr(self):
        if self.kl_divergence is None:
            return [group['lr'] for group in self.optimizer.param_groups]

        new_lrs = []
        for group in self.optimizer.param_groups:
            old_lr = group['lr']
            if self.kl_divergence < self.kl_threshold / self.kl_factor:
                new_lr = min(old_lr * self.alpha, self.max_lr)
            elif self.kl_divergence > self.kl_threshold * self.kl_factor:
                new_lr = max(old_lr * self.beta, self.min_lr)
            else:
                new_lr = old_lr
            new_lrs.append(new_lr)

        return new_lrs