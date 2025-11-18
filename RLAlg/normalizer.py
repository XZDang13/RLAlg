import torch
import torch.nn as nn

class Normalizer(nn.Module):
    def __init__(self, dim:tuple[int]=()):
        super().__init__()
        
        self.epsilon = 1e-8
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.ones(1))

    def update(self, x:torch.Tensor):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count

        self.mean = self.mean + delta * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def forward(self, x:torch.Tensor, update:bool=False):
        if update:
            self.update(x)

        return (x - self.mean) / (torch.sqrt(self.var) + self.epsilon)