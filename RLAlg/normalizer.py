import torch
import torch.nn as nn

class Normalizer(nn.Module):
    def __init__(self, dim: int | tuple[int, ...] = (), epsilon: float = 1e-8):
        super().__init__()

        if isinstance(dim, int):
            dim = (dim,)
        else:
            dim = tuple(dim)
        if any(size < 0 for size in dim):
            raise ValueError(f"dim must contain non-negative sizes, got {dim}.")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")

        self.epsilon = float(epsilon)
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.zeros(1, dtype=torch.float64))

    def _batch_axes_and_count(self, x: torch.Tensor) -> tuple[tuple[int, ...], int]:
        stats_shape = tuple(self.mean.shape)
        if stats_shape:
            if x.ndim < len(stats_shape):
                raise ValueError(
                    f"x must have trailing shape {stats_shape}, got {tuple(x.shape)}."
                )
            x_stats_shape = tuple(x.shape[-len(stats_shape):])
            if x_stats_shape != stats_shape:
                raise ValueError(
                    f"x must have trailing shape {stats_shape}, got {tuple(x.shape)}."
                )
            batch_axes = tuple(range(x.ndim - len(stats_shape)))
            batch_count = 1
            for axis in batch_axes:
                batch_count *= x.shape[axis]
            return batch_axes, batch_count

        return tuple(range(x.ndim)), x.numel()

    def update(self, x: torch.Tensor):
        if x.device != self.mean.device:
            raise ValueError(f"x must be on device {self.mean.device}, got {x.device}.")

        with torch.no_grad():
            x = x.detach().to(dtype=self.mean.dtype)
            batch_axes, batch_count = self._batch_axes_and_count(x)
            if batch_count == 0:
                raise ValueError("cannot update Normalizer with an empty batch.")

            if batch_axes:
                batch_mean = torch.mean(x, dim=batch_axes)
                batch_var = torch.var(x, dim=batch_axes, unbiased=False)
            else:
                batch_mean = x
                batch_var = torch.zeros_like(x)

            batch_count = self.count.new_tensor(batch_count)
            delta = batch_mean - self.mean
            total_count = self.count + batch_count

            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count

            self.mean.copy_(self.mean + delta * batch_count / total_count)
            self.var.copy_((M2 / total_count).clamp_min(0.0))
            self.count.copy_(total_count)

        return self

    def forward(self, x: torch.Tensor, update: bool = False):
        if x.device != self.mean.device:
            raise ValueError(f"x must be on device {self.mean.device}, got {x.device}.")

        if update:
            self.update(x)
        else:
            self._batch_axes_and_count(x)

        mean = self.mean.to(dtype=x.dtype)
        var = self.var.to(dtype=x.dtype)
        return (x - mean) / torch.sqrt(var + self.epsilon)
