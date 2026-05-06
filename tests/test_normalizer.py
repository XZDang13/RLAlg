import torch
import pytest

from RLAlg.normalizer import Normalizer


def test_normalizer_update_with_single_sample_produces_finite_stats():
    normalizer = Normalizer((3,))
    x = torch.tensor([[1.0, 2.0, 3.0]])

    normalizer.update(x)

    assert torch.isfinite(normalizer.mean).all()
    assert torch.isfinite(normalizer.var).all()

    normalized = normalizer(x)
    assert torch.isfinite(normalized).all()


def test_normalizer_single_sample_sets_unbiased_initial_stats():
    normalizer = Normalizer((3,))
    x = torch.tensor([[1.0, 2.0, 3.0]])

    normalizer.update(x)

    assert torch.allclose(normalizer.mean, x.squeeze(0).to(dtype=normalizer.mean.dtype))
    assert torch.allclose(normalizer.var, torch.zeros(3, dtype=normalizer.var.dtype))
    assert torch.allclose(normalizer.count, torch.tensor([1.0], dtype=normalizer.count.dtype))
    assert torch.allclose(normalizer(x), torch.zeros_like(x))


def test_normalizer_combines_multiple_updates():
    normalizer = Normalizer((2,))
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    second = torch.tensor([[5.0, 6.0]])
    combined = torch.cat([first, second], dim=0)

    normalizer.update(first)
    normalizer.update(second)

    assert torch.allclose(normalizer.mean, combined.mean(dim=0).to(dtype=normalizer.mean.dtype))
    assert torch.allclose(normalizer.var, combined.var(dim=0, unbiased=False).to(dtype=normalizer.var.dtype))
    assert torch.allclose(normalizer.count, torch.tensor([3.0], dtype=normalizer.count.dtype))


def test_normalizer_treats_leading_dimensions_as_batch_axes():
    normalizer = Normalizer((2,))
    x = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    flattened = x.reshape(-1, 2)

    normalizer.update(x)

    assert torch.allclose(normalizer.mean, flattened.mean(dim=0).to(dtype=normalizer.mean.dtype))
    assert torch.allclose(normalizer.var, flattened.var(dim=0, unbiased=False).to(dtype=normalizer.var.dtype))
    assert torch.allclose(normalizer.count, torch.tensor([12.0], dtype=normalizer.count.dtype))


def test_normalizer_keeps_running_stats_in_float64_and_preserves_output_dtype():
    normalizer = Normalizer((2,))
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    y = normalizer(x, update=True)

    assert normalizer.mean.dtype == torch.float64
    assert normalizer.var.dtype == torch.float64
    assert normalizer.count.dtype == torch.float64
    assert y.dtype == x.dtype


def test_normalizer_rejects_shape_mismatch():
    normalizer = Normalizer((3,))

    with pytest.raises(ValueError, match="trailing shape"):
        normalizer.update(torch.ones(2, 4))

    with pytest.raises(ValueError, match="trailing shape"):
        normalizer(torch.ones(2, 4))


def test_normalizer_rejects_empty_batches():
    normalizer = Normalizer((3,))

    with pytest.raises(ValueError, match="empty batch"):
        normalizer.update(torch.empty(0, 3))


def test_normalizer_update_does_not_attach_running_stats_to_autograd_graph():
    normalizer = Normalizer((2,))
    x = torch.randn(4, 2, requires_grad=True)

    normalized = normalizer(x, update=True)
    loss = normalized.sum()
    loss.backward()

    assert x.grad is not None
    assert normalizer.mean.grad_fn is None
    assert normalizer.var.grad_fn is None
    assert not normalizer.mean.requires_grad
    assert not normalizer.var.requires_grad
