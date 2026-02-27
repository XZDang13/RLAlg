import torch

from RLAlg.normalizer import Normalizer


def test_normalizer_update_with_single_sample_produces_finite_stats():
    normalizer = Normalizer((3,))
    x = torch.tensor([[1.0, 2.0, 3.0]])

    normalizer.update(x)

    assert torch.isfinite(normalizer.mean).all()
    assert torch.isfinite(normalizer.var).all()

    normalized = normalizer(x)
    assert torch.isfinite(normalized).all()
