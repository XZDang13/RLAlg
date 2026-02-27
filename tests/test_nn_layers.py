import torch

from RLAlg.nn.layers import GaussianHead


def test_gaussian_head_entropy_shape_and_finiteness_with_action_bounds():
    torch.manual_seed(0)
    head = GaussianHead(feature_dim=4, action_dim=2, max_action=1.0)
    x = torch.randn(256, 4)

    step = head(x)

    assert step.entropy.shape == (x.shape[0],)
    assert torch.isfinite(step.entropy).all()


def test_gaussian_head_bounded_entropy_differs_from_base_gaussian_entropy():
    torch.manual_seed(0)
    head = GaussianHead(feature_dim=4, action_dim=2, max_action=1.0)
    x = torch.randn(256, 4)

    step = head(x)
    base_entropy = step.pi.base_dist.entropy().sum(dim=-1)
    entropy_gap = torch.abs(step.entropy - base_entropy).mean()

    assert entropy_gap > 1e-3


def test_gaussian_head_unbounded_entropy_matches_base_distribution():
    torch.manual_seed(0)
    head = GaussianHead(feature_dim=4, action_dim=2, max_action=None)
    x = torch.randn(128, 4)

    step = head(x)
    base_entropy = step.pi.base_dist.entropy().sum(dim=-1)

    assert torch.allclose(step.entropy, base_entropy, atol=1e-6, rtol=1e-5)
