import torch

from RLAlg.nn.layers import GaussianHead, GRULayer


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


def test_gaussian_head_keeps_finite_gradients_at_action_bounds():
    torch.manual_seed(0)
    max_action = 1000.0
    head = GaussianHead(feature_dim=4, action_dim=2, max_action=max_action)
    x = torch.randn(256, 4)

    # Force near-saturation so tanh-squash correction is numerically stressed.
    with torch.no_grad():
        head.mu_layer.bias.fill_(20.0)

    boundary_action = torch.full((256, 2), max_action)
    step = head(x, boundary_action)
    loss = -step.log_prob.mean() + 0.01 * step.entropy.mean()
    loss.backward()

    assert torch.isfinite(step.log_prob).all()
    assert torch.isfinite(step.entropy).all()
    for param in head.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_gru_layer_supports_single_step_and_sequence_inputs():
    torch.manual_seed(0)
    layer = GRULayer(input_size=3, hidden_size=5)

    x_step = torch.randn(4, 3)
    step_output, step_hidden = layer(x_step)
    assert step_output.shape == (4, 5)
    assert step_hidden.shape == (1, 4, 5)

    x_seq = torch.randn(6, 4, 3)
    seq_output, seq_hidden = layer(x_seq)
    assert seq_output.shape == (6, 4, 5)
    assert seq_hidden.shape == (1, 4, 5)


def test_gru_layer_resets_hidden_state_on_episode_starts():
    torch.manual_seed(0)
    layer = GRULayer(input_size=3, hidden_size=4)

    x = torch.randn(1, 2, 3)
    h0 = torch.randn(1, 2, 4)
    episode_starts = torch.tensor([[False, True]])

    out, _ = layer(x, hidden_state=h0, episode_starts=episode_starts)

    out_keep, _ = layer(x[:, 0:1, :], hidden_state=h0[:, 0:1, :], episode_starts=torch.tensor([[False]]))
    out_reset, _ = layer(
        x[:, 1:2, :],
        hidden_state=torch.zeros_like(h0[:, 1:2, :]),
        episode_starts=torch.tensor([[False]]),
    )

    assert torch.allclose(out[:, 0:1, :], out_keep, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out[:, 1:2, :], out_reset, atol=1e-6, rtol=1e-5)
