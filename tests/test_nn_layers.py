import math

import pytest
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

    assert step.entropy.shape == (x.shape[0],)
    assert torch.allclose(step.entropy, base_entropy, atol=1e-6, rtol=1e-5)


def test_gaussian_head_unbounded_entropy_sums_over_action_dimensions():
    x = torch.zeros(3, 4)
    two_actions = GaussianHead(
        feature_dim=4,
        action_dim=2,
        log_std=0.0,
        max_action=None,
    )(x)
    twenty_three_actions = GaussianHead(
        feature_dim=4,
        action_dim=23,
        log_std=0.0,
        max_action=None,
    )(x)

    torch.testing.assert_close(
        twenty_three_actions.entropy,
        two_actions.entropy * (23.0 / 2.0),
    )


def test_gaussian_head_bounded_entropy_with_supplied_action_is_per_sample():
    torch.manual_seed(0)
    head = GaussianHead(feature_dim=4, action_dim=2, max_action=1.0)
    x = torch.randn(128, 4)
    action = torch.tanh(torch.randn(128, 2))

    step = head(x, action)

    assert step.entropy.shape == (x.shape[0],)
    assert torch.isfinite(step.entropy).all()


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


# Direct std is opt-in; all legacy entropy regression tests above remain intact.


def test_direct_std_initial_distribution_matches_legacy_and_sums_23_axes():
    legacy = GaussianHead(4, 23, log_std=0.0, log_std_min=-5.0, max_action=None)
    direct = GaussianHead(4, 23, std_parameterization='std', init_std=1.0,
                          std_min=math.exp(-5), std_max=math.exp(2))
    direct.mu_layer.load_state_dict(legacy.mu_layer.state_dict())
    x = torch.randn(7, 4)
    action = torch.randn(7, 23)
    old, new = legacy(x, action), direct(x, action)
    assert new.entropy.shape == (7,)
    torch.testing.assert_close(new.log_std, torch.zeros(7, 23))
    torch.testing.assert_close(new.pi.base_dist.scale, torch.ones(7, 23))
    torch.testing.assert_close(new.entropy, torch.full((7,), 23 * 0.5 * math.log(2 * math.pi * math.e)))
    for name in ('mean', 'log_std', 'log_prob', 'entropy'):
        torch.testing.assert_close(getattr(new, name), getattr(old, name))
    assert 'std' in dict(direct.named_parameters()) and 'log_std' not in direct.state_dict()
    assert 'log_std' in legacy.state_dict() and 'std' not in legacy.state_dict()
    # No silent conversion of a legacy checkpoint into raw std parameters.
    with pytest.raises(RuntimeError):
        direct.load_state_dict(legacy.state_dict())


@pytest.mark.parametrize('std', [1.0, 2.0])
def test_direct_std_entropy_gradient_is_inverse_std(std):
    head = GaussianHead(4, 23, std_parameterization='std', init_std=std)
    step = head(torch.zeros(8, 4))
    (-0.01 * step.entropy.mean()).backward()
    torch.testing.assert_close(head.std.grad, torch.full((23,), -0.01 / std))


def test_explicit_legacy_mode_preserves_log_std_parameter_and_constant_entropy_gradient():
    head = GaussianHead(4, 23, log_std=math.log(2), std_parameterization='log_std')
    step = head(torch.zeros(8, 4))
    torch.testing.assert_close(step.pi.base_dist.scale, torch.full((8, 23), 2.))
    (-0.01 * step.entropy.mean()).backward()
    torch.testing.assert_close(head.log_std.grad, torch.full((23,), -0.01))
    fixed = GaussianHead(4, 2, log_std=-0.5, learnable_log_std=False)
    assert 'log_std' in dict(fixed.named_buffers())


def test_state_dependent_log_std_retains_existing_transform_and_gradients():
    head = GaussianHead(4, 3, state_dependent_std=True, log_std_min=-5., log_std_max=2.)
    x = torch.randn(5, 4)
    step = head(x)
    expected = -5 + 3.5 * (torch.tanh(head.log_std_layer(x)) + 1)
    torch.testing.assert_close(step.log_std, expected)
    torch.testing.assert_close(step.entropy, head(x).pi.base_dist.entropy().sum(-1))
    (-step.entropy.mean()).backward()
    assert head.log_std_layer.weight.grad.abs().sum() > 0
    assert not hasattr(head, 'std') and not hasattr(head, 'log_std')
    with pytest.raises(ValueError, match='state-independent'):
        GaussianHead(4, 3, state_dependent_std=True, std_parameterization='std')


@pytest.mark.parametrize('supplied_action', [False, True])
def test_direct_std_tanh_retains_finite_summed_entropy(supplied_action):
    head = GaussianHead(4, 23, std_parameterization='std', init_std=1.0, max_action=2.)
    x = torch.randn(8, 4)
    action = torch.full((8, 23), 2.) if supplied_action else None
    seed = 77
    torch.manual_seed(seed)
    step = head(x, action)
    assert step.entropy.shape == (8,)
    assert torch.all(step.action.abs() <= 2.)
    for tensor in (step.action, step.entropy, step.log_prob):
        assert torch.isfinite(tensor).all()
    if supplied_action:
        torch.manual_seed(seed)
        pre = step.pi.base_dist.rsample()
        log_det = torch.log(2. * (1 - torch.tanh(pre).square()) + 1e-6)
        expected = -(step.pi.base_dist.log_prob(pre) - log_det).sum(-1)
    else:
        expected = -step.log_prob
    torch.testing.assert_close(step.entropy, expected)
    (-step.log_prob.mean() - 0.01 * step.entropy.mean()).backward()
    assert torch.isfinite(head.std.grad).all()


def test_direct_std_clamps_effective_tensor_without_mutating_parameter():
    lower, upper = math.exp(-5), math.exp(2)
    head = GaussianHead(4, 3, std_parameterization='std', init_std=1., std_min=lower, std_max=upper)
    raw = torch.tensor([-1., 2., 100.])
    with torch.no_grad():
        head.std.copy_(raw)
    step = head(torch.zeros(2, 4))
    expected = raw.clamp(lower, upper).expand(2, -1)
    torch.testing.assert_close(head.std, raw)
    torch.testing.assert_close(step.pi.base_dist.scale, expected)
    torch.testing.assert_close(step.log_std, expected.log())
    assert torch.isfinite(step.entropy).all()
    (-step.entropy.mean()).backward()
    torch.testing.assert_close(head.std.grad, torch.tensor([0., -0.5, 0.]))
    # The logging identity uses mean(log(std)), not log(mean(std)).
    torch.testing.assert_close(step.entropy.mean() / 3 - 0.5 * math.log(2 * math.pi * math.e), step.log_std.mean())
    fixed = GaussianHead(4, 3, std_parameterization='std', init_std=1., learnable_std=False)
    assert 'std' in dict(fixed.named_buffers())


@pytest.mark.parametrize('kwargs', [
    {'std_parameterization': 'invalid'},
    {'std_parameterization': 'std', 'std_min': 0.},
    {'std_parameterization': 'std', 'init_std': -1.},
    {'std_parameterization': 'std', 'std_max': 0.5},
    {'std_parameterization': 'std', 'std_max': float('inf')},
    {'std_parameterization': 'std', 'init_std': float('nan')},
])
def test_gaussian_head_rejects_invalid_direct_std_settings(kwargs):
    with pytest.raises(ValueError):
        GaussianHead(4, 3, **kwargs)
