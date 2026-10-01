import math

import pytest
import torch
from torch import nn

from RLAlg.alg.fpo import FPO, SuperviseTarget
from RLAlg.nn.steps import ValueStep


class ConstantVelocity(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, obs, action, time):
        return ValueStep(torch.ones_like(action) * self.value)


def inputs():
    return (
        torch.zeros(2, 1), torch.zeros(2, 1),
        torch.ones(2, 2, 1), torch.full((2, 2, 1), 0.5),
    )


def test_default_ratio_averages_monte_carlo_losses_before_exponentiation():
    actor = ConstantVelocity()
    obs, actions, eps, t = inputs()
    old = FPO.compute_cmf_loss(actor, obs, actions, eps, t).detach()
    old += torch.tensor([[-0.2, 0.2], [-0.1, 0.1]])
    result = FPO.compute_policy_loss(actor, obs, actions, eps, t, old, torch.ones(2), 0.2)
    assert result['ratio'].item() == pytest.approx(1.0)
    assert result['clip_fraction'].item() == pytest.approx(0.0)


@pytest.mark.parametrize('average', [False, True])
def test_log_ratio_guard_applies_in_both_aggregation_modes(average):
    actor = ConstantVelocity()
    obs, actions, eps, t = inputs()
    old = torch.full((2, 2), 1000.0)
    result = FPO.compute_policy_loss(
        actor, obs, actions, eps, t, old, -torch.ones(2), 0.2,
        average_losses_before_exp=average, ratio_clip_range=(-3, 3),
    )
    assert all(torch.isfinite(value) for value in result.values())
    assert result['ratio'].item() == pytest.approx(math.exp(3), rel=1e-6)
    result['loss'].backward()
    assert torch.isfinite(actor.value.grad)


def test_rollout_loss_and_advantages_are_frozen_during_updates():
    actor = ConstantVelocity(0.1)
    obs, actions, eps, t = inputs()
    # Deliberately leave the old loss attached to a graph: policy updates must
    # still only differentiate the recomputed current loss.
    old = FPO.compute_cmf_loss(actor, obs, actions, eps, t)
    old.retain_grad()
    advantages = torch.ones(2, requires_grad=True)
    result = FPO.compute_policy_loss(actor, obs, actions, eps, t, old, advantages, 0.2)
    result['loss'].backward()
    assert actor.value.grad.abs().item() > 0
    assert old.grad is None
    assert advantages.grad is None


def test_default_sampling_is_reproducible_given_the_initial_noise():
    actor = ConstantVelocity()
    obs, noise = torch.zeros(3, 1), torch.zeros(3, 1)
    first = FPO.sample_actions(actor, obs, noise, 4)
    second = FPO.sample_actions(actor, obs, noise, 4)
    assert torch.equal(first.action, second.action)
    assert torch.equal(first.action, first.action_path[..., -1, :])


@pytest.mark.parametrize('target', [SuperviseTarget.Velocity, SuperviseTarget.Noise])
def test_consistent_velocity_moves_noise_to_an_oracle_action(target):
    # t=0 is data, t=1 is noise; dx/dt = eps - action. Integrating
    # backwards therefore subtracts velocity and recovers the desired action.
    actor = ConstantVelocity(2.0)
    obs = torch.zeros(2, 1)
    action = torch.ones(2, 1)
    eps = torch.full((2, 4, 1), 3.0)
    t = torch.linspace(0, 1, 4).view(1, 4, 1).expand(2, 4, 1)
    loss = FPO.compute_cmf_loss(actor, obs, action, eps, t, supervise_target=target)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)
    step = FPO.sample_actions(actor, obs, eps[:, 0], 4)
    assert torch.allclose(step.action, action)


@pytest.mark.parametrize('target', [SuperviseTarget.Velocity, SuperviseTarget.Noise])
def test_positive_and_negative_advantages_reverse_the_update_direction(target):
    obs, actions, eps, t = inputs()
    for advantage, sign in ((1.0, -1), (-1.0, 1)):
        actor = ConstantVelocity(0.1)
        old = FPO.compute_cmf_loss(actor, obs, actions, eps, t, supervise_target=target).detach()
        result = FPO.compute_policy_loss(
            actor, obs, actions, eps, t, old, torch.full((2,), advantage), 0.2,
            supervise_target=target,
        )
        result['loss'].backward()
        assert actor.value.grad.item() * sign > 0


@pytest.mark.parametrize('problem', ['empty', 'nan_time', 'invalid_time', 'invalid_guard'])
def test_invalid_cmf_inputs_fail_before_producing_nan(problem):
    actor = ConstantVelocity()
    obs, actions, eps, t = inputs()
    if problem == 'empty':
        eps, t = eps[:, :0], t[:, :0]
    elif problem == 'nan_time':
        t[0, 0] = float('nan')
    elif problem == 'invalid_time':
        t[0, 0] = 1.1
    else:
        old = FPO.compute_cmf_loss(actor, obs, actions, eps, t).detach()
        with pytest.raises(ValueError):
            FPO.compute_policy_loss(actor, obs, actions, eps, t, old, torch.ones(2), 0.2, ratio_clip_range=(3, -3))
        return
    with pytest.raises(ValueError):
        FPO.compute_cmf_loss(actor, obs, actions, eps, t)
