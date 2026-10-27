from collections import deque

import numpy as np
import pytest
import torch
import torch.nn as nn

from RLAlg.alg.ahac import AHAC
from RLAlg.alg.shac import SHAC
from RLAlg.nn.steps import ValueStep


def _reference_contact_norms(contact_forces, accelerations):
    # Port of the contact normalization in the reference AHAC compute_actor_loss.
    acc = accelerations.clone()
    acc[acc >= 0] = torch.maximum(acc[acc >= 0], torch.ones_like(acc[acc >= 0]))
    acc[acc < 0] = torch.maximum(acc[acc < 0], torch.ones_like(acc[acc < 0]))
    return torch.norm(contact_forces / acc, dim=(1, 2))


def _reference_horizon_update(H, lambd, cfs, C, lambd_lr, steps_min, steps_max):
    # Port of the "Train horizon" step and init_buffers in the reference AHAC train loop.
    lambd = lambd - lambd_lr * (C - cfs.mean(-1))
    H = H + lambd_lr * lambd.sum()
    H = torch.clip(H, steps_min, steps_max)
    return H, lambd[0].repeat(round(H.item()))


def _reference_critic_stop(epoch_losses):
    # Port of the early-stopping loop in the reference AHAC train loop.
    last_losses = deque(maxlen=5)
    for j, loss in enumerate(epoch_losses):
        if len(last_losses) == 5 and abs(np.diff(last_losses).mean()) < 2e-1:
            return j + 1
        last_losses.append(loss)
    return len(epoch_losses)


def _ahac_critic_stop(epoch_losses):
    losses = []
    for j, loss in enumerate(epoch_losses):
        if AHAC.critic_converged(losses):
            return j + 1
        losses.append(loss)
    return len(epoch_losses)


def test_contact_norms_match_reference():
    generator = torch.Generator().manual_seed(0)
    contact_forces = 100.0 * torch.randn(5, 4, 3, generator=generator)
    accelerations = 3.0 * torch.randn(5, 4, 3, generator=generator)

    norms = AHAC.compute_contact_norms(contact_forces, accelerations)

    torch.testing.assert_close(norms, _reference_contact_norms(contact_forces, accelerations))
    assert norms.shape == (5,)


@pytest.mark.parametrize("contact_scale", [100.0, 900.0])
def test_horizon_update_matches_reference_over_iterations(contact_scale):
    generator = torch.Generator().manual_seed(1)
    horizon, dual = 8.0, torch.zeros(8)
    ref_H, ref_lambd = torch.tensor(8.0), torch.zeros(8)
    for _ in range(30):
        contact_norms = contact_scale * torch.rand(dual.shape[0], 6, generator=generator)
        horizon, dual = AHAC.update_horizon(
            horizon, dual=dual, contact_norms=contact_norms, contact_threshold=500.0, lr=5e-4,
            min_horizon=8.0, max_horizon=64.0,
        )
        ref_H, ref_lambd = _reference_horizon_update(ref_H, ref_lambd, contact_norms, 500.0, 5e-4, 8, 64)

        assert horizon == pytest.approx(ref_H.item(), rel=1e-5)
        torch.testing.assert_close(dual, ref_lambd)


def test_horizon_grows_under_stiff_contact_and_shrinks_otherwise():
    # Positive multipliers lengthen the horizon, as in the reference and Algorithm 1.
    stiff = torch.full((16, 4), 900.0)
    soft = torch.full((16, 4), 100.0)
    kwargs = dict(contact_threshold=500.0, lr=1e-2, min_horizon=8.0, max_horizon=64.0)

    longer, dual = AHAC.update_horizon(16.0, dual=torch.zeros(16), contact_norms=stiff, **kwargs)
    shorter, _ = AHAC.update_horizon(16.0, dual=torch.zeros(16), contact_norms=soft, **kwargs)

    assert longer == pytest.approx(16.0 + 1e-2 * 16 * 4.0)
    assert shorter == pytest.approx(16.0 - 1e-2 * 16 * 4.0)
    assert dual.shape == (round(longer),)
    torch.testing.assert_close(dual, torch.full((round(longer),), 4.0))


def test_horizon_update_clips_and_does_not_modify_inputs():
    dual = torch.zeros(8)
    contact_norms = torch.full((8, 2), 1e6)

    horizon, new_dual = AHAC.update_horizon(
        8.0, dual=dual, contact_norms=contact_norms, contact_threshold=500.0, lr=1.0,
        min_horizon=8.0, max_horizon=64.0,
    )

    assert horizon == 64.0
    assert new_dual.shape == (64,)
    torch.testing.assert_close(dual, torch.zeros(8))


class DoubleCritic(nn.Module):
    def __init__(self, values1, values2):
        super().__init__()
        self.values1 = nn.Parameter(values1)
        self.values2 = nn.Parameter(values2)

    def forward(self, observations):
        return ValueStep(value=self.values1), ValueStep(value=self.values2)


def test_next_values_take_minimum_with_graph():
    critic = DoubleCritic(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))

    next_values = AHAC.compute_next_values(critic, torch.zeros(2, 3))
    next_values.sum().backward()

    torch.testing.assert_close(next_values.detach(), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(critic.values1.grad, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(critic.values2.grad, torch.tensor([0.0, 1.0]))


def test_value_loss_equals_reference_mean_over_both_critics():
    critic = DoubleCritic(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]))
    targets = torch.tensor([0.0, 4.0], requires_grad=True)

    result = AHAC.compute_value_loss(critic, observations=torch.zeros(2, 3), targets=targets)
    result["loss"].backward()

    predicted = torch.stack([critic.values1, critic.values2], dim=-1).detach()
    reference = ((predicted - targets.detach().unsqueeze(-1)) ** 2).mean()
    torch.testing.assert_close(result["loss"], reference)
    assert targets.grad is None
    torch.testing.assert_close(critic.values2.grad, torch.tensor([0.0, -2.0]))


@pytest.mark.parametrize(
    "epoch_losses",
    [
        [10.0, 8.0, 6.0, 4.0, 2.0, 1.9, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8],
        [5.0, 5.1, 5.0, 5.1, 5.0, 5.1, 5.0],
        list(np.linspace(100.0, 0.0, 64)),
    ],
)
def test_critic_early_stop_matches_reference(epoch_losses):
    assert _ahac_critic_stop(epoch_losses) == _reference_critic_stop(epoch_losses)


def test_critic_converged():
    assert not AHAC.critic_converged([1.0, 1.0, 1.0, 1.0])
    assert AHAC.critic_converged([1.0, 1.1, 1.0, 1.1, 1.0])
    assert not AHAC.critic_converged([5.0, 4.0, 3.0, 2.0, 1.0])


def test_policy_loss_and_td_lambda_are_shac():
    assert AHAC.compute_policy_loss is SHAC.compute_policy_loss
    assert AHAC.compute_td_lambda_targets is SHAC.compute_td_lambda_targets


def test_validation():
    kwargs = dict(contact_threshold=500.0, lr=0.1, min_horizon=8.0, max_horizon=64.0)
    with pytest.raises(ValueError, match=r"\[H, N\]"):
        AHAC.update_horizon(8.0, dual=torch.zeros(8), contact_norms=torch.zeros(8), **kwargs)
    with pytest.raises(ValueError, match="dual"):
        AHAC.update_horizon(8.0, dual=torch.zeros(7), contact_norms=torch.zeros(8, 2), **kwargs)
    with pytest.raises(ValueError, match="min_horizon"):
        AHAC.update_horizon(
            8.0, dual=torch.zeros(8), contact_norms=torch.zeros(8, 2), contact_threshold=500.0, lr=0.1,
            min_horizon=10.0, max_horizon=5.0,
        )
    with pytest.raises(ValueError, match="same shape"):
        AHAC.compute_contact_norms(torch.zeros(2, 3), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="window"):
        AHAC.critic_converged([1.0], window=1)
    with pytest.raises(TypeError):
        AHAC.update_horizon(8.0, torch.zeros(8), torch.zeros(8, 2), 500.0, 0.1, 8.0, 64.0)
