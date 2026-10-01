import pytest
import torch
from torch import nn

from RLAlg.alg.fpo import FPO
from RLAlg.nn.steps import ValueStep
from RLAlg.ode_solver.euler_ode_solver import EulerODESolver


class KeywordTimeVelocity(nn.Module):
    def forward(self, obs, action, *, time):
        assert time.shape == (*action.shape[:-1], 1)
        assert obs.shape[:-1] == action.shape[:-1]
        return ValueStep(torch.ones_like(action))


class KeywordTVelocity(nn.Module):
    def forward(self, obs, action, *, t):
        return ValueStep(torch.ones_like(action) * t)


@pytest.mark.parametrize('shape', [(2,), (3, 2), (4, 3, 2)])
def test_solver_preserves_arbitrary_batch_dimensions(shape):
    noise = torch.zeros(shape)
    obs = torch.zeros((*shape[:-1], 5))
    actions, path = EulerODESolver.denoise(KeywordTimeVelocity(), 4, obs, noise, deterministic=True)
    assert torch.allclose(actions, -torch.ones_like(actions))
    assert path.shape == (*shape[:-1], 5, shape[-1])
    assert torch.equal(actions, path[..., -1, :])


def test_loss_and_solver_accept_the_same_keyword_only_time_signatures():
    obs = torch.zeros(3, 2)
    actions = torch.zeros(3, 1)
    eps = torch.ones(3, 2, 1)
    t = torch.full((3, 2, 1), 0.5)
    for actor in (KeywordTimeVelocity(), KeywordTVelocity()):
        FPO.compute_cmf_loss(actor, obs, actions, eps, t)
        EulerODESolver.denoise(actor, 4, obs, actions, deterministic=True)


def test_internal_model_type_error_is_not_retried_or_hidden():
    class BrokenVelocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, obs, action, time=None):
            self.calls += 1
            raise TypeError('velocity implementation failed')

    actor = BrokenVelocity()
    obs, noise = torch.zeros(2, 1), torch.zeros(2, 1)
    with pytest.raises(TypeError, match='velocity implementation failed'):
        EulerODESolver.denoise(actor, 4, obs, noise)
    assert actor.calls == 1
