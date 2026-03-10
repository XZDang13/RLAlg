import torch
import torch.nn as nn

from RLAlg.nn.steps import ValueStep
from RLAlg.ode_solver.euler_ode_solver import EulerODESolver


class TimeAwareModel(nn.Module):
    def forward(self, obs: torch.Tensor, action: torch.Tensor, time: torch.Tensor) -> ValueStep:
        del obs, time
        return ValueStep(value=torch.ones_like(action))


class ZeroDriftModel(nn.Module):
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> ValueStep:
        del obs
        return ValueStep(value=torch.zeros_like(action))


class RecurrentLikeModel(nn.Module):
    def forward(self, obs: torch.Tensor, action: torch.Tensor, time: torch.Tensor):
        del obs, time
        return ValueStep(value=torch.ones_like(action)), torch.zeros(1)


def test_denoise_returns_actions_and_denoised_path():
    model = TimeAwareModel()
    obs = torch.randn(2, 3)
    init_noise = torch.zeros(2, 4)

    actions, denoised_path = EulerODESolver.denoise(
        model=model,
        sample_steps=4,
        obs=obs,
        init_noise=init_noise,
        deterministic=True,
    )

    assert actions.shape == init_noise.shape
    assert denoised_path.shape == (init_noise.shape[0], 5, init_noise.shape[1])
    assert torch.allclose(denoised_path[:, 0], init_noise)
    assert torch.allclose(actions, torch.full_like(init_noise, -1.0))


def test_denoise_adds_noise_when_not_deterministic():
    model = ZeroDriftModel()
    obs = torch.randn(3, 2)
    init_noise = torch.zeros(3, 2)
    sample_steps = 5

    torch.manual_seed(7)
    stochastic_actions, denoised_path = EulerODESolver.denoise(
        model=model,
        sample_steps=sample_steps,
        obs=obs,
        init_noise=init_noise,
        deterministic=False,
    )

    torch.manual_seed(7)
    expected_noise = torch.randn_like(init_noise) * (1.0 / sample_steps)
    expected_actions = init_noise + expected_noise

    assert torch.allclose(denoised_path[:, -1], init_noise)
    assert torch.allclose(stochastic_actions, expected_actions)


def test_denoise_accepts_tuple_model_output():
    model = RecurrentLikeModel()
    obs = torch.randn(2, 3)
    init_noise = torch.zeros(2, 2)

    actions, denoised_path = EulerODESolver.denoise(
        model=model,
        sample_steps=2,
        obs=obs,
        init_noise=init_noise,
        deterministic=True,
    )

    assert torch.allclose(actions, torch.full_like(init_noise, -1.0))
    assert denoised_path.shape == (2, 3, 2)
