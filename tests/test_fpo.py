import torch
import torch.nn as nn

from RLAlg.alg.fpo import FPO
from RLAlg.nn.steps import ValueStep


class ConstantFlowActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor, current_action: torch.Tensor, t: torch.Tensor) -> ValueStep:
        del current_action, t
        velocity = obs[..., -self.action_dim :]
        return ValueStep(value=velocity)


class BadShapeActor(nn.Module):
    def forward(self, obs: torch.Tensor, current_action: torch.Tensor, t: torch.Tensor) -> ValueStep:
        del obs, current_action, t
        return ValueStep(value=torch.zeros(2, 3))


def test_compute_cmf_loss_u_mode_uses_valuestep_output():
    action_dim = 2
    actor = ConstantFlowActor(action_dim=action_dim)
    obs = torch.ones(4, 5)
    action = torch.zeros(4, action_dim)
    eps = torch.ones(4, 1, action_dim)
    t = torch.full((4, 1, 1), 0.5)

    output = FPO.compute_cmf_loss(
        actor=actor,
        obs=obs,
        action=action,
        eps=eps,
        t=t,
        n_samples_per_action=1,
        output_mode="u",
    )

    assert torch.allclose(output["cmf_loss"], torch.tensor(0.0))


def test_compute_cmf_loss_u_but_supervise_as_eps_mode():
    action_dim = 3
    actor = ConstantFlowActor(action_dim=action_dim)
    obs = torch.ones(2, 6)
    action = torch.zeros(2, action_dim)
    eps = torch.ones(2, 1, action_dim)
    t = torch.full((2, 1, 1), 0.8)

    output = FPO.compute_cmf_loss(
        actor=actor,
        obs=obs,
        action=action,
        eps=eps,
        t=t,
        n_samples_per_action=1,
        output_mode="u_but_supervise_as_eps",
    )

    assert torch.allclose(output["cmf_loss"], torch.tensor(0.0), atol=1e-7)


def test_compute_cmf_loss_raises_on_actor_shape_mismatch():
    actor = BadShapeActor()
    obs = torch.zeros(2, 4)
    action = torch.zeros(2, 3)

    try:
        FPO.compute_cmf_loss(
            actor=actor,
            obs=obs,
            action=action,
            n_samples_per_action=2,
        )
        assert False, "Expected ValueError due to actor prediction shape mismatch."
    except ValueError as error:
        assert "shape" in str(error)
