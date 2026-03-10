import torch
import torch.nn as nn

from RLAlg.alg.fpo import (
    FPO,
    SuperviseTarget,
    TimeStepSampler,
    TimeStepSamplerStrategy,
)
from RLAlg.nn.steps import ValueStep


class ConstantFlowActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor, current_action: torch.Tensor, t: torch.Tensor) -> ValueStep:
        del current_action, t
        velocity = obs[..., -self.action_dim :]
        return ValueStep(value=velocity)


class ZeroFlowActor(nn.Module):
    def forward(self, obs: torch.Tensor, current_action: torch.Tensor, t: torch.Tensor) -> ValueStep:
        del obs, t
        return ValueStep(value=torch.zeros_like(current_action))


class DictZeroFlowActor(nn.Module):
    def forward(
        self,
        obs: dict[str, torch.Tensor],
        current_action: torch.Tensor,
        t: torch.Tensor,
    ) -> ValueStep:
        del obs, t
        return ValueStep(value=torch.zeros_like(current_action))


class BadShapeActor(nn.Module):
    def forward(self, obs: torch.Tensor, current_action: torch.Tensor, t: torch.Tensor) -> ValueStep:
        del obs, current_action, t
        return ValueStep(value=torch.zeros(2, 3))


def test_continuous_time_step_sampler_returns_unit_interval_samples():
    t = TimeStepSampler.sample(
        strategy=TimeStepSamplerStrategy.Continuous,
        sample_shape=(4, 3),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert t.shape == (4, 3, 1)
    assert torch.all(t >= 0.0)
    assert torch.all(t < 1.0)


def test_discrete_time_step_sampler_returns_euler_grid_samples():
    flow_steps = 4
    t = TimeStepSampler.sample(
        strategy=TimeStepSamplerStrategy.Discrete,
        sample_shape=(128,),
        dtype=torch.float32,
        device=torch.device("cpu"),
        flow_steps=flow_steps,
    )

    scaled_t = t * flow_steps
    assert t.shape == (128, 1)
    assert torch.all(scaled_t >= 1.0)
    assert torch.all(scaled_t <= float(flow_steps))
    assert torch.allclose(scaled_t, scaled_t.round())


def test_compute_cmf_loss_noise_mode_uses_provided_targets():
    actor = ConstantFlowActor(action_dim=2)
    obs = torch.ones(4, 5)
    action = torch.zeros(4, 2)
    eps = torch.ones(4, 3, 2)
    t = torch.full((4, 3, 1), 0.5)

    prev_target = FPO.supervise_target
    try:
        FPO.supervise_target = SuperviseTarget.Noise
        output = FPO.compute_cmf_loss(
            policy=actor,
            obs=obs,
            action=action,
            eps=eps,
            t=t,
        )
    finally:
        FPO.supervise_target = prev_target

    assert output.shape == (4, 3)
    assert torch.allclose(output, torch.zeros(4, 3), atol=1e-7)


def test_compute_cmf_loss_supports_dict_observations():
    actor = DictZeroFlowActor()
    obs = {
        "state": torch.zeros(3, 4),
        "goal": torch.ones(3, 2),
    }
    action = torch.zeros(3, 2)
    eps = torch.zeros(3, 5, 2)
    t = torch.full((3, 5, 1), 0.25)

    output = FPO.compute_cmf_loss(
        policy=actor,
        obs=obs,
        action=action,
        eps=eps,
        t=t,
    )

    assert output.shape == (3, 5)
    assert torch.allclose(output, torch.zeros(3, 5), atol=1e-7)


def test_compute_cmf_loss_supports_non_flat_tensor_observations():
    actor = ZeroFlowActor()
    obs = torch.zeros(3, 1, 8, 8)
    action = torch.zeros(3, 2)
    eps = torch.zeros(3, 4, 2)
    t = torch.full((3, 4, 1), 0.25)

    output = FPO.compute_cmf_loss(
        policy=actor,
        obs=obs,
        action=action,
        eps=eps,
        t=t,
    )

    assert output.shape == (3, 4)
    assert torch.allclose(output, torch.zeros(3, 4), atol=1e-7)


def test_sample_actions_with_cmf_info_samples_discrete_t_when_missing():
    actor = ZeroFlowActor()
    obs = torch.zeros(2, 4)
    init_noise = torch.zeros(2, 3)

    prev_sampler = FPO.time_step_sampler
    try:
        FPO.time_step_sampler = TimeStepSamplerStrategy.Discrete
        output = FPO.sample_actions_with_cmf_info(
            policy=actor,
            obs=obs,
            init_noise=init_noise,
            flow_steps=4,
            n_samples_per_action=5,
        )
    finally:
        FPO.time_step_sampler = prev_sampler

    scaled_t = output.time_step * 4
    assert output.time_step.shape == (2, 5, 1)
    assert torch.allclose(scaled_t, scaled_t.round())
    assert output.eps.shape == (2, 5, 3)
    assert output.init_cmf_loss.shape == (2, 5)


def test_compute_cmf_loss_raises_on_actor_shape_mismatch():
    actor = BadShapeActor()
    obs = torch.zeros(2, 4)
    action = torch.zeros(2, 3)
    eps = torch.zeros(2, 2, 3)
    t = torch.full((2, 2, 1), 0.5)

    try:
        FPO.compute_cmf_loss(
            policy=actor,
            obs=obs,
            action=action,
            eps=eps,
            t=t,
        )
        assert False, "Expected ValueError due to actor prediction shape mismatch."
    except ValueError as error:
        assert "shape" in str(error)


def test_compute_policy_loss_matches_unclipped_ratio_objective():
    policy = ConstantFlowActor(action_dim=2)
    obs = torch.ones(4, 5)
    actions = torch.zeros(4, 2)
    eps = torch.ones(4, 3, 2)
    t = torch.full((4, 3, 1), 0.5)
    init_cmf_loss = torch.zeros(4, 3)
    advantages = torch.ones(4)

    prev_target = FPO.supervise_target
    try:
        FPO.supervise_target = SuperviseTarget.Noise
        output = FPO.compute_policy_loss(
            policy=policy,
            observations=obs,
            actions=actions,
            eps=eps,
            t=t,
            init_cmf_loss=init_cmf_loss,
            advantages=advantages,
            clip_ratio=0.2,
        )
    finally:
        FPO.supervise_target = prev_target

    assert torch.allclose(output["loss"], torch.tensor(-1.0))
    assert torch.allclose(output["ratio"], torch.tensor(1.0))
    assert torch.allclose(output["clip_fraction"], torch.tensor(0.0))
    assert torch.allclose(output["kl_divergence"], torch.tensor(0.0))
    assert torch.allclose(output["cmf_loss"], torch.tensor(0.0))


def test_compute_policy_loss_uses_per_sample_ratios_by_default():
    policy = ZeroFlowActor()
    obs = torch.zeros(2, 4)
    actions = torch.zeros(2, 2)
    eps = torch.ones(2, 2, 2)
    t = torch.full((2, 2, 1), 0.5)
    init_cmf_loss = torch.tensor([[0.0, 0.5], [0.0, 0.5]])
    advantages = torch.ones(2)

    prev_target = FPO.supervise_target
    try:
        FPO.supervise_target = SuperviseTarget.Noise
        output = FPO.compute_policy_loss(
            policy=policy,
            observations=obs,
            actions=actions,
            eps=eps,
            t=t,
            init_cmf_loss=init_cmf_loss,
            advantages=advantages,
            clip_ratio=10.0,
        )
    finally:
        FPO.supervise_target = prev_target

    expected_ratios = torch.tensor(
        [[torch.exp(torch.tensor(-0.25)), torch.exp(torch.tensor(0.25))]] * 2
    )
    expected_loss = -expected_ratios.mean()

    assert torch.allclose(output["loss"], expected_loss)
    assert torch.allclose(output["ratio"], expected_ratios.mean())
    assert torch.allclose(output["ratio_min"], expected_ratios.min())
    assert torch.allclose(output["ratio_max"], expected_ratios.max())
    assert torch.allclose(output["surrogate_loss1"], expected_ratios.mean())
    assert torch.allclose(output["surrogate_loss2"], expected_ratios.mean())


def test_compute_policy_loss_can_average_losses_before_exponentiation():
    policy = ZeroFlowActor()
    obs = torch.zeros(2, 4)
    actions = torch.zeros(2, 2)
    eps = torch.ones(2, 2, 2)
    t = torch.full((2, 2, 1), 0.5)
    init_cmf_loss = torch.tensor([[0.0, 0.5], [0.0, 0.5]])
    advantages = torch.ones(2, 1)

    prev_target = FPO.supervise_target
    try:
        FPO.supervise_target = SuperviseTarget.Noise
        output = FPO.compute_policy_loss(
            policy=policy,
            observations=obs,
            actions=actions,
            eps=eps,
            t=t,
            init_cmf_loss=init_cmf_loss,
            advantages=advantages,
            clip_ratio=10.0,
            average_losses_before_exp=True,
        )
    finally:
        FPO.supervise_target = prev_target

    assert torch.allclose(output["loss"], torch.tensor(-1.0))
    assert torch.allclose(output["ratio"], torch.tensor(1.0))
    assert torch.allclose(output["ratio_min"], torch.tensor(1.0))
    assert torch.allclose(output["ratio_max"], torch.tensor(1.0))
    assert torch.allclose(output["surrogate_loss1"], torch.tensor(1.0))
    assert torch.allclose(output["surrogate_loss2"], torch.tensor(1.0))
