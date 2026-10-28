import pytest
import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution

from RLAlg.alg.rpo import RPO
from RLAlg.alg.sapo import SAPO
from RLAlg.alg.shac import SHAC
from RLAlg.nn.layers import CategoricalHead, GaussianHead
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep


class EnsembleCritic(nn.Module):
    def __init__(self, obs_dim=3):
        super().__init__()
        self.v1, self.v2 = nn.Linear(obs_dim, 1), nn.Linear(obs_dim, 1)

    def forward(self, observations):
        return ValueStep(value=self.v1(observations).squeeze(-1)), ValueStep(value=self.v2(observations).squeeze(-1))


def _rollout(policy, critic, horizon=5, num_envs=4, seed=0):
    torch.manual_seed(seed)
    state = torch.randn(num_envs, 3)
    observations, actions, rewards, entropies, next_values = [], [], [], [], []
    for _ in range(horizon):
        observations.append(state.detach())
        step = policy(state)
        actions.append(step.action)
        state = state + 0.1 * torch.cat([step.action, step.action[:, :1]], dim=-1)
        rewards.append(-(state ** 2).sum(-1))
        entropies.append(step.entropy)
        next_values.append(SAPO.compute_next_values(critic, state)[1])
    return dict(
        observations=torch.stack(observations), actions=actions, rewards=torch.stack(rewards),
        entropies=torch.stack(entropies), next_values=torch.stack(next_values),
        terminated=torch.zeros(horizon, num_envs, dtype=torch.bool),
    )


def _behavior(policy, observations, actions):
    with torch.no_grad():
        step = policy(observations, actions)
    base = step.pi.base_dist
    return dict(log_probs_hat=step.log_prob, means_hat=base.loc.clone(), stds_hat=base.scale.clone())


def _policy_grads(policy):
    return [p.grad.clone() for p in policy.parameters()]


@pytest.mark.parametrize("max_action", [None, 1.0])
def test_regenerated_actions_match_stored_and_follow_the_policy(max_action):
    torch.manual_seed(0)
    policy = GaussianHead(3, 2, max_action=max_action, state_dependent_std=True)
    observations = torch.randn(6, 3)
    with torch.no_grad():
        actions = policy(observations).action
    step = policy(observations, actions)
    regenerated = RPO.regenerate_actions(step.pi, actions)
    torch.testing.assert_close(regenerated, actions, atol=1e-5, rtol=1e-5)

    base = step.pi.base_dist
    loc, scale = base.loc.detach().requires_grad_(True), base.scale.detach().requires_grad_(True)
    pi = Normal(loc, scale)
    regenerated = RPO.regenerate_actions(pi, actions if max_action is None else torch.atanh(actions))
    regenerated.sum().backward()
    torch.testing.assert_close(loc.grad, torch.ones_like(loc))
    noise = ((actions if max_action is None else torch.atanh(actions)) - base.loc) / base.scale
    torch.testing.assert_close(scale.grad, noise.detach(), atol=1e-4, rtol=1e-4)


def test_regenerate_rejects_non_gaussian_policy():
    step = CategoricalHead(3, 2)(torch.randn(4, 3))
    with pytest.raises(ValueError, match="Gaussian"):
        RPO.regenerate_actions(step.pi, torch.zeros(4))


@pytest.mark.parametrize("max_action", [None, 1.0])
def test_first_epoch_gradient_is_the_bptt_gradient(max_action):
    policy = GaussianHead(3, 2, max_action=max_action, state_dependent_std=True)
    critic = EnsembleCritic()
    kwargs = dict(gamma=0.99, alpha=0.0, target_entropy=-1.0)

    rollout = _rollout(policy, critic)
    shac = SHAC.compute_policy_loss(
        rewards=rollout["rewards"], next_values=rollout["next_values"], terminated=rollout["terminated"],
        gamma=0.99,
    )
    shac["loss"].backward()
    bptt_grads = _policy_grads(policy)
    policy.zero_grad()

    rollout = _rollout(policy, critic)
    cached = RPO.compute_action_gradients(
        actions=rollout["actions"], rewards=rollout["rewards"], entropies=rollout["entropies"],
        next_values=rollout["next_values"], terminated=rollout["terminated"], **kwargs,
    )
    torch.testing.assert_close(cached["loss"], shac["loss"].detach())
    actions = torch.stack(rollout["actions"]).detach()
    result = RPO.compute_policy_loss(
        policy, observations=rollout["observations"], actions=actions,
        action_gradients=cached["action_gradients"], clip_low=0.8, clip_high=1.0, kl_coef=0.5,
        **_behavior(policy, rollout["observations"], actions),
    )
    result["loss"].backward()

    for rpo_grad, bptt_grad in zip(_policy_grads(policy), bptt_grads):
        torch.testing.assert_close(rpo_grad, bptt_grad, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(result["ratio"], torch.tensor(1.0))
    assert result["kl_divergence"].abs() < 1e-6
    assert result["policy_clip_fraction"] == 0.0


def test_action_gradients_accept_tensor_and_zero_fill_unused_actions():
    actions = torch.randn(3, 2, 1, requires_grad=True)
    rewards = actions[:2].sum(-1) ** 2
    rewards = torch.cat([rewards, torch.zeros(1, 2)])
    unused = torch.randn(2, 1, requires_grad=True)
    kwargs = dict(
        rewards=rewards, entropies=torch.zeros(3, 2), next_values=torch.zeros(3, 2),
        terminated=torch.zeros(3, 2, dtype=torch.bool), gamma=1.0, alpha=0.0, target_entropy=-1.0,
    )
    from_tensor = RPO.compute_action_gradients(actions=actions, **kwargs)["action_gradients"]
    torch.testing.assert_close(from_tensor[:2], -2.0 * actions[:2].detach() / 6)
    torch.testing.assert_close(from_tensor[2], torch.zeros(2, 1))

    rewards = torch.stack([unused.sum(-1) * 0.0 + 1.0] * 3)
    kwargs["rewards"] = rewards
    from_list = RPO.compute_action_gradients(actions=[unused, unused.detach().clone().requires_grad_(), unused], **kwargs)
    assert from_list["action_gradients"].shape == (3, 2, 1)
    torch.testing.assert_close(from_list["action_gradients"][1], torch.zeros(2, 1))

    with pytest.raises(ValueError, match="H, N"):
        RPO.compute_action_gradients(actions=[unused], **kwargs)


class FixedGaussian(nn.Module):
    """State-independent Gaussian with learnable mean, for checking the surrogate by hand."""

    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.std = std

    def forward(self, observations, actions):
        mean, std = self.mean.expand_as(actions), self.std.expand_as(actions)
        base = Normal(mean, std)
        return StochasticContinuousPolicyStep(
            pi=TransformedDistribution(base, []), action=actions, log_prob=base.log_prob(actions).sum(-1),
            mean=mean, log_std=std.log(), entropy=base.entropy().sum(-1),
        )


def test_clipped_ratio_weights_and_zeroes_cached_gradients():
    std = torch.tensor([1.0])
    policy = FixedGaussian(torch.tensor([0.0]), std)
    actions = torch.tensor([[0.0], [1.0], [2.0]])
    gradients = torch.tensor([[1.0], [2.0], [3.0]])
    log_probs = Normal(0.0, 1.0).log_prob(actions).sum(-1)
    # Behavior log-probs chosen so the ratios are 1.5, 3.0 (above 1 + 1.0) and 0.1 (below 1 - 0.8).
    ratios = torch.tensor([1.5, 3.0, 0.1])
    log_probs_hat = log_probs - ratios.log()

    result = RPO.compute_policy_loss(
        policy, observations=torch.zeros(3, 1), actions=actions, action_gradients=gradients,
        log_probs_hat=log_probs_hat, means_hat=torch.zeros(3, 1), stds_hat=torch.ones(3, 1),
        clip_low=0.8, clip_high=1.0, kl_coef=0.0,
    )
    torch.testing.assert_close(result["surrogate"], torch.tensor(1.5 * 0.0 * 1.0))
    torch.testing.assert_close(result["policy_clip_fraction"], torch.tensor(2.0 / 3.0))
    result["loss"].backward()
    # d a_reg / d mean = 1, so only the in-range sample contributes rho * g; rho itself is not differentiated.
    torch.testing.assert_close(policy.mean.grad, torch.tensor([1.5 * 1.0]))


def test_kl_penalty_is_analytic_behavior_to_current_kl():
    policy = FixedGaussian(torch.tensor([0.5, -1.0]), torch.tensor([2.0, 0.5]))
    actions = torch.zeros(4, 2)
    means_hat, stds_hat = torch.zeros(4, 2), torch.ones(4, 2)
    result = RPO.compute_policy_loss(
        policy, observations=torch.zeros(4, 1), actions=actions, action_gradients=torch.zeros(4, 2),
        log_probs_hat=Normal(0.0, 1.0).log_prob(actions).sum(-1), means_hat=means_hat, stds_hat=stds_hat,
        clip_low=0.8, clip_high=1.0, kl_coef=0.3, entropy_coef=0.1,
    )
    mean, std = policy.mean.detach(), policy.std
    reference_kl = (torch.log(std) + (1.0 + mean ** 2) / (2 * std ** 2) - 0.5).sum()
    torch.testing.assert_close(result["kl_divergence"], reference_kl)
    entropy = Normal(mean, std).entropy().sum()
    torch.testing.assert_close(result["loss"], 0.3 * reference_kl - 0.1 * entropy)
    result["loss"].backward()
    torch.testing.assert_close(policy.mean.grad, 0.3 * mean / std ** 2)


def test_policy_loss_validates_arguments():
    policy = FixedGaussian(torch.zeros(2), torch.ones(2))
    kwargs = dict(
        observations=torch.zeros(4, 1), actions=torch.zeros(4, 2), action_gradients=torch.zeros(4, 2),
        log_probs_hat=torch.zeros(4), means_hat=torch.zeros(4, 2), stds_hat=torch.ones(4, 2),
        clip_low=0.8, clip_high=1.0, kl_coef=0.0,
    )
    with pytest.raises(ValueError, match="clip_low"):
        RPO.compute_policy_loss(policy, **{**kwargs, "clip_low": 1.5})
    with pytest.raises(ValueError, match="action_gradients"):
        RPO.compute_policy_loss(policy, **{**kwargs, "action_gradients": torch.zeros(4, 3)})
    with pytest.raises(ValueError, match="log_probs_hat"):
        RPO.compute_policy_loss(policy, **{**kwargs, "log_probs_hat": torch.zeros(3)})


def test_off_policy_epochs_reuse_cached_gradients_with_soft_rewards():
    policy = GaussianHead(3, 2, max_action=1.0, state_dependent_std=True)
    critic = EnsembleCritic()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    rollout = _rollout(policy, critic, seed=3)
    cached = RPO.compute_action_gradients(
        actions=rollout["actions"], rewards=rollout["rewards"], entropies=rollout["entropies"],
        next_values=rollout["next_values"], terminated=rollout["terminated"], gamma=0.99, alpha=0.5,
        target_entropy=-1.0,
    )
    actions = torch.stack(rollout["actions"]).detach()
    behavior = _behavior(policy, rollout["observations"], actions)
    kls = []
    for _ in range(4):
        result = RPO.compute_policy_loss(
            policy, observations=rollout["observations"], actions=actions,
            action_gradients=cached["action_gradients"], clip_low=0.8, clip_high=1.0, kl_coef=0.2,
            entropy_coef=0.01, **behavior,
        )
        optimizer.zero_grad()
        result["loss"].backward()
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters())
        optimizer.step()
        kls.append(result["kl_divergence"].item())
    assert kls[0] < 1e-6 < kls[-1]
    assert result["ratio"] != 1.0
