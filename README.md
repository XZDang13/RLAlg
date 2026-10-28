# RLAlg

**English** | [简体中文](README.zh-CN.md)

RLAlg is a PyTorch library of reinforcement learning losses, neural network
components, and replay utilities. You supply the models, environment interaction,
optimizers, and update schedule; RLAlg provides the loss calculations and shared
building blocks. For complete training examples, see
[RLDemos](https://github.com/XZDang13/RLDemos).

## Installation

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install torch numpy
python -m pip install -e .
```

The current package metadata does not declare runtime dependencies, so install
PyTorch and NumPy explicitly. `wandb` is optional and is needed only when using
`WandbLogger`. Environment dependencies such as Gymnasium and MuJoCo are managed
by your training project; the core library does not require them.

## Algorithms

Import algorithm classes from their modules, for example
`from RLAlg.alg.ppo import PPO`.

| Class | Module | Main functionality |
| --- | --- | --- |
| `PPO` | [ppo.py](RLAlg/alg/ppo.py) | Clipped policy and value losses, entropy bonus, multiple advantage streams, recurrent variants |
| `DDPG` | [ddpg.py](RLAlg/alg/ddpg.py) | Deterministic actor and critic losses, target actor/critic updates |
| `DDPGDoubleQ` | [ddpg_double_q.py](RLAlg/alg/ddpg_double_q.py) | Noisy deterministic policy, twin critics, separate `std` and `gamma` arguments |
| `SAC` | [sac.py](RLAlg/alg/sac.py) | Stochastic policy, twin critics, entropy temperature optimization |
| `DSAC` | [dsac.py](RLAlg/alg/dsac.py) | Distributional twin critics, bounded TD targets, target actor and critic |
| `DSACT` | [dsact.py](RLAlg/alg/dsact.py) | Distributional twin critics with running standard deviation statistics |
| `FPO` | [fpo.py](RLAlg/alg/fpo.py) | Flow action sampling, CMF losses, clipped flow policy optimization |
| `IQL` | [iql.py](RLAlg/alg/iql.py) | Offline expectile value regression and advantage-weighted policy fitting |
| `SHAC` | [shac.py](RLAlg/alg/shac.py) | Short-horizon analytic policy gradients through a differentiable simulator, TD(λ) value targets |
| `AHAC` | [ahac.py](RLAlg/alg/ahac.py) | SHAC with a contact-adapted horizon, double critic, and critic training until convergence |
| `SAPO` | [sapo.py](RLAlg/alg/sapo.py) | Maximum-entropy SHAC with soft rewards, an auto-tuned temperature, and a double critic |
| `RPO` | [rpo.py](RLAlg/alg/rpo.py) | SAPO with BPTT action gradients reused over several clipped, KL-regularized policy epochs |

All loss functions share one calling convention:

- Models are positional; every tensor and hyperparameter after them is
  keyword-only, for example
  `SAC.compute_critic_loss(actor, critic, critic_target, observations=..., actions=...,
  rewards=..., next_observations=..., terminated=..., alpha=0.2, gamma=0.99)`.
  `terminated` is the true-termination mask; timeouts should still bootstrap.
- Every result is a dictionary whose optimization tensor is `"loss"`, including
  `compute_alpha_loss`. Other entries are detached metrics, such as `"q1"`,
  `"q2"`, and `"q_target"` for twin critics.
- Value and Q regressions use `0.5 * mean((prediction - target) ** 2)`, summed
  over twin critics. Predictions, targets, rewards, and termination masks must
  have the same shape; a `[B, 1]` reward against `[B]` values raises an error.
- Off-policy losses take `critic_observations` / `next_critic_observations` for
  asymmetric actor-critic setups; by default critics see the actor's
  observations. Policy losses accept a list of critics with `critic_weights`.
  PPO accepts a list of advantages with `advantage_weights`, and
  `episode_starts`, `initial_state`, and `valid_mask` for recurrent models.

Target networks are created and scheduled by the caller.
`update_target_param(model, model_target, tau)` updates target parameters as
`(1 - tau) * target + tau * source` and copies buffers such as normalizer
statistics.

DSACT keeps running averages of the critic standard deviations. Pass
`q_mean_std=None` on the first critic update and the returned
`result["q_mean_std"]` on later updates.

## Model interfaces

Networks return the dataclasses defined in [steps.py](RLAlg/nn/steps.py).
Use matching devices, dtypes, and batch dimensions for inputs and targets.
For a flat batch of size `B`, scalar values, rewards, log probabilities, and
advantages normally have shape `[B]`, rather than `[B, 1]`.

| Model output | Key fields | Typical use |
| --- | --- | --- |
| `DiscretePolicyStep` | `pi`, `action`, `log_prob`, `entropy` | Discrete PPO or IQL policy |
| `StochasticContinuousPolicyStep` | `pi`, `action`, `log_prob`, `mean`, `log_std`, `entropy` | Continuous PPO, SAC, DSAC, DSACT, or IQL policy |
| `DeterministicContinuousPolicyStep` | `mean`, `pre_tanh_mean`, `max_action` | DDPG or DDPGDoubleQ policy |
| `ValueStep` | `value` | State value or scalar Q prediction |
| `DistributionStep` | `pi`, `mean`, `std`, `sample` | Distributional Q prediction |
| `FPOStep` | `action`, `action_path`, `eps`, `time_step`, `init_cmf_loss` | Flow sampling result and optional stored CMF targets |

PPO and IQL policies evaluate supplied actions with `policy(observations, actions)`.
SAC-family policies sample actions with `policy(observations)`. DDPGDoubleQ passes
`std` to its policy, so the model must accept that argument. DDPG critics return
one `ValueStep`; DDPGDoubleQ, SAC, and IQL critics return two `ValueStep` objects.
DSAC and DSACT critics return two `DistributionStep` objects.

## Minimal PPO update

This self-contained example demonstrates the API with a synthetic batch.
In training, collect actions and old log probabilities under `torch.no_grad()`
and compute returns and advantages from actual rollouts.

```python
import torch
from torch import nn

from RLAlg.alg.ppo import PPO
from RLAlg.nn.layers import CategoricalHead, CriticHead, NormPosition, make_mlp_layers
from RLAlg.utils import set_seed_everywhere

set_seed_everywhere(0)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers, features = make_mlp_layers(
            4, [64, 64], nn.SiLU(), NormPosition.POST,
        )
        self.head = CategoricalHead(features, 2)

    def forward(self, observations, actions=None):
        return self.head(self.layers(observations), actions)


policy = Policy()
value_model = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), CriticHead(64))
optimizer = torch.optim.Adam(
    list(policy.parameters()) + list(value_model.parameters()), lr=3e-4,
)
observations = torch.randn(32, 4)
with torch.no_grad():
    old_step = policy(observations)

advantages = torch.randn(32)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
returns = torch.randn(32)
policy_result = PPO.compute_policy_loss(
    policy, log_probs_hat=old_step.log_prob, observations=observations,
    actions=old_step.action, advantages=advantages, clip_ratio=0.2, entropy_coef=0.01,
)
value_result = PPO.compute_value_loss(value_model, observations=observations, returns=returns)
loss = policy_result["loss"] + 0.5 * value_result["loss"]
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
print({key: metric.detach().item() for key, metric in policy_result.items()})
```

`GaussianHead` provides continuous policies, optionally with scaled tanh action
bounds. Its standard deviation is configured in std units for every mode:
`init_std` is the initial value and `[std_min, std_max]` the allowed range
(defaults `1`, `e^-20`, and `e^2`). The default `std_parameterization="log_std"`
learns log standard deviation, `"std"` learns it directly, `learnable_std=False`
fixes it, and `state_dependent_std=True` predicts it from the features. With tanh
bounds, `entropy` is the one-sample estimate `-log_prob(a')` with `a'` drawn from
the current policy, so it does not use a supplied action from an older policy.
`DeterministicHead` returns the deterministic action only; DDPGDoubleQ adds its
exploration and target noise with `DDPGDoubleQ.perturb_action(step, std)`, before
the tanh.
The other heads and layer builders are in [layers.py](RLAlg/nn/layers.py).
`NormPosition.NONE`, `PRE`, and `POST` control normalization placement.

## Replay, GAE, and episode boundaries

`ReplayBuffer(num_envs, steps, device)` stores tensors as `[T, N, ...]`, where
`T` is the number of retained steps and `N` is the number of environments.
Create storage for each field before adding records. One record contains one
transition per environment, and total capacity is `steps × num_envs`.

```python
import torch
from RLAlg.buffer import ReplayBuffer, compute_gae

buffer = ReplayBuffer(num_envs=4, steps=8, device=torch.device("cpu"))
buffer.create_storage_space("observations", (3,))
buffer.create_storage_space("actions", (1,))
buffer.add_records({
    "observations": torch.randn(4, 3),
    "actions": torch.randn(4, 1),
})
batch = buffer.sample_batch(["observations", "actions"], batch_size=16)
assert batch["observations"].shape == (16, 3)

rewards = torch.randn(8, 4)
values = torch.zeros(8, 4)
terminated = torch.zeros(8, 4, dtype=torch.bool)
truncated = torch.zeros_like(terminated)
returns, advantages = compute_gae(
    rewards=rewards, values=values, terminated=terminated, last_values=torch.zeros(4),
    gamma=0.99, lambda_=0.95, truncated=truncated,
)
assert returns.shape == advantages.shape == (8, 4)
```

`sample_batch` samples with replacement. `sample_batchs` (the API's spelling)
iterates over shuffled minibatches of stored transitions. `reset` clears a
rollout; `save` and `load` persist the buffer. `sample_sequence_batches` returns
time-major chunks with a `valid_mask` and optional initial-state fields. For
recurrent on-policy updates, collect a fresh rollout before sampling sequences.
Storage is circular: once full, new steps overwrite the oldest, so
`data[:current_size]` is no longer in time order. `time_order()` returns the
storage slots oldest first, and `sample_sequence_batches` uses it.

Return and advantage estimators live in
[advantages.py](RLAlg/buffer/advantages.py) and take keyword arguments only.
`compute_gae` and `compute_advantage_mc` return returns and advantages, using both
`terminated` and `truncated` as episode boundaries; `compute_returns` does the
same for discounted returns. Advantages are standardized unless `normalize=False`
(for example when normalizing per minibatch or combining several critics);
normalization needs at least two transitions. When bootstrapping a timeout from its actual final observation,
add `gamma * V(final_obs)` to that reward yourself and keep
`bootstrap_timeouts=False`. The optional `True` mode instead uses the
current-value convention, adding `gamma * values[t]` at timeouts; do not apply
both corrections. Off-policy `done` masks should represent true termination
when timeouts are intended to bootstrap from the actual next observation.

For goal-conditioned tasks, [HERReplayBuffer](RLAlg/buffer/her.py) is a
`ReplayBuffer` that records episode boundaries as transitions are added and
relabels goals when sampling:

```python
from RLAlg.buffer import HERReplayBuffer, env_reward_fn

buffer = HERReplayBuffer(
    num_envs, steps, device,
    reward_fn=env_reward_fn(envs),  # or any f(achieved, desired) -> reward
    strategy="future", replay_k=4,
)
for key, shape in (("observations", obs_shape), ("desired_goal", goal_shape),
                   ("next_achieved_goal", goal_shape), ("rewards", ())):
    buffer.create_storage_space(key, shape)
buffer.create_storage_space("terminated", ())
buffer.create_storage_space("episode_end", (), torch.bool)

buffer.add_records({..., "terminated": terminated, "episode_end": terminated | truncated})
batch = buffer.sample_batch(["observations", "desired_goal", "rewards", "terminated"], 256)
```

A fraction `1 - 1 / (1 + replay_k)` of the sampled transitions gets the goal
achieved after a step `t'` of its own episode, and its reward is recomputed as
`reward_fn(next_achieved_goal_t, goal)`. `future` draws `t'` uniformly from
`t` to the episode's end (so `t' = t` is an immediate success), `final` uses the
last step, and `episode` draws from the episode's steps still in the buffer. An
unfinished episode ends, for now, at its latest stored step. `batch["her_mask"]`
marks the relabeled rows. `episode_end` must include timeouts; keep true
terminations in a separate key for the Bellman mask. Pass `done_fn` and
`done_key` to also recompute termination for tasks that end on success.
`env_reward_fn` wraps a Gymnasium-Robotics `compute_reward`; for a vector env it
uses the first sub-env, which computes rewards for a whole batch.

## Short-Horizon Actor-Critic

SHAC needs a differentiable environment, such as DFlex, Warp, MJX, Brax, or
Genesis. The actor gradient comes from backpropagating through the simulator,
so `rewards` and `next_values` must keep their autograd graph. All rollout
tensors are time-major `[H, N]`. `next_values[t]` is `V(s_{t+1})` on the actual
next observation, which is the final observation before an automatic reset.
The reference implementation computes it with a target critic.

```python
import torch
import torch.nn as nn
from RLAlg.alg.shac import SHAC
from RLAlg.nn.steps import ValueStep

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 1)

    def forward(self, obs):
        return ValueStep(value=self.net(obs).squeeze(-1))

policy, critic = nn.Linear(3, 3), Critic()
H, N, gamma = 8, 4, 0.99
state = torch.randn(N, 3)  # usually env.reset(); must stay in the autograd graph
observations, rewards, next_values = [], [], []
terminated = torch.zeros(H, N, dtype=torch.bool)
truncated = torch.zeros_like(terminated)
for t in range(H):
    observations.append(state.detach())
    state = state + 0.1 * torch.tanh(policy(state))  # differentiable env.step
    rewards.append(-(state ** 2).sum(-1))
    next_values.append(critic(state).value)  # final obs before reset when done
rewards, next_values = torch.stack(rewards), torch.stack(next_values)

policy_result = SHAC.compute_policy_loss(
    rewards=rewards, next_values=next_values, terminated=terminated,
    gamma=gamma, truncated=truncated,
)
policy_result["loss"].backward()  # then step only the policy optimizer

targets = SHAC.compute_td_lambda_targets(
    rewards=rewards, next_values=next_values, terminated=terminated,
    gamma=gamma, lambda_=0.95, truncated=truncated,
)
value_result = SHAC.compute_value_loss(
    critic, observations=torch.stack(observations), targets=targets,
)
assert targets.shape == (H, N)
```

The policy loss is the negative return `sum_t gamma^t r_t + gamma^h V(s_h)`,
divided by `H * N`. An episode that ends inside the rollout contributes its own
segment, and the discount restarts after it. Terminated steps never bootstrap,
and non-finite values at terminal observations are masked out. Truncated steps
and the last rollout step bootstrap from `next_values`. `compute_td_lambda_targets`
detaches its inputs and uses the same boundaries; `lambda_=0` gives one-step
targets. `compute_value_loss` uses the library's `0.5` MSE; the reference SHAC
omits the `0.5`, which with Adam only changes the effect of gradient clipping.

The policy backward pass also writes gradients into the critic. Zero the critic
gradients before its update, or compute `next_values` with a target critic that
`SHAC.update_target_param` keeps up to date. To cut gradients between
rollouts, detach the simulator state before the next rollout starts.

## Adaptive Horizon Actor-Critic

AHAC ([Georgiev et al., 2024](https://arxiv.org/abs/2405.17784);
[reference code](https://github.com/imgeorgiev/DiffRL)) builds on SHAC and
reuses its policy loss, rollout tensors, and TD(λ) targets. It changes three
things. Dual variables on a contact-stiffness constraint adapt the horizon `H`.
A double critic, `min(V1, V2)`, replaces the target critic. The critic trains
until its loss stops changing. The simulator must report contact forces and
accelerations at each step.

```python
import torch
import torch.nn as nn
from RLAlg.alg.ahac import AHAC
from RLAlg.nn.steps import ValueStep

class DoubleCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.v1, self.v2 = nn.Linear(3, 1), nn.Linear(3, 1)

    def forward(self, obs):
        return ValueStep(value=self.v1(obs).squeeze(-1)), ValueStep(value=self.v2(obs).squeeze(-1))

policy, critic = nn.Linear(3, 3), DoubleCritic()
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=2e-3)
N, gamma, C, H_min, H_max = 4, 0.99, 500.0, 8.0, 64.0
horizon, dual = H_min, torch.zeros(round(H_min))  # kept across iterations

H = len(dual)  # round(horizon)
state = torch.randn(N, 3)  # usually env.reset(); must stay in the autograd graph
observations, rewards, next_values, contact_norms = [], [], [], []
terminated = torch.zeros(H, N, dtype=torch.bool)
for t in range(H):
    observations.append(state.detach())
    action = torch.tanh(policy(state))
    state = state + 0.1 * action  # differentiable env.step
    rewards.append(-(state ** 2).sum(-1))
    contact_forces, accelerations = 1e3 * action.unsqueeze(-1), torch.randn(N, 3, 1)  # from the simulator
    contact_norms.append(AHAC.compute_contact_norms(contact_forces, accelerations))
    next_values.append(AHAC.compute_next_values(critic, state))  # final obs before reset when done
rewards, next_values = torch.stack(rewards), torch.stack(next_values)

policy_result = AHAC.compute_policy_loss(
    rewards=rewards, next_values=next_values, terminated=terminated, gamma=gamma,
)
policy_result["loss"].backward()  # then step only the policy optimizer

targets = AHAC.compute_td_lambda_targets(
    rewards=rewards, next_values=next_values, terminated=terminated, gamma=gamma, lambda_=0.95,
)
losses = []
for epoch in range(64):
    value_result = AHAC.compute_value_loss(
        critic, observations=torch.stack(observations), targets=targets,
    )  # average over mini-batches in practice
    critic_optimizer.zero_grad(set_to_none=True)
    value_result["loss"].backward()
    critic_optimizer.step()
    if AHAC.critic_converged(losses):
        break
    losses.append(value_result["loss"].item())

horizon, dual = AHAC.update_horizon(
    horizon, dual=dual, contact_norms=torch.stack(contact_norms), contact_threshold=C,
    lr=5e-4, min_horizon=H_min, max_horizon=H_max,
)
```

`compute_contact_norms` returns `||contact_forces / max(accelerations, 1)||` per
environment. `update_horizon` applies `dual <- dual - lr * (C - mean_n contact_norms)`
and `H <- clip(H + lr * sum(dual), min_horizon, max_horizon)`. It returns the new
horizon and the next rollout's multipliers, `dual[0]` repeated `round(H)` times.
The next rollout runs `len(dual)` steps. As in the reference code and
Algorithm 1, the constraint adds no term to the actor loss. Contact norms above
`C` make the multipliers positive and lengthen the horizon. The reference
defaults are `C = 500`, horizon `[8, 64]` starting at 8, and `lr = 5e-4`,
decayed linearly to `1e-5`. `critic_converged` stops critic training when
`|mean(diff(last 5 epoch losses))| < 0.2`, checked before the current epoch's
loss is recorded, with at most 64 epochs. `compute_value_loss` sums the 0.5 MSE
of both critics. This equals the reference squared error averaged over the two critics.

## Soft Analytic Policy Optimization

SAPO ([Xing et al., ICLR 2025](https://arxiv.org/abs/2412.12089);
[reference code](https://github.com/etaoxing/mineral)) is maximum-entropy SHAC.
Each reward becomes `r_t + alpha * normalized_entropy_t`, both in the analytic
policy return and in the TD(λ) targets, so the entropy bonus is discounted with
the rewards. The temperature `alpha` is tuned toward a target entropy. The policy
is a tanh-squashed Gaussian, whose entropy is the one-sample estimate
`-log_prob(a)` of the reparameterized action. Two critics replace the target
critic: the policy bootstraps from their mean, the value targets from their minimum.

```python
import torch
import torch.nn as nn
from RLAlg.alg.sapo import SAPO
from RLAlg.nn.layers import GaussianHead
from RLAlg.nn.steps import ValueStep

class DoubleCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.v1, self.v2 = nn.Linear(3, 1), nn.Linear(3, 1)

    def forward(self, obs):
        return ValueStep(value=self.v1(obs).squeeze(-1)), ValueStep(value=self.v2(obs).squeeze(-1))

policy, critic = GaussianHead(3, 3, max_action=1.0, state_dependent_std=True), DoubleCritic()
log_alpha = nn.Parameter(torch.zeros(()))  # init_alpha = 1
alpha_optimizer = torch.optim.AdamW([log_alpha], lr=5e-3, betas=(0.7, 0.95))
N, H, gamma = 4, 32, 0.99
target_entropy = -0.5 * 3  # -0.5 * action_dim

alpha = log_alpha.exp().item()  # fixed for the rollout
state = torch.randn(N, 3)  # usually env.reset(); must stay in the autograd graph
observations, rewards, entropies, min_values, mean_values = [], [], [], [], []
terminated = torch.zeros(H, N, dtype=torch.bool)
for t in range(H):
    observations.append(state.detach())
    step = policy(state)  # tanh-squashed; step.entropy = -log_prob(step.action)
    state = state + 0.1 * step.action  # differentiable env.step
    rewards.append(-(state ** 2).sum(-1))
    entropies.append(step.entropy)
    v_min, v_mean = SAPO.compute_next_values(critic, state)  # final obs before reset when done
    min_values.append(v_min)
    mean_values.append(v_mean)
rewards, entropies = torch.stack(rewards), torch.stack(entropies)

policy_result = SAPO.compute_policy_loss(
    rewards=rewards, entropies=entropies, next_values=torch.stack(mean_values), terminated=terminated,
    gamma=gamma, alpha=alpha, target_entropy=target_entropy,
)
policy_result["loss"].backward()  # then step only the policy optimizer

alpha_result = SAPO.compute_alpha_loss(log_alpha, entropies, target_entropy)
alpha_optimizer.zero_grad(set_to_none=True)
alpha_result["loss"].backward()
alpha_optimizer.step()

targets = SAPO.compute_td_lambda_targets(
    rewards=rewards, entropies=entropies, next_values=torch.stack(min_values), terminated=terminated,
    gamma=gamma, lambda_=0.95, alpha=log_alpha.exp().item(), target_entropy=target_entropy,
)
value_result = SAPO.compute_value_loss(critic, observations=torch.stack(observations), targets=targets)
```

`normalize_entropy` maps an entropy to `(entropy + |target|) / (2 |target|)`, so
the target entropy maps to 0. `compute_alpha_loss` is
`|target| * alpha * mean(entropy - target)` on the raw entropies; the `|target|`
factor undoes the normalization, as in the reference. The reference updates
`alpha` after the actor step and before computing the value targets, so the
targets use the new temperature. `compute_value_loss` is the reference MSE
averaged over the critics, without the library's usual `0.5`. The reference
DFlex Ant defaults are horizon 32, target entropy `-0.5 * action_dim`, initial
`alpha = 1`, AdamW with betas `(0.7, 0.95)` and learning rates `2e-3` (actor),
`5e-4` (critic), and `5e-3` (alpha), gradient-norm clipping at 0.5, a fixed 16
critic epochs of 4 minibatches, log-std clamped to `[-5, 2]`, and LayerNorm SiLU
networks of `[128, 64, 32]` (actor) and `[64, 64]` (two critics).

## Reparameterization Proximal Policy Optimization

RPO ([Zhong et al., ICML 2026](https://arxiv.org/abs/2508.06214); no official
code released) reuses one batch of BPTT gradients for several policy epochs.
After a SAPO-style rollout, `compute_action_gradients` takes the gradient of the
soft short-horizon loss with respect to every action once and caches it. Each
policy epoch regenerates the stored actions under the current policy,
`a = mu + sigma * eps_reg` with `eps_reg = (a - mu) / sigma` held fixed (tanh
squashing is inverted first), and backpropagates the cached gradients through
this path. Each sample is weighted by the detached ratio `rho = pi / pi_old`,
and zeroed outside `[1 - clip_low, 1 + clip_high]` whatever the gradient's sign.
An analytic `KL(pi_old || pi)` penalty keeps the update close to the behavior
policy. The value side is SAPO's.

```python
import torch
import torch.nn as nn
from RLAlg.alg.rpo import RPO
from RLAlg.nn.layers import GaussianHead
from RLAlg.nn.steps import ValueStep

class DoubleCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.v1, self.v2 = nn.Linear(3, 1), nn.Linear(3, 1)

    def forward(self, obs):
        return ValueStep(value=self.v1(obs).squeeze(-1)), ValueStep(value=self.v2(obs).squeeze(-1))

policy, critic = GaussianHead(3, 3, max_action=1.0, state_dependent_std=True), DoubleCritic()
policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4, betas=(0.7, 0.95))
N, H, M, gamma = 4, 32, 5, 0.99
alpha, target_entropy = 0.2, -0.5 * 3  # paper: fixed entropy coefficient, target -0.5 * action_dim

state = torch.randn(N, 3)  # usually env.reset(); must stay in the autograd graph
observations, actions, rewards, entropies, mean_values = [], [], [], [], []
terminated = torch.zeros(H, N, dtype=torch.bool)
for t in range(H):
    observations.append(state.detach())
    step = policy(state)
    actions.append(step.action)  # keep the graph: gradients are taken w.r.t. these tensors
    state = state + 0.1 * step.action  # differentiable env.step
    rewards.append(-(state ** 2).sum(-1))
    entropies.append(step.entropy)
    mean_values.append(RPO.compute_next_values(critic, state)[1])  # final obs before reset when done
observations, rewards, entropies = torch.stack(observations), torch.stack(rewards), torch.stack(entropies)

mean_values = torch.stack(mean_values)
cached = RPO.compute_action_gradients(  # one BPTT pass, frees the rollout graph
    actions=actions, rewards=rewards, entropies=entropies, next_values=mean_values,
    terminated=terminated, gamma=gamma, alpha=alpha, target_entropy=target_entropy,
)
actions = torch.stack(actions).detach()
with torch.no_grad():  # behavior policy on the stored actions
    behavior = policy(observations, actions)
log_probs_hat = behavior.log_prob
means_hat, stds_hat = behavior.pi.base_dist.loc, behavior.pi.base_dist.scale

for epoch in range(M):  # epoch 0 is on-policy, later epochs reuse the cached gradients
    policy_result = RPO.compute_policy_loss(
        policy, observations=observations, actions=actions, action_gradients=cached["action_gradients"],
        log_probs_hat=log_probs_hat, means_hat=means_hat, stds_hat=stds_hat,
        clip_low=0.8, clip_high=1.0, kl_coef=0.25,
    )
    policy_optimizer.zero_grad(set_to_none=True)
    policy_result["loss"].backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    policy_optimizer.step()

targets = RPO.compute_td_lambda_targets(  # the paper uses the critics' mean, SAPO the minimum
    rewards=rewards, entropies=entropies, next_values=mean_values, terminated=terminated,
    gamma=gamma, lambda_=0.95, alpha=alpha, target_entropy=target_entropy,
)
value_result = RPO.compute_value_loss(critic, observations=observations, targets=targets)
```

In the first epoch `rho = 1` and the KL gradient is zero, so the policy gradient
is exactly the SHAC/SAPO BPTT gradient through the actions. The surrogate is
summed, not averaged, because the cached gradients already carry SAPO's
`1 / (H * N)`; when splitting the batch into minibatches, rescale it. Recompute
`log_probs_hat` on the stored actions, as above, so the first ratio is exactly 1.
The paper's defaults are horizon 32, `M = 5` policy and 16 critic epochs,
`clip_low = 0.8`, `clip_high = 1.0`, target entropy `-0.5 * action_dim`,
AdamW `(0.7, 0.95)` with learning rate `5e-4` for both networks on an
exponential schedule, two LayerNorm SiLU `[400, 200, 100]` critics, and
per-task coefficients, for example DFlex Ant `kl_coef = 0.25`, entropy `0.2`.

The paper releases no code and states RPO is built on the SAPO repository, so
some details are inferred:
- The paper's entropy term is SAPO's soft reward (its Appendix C.2), so its
  entropy coefficient is passed as a fixed `alpha`. `compute_policy_loss` also
  takes an optional per-epoch `entropy_coef` bonus (equation 13), which defaults to 0.
  `compute_alpha_loss` is inherited if you prefer SAPO's tuned temperature.
- The paper says the value targets use the critics' mean. SAPO's reference uses
  the minimum. `compute_next_values` returns both.
- `lambda_surr` is left out, since it only rescales the surrogate.
- By the column alignment of the paper's Table 2, RPO clips gradient norms at
  0.5 (the 1.0 is GI-PPO's) and uses no temperature learning rate, so the
  entropy coefficient stays fixed.

## Flow Policy Optimization

FPO uses `t=0` for actions and `t=1` for Gaussian noise. The network predicts
velocity `dx/dt = noise - action`; Euler integration runs backward from 1 to 0.
Both `SuperviseTarget.Velocity` and `SuperviseTarget.Noise` use a velocity-output
network. The latter reconstructs noise as `x_t + (1 - t) * velocity`.

```python
import torch
from torch import nn

from RLAlg.alg.fpo import FPO, SuperviseTarget, TimeStepSamplerStrategy
from RLAlg.nn.layers import DiffusionHead, NormPosition, make_mlp_layers


class FlowPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers, features = make_mlp_layers(
            4 + 2 + 1, [64, 64], nn.SiLU(), NormPosition.POST,
        )
        self.head = DiffusionHead(features, 2)

    def forward(self, observations, actions, time):
        inputs = torch.cat([observations, actions, time], dim=-1)
        return self.head(self.layers(inputs))


policy = FlowPolicy()
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
observations = torch.randn(32, 4)
target = SuperviseTarget.Noise
step = FPO.sample_actions_with_cmf_info(
    policy, observations=observations, init_noise=torch.randn(32, 2), flow_steps=8,
    n_samples_per_action=10, supervise_target=target,
    time_step_sampler=TimeStepSamplerStrategy.Continuous,
)
advantages = torch.linspace(-1, 1, 32)  # Replace with rollout advantages.
result = FPO.compute_policy_loss(
    policy, observations=observations, actions=step.action, eps=step.eps,
    t=step.time_step, init_cmf_loss=step.init_cmf_loss, advantages=advantages,
    clip_ratio=0.05, supervise_target=target,
)
optimizer.zero_grad(set_to_none=True)
result["loss"].backward()
optimizer.step()
print(result["ratio"].detach().item())
```

For a batch of `B` actions of dimension `A` and `M` CMF samples, stored shapes
are `actions: [B, A]`, `eps: [B, M, A]`, `time_step: [B, M, 1]`, and
`init_cmf_loss: [B, M]`. Reuse the same latent actions, noise/time pairs,
supervision target, and observation transformation throughout all updates.
If actions are transformed to satisfy environment bounds, store their latent
values for the CMF calculation.

The default ratio is `exp(mean(old_loss - new_loss))`, one ratio per action.
The default log-ratio guard is `(-3, 3)` and is applied after averaging.
`average_losses_before_exp=False` selects the per-sample ablation.
`approx_kl` is a surrogate-ratio drift metric, not an exact flow-policy KL;
`kl_divergence` is its compatibility alias.

Sampling defaults to `deterministic=True`: given the same initial noise, it
integrates the same action without extra endpoint jitter. Fresh initial noise
still provides exploration. A single discrete timestep with noise supervision
has zero loss at `t=1`; use continuous timesteps or velocity supervision.
Observation normalization, reward scaling, and value clipping are training
choices controlled by the caller. The
[RLDemos FPO guide](https://github.com/XZDang13/RLDemos#online-examples) includes
these choices and measured learning experiments.

## Other components

| Component | Purpose |
| --- | --- |
| [GRULayer](RLAlg/nn/layers.py) and PPO recurrent methods | Hidden-state resets with `episode_starts`, initial states, and padded-sequence masks |
| [Normalizer](RLAlg/normalizer.py) | Running mean/variance; `update(x)` changes statistics, `forward(x)` leaves them frozen by default |
| [GAN](RLAlg/alg/gan.py) | BCE, smoothed BCE, hinge, least-squares, Wasserstein, and generator losses, with gradient penalties |
| [MetricsTracker / WandbLogger](RLAlg/logger.py) | Local metric accumulation and optional W&B logging |
| [KLAdaptiveLR](RLAlg/scheduler.py) | Adjust learning rates using a supplied KL metric via `set_kl` |
| [set_seed_everywhere / weight_init](RLAlg/utils.py) | Random seed setup and network initialization |

IQL expects a fixed dataset and computes value, policy, and twin-critic losses.
The caller supplies the dataset, target critic, and update schedule. This
implementation requires `expectile` in `[0, 1]`, a positive `temperature`, and
weights policy fitting with `exp((Q - V) * temperature)`, capped at 100.

## Examples and tests

[RLDemos](https://github.com/XZDang13/RLDemos) contains runnable PPO, DDPG,
DDPGDoubleQ, SAC, DSAC, DSACT, FPO, and IQL examples, including offline dataset
collection and FPO evaluation. Its README is available in English and Chinese.
[tests/test.ipynb](tests/test.ipynb) demonstrates FPO sampling and replay storage.

Run the library tests from the repository root:

```bash
python -m pip install pytest
python -m pytest -q
```

Tests cover algorithm losses, shape checks, termination masks, recurrent
networks, replay/HER, normalization, logging, FPO sampling and gradients, and
SHAC/AHAC/SAPO/RPO gradients through a differentiable toy rollout.

## License

[MIT](LICENSE).
