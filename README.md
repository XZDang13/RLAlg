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

DDPG, DDPGDoubleQ, and SAC also provide asymmetric actor/critic observation
interfaces and weighted multiple-critic policy losses. SAC, DSAC, and DSACT
expose `compute_alpha_loss`; its optimization tensor is stored under
`"alpha_loss"`. Policy, value, and critic loss dictionaries use `"loss"`, with
additional metrics depending on the method.

Target networks are created and scheduled by the caller. The corresponding
`update_target_param(model, model_target, tau)` methods update target parameters
as `(1 - tau) * target + tau * source`.

DSACT stores running statistics in the class attributes `q1_mean_std` and
`q2_mean_std`. Set both to `None` before starting an independent training run.

## Model interfaces

Networks return the dataclasses defined in [steps.py](RLAlg/nn/steps.py).
Use matching devices, dtypes, and batch dimensions for inputs and targets.
For a flat batch of size `B`, scalar values, rewards, log probabilities, and
advantages normally have shape `[B]`, rather than `[B, 1]`.

| Model output | Key fields | Typical use |
| --- | --- | --- |
| `DiscretePolicyStep` | `pi`, `action`, `log_prob`, `entropy` | Discrete PPO or IQL policy |
| `StochasticContinuousPolicyStep` | `pi`, `action`, `log_prob`, `mean`, `log_std`, `entropy` | Continuous PPO, SAC, DSAC, DSACT, or IQL policy |
| `DeterministicContinuousPolicyStep` | `pi`, `mean` | DDPG or DDPGDoubleQ policy |
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
    policy, old_step.log_prob, observations, old_step.action,
    advantages, clip_ratio=0.2, entropy_coef=0.01,
)
value_result = PPO.compute_value_loss(value_model, observations, returns)
loss = policy_result["loss"] + 0.5 * value_result["loss"]
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
print({key: metric.detach().item() for key, metric in policy_result.items()})
```

`GaussianHead` provides continuous policies, optionally with scaled tanh action
bounds. Its default `std_parameterization="log_std"` learns log standard deviation;
`std_parameterization="std"` with `init_std` learns standard deviation directly.
The other heads and layer builders are in [layers.py](RLAlg/nn/layers.py).
`NormPosition.NONE`, `PRE`, and `POST` control normalization placement.

## Replay, GAE, and episode boundaries

`ReplayBuffer(num_envs, steps, device)` stores tensors as `[T, N, ...]`, where
`T` is the number of retained steps and `N` is the number of environments.
Create storage for each field before adding records. One record contains one
transition per environment, and total capacity is `steps × num_envs`.

```python
import torch
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae

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
    rewards, values, terminated, last_values=torch.zeros(4),
    gamma=0.99, lambda_=0.95, truncated=truncated,
    bootstrap_timeouts=False,
)
assert returns.shape == advantages.shape == (8, 4)
```

`sample_batch` samples with replacement. `sample_batchs` (the API's spelling)
iterates over shuffled minibatches of stored transitions. `reset` clears a
rollout; `save` and `load` persist the buffer. `sample_sequence_batches` returns
time-major chunks with a `valid_mask` and optional initial-state fields. For
recurrent on-policy updates, collect a fresh rollout before sampling sequences.

`compute_gae` returns returns and normalized advantages, using both `terminated`
and `truncated` as episode boundaries. Use at least two transitions for advantage
normalization. When bootstrapping a timeout from its actual final observation,
add `gamma * V(final_obs)` to that reward yourself and keep
`bootstrap_timeouts=False`. The optional `True` mode instead uses the
current-value convention, adding `gamma * values[t]` at timeouts; do not apply
both corrections. Off-policy `done` masks should represent true termination
when timeouts are intended to bootstrap from the actual next observation.

For goal-conditioned tasks, [HindsightExperienceReplay](RLAlg/buffer/her.py)
relabels goals and optionally recomputes rewards and termination flags.
`ReplayBuffer.sample_batch(..., her_strategies=...)` supplies the goal candidates
for the `future`, `final`, `episode`, and `random` strategies.

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
    policy, observations, torch.randn(32, 2), flow_steps=8,
    n_samples_per_action=10, supervise_target=target,
    time_step_sampler=TimeStepSamplerStrategy.Continuous,
)
advantages = torch.linspace(-1, 1, 32)  # Replace with rollout advantages.
result = FPO.compute_policy_loss(
    policy, observations, step.action, step.eps, step.time_step,
    step.init_cmf_loss, advantages, clip_ratio=0.05,
    supervise_target=target,
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
networks, replay/HER, normalization, logging, and FPO sampling and gradients.

## License

[MIT](LICENSE).
