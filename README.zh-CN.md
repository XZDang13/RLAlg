# RLAlg

[English](README.md) | **简体中文**

RLAlg 是基于 PyTorch 的强化学习损失函数、神经网络组件和经验回放工具库。
你负责模型、环境交互、优化器和更新调度，RLAlg 提供损失计算及通用组件。
完整训练例子见 [RLDemos](https://github.com/XZDang13/RLDemos)。

## 安装

使用 Python 3.10 或更高版本，在仓库根目录执行：

```bash
python -m pip install torch numpy
python -m pip install -e .
```

当前打包配置没有声明运行依赖，因此需要显式安装 PyTorch 和 NumPy。
`wandb` 是可选依赖，仅使用 `WandbLogger` 时需要安装。
Gymnasium、MuJoCo 等环境依赖由训练项目管理，核心库无需安装它们。

## 算法

从对应模块导入算法类，例如 `from RLAlg.alg.ppo import PPO`。

| 类 | 模块 | 主要功能 |
| --- | --- | --- |
| `PPO` | [ppo.py](RLAlg/alg/ppo.py) | 裁剪策略和价值损失、熵奖励、多路优势、循环网络接口 |
| `DDPG` | [ddpg.py](RLAlg/alg/ddpg.py) | 确定性 actor 和 critic 损失、目标 actor/critic 更新 |
| `DDPGDoubleQ` | [ddpg_double_q.py](RLAlg/alg/ddpg_double_q.py) | 带噪声的确定性策略、双 critic、独立的 `std` 和 `gamma` 参数 |
| `SAC` | [sac.py](RLAlg/alg/sac.py) | 随机策略、双 critic、熵温度优化 |
| `DSAC` | [dsac.py](RLAlg/alg/dsac.py) | 预测 Q 分布的双 critic、有界 TD 目标、目标 actor 和 critic |
| `DSACT` | [dsact.py](RLAlg/alg/dsact.py) | 使用运行标准差统计的 Q 分布双 critic |
| `FPO` | [fpo.py](RLAlg/alg/fpo.py) | 流动作采样、CMF 损失、裁剪流策略优化 |
| `IQL` | [iql.py](RLAlg/alg/iql.py) | 离线 expectile 价值回归和优势加权策略拟合 |
| `SHAC` | [shac.py](RLAlg/alg/shac.py) | 通过可微仿真器的短视界解析策略梯度、TD(λ) 价值目标 |
| `AHAC` | [ahac.py](RLAlg/alg/ahac.py) | 按接触刚度自适应视界的 SHAC、双 critic、训练 critic 至收敛 |
| `SAPO` | [sapo.py](RLAlg/alg/sapo.py) | 最大熵 SHAC：软奖励、自动调节温度、双 critic |

所有损失函数采用同一套调用约定：

- 模型按位置传入；模型之后的张量和超参数都只能按关键字传，例如
  `SAC.compute_critic_loss(actor, critic, critic_target, observations=..., actions=...,
  rewards=..., next_observations=..., terminated=..., alpha=0.2, gamma=0.99)`。
  `terminated` 表示真正的终止，超时仍应 bootstrap。
- 返回值都是字典，优化用的张量放在 `"loss"` 中，`compute_alpha_loss` 也一样。
  其余条目是 detach 后的指标，例如双 critic 的 `"q1"`、`"q2"`、`"q_target"`。
- 价值和 Q 的回归统一用 `0.5 * mean((prediction - target) ** 2)`，双 critic
  求和。预测、目标、奖励和终止掩码的形状必须一致；`[B, 1]` 的奖励配 `[B]`
  的价值会报错。
- off-policy 损失用 `critic_observations` / `next_critic_observations` 支持
  非对称 actor-critic，默认 critic 看到的观测与 actor 相同。策略损失可以传入
  critic 列表并配 `critic_weights`。PPO 可以传入优势列表并配 `advantage_weights`，
  循环网络用 `episode_starts`、`initial_state`、`valid_mask`。

目标网络的创建和更新调度由调用方负责。
`update_target_param(model, model_target, tau)` 按
`(1 - tau) * target + tau * source` 更新目标参数，并复制归一化统计量等 buffer。

DSACT 维护 critic 标准差的滑动平均。第一次更新 critic 时传 `q_mean_std=None`，
之后传入上一次返回的 `result["q_mean_std"]`。

## 模型接口

网络返回 [steps.py](RLAlg/nn/steps.py) 中定义的数据类。
输入和目标需要使用一致的设备、数据类型和批次维度。
对于大小为 `B` 的普通批次，标量价值、奖励、对数概率和优势通常使用
`[B]` 形状，而不是 `[B, 1]`。

| 模型输出 | 主要字段 | 常见用途 |
| --- | --- | --- |
| `DiscretePolicyStep` | `pi`、`action`、`log_prob`、`entropy` | 离散 PPO 或 IQL 策略 |
| `StochasticContinuousPolicyStep` | `pi`、`action`、`log_prob`、`mean`、`log_std`、`entropy` | 连续 PPO、SAC、DSAC、DSACT 或 IQL 策略 |
| `DeterministicContinuousPolicyStep` | `mean`、`pre_tanh_mean`、`max_action` | DDPG 或 DDPGDoubleQ 策略 |
| `ValueStep` | `value` | 状态价值或标量 Q 预测 |
| `DistributionStep` | `pi`、`mean`、`std`、`sample` | Q 分布预测 |
| `FPOStep` | `action`、`action_path`、`eps`、`time_step`、`init_cmf_loss` | 流采样结果和可选的缓存 CMF 目标 |

PPO、IQL 使用 `policy(observations, actions)` 计算指定动作的概率。
SAC 系列使用 `policy(observations)` 采样动作。
DDPGDoubleQ 会向策略传入 `std`，因此模型需要接受该参数。
DDPG 的 critic 返回一个 `ValueStep`；DDPGDoubleQ、SAC、IQL 的 critic
返回两个 `ValueStep`；DSAC、DSACT 的 critic 返回两个 `DistributionStep`。

## 最小 PPO 更新例子

以下例子使用合成批次演示接口，可以独立运行。
实际训练时，在 `torch.no_grad()` 下采集动作和旧对数概率，
并根据真实 rollout 计算回报和优势。

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

`GaussianHead` 提供连续策略，可使用带缩放的 tanh 动作范围。
各种模式的标准差都用标准差本身的单位配置：`init_std` 是初始值，
`[std_min, std_max]` 是允许范围（默认 `1`、`e^-20`、`e^2`）。
默认 `std_parameterization="log_std"` 学习对数标准差，`"std"` 直接学习标准差，
`learnable_std=False` 固定标准差，`state_dependent_std=True` 由特征预测标准差。
带 tanh 范围时，`entropy` 是单样本估计 `-log_prob(a')`，`a'` 从当前策略采样，
因此不会使用传入的旧策略动作。
`DeterministicHead` 只输出确定性动作；DDPGDoubleQ 的探索噪声和目标噪声由
`DDPGDoubleQ.perturb_action(step, std)` 在 tanh 之前加入。
其他输出头和网络构建接口见 [layers.py](RLAlg/nn/layers.py)。
`NormPosition.NONE`、`PRE`、`POST` 控制归一化的位置。

`GaussianHead` 和 `CategoricalHead` 支持 `validate_args` 参数，默认 `None`
保留 PyTorch 分布的默认校验行为。对已验证的 GPU 训练输入，可显式设置
`validate_args=False`，避免分布参数/样本校验中的 CPU 标量读取和 GPU 同步。
该选项不改变采样、log_prob 或梯度公式；调用方仍应检查动作范围和训练数值，
例如使用异步有限值断言。它不会修改 PyTorch 的全局校验设置。

## 经验回放、GAE 与 episode 边界

`ReplayBuffer(num_envs, steps, device)` 按 `[T, N, ...]` 保存张量，
`T` 为保留步数，`N` 为环境数量。添加记录前先为每个字段创建存储。
每条记录包含各环境的一次 transition，总容量为 `steps × num_envs`。

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

`sample_batch` 有放回地采样；`sample_batchs`（接口使用此拼写）遍历打乱后的
transition 小批次。`reset` 清空 rollout，`save` 和 `load` 保存、读取缓冲区。
`sample_sequence_batches` 返回时间优先的序列块、`valid_mask` 和可选的初始状态字段。
循环策略的 on-policy 更新应先采集一段新的 rollout，再采样序列。
存储是循环的：写满后新数据覆盖最旧的数据，此时 `data[:current_size]`
不再按时间排序。`time_order()` 按从旧到新返回存储位置，
`sample_sequence_batches` 就是按它切分序列的。

回报和优势估计函数在 [advantages.py](RLAlg/buffer/advantages.py) 中，
参数只能按关键字传。`compute_gae` 和 `compute_advantage_mc` 返回回报和优势，
同时将 `terminated`、`truncated` 视为 episode 边界；`compute_returns`
对折扣回报采用相同规则。优势默认会标准化，`normalize=False` 可关闭
（例如按 minibatch 归一化或组合多个 critic 时）；标准化需要至少两条 transition。
若希望超时从真实末状态 bootstrap，请先自行给相应奖励加上
`gamma * V(final_obs)`，并保持 `bootstrap_timeouts=False`。
可选的 `True` 模式采用当前状态价值约定，在超时处加上
`gamma * values[t]`；不要同时应用两种修正。
Off-policy 方法若希望超时从真实下一观测 bootstrap，则 `done` 应仅表示真正终止。

对于目标条件任务，[HERReplayBuffer](RLAlg/buffer/her.py) 是一个
`ReplayBuffer`：写入 transition 时记录 episode 边界，采样时重标记目标：

```python
from RLAlg.buffer import HERReplayBuffer, env_reward_fn

buffer = HERReplayBuffer(
    num_envs, steps, device,
    reward_fn=env_reward_fn(envs),  # 或任意 f(achieved, desired) -> reward
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

采样到的 transition 中有 `1 - 1 / (1 + replay_k)` 的比例会被换成同一 episode
里某一步 `t'` 之后达到的目标，并用 `reward_fn(next_achieved_goal_t, goal)`
重算奖励。`future` 在 `t` 到 episode 结束之间均匀取 `t'`（`t' = t` 即立即成功），
`final` 取最后一步，`episode` 在该 episode 仍留在 buffer 中的步里均匀取。
尚未结束的 episode 暂时以它最新写入的一步作为结束。`batch["her_mask"]`
标出被重标记的行。`episode_end` 必须包含超时；真正的终止另存一个键，
用作 Bellman 掩码。对成功即结束的任务，可传入 `done_fn` 和 `done_key`
同时重算终止标记。`env_reward_fn` 包装 Gymnasium-Robotics 的 `compute_reward`；
对向量环境，它使用第一个子环境，一次计算整批奖励。

## Short-Horizon Actor-Critic

SHAC 需要可微环境，例如 DFlex、Warp、MJX、Brax、Genesis。
actor 梯度通过对仿真器反向传播得到，因此 `rewards` 和 `next_values`
必须保留计算图。所有 rollout 张量都是时间优先的 `[H, N]`。
`next_values[t]` 是真实下一观测上的 `V(s_{t+1})`，即自动重置前的最终观测。
参考实现用目标 critic 计算它。

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
state = torch.randn(N, 3)  # 通常来自 env.reset()；必须保留在计算图中
observations, rewards, next_values = [], [], []
terminated = torch.zeros(H, N, dtype=torch.bool)
truncated = torch.zeros_like(terminated)
for t in range(H):
    observations.append(state.detach())
    state = state + 0.1 * torch.tanh(policy(state))  # 可微的 env.step
    rewards.append(-(state ** 2).sum(-1))
    next_values.append(critic(state).value)  # done 时用重置前的最终观测
rewards, next_values = torch.stack(rewards), torch.stack(next_values)

policy_result = SHAC.compute_policy_loss(
    rewards=rewards, next_values=next_values, terminated=terminated,
    gamma=gamma, truncated=truncated,
)
policy_result["loss"].backward()  # 之后只更新策略优化器

targets = SHAC.compute_td_lambda_targets(
    rewards=rewards, next_values=next_values, terminated=terminated,
    gamma=gamma, lambda_=0.95, truncated=truncated,
)
value_result = SHAC.compute_value_loss(
    critic, observations=torch.stack(observations), targets=targets,
)
assert targets.shape == (H, N)
```

策略损失是负回报 `sum_t gamma^t r_t + gamma^h V(s_h)`，再除以 `H * N`。
在 rollout 中途结束的 episode 单独计算一段回报，之后折扣重新开始。
终止步不做 bootstrap，终止观测上的非有限价值会被屏蔽。
截断步和 rollout 的最后一步从 `next_values` bootstrap。
`compute_td_lambda_targets` 会 detach 输入，并使用相同的边界；
`lambda_=0` 即单步目标。`compute_value_loss` 使用库里统一的 `0.5` 倍均方误差；
参考实现没有 `0.5`，在 Adam 下这只影响梯度裁剪的效果。

策略的反向传播也会给 critic 写入梯度。更新 critic 前先清零其梯度，
或者用目标 critic 计算 `next_values`，并用 `SHAC.update_target_param` 更新它。
若要在 rollout 之间截断梯度，请在下一次 rollout 开始前 detach 仿真器状态。

## Adaptive Horizon Actor-Critic

AHAC（[Georgiev 等，2024](https://arxiv.org/abs/2405.17784)；
[参考代码](https://github.com/imgeorgiev/DiffRL)）基于 SHAC，
复用其策略损失、rollout 张量和 TD(λ) 目标，并做了三处改动：
用接触刚度约束的对偶变量自适应调整视界 `H`；
用双 critic `min(V1, V2)` 取代目标 critic；训练 critic 直到损失不再变化。
仿真器需要在每一步给出接触力和加速度。

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

`compute_contact_norms` 对每个环境返回 `||contact_forces / max(accelerations, 1)||`。
`update_horizon` 执行 `dual <- dual - lr * (C - mean_n contact_norms)` 和
`H <- clip(H + lr * sum(dual), min_horizon, max_horizon)`，
返回新视界和下一次 rollout 的乘子，即 `dual[0]` 重复 `round(H)` 次；
下一次 rollout 运行 `len(dual)` 步。与参考代码和算法 1 一致，约束不在 actor 损失中加项，
接触范数超过 `C` 时乘子变为正值并使视界变长。
参考默认值为 `C = 500`，视界范围 `[8, 64]` 且从 8 开始，`lr = 5e-4` 线性衰减到 `1e-5`。
`critic_converged` 在 `|mean(diff(最近 5 个 epoch 的损失))| < 0.2` 时停止训练 critic，
判断发生在记录当前 epoch 损失之前，最多 64 个 epoch。
`compute_value_loss` 对两个 critic 的 0.5 倍均方误差求和，等于参考实现中两个 critic 平方误差的平均。

## Soft Analytic Policy Optimization

SAPO（[Xing 等，ICLR 2025](https://arxiv.org/abs/2412.12089)；
[参考代码](https://github.com/etaoxing/mineral)）是最大熵版本的 SHAC。
每个奖励都替换为 `r_t + alpha * normalized_entropy_t`，同时用于解析策略回报和 TD(λ) 目标，
因此熵奖励和奖励一起折扣。温度 `alpha` 朝目标熵自动调整。
策略是 tanh 压缩的高斯分布，熵用重参数化动作的单样本估计 `-log_prob(a)`。
两个 critic 取代目标 critic：策略用它们的均值做 bootstrap，价值目标用它们的最小值。

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

`normalize_entropy` 把熵映射为 `(entropy + |target|) / (2 |target|)`，目标熵对应 0。
`compute_alpha_loss` 在原始熵上计算 `|target| * alpha * mean(entropy - target)`，
与参考实现一样，`|target|` 因子抵消了归一化。参考实现在 actor 更新之后、计算价值目标之前更新 `alpha`，
所以价值目标使用新的温度。`compute_value_loss` 是参考实现中各 critic 均方误差的平均，
不带库中其他价值损失的 `0.5`。参考的 DFlex Ant 默认值为：视界 32、目标熵 `-0.5 * action_dim`、
初始 `alpha = 1`、betas 为 `(0.7, 0.95)` 的 AdamW，学习率 `2e-3`（actor）、`5e-4`（critic）、
`5e-3`（alpha），梯度范数裁剪 0.5，固定 16 个 critic epoch、每个 4 个 minibatch，
log-std 限制在 `[-5, 2]`，LayerNorm + SiLU 网络 `[128, 64, 32]`（actor）和 `[64, 64]`（两个 critic）。

## Flow Policy Optimization

FPO 约定 `t=0` 为动作、`t=1` 为高斯噪声。网络预测速度
`dx/dt = noise - action`，Euler 积分从 1 向 0 运行。
`SuperviseTarget.Velocity` 和 `SuperviseTarget.Noise` 都使用输出速度的网络；
后者通过 `x_t + (1 - t) * velocity` 重建噪声。

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

对于大小为 `B`、动作维度为 `A`、每个动作有 `M` 个 CMF 样本的批次，缓存形状为
`actions: [B, A]`、`eps: [B, M, A]`、`time_step: [B, M, 1]`、
`init_cmf_loss: [B, M]`。所有更新应复用相同的 latent action、噪声/时间、
监督目标和观测变换。如果为满足环境范围而变换动作，CMF 计算仍应保存和使用其 latent 值。

默认比率为 `exp(mean(old_loss - new_loss))`，每个动作只有一个比率。
默认 log-ratio 限幅范围为 `(-3, 3)`，在平均之后应用。
`average_losses_before_exp=False` 选择逐样本比率的消融版本。
`approx_kl` 是基于 surrogate ratio 的漂移指标，不是流策略的精确 KL；
`kl_divergence` 是其兼容别名。

采样默认 `deterministic=True`：给定相同初始噪声时积分得到相同动作，不添加额外末端噪声。
重新生成初始噪声仍可提供探索。噪声监督下，单步离散时间采样在 `t=1` 的损失为零，
应改用连续时间或速度监督。观测归一化、奖励缩放、价值裁剪属于调用方控制的训练设置。
[RLDemos FPO 文档](https://github.com/XZDang13/RLDemos#online-examples)
提供这些设置和真实学习实验。

## 其他组件

| 组件 | 用途 |
| --- | --- |
| [GRULayer](RLAlg/nn/layers.py) 与 PPO 循环接口 | 使用 `episode_starts` 重置隐状态，支持初始状态和填充序列掩码 |
| [Normalizer](RLAlg/normalizer.py) | 运行均值/方差；`update(x)` 更新统计，`forward(x)` 默认冻结统计 |
| [GAN](RLAlg/alg/gan.py) | BCE、平滑 BCE、hinge、最小二乘、Wasserstein、生成器损失及梯度惩罚 |
| [MetricsTracker / WandbLogger](RLAlg/logger.py) | 本地指标累计和可选的 W&B 日志 |
| [KLAdaptiveLR](RLAlg/scheduler.py) | 通过 `set_kl` 提供 KL 指标并调整学习率 |
| [set_seed_everywhere / weight_init](RLAlg/utils.py) | 随机种子设置和网络初始化 |

IQL 使用固定数据集，计算价值、策略和双 critic 损失。
调用方提供数据集、目标 critic 和更新调度。本实现要求 `expectile` 在 `[0, 1]` 内、
`temperature` 为正，策略拟合权重为 `exp((Q - V) * temperature)`，最大截断为 100。

## 例子与测试

[RLDemos](https://github.com/XZDang13/RLDemos) 提供可运行的 PPO、DDPG、
DDPGDoubleQ、SAC、DSAC、DSACT、FPO、IQL 例子，包含离线数据采集和 FPO 评估。
其 README 同时提供中英文版本。[tests/test.ipynb](tests/test.ipynb)
展示 FPO 采样和 replay 存储。

在仓库根目录运行测试：

```bash
python -m pip install pytest
python -m pytest -q
```

测试覆盖算法损失、形状检查、终止掩码、循环网络、replay/HER、归一化、日志，
FPO 的采样和梯度，以及 SHAC/AHAC/SAPO 在可微玩具 rollout 上的梯度。

## 许可证

[MIT](LICENSE)。
