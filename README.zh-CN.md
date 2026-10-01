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

DDPG、DDPGDoubleQ、SAC 还支持 actor/critic 使用不同观测的非对称接口，
以及多个 critic 加权的策略损失。SAC、DSAC、DSACT 提供
`compute_alpha_loss`，优化用的张量放在 `"alpha_loss"` 键中。
策略、价值、critic 的损失字典使用 `"loss"` 键，不同方法还返回相应指标。

目标网络的创建和更新调度由调用方负责。相应的
`update_target_param(model, model_target, tau)` 方法按
`(1 - tau) * target + tau * source` 更新目标参数。

DSACT 将运行统计保存在类属性 `q1_mean_std` 和 `q2_mean_std` 中。
开始独立训练前，将两者设为 `None`。

## 模型接口

网络返回 [steps.py](RLAlg/nn/steps.py) 中定义的数据类。
输入和目标需要使用一致的设备、数据类型和批次维度。
对于大小为 `B` 的普通批次，标量价值、奖励、对数概率和优势通常使用
`[B]` 形状，而不是 `[B, 1]`。

| 模型输出 | 主要字段 | 常见用途 |
| --- | --- | --- |
| `DiscretePolicyStep` | `pi`、`action`、`log_prob`、`entropy` | 离散 PPO 或 IQL 策略 |
| `StochasticContinuousPolicyStep` | `pi`、`action`、`log_prob`、`mean`、`log_std`、`entropy` | 连续 PPO、SAC、DSAC、DSACT 或 IQL 策略 |
| `DeterministicContinuousPolicyStep` | `pi`、`mean` | DDPG 或 DDPGDoubleQ 策略 |
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

`GaussianHead` 提供连续策略，可使用带缩放的 tanh 动作范围。
默认 `std_parameterization="log_std"` 学习对数标准差；
`std_parameterization="std"` 配合 `init_std` 可直接学习标准差。
其他输出头和网络构建接口见 [layers.py](RLAlg/nn/layers.py)。
`NormPosition.NONE`、`PRE`、`POST` 控制归一化的位置。

## 经验回放、GAE 与 episode 边界

`ReplayBuffer(num_envs, steps, device)` 按 `[T, N, ...]` 保存张量，
`T` 为保留步数，`N` 为环境数量。添加记录前先为每个字段创建存储。
每条记录包含各环境的一次 transition，总容量为 `steps × num_envs`。

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

`sample_batch` 有放回地采样；`sample_batchs`（接口使用此拼写）遍历打乱后的
transition 小批次。`reset` 清空 rollout，`save` 和 `load` 保存、读取缓冲区。
`sample_sequence_batches` 返回时间优先的序列块、`valid_mask` 和可选的初始状态字段。
循环策略的 on-policy 更新应先采集一段新的 rollout，再采样序列。

`compute_gae` 返回回报和归一化后的优势，同时将 `terminated`、`truncated`
视为 episode 边界。优势归一化需要至少两条 transition。
若希望超时从真实末状态 bootstrap，请先自行给相应奖励加上
`gamma * V(final_obs)`，并保持 `bootstrap_timeouts=False`。
可选的 `True` 模式采用当前状态价值约定，在超时处加上
`gamma * values[t]`；不要同时应用两种修正。
Off-policy 方法若希望超时从真实下一观测 bootstrap，则 `done` 应仅表示真正终止。

对于目标条件任务，[HindsightExperienceReplay](RLAlg/buffer/her.py)
负责重标记目标，并可重新计算奖励和终止标记。
`ReplayBuffer.sample_batch(..., her_strategies=...)` 为
`future`、`final`、`episode`、`random` 策略提供候选目标。

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
以及 FPO 的采样和梯度。

## 许可证

[MIT](LICENSE)。
