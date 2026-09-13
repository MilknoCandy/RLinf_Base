# RLT + 固定短期记忆（Step 2A）

## 范围

本文档描述 `RLT_Hierarchical_RL_Memory_Step2_Design.md` 中 **Step 2A** 的实现。

Step 2A 冻结 VLM / RLT 表示与 RL-token encoder，只新增可训练的短期记忆头：

```text
o_t -> frozen RLT -> z_t -> STM read -> m_t -> z'_t = z_t + W_m m_t -> Actor/Critic
```

记忆条目为 `e_t = (z_t, a_t, r_t)`；读取器是固定大小的近期窗口 `K`；编码器为 GRU；
融合方式为残差。由于 `W_m` 初始化为零，模型可严格退化为 RLT baseline。

## 入口

RLinf 没有 `train_sync.py` 入口。RLT Stage 2 有两个官方 embodied 入口：

- `examples/embodiment/train_async.py` — 异步 / disaggregated 管线，rollout 与训练重叠，
  是 RLT Stage 2 的高效模式。
- `examples/embodiment/train_embodied_agent.py` — 同步入口，rollout 与训练串行。
  两者都会根据 `loss_type: rlt_ac` 分派到带 replay buffer 的 off-policy SAC，并非
  on-policy PPO。

下面命令统一使用异步入口以获得吞吐。同步配置
（`maniskill_rlt_stage2_ac_mlp.yaml`、`maniskill_rlt_stage2_stm_ac_mlp.yaml`）仍保留，
用于与同步 RLT baseline 对齐。

## 实现位置

- `rlinf/algorithms/rlt/memory/buffer.py` — `STMBuffer`（read / write / reset；recent /
  shuffled / random 三种读取；`(z, a, r)` 条目组合）。
- `rlinf/algorithms/rlt/memory/encoder.py` — `GRUMemoryEncoder`。
- `rlinf/algorithms/rlt/memory/fusion.py` — `ResidualMemoryFusion`。
- `rlinf/algorithms/rlt/memory/module.py` — `RLTSTMModule` 与 `build_memory_module`。
- `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` — Actor 与 Critic 都使用融合后
  的 token `z'_t`。
- `rlinf/workers/rollout/hf/huggingface_worker.py` 与
  `rlinf/workers/env/env_worker.py` — rollout/eval 时维护 per-env STM buffer。
- `examples/embodiment/config/maniskill_rlt_stage2_stm_ac_mlp_async.yaml` — 可运行的
  异步配置；`maniskill_rlt_stage2_stm_ac_mlp.yaml` 是对应的同步配置。

## 运行

Baseline（RLT，无记忆）：

```bash
python examples/embodiment/train_async.py \
  --config-name maniskill_rlt_stage2_ac_mlp_async
```

Step 2A 主实验（RLT + 固定 STM，recent，`K=8`，`(z,a,r)` 条目）：

```bash
python examples/embodiment/train_async.py \
  --config-name maniskill_rlt_stage2_stm_ac_mlp_async
```

## 递进消融矩阵

同一配置可通过 Hydra overrides 覆盖 Step 2A 消融矩阵。

| 实验 | Overrides | 回答的问题 |
|---|---|---|
| A. RLT baseline | `maniskill_rlt_stage2_ac_mlp_async` | 参照 |
| B. RLT + recent STM | `maniskill_rlt_stage2_stm_ac_mlp_async` | 近期历史是否有用？ |
| B-K4 / B-K16 | `algorithm.memory.window_size=4` / `=16` | 窗口长度敏感性 |
| C. random context | `algorithm.memory.retrieval=random` | 排除“只是增加参数/上下文” |
| D. shuffled context | `algorithm.memory.retrieval=shuffled` | 时间顺序是否重要？ |
| E. state-only entry | `algorithm.memory.entry.action=False algorithm.memory.entry.reward=False` | `(a,r)` 是否必要？ |

例如：

```bash
python examples/embodiment/train_async.py \
  --config-name maniskill_rlt_stage2_stm_ac_mlp_async \
  algorithm.memory.window_size=4
```

## 因果性

rollout 路径遵循 `read -> act -> write`：

1. 用上一个 chunk 的 reward 补全上一个 pending `(z_{t-1}, a_{t-1})`，
2. 读取 `M_t = {e_0, ..., e_{t-1}}`，
3. 基于 `z'_t` 计算 Actor/Critic 动作，
4. 将 `(z_t, a_t)` 存为下一个 pending 条目。

episode 结束会清空 buffer，因此记忆不会跨 episode 泄漏，也不会读到当前动作的 reward。
