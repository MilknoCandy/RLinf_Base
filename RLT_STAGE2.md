# RLT Stage 2 — Actor-Critic RL Training with RLT Feature Model

## 1. Overview

RLT Stage 2 是一个 **off-policy actor-critic RL 训练**阶段。它使用 Stage 1 训练好的 **OpenPI + RLT Module** 作为冻结的 feature model，从环境观测中提取紧凑特征 `z_rl`，然后用一个**小型 MLP actor-critic 头**在 ManiSkill PegInsertion 环境中进行 SAC 风格的在线强化学习。

**核心架构：**
```
环境观测 (images + state)
  │
  └─→ [冻结] Stage1 Pi0 + RLTModule ──→ z_rl [B, 2048]      ← feature model
         │
         ├─→ ref_chunk (Pi0 预测的参考动作 chunk)
         │
         └─→ [可训练] RLT MLP Actor-Critic ← Stage2 头
                ├─ Actor:  [ref_chunk, z_rl, proprio] ──→ action
                ├─ Q1:     [action, z_rl, proprio] ──→ q1
                └─ Q2:     [action, z_rl, proprio] ──→ q2
```

**训练目标 (RLT AC)：**
```
L_actor  = -q_weight * Q1(s, π(s)) + bc_weight * BC_loss(π(s), a_ref)
L_critic = MSE(Q_i(s, a), r + γ * min_j Q_j_target(s'', π(s'')))
```

---

## 2. 代码文件索引

| 文件 | 作用 |
|---|---|
| `examples/embodiment/train_embodied_agent.py` | 入口（同步）/ `train_async.py`（异步） |
| `rlinf/runners/embodied_runner.py` | 同步 embodied runner |
| `rlinf/runners/async_embodied_runner.py` | 异步 embodied runner（支持 disaggregated pipeline） |
| `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` | RLT AC actor worker：损失函数、RLT schedule、replay buffer |
| `rlinf/workers/actor/fsdp_sac_policy_worker.py` | SAC actor worker 基类：replay buffer、SAC update loop |
| `rlinf/workers/actor/fsdp_actor_worker.py` | Embodied FSDP actor 基类 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | Rollout worker：含 RLT feature model + expert model |
| `rlinf/workers/env/env_worker.py` | Env worker：含 RLT transition replay 逻辑 |
| `rlinf/algorithms/rlt/rollout.py` | `predict_rlt_actions()` — RLT rollout 核心函数 |
| `rlinf/algorithms/rlt/route.py` | `SimulatorRLTRoute` / `RealworldRLTRoute` — 动作路由 |
| `rlinf/algorithms/rlt/transition.py` | RLT transition 收集与 replay buffer 写入 |
| `rlinf/algorithms/rlt/expert.py` | `predict_expert_actions()` — expert takeover 推理 |
| `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` | `RLTMLPPolicy` — Stage 2 MLP actor-critic |
| `rlinf/models/embodiment/openpi_rlinf/eval_action_model.py` | `extract_rlt_obs()` — feature extraction |
| `rlinf/models/embodiment/openpi_rlinf/openpi_action_model.py` | `_encode_rlt_flat()` — RLT encoding |
| `rlinf/envs/maniskill/maniskill_rlt_env.py` | `ManiskillRLTEnv` — policy switch、expert takeover |
| `rlinf/envs/maniskill/peg_insertion_side_variants.py` | RLT obs wrapping、event state、env variant 注册 |
| `examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml` | Stage 2 训练配置 |
| `examples/embodiment/config/env/maniskill_rlt.yaml` | RLT 环境配置模板 |

---

## 3. 启动流程

```
train_embodied_agent.py (main)
  │
  ├─ validate_cfg(cfg)
  ├─ Cluster(cluster_cfg)           → Ray 集群
  ├─ HybridComponentPlacement()     → 放置策略
  ├─ 创建 Worker Groups:
  │     ├─ ActorWorker.create_group().launch()     → RLTACFSDPPolicy
  │     ├─ RolloutWorker.create_group().launch()   → HuggingfaceWorker (含 feature model)
  │     └─ EnvWorker.create_group().launch()       → EnvWorker
  ├─ EmbodiedRunner(cfg, actor, rollout, env)
  ├─ runner.init_workers()
  └─ runner.run()
```

### 3.1 Worker 初始化顺序

```
1. EnvWorker.init_worker()
     └─ 创建 ManiskillRLTEnv 实例（含 PegInsertion 变体注册）

2. RolloutWorker.init_worker()
     ├─ 加载主模型（rlt_mlp_policy，从配置中 model_type 决定）
     ├─ 加载 RLT feature model（OpenPI + RLTModule，冻结）
     │     └─ rlt_feature_model = get_model(rlt_feature_model_config)
     │           └─ openpi/eval → extract_rlt_obs()
     │                 └─ Pi0.build_prefix_cache() → prefix embeddings
     │                 └─ RLTModule.encode_flat() → z_rl [B, 2048]
     │                 └─ Pi0.sample_actions() → ref_chunk
     ├─ 构建 RLT route: build_rlt_route(cfg)
     └─ [可选] 加载 expert model（用于 expert takeover）

3. ActorWorker.init_worker()
     ├─ 构建 RLTMLPPolicy（MLP actor-critic）
     ├─ 构建 target model（用于 SAC soft update）
     └─ 初始化 replay buffer
```

---

## 4. 训练循环（同步模式）

```
EmbodiedRunner.run()
  │
  for epoch in range(max_epochs):
    │
    ├─ 1. ROLLOUT 阶段
    │     env_worker.collect_rollout(actor_worker)
    │       ├─ env.reset() → obs
    │       ├─ for step in max_steps:
    │       │     ├─ rollout_worker.predict(obs)     ← RLT rollout
    │       │     │     └─ predict_rlt_actions()     (见 §5)
    │       │     ├─ env.step(actions) → reward, done, info
    │       │     ├─ calculate_adv_and_returns()     ← GAE / SAC advantage
    │       │     └─ 收集 trajectory → replay buffer
    │       └─ 发送 rollout data 到 actor
    │
    ├─ 2. ACTOR UPDATE 阶段
    │     actor_worker.run_training()
    │       ├─ _rlt_updates_to_run()                 ← RLT schedule 控制更新频率
    │       ├─ for update in range(N):
    │       │     ├─ replay buffer 采样 batch
    │       │     ├─ critic update: Q loss (critic_actor_ratio 次)
    │       │     │     ├─ current Q: Q(s, a)
    │       │     │     ├─ target Q: r + γ * min_j Q_target_j(s'', π(s''))
    │       │     │     └─ loss = MSE(Q, target_Q)
    │       │     └─ actor update: actor loss (每 critic_actor_ratio 步一次)
    │       │           ├─ L_q = -q_weight * Q1(s, π(s))
    │       │           ├─ L_bc = bc_weight * MSE(π(s), a_ref)
    │       │           └─ L = L_q + L_bc
    │       └─ soft update target model: θ_target ← τ * θ + (1-τ) * θ_target
    │
    ├─ 3. WEIGHT SYNC 阶段
    │     actor → rollout worker: 同步最新 actor 权重
    │
    └─ 4. CHECKPOINT + EVAL
          save_interval: 保存 checkpoint
          val_check_interval: 运行 evaluation rollout
```

---

## 5. RLT Rollout 详细流程

### 5.1 `predict_rlt_actions()` (`algorithms/rlt/rollout.py`)

这是 RLT Stage 2 最核心的 rollout 函数，每次 env step 调用：

```
predict_rlt_actions(policy_model, feature_model, rlt_route, env_obs, ...)

  Step 1: 提取 RLT 观测
    rlt_obs = feature_model.extract_rlt_obs(env_obs)
      │
      │  feature_model 是 Stage 1 训练好的 OpenPI + RLTModule（冻结）
      │
      ├─ repack env_obs → OpenPI 标准格式
      ├─ Pi0.build_prefix_cache(obs) → prefix embeddings [B, S, D]
      ├─ RLTModule.encode_flat(prefix) → z_rl [B, 2048]
      ├─ Pi0.sample_actions() → ref_chunk (OpenPI 预测的参考动作)
      └─ return {"z_rl": z_rl, "proprio": state, "ref_chunk": ref_chunk}

  Step 2: Stage 2 Actor 推理
    actions, result = policy_model.predict_action_batch(env_obs=rlt_obs)
      │
      │  policy_model = RLTMLPPolicy (可训练的 MLP)
      │
      ├─ actor_state = [ref_chunk, z_rl, proprio]
      ├─ backbone(actor_state) → hidden
      ├─ actor_mean(hidden) → action_mean
      ├─ action = tanh(action_mean + fixed_std * noise)   (SAC 采样)
      └─ return {action, forward_inputs: {z_rl, proprio, ref_chunk}}

  Step 3: 动作路由 (RLT Route)
    route_output = rlt_route.route(ctx)
      │
      │  SimulatorRLTRoute: 决定用谁的动作
      │
      ├─ 判断 critical_phase? (env info: peg 接近 hole)
      ├─ 判断 ready_for_online? (learner 已 warmup)
      │
      ├─ IF critical_phase AND ready_for_online:
      │     → 使用 Stage 2 Actor 的动作 (actor_switch=True)
      │
      ├─ ELIF expert_takeover 触发 (stalled progress):
      │     → 使用 Expert Model (Stage 1 SFT 模型) 的动作
      │     → 标记 intervene_flags = True
      │
      └─ ELSE:
            → 使用 ref_chunk (OpenPI base policy 的参考动作)

  Step 4: 附加 RLT transition 数据
    result["forward_inputs"]["rlt_transition_z_rl"] = z_rl  ← 下一步的 obs
    result["forward_inputs"]["rlt_transition_proprio"] = proprio
    result["forward_inputs"]["rlt_transition_ref_chunk"] = ref_chunk
```

### 5.2 `SimulatorRLTRoute.route()` (`algorithms/rlt/route.py`)

动作路由是 RLT 的关键设计——它实现了从 base policy → RL actor 的渐进切换：

```
输入: RLTRouteContext
  ├─ student_actions: Stage 2 Actor 预测的动作
  ├─ rlt_obs.ref_chunk: OpenPI base policy 的参考动作
  ├─ rlt_switch_flags: env 计算的 critical_phase 标志
  ├─ intervene_requested: env 计算的 expert takeover 标志
  └─ version: learner 的 update 计数（用于 warmup gate）

路由决策树:

  critical_phase = (peg 已抓取) AND (peg_head_x > near_hole_x_min)
  
  IF use_schedule AND version < warmup_updates:
      actor_switch = False            ← warmup: 全部用 base policy
  ELSE:
      actor_switch = critical_phase   ← online: critical phase 才用 actor

  IF expert_takeover requested AND ready AND mode=="train":
      使用 expert_model 动作          ← 人类/expert 干预
      actor_switch_mask = False       ← 标记为 expert 动作（非 actor）
  ELSE:
      使用 torch.where(actor_switch, student_actions, ref_chunk)

输出: RLTRouteOutput
  ├─ actions: 最终执行的动作
  └─ result.forward_inputs: 含 actor_switch、intervene_flags、record_transition
```

---

## 6. RLT 环境：`ManiskillRLTEnv` (`maniskill_rlt_env.py`)

### 6.1 观测包装

`_wrap_obs()` — 在 `wrap_obs_mode == "rlt_openpi_joint"` 时：

```
raw ManiSkill obs → wrap_rlt_openpi_joint_obs()
  ├─ 提取 wrist camera + 3rd-view camera 图像 (384×384 RGB)
  ├─ 提取关节状态 (9-dim: joint positions)
  ├─ 构建 task prompt: "insert the peg in the hole"
  └─ 返回 {"images": {...}, "state": [...], "prompt": "..."}
```

### 6.2 Policy Switch 逻辑

`step()` 每步调用 `_update_rlt_switch()` 来决定是否激活 Stage 2 actor：

```
_update_rlt_switch(infos)
  │
  ├─ _rlt_auto_enter_actor(infos)
  │     ├─ grasped = (infos["consecutive_grasp_current"] > 0)
  │     ├─ near_hole = (peg_head_x > near_hole_x_min)
  │     └─ return grasped AND near_hole        ← 进入 critical phase
  │
  ├─ latch_until_done:
  │     rlt_switch_flags = previous_flags | enter_actor  ← 一旦进入就不再退出
  │
  └─ expert takeover check (_rlt_expert_takeover_mask)
        ├─ trigger_mode == "stalled_progress":
        │     跟踪 peg 插入进展 (x, yz distance)
        │     如果 stuck_chunks > threshold → 请求 expert
        └─ trigger_mode == "critical_phase": 进入 critical phase 就请求

输出: {rlt_switch_flags, intervene_flag, ...}
```

### 6.3 Expert Takeover 进展跟踪

`_update_rlt_stalled_progress_expert_takeover()` — 跟踪每个 env 的插入进展：

```
每个 action chunk 后:
  progress_score = peg_head_x - yz_weight * sqrt(y^2 + z^2)

  IF progress_improved (x 前进 / yz 对齐改善):
      reset stalled_chunks = 0
  ELSE:
      stalled_chunks += 1

  IF stalled_chunks >= stuck_chunks_before_takeover (默认 3):
      触发 expert takeover → 下一个 step 用 expert 动作
```

---

## 7. Stage 2 Actor：`RLTMLPPolicy` (`rlt_mlp_policy.py`)

### 7.1 模型结构

```
RLTMLPPolicy (继承 MLPPolicy)
  │
  ├─ Actor 输入: [ref_chunk (80,) + z_rl (2048,) + proprio (9,)] = 2137-dim
  │     └─ backbone (MLP) → hidden
  │     └─ actor_mean (Linear) → action (80-dim = 10 chunks × 8 dof)
  │     └─ fixed_std = 0.002 (确定性略加噪声)
  │
  └─ Critic (Twin Q):
        ├─ Q1: [action (80,) + z_rl (2048,) + proprio (9,)] → q1
        └─ Q2: [action (80,) + z_rl (2048,) + proprio (9,)] → q2
```

关键点：Critic 输入**不含 ref_chunk**，只含 `[action, z_rl, proprio]`。

### 7.2 关键方法

**`_actor_state(obs, reference_dropout_prob)`** — 拼接 `[ref_chunk, z_rl, proprio]` 作为 actor 输入。
- `reference_dropout_prob=0.5` 时随机将 50% 样本的 ref_chunk 置零 → 防止过度依赖 reference

**`_critic_state(obs)`** — 拼接 `[z_rl, proprio]` 作为 critic 输入。

**`sac_forward(obs, ...)`** — SAC actor 前向：输出 action mean + fixed std → TanhNormal 采样

**`sac_q_forward(obs, actions)`** — Q 网络前向

**`predict_action_batch(env_obs, mode)`** — 推理接口
- `mode=="train"`: 随机采样（SAC exploration）
- `mode=="eval"`: 确定性动作（取 mean）

---

## 8. RLT Actor Worker：`RLTACFSDPPolicy` (`fsdp_rlt_ac_policy_worker.py`)

### 8.1 继承链

```
RLTACFSDPPolicy
  ├─ RLTACLossMixin      ← RLT 特有的损失函数
  ├─ RLTACReplayMixin    ← RLT schedule + replay buffer 逻辑
  └─ EmbodiedSACFSDPPolicy  ← SAC update loop (replay buffer, target network)
```

### 8.2 RLT 损失函数 (`RLTACLossMixin`)

**Actor Loss:**
```python
L_actor = -q_weight * Q1(s, π(s)) + bc_weight * MSE(π(s), a_ref_chunk)
```
- `q_weight` / `bc_weight` 由 schedule 控制（warmup → online transition）
- 前 20000 步 warmup: `bc_weight=7.0, q_weight=0.05`（强调 BC）
- 50000 步 ramp 后: `bc_weight=2.5, q_weight=0.45`（强调 Q 优化）

**Critic Loss:**
```python
target_q = r + γ * min_j Q_target_j(s'', π(s''))
L_critic = MSE(Q_j(s, a), target_q)
```
- 使用 clipped double Q-learning（取两个 Q 的最小值）

**BC 正则化:**
- BC target 优先使用 human/expert 动作 (intervene_flags=True 时)
- 否则使用 ref_chunk（OpenPI base policy 动作）

### 8.3 RLT Training Schedule (`RLTACReplayMixin`)

`_rlt_updates_to_run()` — 控制何时以及多少次 actor 更新：

```
IF NOT use_rlt_schedule: 直接返回 super().run_training()

计算 desired updates:
  IF train_every_transitions > 0:
      desired = accumulated_transitions / train_every_transitions
  IF train_every_episodes > 0:
      desired = completed_episodes / train_every_episodes

clamp to max_updates_per_train_step (默认 400)

检查 warmup:
  IF total_transitions < warmup_min_size (10000): skip
  IF warmup_post_collect_updates (30000) not reached: skip

IF updates_to_run > 0:
    执行 N 次 SAC update（critic_actor_ratio=4，4:1 critic:actor）
    update_step++ (用作 warmup gate)
```

### 8.4 Replay Buffer

RLT 使用 `TrajectoryReplayBuffer`：
- `cache_size: 10000` — 内存缓存大小
- `sample_window_size: 50000` — 采样窗口
- `min_buffer_size: 1000` — 开始训练的最小数据量

---

## 9. RLT Transition 收集 (`algorithms/rlt/transition.py`)

### 9.1 Transition 结构

每个 RLT transition 包含：
```
{"z_rl": ..., "proprio": ..., "ref_chunk": ...}
```
即 `RLT_OBS_KEYS = ("z_rl", "proprio", "ref_chunk")`。

### 9.2 收集流程

`update_rlt_transitions()` — env worker 在每个 action chunk 边界调用：

```
stage_0: pending_obs[0] = {z_rl_t, proprio_t, ref_chunk_t}
stage_1: 下一帧到达

IF pending_obs[0] is not None:
    next_obs = extract from result.forward_inputs (rlt_transition_*)
    trajectory_builder.append_transitions(pending_obs[0], next_obs)
    pending_obs[0] = None

IF cache_current:
    pending_obs[0] = {z_rl, proprio, ref_chunk}  (当前帧)
```

### 9.3 Expert Intervention 的 Transition 修正

当 expert takeover 触发时，`ref_chunk` 被替换为 expert 动作：
```
IF intervene_flags.any():
    ref_actions[current_obs["ref_chunk"]] = expert_actions
```
这确保了 replay buffer 中 BC target 指向专家动作。

---

## 10. Stage 1 → Stage 2 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        Stage 2 数据流                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ManiSkill Env                                                   │
│    │                                                             │
│    │  raw obs (images + qpos)                                    │
│    ▼                                                             │
│  ManiskillRLTEnv._wrap_obs()                                     │
│    │  repack → {"images": [wrist, 3rd], "state": joints,         │
│    │            "prompt": "insert the peg..."}                   │
│    ▼                                                             │
│  feature_model.extract_rlt_obs()                                 │
│    │  ┌──────────────┐                                           │
│    │  │ Stage1 Pi0    │  build_prefix_cache(obs)                 │
│    │  │  (frozen)     │  → prefix embeddings                    │
│    │  ├──────────────┤                                           │
│    │  │ Stage1 RLT    │  encode_flat(prefix)                     │
│    │  │  Module       │  → z_rl [B, 2048]                       │
│    │  │  (frozen)     │                                         │
│    │  ├──────────────┤                                           │
│    │  │ Stage1 Pi0    │  sample_actions()                        │
│    │  │  (frozen)     │  → ref_chunk [B, 10, 8]                 │
│    │  └──────────────┘                                           │
│    │                                                             │
│    │  return {"z_rl": z_rl, "proprio": state, "ref_chunk": ...} │
│    ▼                                                             │
│  Stage2 RLTMLPPolicy.predict_action_batch()                      │
│    │  ┌──────────────┐                                           │
│    │  │ Actor MLP     │  [ref_chunk, z_rl, proprio] → action     │
│    │  │  (trainable)  │                                          │
│    │  └──────────────┘                                           │
│    │                                                             │
│    ▼                                                             │
│  SimulatorRLTRoute.route()                                       │
│    │  决定最终动作: actor / ref / expert                         │
│    ▼                                                             │
│  env.step(final_actions)                                         │
│    │  → reward, done, info (含 rlt_switch_flags)                 │
│    ▼                                                             │
│  update_rlt_transitions()                                        │
│    │  存储 (z_rl_t, proprio_t, ref_chunk_t) → replay buffer      │
│    ▼                                                             │
│  Actor: SAC update                                               │
│    │  从 replay buffer 采样 → Q loss + Actor loss                │
│    ▼                                                             │
│  soft update: θ_target ← τ*θ + (1-τ)*θ_target                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. 配置解读

### 11.1 核心配置 (`maniskill_rlt_stage2_ac_mlp.yaml`)

```yaml
algorithm:
  loss_type: rlt_ac                    # RLT Actor-Critic
  adv_type: embodied_sac               # SAC advantage 计算
  gamma: 0.96                          # 折扣因子
  tau: 0.005                           # target network soft update
  
  q_weight: 1.0                        # Q loss 权重
  bc_weight: 1.0                       # BC loss 权重
  reference_dropout_prob: 0.5          # ref_chunk dropout 概率
  
  actor_weight_schedule:               # Q/BC 权重调度
    enable: True
    warmup_updates: 20000              # warmup 阶段纯 BC
    ramp_updates: 50000                # ramp 阶段逐渐增加 Q 权重
    warmup_bc_weight: 7.0
    warmup_q_weight: 0.05
    online_bc_weight: 2.5
    online_q_weight: 0.45
  
  rlt_schedule:                        # RLT 更新频率控制
    enable: True
    max_updates_per_train_step: 400    # 每步最多更新次数
    warmup_min_size: 10000             # 最小 buffer 大小
    warmup_post_collect_updates: 30000 # warmup 更新数
    train_every_transitions: 5         # 每 5 个 transition 触发一次更新
  
  replay_buffer:                       # Replay buffer 参数
    cache_size: 10000
    sample_window_size: 50000
    min_buffer_size: 1000
  
  target_update_freq: 1                # 每次更新都 soft update
  critic_actor_ratio: 4                # Q 更新:Actor 更新 = 4:1

actor:
  model:
    model_type: "rlt_mlp_policy"       # RLT MLP 头
    z_dim: 2048                        # RLT feature 维度
    proprio_dim: 9                     # 关节状态维度
    action_dim: 8                      # 动作维度
    num_action_chunks: 10              # action chunk 长度
    add_q_head: True                   # 启用 Critic
    fixed_std: 0.002                   # SAC 固定标准差

rollout:
  rlt_feature_model:                   # Stage1 feature model
    model_type: "openpi"              # 使用 Stage1 训练好的 OpenPI
    model_path: "/path/to/stage1/checkpoint"
    openpi:
      config_name: "pi05_rlt_maniskill_joint"
      use_rlt: True                   # 启用 RLT Module
      ...

  expert_model:                        # Expert takeover 模型
    model_path: "/path/to/sft_model"
    openpi:
      use_rlt: False                  # 不需要 RLT module

env:
  train:
    env_type: maniskill_rlt           # RLT 环境类型
    wrap_obs_mode: rlt_openpi_joint   # RLT 观测包装
    rlt_policy_switch:                # Policy switch 配置
      enable: True
      task_mode: full_task            # 全任务模式
      trigger_mode: auto              # 自动进入 critical phase
      latch_until_done: True          # 一旦进入就保持
      auto_gate:
        require_grasp: True
        require_not_success: True
        near_hole_x_min: -0.16
      expert_takeover:
        enable: True                  # 启用 expert takeover
        trigger_mode: stalled_progress # 进展停滞时触发
```

---

## 12. 函数衔接图

```
EmbodiedRunner.run()
  │
  ├─ EnvWorker.collect_rollout()
  │     ├─ ManiskillRLTEnv.step()
  │     │     ├─ BaseEnv.step(actions) → raw_obs, reward, info
  │     │     ├─ _update_rlt_switch(infos)            ← 计算 critical_phase + expert takeover
  │     │     │     ├─ _rlt_auto_enter_actor()        ← 判断是否进入 critical phase
  │     │     │     └─ _rlt_expert_takeover_mask()    ← 判断是否需要 expert
  │     │     └─ _wrap_obs(raw_obs) → OpenPI format   ← rlt_openpi_joint
  │     │
  │     └─ HuggingfaceWorker.predict(obs)
  │           └─ predict_rlt_actions()
  │                 ├─ feature_model.extract_rlt_obs(env_obs)
  │                 │     ├─ Pi0.build_prefix_cache() → prefix embeddings
  │                 │     ├─ RLTModule.encode_flat() → z_rl
  │                 │     └─ Pi0.sample_actions() → ref_chunk
  │                 │
  │                 ├─ policy_model.predict_action_batch(rlt_obs)
  │                 │     └─ RLTMLPPolicy.sac_forward()
  │                 │           ├─ _actor_state() → [ref_chunk, z_rl, proprio]
  │                 │           └─ backbone → action_mean → Tanh → action
  │                 │
  │                 ├─ rlt_route.route(ctx)
  │                 │     └─ SimulatorRLTRoute.route()
  │                 │           ├─ critical_phase? → actor_switch
  │                 │           ├─ ready_for_online? → warmup gate
  │                 │           ├─ expert_takeover? → expert_actions
  │                 │           └─ torch.where → final_actions
  │                 │
  │                 └─ _append_rlt_transition_obs()
  │                       └─ result["forward_inputs"]["rlt_transition_*"] = z_rl, proprio, ref_chunk
  │
  ├─ ActorWorker.run_training()
  │     └─ RLTACFSDPPolicy.run_training()
  │           ├─ _rlt_updates_to_run()                ← 计算需要多少次更新
  │           └─ for update in range(updates_to_run):
  │                 └─ update_one_epoch()
  │                       ├─ replay buffer 采样 batch
  │                       ├─ critic update:
  │                       │     ├─ sac_q_forward(obs, actions) → current_Q
  │                       │     ├─ target_model.sac_q_forward(next_obs, next_actions) → target_Q
  │                       │     ├─ target_q = reward + γ * min(target_Q1, target_Q2)
  │                       │     └─ loss = MSE(current_Q, target_q)
  │                       └─ actor update (每 critic_actor_ratio 步):
  │                             ├─ sac_forward(obs) → actions
  │                             ├─ Q1(s, actions)
  │                             ├─ L_q = -q_weight * Q1
  │                             ├─ L_bc = bc_weight * MSE(actions, ref_chunk)
  │                             └─ loss = L_q + L_bc
  │
  └─ WeightSyncer.sync()
        └─ actor → rollout worker: 同步最新权重
```

---

## 13. 关键设计要点

### 13.1 为什么分 Stage 1 和 Stage 2？

| | Stage 1 | Stage 2 |
|---|---|---|
| 训练模型 | Pi0.5 (2.5B) + RLTModule | MLP (~几M) |
| 训练方式 | SFT（监督学习） | SAC RL（强化学习） |
| 数据 | 离线数据集 | 在线交互 |
| 计算量 | 大（需要 SGLang/vLLM 推理） | 小（MLP 推理极快） |
| 目标 | 学习紧凑的视觉-语言-动作表征 | 在线 fine-tune 策略 |

分离的好处：Stage 2 只需要运行轻量 MLP，feature model 冻结，推理成本极低，支持高频在线交互。

### 13.2 Warmup Schedule

RLT 使用渐进式 warmup，避免 RL 早期的不稳定：

1. **Warmup 阶段** (steps 0-20000): 纯 BC，`bc_weight=7.0, q_weight=0.05`
2. **Ramp 阶段** (steps 20000-70000): 逐渐增加 Q 权重
3. **Online 阶段** (steps 70000+): BC+Q 联合优化

同时 `SimulatorRLTRoute` 在 `version < warmup_updates` 时始终使用 base policy(ref_chunk)，不给 actor 控制权。

### 13.3 Expert Takeover

模拟真实世界的人类干预机制：
- 当 peg 在 critical phase 但插入进展停滞时，自动切换到 expert model
- Expert 动作替代 ref_chunk → BC target 指向更优动作
- 仅在 train mode 触发，eval 始终禁用

### 13.4 为什么 Actor Critic 输入不含 ref_chunk？

Critic 评估的是 **状态-动作对** 的价值，不应依赖参考动作。`critic_state = [z_rl, proprio]`（不含 ref_chunk）确保 Q 值纯粹反映当前状态。
