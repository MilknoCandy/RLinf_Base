# RLT Stage 1 — SFT Training with OpenPI + RLT Module

## 1. Overview

RLT (Reinforcement Learning with Teacher) Stage 1 是对 OpenPI Pi0.5 VLA 模型进行 **有监督微调 (SFT)**，并在 VLM prefix 之上附加一个 **RLT Token 自编码器 (RLTTokenTransformer)**，通过自重建损失学习将 prefix embeddings 压缩为紧凑的 `z_rl` 特征。

**核心目标：**
- 在 ManiSkill PegInsertion 数据上微调 Pi0.5，保持 flow-matching 行为克隆能力
- 训练 RLT Module（encoder-decoder transformer），将 VLM 输出的 prefix embeddings 压缩为低维特征 `z_rl`
- 产出 checkpoint → Stage 2 的 RLT Actor-Critic RL 训练的 feature model

**总损失函数：**
```
loss = rlt_loss + rlt_alpha * vla_loss
```
其中 `vla_loss` 是 Pi0 的 flow-matching MSE，`rlt_loss` 是 RLT Module 的自重建 MSE。

---

## 2. 代码文件索引

| 文件 | 作用 |
|---|---|
| `examples/sft/train_vla_sft.py` | SFT 训练入口 |
| `rlinf/runners/sft_runner.py` | SFT 训练循环 |
| `rlinf/workers/sft/fsdp_sft_worker.py` | FSDP SFT Worker 基类 |
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | VLA SFT Worker（dispatch dataloader） |
| `rlinf/models/embodiment/openpi_rlinf/__init__.py` | get_model 入口，加载 Pi0 + 选择 task wrapper |
| `rlinf/models/embodiment/openpi_rlinf/sft_action_model.py` | SFT forward：flow-matching loss + RLT loss |
| `rlinf/models/embodiment/openpi_rlinf/openpi_action_model.py` | 基类：封装 Pi0 + RLTModule |
| `rlinf/models/embodiment/openpi_rlinf/utils/model_builders.py` | _build_sft_model / _build_rl_model / _build_eval_model |
| `rlinf/models/embodiment/openpi_rlinf/utils/rlt_utils.py` | RLT 配置、checkpoint 加载/转换 |
| `rlinf/models/embodiment/modules/rlt_token_transformer.py` | RLTTokenTransformer：encoder-decoder 自重建 |
| `rlinf/models/embodiment/openpi_rlinf/pi0_model/pi0.py` | Pi0 核心：flow-matching 前向/采样 |
| `rlinf/models/embodiment/openpi_rlinf/pi0_model/pi0_config.py` | Pi0Config 模型形状定义 |
| `rlinf/models/embodiment/openpi_rlinf/pi0_model/model.py` | Observation 数据结构 |
| `rlinf/models/embodiment/openpi_rlinf/pi0_model/gemma.py` | Gemma LLM decoder 实现 |
| `rlinf/models/embodiment/openpi_rlinf/pi0_model/siglip.py` | SigLIP 视觉编码器 |
| `rlinf/models/embodiment/openpi/dataconfig/__init__.py` | openpi TrainConfig 注册表 + get_openpi_config |
| `rlinf/models/embodiment/openpi/dataconfig/maniskill_rlt_dataconfig.py` | ManiSkill RLT 数据配置 |
| `rlinf/data/datasets/openpi_rlinf/__init__.py` | SFT dataloader dispatch |
| `rlinf/data/datasets/openpi_rlinf/official_sft_data_loader.py` | 官方 openpi data_loader 适配（RLT 使用） |
| `rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py` | FSDP checkpoint → HuggingFace 转换（Stage 1→2 桥接） |
| `examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml` | Stage 1 训练配置 |
| `examples/sft/config/model/pi0_5_rlinf.yaml` | Pi0.5 模型配置模板 |
| `tests/e2e_tests/sft/maniskill_rlt_stage1_sft_openpi_pi05.yaml` | Stage 1 CI e2e 测试配置 |

---

## 3. 启动流程

```
train_vla_sft.py (main)
  │
  ├─ validate_cfg(cfg)            → 校验配置
  ├─ Cluster(cluster_cfg)         → 创建 Ray 集群管理器
  ├─ HybridComponentPlacement()   → 计算 actor 放置策略
  ├─ FSDPVlaSftWorker.create_group(cfg).launch(...)
  │     └─ 启动 N 个 Ray actor，每个 = 1 个 SFT worker
  ├─ SFTRunner(cfg, actor_group)  → 创建 runner
  ├─ runner.init_workers()        → 初始化模型 + optimizer
  └─ runner.run()                 → 训练循环
```

### 3.1 入口: `train_vla_sft.py`

```python
@hydra.main(config_path="config", config_name="maniskill_rlt_stage1_sft_openpi_pi05")
def main(cfg):
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_group = FSDPVlaSftWorker.create_group(cfg).launch(...)
    runner = SFTRunner(cfg=cfg, actor=actor_group)
    runner.init_workers()
    runner.run()
```

### 3.2 SFT Runner: `sft_runner.py`

**`SFTRunner.__init__`** — 初始化 metric_logger、max_steps、early_stop 控制器。

**`init_workers()`** — 调用 `actor.init_worker()` 初始化模型和 optimizer；支持 resume。

**`run()`** — 主训练循环：
```
for step in range(max_steps):
    actor_metrics = actor.run_training().wait()   # 一个 gradient accumulation 周期
    if save_interval:  _save_checkpoint()          # 保存 FSDP checkpoint
    if val_check_interval:  eval_metrics = actor.run_eval().wait()
    metric_logger.log(...)
```

**`_save_checkpoint()`** — 保存到 `{log_path}/{exp_name}/checkpoints/global_step_{N}/actor/`。

---

## 4. 模型加载链路

```
FSDPVlaSftWorker.init_worker()
  → FSDPSftWorker.model_provider_func()
    → get_model(cfg.actor.model)                     # rlinf/models/__init__.py
      → 根据 model_type="openpi_rlinf" 分发到
      → openpi_rlinf/__init__.py : get_model(cfg)
```

### 4.1 `openpi_rlinf/__init__.py` : `get_model()`

这是整个 Stage 1 模型加载的核心，按以下步骤执行：

1. **解析配置** — 从 `cfg.openpi` 读取 `pi05`、`paligemma_variant`、`action_expert_variant` 等构造 `Pi0Config`
2. **创建 Pi0 核心模型** — `pi0_config.create()` 构建完整的 Pi0 模型（SigLIP + Gemma LLM + Action Expert + 投影层）
3. **加载 base checkpoint** — `load_base_safetensors()` 从 `model.safetensors` 加载预训练权重（支持自动转换旧版 OpenPI 格式到新 key layout）
4. **选择 task wrapper** — 根据 `cfg.openpi.task` 分发：
   - `"sft"` → `_build_sft_model()` → `OpenPiPytorchSFTActionModel`
   - `"rl"` → `_build_rl_model()` → `OpenPiPytorchRLActionModel`
   - `"eval"` → `_build_eval_model()` → `OpenPiPytorchEvalActionModel`
5. **加载 wrapper checkpoint**（可选）— 如果路径包含 `full_weights.pt`，调用 `load_full_wrapper_weights()` 加载之前训练好的完整 wrapper（含 RLT module 权重）

### 4.2 Pi0 模型架构

```
Observation (images + state + prompt)
  │
  ├─ SigLIP  ──────────→ image embeddings
  ├─ PointNet ─────────→ state embeddings   (optional)
  ├─ Token Embedding ──→ language tokens
  │
  └─ [prefix embeddings]  ──→  Gemma LLM  ──→  suffix output
                                   │
  noisy_actions + time ───────────┘
                                   │
                     action_out_proj ──→ velocity prediction v_t
                                           │
                                    loss = MSE(v_t, noise - actions)
```

关键子模块：
- **`embed_prefix()`** — 将图像（SigLIP）、语言、点云投影为 prefix token embeddings
- **`embed_suffix()`** — 将噪声动作 + 时间编码投影为 suffix token embeddings
- **`llm()`** — Gemma decoder 处理 [prefix, suffix] tokens
- **`velocity_from_suffix()`** — 从 suffix hidden states 投影出速度预测
- **`compute_loss()`** — Flow-matching 损失计算

---

## 5. SFT Action Model: `sft_action_model.py`

### 5.1 `OpenPiPytorchSFTActionModel` (继承 `OpenPiPytorchActionModel`)

**`__init__(pi0_model, num_steps, action_env_dim, rlt_cfg)`**
- 存储 Pi0 模型引用和 RLT 配置
- 如果 `rlt_cfg.use_rlt=True`，基类构造 `RLTTokenTransformer`

**`forward(forward_type=ForwardType.SFT, **kwargs)`** — 只支持 SFT forward。

**`sft_forward(data)`** — 核心训练前向：
- 解包 `(observation, actions)` 从 data batch
- 如果 `use_rlt=False`：直接调用 `pi0_model.compute_loss()` → 纯 VLA loss
- 如果 `use_rlt=True`：
  1. `_sft_forward_with_rlt_prefix()` — 计算 flow-matching loss，同时截取 prefix hidden states
  2. `_rlt_forward(prefix_output, prefix_mask)` — RLT Module 自重建 → `rlt_loss`
  3. 返回 `{"loss": rlt_loss + rlt_alpha * vla_loss, "vla_loss": ..., "rlt_loss": ...}`

### 5.2 `_sft_forward_with_rlt_prefix(observation, actions)`

这是 `use_rlt=True` 时的关键前向，与普通 `compute_loss` 的区别是它**手动展开**了 flow-matching 的内部步骤以截取 prefix hidden states：

```
1. preprocess_observation → embed_prefix → prefix_tokens
2. embed_suffix(noisy_actions, time) → suffix_tokens
3. llm([prefix_tokens, suffix_tokens]) → (prefix_out, suffix_out)
4. velocity_from_suffix(suffix_out[:, -action_horizon:]) → v_t
5. vla_loss = MSE(v_t, noise - actions)  per-timestep
6. _select_rlt_prefix_embeddings() → 截取 prefix_out（可选仅保留 image tokens）
7. return vla_loss, prefix_out, prefix_mask
```

### 5.3 基类 `OpenPiPytorchActionModel` (`openpi_action_model.py`)

提供：
- **`_rlt_forward(prefix_output, prefix_mask)`** — 调用 `RLTTokenTransformer.forward()` 计算自重建损失
- **`_encode_rlt_flat(prefix_output, prefix_mask)`** — 编码 prefix 为扁平 `z_rl`（Stage 2 推理时使用）
- **`_select_rlt_prefix_embeddings()`** — 根据 `rlt_image_only` 决定是否排除 language tokens
- **`gradient_checkpointing_enable/disable()`** — 透传到 Pi0 模型
- **`_mark_fsdp_wrap_names()`** — 标记模块名供 FSDP lambda policy 使用

---

## 6. RLT Token Transformer: `rlt_token_transformer.py`

### 6.1 整体结构

```
RLTTokenTransformer
  ├─ RLTTokenEncoder
  │     ├─ input_proj:  Linear(input_dim → embed_dim)
  │     ├─ pos_embed:   Sinusoidal PE
  │     ├─ rl_token:    learnable [1, embed_dim]
  │     ├─ layers:      [RLTSelfAttentionLayer × num_layers]
  │     └─ output_proj: Linear(embed_dim → embed_dim)
  │
  └─ RLTTokenDecoder
        ├─ input_proj:  Linear(input_dim → embed_dim)
        ├─ pos_embed:   Sinusoidal PE
        ├─ rl_token:    learnable [1, embed_dim]
        ├─ layers:      [RLTSelfAttentionLayer × num_layers]
        └─ output_proj: Linear(embed_dim → input_dim)
```

### 6.2 核心方法

**`encode(prefix_embs, mask)`**
- 输入：`[B, prefix_seq_len, input_dim]` prefix embeddings
- 将 learnable `rl_token` 拼接到 prefix 序列前
- 通过 cross-attention transformer layers 压缩整个 prefix → 输出 `rl_token` 位置的 hidden state
- 输出：`[B, 1, embed_dim]`

**`encode_flat(prefix_embs, mask)`**
- 调用 `encode()` 后 reshape 为 `[B, embed_dim]`（Stage 2 actor 使用）

**`decode(rl_tokens, target_embeddings, mask)`**
- 输入：`rl_tokens [B, 1, embed_dim]` + `target_embeddings [B, seq_len, input_dim]`
- 自回归重建：用 rl_token 作为条件，通过 causal attention 逐 token 重建 prefix

**`reconstruct(prefix_embs, mask)`**
- `rl_tokens = encode(prefix_embs)` → `reconstructed = decode(rl_tokens, prefix_embs)`

**`loss(prefix_embs, mask)`**
- 调用 `reconstruct()` 后计算 `MSE(reconstructed, target)` 
- 如果提供 mask，仅计算有效 token 位置的 MSE
- 返回 `(mse_loss, {"mse": ..., "z_rl": rl_tokens})`

**`forward(prefix_embs, mask)`** = `loss(prefix_embs, mask)`

### 6.3 `RLTSelfAttentionLayer`

标准 Transformer block with:
- Pre-LayerNorm
- MultiheadAttention（支持 `key_padding_mask`）
- GeGLU MLP（Gated GELU Linear Unit）

---

## 7. 数据管线

### 7.1 Dataloader 构建

```
FSDPVlaSftWorker.build_dataloader(data_paths)
  → model_type == SupportedModel.OPENPI_RLINF
    → build_openpi_rlinf_sft_dataloader(cfg, world_size, rank, data_paths)
      → use_rlt=True
        → build_official_openpi_sft_dataloader()
```

### 7.2 `build_official_openpi_sft_dataloader()` (`official_sft_data_loader.py`)

1. **解析数据集路径** — `resolve_lerobot_repo_id(data_paths)` → 本地路径或 HF repo ID
2. **获取 TrainConfig** — `get_openpi_config(config_name="pi05_rlt_maniskill_joint", ...)` 
3. **验证模型形状** — `_validate_openpi_rlinf_model_shape()` 确保 YAML 与 openpi config 的 `action_horizon` 一致
4. **创建 data_loader** — `openpi_data_loader.create_data_loader(config, framework="pytorch")` — 使用 openpi 官方 `_data_loader` 模块
5. 返回 `(data_loader, data_config)`

### 7.3 `LeRobotRLTManiSkillJointDataConfig` (`maniskill_rlt_dataconfig.py`)

定义 RLT ManiSkill 数据的 transform pipeline：

- **repack**: 将 LeRobot 键名映射到标准 Observation 键名（`image` → `observation/image` 等）
- **data_transforms**: `ManiSkillInputs` + `ManiSkillOutputs`（处理 image resize、state normalize、action pad 等）
- **model_transforms**: `ModelTransformFactory()(model_config)` — 标准的 Pi0 tokenization/padding

### 7.4 配置 key: `openpi_data`

```yaml
openpi_data:
  repo_id: "maniskill_peginsertionside_joint"   # LeRobot 数据集路径
  default_prompt: "insert the peg in the hole"   # 默认语言指令
  norm_stats_path: null                          # 可选：指定 norm stats 路径
```

---

## 8. 训练循环细节

### 8.1 `FSDPSftWorker.run_training()`

每个 step 执行 `gradient_accumulation` 个 micro-batch：

```
for micro_step in range(gradient_accumulation):
    batch = next(data_iter)
    loss, metrics = get_train_model_output(batch)
    loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### 8.2 `FSDPVlaSftWorker.get_train_model_output(batch)`

```python
with amp_context:
    output = self.model(forward_type=ForwardType.SFT, data=batch)
# output → {"loss": rlt_loss + rlt_alpha * vla_loss, "vla_loss": ..., "rlt_loss": ...}
```

### 8.3 Checkpoint 保存

**`FSDPSftWorker.save_checkpoint()`** — 调用 FSDP strategy 的 `save_checkpoint(save_full_model_weights=True)`，保存：
- `dcp_checkpoint/` — 分布式 checkpoint（各 rank 分片）
- `model_state_dict/full_weights.pt` — 完整权重（rank 0 gather）

**`FSDPVlaSftWorker.save_checkpoint()`** — 额外保存 dataloader state 和 RNG state 用于 resume。

---

## 9. Stage 1 → Stage 2 的桥接

### 9.1 产出物

Stage 1 训练完成后，checkpoint 目录结构：
```
checkpoints/global_step_{N}/
  └─ actor/
       ├─ dcp_checkpoint/       # 分布式 checkpoint
       └─ model_state_dict/
            └─ full_weights.pt  # 完整权重，含 Pi0 + RLTModule
```

### 9.2 Stage 2 如何使用

Stage 2 配置中的 `rlt_feature_model.model_path` 指向 Stage 1 的 `full_weights.pt`：

```yaml
rollout:
  rlt_feature_model:
    model_type: "openpi"
    model_path: "/path/to/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_<N>/actor"
    openpi:
      use_rlt: True
      config_name: "pi05_rlt_maniskill_joint"
```

加载时 `openpi_rlinf/__init__.py:get_model()` 会：
1. 识别 `model_path` 指向 `full_weights.pt`
2. 先创建 Pi0 核心（从配置），加载 base safetensors
3. 创建 wrapper（eval task 模式用于 rollout feature extraction）
4. `load_full_wrapper_weights()` 加载完整的 wrapper checkpoint（含 RLT Module）
5. 验证 RLT module 权重是否正确加载

### 9.3 权重转换

如果 Stage 2 使用不同的 openpi 变体（如 `openpi` → `openpi_pytorch`），需要先用 `convert_pt_to_hf.py` 转换：
```bash
python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
    --config-name fsdp_openpi_convertor \
    convertor.train_config_path=/path/to/maniskill_rlt_stage1_sft_openpi_pi05.yaml \
    convertor.ckpt_path=/path/to/model.pt \
    convertor.save_path=/path/to/hf_model
```

---

## 10. 配置解读

### 10.1 核心配置项 (`maniskill_rlt_stage1_sft_openpi_pi05.yaml`)

```yaml
actor:
  training_backend: "fsdp"            # FSDP 分布式训练
  micro_batch_size: 8                 # 每 GPU 每步 batch
  global_batch_size: 256              # 总 batch = 256
  model:
    model_type: "openpi_rlinf"        # 使用 openpi_rlinf 模型变体
    model_path: "/path/to/pi05_base"  # Pi0.5 base checkpoint
    num_action_chunks: 10             # action horizon = 10
    action_dim: 8                     # 实际 action 维度
    num_steps: 5                      # 推理时 ODE solver 步数
    openpi:
      task: sft                       # SFT 模式
      config_name: "pi05_rlt_maniskill_joint"  # 数据配置名
      use_rlt: True                   # 启用 RLT Module
      rlt_alpha: 1.0                  # RLT loss 权重
      rlt_input_dim: 2048             # RLT encoder 输入维度 = prefix embedding dim
      rlt_embed_dim: 2048             # RLT z_rl 维度
      rlt_prefix_seq_len: 1024        # prefix 序列最大长度
      rlt_num_layers: 2               # RLT transformer 层数
      rlt_num_heads: 8                # 注意力头数
      rlt_image_only: False           # False = 保留 language tokens
      rlt_use_mask: True              # 使用 mask 计算 RLT loss
    openpi_data:
      repo_id: "maniskill_peginsertionside_joint"
      default_prompt: "insert the peg in the hole"
  fsdp_config:
    strategy: "fsdp"                  # FSDP v1
    sharding_strategy: "no_shard"     # DDP 模式（不切分参数）
    mixed_precision:
      param_dtype: bf16               # bf16 混合精度
```

### 10.2 `config_name: "pi05_rlt_maniskill_joint"` 对应配置

在 `rlinf/models/embodiment/openpi/dataconfig/__init__.py` 的 `_CONFIGS` 中定义：

```python
TrainConfig(
    name="pi05_rlt_maniskill_joint",
    model=Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True),
    data=LeRobotRLTManiSkillJointDataConfig(
        repo_id="physical-intelligence/maniskill",
        default_prompt="insert the peg in the hole",
        output_action_dim=8,
    ),
    weight_loader=CheckpointWeightLoader("checkpoints/jax/pi05_base"),
    pytorch_weight_path="checkpoints/torch/pi05_base",
)
```

---

## 11. 函数衔接图

```
train_vla_sft.py:main()
  │
  ├─ FSDPVlaSftWorker.launch()          ─── N 个 Ray Worker 进程
  │     └─ init_worker()
  │           └─ setup_model_and_optimizer()
  │                 └─ model_provider_func()
  │                       └─ get_model(cfg.actor.model)        ← rlinf/models/__init__.py
  │                             └─ openpi_rlinf/__init__.py:get_model()
  │                                   ├─ Pi0Config.create()    ← 构建 Pi0 核心
  │                                   ├─ load_base_safetensors() ← 加载预训练
  │                                   ├─ _build_sft_model()    ← model_builders.py
  │                                   │     └─ OpenPiPytorchSFTActionModel(pi0, rlt_cfg)
  │                                   │           └─ OpenPiPytorchActionModel.__init__()
  │                                   │                 └─ if use_rlt: RLTTokenTransformer()
  │                                   └─ load_full_wrapper_weights()  ← 可选 resume
  │
  └─ SFTRunner.run()
        └─ actor.run_training()
              └─ FSDPSftWorker.run_training()
                    └─ get_train_model_output(batch)
                          └─ model(forward_type=SFT, data=batch)
                                └─ OpenPiPytorchSFTActionModel.forward()
                                      └─ sft_forward(data)
                                            ├─ Unpack: (observation, actions)
                                            ├─ _sft_forward_with_rlt_prefix()
                                            │     ├─ embed_prefix() → prefix_tokens
                                            │     ├─ embed_suffix() → suffix_tokens
                                            │     ├─ llm() → (prefix_out, suffix_out)
                                            │     ├─ velocity_from_suffix() → v_t
                                            │     ├─ vla_loss = MSE(v_t, noise - actions)
                                            │     └─ _select_rlt_prefix_embeddings()
                                            └─ _rlt_forward(prefix_out, prefix_mask)
                                                  └─ RLTTokenTransformer.loss()
                                                        ├─ encoder.encode() → rl_tokens
                                                        ├─ decoder.decode() → reconstructed
                                                        └─ MSE(reconstructed, target)
```

---

## 12. Stage 1 → Stage 2 数据流

```
Stage 1 (SFT)                          Stage 2 (RLT AC)
─────────────                          ─────────────────
Pi0 + RLTModule training               RLT AC (MLP) RL 训练
     │                                       │
     │  checkpoint:                           │
     │  full_weights.pt                       │  rlt_feature_model:
     │  ├─ model.* (Pi0)                      │    model_type: "openpi"
     │  └─ rlt_module.* (RLT)                 │    openpi.use_rlt: True
     │                                       │
     └──────────────────────────────────────→│  rollout worker 加载为 feature model
                                             │
                                             │  predict_rlt_actions():
                                             │    obs → Pi0.embed_prefix()
                                             │        → RLTModule.encode_flat()
                                             │        → z_rl [B, 2048]
                                             │
                                             │  z_rl → Stage 2 MLP Actor → action
```
