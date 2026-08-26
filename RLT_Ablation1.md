# RLT 消融实验 1：Stage 2 特征来源与 RLT Token 的必要性

## 1. 目标

回答两个问题：

1. Stage 2 是否还需要 Stage 1 的 RLT token，还是可以直接用 VLM 输出特征替代。
2. 用哪种 VLM 特征作为 `z_rl` 最有用：视觉、语言/文本、动作，还是整段前缀。

## 2. 固定不变的部件

所有实验都冻结 Stage 1，只改 Stage 2 的 `rollout.rlt_feature_model`，因此以下部件保持一致：

- **Stage 1 checkpoint**：`RLT + Openpi`（`pi05_rlt_maniskill_joint`），`use_rlt: True`。
- **Stage 2 head**：`rlt_mlp_policy`，RLT-AC / SAC 训练。
- **输入结构**：`[ref_chunk, z_rl, proprio]`。
- **环境 / 奖励 / 超参**：来自 `examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml`。

## 3. 两个消融模式

### Mode A：`rlt_mode: token`

```
selected VLM feature (B, S, D)
  -> frozen RLT token transformer (Stage 1)
  -> z_rl (B, rlt_embed_dim)
```

### Mode B：`rlt_mode: none`

```
selected VLM feature (B, S, D)
  -> mean-pool
  -> z_rl (B, rlt_embed_dim)
```

Mode B 不经过 RLT token transformer，直接对 VLM 输出特征做 mean-pool。

## 4. 特征来源定义

| source | 数据来源 | 消融问题 |
| --- | --- | --- |
| `all` | PaliGemma prefix 全段：`image + language` | 控制组，等价于 Stage 1 RLT 的原始输入（`rlt_image_only: False`） |
| `image` | prefix 中所有视觉 token | 仅视觉是否足以压缩出 RL 状态 |
| `language` / `text` | prefix 中语言 prompt token | 文本/任务指令是否单独携带任务语义 |
| `action` | action-expert 对“冻结 VLM 采样的参考动作”的 hidden states | 参考动作的隐表征是否比原始 `ref_chunk` 更适合作为 RL 状态 |

实现位于 `rlinf/models/embodiment/openpi_rlt_probe/feature_sources.py`，通过
`@register_feature_source` 注册，`select_features()` 按区间切片，`mean_pool_features()`
完成 Mode B 的归约。

## 5. 消融矩阵

理论上 `2 modes × 5 sources = 10` 组。建议顺序：

1. 先跑控制组：`token + all` vs `none + all`，判断 RLT token 是否有增益。
2. 再固定有增益/有差异的模式，扫 `image / language / action`。

| run | rlt_mode | rlt_feature_source |
| --- | --- | --- |
| A0 | token | all |
| A1 | token | image |
| A2 | token | language |
| A3 | token | action |
| B0 | none | all |
| B1 | none | image |
| B2 | none | language |
| B3 | none | action |

## 6. 已生成的两个 YAML

- `examples/embodiment/config/maniskill_rlt_stage2_ablation1_rlt_token.yaml`
  - `model_type: openpi_rlt_probe`
  - `rlt_mode: token`
  - `rlt_feature_source: all`
- `examples/embodiment/config/maniskill_rlt_stage2_ablation1_no_rlt.yaml`
  - `model_type: openpi_rlt_probe`
  - `rlt_mode: none`
  - `rlt_feature_source: all`

两个文件都通过 `rollout.rlt_feature_model.model_path` 加载同一个 Stage 1 RLT+Openpi
checkpoint。切换特征来源只需改 `rlt_feature_model.openpi.rlt_feature_source`。

## 7. 运行方式

1. 把两个 YAML 中的 `model_path` 替换成你的 Stage 1 RLT+Openpi 检查点目录，例如：

   ```yaml
   model_path: "/path/to/maniskill_rlt_stage1_sft_openpi/checkpoints/global_step_<step>/actor"
   ```

2. 分别启动：

   ```bash
   python examples/embodiment/train_embodied_agent.py \
     --config-name maniskill_rlt_stage2_ablation1_rlt_token

   python examples/embodiment/train_embodied_agent.py \
     --config-name maniskill_rlt_stage2_ablation1_no_rlt
   ```

   （也可用 `bash examples/embodiment/run_embodiment.sh <config_name>`。）

### 运行时切换特征来源

`rlt_feature_source` 和 `rlt_mode` 都是 Hydra 配置项，可以直接用命令行覆盖，无需改 YAML：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-name maniskill_rlt_stage2_ablation1_rlt_token \
  rollout.rlt_feature_model.openpi.rlt_feature_source=image
```

串行脚本已提供：`examples/embodiment/run_rlt_ablation1_serial.sh`。它按顺序遍历来源，
并为每个 run 生成独立日志目录和 experiment name：

```bash
MODEL_PATH=/path/to/stage1/actor \
  bash examples/embodiment/run_rlt_ablation1_serial.sh token

MODEL_PATH=/path/to/stage1/actor \
  bash examples/embodiment/run_rlt_ablation1_serial.sh none "image action"

MODEL_PATH=/path/to/stage1/actor \
  bash examples/embodiment/run_rlt_ablation1_serial.sh all "all image language action"
```

`MODEL_PATH` 会以 `rollout.rlt_feature_model.model_path=...` 的形式传入，避免每次改 YAML。

## 8. 观察指标

- 评估成功率 / reward（`eval/success_rate`、`env/reward`）。
- Stage 2 训练曲线：`train/actor_loss`、`train/critic_loss`、Q 值。
- 相同环境交互量下的样本效率，而不是只看最终分数。

## 9. 预期与假设

- 若 `token + all` 与 `none + all` 无明显差距，说明 Stage 2 不依赖 RLT token 压缩，
  可直接使用 VLM 特征。
- 若 `image` 明显差于 `all` 或 `language`，说明纯视觉前缀信息不足。
- `action` 来源是“当前观测 → 冻结 VLM 参考动作 → 隐表征”，不含未来信息，因此作为
  RL 状态没有泄漏；但它与 `ref_chunk` 信息高度重叠，解释时应视为“参考动作的另一种编码”。

## 10. 注意事项

- 两个模式都加载同一个 RLT+Openpi checkpoint。`none` 模式仍会实例化 RLT 模块并加载其
  权重，只是在 `extract_rlt_obs` 中不调用 `rlt_module.encode_flat`。
- `action` 特征的原始维度是 action-expert 宽度（约 1024），会通过新增的 `z_proj`
  投影到 `rlt_embed_dim`（2048）；该投影在 rollout 中冻结且随机初始化，比较 `action`
  结果时需注意这一点。
- 不要把数据集里的真实动作标签当作 `z_rl`，否则会造成标签泄漏。
