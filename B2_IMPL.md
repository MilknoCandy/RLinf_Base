# B2 实现说明

本文描述 **当前代码里实际跑的 B2**，不是早期 dump / 读出头 SFT 草案。
方案动机见 `RLT_MEM.md`。入口配置是完整独立的
`examples/embodiment/config/maniskill_rlt_stage2_b2_loop.yaml`（不继承 Stage 2 AC yaml）。

启动：

```bash
python examples/embodiment/train_embodied_agent.py --config-name maniskill_rlt_stage2_b2_loop \
  rollout.rlt_feature_model.model_path=/path/to/stage1/.../actor
```

`encoder_ckpt` 与 `rollout.rlt_feature_model.model_path` 都指向 **Stage 1**。
`runner.ckpt_path` 与原 Stage 2 一样默认 `null`：它不是 Stage 2 训练要加载的内容，只在 eval/resume 时才把一份已训好的 **MLP** `state_dict` 灌进 `hf_model`。

---

## 1. 一句话

B2 不另建 memory bank。记忆就是 **RLT prefix encoder 的 Loop**：

\[
z_t = \mathrm{Encoder}([I_t,\; z_{t-1}.\mathrm{detach()}]),\qquad z_{-1}=z_{\mathrm{init}}
\]

- \(I_t\)：冻结 VLM 对当前图像（及 \(t>0\) 的固定反馈句）做完 prefix 之后，用最后一层 **文本→图像注意力 TopK ~50%** 留下的 image tokens。
- \(z_t\)：当前 chunk 的 RL token，给 MLP 去残差调整 VLA 的 `ref_chunk`。
- 下一步把 \(z_t\) 当作 RL token 再压一次。episode / 场景 `done` 时回到可学习的 \(z_{\mathrm{init}}\)。

VLA **从来不吃 \(z\)**。同一套 prefix KV cache 只用来采样 `ref_chunk`。RLT 仍然是「VLA 出 reference + MLP 用 \(z\) 微调」，不是丢掉 `ref_chunk` 另训一条策略。

---

## 2. 和原 Stage 2 的差别

| | 原 Stage 2 (`maniskill_rlt_stage2_ac_mlp`) | B2 (`maniskill_rlt_stage2_b2_loop`) |
| --- | --- | --- |
| VLM / VLA | 冻结 feature model（Stage 1 `model_path`） | 同样冻结 |
| Encoder | 冻在 feature model 里，每帧独立压当前 prefix → `z_rl` | 从 Stage 1 抽出 `rlt_module.encoder.*` 挂到 MLP 的 `rlt_loop`，在线 Loop，随 RL 更新 |
| MLP | **随机初始化**，`model_path=""`，`ckpt_path: null` | 同样随机初始化。`ckpt_path` 仍只用于之后 eval/resume，不是训练输入 |
| VLM prompt | 只有 instruction | instruction + 固定模板反馈句（\(t=0\) / `done` 为空） |
| Replay | `{z_rl, proprio, ref_chunk}` | 再加上 `{I_t, mask, z_prev}`。Actor 用 \(I_t,z_{\mathrm{prev}}\) **重算** \(z\) |
| \(z\) 进 VLA？ | 否 | 否 |

Dump + dist/success 读出头（`maniskill_rlt_b2_dump.yaml` / `examples/sft/train_rlt_b2.py`）还在仓库里，**不是主训练路径**。主路径是：丢掉 dump，把 Stage 1 encoder 接到 actor 上做在线 Loop RL。

---

## 3. 谁冻、谁训

Feature model 仍是整份 Stage 1 OpenPI。里面那份 encoder **不是丢掉**，而是 **拷到 MLP 的 `rlt_loop` 上继续训**。Feature model 本体冻结，只负责 VLM prefix 和 VLA `ref_chunk`；出 \(z\)、学历史压缩的是 actor 上那份拷贝。

```
冻结（整份 Stage 1 OpenPI，rollout.rlt_feature_model）
  VLM + action expert     每步跑 prefix，出 I_t 和 ref_chunk
  rlt_module.encoder      权重还在，但 encode_z=False，不再出 z
                          （避免和 rlt_loop 两套 encoder 分叉）

可训（RLTMLPPolicy，和原 Stage 2 一样从头建 MLP 头）
  rlt_loop     Stage 1 encoder 的拷贝：load_encoder_weights(encoder_ckpt)
               z_t = Encoder(I_t, z_{t-1}.detach())，随 actor -Q+BC 更新
  backbone / actor_mean    MLP 策略头（随机初始化）
  q_head                   critic（随机初始化）
```

历史压缩能力从哪来：

1. **初始化**：`encoder_ckpt` 指向同一份 Stage 1，`rlt_loop` 继承单帧压缩（独立帧 + \(z_{\mathrm{init}}\)）。
2. **在线 Loop**：每步把上一步 \(z\) 当 RL token 再压，\(I_t\) 里已经含反馈句调制。
3. **RL 更新**：`rlt_loop` 在 actor 优化器里，\(-Q\) 和 BC 回传到 encoder，学的是「怎么把 \(I_t\) 和 \(z_{t-1}\) 压成对关键段微调有用的 \(z_t\)」。

如果继续用 feature model 里冻住的 encoder 出 \(z\)，那就是原 Stage 2：encoder 不再更新，历史 Loop 也学不到。B2 正是为了让这份 encoder 参与 RL，才把它从 feature model 挪到 `rlt_loop`。

`rlt_loop` 这个名字是故意的。FSDP SAC 用 `"encoder" in param_name` 把参数丢给 **critic 优化器**。如果模块叫 `rlt_encoder`，Loop 权重会进 Q 优化器，actor 的 \(-Q+\mathrm{BC}\) 更新不到它。

权重同步：现有 `hf_model` weight_sync 整模同步（含 `rlt_loop`）。Rollout 侧 MLP 是 actor 的 inference 副本。Feature model 始终冻结，**不同步**；B2 开着时它也不负责出 \(z\)。

加载路径（和原 Stage 2 对齐，多一项 encoder）：

```
HuggingFaceWorker.init_worker / FSDPActorWorker.model_provider_func
  hf_model = get_model(rollout.model / actor.model)
      RLTMLPPolicy 随机初始化        # rollout.model.model_path 在 Stage 2 yaml 里是 ""
      若 rlt_loop：build_encoder + load_encoder_weights(encoder_ckpt)
          encoder_ckpt → Stage 1 的 rlt_module.encoder.*（不是整份 Stage 2 actor）
      仅当 runner.ckpt_path 非空：hf_model.load_state_dict(ckpt_path)
          这是 eval/resume，不是 Stage 2 训练配方
  rlt_feature_model = get_model(rollout.rlt_feature_model)
      model_path → Stage 1 OpenPI（VLM + encoder + VLA），eval + requires_grad_(False)
```

原 Stage 2 训练：`ckpt_path: null`，MLP 不从任何 Stage 2 权重起步。
B2 训练：同样 `ckpt_path: null`。多出来的只是 `encoder_ckpt` 把 Stage 1 **encoder 子集**拷进 `rlt_loop`。
`maniskill_rlt_b2_dump.yaml` 才会设 `runner.ckpt_path` 指向一份已经训完的 Stage 2 actor，那是 **eval dump**，不是 Stage 2 训练。

---

## 4. 文件地图

| 文件 | 做什么 |
| --- | --- |
| `examples/embodiment/config/maniskill_rlt_stage2_b2_loop.yaml` | 完整训练配置 |
| `rlinf/algorithms/rlt/rollout.py` | 每步：反馈、VLM extract、Loop encode、写 replay |
| `rlinf/algorithms/rlt/b2_loop.py` | 每 env 的 \(z_{t-1}\) 与反馈句状态机 |
| `rlinf/algorithms/rlt/b2_feedback.py` | Peg Insertion 模板句；从 `env_infos` 抽特权量 |
| `rlinf/algorithms/rlt/transition.py` | replay 键：`RLT_OBS_KEYS` + `RLT_B2_OBS_KEYS` |
| `rlinf/models/embodiment/openpi_rlinf/eval_action_model.py` | `extract_rlt_obs`：拼 prompt、prefix、TopK、`ref_chunk` |
| `rlinf/models/embodiment/openpi_rlinf/pi0_model/{pi0,gemma}.py` | 最后一层 attn 钩子（仅 B2） |
| `rlinf/models/embodiment/modules/rlt_b2_select.py` | 文本→图像注意力 TopK |
| `rlinf/models/embodiment/modules/rlt_token_transformer.py` | `RLTTokenEncoder([I_t, rl_token]) → z` |
| `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` | `rlt_loop` / `encode_rlt` / `_get_z` |
| `rlinf/models/embodiment/mlp_policy/__init__.py` | `rlt_loop=True` 时 `build_encoder` + `load_encoder_weights` |
| `rlinf/algorithms/rlt/b2_sft.py` | `build_encoder` / `load_encoder_weights`（从 Stage 1 抽 `rlt_module.encoder.*`） |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 按 stage 持有 `B2LoopState`，把 `dones`/`env_infos` 传进 predict |
| `rlinf/workers/env/env_worker.py` | B2 时把 `dones` 和精简后的 `env_infos` 发给 rollout |
| `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` | Loop 时 actor Q **不** detach encoder；critic TD **detach** |
| `tests/unit_tests/test_rlt_b2.py` | 反馈句、TopK、Loop、`z_prev` detach、replay 可选键 |

---

## 5. 配置开关（B2 yaml 相对原 Stage 2 多出来的）

Actor / rollout MLP：

```yaml
rlt_loop: True
encoder_ckpt: ${rollout.rlt_feature_model.model_path}
rlt_input_dim: 2048
rlt_embed_dim: 2048
rlt_prefix_seq_len: 1024
rlt_num_layers: 2
rlt_num_heads: 8
rlt_mlp_ratio: 4.0
rlt_dropout_rate: 0.0
```

冻结 VLM（`rollout.rlt_feature_model.openpi`）：

```yaml
rlt_b2: True
rlt_b2_image_keep_ratio: 0.5
rlt_image_only: True
rlt_use_mask: True
```

`rlt_b2: True` 会：

1. rollout 创建 `B2LoopState`，往 VLM 拼反馈句；
2. Gemma 最后一层记下 attn，供 TopK；
3. env worker 额外发送 `dones` + Peg 特权 `env_infos`。

`get_model`：`rlt_loop` 或 `encoder_ckpt` 任一为真，就往 `RLTMLPPolicy` 上挂 `rlt_loop`。

---

## 6. 逐步：一次 rollout 决策

调用链：

```
EnvWorker._build_rollout_input_data
  → obs, dones, env_infos={success*, peg_head_hole_*}
HuggingFaceWorker._predict_rollout_actions
  → predict_rlt_actions(..., b2_loop=B2LoopState)
```

`predict_rlt_actions` 在 `rlt_b2` 且 **不是 dump** 时走下面这条路（全程 `torch.no_grad()`）。

### 6.1 取 \(z_{t-1}\)

`B2LoopState.build_rl_token`：

- 第一次，或 batch 对不上：返回 encoder 的可学习 `rl_token_embed`，即 \(z_{\mathrm{init}}\)。
- 该 env `done`：同样回到 \(z_{\mathrm{init}}\)（新 episode / 新场景）。
- 否则：上一步 `commit_z` 存下的 \(z_t\)。

同场景换任务 **不** reset（没有额外 task-switch 标志）。只有 `done` 重置。

### 6.2 写反馈句

`B2LoopState.feedback_sentences`：

- 没有 `env_infos` / 缺少距离字段 / 还没有 history / 当前 env `done` → 空字符串。
- 否则用当前观测里的特权量（这是 **上一步动作执行之后** 的结果）写一句固定英文。

距离：

\[
d = \sqrt{y^2+z^2} - x
\qquad
\text{（}x=\texttt{peg\_head\_hole\_x},\ yz=\texttt{abs\_y/abs\_z}\text{）}
\]

\(\Delta d = d_{\mathrm{now}} - d_{\mathrm{prev}}\)。更小表示更靠近插入。

模板（`format_peg_insertion_feedback`）必须同时含成败和远近：

- 成败：`Insertion succeeded.` / `Insertion has not succeeded.`
- 远近：`moved closer` / `moved farther` / `distance unchanged`
- 可选：`yz > 0.01` 时加 `but is still slightly off-axis`

例：`Insertion has not succeeded. The peg moved closer to the hole, but is still slightly off-axis.`

拼进 VLM 的方式是 **instruction 后面空格接一句**，两句角色分开，不融合成新指令（`append_feedback_to_prompts`）。VLM 冻结，所以这是 **固定模板 prompt**，不是再训语言。

Env 只传这些字段，避免整包 `infos` 过大：

`success_current`, `success`, `peg_head_hole_x`, `peg_head_hole_abs_y`, `peg_head_hole_abs_z`。

### 6.3 冻结 VLM 一次 prefix

`OpenPi0RlinfEvalActionModel.extract_rlt_obs`：

1. 若有反馈句，改 `prompt`。
2. `build_prefix_cache(..., capture_last_attn=True)`。Gemma **只钩最后一层**，把 attn 放到 `last_attn_probs`。
3. **同一套 KV cache** 上跑 Pi0 Euler，得到 `ref_chunk`。**不**把 \(z\) 当 extra prefix token。
4. `encode_z=False`（因为 MLP 上已经有 `rlt_loop`）：feature model 里的 Stage 1 encoder **不算 \(z\)**。
5. `return_image_tokens=True`：用文本 query 对 image key 的注意力，保留 `keep_ratio=0.5` 的 image tokens 作为 \(I_t\)（padded 位永不入选；每个 batch 项 TopK 长度相同）。

语言 token 不进 encoder（`rlt_image_only: True` + TopK 本身就丢掉 language 位）。Instruction 和反馈只通过 VLM 最后一层去调制、筛选 image。

### 6.4 用 Loop 出 \(z_t\)

```
z_t = policy.encode_rlt(I_t, mask, z_prev)   # 内部 z_prev.detach()
rlt_obs["z_rl"] = z_t
rlt_obs["z_prev"] = z_prev
rlt_obs["rlt_image_tokens"] = I_t.float16     # 省 replay 体积
b2_loop.commit_z(z_t)                         # 下一步的 z_{t-1}
```

`RLTTokenEncoder`：把 \(I_t\) 和 RL token 拼成序列，self-attention，读出 RL token 位置 → \(z_t\in\mathbb{R}^{2048}\)。

### 6.5 MLP + route

`RLTMLPPolicy.predict_action_batch` 与原 Stage 2 相同：

```
actor_state = concat(ref_chunk, z, proprio)
a = tanh(μ(MLP(actor_state)))     # 固定 std
```

然后 `SimulatorRLTRoute`：非关键段执行 VLA `ref_chunk`，关键段执行 MLP。关键段判定、expert takeover 与原 Stage 2 相同。

`predict_action_batch(..., return_obs=True)` 把整个 `rlt_obs` 拷进 `forward_inputs`，所以 replay 能拿到 \(I_t\) 和 `z_prev`。

### 6.6 写 transition（next_obs）

原 Stage 2 在 `done` 时会对 `final_obs` **再跑一次** VLM，作为 bootstrap 的 next 特征。

B2 **跳过**这次重提取（`final_obs=None`）。原因：

- 再跑 VLM 需要对齐 Loop 的 \(z\) 和反馈，而 reset 后的观测已经不是「当前 Loop 的下一步」；
- SAC 在 `done=True` 时会 mask 掉 next Q，终端 next_obs 用不上。

非终端步：现有 `update_rlt_transitions` 用 **下一步 predict 的当前特征** 当作上一步的 `next_obs`。于是

```
curr  = {I_t, z_{t-1}, proprio_t, ref_t, z_rl_t}
next  = {I_{t+1}, z_t, proprio_{t+1}, ref_{t+1}, z_rl_{t+1}}
```

对齐 Loop：next 的 `z_prev` 就是 curr 的 \(z_t\)。

Replay 键：

- 必有：`z_rl`, `proprio`, `ref_chunk`
- B2 额外：`rlt_image_tokens`, `rlt_image_mask`, `z_prev`（缺了也不崩，那时 actor 退回用存好的 `z_rl`）

---

## 7. Actor / Critic 怎么用这些键

`RLTMLPPolicy._get_z`：

```
若挂了 rlt_loop 且 obs 里有 rlt_image_tokens:
    z = Encoder(I_t, z_prev.detach())     # 每次前向重算
否则:
    z = obs["z_rl"]                       # 原 Stage 2
```

因此：

- **Rollout**：已经算过 \(z_t\) 并写入 `z_rl`；`predict_action_batch` 若再次 `_get_z` 会再 encode 一次（`no_grad`，只是重复计算）。
- **训练**：replay 里的 \(I_t\)（反馈已经在 VLM 里调制过）+ `z_prev`。Actor **重跑 encoder**，梯度能回到 `rlt_loop`。`z_prev.detach()`，没有跨步 BPTT。

损失仍是原 RLT AC：

\[
\mathcal L_{\mathrm{actor}} = -q_{\mathrm{weight}}\,Q(s,\pi) + bc_{\mathrm{weight}}\,\mathrm{BC}(\pi,\;\mathrm{ref\_chunk})
\]

Loop 打开时与原 SAC 的 detach 约定相反：

| | 原 Stage 2（encoder 冻在 feature model） | B2（encoder 在 actor 优化器） |
| --- | --- | --- |
| Critic TD → encoder | 无所谓（冻） | **detach**：不把没用的 encoder 梯度算出来（反正不在 critic 优化器里） |
| Actor \(Q(\pi)\) → encoder | detach（标准 SAC） | **不 detach**：\(-Q\) 必须流进 `rlt_loop` |

否则 encoder 只会吃到 BC（把 MLP 拉向 `ref_chunk`），没有任务 Q 信号。

Critic 输入仍是 `[z, proprio]`；actor 输入仍是 `[ref_chunk, z, proprio]`。`reference_dropout` 只作用于 `ref_chunk`。

---

## 8. 反馈语义存在哪

Replay **不存反馈原文**。

反馈只在 rollout 进冻结 VLM。存下来的 \(I_t\) 已经是「看过 instruction + 反馈句」的 TopK image tokens。Actor 不再跑 VLM，也不再拼句子。

所以：

- 训练时反馈语义在 \(I_t\) 里，不在 prompt 字符串里；
- 换模板不会改变已写入 buffer 的旧 \(I_t\)；新 rollout 才会带新模板。

---

## 9. 和「不要把 z 注入 Stage 1 VLA」的关系

Stage 1 训练分布：VLM 只有 instruction，encoder 吃当前帧 image（加 \(z_{\mathrm{init}}\) 那种静态 RL token），VLA 从 prefix KV 出 `ref_chunk`。

B2 **没有**改 `build_prefix_cache` 去加 `extra_prefix_tokens`。`rl_token` 只在有 `rlt_loop` 时作为 encoder 的额外 token，且那次 encode 发生在 MLP 上，不在 VLM 里。

因此：

- `ref_chunk` 仍是冻结 VLA 的输出（prompt 多了一句反馈，这是 B2 故意的 OOD；VLM 不训，只当固定 prompt）；
- MLP 和原 Stage 2 一样随机初始化。它要学的是用 **Loop 之后的 \(z\)**（当前帧或融过历史）去残差调 `ref_chunk`，所以不要把一份已经训完的 Stage 2 actor `state_dict` 当训练起点（那份权重没有 `rlt_loop`，obs 也不是 \(I_t,z_{\mathrm{prev}}\)）。

Dump 路径若打开反馈或 Loop，会让 **已训练的 Stage 1/2 同时 OOD**，所以 dump 配置保持 `rlt_b2: False`、不用反馈。主实验不需要 dump。

---

## 10. 逐步状态机（单个 env）

```
t = 0 (reset / bootstrap)
  z_prev = z_init
  feedback = ""
  VLM(image_0, instruction)
  I_0 = TopK(attn)
  z_0 = Encoder(I_0, z_init)
  a_0 = route(MLP(z_0, proprio, ref_0) or ref_0)
  commit z_0, 记下 d_0

t ≥ 1
  环境已 step，obs_t 带着结果特权量
  z_prev = z_{t-1}（若 done 则 z_init）
  feedback = 模板(success, d_t - d_{t-1}, off-axis)（done 则为空）
  VLM(image_t, instruction + feedback)
  I_t = TopK(attn)          # 反馈已经在这些 token 里
  z_t = Encoder(I_t, z_prev.detach())
  a_t = route(...)
  commit z_t
```

`has_history` 在第一次 `commit_z` 后为真。第一步没有「上一动作的结果变化」，所以没有反馈句。

---

## 11. 优化器与 FSDP

`EmbodiedSACFSDPPolicy.build_optimizers`：

- critic 组：名字含 `encoders` / `encoder` / `q_head` / `state_proj`
- actor 组：其余可训参数（MLP backbone、`rlt_loop.*`）

`rlt_loop.layers.*.self_attn...` 不含子串 `"encoder"`，所以进 actor Adam。
`q_head` 进 critic。

`clip_grad_norm_` 打在整模上；真正 `step` 的仍是各自分组。

Target 网络是整份 policy 的 EMA（含 `rlt_loop`）。算 target Q 时 `_get_z` 用 **target encoder** 重算 next 的 \(z\)。

---

## 12. 明确没有做的事

- 不把 \(z\) 注入 Pi0 prefix / suffix。
- 不把 dump SFT（dist/success 头、截断 BPTT）当主路径。
- 不在 actor 上重跑 VLM。
- 不把 encoder 冻在 feature model 里做「只训 MLP」的 B2 正实验（那是消融，需要另改 yaml：去掉 `rlt_loop` / `encoder_ckpt`）。
- CALVIN 反馈字段还没接（`RLT_MEM` 的 B2-4）。
- 同场景换任务只换 yaml 里的 `default_prompt`；encoder 权重复用，不 reset \(z\)，除非 `done`。

---

## 13. 怎么读代码（最短路径）

1. yaml：`rlt_b2`、`rlt_loop`、`encoder_ckpt`（Stage 1）；`ckpt_path` 与原 Stage 2 一样保持 `null`
2. `env_worker.py`：B2 时发送 `dones` / `env_infos`
3. `huggingface_worker.py`：`B2LoopState` → `predict_rlt_actions`
4. `rollout.py`：第 75–180 行整条决策
5. `eval_action_model.extract_rlt_obs`：VLM + TopK + `ref_chunk`
6. `rlt_mlp_policy.encode_rlt` / `_get_z`：训练时重算 \(z\)
7. `fsdp_rlt_ac_policy_worker.forward_actor/critic`：Loop 的 detach 约定
