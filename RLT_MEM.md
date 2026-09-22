# B2：RLT 面向长时程任务的历史信息压缩扩展

## 1. 研究目标

现有 RLT 通过 chunk-level representation compression 降低在线 RL 的训练和推理开销。其作用不是重做整条策略，而是在 **VLM 给出当前理解之后，于关键阶段做精细调整**。

当前这一调整主要依赖**当前 chunk**：VLM 对当前图像与 instruction 的编码，以及 reference action chunk。历史交互没有进入同一套压缩，因此关键阶段的 residual 是短视的；换任务时也无法复用已经学到的场景交互读法。

本工作的目标不是增强 RLT 的模型容量，也不是引入任务分解或独立 memory bank，而是：

> **在保持 RLT 轻量化的前提下，让基于 prefix 的 encoder 能够利用历史交互与执行反馈，使关键阶段的精细调整更准，并使这套能力可迁移到同场景的不同任务。**

核心问题：

$$
\boxed{
\text{如何让 RLT 的 prefix encoder 在压缩当前视觉的同时，持续吸收场景交互历史？}
}
$$

---

## 2. 核心思想

记忆模块就是 **RLT 的 prefix encoder**，不是一份存好的历史向量。被迁移、被复用的是 encoder 如何把当前视觉与上一时刻表征压在一起的能力；换任务时只换 instruction，不换这个 encoder。

原始 RLT 已经用提示学习把当前 prefix 压进一个 RL token。本扩展不另建序列模型，只把这件事沿 chunk 循环下去：

* 当前决策仍以**筛选后的当前 image** 为主；
* 上一 chunk 的表征 \(z_t\) 作为下一步的 RL token 回注；
* 执行反馈写成短自然语言，进入下一步 VLM prefix，通过最后一层文本注意力真正影响后续看什么、怎么调。

因此：

$$
\boxed{
\text{记忆模块}=\text{基于 prefix 的 RLT encoder}
}
$$

\(z\) 不是独立存储的 memory content，只是下一轮 prefix 里那个 RL token。

---

## 3. 模块分工

| 模块 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| VLM | 当前 instruction 与执行反馈的语言理解；用最后一层文本注意力筛选、调制 image token | 不直接输出精细动作 |
| Prefix Encoder | 把筛选后的当前 image 与上一步 \(z\) 压成当前表征 | 不是 memory bank，不单独解释长指令 |
| RLT 策略头 | 用 \(z_t\) 在关键阶段对 reference chunk 做精细调整 | 不替代 VLM 的场景理解 |

任务 instruction 是语言问题，留在 VLM。经过 VLM 之后，进入 encoder 的主体只有 **被文本注意力筛选过的 image token**，加上作为 RL token 的 \(z\)。

同场景换任务：只换 instruction；encoder 与 \(z\) 的回注方式不变。新 episode / 新场景才把 RL token 重置为初始可学习向量。

---

## 4. 当前视觉如何进入 Encoder

每个 chunk 先走 VLM，再筛 image，最后才进 encoder。

$$
\text{image}_t
\rightarrow
\text{VLM}
\rightarrow
\text{最后一层 hidden states}
$$

VLM prefix 的排布仍是 image 在前、language 在后。pi05 / ManiSkill 上典型为：每路相机 256 个 SigLIP patch，三路槽位共 768 个 image token，随后是 instruction（及下一步加入的反馈句）。

进 encoder 之前，用 **VLM 最后一层中文本对 image 的注意力** 筛选 image token。第一版采用固定比例，例如保留 50%。被保留的 image token 已经含有当前语言（instruction + 反馈句）的调制，不再把长 instruction token 送入 encoder。

$$
\boxed{
I_t=\operatorname{TopK}_{50\%}\!\left(\operatorname{Attn}_{\text{text}\rightarrow\text{image}}(\text{VLM last layer})\right)
}
$$

---

## 5. Loop：用 \(z\) 作为下一时刻的 RL token

记初始 RL token 为可学习向量 \(z_{\text{init}}\)。第 \(t\) 步：

$$
z_t
=
\operatorname{Encoder}\big([I_t,\; z_{t-1}]\big),
\qquad
z_0=z_{\text{init}}
$$

\(z_t\) 是**当前 chunk 的表征**，用于关键阶段精细调整，得到 \(a_t\)。传到 \(t+1\) 时，\(z_t\) 替换 RL token：

$$
\boxed{
[I_{t+1},\; z_t]
\xrightarrow{\text{Encoder}}
z_{t+1}
}
$$

这就是 Loop：每个 chunk 再走同一套 encoder；压缩成功的标志是当前筛选后的 image 与上一步 \(z\) 被压成新的 \(z\)。不维护 \(C_{1:t}\) 的窗口，也不另存 \(h_t\)。

---

## 6. 执行反馈如何进入下一步

\(z_t\) 在动作执行前算出，当前决策不以尚未发生的反馈为主。执行之后，环境状态变成结果，必须回到 **VLM**，否则反馈到不了最后一层注意力，仅靠 encoder 带不动后续动作。

反馈不用离散短标签（如 `success: fail`），以免浪费 VLM 的多模态语义。也不让 VLM 自由看图写长描述。而是 **由环境状态写成一句短自然语言**，直接反映结果变化，且必须包含：

* 成功与否 \(s_t\)；
* 距离变化 \(\Delta d_t=d_{t+1}-d_t\)（靠近 / 远离）。

Peg Insertion 示例：

> Insertion has not succeeded. The peg moved closer to the hole, but is still slightly off-axis.

失败且远离：

> Insertion failed. The peg moved farther from the hole.

\(t+1\) 的 VLM 文本 prefix 为：

$$
\boxed{
\text{prompt}_{t+1}
=
[\text{当前 instruction}]
\;+\;
[\text{上一 chunk 的结果变化句}]
}
$$

两条语言角色不同、不相揉：instruction 说当前要做什么；反馈句说上一步结果变了什么。换任务只换前一句；后一句仍由环境状态按同一规则生成。

于是 \(t+1\) 的完整路径是：

$$
\text{VLM}(\text{image}_{t+1},\; \text{instruction}_{t+1},\; \text{feedback}_t)
\rightarrow
I_{t+1}
\rightarrow
\operatorname{Encoder}([I_{t+1},\; z_t])
\rightarrow
z_{t+1}
\rightarrow
a_{t+1}
$$

离散的 \(s_t\)、\(\Delta d_t\) 同时保留为 SFT 读出监督，**不**作为 VLM prefix 的正文。

---

## 7. Stage 1：SFT 学习如何注入与压缩

Stage 1 不直接学最终控制策略，而是让 prefix encoder 学会：当前筛选后的 image、回注的 \(z\)、以及已经进入 VLM 的反馈句，如何压成对进展有意义的表征。

数据来自已有 trajectory。在每一步用当时的 instruction 与规则生成的反馈句构造 VLM prefix，筛选 image，按 Loop 得到 \(z_t\)。读出头要求 \(z_t\) 能够预测距离与成败：

$$
\mathcal L_{dist}
=
\left\|
\hat d_{t+k}-d_{t+k}
\right\|_2^2
$$

$$
\mathcal L_{success}
=
\operatorname{BCE}(\hat p_{success},\, s)
$$

$$
\boxed{
\mathcal L_{SFT}
=
\lambda_d\mathcal L_{dist}
+
\lambda_s\mathcal L_{success}
}
$$

SFT 决定 **怎么注入、压缩表示什么**。VLM 保持其语言–视觉通路，encoder 学习在这条已经含反馈的视觉 prefix 上做跨 chunk 压缩。

---

## 8. Stage 2：Online RL 学习如何使用

SFT 之后，encoder 已具备基本的历史压缩。Online RL 不再重做注入机制，而是让 encoder 根据任务 reward 学习 **如何使用** 这些已经注入的信息去做关键阶段调整：

$$
z_t
\rightarrow
\text{RLT policy}
\rightarrow
a_t
\rightarrow
r_t
$$

| 阶段 | 作用 |
| --- | --- |
| Stage 1 SFT | 学会注入与压缩：反馈句进 VLM，\(z\) 回注为 RL token，\(z\) 能反映距离与成败 |
| Stage 2 Online RL | 让 encoder 决定这些信息如何服务于精细调整 |

---

## 9. 与原始 RLT 的关系

### 原始 RLT

$$
\boxed{
[\text{VLM prefix: image + instruction}]
+
[\text{learnable RL token}]
\rightarrow
z_t
\rightarrow
a_t
}
$$

RL token 是静态可学习向量；encoder 只压缩当前 chunk。

### 扩展后的 RLT

$$
\boxed{
\text{VLM prefix: image + 当前 instruction + 上一 chunk 反馈句}
}
$$

$$
\boxed{
I_t=\text{文本注意力筛选后的 image tokens}
}
$$

$$
\boxed{
[I_t,\; z_{t-1}]
\xrightarrow{\text{同一 Encoder}}
z_t
\xrightarrow{\text{关键阶段精细调整}}
a_t
}
$$

不是在 RLT 外再加 Loop Transformer，而是让原有 encoder 沿 chunk 循环，并把执行反馈接到 VLM 的语言通道上。

---

## 10. ManiSkill 验证方案

先在 Peg Insertion 上验证。目的不是证明长时程，而是验证：

$$
\boxed{
\text{筛选后的当前 image}
+
\text{回注的 } z
+
\text{VLM 中的结果变化句}
}
$$

能否在不破坏原始 RLT 基础能力的前提下，让关键阶段调整用上历史交互。

环境状态已提供距离、姿态、success 等，足以生成短反馈句，并提供 SFT 的 \(\Delta d\)、\(s\) 监督。若简单任务无法稳定完成，再讨论 CALVIN 没有意义。

---

## 11. 研究路径

$$
\boxed{
\text{image}_t+\text{instruction}_t+\text{feedback}_{t-1}
\xrightarrow{\text{VLM}}
\text{last-layer tokens}
}
$$

$$
\boxed{
\text{文本注意力筛选 image（如 50\%）}
\rightarrow
I_t
}
$$

$$
\boxed{
\operatorname{Encoder}([I_t,\, z_{t-1}])
\rightarrow
z_t
\rightarrow
a_t
}
$$

$$
\boxed{
\text{执行}
\rightarrow
\text{环境状态}\rightarrow\text{短自然语言反馈}_t
}
$$

$$
\boxed{
z_t \text{ 作为 } t+1 \text{ 的 RL token};\quad
\text{feedback}_t \text{ 进入 } t+1 \text{ 的 VLM prefix}
}
$$

验证顺序：

$$
\boxed{\text{ManiSkill}\rightarrow\text{CALVIN}}
$$

ManiSkill 验证关键阶段调整是否因历史更稳；CALVIN 验证同场景换 instruction 时，encoder 与 \(z\) 回注是否仍可用。

### 一句话概括

> **B2：记忆模块就是 RLT 的 prefix encoder。当前图像经 VLM 后，由文本注意力筛选再压缩为 \(z\)；\(z\) 作为下一时刻的 RL token 回注，执行结果由环境状态写成短自然语言进入 VLM，从而在关键阶段利用场景交互历史做精细调整，并在同场景不同任务间复用这一压缩能力。**

---

## 12. 训练时何时算损失

每个 chunk 只跑 **1 次** encoder，没有同一帧上的空转迭代。一次 Loop 对应一次环境过渡（在线 step，或离线回放里的下一条记录）。

第一次完整压缩发生在 \(t=1\)：此时 VLM 已有 \(\mathrm{feedback}_0\)，encoder 输入才同时具备当前筛选后的 \(I_1\) 与回注的 \(z_0\)。

| | 每个 chunk 的 encoder 次数 | 何时算损失 | 梯度穿过几步 Loop |
| --- | --- | --- | --- |
| Stage 1 SFT | 1 | \(t\ge 1\) 逐步算 \(\mathcal L_{dist}+\mathcal L_{success}\) | 截断 \(K=2\sim 4\) |
| Stage 2 RL | 1 | 每步都算出动作与策略损失（含 \(t=0\)） | 第一版 \(z_{t-1}\) detach |

SFT 不在线进仿真：反馈句与 \(I_{t+1}\) 来自已执行过的轨迹记录。RL 必须 `env.step`，与现有 Stage 2 同频。

---

## 13. 训练效率

VLM 与仿真是成本主体；encoder Loop 很便宜，但一条轨迹内不能按时间并行。

- **SFT：** VLM 冻结。对离线轨迹 dump 一次 \(I_t\)（含反馈句与 TopK），之后每个 epoch 只在缓存的 \(I_t\) 上训 encoder + 读出头。
- **RL：** 不增加仿真次数。相对现有 `extract_rlt_obs`，只多一句反馈、一次最后一层 text→image TopK、一次小 encoder；\(z_{t-1}\) detach。
- 不要对整条 episode 做 BPTT，不要为 mem 再开一轮在线 SFT 交互。

---

## 14. B2 实现分析

相对现有 RLT，B2 只改四件事：VLM prompt 拼反馈句、最后一层注意力筛 image、encoder 的 RL token 可被上一步 \(z\) 替换、SFT 目标从 prefix 重建改为逐步距离/成败读出。不另建 memory bank，不另建 Loop Transformer。

### 14.1 现状与缺口

| 现有位置 | 现在做什么 | B2 缺口 |
| --- | --- | --- |
| `RLTTokenEncoder.forward` | 始终使用静态 `rl_token_embed` | 需要可选地喂入 \(z_{t-1}\) |
| `_select_rlt_prefix_embeddings` | 按 `rlt_image_only` 切掉 language 位置 | 还要用文本注意力 TopK 50% |
| `gemma.Attention` | 不算、不返回 attn | 需要最后一层 text→image 分数 |
| `extract_rlt_obs` | 每步独立编码，prompt 只有 instruction | 要拼反馈句，并吃 Loop 状态 |
| `predict_rlt_actions` | `del dones, rewards, success` | 要用 `dones` 重置 \(z\)，用 info 写反馈 |
| Stage 1 SFT | 独立帧 + prefix 重建 MSE | 轨迹窗口 + Loop + dist/success 头 |
| Stage 2 MLP | 消费冻结的 `z_rl` | encoder 在 RL 中仍要更新「如何用」 |

LeRobot SFT 数据（`maniskill_peginsertionside_joint`）通常只有图像、关节 state、action、prompt，**没有** `peg_head_hole_*` / `success_current`。因此 B2 的 \(I_t\) dump **不应假设现成数据集带特权进度**，而应在 `ManiskillRLTEnv` 上用已训练的 Stage 2 RLT（MLP 消费 \(z\)）采一次。

### 14.2 建议新增与修改

**新增（B2 专用，默认不改原始 RLT 行为）**

| 模块 | 职责 |
| --- | --- |
| `rlinf/algorithms/rlt/b2_feedback.py` | 由 `success`、\(\Delta d\)、可选偏轴写成一句短自然语言 |
| `rlinf/models/embodiment/modules/rlt_b2_select.py` | 用最后一层 text→image 注意力对 valid image token 做固定比例 TopK |
| `rlinf/algorithms/rlt/b2_loop.py` | 每 env 维护 \(z\)、上一距离、待注入反馈句；`done` 时重置为 \(z_{\mathrm{init}}\) |
| `rlinf/models/embodiment/modules/rlt_b2_heads.py` | SFT 读出：\(\hat d\)、\(\hat p_{\mathrm{success}}\) |
| dump / SFT 小循环 | 冻结 VLM → 存 \(I_t,d_t,s_t\) → 只训 encoder + heads |

**改动（加开关，例如 `openpi.rlt_b2: true`）**

1. `RLTTokenEncoder.forward(..., rl_token=None)`  
   `rl_token is None` 时用 `rl_token_embed`（\(t=0\) / reset）；否则用传入的 \(z_{t-1}\)（形状 `[B, 1, D]`）。
2. `Pi0.build_prefix_cache` / last-layer attention  
   只在 B2 打开时，对最后一层用 language query × image key 算分数（不要全层 `output_attentions`）。image 段长度仍按 `S - lang_len`，mask 掉 dummy 相机。
3. prompt  
   `prompt = instruction + " " + feedback_{t-1}`（\(t=0\) 无反馈句）。走现有 tokenizer，反馈必须是独立短句，不写进 instruction 模板。
4. `extract_rlt_obs` / `predict_rlt_actions`  
   输入增加 Loop 状态；输出 `z_rl` 的同时写回下一轮 RL token。`dones` 为 True 的 env 清零反馈并改回 `z_init`。
5. env info  
   已有 `success_current`、`peg_head_hole_x/y/z`。在 env 或 rollout 侧缓存上一 chunk 的距离，step 后算 \(\Delta d\) 并生成反馈句，供**下一次** VLM 使用。

### 14.3 运行时数据流（Stage 2 rollout）

```
env_obs_t, feedback_{t-1}, z_{t-1}
  → 拼 prompt（instruction + 反馈句）
  → VLM last layer
  → I_t = TopK_50%(Attn_text→image)
  → z_t = Encoder([I_t, z_{t-1}])          # t=0 时 z_{-1}=z_init
  → MLP(z_t, proprio, ref_chunk) → a_t
  → env.step → success, d_{t+1}, image_{t+1}
  → feedback_t = NL(success, Δd)
  → 若 done：z ← z_init，反馈清空
```

`ref_chunk` 仍由同一套 prefix KV cache 采样，与现有 OpenPI RLT 一致。反馈进 VLM 后 KV 也会被反馈句调制，这是预期行为：后续精细调整和 reference 都看得到结果变化。

### 14.4 Stage 1 SFT 实现路径

Dump + dist/success 读出头是**可选**路径，不是 B2 主训练。主路径见 14.5：在线 Loop RL。

不要在现有「独立帧 + 重建 MSE」的 `OpenPiPytorchSFTActionModel` 上硬接 Loop。B2 SFT 分两段：

**Dump（一次，VLM 冻结）**

1. 加载已训练的 Stage 1（`rollout.rlt_feature_model`：VLM + encoder + VLA）和 Stage 2（`runner.ckpt_path`：MLP）。执行必须与这两个 checkpoint 的训练分布一致：VLM prompt **只有 instruction**，不用反馈句，encoder 不用 Loop \(z_{t-1}\)。反馈句和 Loop 是后续 B2 的事；为了 dump 硬塞进去会让 `ref_chunk` 和 MLP 都偏出分布。
2. 每步走现有 RLT：同一套 prefix 出静态 RL token \(z\) 与 VLA `ref_chunk`；MLP 用 \((z,\mathrm{proprio},\mathrm{ref\_chunk})\) 做残差微调；`rlt_route` 非关键段执行 `ref_chunk`、关键段执行 MLP。
3. 从这次 in-distribution prefix 另取 TopK image tokens 作为 \(I_t\)，连同环境特权 \(d_t,s_t\) 存盘。之后 SFT 不再跑 VLM / 仿真。Loop 只发生在离线 SFT 的 encoder 上。

**Train（多次 epoch，只训 encoder + 读出头）**

1. 按 episode 取窗口，\(z_0=z_{\mathrm{init}}\)，逐步 \(z_t=\mathrm{Encoder}([I_t,z_{t-1}])\)。
2. \(t\ge 1\) 上 \(\lambda_d\mathcal L_{dist}+\lambda_s\mathcal L_{success}\)。
3. BPTT 截断 \(K=2\sim 4\)。
4. 原始重建 decoder 第一版关掉，避免和「encoder 只吃筛选后 image」抢目标。

可从现有 Stage 1 checkpoint 初始化 `rl_token_embed` 与 encoder 权重，再在 B2 目标上继续训。

### 14.5 Stage 2 B2：在线 Loop RL（主路径）

VLM/VLA 冻结。MLP 与原 Stage 2 相同：`get_model` 随机初始化，`runner.ckpt_path` 训练时为 `null`（那只是 eval/resume，不是 Stage 2 要加载的内容）。B2 多出来的是：把 Stage 1 encoder 挂在 `RLTMLPPolicy.rlt_loop`（`encoder_ckpt`），经现有 MLP `weight_sync` 同步到 rollout。

Replay 存 VLM 之后的 \(I_t\)（TopK tokens，反馈已调制）、`z_prev`、`proprio`、`ref_chunk`。不存反馈原文。Actor：\(z_t=\mathrm{Encoder}(I_t, z_{\mathrm{prev}}.\mathrm{detach}())\)。

配置：`examples/embodiment/config/maniskill_rlt_stage2_b2_loop.yaml`。`encoder_ckpt` 指向 Stage 1。

若第一版要降低风险，可先冻结 encoder、只训 MLP，作为消融；正实验仍应让 encoder 参与 RL。

### 14.6 反馈句（Peg Insertion 第一版）

用环境标量，固定句式，必须含成败与远近：

- \(s_t\)：`Insertion succeeded.` / `Insertion has not succeeded.` / `Insertion failed.`
- \(\Delta d_t\)：由 `peg_head_hole_x` 与 yz 距离合成标量 \(d\)，比较 \(d_{t+1}-d_t\) → `moved closer` / `moved farther` / `distance unchanged`
- 可选半句偏轴：若 yz 仍大，则 `but is still slightly off-axis`

例：`Insertion has not succeeded. The peg moved closer to the hole, but is still slightly off-axis.`

CALVIN 换同一接口、另一套环境字段，不改 encoder。

### 14.7 实现顺序

1. **B2-0 开关与单测**：`rl_token` 可注入；反馈句生成；TopK 在假 attn 上形状正确。  
2. **B2-1 推理 Loop**：rollout 维护 \(z\) 与反馈；VLM prompt 拼接；`done` 重置。用现成 Stage 1 encoder，不改损失，先看 Peg Insertion 是否不崩。  
3. **B2-2 Dump + SFT**：可选，非主路径。  
4. **B2-3 Stage 2 RL**：`maniskill_rlt_stage2_b2_loop.yaml`，encoder+MLP，detach \(z_{t-1}\)。  
5. **B2-4 CALVIN**：换反馈字段与 instruction 切换，验证同场景换任务。

### 14.8 主要风险

- 最后一层 attn 要改 `gemma.py` 的 Attention；必须只开 B2，且只钩最后一层，避免打断现有 eager prefix cache。  
- Dump 必须来自带 `peg_head_*` 的 env，不能从缺特权信息的 LeRobot 帧硬编距离。  
- 反馈句拉长 prompt 会改变 KV cache 与 `max_token_len=200` 预算，句子必须短。  
- Stage 2 encoder 挂在 MLP 的 `rlt_loop` 上（不要叫 `encoder`，以免进 critic 优化器）。通过现有 `hf_model` weight_sync 同步到 rollout；冻结的 VLM feature encoder 不再出 \(z\)。

