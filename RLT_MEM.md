# RLT 记忆扩展：同场景关键阶段的可迁移精细调整

本文描述当前 Stage 2 记忆方案（相对原版 RLT），以及它和已撤回路径（B2 反馈句、特权 progress 进 \(Q\)、GRU、256 维 sidecar）的差别。实现入口是 `use_mem: True` 的 `examples/embodiment/config/maniskill_rlt_stage2_ac_mlp_mem.yaml`。原版 `maniskill_rlt_stage2_ac_mlp.yaml` 不改。

---

## 1. 要解决什么

原版 RLT 只在**关键阶段**对冻结 VLA 做精细调整，不重训 Pi0。Stage 1 把当前 prefix 压成 \(z_{\mathrm{rl}}\)；Stage 2 小 MLP 在 \(z\)、proprio 和 VLA 的 `ref_chunk` 上出动作，并用 BC 拉向 `ref`。

两个限制：

1. **短视。** 冻结 Stage 1 的 \(z_{\mathrm{rl}}\) 只压缩当前帧。Peg 上像素几乎不变，「上一 chunk 更近了 / 插偏了」进不去。
2. **`ref → a` 短路。** Actor 输入里就有整段 `ref_chunk`，再加 \(\|\pi-\tilde a\|^2\)，最容易的解是抄提案。后面再拼任何历史向量也很难改到 \(a\)。

优化目标不是「让 VLA 自己会记」，而是：

> 同一套可训的 **encoder loop**，在**同一个 ManiSkill 场景**里换 instruction 后不训还能用；新 episode 重置 \(z\)；VLA 继续出同分布 `ref_chunk`。

能 zero-shot 的是**写法**（当前 \(I_t\) 如何和上一 \(z\)、上一执行 \((a,r)\) 一起压成新的 \(z_t\)），不是某条轨迹里的 \(z\) 向量。

---

## 2. 和原版 RLT 的分工（没改的部分）

| 模块 | 仍然做什么 |
| --- | --- |
| 冻结 VLA（官方 OpenPI / Pi0.5） | 当前 \(I_t\) + instruction → `ref_chunk`；并给出 VLM prefix embeddings |
| \(z\) 不注入 VLA | 记忆只在 Stage 2 的 loop encoder + MLP 里 |
| Route | 进入关键阶段前执行 VLA。原版之后执行 actor；mem 的 train 采集在 \(\mathrm{EMA}(\|\pi-\tilde a\|_1)\) 降到阈值前仍执行 VLA，eval 过 warmup 后看学生 |
| Critic 结构 | Twin-Q，chunk 级 TD。原版 `only_success`；mem 训练用 `success_potential`（\(1[\mathrm{success}]+\mathrm{coef}(\gamma\Phi'-\Phi)\)），\(\Phi\) 只进 \(r\) 不进 \(Q\) |
| Actor 形式 | 固定标准差的 \(\tanh\) MLP，损失里仍有 \(-Q+\beta\,\mathrm{BC}\) |

冻结特征模型**不再**把已经独立算完的 \(z_{\mathrm{rl}}\) 交给 policy 当状态。它只提供本帧 \(I_t\)（pool 后的 prefix）和 `ref_chunk`。跨 chunk 的 \(z\) 由可训 encoder 递推。

---

## 3. 改了什么

原版：

$$
z_t=\mathrm{Enc}_{\mathrm{S1}}(I_t),\qquad
\pi([\tilde a,\,z_t,\,\mathrm{proprio}]),\qquad
Q([z_t,\,\mathrm{proprio}],\,a)
$$
每一步的 \(z_t\) 和 \(z_{t-1}\) 无关。Policy 即使拼一个 sidecar 向量，输入里仍有已经完成的 \(z_t\) 和整段 \(\tilde a\)，历史进不去动作。

现在：

$$
z_t=\mathrm{Enc}\big([I_t;\,a_{t-1};\,r_{t-1};\,z_{t-1}]\big)
$$

$$
\pi([z_t,\,\mathrm{proprio}]),\qquad
Q([z_t,\,\mathrm{proprio}],\,a)
$$

`ref_chunk` **不进入** encoder，也 **不进入** \(\pi\) 和 \(Q\)。它只当预测头和 BC 的拟合目标。Encoder 的输入只有当前 \(I_t\)、上一时刻写出的 \(z_{t-1}\)、上一 chunk **实际执行**的 \(a_{t-1}\) 和 \(r_{t-1}\)。Stage 1 从未见过 `ref`，把 `ref` 拼进 encoder 会让写函数抄提案，历史进不去。

---

## 4. `mem` 就是这条 encoder loop

没有单独的 256 维记忆槽。记忆状态就是 Stage 1 同构的 RL token 
$$
z\in\mathbb{R}^{2048}
$$
。第一步没有历史时，\(z_{0}=z_{\mathrm{init}}\)（可学习），\(a_{0}=0\)，\(r_{0}=0\)。

### 4.1 冻结特征模型只出 \(I_t\) 和 `ref`

`extract_rlt_obs`（官方 `openpi`）在 `rlt_return_prefix: True` 时额外返回：

- `prefix_embs`：当前 VLM prefix，用 `adaptive_avg_pool1d` 收到 `loop_prefix_len=64`（整段 1024×2048 无法进 replay）
- `prefix_mask`
- `ref_chunk`、`proprio`（与原版相同）

冻结 Stage 1 encoder 仍会算一个 \(z_{\mathrm{rl}}\) 以兼容原版键；`use_mem=True` 时 policy **丢掉**这个值，用 loop 重写。

### 4.2 可训 loop：`RLTLoopEncoder`

类：`rlinf/models/embodiment/modules/rlt_mem_write.py`，挂在 `RLTMLPPolicy.loop_encoder`。名字含 `encoder`，FSDP 把它和 Q 头放进 **critic 优化器**。

内部是和 Stage 1 同构的 `RLTTokenEncoder`（默认 2 层、8 head，`prefix_seq_len=64`，没有 decoder），外加：

| token | 来源 |
| --- | --- |
| \(I_t\) | pool 后的 `prefix_embs`（冻结 VLM，不反传进 Pi0） |
| \(a_{t-1}\) | `action_proj`，上一 chunk **route 之后实际执行**的动作，展平 80 维 |
| \(r_{t-1}\) | `reward_proj`，上一 chunk 各步 reward 的和 |
| \(z_{t-1}\) | 当作 encoder 的 RL token（不再用 Stage 1 那个与历史无关的 learnable embed） |

序列是 \([I_1\ldots I_{64};\,a;\,r;\,z]\)。Encoder 读出最后一位得到 \(z_t\)。这和 Stage 1「一个 RL token 去读 prefix」是同一条计算，只是 token 换成上一时刻的 \(z\)，并多了执行反馈 \((a,r)\)。**没有 decoder，不重建 \(I_t\)。**

同一模块上的约束头（只在训练用）：

- `pred_ref`：\(z_t\to\) 当前 `ref_chunk` 展平。这是 Stage 2 里 encoder 的主监督：\(z\) 必须能从 \(I_t\) 和历史推出 VLA 提案。
- `pred_r`：\(z_t\to\) 当前 chunk 的折现/加总回报。

### 4.3 Policy 只看见 loop 后的 \(z\)

ManiSkill 默认：\(z\in\mathbb{R}^{2048}\)，proprio 9。Actor / critic 输入是 `[z | proprio]` = 2057，**不再拼** 80 维 `ref`，也**不再拼**冻结 Stage 1 的 \(z\)。

因此 \(\pi\) 要出接近 VLA 的动作，必须从 \(z_t\) 里读出提案；而 \(z_t\) 只有在 encoder 真正融合了 \(I_t\) 和历史时才稳定。这就是 history 进入 action 的路径。

### 4.4 Rollout：\(z\) 存在 policy 上，不进 checkpoint

`RLTMLPPolicy` 上两份 **非参数** 状态（每个 env 一行）：

- `_rollout_mem`：上一步写出的 \(z_t\)，下一步当 \(z_{t-1}\)
- `_rollout_prev_action`：上一步 commit 的执行动作

一次 `predict_rlt_actions` 的顺序：

1. 冻结 OpenPI：`extract_rlt_obs` → `prefix_embs`, `ref_chunk`, `proprio`。
2. `_prepare_rollout_mem`：若上一 chunk `dones` 为真，该 env 的 \(z\)、\(a\) 置回 \(z_{\mathrm{init}}\) / 0，\(r\) 也置 0；否则 \(r_{t-1}\) = 上一 chunk reward 求和。然后 \(z_t=\mathrm{Enc}([I_t;a_{t-1};r_{t-1};z_{t-1}])\)，写回 `_rollout_mem`。
3. \(\pi(z_t,\mathrm{proprio})\) 出学生动作。
4. route 在关键阶段前换成 `ref`。mem 的 train 在 clone 未就绪时关键阶段也继续执行 `ref`；eval 过 warmup 后执行学生。
5. `commit_rollout_action(实际执行的 a)`：下一 chunk 的 \(a_{t-1}\) 必须是环境里真正走的。

`dones` / `rewards` 来自 **上一** 次 env step。本步的 \(r_t\) 要等环境走完，下一次 predict 才进入写。

### 4.5 一条轨迹里已经有很多个 chunk，K 是其中最近几段

这里的「一步」不是环境里的 1 个 `env.step`，也不是整条 episode，而是 **一次 policy 调用 = 一个 action chunk**。

当前 ManiSkill 配置：

| 单位 | 多长 | 对应什么 |
| --- | --- | --- |
| env step | 1 个关节指令 | `env.step(a)` |
| chunk（记忆的 1 步） | 10 个 env step | 一次 `predict`，`num_action_chunks: 10` |
| episode | 最多 500 个 env step | 同一颗 peg、同一局，约 **50 个 chunk** |
| 一次 rollout epoch | `max_steps_per_rollout_epoch: 500` | `500/10=50` 次 chunk 循环；`auto_reset: False`，每个 env 正好走完一局 |

采集循环是：

```
for chunk_t in 0 .. 49:          # 同一 env、同一局
    看 I_t，出 10 维动作 chunk
    环境连续 step 10 次
    记下 (I_t, a_t, r_t, done)
    下一圈用新的图像 I_{t+1}
```

`EmbodiedTrajectoryBuilder` 每个 chunk `append` 一次。送到 actor 的那条 `Trajectory` 的时间维是 **50，不是 1**。原版训练再把它 **拆成 50 条单步 replay**，所以看起来像「关键阶段只预测一次」。数据里其实已经有「上一 chunk 插偏了、这一 chunk 再试」的顺序。

默认 **不在 replay 里再存 K 段图像**（`mem_unroll_len: 1`）。采集时 \(z\) 仍沿 50 个 chunk 递推；训练只对当前 \(I_t\) 和存下来的 \(z_{t-1},a_{t-1},r_{t-1}\) 写一次。把 4 段 `64×2048` prefix 拷进每一行、再在 critic/actor/预测里各展开一遍，单步会到小时级。

若要训练期反传多段视觉，把 `mem_unroll_len` 调到 2–4，并接受更慢、更大的 replay。默认 prefix pool 到 16 token；encoder **2 层**（与 Stage 1 相同），不要用 1 层。

记忆就是这个递推：

```
chunk 34: Enc(I_34, z_init, 0, 0)     → z_34
chunk 35: Enc(I_35, z_34, a_34, r_34) → z_35
chunk 36: Enc(I_36, z_35, a_35, r_35) → z_36
chunk 37: Enc(I_37, z_36, a_36, r_36) → z_37   ← π 和 Q 用这个
```

\(z_{37}\) 里有「前几段实际怎么走、回报怎样」。只看 \(I_{37}\) 的原版 RLT 没有这段。局与局之间 `done` 断开，窗口不跨 episode。

### 4.6 为什么训练也要展开，而不是只在 rollout 里记 \(z\)

原版 Stage 2 是一步 TD：关键阶段拿当前 \(I_t\) / `ref` 预测一次 \(a\)。如果训练也只做
\(z_t=\mathrm{Enc}(I_t,z_{t-1}^{\mathrm{detach}})\)，再让 \(D_{\mathrm{ref}}(z_t)\approx\tilde a_t\)，encoder 可以完全丢掉 \((a,r)\) 历史——当前 `ref` 本来就由 \(I_t\) 决定。那样 actor 仍是「关键阶段一次动作预测」，和原版没有区别。

因此：

1. **采集**仍按 chunk 顺序走。Policy 上的 `_rollout_mem` 跨 chunk 递推 \(z\)，`done` 清掉。这是推理时的 loop。
2. **写入 replay 时**，同一条 episode 上把最近 `mem_unroll_len`（默认 4）步的 \(I,a,r,\mathrm{ref}\) 拷到这一行（`hist_*`）。中间遇到 `done` 的步标成 invalid。
3. **训练**用当前 encoder **展开**这个窗口（步与步之间不 detach）：

\[
z_k=\mathrm{Enc}(I_k,z_{k-1},a_{k-1},r_{k-1}),\quad k=t-K+1,\ldots,t
\]

然后 \(\pi(z_t)\)、\(Q(z_t,a_t)\)、以及窗口内每一步的 \(D_{\mathrm{ref}}(z_k)\approx\tilde a_k\)。TD 仍是一步，但 **状态 \(z_t\) 是这段历史的函数**，改早期的执行 \(a\) 会改现在的 \(\pi/Q\)。

`ref` 只当每一步的预测目标，不进 encoder。Actor 的 \(-Q\) 仍 detach 最终 \(z_t\)，encoder 由 critic 的 TD 和 `pred_ref` 更新。

---

## 5. 两条约束：BC 在动作上，预测 `ref` 在 \(z\) 上

\[
\mathcal L_{\pi}=-q\,Q(z,\pi)+\beta\,\|\pi-\tilde a\|^2
\]

\[
\mathcal L_{Q}=\underbrace{\|Q-y\|^2}_{\mathrm{TD}}
+\lambda_{\mathrm{ref}}\|D_{\mathrm{ref}}(z)-\tilde a\|^2
+\lambda_{r}\|D_{r}(z)-r\|^2
\]

| 约束 | 打在谁身上 | 作用 |
| --- | --- | --- |
| BC \(\|\pi-\tilde a\|\) | **执行动作** | 输出必须还在 VLA 先验里 |
| \(D_{\mathrm{ref}}(z)\approx\tilde a\) | **loop encoder** | Stage 2 的主辅助：\(z\) 要根据 \(I_t\) 和历史推出当前 `ref_chunk` |
| \(D_r(z)\approx r\) | **loop encoder** | \(z\) 里要留得住上一段结果 |
| \(-Q\) / TD | \(\pi\) 和 \(Q\) | 只在 BC 附近做对任务更好的微调 |

`ref` 只做三件事：route 提案、BC 目标、\(D_{\mathrm{ref}}\) 标签。不进 encoder，不进 \(\pi\)，不进 \(Q\)。不重建 \(I_t\)。

ManiSkill mem yaml 的 BC / \(Q\) 日程与原版相同（warmup \(\beta=7,q=0.05\)，online \(\beta=2.5,q=0.45\)）。`reference_dropout` 关掉：输入里已经没有 `ref`。

---

## 6. 数据流（一个 chunk）

```
冻结 OpenPI
    I_t, ℓ  →  prefix_embs (pool 64), ref_chunk

loop encoder（可训）
    Enc([I_t; a_{t-1}; r_{t-1}; z_{t-1}])  →  z_t

actor（可训）
    (z_t, proprio)  →  π
    route：关键阶段前用 ref；之后用 π

环境
    执行 a_t  →  r_t, done, I_{t+1}

训练（同一 episode 展开 K 步）
    z_t = Enc(I_{t-K+1:t}, a, r)
    π：-Q(z_t) + BC(π, ref_t)
    Q：TD(z_t, a_t) + 预测(z_k → ref_k)
```

---

## 7. 同场景 zero-shot 怎么成立

| 随任务变 | 固定、可迁 | 每条 episode 清掉 |
| --- | --- | --- |
| \(\ell\) → VLA → 新的 `ref` 和新的 \(I_t\) | loop 写法、\(D_{\mathrm{ref}}\)、残差 MLP | \(z\) 内容 |
| 当前目标上的 \(r\)（训练用） | BC：输出跟**当时**的 `ref` | 本局 \(a,r\) 历史 |

换任务只换 instruction。VLA 出新提案和新 prefix；BC 和 \(D_{\mathrm{ref}}\) 跟着新 `ref` 走；encoder 权重不动。不要把任务名或当前 success 写进 \(Q\)。

---

## 8. 明确不做

- \(z\) 注入 Stage 1 VLA。
- 自然语言反馈进冻结 VLM（B2）。
- 特权 \(d\) / success 拼进 \(Q\)。
- GRU 或 256 维 sidecar 去混合**已经算完**的 Stage 1 \(z\)：当前 \(z\) 与历史无关，policy 也仍在抄 `ref`。
- \(\beta=0\)：输出可以完全离开 VLA。
- 重建 \(I_t\)：Stage 2 要的是从历史推出 `ref`，不是再做一遍 Stage 1 reconstruction。
- `ref_chunk` 进 encoder。
- 残差 \(a=\tilde a+\Delta(z)\)：那是原文 pass-through，不是 mem。
- 特权 \(d\) 进 \(Q\) 特征。势函数只允许出现在 \(r\)。

---

## 9. 文件与配置

| 路径 | 作用 |
| --- | --- |
| `rlinf/models/embodiment/modules/rlt_mem_write.py` | `pool_rlt_prefix` + `RLTLoopEncoder` |
| `rlinf/models/embodiment/modules/rlt_token_transformer.py` | encoder 可传入 `rl_token` / `extra_tokens` |
| `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` | `use_mem` 时 \(\pi/Q\) 只吃 loop 后的 \(z\) |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | 官方 OpenPI：`rlt_return_prefix` 出 pool 后的 \(I_t\) |
| `rlinf/algorithms/rlt/rollout.py` | 跨 chunk 携带 \(z,a,r\)；`done` 重置 |
| `rlinf/algorithms/rlt/transition.py` | replay 可选 prefix / \(z_{\mathrm{prev}}\) 键 |
| `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` | critic：TD + 预测 ref/\(r\)；actor：\(-Q+\mathrm{BC}\) |
| `examples/embodiment/config/maniskill_rlt_stage2_ac_mlp_mem.yaml` | 完整 Stage 2 配置（官方 OpenPI + TensorBoard，不继承） |
| `rlinf/envs/maniskill/rlt_potential.py` | \(\Phi\) 只进 \(r\) |
| `tests/unit_tests/test_rlt_mem.py` | actor 维数、历史改变动作、encoder 不看 ref、预测 ref、done 重置 |
| `tests/unit_tests/test_rlt_route.py` | train 在 clone 未就绪时继续执行 VLA |
| `tests/unit_tests/test_rlt_potential.py` | 接近 hole 时 \(r>0\)，首步无 shaping |

原版配置保持 `use_mem` 默认关，也不开 `rlt_return_prefix`。

建议看的 TensorBoard 标：

| 标 | 含义 |
| --- | --- |
| `env/` train success | VLA 采集是否还在成功。clone 未就绪时应接近原版 warmup |
| `eval/` success（warmup 之后） | \(\pi(z)\) 自己插得进不算。未过 `student_action_ref_max` 前允许为 0 |
| `action_ref_abs_mean` / `action_ref_ema` / `student_clone_ready` | 克隆是否够格切学生采集 |
| `replay/reward_mean`、`reward_positive_rate` | 势函数是否让 \(r\) 在成功前就有正负，而不是全 0 |
| `pred_ref_loss`、`bc_loss`、`q_data` / `q_pi` | \(z\) 是否在恢复 `ref`；\(Q\) 是否不再贴 0 |

`mem_unroll_len` 过小会退回「只看当前帧」。

---

## 10. 和原文 RLT、DSRL 的位置

- **原版 RLT：** 条件生成 + BC，在 `ref` 附近局部编辑。本方案保留 BC 和冻结 VLA，把「看见 `ref`」改成「从 loop 后的 \(z\) 恢复 `ref`」，并让 \(z\) 带上 \((a,r)\) 历史。
- **原文消融：** w/o Pass-Through（去掉 actor 里的 `ref`）仍能收敛但更慢；\(\beta=0\) 掉点最大。本方案对应前者 + encoder loop，**不**对应后者。
- **DSRL：** SAC 的动作是扩散初始噪声。本方案仍在真实动作空间里做 BC。

---

## 11. 为了让 mem 的优点能被看见

原版成功率曲线比的是「当前帧能不能贴住 VLA」。那条曲线上 mem 没有优势，残差也只是复现 RLT。

mem 只多 \(z_t(I_t,z_{t-1},a_{t-1},r_{t-1})\)。要让这段历史有用：

1. **\(r_{t-1}\) 在成功前就要有差别。** 训练 `reward_mode: success_potential`，\(\Phi=x_{\mathrm{hole}}-w\cdot d_{yz}\)，\(r=1[\mathrm{success}]+\mathrm{coef}(\gamma\Phi'-\Phi)\)。eval 仍是 `only_success`，成功率口径不变。
2. **采集不能被学生写死。** `collect_student_when_ready: True`：train 关键阶段继续走 VLA，直到 `action_ref_ema <= 0.03` 才切学生。eval 过 `warmup_post_collect_updates`（默认 2000）后看学生，用来量克隆，不是拿来填 replay。单步更新上限 80，不加载 expert OpenPI。

主指标应是：同一局里 miss 后再插、以及 `Enc(I)` vs `Enc(I,a,r,z)` 的消融；不是第一段关键阶段超过原版 RLT。
