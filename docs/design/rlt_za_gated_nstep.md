# ZAP：(Z, A) Path 度量门控 n-step

谱系（RLT → BEE / eRLT → T-SAC / QC / AQC / DQC → ZAP）见 [`rlt_improvements_and_zap.md`](rlt_improvements_and_zap.md)。下文是 ZAP 的公式与落地清单。

**方法名：ZAP**（**(Z, A) Path**）。

- 损失名：`loss_type: rlt_zap`
- Spatial 配置：`examples/embodiment/config/libero_spatial_rlt_stage2_zap.yaml`（`rlt_phase_gate.mode: vlm`）
- Bowl 短任务：`examples/embodiment/config/libero_bowl_rlt_stage2_zap.yaml`（`rlt_phase_gate.mode: none`）
- MetaWorld MT50：`examples/embodiment/config/metaworld_50_rlt_stage2_zap.yaml`（suite 默认 `vlm`；单任务改 `none`）
- ManiSkill PegInsertion：`examples/embodiment/config/maniskill_rlt_stage2_zap.yaml`（几何 `rlt_policy_switch`，`rlt_phase_gate.mode: none`）
- 代码：`rlinf/algorithms/zap/`，`rlinf/workers/actor/fsdp_zap_policy_worker.py`

精度阶段门控与 ZAP 正交，由 `algorithm.rlt_phase_gate.mode` 控制：`vlm` 用冻结 VLM yes/no；`none` 表示全程 residual（短单任务）。

Stage-2 对比（同一 pi05-SFT，去掉论文 offline 预训练，只跑 online）：

- `rlt_qc`：Q-chunking（chunk-Q + Best-of-N 探索/备份）
- `rlt_dqc`：Decoupled Q-Chunking（长 horizon chunk-Q，蒸馏短 policy-Q）
- `rlt_aqc`：Adaptive Q-Chunking（多尺度 Q^k + 折扣归一化优势选择）
- `rlt_zap`：本方法

配置：`examples/embodiment/config/libero_{spatial,bowl}_rlt_stage2_{qc,dqc,aqc,zap}.yaml`，`examples/embodiment/config/metaworld_50_rlt_stage2_{ac_mlp,qc,dqc,aqc,zap}.yaml`，`examples/embodiment/config/maniskill_rlt_stage2_{ac_mlp,qc,dqc,aqc,zap}.yaml`。Stage 1 SFT：`examples/sft/config/{libero_pro_spatial,metaworld_50,maniskill}_rlt_stage1_sft_openpi_pi05.yaml`。

含义：backup 沿一条路径走；$z$ 与 $a$ 决定还能走多远，截断后再向同类格点借 $Q$。$Q_\theta(z,a)$ 仍只吃当前原子，不扩成 Q-chunking 的多 action 定义域。

行内公式用 `$...$`，独占一行的行间公式用 `$$...$$`。

---

## 1. 要解决的问题

Stage 2 在冻结 VLA 上训练 residual actor。每一步能拿到两样东西：

- $z=\varphi(o)$：冻结 VLM prefix 的语义向量，不是马尔可夫状态。
- $a$：这次真正执行的动作（control step；chunk 只是环境执行方式，不进 $Q$ 的动作自变量）。

评论家仍是

$$Q_\theta(z,a)$$

只吃**当前**这一对，不吃未来动作序列，也不吃 $(z,a)$ 对序列。

稀疏成功奖励出现在 episode 末尾。一步 TD 要把值沿时间爬回精细段，太慢。朴素 n-step 把行为策略的中间动作累进目标，off-policy 有偏。Q-chunking 用「把 $a_{t:t+h}$ 写进 $Q$」消这个偏——那是他们的贡献，这里不允许照搬。RLT 的 $\pi$ 又是近确定的（`fixed_std ≈ 0.002`），密度比 $\pi/\pi_\beta$ 几乎是 0/1，Retrace 的 IS 退化。

因此需要一条**新的** n-step 路：不扩 $Q$ 的动作维，不用失效的密度比，用 $(z,a)$ 决定「这条路径还能不能往前走」以及「停下后向谁借 $Q$」。

---

## 2. 核心思路

Q-chunking 的**问题**（可以借）：产生这段 backup 的变量，不能待在估计器外面。

Q-chunking 的**构造**（不能借）：把中间动作写进 $Q$ 的定义域，使一步 TD 等于无偏 n-step。

我们的构造不对称：

- $a$ 仍只作为当前动作出现在 $Q_\theta(z,a)$ 里，不拉长成 $a_{t:t+h}$。
- $z$ 不当马尔可夫 $s_t$，不写进 $z'=f(z,a)$。
- $(z,a)$ 只出现在**续步开关** $c$ 和**截断后的借值权重** $w_j$ 里。

直观上：还在同一个 VLM 语义格、动作还贴着当前 $\pi$，就把后面的 $r$ 累进来（沿时间 n-step）；一旦格子裂开或动作离开 $\pi$，立刻停，避免 off-policy 中间段；停下来的续值，向 replay 里同一个格点上已经更新过的 $Q$ 借（跨轨迹传播）。

---

## 3. 核、续步许可 $c$、连乘 $C_k$

### 3.1 两个核

$$k_z(z,z')=\exp\bigl((\cos(z,z')-1)/\tau_z\bigr)$$

$$k_a(a,a')=\exp\bigl(-\|a-a'\|^2/\tau_a\bigr)$$

两者都在 $(0,1]$。$\tau_z,\tau_a>0$ 是带宽。不另训分类器。

### 3.2 为什么 $c$ 是乘积

从 backup 起点 $(z_t,a_t)$ 沿同一条 episode 走到 $t+k+1$：

$$c_{k+1}=k_z(z_t,z_{t+k+1})\cdot k_a\bigl(a^\pi_{t+k+1},\,a^{\mathrm{buf}}_{t+k+1}\bigr)$$

- $k_z$ 管语义：$z$ 已经换任务/换阶段，再累 $r$ 会把别人的奖励记到起点。
- $k_a$ 管策略：buffer 里的 $a^{\mathrm{buf}}$ 已经离开当前 $\pi(z)$，再累就是朴素 n-step 的 off-policy 偏置。

缺任何一边，续步都不合法，所以是乘积，不是加和、也不是只看 $z$ 或只看 $a$。

$a^\pi_{t+k+1}$ 是**当前** actor 在 $z_{t+k+1}$ 上的动作（推理，无梯度）。$a^{\mathrm{buf}}$ 是当时环境执行的动作。

### 3.3 什么叫 $c$ 大、什么叫小

不必做硬分类。像 Retrace 一样连乘：

$$C_0=1,\qquad C_{k+1}=C_k\cdot c_{k+1}$$

- $C_k$ 接近 $1$：从起点到这一步，语义一直近、动作一直贴 $\pi$，继续

$$y\leftarrow y+\gamma^k r_{t+k}$$

- $C_k<\varepsilon$（建议 $\varepsilon=10^{-2}$）：停。后面的奖励一律不进 $y$。

精细阶段里 $z$ 往往几乎不动，residual 又贴 VLA ref，$c$ 会连续接近 $1$，一次更新可以跨很多 control step。一偏转或跳出该阶段，$C_k$ 掉下来，自动退化成一步 TD。

---

## 4. 截断后的 $w_j$：向谁借 $Q$

$c$ 管**这一条路径**还能走多远。$w_j$ 只在截断点（或 episode 结束）算一次，管**向哪些别的轨迹借续值**。两者不是同一个权重。

设停在 $\tau$，要用当前 $\pi$ 在 $z_\tau$ 上的续值。Replay 第 $j$ 行是 $(z_j,a_j)$：

$$w_j\propto k_z(z_\tau,z_j)\cdot k_a\bigl(\pi(z_\tau),a_j\bigr),\qquad \sum_j w_j=1$$

- $w_j$ 大：别人也在这个语义格，动作也像当前 $\pi$，它的 $Q_\theta(z_j,\pi(z_j))$ 是同一格点的重复观测。
- $w_j\approx 0$：别的任务或完全不同的动作，不借。
- 没有邻居（$k_z$ 在候选集上接近均匀）：退回单点 $Q_\theta(z_\tau,\pi(z_\tau))$，就是普通一步 bootstrap。

借的是**当前 $\pi$ 的 $Q$**，不是行为策略的整段回报 $G$。这和「核平均 $G$」不是一回事：后者没有 TD 链，也不消除 off-policy MC 偏置。

候选集不要对整个 replay 做全量 softmax。实现上对 $z_\tau$ 做 top-$K$ 近邻（$K$ 如 32），只在这 $K$ 个上算 $w_j$。

---

## 5. 完整 backup

$$y_t=\sum_{k=0}^{\tau-1}\gamma^k r_{t+k}+\gamma^\tau\sum_j w_j\,Q_{\bar\theta}(z_j,\pi(z_j))$$

$$\tau=\min\{k\ge 1:C_k<\varepsilon\text{ 或 episode 结束}\}$$

$Q_{\bar\theta}$ 是 target 网络。Actor 目标不变：

$$L_\pi=-q_w\,Q_\theta(z,\pi(\mathrm{ref},z,\mathrm{proprio}))+b_w\|\pi-\mathrm{ref}\|^2$$

评论家：

$$L_Q=\bigl(Q_\theta(z_t,a_t)-y_t\bigr)^2$$

时间索引是 **control step**，不是 action chunk。环境仍可按 chunk 执行；存 replay 时把 chunk 拆成逐步 $(z,a,r,z_{\text{next}})$。同一 chunk 内 $z$ 不变（一次 prefix），$k_z=1$，续步只由 $k_a$ 决定。跨 chunk 时 $z$ 更新，$k_z$ 开始起作用。

---

## 6. 信用如何变快

### 沿时间

成功往往在末尾才给 $1$。一步 TD 要 $T$ 次更新才能爬回精细段起点。$C_k$ 一直大时，一次更新就是把 $r_{t:t+\tau-1}$ 整段折现到 $(z_t,a_t)$。末尾成功乘 $\gamma^\tau$ 直接打到起点。$\tau$ 是这段还贴着 $\pi$ 的长度。

这是同一条轨迹上的 n-step 信用。无偏靠的是「路径仍在 $(z,\pi)$ 格内」，不是把中间 $a$ 写进 $Q$。

### 跨时间

截断处 $\bar Q_\tau=\sum_j w_j Q_j$。另一条已经成功、且当时 $(z,a)$ 很像你的轨迹，它的 $Q$ 已经偏高。你还没成功，但格点相同，这次更新把那个高 $Q$ 借过来。不必等你这条自己把 TD 爬完。

$c$：沿这条路径能跳几步。$w_j$：跳完后向哪些同类格点取值。

---

## 7. 为什么这能对上三件事

| 要求 | 做法 |
|---|---|
| 无偏 n-step（新思路） | 续步当且仅当语义还在格内且 $a$ 还贴 $\pi$。近确定连续动作下，用度量代替失效的 $\pi/\pi_\beta$。不是 Q-chunking 的「动作进 $Q$」。 |
| 沿时间加快 | $c$ 连续大时，一次 backup 跨过 $\tau$ 步。 |
| 跨时间加快 | 截断 bootstrap 来自别的轨迹上同一 $(z,a)$ 格。 |
| 消除 off-policy n-step 偏置 | 从不沿偏离 $\pi$ 的路径做固定地平 n-step；偏离由 $k_a$ 切断。也不把 $z'$ 当 $a\mapsto z'$。 |

Tabular + $c_k\le \pi/\pi_\beta$ 时，这就是 Retrace 的保守备份，对 $Q^\pi$ 无偏。我们用度量球代替密度比：当核 $\approx 1$ 当且仅当这些 $(z,a)$ 在 $\pi$ 下可互换时，近似成立。核不准时偏置会回来，这是这个方法的主要风险。

---

## 8. 完整实现思路

### 8.1 数据

Rollout 仍用冻结 VLA 产 $z$ 与 `ref_chunk`，residual actor 产 $a$。Replay **逐步**存：

- `z_rl`：该 step 的冻结嵌入（chunk 内可复制同一向量）
- `action`：该 control 的 $a_t$
- `reward`：该 step 的 $r_t$
- `done`
- `episode_id` 与 `time_in_episode`：沿轨迹走 $c$ 必须能取到 $t,t+1,\ldots$

不要只存一个 flatten 的 chunk 行却在评论家里当单步。

### 8.2 估 $y_t$（无梯度）

对 minibatch 里每个 $t$：

1. 沿 `episode_id` 向前取最多 $N_{\max}$ 步（如 32）。
2. 对每个前瞻步算 $a^\pi$（target 或当前 actor，eval 模式）。
3. 算 $c_{k+1}$、$C_{k+1}$，累加 $\gamma^k r$，直到 $C<\varepsilon$ 或 done。
4. 用截断处的 $z_\tau$ 与 $\pi(z_\tau)$，在 replay（或当前 sample window）里对 $z$ 取 top-$K$，算 $w_j$，bootstrap $\sum_j w_j Q_{\bar\theta}(z_j,\pi(z_j))$。
5. 无邻居则 $Q_{\bar\theta}(z_\tau,\pi(z_\tau))$。

### 8.3 网络

- $Q_\theta(z,a)$：MLP，输入当前 $z$ 与当前 $a$，不是 Transformer 序列头，不是 $Q(s,a_{1:h})$。
- Actor：现有 residual $\pi(\mathrm{ref},z,\mathrm{proprio})$。
- Target 网络软更新，与 SAC 相同。

### 8.4 损失与配置（建议）

```yaml
algorithm:
  loss_type: rlt_zap
  gamma: 0.99
  za_tau_z: 0.1
  za_tau_a: 0.05
  za_truncate_eps: 0.01
  za_max_horizon: 32
  za_neighbor_k: 32
  za_enable_neighbors: True
  q_weight: 1.0
  bc_weight: 1.0
  reference_dropout_prob: 0.5
```

`n_step` 保持 `1`（不再做固定多 chunk TD）。有效地平由 $C_k$ 决定，上限是 `za_max_horizon`（路径上的 atom 数；当前实现里一个 atom 是一个 action chunk）。

### 8.5 算力

沿轨迹走 $N_{\max}$ 步要反复问 $\pi(z)$，可对窗口里的 $z$ 一次 batched 前向。top-$K$ 用 sample window（如 16k）上的 $z$ 矩阵乘，不要扫整个 buffer。$Q$ 的 bootstrap 对 $K$ 个邻居 batched。

### 8.6 落地顺序

1. Replay 改为逐步 $z,a,r$，带 episode 索引。
2. 只实现 $c$ 截断的 on-trajectory n-step，bootstrap 用单点 $Q(z_\tau,\pi)$。先验证沿时间信用。
3. 再加 top-$K$ 的 $w_j$ 跨轨迹 bootstrap。
4. 消融：固定 n-step 无 $c$、只有 $k_z$、只有 $k_a$、关掉 $w_j$。

---

## 9. 和 RLT、Q-chunking、T-SAC 的异同

### 9.1 RLT

**它做什么。** 冻结 VLA，$z$ 作紧凑状态，residual $\pi(\mathrm{ref},z,\mathrm{proprio})$，BC + 最大化 $Q$。

**优点。** 不 RL 微调海量 VLA 权重；有语言条件先验；BC 限制偏离 ref。

**缺点。** 精细阶段在 ManiSkill 靠几何开关，LIBERO 没有；每条转移独立做一步/chunk TD，稀疏奖励传播慢；$z$ 当 $s$ 用并不马尔可夫。

**同。** 我们仍用冻结 $\varphi$、residual actor、$Q(z,a)$、BC。RLT 是底座，不是贡献。

**异。** 不把几何开关或 VLM yes/no 当方法；backup 由 $(z,a)$ 度量门控，而不是每步独立的 SAC/TD3。

### 9.2 Q-chunking

**它做什么。** $Q(s,a_{t:t+h})$，备份 $R^{(h)}+\gamma^h Q(s',a')$。中间动作在 $Q$ 里，一步 TD 等于无偏 n-step，价值沿时间快传 $h$ 步。

**优点。** 消朴素 n-step 的 off-policy 偏置；与 chunk 策略对齐；不依赖密度比。

**缺点。** 贡献就是「无偏多 action $Q$」；$s$ 仍须马尔可夫；把 $s$ 换成 $z$ 既侵权又引入投影误差；不跨轨迹借值。

**同。** 都针对「n-step 中间段 off-policy」和「要沿时间加快」。

**异。** 他们改 $Q$ 的定义域（齐次拉长 $A\to A^h$）。我们改 backup 的续步测度：$Q$ 仍是一对 $(z,a)$，中间 $a$ 只进 $c$。跨轨迹 $w_j$ 是他们没有的。

### 9.3 T-SAC

**它做什么。** 因果 Transformer，$Q$ 在 $[z,a_0,\ldots,a_{T-1}]$ 每个动作位置出值。

**优点。** 能打分 chunk 内依赖；前缀条件 $Q$。

**缺点。** 序列成本高；不决定何时续 n-step；不消 off-policy n-step；不跨轨迹。

**同。** 都看见 $z$ 和一段 $a$。

**异。** T-SAC 把序列放进 **$Q$ 网络**。我们的序列只出现在 **backup 的路径积分**里，$Q$ 仍是 MLP。T-SAC 不回答 $c$ 和 $w_j$。

### 9.4 对照表

| | RLT Stage 2 | Q-chunking | T-SAC | 本方案 |
|---|---|---|---|---|
| $Q$ 输入 | $(z,a_{\mathrm{chunk}})$ | $(s,a_{t:t+h})$ | 序列 $[z,a_{0:T})$ | 当前 $(z,a_t)$ |
| n-step | 无，或有偏 | 无偏，靠动作进 $Q$ | 不负责 | 无偏/保守，靠 $(z,a)$ 续步 |
| 沿时间 | 一步爬 | 每次跳 $h$ 步 | 不负责 | $C_k$ 大时跳 $\tau$ 步 |
| 跨时间 | 无 | 无 | 无 | 截断处 $w_j$ 借格点 $Q$ |
| off-policy n-step | 未处理 | 动作进 $Q$ 消除 | 未处理 | 偏离 $\pi$ 则切断 |
| $z'$ 当动力学 | 常当作 $s'$ | $s'$ 合法 | 状态 token | 只用于「格是否还相同」 |

---

## 10. 我们解决了什么，没解决什么

**解决了。**

1. 近确定连续控制下，IS Retrace 退化、又不能用 Q-chunking 扩 arity 时，如何做 n-step：用 $(z,a)$ 度量当续步许可。
2. 稀疏成功如何沿本轨迹一次回灌：$c$ 在精细段连续大。
3. 如何让已经成功的同类格点帮还没成功的轨迹：$w_j$ 借 $Q$，不是借行为策略的 $G$。
4. 不把冻结 $z$ 当成马尔可夫 $s$，也不学习 $a\mapsto z'$。

**没解决、也不声称。**

- 无偏多 action 的 MDP $Q(s,a_{1:h})$（Q-chunking 的定理）。
- Stage 1 的 $z$ 质量；核是坏度量时 $c$ 和 $w_j$ 会错。
- 几何精细开关 / VLM yes/no 当科学贡献（那是工程，需要的话只当 replay 过滤）。
- 核平均整段 $G$ 的方差魔术：那条路不做。

**主要风险。** $k_z$ 假邻居会把错误奖励或错误 $Q$ 传过来；$\tau_z,\tau_a$ 过大等于关掉门控（退化成固定 n-step）。必须用消融证明两个核都在工作。

---

## 11. 明确拒绝的旧拼接

- 在 $z$ 上做 Q-chunking：$Q(z,a_{\mathrm{chunk}})\leftarrow R^{(h)}+\gamma^h Q(z',a')$。
- 对 $(z,a)$ 对序列再做一次 Theorem A.1（同一构造换原子）。
- in-batch softmax 平均 TD 目标或平均 $G$。
- $\mathrm{cosine}(z,z_0)$ 或 VLM yes/no 当方法主体。
- InfoNCE / 双线性 $W$ 改冻结 $\varphi$。
- 加性双 $V^s$。

---

## 12. 一句话

Q-chunking 靠「中间动作进 $Q$」做无偏 n-step。我们靠「$z$ 是否还在格内、$a$ 是否还贴 $\pi$」决定 n-step 能走多远，停下来再向同类格点的当前 $Q$ 借值。$Q$ 的形状不变；变的是 backup 沿着哪条路径走、向谁取值。
