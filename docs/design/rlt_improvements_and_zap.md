# RLT 谱系：表示改进、Chunk 价值，以及 ZAP

本文从 **RL Token (RLT)** 出发，按两条改进轴整理仓库里已经落地的方法，最后落到 **ZAP**。

- **轴 A：针对 RLT 本身。** 不改 $Q$ 的动作定义域，改 residual 约束或 $z$ 怎么从冻结 VLA 里取出来。对应 **BEE**、**eRLT**。
- **轴 B：针对 action chunk 的价值估计。** 承认 VLA 按 chunk 执行，把一段时间动作写进 $Q$ 或序列头，换更快、更无偏的 backup。对应 **T-SAC**、**Q-chunking (QC)**、**Adaptive QC (AQC)**、**Decoupled QC (DQC)**。
- **ZAP** 不走轴 B 的「动作进 $Q$」，也不重做轴 A 的表示。它改的是 **backup 沿哪条路径走、停下来向谁借 $Q$**。

仓库对照（同一 $\pi_{0.5}$ Stage-1，Stage-2 只跑 online）：

| 方法 | `loss_type` | 配置示例 | 代码 |
|---|---|---|---|
| RLT AC / TD3 | `rlt_ac` / `rlt_td3` | `*_rlt_stage2_ac_mlp.yaml` | `fsdp_rlt_ac_policy_worker.py` |
| BEE | `bee` | `libero_pro_bowl_bee.yaml` | `rlinf/algorithms/bee/` |
| eRLT | `embodied_sac` + `use_erlt` | `libero_pro_bowl_erlt_*.yaml` | `rlinf/models/embodiment/modules/erlt.py` |
| T-SAC | `rlt_ac` + TSAC critic | `libero_pro_bowl_rlt_stage2_ac_tsac.yaml` | `tsac_transformer_critic.py` |
| QC / AQC / DQC | `rlt_qc` / `rlt_aqc` / `rlt_dqc` | `*_rlt_stage2_{qc,aqc,dqc}.yaml` | `rlinf/algorithms/qc/` |
| ZAP | `rlt_zap` | `*_rlt_stage2_zap.yaml` | `rlinf/algorithms/zap/` |

行内公式用 `$...$`，独占一行的行间公式用 `$$...$$`。更细的 ZAP 公式与落地清单见 [`rlt_za_gated_nstep.md`](rlt_za_gated_nstep.md)。

---

## 1. RLT：定义与留下的缺口

**RL Token**（Xu et al., *RL Token: Bootstrapping Online RL with Vision-Language-Action Models*）把「表示学习」和「online 控制」拆开。RLinf 里简称 **RLT**。

### 1.1 两阶段

**Stage 1（演示）。** 从 VLA 检查点出发，同一批 demo 上优化两个目标：

$$L = L_{\mathrm{rlt}} + \alpha\, L_{\mathrm{vla}}$$

- $L_{\mathrm{vla}}$：普通 flow-matching / 动作模仿。
- $L_{\mathrm{rlt}}$：token transformer 把 VLM prefix 压成向量 $z=\varphi(o)$，再用 decoder 重建 prefix（MSE）。Prefix 对重建 detach，encoder / decoder 受训。

**Stage 2（online）。** 冻结 Stage-1 特征模型（VLA + $\varphi$）。每一步观测变成

$$o^{\mathrm{rl}} = (z,\; x_{\mathrm{proprio}},\; a^{\mathrm{ref}})$$

其中 $a^{\mathrm{ref}}$ 是冻结 VLA 的参考动作 chunk。轻量 residual actor 输出真正执行的 $a$；critic 估 $Q(z,a)$（实现里常把整个 chunk 展平进 $Q$）。Actor 目标是最大化 $Q$ 加对 $a^{\mathrm{ref}}$ 的 BC：

$$L_\pi = -q_w\, Q_\theta(z,\pi(\mathrm{ref},z,x)) + b_w\|\pi-\mathrm{ref}\|^2$$

$z$ **不注回 VLA**。VLA 只提供先验动作和语义嵌入。

### 1.2 核心动机

直接 RL 微调整网 VLA 贵、且容易冲掉语言先验。RLT 的判断是：VLA 已经会合理动作，缺的是一个 **紧凑、可给 actor-critic 用的状态**，以及在参考轨迹附近做 residual 修正。

### 1.3 解决思路

把 $\varphi(o)\to z$ 从控制回路里拆出来，先用重建在 demo 上学；online 只训小 MLP。探索被约束在 VLA 行为附近。

### 1.4 优势

- 不更新海量 VLA 权重，真机 / 仿真都能在有限交互里跑 online RL。
- 语言条件和参考 chunk 还在，策略不会从零摸索。
- 接口干净：后面所有方法都还能复用 $(z,x,a^{\mathrm{ref}})$。

### 1.5 劣势（后面每条改进各打一块）

1. **$z$ 是当前 prefix 的压缩，不是技能，也不保证动作相关。** 重建可以忽略控制所需的细偏差；$z$ 被当成马尔可夫 $s$ 用，其实不是。
2. **Residual + BC 不知道何时该大胆修正。** 固定 $\|a-a^{\mathrm{ref}}\|^2$ 在人介入、或 VLA 已经错模时过紧或过松。
3. **价值沿时间爬得慢。** 成功奖励稀疏、出现在 episode 末尾。一步 / 单 chunk TD 要很多次更新才能回到精细段起点。
4. **朴素 n-step 在 off-policy replay 上有偏。** 中间动作来自旧 $\pi_\beta$，不是当前 $\pi$。RLT 的 $\pi$ 又近确定（`fixed_std ≈ 0.002`），密度比 $\pi/\pi_\beta$ 退化成 0/1，Retrace 几乎失效。
5. **Chunk 只是执行方式。** 环境按 chunk 步进，但标准 RLT 的 $Q$ 并不为「这段动作作为整体」负责，也不处理 chunk 内依赖。

轴 A 针对 1–2，轴 B 针对 3–5。ZAP 针对 3–4，且明确拒绝用轴 B 的「把 $a_{t:t+h}$ 写进 $Q$」来消偏。

---

## 2. 针对 RLT 的改进：BEE 与 eRLT

这两条都不改 n-step 理论。BEE 改 **actor 被允许离 $a^{\mathrm{ref}}$ 多远**；eRLT 改 **$z$ 从 VLA 哪一层、哪些 token 聚合而来**。

### 2.1 BEE：介入自适应的 residual

论文：*Bee*（arXiv 2609.27450）。配置：`loss_type: bee`。

**动机。** 标准 RLT 用固定 BC 锚在 VLA 提案 $\tilde a$ 上。人介入时，真正该学的是 $\Delta^H = a^H-\tilde a$，而且不同状态上「能偏多远」不一样：对齐阶段要严，自由空间可以松。固定 $\eta\|a-\tilde a\|^2$ 无法表达这条约束。

**思路。** 额外训一个 Correction Model，在 $(s,\tilde a)$ 上预测人残差的对角高斯 $(\mu_\phi,\sigma_\phi)$。用马氏距离

$$\rho = \tfrac{1}{D}(a_\theta-\hat a^H)^\top\Sigma^{-1}(a_\theta-\hat a^H)$$

衡量策略是否还落在人可接受的修正椭球里。状态相关乘子 $\lambda_\omega(s)\ge 0$ 把 $\rho\le\varepsilon$ 做成原-对偶：

$$L_\pi = \mathbb{E}\Big[\big(-Q + \eta\|a-\tilde a\|^2 + \lambda\rho\big)/(1+\lambda)\Big]$$

人残差 buffer $\mathcal{B}_H$ 周期性更新 Correction Model；$\lambda$ 对 $\rho-\varepsilon$ 做对偶上升。VLA 仍然冻结。

**优势。**

- 约束随状态变：人常改的维度 $\sigma$ 大，$\rho$ 更松；人几乎不改的维度钉死。
- 人介入直接进椭球，而不是和匿名 BC 混在一起。
- 仍是 residual online RL，不碰 VLA 权重。

**劣势。**

- 依赖介入或种子修正；纯自主、无人数据时椭球退化。
- 不加速稀疏奖励的信用分配，也不消 off-policy n-step 偏置。
- $z$ 仍按 RLT 方式取，表示质量原样继承。

### 2.2 eRLT：动作相关的 token / 层路由

论文：*eRLT: Efficient VLA Reinforcement Learning via Action-Relevant Token Routing*。配置：`openpi.use_erlt: True`。

**动机。** RLT 把 **最后一层、固定 token 集** 重建压缩成 $z$。激励实验表明：对 online RL 最有用的层和 token **随任务变**。固定压缩会丢掉「动作修正和价值估计真正需要的」内部特征；在 VLA 外另训视觉编码器（DSRL）则完全不用这些内部特征。

**思路。** 冻结 VLA。另开一条前向：在 prefix 后追加 $K$ 个 routing token（默认 $K=1$），在选定层 $\{0,3,6,9,12,15,18\}$ 读这些位置并均值池化，再用任务级 softmax 层权重合成

$$z_t = \sum_i \alpha_i\, u_t^{(i)},\qquad \alpha=\mathrm{softmax}(\eta_{\mathrm{lay}}/\tau)$$

两阶段监督：

1. **离线初始化。** 临时 action probe 从 $z$ 回归专家动作 chunk（MSE）。probe 只看见 $z$，所以路由必须留下「能区分不同专家动作」的信息。按验证 MSE 选 $\eta$，丢掉 probe。
2. **Online。** critic 的 Bellman 误差回传到 routing；actor 用 detach 的 $z$。每 $U=200$ 步才开一次路由梯度，避免表示抖得太快。Replay 存原观测，路由更新时重算 $z$。

仿真里 actor 走 DSRL 式 latent-noise；真机走 RLT residual。主指标是归一化学习曲线 AUC，不是最终 SR。

**优势。**

- $z$ 按任务选层和 token，比最终层重建更贴动作 / 价值。
- 参数极少（约 $KD+m$），VLA 始终冻。
- 离线动作预测给出可学起点，online critic 再按成败微调。

**劣势。**

- 多一层 VLM 前向，算力和显存高于固定最终层 $z$。
- $z$ 仍然是 **状态摘要**，不是可选技能；不能像换 prompt 那样换 VLA 行为。
- 不处理 chunk backup / off-policy n-step。信用分配还是标准 SAC/DSRL TD。
- 受冻结 VLA 内部信息上限约束；路由学错层会引入噪声。

BEE 和 eRLT 互补：一个管「离提案多远」，一个管「$z$ 里留什么」。两者都默认 $Q$ 仍吃当前原子（或当前 latent），把「值如何沿时间走」留给下一节。

---

## 3. 针对 Chunk 的改进：T-SAC、QC、AQC、DQC

VLA 一次出一段动作再开环执行。轴 B 的共同判断是：若 $Q$ 只看见当前一步 / 当前短 chunk，backup 地平太短，稀疏成功传不回来；若强行做固定 n-step 又不把中间动作写进估计器，off-policy 有偏。

### 3.1 T-SAC：序列 $Q$ 头

配置：RLT actor + `TSACTransformerQHead`。

**动机。** 展平 MLP $Q(z,\mathrm{vec}(a_{1:h}))$ 把 chunk 当成无结构向量，看不到「先抬再合爪」这种前缀依赖。需要一个能对 **每个动作位置** 打分的 critic。

**思路。** 因果 Transformer，序列为 $[z,a_0,\ldots,a_{T-1}]$，只在动作位输出 $Q_t=Q(z,a_{0:t})$。训练时把相邻 RLT 转移打成多 chunk 前缀窗口（`pack_tsac_prefix_windows`）。Twin-Q，无共享权重。Actor 仍是 residual MLP。

**优势。**

- 前缀条件价值：能说「这段前缀已经好/坏」，而不只是整段一个标量。
- 与 chunk 内时间结构对齐，容量大于扁平 MLP。

**劣势。**

- 序列成本高，窗口一长就贵。
- **不决定 n-step 能走多远**，也不消中间段 off-policy 偏置。Backup 仍是逐步 / 按窗的 TD。
- 不跨轨迹借值。$z$ 仍当状态 token。

T-SAC 改的是 **$Q$ 网络形状**，不是 backup 测度。

### 3.2 Q-chunking（QC）

配置：`loss_type: rlt_qc`。论文构造的 online 适配：无单独 offline critic 预训练，冻结 VLA + residual 当行为先验。

**动机。** 朴素 n-step 把 $\sum_{k=0}^{h-1}\gamma^k r_{t+k}+\gamma^h Q(s_{t+h},\cdot)$ 当作目标，但中间 $a_{t+1:t+h-1}$ 来自 $\pi_\beta$。要无偏，产生这段回报的变量必须进估计器。

**思路。** 把动作定义域从 $\mathcal{A}$ 拉成 $\mathcal{A}^h$：

$$Q(s,a_{t:t+h}),\qquad y = R^{(h)}+\gamma^h Q(s',a'),\qquad R^{(h)}=\sum_{k=0}^{h-1}\gamma^k r_{t+k}$$

一步 TD 在数学上等于这段 n-step。探索和 bootstrap 的 $a'$ 都用 **Best-of-N**：从行为策略（此处为 residual + `ref_chunk`）采 $N$ 个 chunk，取 $Q$ 最高者。`n_step` 必须为 1，backup 长度等于 critic 的动作 chunk。

**优势。**

- 无偏多步 backup，不依赖密度比——正好避开 RLT 近确定策略下 Retrace 失效。
- 与「一次执行一个 chunk」对齐；信用一次跳 $h$ 步。
- Best-of-N 在先验支持里筛好 chunk，实现简单。

**劣势。**

- **贡献就是改 $Q$ 的 arity。** $s$ 仍须马尔可夫。把 $s$ 换成冻结 $z$ 会引入投影误差，且 $z'=\varphi(o')$ 不是 $a\mapsto z'$ 的动力学。
- $h$ 固定：太短传不回成功，太长动作空间维度爆炸、Best-of-N 难覆盖。
- 不跨轨迹传播已经学到的格点价值。
- 仓库约定：这是对照方法，ZAP **不允许**照搬「中间动作进 $Q$」。

### 3.3 Adaptive QC（AQC）

配置：`loss_type: rlt_aqc`。

**动机。** 固定 $h$ 对所有状态一刀切。精细对齐需要短地平、可反应；粗定位可以长地平。QC 不能按状态选尺度。

**思路。** 同时训一个长地平 $Q^h$（QC 式 Best-of-N backup）和一组前缀 critic $Q^k$（$k\in K$）。$Q^k$ 的目标是前缀回报再 bootstrap $V^h$（AQC Eq. 12–15）。Rollout 对候选 chunk 打分

$$A^k = (Q^k-V^k)/\gamma^k$$

再在样本维 z-score（Eq. 16），取 $\arg\max_{n,k} \tilde A^{k}(s,a^{(n)}_{1:k})$。RLinf 的向量化 LIBERO chunk 长度固定，所以 $k^*$ 选的是 **提交哪一个候选 chunk**，不是把单条样本截成退缩地平。

**优势。**

- 多尺度价值：同一批候选上比长短前缀，比死 $h$ 灵活。
- 折扣归一化后，不同 $k$ 的优势可比较。
- 仍享受 QC 的无偏长 backup（$Q^h$）。

**劣势。**

- 多个 critic + value 头，超参（$\kappa$、地平集合、N）多。
- $Q$ 定义域仍是「状态 + 一段动作」；$z$ 非马尔可夫的问题还在。
- 环境一步仍执行完整 policy chunk，自适应发生在 **选哪条候选**，不是真退缩地平控制。
- 同样不跨轨迹借 $Q$。

### 3.4 Decoupled QC（DQC）

配置：`loss_type: rlt_dqc`。

**动机。** 希望 critic 看很长的动作窗（多 chunk 无偏 backup），但 actor / Best-of-N 仍在短 policy chunk 上选，以免动作维和执行延迟一起爆。一个网络无法同时当「长价值」和「短策略」。

**思路。** 拆成两个 critic：

1. **$Q^h$（chunk critic）。** 输入拼接起来的多 chunk 动作窗，backup 跨 `dqc_backup_horizon`（须为 `num_action_chunks` 的倍数）。这是长地平无偏 TD。
2. **$Q^k$（policy critic）。** 只吃一个 policy chunk。用 expectile 从 $Q^h$ 蒸馏（$\kappa_d$），另训 $V$（$\kappa_b$）。

Rollout 的 Best-of-N 打在 $Q^k$ 上。没有 offline critic 预训练。

**优势。**

- 价值地平和策略地平解耦：backup 可以很长，执行 / 选动作仍短。
- Expectile 蒸馏比硬回归更稳，适合「长 $Q$ 教短 $Q$」。
- 保留 QC 的无偏长 backup。

**劣势。**

- 两套 $Q$ + $V$，训练比 QC 重，蒸馏误差会漏到短策略。
- 长窗打包（`pack_dqc_windows`）要求 replay 里有连续 chunk；边界和 done 要小心。
- 本质仍是「动作进 $Q$」家族；$z$ 当 $s$、不跨轨迹借值，与 QC 相同。

轴 B 四条共享一句话：**用估计器内部的一段时间动作，换无偏（或近似无偏）的多步信用。** 代价是 $Q$ 的定义域变大，并且都默认 $z$ 可以当 $s'$。

---

## 4. ZAP：不扩 $Q$，改 backup 路径

**ZAP** = **(Z, A) Path**。`loss_type: rlt_zap`。

精度阶段门控（`rlt_phase_gate.mode: vlm | none`）与 ZAP **正交**：前者是工程上的 residual 开关，不是方法主体。

### 4.1 在谱系里要解决什么

Stage 2 仍是冻结 $\varphi$、residual $\pi(\mathrm{ref},z,x)$、评论家

$$Q_\theta(z,a)$$

只吃 **当前** 一对。时间索引是 control step（实现里一个 atom 可以是一个 action chunk，但 $Q$ 不把未来动作当自变量）。

此时同时成立：

- 稀疏成功在末尾，一步 TD 太慢（RLT / eRLT / BEE 都没修）。
- 朴素 n-step off-policy 有偏。
- Retrace 的 $\pi/\pi_\beta$ 在近确定连续动作上退化。
- **不能** 采用 QC/AQC/DQC 的构造：那是「中间动作进 $Q$」，会变成对照方法而不是新贡献；再把 $s$ 换成 $z$ 还有投影误差。
- T-SAC 只改网络形状，不回答「这段 backup 还能不能往前走」。

因此需要第三条路：**$Q$ 的形状不变；用 $(z,a)$ 决定路径还能走多远，停下后再向同类格点借当前 $\pi$ 的 $Q$。**

### 4.2 实现思想

Q-chunking 可借的问题：产生这段 backup 的变量，不能待在估计器外面。

Q-chunking 不能借的构造：把中间动作写进 $Q$ 的定义域，使一步 TD 恒等于无偏 n-step。

ZAP 的构造不对称：

| 对象 | 放进哪里 | 不放进哪里 |
|---|---|---|
| 当前 $a_t$ | $Q_\theta(z_t,a_t)$ | 不拉长成 $a_{t:t+h}$ |
| 冻结 $z$ | 核 $k_z$（格是否还相同） | 不当马尔可夫 $s$，不学 $z'=f(z,a)$ |
| 中间动作 | 只进续步开关 $c$ | 不进 $Q$ 的输入 |
| 别的轨迹 | 截断后的借值权重 $w_j$ | 不借行为策略的整段回报 $G$ |

直观：还在同一个 VLM 语义格、动作还贴着当前 $\pi$，就把后面的 $r$ 累进来；格子裂开或动作离开 $\pi$，立刻停；停下来的续值，向 replay 里同一格点上已经更新过的 $Q_{\bar\theta}(z_j,\pi(z_j))$ 借。

两个核（带宽 $\tau_z,\tau_a>0$，不另训分类器）：

$$k_z(z,z')=\exp\bigl((\cos(z,z')-1)/\tau_z\bigr),\qquad k_a(a,a')=\exp\bigl(-\|a-a'\|^2/\tau_a\bigr)$$

从 backup 起点沿同一 episode 走一步：

$$c_{k+1}=k_z(z_t,z_{t+k+1})\cdot k_a\bigl(a^\pi_{t+k+1},\,a^{\mathrm{buf}}_{t+k+1}\bigr)$$

- $k_z$：语义换任务 / 换阶段则不能再把别人的奖励记到起点。
- $k_a$：buffer 动作已离当前 $\pi(z)$，再累就是朴素 n-step 偏置。
- 必须乘积。缺一边续步都不合法。
- $a^\pi$ 是 **当前** actor 在该 $z$ 上的动作（推理、无梯度）；$a^{\mathrm{buf}}$ 是当时环境执行的动作。

连乘（Retrace 风格，不用密度比）：

$$C_0=1,\qquad C_{k+1}=C_k\cdot c_{k+1}$$

$C_k\approx 1$：继续 $y\leftarrow y+\gamma^k r_{t+k}$。$C_k<\varepsilon$（默认 $10^{-2}$）：截断。精细段里 $z$ 几乎不动、residual 贴 ref 时，$c$ 连续接近 1，一次更新跨过很多 control step；一偏转就退化成一步 TD。

截断点 $\tau$ 上的跨轨迹借值（只算一次，和 $c$ 不是同一个权重）：

$$w_j\propto k_z(z_\tau,z_j)\cdot k_a\bigl(\pi(z_\tau),a_j\bigr),\quad\sum_j w_j=1$$

对 $z_\tau$ 在 sample window 上取 top-$K$（如 32），不做全 replay softmax。无邻居则退回 $Q_{\bar\theta}(z_\tau,\pi(z_\tau))$。借的是 **当前 $\pi$ 的 $Q$**，不是 $\pi_\beta$ 的 Monte Carlo $G$。

完整目标：

$$y_t=\sum_{k=0}^{\tau-1}\gamma^k r_{t+k}+\gamma^\tau\sum_j w_j\, Q_{\bar\theta}(z_j,\pi(z_j))$$

$$\tau=\min\{k\ge 1:C_k<\varepsilon\text{ 或 episode 结束}\}$$

$$L_Q=\bigl(Q_\theta(z_t,a_t)-y_t\bigr)^2$$

Actor 与 RLT 相同：最大化 $Q$ + BC。同一 chunk 内 $z$ 可复制，$k_z=1$，续步只由 $k_a$ 决定；跨 chunk 后 $z$ 更新，$k_z$ 开始起作用。

### 4.3 路线（实现顺序）

与「先扩 $Q$ 再蒸馏」相反。路线是 **先让路径可索引，再开门控，最后才借邻居**。

**数据。** Rollout 仍用冻结 VLA 产 $z$ 与 `ref_chunk`，residual 产 $a$。Replay **按 atom 存**，并带 `episode_id`、`time_in_episode`。不要只存一条 flatten 的 chunk 行却在评论家里当单步——否则 $c$ 无法沿轨迹走。

**估 $y_t$（无梯度）。** 对 minibatch 每个 $t$：

1. 沿 `episode_id` 向前最多 $N_{\max}$ 步（`za_max_horizon`，如 32）。
2. 对窗口里的 $z$ 一次 batched 前向，得到 $a^\pi$（eval 模式）。
3. 算 $c_{k+1}$、$C_{k+1}$，累加 $\gamma^k r$，直到 $C<\varepsilon$ 或 done。
4. 用截断处的 $z_\tau,\pi(z_\tau)$ 在 sample window 上对 $z$ 做 top-$K$，算 $w_j$，bootstrap $\sum_j w_j Q_{\bar\theta}(z_j,\pi(z_j))$。
5. 无邻居则单点 $Q_{\bar\theta}(z_\tau,\pi(z_\tau))$。

**网络。** $Q_\theta(z,a)$ 仍是 MLP，不是 Transformer 序列头，不是 $Q(s,a_{1:h})$。Actor 仍是 residual。Target 软更新与 SAC 相同。`algorithm.n_step` 保持 1；有效地平由 $C_k$ 决定。

**落地顺序（必须按此消融）。**

1. Replay 改为逐步 / 按 atom 的 $(z,a,r)$ + episode 索引。
2. 只做 $c$ 截断的 on-trajectory n-step，bootstrap 用单点 $Q(z_\tau,\pi)$。先验证 **沿时间** 信用。
3. 再打开 top-$K$ 的 $w_j$，验证 **跨轨迹** 借值。
4. 消融：固定 n-step 无 $c$、只有 $k_z$、只有 $k_a$、关掉 $w_j$。两个核都必须被证明在工作。

关键配置：`za_tau_z`、`za_tau_a`、`za_truncate_eps`、`za_max_horizon`、`za_neighbor_k`、`za_enable_neighbors`。$\tau_z,\tau_a$ 过大等于关掉门控，退化成固定 n-step。

### 4.4 解决了什么，没解决什么

**解决了。**

1. **近确定连续控制下的 n-step。** 不用失效的 $\pi/\pi_\beta$，也不扩 $Q$ 的 arity；用 $(z,a)$ 度量当续步许可。路径仍在 $(z,\pi)$ 格内时，backup 对 $Q^\pi$ 近似无偏 / 保守（tabular 且 $c_k\le\pi/\pi_\beta$ 时即 Retrace）。
2. **沿时间加快。** $C_k$ 连续大时，一次更新把 $r_{t:t+\tau-1}$ 折现到 $(z_t,a_t)$。末尾成功乘 $\gamma^\tau$ 直接打到精细段起点。$\tau$ 是这段还贴着 $\pi$ 的长度，不是超参 $h$。
3. **跨时间加快。** 另一条已经成功、格点相同的轨迹，其 $Q$ 已经偏高；截断处 $w_j$ 把这个高 $Q$ 借过来，不必等本轨迹自己把 TD 爬完。借的是当前 $\pi$ 的 $Q$，不是 $\pi_\beta$ 的 $G$。
4. **不把冻结 $z$ 当成 $s$，也不学 $a\mapsto z'$。** $z'$ 只回答「格是否还相同」。

**明确拒绝（对照方法，不是 ZAP）。**

- 在 $z$ 上做 Q-chunking：$Q(z,a_{\mathrm{chunk}})\leftarrow R^{(h)}+\gamma^h Q(z',a')$。
- 对 $(z,a)$ 对序列再套一遍 QC 定理（同一构造换原子）。
- in-batch softmax 平均 TD 目标或平均 $G$。
- 把 $\mathrm{cosine}(z,z_0)$ 或 VLM yes/no 当方法主体（`rlt_phase_gate` 只是工程过滤）。
- InfoNCE / 双线性头去改冻结 $\varphi$（那是 Stage-1 / eRLT 的事）。
- 核平均整段回报 $G$（无 TD 链，消不掉 off-policy MC 偏置）。

**没解决、也不声称。**

- QC 的定理：无偏多 action MDP $Q(s,a_{1:h})$。
- Stage 1 的 $z$ 质量。核是坏度量时 $c$ 和 $w_j$ 会传错奖励或错 $Q$——这是主风险。
- BEE 式介入椭球、eRLT 式路由。ZAP 吃现成 $z$。
- 多样 / 合适的技能 $z$（那是另一条表示问题，见 Stage-1 讨论）。

### 4.5 和前面方法怎么对齐

| | RLT / eRLT / BEE | T-SAC | QC / AQC / DQC | **ZAP** |
|---|---|---|---|---|
| 改的是什么 | 约束或 $z$ 来源 | $Q$ 网络形状 | $Q$ 的动作定义域 | **backup 路径 + 借值** |
| $Q$ 输入 | $(z,a)$ 或 $(z,a_{\mathrm{chunk}})$ | 序列 $[z,a_{0:T})$ | $(s\text{ 或 }z,\,a_{t:t+h})$ | 当前 $(z,a_t)$ |
| n-step | 无，或有偏 | 不负责 | 无偏，靠动作进 $Q$ | 门控续步，靠 $(z,a)$ 度量 |
| 沿时间 | 一步爬 | 不负责 | 每次跳固定 / 多尺度 $h$ | $C_k$ 大时跳 $\tau$ 步 |
| 跨时间 | 无 | 无 | 无 | 截断处 $w_j$ 借格点 $Q$ |
| off-policy 中间段 | 未处理 | 未处理 | 动作进 $Q$ 消除 | 偏离 $\pi$ 则切断 |
| $z'$ | 常当作 $s'$ | 状态 token | $s'$ 合法 / $z'$ 近似 | 只用于「格是否相同」 |

一句话：QC 家族靠「中间动作进 $Q$」做无偏 n-step；ZAP 靠「$z$ 是否还在格内、$a$ 是否还贴 $\pi$」决定 n-step 能走多远，停下来再向同类格点的当前 $Q$ 借值。**$Q$ 的形状不变；变的是 backup 沿着哪条路径走、向谁取值。**

---

## 5. 怎么读这条谱系

若问题是 **VLA 提案不能乱离、人在纠偏**：BEE。  
若问题是 **固定最终层 $z$ 丢掉了动作相关特征**：eRLT。  
若问题是 **chunk 内部谁先谁后**：T-SAC。  
若问题是 **要无偏多步、且接受改 $Q$ 定义域**：QC；地平要自适应则 AQC；价值地平要长、策略地平要短则 DQC。  
若问题是 **不能扩 $Q$、Retrace 又废、还要把稀疏成功沿本轨迹和跨轨迹传开**：ZAP。

ZAP 默认站在 RLT Stage-2 底座上：冻结 $\varphi$、residual actor、$Q(z,a)$、BC。eRLT 换 $\varphi$、BEE 换约束，都可以和 ZAP 的 backup **正交组合**；QC 家族与 ZAP **互斥**（同一套 $Q$ 不能既吃 $a_{1:h}$ 又声称只吃当前原子）。
