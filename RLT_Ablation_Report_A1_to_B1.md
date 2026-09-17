# RLT Stage-2 扩展消融：从 A1 到 B1

## 0. 设计动机（先讲清楚在改什么）

### 0.1 原版 RLT Stage-2 在做什么（白话）

训练分两阶段：

1. **Stage-1**：在大模型（OpenPI / VLA）上训出 **RL Token 模块**，把模型内部的视觉（等）表征压成一个短向量。
2. **Stage-2（本文关注）**：把大模型**冻住**，只用一个很小的 Actor-Critic，在仿真里做在线 RL。

每一个控制步，Stage-2 大致做三件事：

1. 用冻结的 VLA + RLT 模块，从当前相机图像等观测里读出一个 **RL Token**（下文统一记为 \(z_t\)）。
2. 同一套冻结 VLA 再给出一段 **参考动作**（记为 \(a^{\mathrm{ref}}_t\)），以及机器人 **本体感觉**（记为 \(s_t\)）。
3. 小网络根据 \((z_t,\, s_t,\, a^{\mathrm{ref}}_t)\) 输出要执行的动作；Critic 估计价值并做 TD 更新。

另外还有 **临界相（critical phase）** 开关：环境给出布尔标志（代码里叫 `rlt_switch_flags`）。只有进入「难做的阶段」（例如 PegInsertion 的对准/进孔）才让小 Actor 接管；其它时候直接执行参考动作。这与原版 RLT「只在难相上做 RL」一致。

**动机一句话**：原版 Stage-2 里的 \(z_t\) 基本是**当前这一帧**压出来的；PegInsertion 的难相却很依赖「最近怎么动、有没有进展」。本系列消融都是在问——**怎样给 Stage-2 补上时序信息，同时仍然不微调大 VLA**。

### 0.2 两条技术路线

1. **显式记忆（A1 → A21 → A22 / A23）**：把过去的 \(z\)（或 \(z,a\)、回报）存进窗口/银行，再检索或拼接给策略/价值用。
2. **流式动态表征（B1）**：不存一长串历史，只维护一个固定大小的状态 \(h_t\)，每步用当前外观 token 更新它，并用「预测未来外观 token」的损失约束 \(h_t\)。

中间试过可学习短期记忆 **A2**，后为方便消融分解而回退。下文按实现顺序写机制和公式；**请先看完 0.3 符号表，再读后面的公式**。

### 0.3 符号表（全文通用，先定义再出现公式）

| 符号 | 是什么 | 典型取值 |
|------|--------|----------|
| \(t\) | 当前时间步（一次 action-chunk 决策） | 整数 |
| \(z_t\) | 当前 RL Token（Stage-2 的状态向量） | 长度 \(d_z=2048\) 的向量 |
| \(z^{\mathrm{app}}_t\) | B1 用的「外观」token：只用图像 token 做的 RLT 读出（`rlt_image_only`） | 同 \(d_z\) |
| \(z^{\mathrm{RL}}_t\) | B1 把动态状态融进外观 token 之后的结果，再送给 Actor/Critic | 同 \(d_z\) |
| \(a_t\) | 当前执行的动作 chunk（常展平成一条向量） | 长度 \(d_a\)，如 \(10\times 8=80\) |
| \(a^{\mathrm{ref}}_t\) | 冻结 VLA 给出的参考动作 | 与 \(a_t\) 同维 |
| \(s_t\) | 本体感觉 / proprio | 低维向量 |
| \(r_t\) | 该步标量奖励（若原始按 chunk 展开，会先 collapse） | 标量 |
| \(R_i\) | 从第 \(i\) 步算到 episode 结束的蒙特卡洛 return-to-go | 标量 |
| \(c_t\) | 是否临界相：1=是，0=否（即 `rlt_switch_flags`） | 0 或 1 |
| \(\pi_\theta\) | Stage-2 小 Actor | 参数 \(\theta\) |
| \(Q_\theta\), \(Q_{\bar\theta}\) | Critic / 目标网络 Critic | — |
| \(\gamma\) | 折扣因子 | 默认 0.99 |
| \(\tau\) | 检索温度 | 默认 0.07 |
| \(K\) | Top-K 检索条数 | 默认 16 |
| \(h_t\) | B1 的流式动态状态（固定大小，**不是**历史列表） | 默认 \(16\times 256\) |

**原版 Actor 损失**（在符号已定义之后）：一边用 Critic 拉高价值，一边用行为克隆贴住参考动作：

\[
L_{\mathrm{actor}}
=
- q_{\mathrm{weight}}\, Q_\theta(z_t, s_t, \pi_\theta)
+ bc_{\mathrm{weight}}\,
\big\| \pi_\theta(z_t, s_t, a^{\mathrm{ref}}_t) - a^{\mathrm{ref}}_t \big\|
\]

Critic 仍是 RLT-AC 风格的 bootstrap；A23 会改其中的价值混合项，见第 5 章。

---

## 1. A1：规则 Short-Term Memory（不可学习 Verifier）

### 1.1 要验证什么

显式短期记忆、**固定规则写库 + 余弦检索 + 固定门控残差**，能否提升 PegInsertion 样本效率。A1 **不引入可训参数**，以便把效果归因于「有没有 STM」，而不是「STM 网络学得好不好」。

### 1.2 数据结构

- 每个 env 维护长度为 \(W\) 的滑动窗（默认 `window_size=64`），缓存本 episode 的 \(z\)。  
- 两个跨 episode 环形库：成功库 \(\mathcal{M}^{+}\)、失败库 \(\mathcal{M}^{-}\)（容量默认各 2048），条目为 \(z\in\mathbb{R}^{d_z}\)。

### 1.3 写入（规则）

每步先把当前 \(z_t\) 记入 pending；在得到 env 反馈后：

1. 若 `only_critical=True`，仅当 \(c_t=1\) 时把 \(z_t\) 写入该 env 的 episode 窗。  
2. Episode 结束（done）时 consolidate：  
   - 成功 → 窗内（或规则选出的）\(z\) 写入 \(\mathcal{M}^{+}\)；  
   - 失败 → 常取失败尾段 `fail_tail_steps` 写入 \(\mathcal{M}^{-}\)。

成功标志来自 `success`，否则用 \(\sum r > 0\)。

### 1.4 检索与增强（完整公式）

检索记忆集合为当前各 env episode 窗与 \(\mathcal{M}^{+}\cup\mathcal{M}^{-}\) 的并集，记为 \(\{m_j\}_{j=1}^{N}\)。

对 batch 中查询 \(z_t\)：

\[
q_t = \frac{z_t}{\|z_t\|_2},\qquad
k_j = \frac{m_j}{\|m_j\|_2}
\]

\[
s_{t,j} = \frac{q_t^\top k_j}{\tau}
\]

取 Top-\(K\) 得分 \(\{s_{t,(i)}\}_{i=1}^{K}\) 及对应索引，再：

\[
\alpha_{t,i}
=
\frac{\exp(s_{t,(i)})}{\sum_{i'=1}^{K}\exp(s_{t,(i')})}
\]

\[
\tilde{m}_t
=
\sum_{i=1}^{K} \alpha_{t,i}\, k_{(i)}
\]

固定门控 \(g\in\mathbb{R}\)（配置 `gate`，默认 \(0.5\)）：

\[
z'_t = z_t + g\,\tilde{m}_t
\]

随后 Actor/Critic 使用 \(z'_t\) 代替 \(z_t\)。

### 1.5 实现位置（历史）

- 模块：`rlinf/algorithms/rlt/a1_stm.py`（commit `f11845f0`）  
- 接线：rollout 在 policy 前调用 `enhance`  
- 后被 `d9cac711` 删除（由 A21/A22 路线取代）

---

## 2. A2：可学习 Short-Term Memory（中间尝试，已回退）

### 2.1 要验证什么

相对 A1，让 **检索与门控可学习**，并由 Stage-2 RL 目标反传，检验「学着用记忆」是否优于规则残差。

### 2.2 与 A1 的差异

- 银行条目改为 \((z,a)\)。  
- Encoder 挂在 policy 上：可学习 \(W_q,W_k,W_v\) 与 gate 网络。  
- Replay 存原始 \(z\)；actor 更新时 **重算** \(z'\)，使梯度进入 STM 参数。  
- Bank 本身不是模型参数（不做 weight sync）。

### 2.3 增强公式

\[
q_t = \mathrm{normalize}(W_q z_t),\qquad
k_j = \mathrm{normalize}(W_k z_j)
\]

\[
s_{t,j}=\frac{q_t^\top k_j}{\tau}
\]

Top-\(K\) 后：

\[
v_i = W_v\,[z_{(i)};\,a_{(i)}],\qquad
\tilde{m}_t=\sum_{i=1}^{K}\alpha_{t,i}\,v_i
\]

\[
g_t=\sigma\big(\mathrm{MLP}_g([z_t;\tilde{m}_t])\big)\in\mathbb{R}^{d_z}
\]

\[
z'_t = z_t + g_t \odot \tilde{m}_t
\]

`gate` 末层 bias 初始化为约 \(-2\)（\(\sigma(-2)\approx 0.12\)），起步保守。

### 2.4 状态

引入于 `9cd6af10`，由 `d8859c22` revert。之后主线改为可分解的 A21 / A22 / A23。

---

## 3. A21：本 Episode 内 \((z,a)\) 上下文

### 3.1 要验证什么

在引入跨 episode 银行之前，先问：

> **仅把当前 episode 最近 \(L\) 步的 \((z,a)\) 拼进 Actor/Critic，是否已足够补时序？**

无跨 episode 检索、无额外 memory loss，变量最少。

### 3.2 数据结构

对每个 env 维护长度 \(L=\texttt{ctx\_len}\)（默认 8）的环形缓冲：

\[
\big(z_{t-L+1:t},\; a_{t-L+1:t},\; \mathrm{mask}_{t-L+1:t}\big)
\]

Episode 开始时全零；有效步 `mask=1`。done 后清空。

### 3.3 特征构造（完整公式）

记 \(\mathrm{ctx\_z}\in\mathbb{R}^{B\times L\times d_z}\)，\(\mathrm{ctx\_a}\in\mathbb{R}^{B\times L\times d_a}\)，\(\mathrm{mask}\in\mathbb{R}^{B\times L}\)。展平：

\[
u_t^{z}
=
\mathrm{vec}\big(\mathrm{ctx\_z}_t \odot \mathrm{mask}_t\big)
\in\mathbb{R}^{L d_z}
\]

\[
u_t^{a}
=
\mathrm{vec}\big(\mathrm{ctx\_a}_t \odot \mathrm{mask}_t\big)
\in\mathbb{R}^{L d_a}
\]

\[
u_t = [u_t^{z};\, u_t^{a}]
\]

Stage-2 骨干输入在原 \((a^{\mathrm{ref}}_t,\, z_t,\, s_t)\) 基础上再拼 \(u_t\)（实现见历史 `a21_context.py`）。

### 3.4 状态

与 A22 一同引入（`5b93e292`）；在 A23 落地时为减交互而删除（`69c3f572`）。复现需检出历史 commit。

---

## 4. A22（一段话）

A22 在 episode 结束后，把临界相与收尾窗口里的经验写入跨 episode 的成功/失败记忆库，每条带上用整段奖励回算的蒙特卡洛 return-to-go（写库不用 TD，避免泄漏）；训练 Actor 时按当前 RL Token 相似度检索若干条，用「记忆回报减去当前 Critic（stop-grad）」当优势，只对优势为正的记忆动作做加权行为约束，加到原版 Actor 损失上。配置见 `algorithm.a22_memory`，代码 `rlinf/algorithms/rlt/a22_memory.py`。

---

## 5. A23（一段话）

A23 写库方式与 A22 同类（episode 末填 return-to-go，成功/失败库），但不改 Actor：检索改为「当前 (token, 动作)」对历史 (token, 动作) 的相似度，把检索到的回报聚成记忆价值 Q_M，再按权重 alpha 混进 Critic 的 bootstrap 目标，让价值估计吸收跨 episode 证据。配置 `maniskill_rlt_stage2_a23.yaml` / `algorithm.a23_memory`，代码 `rlinf/algorithms/rlt/a23_memory.py`。

---

## 6. B1（一段话）

B1 不存历史列表，只维护固定大小的递归状态 h：每步用当前图像-only 的外观 token 更新 h，再经门控融合成送给 Actor/Critic 的 RL Token；默认仅在临界相更新与注入。训练时存上一步 h 与当前外观 token，重算 h 后附加多步未来外观预测损失（如超前 1/4/8 步），与原版 RL 损失加权相加。配置 `maniskill_rlt_stage2_b1_dynamic.yaml`，代码 `rlinf/algorithms/rlt/b1_dynamic.py`。

---

## 7. 系列关系（实现视角）

```text
基线 RLT
  └─ z_t 仅当前帧

A1  规则：z' = z + g * Attn_cos(z, M)          （改表征，无可训 STM）
A2  可学：z' = z + g_θ ⊙ Attn_θ(z, M_(z,a))   （改表征，可训；已回退）
A21 拼接：输入 ← […; vec(z_{t-L:t}, a_{t-L:t})] （改输入，无银行）
A22 银行：L ← L_RLT + λ_m L_M(R_MC, π)         （改 Actor）
A23 银行：y ← … γ^H [(1-α)Q_bar + α Q_M]       （改 Critic）
B1  递归：h_t=U(h_{t-1},z_app); z_RL=Fusion; + λ_pred L_pred
```

| 变体 | 时序载体 | 额外损失 / 目标改写 | 默认 critical 限制 |
|------|----------|---------------------|-------------------|
| A1 | 显式 \(z\) 库 + 窗 | 无（只改 \(z\to z'\)） | 写库 `only_critical` |
| A2 | 显式 \((z,a)\) 库 | 经 \(z'\) 吃 RL 梯度 | 写库 `only_critical` |
| A21 | 本 ep 窗 | 无 | 无（全程填窗） |
| A22 | 跨 ep 库 + \(R_i\) | \(+\lambda_m L_M\) | 写库 E1 用 critical |
| A23 | 跨 ep 库 + \(R_i\) | bootstrap 混 \(Q_M\) | 同左 |
| B1 | \(h_t\) | \(+\lambda_{\mathrm{pred}} L_{\mathrm{pred}}\) | 更新/注入 `only_critical` |

---

## 8. 代码与提交索引

| 变体 | 关键提交 | 现存路径 |
|------|----------|----------|
| A1 | `f11845f0` | 已删；见该 commit |
| A2 | `9cd6af10` / revert `d8859c22` | 已删 |
| A21 | `5b93e292`；删于 `69c3f572` | 已删 |
| A22 | `5b93e292`, `06b399c3` | `rlinf/algorithms/rlt/a22_memory.py` |
| A23 | `69c3f572` | `rlinf/algorithms/rlt/a23_memory.py` |
| B1 | `339561de`, `510ef9da`, `f99716f9` | `rlinf/algorithms/rlt/b1_dynamic.py` |

相关设计文档：`RLT_Hierarchical_RL_Memory_Design.md`（记忆总设想）、`RLT_Dynamic_说明文档.md`（B1 / Dynamic）、`RLT_STAGE2.md`（基线 Stage-2）。

> **命名注意**：`RLT_Ablation1.md` 中的 A0/A1/B0/B1 指「特征来源 × 是否走 RLT token」矩阵，与本文 A1–B1 **不是同一套编号**。
