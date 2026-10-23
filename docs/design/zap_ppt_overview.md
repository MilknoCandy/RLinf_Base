# ZAP 整体设计（PPT 稿）

面向汇报 / 组会的幻灯片文稿。每节对应一页或一组页；公式可直接拷到幻灯片。更细的公式与落地清单见 [`rlt_za_gated_nstep.md`](rlt_za_gated_nstep.md)，谱系对照见 [`rlt_improvements_and_zap.md`](rlt_improvements_and_zap.md)。

**方法名：ZAP** = **(Z, A) Path**（$(z,a)$ 路径度量门控 n-step）

**一句话口号（封面可用）：**  
$Q$ 的形状不变；变的是 backup 沿着哪条路径走、向谁取值。

---

## 幻灯片结构总览

| 板块 | 建议页数 | 内容 |
|---|---|---|
| 封面 | 1 | 题目、一句话、机构/日期 |
| 研究背景 | 3–4 | VLA RL、稀疏奖励、n-step 困境 |
| 相关工作 | 3–4 | RLT / Retrace / QC 家族 / T-SAC |
| 我们的方法 ZAP | 5–7 | 思路、核、$c$/$C_k$、$w_j$、目标、对照表 |
| 算法图 | 1–2 | 按文末「绘制说明」画主图 |
| 总结与展望 | 1 | 贡献、不做的事、风险 |

---

# 第一部分：研究背景

## Slide 1｜封面

**标题：** ZAP: Metric-Gated Path Backup for Residual RL on Frozen VLAs

**副标题：** $(Z, A)$ Path —— 不扩 $Q$，用语义–动作度量决定 n-step 能走多远

**要点（可选）：**

- 底座：RL Token (RLT) Stage-2（冻结 VLA + residual actor）
- 问题：稀疏成功 + 近确定连续控制下，标准 n-step / Retrace 失效
- 贡献：路径续步许可 $c$ + 截断后跨轨迹借值 $w_j$

---

## Slide 2｜具身 RL 与 VLA：为什么还要 Online

**背景叙述（讲稿）：**

大规模 Vision-Language-Action（VLA）模型在演示上已经会「合理动作」，但真机 / 仿真里仍常卡在精细段：对齐、插入、合爪时机。直接对整网做 RL 微调贵，且容易冲掉语言先验。

**要点：**

1. VLA 提供强先验（语言条件、参考动作 chunk）
2. 缺的是：在有限交互里，把稀疏成功信号变成策略修正
3. 工程共识：冻结大模型，只训轻量 head / residual（RLT 路线）

**图意（可选小图）：**  
「大 VLA 冻结」→「小 residual」→「环境成功/失败」→「价值回传」

---

## Slide 3｜RLT 底座：我们站在什么上面

**两阶段（Xu et al., RL Token）：**

| 阶段 | 做什么 | Online 是否更新 |
|---|---|---|
| Stage 1 | demo 上训 $\varphi(o)\to z$，重建 VLM prefix | 训 encoder |
| Stage 2 | 冻结 $\varphi$ 与 VLA；训 residual $\pi$ 与 $Q$ | 只训小网络 |

**Stage 2 接口（每步）：**

$$
o^{\mathrm{rl}}=(z,\; x_{\mathrm{proprio}},\; a^{\mathrm{ref}}),\qquad
z=\varphi(o),\quad
a=\pi(\mathrm{ref},z,x)
$$

评论家：

$$
Q_\theta(z,a)
$$

Actor：最大化 $Q$ + 对 $a^{\mathrm{ref}}$ 的 BC。

**讲稿一句：** RLT 解决了「能不能便宜地 online」；**没有**解决「稀疏奖励如何又快又稳地回传」。

---

## Slide 4｜核心痛点：信用回传太慢

**现象：**

- 成功奖励多为 episode 末尾的稀疏 $0/1$
- 精细段往往发生在早期–中期；一步 / 单 chunk TD 要很多次更新才爬回起点
- VLA 按 **action chunk** 开环执行，但标准 RLT 的 $Q$ 并不为「整段中间动作」负责

**朴素对策与失败原因：**

| 对策 | 为什么不够 |
|---|---|
| 固定 n-step | 中间动作来自旧行为策略 $\pi_\beta$，off-policy 有偏 |
| Retrace / IS | RLT 近确定（`fixed_std≈0.002`），$\pi/\pi_\beta$ 退化成 0/1 |
| 把 $a_{t:t+h}$ 写进 $Q$（QC） | 有效，但是**改 $Q$ 定义域**；且默认 $z$ 可当马尔可夫 $s$ |

**本工作要回答的问题（高亮）：**

> 在 **不扩 $Q$ arity**、**不用失效密度比** 的前提下，如何做近似无偏 / 保守的多步 backup，并把信用沿本轨迹与跨轨迹传开？

---

## Slide 5｜问题形式化（可选技术页）

给定冻结嵌入 $z_t$ 与执行动作 $a_t$，目标是估 $Q^\pi(z_t,a_t)$。

- $z$ **不是**马尔可夫状态：$z'=\varphi(o')$ 不是 $a\mapsto z'$ 的动力学
- 允许的估计器形状仍是 **当前一对**：

$$
Q_\theta(z_t,a_t)\qquad\text{（不吃 }a_{t:t+h}\text{，不吃 }(z,a)\text{ 序列）}
$$

- 需要决定的是 **backup 路径**：从 $t$ 沿同一 episode 能累多少步 $r$，停在哪，bootstrap 取谁的 $Q$

---

# 第二部分：相关工作

## Slide 6｜相关工作总览（一张图）

建议画三条轴上的方法树：

```
稀疏奖励 / 多步信用
        │
        ├─ 密度比门控 ── Retrace / V-trace（离散或高熵策略友好）
        │
        ├─ 动作进 Q ─── Q-chunking / AQC / DQC
        │                 （一步 TD ≡ 无偏 n-step）
        │
        ├─ 改 Q 网络形 ─ T-SAC（序列头，不负责 n-step 合法性）
        │
        └─ 改 backup 路径 ─ ★ ZAP（本文）
              Q 形状不变；(z,a) 度量决定续步与借值
```

**讲稿：** 大家解决的是同一类问题；差别在「把中间段动作放进估计器」还是「放进路径开关」。

---

## Slide 7｜RLT：底座而非贡献

**它做什么**

- 冻结 VLA + $\varphi$；residual actor；BC 锚定 ref
- $z$ 作紧凑条件，不注回 VLA

**优点**

- 算力与稳定性：不 RL 微调海量权重
- 保留语言条件与参考 chunk

**缺口（我们接着打）**

1. 价值沿时间爬得慢  
2. 朴素 n-step / Retrace 在近确定连续动作上失效  
3. $z$ 常被当成 $s'$，但并不马尔可夫  

**与 ZAP 关系：** ZAP **复用** RLT Stage-2 接口；**贡献不在** Stage-1 表示，而在 Stage-2 的 backup。

---

## Slide 8｜多步 Off-policy：Retrace 与为什么失效

**Retrace 思想（简）：**

- 沿轨迹累回报，用 $\min(1,\pi/\pi_\beta)$ 一类系数连乘截断
- 系数小 → 少信这段 off-policy 中间段

**在 RLT 设定下：**

- $\pi$ 近确定、连续动作 → 密度比数值上接近 **0 或 1**
- 几乎变成「要么一步 TD，要么全信行为策略」——门控失效

**启示（给 ZAP）：**  
仍要「连乘截断」的 **形式**，但截断信号不能来自 $\pi/\pi_\beta$，而来自 **$(z,a)$ 是否仍可互换**。

---

## Slide 9｜Q-chunking 家族：动作进 $Q$

**核心构造（QC）：**

$$
Q(s,a_{t:t+h}),\qquad
y=R^{(h)}+\gamma^h Q(s',a'),\qquad
R^{(h)}=\sum_{k=0}^{h-1}\gamma^k r_{t+k}
$$

一步 TD **数学上等于** 这段 n-step —— 因为产生这段回报的中间动作 **在估计器里面**。

| 变体 | 一句话 |
|---|---|
| QC | 固定 $h$；Best-of-N 选 chunk |
| AQC | 多尺度 $Q^k$；折扣归一化优势选尺度 |
| DQC | 长 horizon $Q^h$ 蒸馏短 policy $Q^k$ |

**可借的问题：** 中间段变量不能待在估计器外面。  
**不能借的构造（本文约束）：** 把 $a_{t:t+h}$ 写进 $Q$；把冻结 $z$ 直接当马尔可夫 $s$。

---

## Slide 10｜T-SAC：改网络形状，不改 backup 合法性

- 因果 Transformer，$Q$ 在序列 $[z,a_0,\ldots]$ 每个动作位出值
- **优点：** 看见 chunk 内前缀依赖  
- **不做：** 何时允许续 n-step；如何消 off-policy 中间段；跨轨迹借值  

**定位：** 正交于 ZAP；解决的是「$Q$ 怎么看序列」，不是「backup 走哪条路」。

---

## Slide 11｜相关工作对照（汇总表，可整页）

| | RLT | Retrace | QC / AQC / DQC | T-SAC | **ZAP** |
|---|---|---|---|---|---|
| 改什么 | Stage-2 底座 | IS 门控 | $Q$ 动作定义域 | $Q$ 网络形状 | **backup 路径** |
| $Q$ 输入 | $(z,a)$ | 通常 $(s,a)$ | $(s,a_{1:h})$ | 序列 | **当前 $(z,a)$** |
| 多步无偏手段 | 无 | $\pi/\pi_\beta$ | 动作进 $Q$ | 不负责 | **$(z,a)$ 度量续步** |
| 跨轨迹借 $Q$ | 无 | 无 | 无 | 无 | **有（$w_j$）** |
| 近确定连续控制 | — | 易失效 | 可用 | — | **专为该设定** |

---

# 第三部分：我们的方法 —— ZAP

## Slide 12｜ZAP 一句话与设计原则

**定义：** ZAP = **(Z, A) Path**  
用当前冻结语义 $z$ 与动作 $a$，度量「这条 backup 路径还能不能往前走」；截断后再向 replay 里 **同类 $(z,\pi)$ 格点** 借当前策略的 $Q$。

**三条原则（务必上屏）：**

1. **$Q$ 形状不变** —— 仍是 $Q_\theta(z_t,a_t)$，不拉成 $a_{t:t+h}$  
2. **$(z,a)$ 只进路径开关** —— 进续步 $c$ 与借值 $w_j$，不当动力学 $z'=f(z,a)$  
3. **双加速** —— 沿时间（本轨迹 n-step）+ 跨时间（邻居格点 $Q$）

---

## Slide 13｜直觉：语义格 + 贴策略的路径

**画在幻灯片上的直觉（配合主算法图）：**

1. 从 $(z_t,a_t)$ 出发，沿同一 episode 向前看  
2. 若仍在 **同一 VLM 语义格**，且 buffer 动作仍 **贴着当前 $\pi$** → 把后面的 $r$ 累进目标  
3. 一旦格子裂开，或动作离开 $\pi$ → **立刻停**（避免 off-policy 中间段）  
4. 停在 $\tau$ 后，不借行为策略的整段 $G$，而向 replay 中「同格、动作像 $\pi$」的样本借 $Q_{\bar\theta}(z_j,\pi(z_j))$

**类比（可选）：**  
像 Retrace 的连乘截断，但把密度比换成 **度量球上的邻接**。

---

## Slide 14｜两个核：不另训分类器

$$
k_z(z,z')=\exp\bigl((\cos(z,z')-1)/\tau_z\bigr)
$$

$$
k_a(a,a')=\exp\bigl(-\|a-a'\|^2/\tau_a\bigr)
$$

- 值域 $(0,1]$；$\tau_z,\tau_a>0$ 为带宽  
- $k_z$：**语义是否还在同一格**（换任务 / 换阶段则下降）  
- $k_a$：**动作是否还贴当前策略**（离 $\pi$ 则下降）  

**超参风险一句：** $\tau$ 过大 ≈ 关掉门控，退回固定 n-step。

---

## Slide 15｜续步许可 $c$ 与连乘 $C_k$

从起点 $(z_t,a_t)$ 沿轨迹走到 $t+k+1$：

$$
c_{k+1}=k_z(z_t,z_{t+k+1})\cdot k_a\bigl(a^\pi_{t+k+1},\,a^{\mathrm{buf}}_{t+k+1}\bigr)
$$

$$
C_0=1,\qquad C_{k+1}=C_k\cdot c_{k+1}
$$

| 符号 | 含义 |
|---|---|
| $a^\pi$ | **当前** actor 在该 $z$ 上的动作（无梯度） |
| $a^{\mathrm{buf}}$ | 当时环境执行、存进 replay 的动作 |
| $C_k\approx 1$ | 继续累 $\gamma^k r_{t+k}$ |
| $C_k<\varepsilon$ | 截断（建议 $\varepsilon=10^{-2}$） |

**为何必须乘积：** 缺语义或缺策略贴合，续步都不合法。

**精细段行为：** $z$ 几乎不动 + residual 贴 ref → $c$ 连续≈1 → 一次更新跨很多步；一偏转 → 自动退回一步 TD。

---

## Slide 16｜截断后借值 $w_j$（跨轨迹）

$c$ / $C_k$：**这一条路径**能走多远。  
$w_j$：只在截断点算一次，**向谁借续值**。

$$
w_j\propto k_z(z_\tau,z_j)\cdot k_a\bigl(\pi(z_\tau),a_j\bigr),\qquad\sum_j w_j=1
$$

- 候选：对 $z_\tau$ 在 sample window 上 **top-$K$**（如 $K=32$），不做全 buffer softmax  
- 无邻居：退回单点 $Q_{\bar\theta}(z_\tau,\pi(z_\tau))$  
- **借的是当前 $\pi$ 的 $Q$，不是 $\pi_\beta$ 的 Monte Carlo $G$**

---

## Slide 17｜完整 backup 与损失

$$
y_t=\sum_{k=0}^{\tau-1}\gamma^k r_{t+k}+\gamma^\tau\sum_j w_j\,Q_{\bar\theta}(z_j,\pi(z_j))
$$

$$
\tau=\min\{k\ge 1:C_k<\varepsilon\text{ 或 episode 结束}\}
$$

$$
L_Q=\bigl(Q_\theta(z_t,a_t)-y_t\bigr)^2
$$

Actor（与 RLT 相同）：

$$
L_\pi=-q_w\,Q_\theta(z,\pi(\mathrm{ref},z,x))+b_w\|\pi-\mathrm{ref}\|^2
$$

**实现提示（一页脚注即可）：**

- 时间索引是 control / atom step；chunk 只是环境执行方式  
- 同一 chunk 内 $z$ 可复制 → $k_z=1$，续步主要靠 $k_a$  
- `loss_type: rlt_zap`；有效地平由 $C_k$ 决定，上限 `za_max_horizon`

---

## Slide 18｜信用如何变快（双通道）

**沿时间（本轨迹）**

- $C_k$ 一直大时：一次更新把 $r_{t:t+\tau-1}$ 折现到 $(z_t,a_t)$  
- 末尾成功 $\times\gamma^\tau$ 可直接打到精细段起点  
- $\tau$ =「还贴着 $\pi$ 的路径长度」，不是固定超参 $h$

**跨时间（跨轨迹）**

- 另一条已成功、格点相同的轨迹，$Q$ 已偏高  
- 截断处 $w_j$ 把高 $Q$ 借过来，不必等本轨迹自己爬完 TD

---

## Slide 19｜与 QC 的不对称构造（强调页）

| | Q-chunking | ZAP |
|---|---|---|
| 中间动作放哪 | **进 $Q$ 输入** | **只进续步 $c$** |
| $Q$ 定义域 | $\mathcal{A}\to\mathcal{A}^h$ | **不变** |
| 无偏手段 | 一步 TD ≡ n-step（定理） | 路径仍在 $(z,\pi)$ 格内时保守续步 |
| $z$ 角色 | 常当 $s$ / $s'$ | 只回答「格是否相同」 |
| 跨轨迹 | 无 | **$w_j$ 借格点 $Q$** |

**可借 / 不可借：**

- ✅ 可借：**问题** —— 产生 backup 的变量不能待在估计器外面  
- ❌ 不可借：**构造** —— 把中间动作写进 $Q$

---

## Slide 20｜解决了什么 / 没解决什么

**解决了**

1. 近确定连续控制下的多步 backup（度量代替 $\pi/\pi_\beta$，不扩 $Q$）  
2. 稀疏成功沿本轨迹一次回灌（$C_k$ 大时）  
3. 已成功格点帮未成功轨迹（$w_j$）  
4. 不把冻结 $z$ 当马尔可夫 $s$，不学 $a\mapsto z'$

**不声称**

- QC 的无偏多 action MDP 定理  
- Stage-1 $z$ 质量；核坏时偏置会回来（主风险）  
- 几何 / VLM 精度门控当科学贡献（`rlt_phase_gate` 仅工程，与 ZAP 正交）

---

## Slide 21｜总结

**ZAP：** 在 RLT Stage-2 上，用 $(z,a)$ 路径度量门控 n-step，并在截断处向同类格点借当前 $\pi$ 的 $Q$。

**相对 QC 家族的定位：** 同一「要快、要稳的多步信用」问题；**不同答案** —— 改路径，不改 $Q$ 的 arity。

**口号收束：**

> Q-chunking 靠「中间动作进 $Q$」做无偏 n-step。  
> ZAP 靠「$z$ 是否还在格内、$a$ 是否还贴 $\pi$」决定能走多远，停下来再向同类格点借 $Q$。

---

# 第四部分：算法图绘制说明

以下专供画 **方法主图 / 算法框图**。可直接交给画图同学或自己按层拼。

## 推荐主图：三栏时间线 + 右侧借值

### 整体构图（横版 16:9）

从左到右四块，用箭头串联：

```
[A. Rollout / Replay]  →  [B. Path Gate (c, C_k)]  →  [C. Truncation τ]  →  [D. Neighbor Bootstrap w_j]
```

标题条写：**ZAP: (Z, A) Path Metric-Gated Backup**  
副标题：**$Q_\theta(z,a)$ 不变 · 改的是路径与借值**

---

### 块 A｜数据从哪来（左，约占 20%）

**画：**

1. 上方小图标：冻结 VLA / $\varphi$ → 输出 $z$、$a^{\mathrm{ref}}$  
2. 下方小图标：residual $\pi$ → 输出执行动作 $a$  
3. 中间一条水平 **episode 时间轴**，若干圆点：

$$
(z_t,a_t,r_t)\;\rightarrow\;(z_{t+1},a_{t+1},r_{t+1})\;\rightarrow\;\cdots\;\rightarrow\;(z_{t+\tau},\ldots)
$$

**标注：**

- Replay 按 atom / control step 存，带 `episode_id`  
- 高亮起点圆点：$(z_t,a_t)$（当前 TD 更新点）  
- 末尾可画一个「Success?」旗标（稀疏 $r$）

**不要画进这块：** $Q$ 吃整段 $a_{t:t+h}$（那是 QC，需用红色禁止角标区分时可另开小对照气泡）

---

### 块 B｜路径门控（中左，约占 35%，主视觉）

**画一条沿时间轴向前的粗箭头「backup path」。**

每个前瞻步 $t+k+1$ 画一个 **闸门图标**，闸门上写两个因子相乘：

```
        k_z(z_t, z_{t+k+1})
 c = ─────────────────────────
        ×  k_a(a^π, a^buf)
```

闸门下方用颜色编码（建议）：

| 颜色 | 含义 |
|---|---|
| 绿 | $C_k\approx 1$，奖励 $\gamma^k r$ 流入 $y$ |
| 黄 | $C_k$ 下降，仍可走但衰减 |
| 红 | $C_k<\varepsilon$，**截断**，后面 $r$ 不再进入 |

**旁注两个小公式框：**

$$
C_0=1,\quad C_{k+1}=C_k\cdot c_{k+1}
$$

$$
y \leftarrow y + \gamma^k r_{t+k}\quad\text{仅当 }C_k\text{ 大}
$$

**视觉隐喻：**  
绿色实线路段 = 合法路径；红色虚线叉掉 = 不再沿行为策略中间段前进。

---

### 块 C｜截断点 $\tau$（中右，约占 15%）

**画：**

- 时间轴上一个醒目的竖切刀 / 「STOP」标记在 $\tau$  
- 公式条：

$$
\tau=\min\{k:C_k<\varepsilon\text{ 或 done}\}
$$

- 从 $\tau$ 引出一条箭头指向右侧块 D，箭头标注：$\gamma^\tau \times$（bootstrap）

**旁注：**  
此处 **不再** 累后续 $r$；改为取续值。

---

### 块 D｜跨轨迹借值（右，约占 30%）

**画：**

1. 一个云状「Replay / sample window」  
2. 中心点：$(z_\tau,\pi(z_\tau))$  
3. 周围 $K$ 个邻居点 $(z_j,a_j)$，连线粗细 ∝ $w_j$  

$$
w_j\propto k_z(z_\tau,z_j)\cdot k_a(\pi(z_\tau),a_j)
$$

4. 汇聚成：

$$
\bar Q_\tau=\sum_j w_j\,Q_{\bar\theta}(z_j,\pi(z_j))
$$

5. 最终公式框（可放图底通栏）：

$$
y_t=\underbrace{\sum_{k=0}^{\tau-1}\gamma^k r_{t+k}}_{\text{沿路径累计}}+\underbrace{\gamma^\tau\bar Q_\tau}_{\text{格点借值}}
$$

**关键标注（防误解）：**

- 借的是 **$Q$（当前 $\pi$）**，不是整段回报 $G$  
- top-$K$，不是全库 softmax  

---

### 底部通栏（可选）

左：**Critic** $L_Q=(Q_\theta(z_t,a_t)-y_t)^2$  
右：**Actor** $\max Q$ + BC to $a^{\mathrm{ref}}$  
中间小徽章：**$Q$ 输入始终只有当前 $(z,a)$**

---

## 推荐对照小图（可作插页或主图角落）

并排两个气泡：

| QC | ZAP |
|---|---|
| $Q(s,a_{t:t+h})$ 大括号包住整段动作 | $Q(z_t,a_t)$ 只包当前一步 |
| 箭头写「动作进估计器」 | 箭头写「动作进闸门 $c$」 |

用禁止符号或「≠」强调：**ZAP ≠ 在 $z$ 上做 Q-chunking**。

---

## 绘制用文字描述（可直接给设计师）

请画一张横版方法示意图，标题为 “ZAP: (Z, A) Path Metric-Gated Backup”。画面从左到右分为四段。第一段展示冻结 VLA 产出语义向量 $z$ 与参考动作，residual 策略产出执行动作 $a$，下方是一条带若干状态圆点的 episode 时间轴，高亮起点 $(z_t,a_t)$。第二段是主视觉：沿时间轴有一条 backup 路径，每一步经过一个由两个核相乘构成的闸门——语义核 $k_z$ 与动作核 $k_a$；闸门开启时把折扣奖励累进目标 $y$，并用连乘系数 $C_k$ 表示路径仍合法；当 $C_k$ 低于阈值时闸门变红截断。第三段在截断时刻 $\tau$ 画一个明显停止标记。第四段从 $\tau$ 指向 replay 中的邻居集合：以 $(z_\tau,\pi(z_\tau))$ 为中心，按同样的 $(z,a)$ 核加权平均邻居的目标网络 $Q$，得到 bootstrap；图底用一条公式把「路径上累计的奖励」与「截断处借来的 $Q$」加总为 $y_t$。全图反复强调评论家 $Q$ 始终只接收当前一对 $(z,a)$，中间动作只出现在闸门里而不进入 $Q$ 的输入；借值借的是当前策略的 $Q$ 而不是行为策略的蒙特卡洛回报。配色建议：合法路径绿色、截断红色、借值邻居蓝色连线。可在角落用小对照说明 Q-chunking 把动作序列写进 $Q$，而 ZAP 不这样做。

---

## 演讲顺序建议（约 12–15 分钟）

1. 背景：VLA 要 online，但信用回传慢（Slide 2–4）  
2. 相关：RLT 底座 → Retrace 失效 → QC 的「动作进 $Q$」（Slide 7–9）  
3. 方法：原则 → 核 → $c$/$C_k$ → $w_j$ → 公式（Slide 12–17）  
4. 主图走读一遍（按块 A→D）  
5. 对照与边界：解决了什么 / 不声称什么（Slide 19–20）  
6. 收束口号（Slide 21）

---

## 与仓库文档的索引

| 需要更细时读 | 文档 |
|---|---|
| 公式、实现顺序、配置键 | [`rlt_za_gated_nstep.md`](rlt_za_gated_nstep.md) |
| RLT→T-SAC/QC/AQC/DQC→ZAP 谱系 | [`rlt_improvements_and_zap.md`](rlt_improvements_and_zap.md) |
| 代码 | `rlinf/algorithms/zap/`，`rlinf/workers/actor/fsdp_zap_policy_worker.py` |
| 配置 | `examples/embodiment/config/*_rlt_stage2_zap.yaml` |
