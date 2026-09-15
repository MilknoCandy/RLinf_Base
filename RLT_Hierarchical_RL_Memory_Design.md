# RLT + Hierarchical RL Memory：设计与实现方案

## 1. 核心思想

在 RLT（RL Token）基础上，引入两层记忆：

- **Short-term Memory (STM)**：短期记忆，保存当前/近期 trajectory 中的 reward-conditioned experience。
- **Long-term Memory (LTM)**：长期记忆，将具有复用价值的 trajectory 片段进一步抽象成 event / skill-level knowledge。

关键不是简单增加一个 Memory，而是：

> **让 Memory 本身成为 Online RL 的可学习决策组件，使 RL 同时优化 Policy、Memory 的写入、检索和长期 consolidation。**

整体闭环：

```text
                         ┌─────────────────────────────┐
                         │      Long-term Memory       │
                         │   Skill / Event / Procedure │
                         └──────────────▲──────────────┘
                                        │
                                  Consolidation
                                        │
                         ┌──────────────┴──────────────┐
                         │     Short-term Memory       │
                         │   Recent Trajectory         │
                         │   RL Token + Action + R     │
                         └──────────────▲──────────────┘
                                        │
                                        │ Write
Observation ──→ VLM ──→ RL Token z_t ──→ Memory Controller
                            │                    │
                            │                    ▼
                            │              Retrieval
                            │                    │
                            └────────────┬───────┘
                                         ▼
                                  Memory Fusion
                                         │
                                         ▼
                                 Enhanced RL Token
                                         │
                                         ▼
                                      Actor
                                         │
                                         ▼
                                      Action
                                         │
                                         ▼
                                    Environment
                                         │
                                         ▼
                                      Reward
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                        Policy Update         Memory Update
```

---

## 2. Short-term Memory：短期轨迹记忆

### 2.1 Memory Unit

不建议 STM 只保存 RL Token，而应保存 reward-conditioned experience：

\[
e_t=(z_t,a_t,r_t,A_t,z_{t+1})
\]

其中：

- \(z_t\)：RLT 的 RL Token
- \(a_t\)：执行动作
- \(r_t\)：环境 reward
- \(A_t\)：advantage
- \(z_{t+1}\)：下一状态 token

一个短期记忆窗口：

\[
S_t=\{e_{t-K+1},\ldots,e_t\}
\]

因此 STM 本质上是一个近期 trajectory buffer。

### 2.2 不使用简单 FIFO

可以加入可学习 Write Controller：

\[
p_t^{write}
=
\sigma(f_{\phi}^{write}(z_t,a_t,r_t,A_t))
\]

决定当前 experience 是否值得进入 STM。

建议同时保留：

\[
STM=STM^+\cup STM^-
\]

即：

- Success memory：保存高 reward / 高 advantage 的经验
- Failure memory：保存具有代表性的失败经验

失败经验也很重要，因为它可以告诉 Policy：

> 类似状态下，过去什么 action 没有效果。

### 2.3 STM Retrieval

当前 RL Token：

\[
z_t
\]

作为 query：

\[
q_t=f_{\phi}^{query}(z_t)
\]

Memory entry：

\[
k_i=f_{\phi}^{key}(e_i)
\]

计算：

\[
s_i=q_t^\top k_i
\]

然后：

\[
\alpha_i=
\operatorname{softmax}(s_i/\tau)
\]

得到：

\[
m_t^{STM}
=
\sum_i\alpha_i v_i
\]

其中：

\[
v_i=f_\phi^{value}(e_i)
\]

### 2.4 STM 如何影响 RLT

不建议 Memory 直接连接 Action Head。

而是重新增强 RL Token：

\[
z_t'
=
F_\phi(z_t,m_t^{STM},m_t^{LTM})
\]

然后：

\[
a_t\sim\pi_\theta(a|z_t')
\]

这样始终保持 RLT 的核心：**RL Token 是决策中间表示**。

---

# 3. Long-term Memory：长期技能记忆

LTM 不应该只是一个更大的数据库。

核心定义：

> **LTM stores temporally extended, semantically meaningful, reusable skills/events extracted from trajectories.**

例如 STM 保存：

```text
Step 1
Step 2
Step 3
Step 4
Step 5
...
```

而 LTM 抽象成：

```text
approach object
      ↓
align gripper
      ↓
grasp
      ↓
lift
      ↓
insert
```

因此：

\[
LTM=
\{\text{Event / Skill / Procedure}\}
\]

---

# 4. STM → LTM：Skill Consolidation

这是整个系统最值得研究的部分之一。

不要人工定义 skill，而应该从 trajectory 自动形成：

\[
STM
\xrightarrow{\text{Segmentation}}
Trajectory\ Segments
\xrightarrow{\text{Skill Encoder}}
Skill\ Tokens
\]

例如：

```text
t1 t2 t3 t4 | t5 t6 t7 | t8 t9 t10
      ↓            ↓           ↓
   approach       grasp       insert
```

每个 segment：

\[
e_j=
(z_{start},
z_{end},
a_{start:end},
R_{segment})
\]

再经过 Skill Encoder：

\[
s_j=
Enc_{\phi}^{skill}(e_j)
\]

得到：

\[
s_j\in\mathbb R^d
\]

---

# 5. Event / Skill Segmentation

可以利用三个信号：

### 5.1 Reward transition

例如：

\[
r_t:0,0,0,0,1,1
\]

reward 发生明显变化时可能对应关键事件。

### 5.2 Advantage transition

\[
A_t
\]

发生明显变化时可能对应行为阶段切换。

### 5.3 Representation transition

\[
\Delta z_t=\|z_t-z_{t-1}\|
\]

如果表示空间发生明显变化，也可能是事件边界。

因此可以定义：

\[
boundary_t=
f(\Delta z_t,\Delta A_t,\Delta r_t)
\]

---

# 6. LTM Memory Entry

一个长期 skill memory 可以设计成：

```text
Skill ID:
#023

Key:
Object + spatial configuration

Precondition:
Object aligned with gripper

Procedure:
approach → align → grasp → lift

Outcome:
SUCCESS

Return:
+8.7

Embedding:
s_023
```

形式化表示：

\[
M_j=
(k_j,
s_j,
P_j,
a_{j,1:T},
R_j,
Outcome_j)
\]

其中：

- \(k_j\)：retrieval key
- \(s_j\)：skill embedding
- \(P_j\)：precondition
- \(a_{j,1:T}\)：skill action sequence
- \(R_j\)：skill return
- \(Outcome_j\)：成功/失败

---

# 7. LTM Retrieval

当前状态：

\[
z_t
\]

查询：

\[
q_t=f_{\phi_q}(z_t)
\]

LTM：

\[
M^L=\{s_1,s_2,\ldots,s_N\}
\]

计算：

\[
\alpha_j=
\operatorname{softmax}(q_t^\top k_j)
\]

得到：

\[
m_t^{LTM}
=
\sum_j\alpha_j s_j
\]

因此：

- STM 回答：**“我刚刚做了什么？”**
- LTM 回答：**“以前遇到类似情况，我应该怎么做？”**

---

# 8. STM + LTM Fusion

建议先使用简单、稳定的 gated fusion：

\[
g_s=\sigma(W_s[z_t;m_s])
\]

\[
g_l=\sigma(W_l[z_t;m_l])
\]

最终：

\[
\boxed{
z_t'
=
z_t+
g_s\odot m_s+
g_l\odot m_l
}
\]

其中：

- \(g_s\)：控制短期经验的影响
- \(g_l\)：控制长期技能的影响

后续可以研究：

> 在什么状态下应该依赖 STM？什么时候应该依赖 LTM？

---

# 9. 让 Memory 本身接受 RL 优化

这是方法的核心。

整个系统：

\[
o_t
\rightarrow
z_t
\rightarrow
M
\rightarrow
z_t'
\rightarrow
\pi
\rightarrow
a_t
\rightarrow
r_t
\]

最终：

\[
J=
E[R]
\]

联合优化：

\[
\theta_{policy},
\theta_{STM},
\theta_{LTM}
\]

即：

\[
\boxed{
\text{Policy + STM + LTM}
}
\]

共同通过 Online RL 学习。

---

# 10. Policy RL Objective

例如使用 PPO：

\[
L_{policy}
=
-\mathbb E_t
[
\min(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
)
]
\]

其中：

\[
\rho_t=
\frac{
\pi_{\theta}(a_t|z_t')
}{
\pi_{\theta_{old}}(a_t|z_t')
}
\]

如果 Memory Fusion 是 policy computation graph 的一部分，则 Memory Controller 的参数也会受到 RL objective 的影响。

---

# 11. RL 优化 STM Retrieval

假设 Retriever：

\[
i_t\sim p_\phi(i|z_t)
\]

则可以使用：

\[
L_{STM}
=
-A_t\log p_\phi(i_t|z_t)
\]

含义：

- 如果 retrieval 带来了好的 action / trajectory：
  \[
  A_t>0
  \]
  强化这种 retrieval。
- 如果 retrieval 对决策产生负面影响：
  \[
  A_t<0
  \]
  抑制这种 retrieval。

因此 Retriever 会逐渐学会：

> **当前状态应该参考哪一类近期经验。**

---

# 12. RL 优化 LTM Retrieval

同理：

\[
j_t\sim p_\phi(j|z_t)
\]

可以定义：

\[
L_{LTM}
=
-A_t\log p_\phi(j_t|z_t)
\]

因此：

\[
\boxed{
\text{RL learns what skill to retrieve}
}
\]

---

# 13. RL 优化 Memory Write

对于：

\[
w_i\sim p_\phi(w|e_i)
\]

可以定义：

\[
L_{write}
=
-A_i\log p_\phi(w_i|e_i)
\]

这样 Memory 会学习：

> 哪些经验值得记住。

但实际实现时，不建议只根据即时 advantage 判断长期价值。

---

# 14. Long-term Memory 的长期 Credit Assignment

这是 LTM 比 STM 更有研究价值的地方。

例如：

```text
Episode 1
   ↓
发现一个有效 skill
   ↓
保存到 LTM
   ↓
Episode 2
   ↓
没有使用
   ↓
Episode 5
   ↓
再次遇到相似状态
   ↓
Retrieve skill
   ↓
成功
```

因此：

\[
Memory
\rightarrow
Future\ Action
\rightarrow
Future\ Reward
\]

而不是：

\[
Memory
\rightarrow
Immediate\ Reward
\]

可以为 Memory 定义 value：

\[
V_M(m_t)
=
E[R_{future}|m_t]
\]

甚至：

\[
Q_M(z_t,m_t)
=
E[R|z_t,m_t]
\]

于是 Memory Retrieval 可以理解成：

\[
m_t^*
=
\arg\max_m Q_M(z_t,m)
\]

这使 Memory 真正成为 Online RL 中的决策模块。

---

# 15. 两个 Policy

最终可以将整个系统理解为两个相互协作的 Policy。

## Action Policy

\[
\boxed{
\pi_A(a_t|z_t,m_t)
}
\]

回答：

> **做什么？**

## Memory Policy

\[
\boxed{
\pi_M(m_t|z_t)
}
\]

回答：

> **参考什么？**

进一步：

\[
\pi_M=
\{
\pi_{write},
\pi_{retrieve},
\pi_{consolidate}
\}
\]

即：

```text
Memory Policy

       ┌───────────────┐
       │               │
     WRITE          RETRIEVE
       │               │
       ↓               ↓
    记什么？          查什么？
       │               │
       └───────┬───────┘
               ↓
          CONSOLIDATE
               ↓
          长期保留什么？
```

---

# 16. 最终 Loss

一个初始版本可以写成：

\[
\boxed{
L=
L_{RL}
+
\lambda_sL_{STM}
+
\lambda_lL_{LTM}
+
\lambda_wL_{write}
}
\]

其中：

- \(L_{RL}\)：Actor / PPO 等 Online RL loss
- \(L_{STM}\)：短期 Memory retrieval loss
- \(L_{LTM}\)：长期 Memory retrieval loss
- \(L_{write}\)：Memory write / selection loss

后续可以进一步加入：

\[
L_{consolidation}
\]

用于约束 STM → LTM 的技能抽象质量。

---

# 17. 工程上需要实现的模块

建议拆成以下模块：

| 模块 | 功能 | 是否训练 |
|---|---|---|
| `RLTokenEncoder` | VLM → RL Token | RLT 原有 |
| `ShortTermMemory` | 保存短轨迹 | ✓ |
| `STMRetriever` | 检索近期经验 | ✓ |
| `TrajectorySegmenter` | trajectory → event | ✓ / 弱监督 |
| `SkillEncoder` | event → skill token | ✓ |
| `LongTermMemory` | 保存 skill | ✓ |
| `LMTRetriever` | 检索历史 skill | ✓ |
| `MemoryFusion` | STM + LTM + RL Token | ✓ |
| `MemoryController` | write / retrieve / consolidate / forget | ✓ |

---

# 18. Online RL 完整生命周期

一个 episode 可以按照以下流程运行：

```text
1. Observation
        ↓
2. VLM
        ↓
3. RL Token z_t
        ↓
4. STM/LTM Retrieval
        ↓
5. Memory Fusion
        ↓
6. Enhanced RL Token z'_t
        ↓
7. Actor
        ↓
8. Action
        ↓
9. Environment
        ↓
10. Reward
        ↓
11. Advantage
        ↓
12. Policy Update
        ↓
13. STM Update
        ↓
14. Episode End
        ↓
15. Trajectory Segmentation
        ↓
16. Skill Extraction
        ↓
17. LTM Consolidation
        ↓
18. Next Episode
```

下一 episode 同时利用：

\[
\boxed{
\text{Recent Experience (STM)}
+
\text{Historical Skills (LTM)}
}
\]

---

# 19. 建议的三阶段实现路线

不要一开始就同时训练所有组件。

## Phase 1：RLT + STM

实现：

\[
RLT+ShortTermMemory
\]

先验证：

> 最近 trajectory experience 是否提高 Online RL sample efficiency？

---

## Phase 2：STM → LTM

加入：

\[
Trajectory
\rightarrow
Segment
\rightarrow
Skill
\rightarrow
LTM
\]

先固定 LTM 的 write / retrieval 策略。

验证：

> Skill-level memory 是否比单纯 trajectory memory 更有效？

---

## Phase 3：Memory Online RL

最后训练：

\[
\boxed{
Write+Retrieve+Consolidate
}
\]

全部接受 RL 信号。

推荐最终 ablation：

```text
RLT
 ↓
RLT + STM
 ↓
RLT + STM + LTM
 ↓
RLT + RL-trained STM
 ↓
RLT + RL-trained STM + LTM
 ↓
RLT + RL-trained Hierarchical Memory
```

---

# 20. 最核心的研究假设

可以将方法概括为：

> **RLT converts multimodal observations into compact RL-token representations, while conventional Online RL mainly stores learned experience implicitly in policy parameters. We extend RLT with hierarchical memory: short-term memory preserves recent reward-conditioned trajectory experiences, while long-term memory consolidates reusable event- and skill-level knowledge. Crucially, memory formation, retrieval, and consolidation are optimized through environmental rewards, allowing memory to improve jointly with the policy.**

中文：

> **RLT 通过 RL Token 将当前多模态观测压缩为紧凑的决策表示，而传统 Online RL 主要通过更新 Policy 参数隐式保留经验。我们进一步引入层次化记忆：短期记忆保留近期 reward-conditioned trajectory experience，长期记忆则将具有复用价值的轨迹片段进一步抽象为 event/skill-level knowledge。更重要的是，记忆的形成、检索与 consolidation 均受到 Online RL 环境反馈的优化，使 Memory 不再是静态经验库，而成为能够与 Policy 共同学习的决策组件。**

最终可以将核心机制浓缩成：

\[
\boxed{
Trajectory
\overset{\text{RL}}{\longrightarrow}
Experience
\overset{\text{Consolidation}}{\longrightarrow}
Skill
\overset{\text{RL}}{\longrightarrow}
Reusable\ Knowledge
}
\]

这也是整个方法区别于普通 Memory-Augmented VLA / RAG 的核心。
