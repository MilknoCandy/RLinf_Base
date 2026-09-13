# RLT + Hierarchical Memory：Online RL 中的 Memory 学习方案

## 1. 研究目标

本项目希望在 RLT（Robot Learning Transformer / RL Token）框架上加入 Hierarchical Memory，使 Memory 不只是一个固定的数据结构，而是逐步被 Online RL 优化，并反过来帮助 Policy 学习。

完整目标可以表示为：

\[
o_t \rightarrow \text{VLM/RLT} \rightarrow z_t
\rightarrow \text{Memory Retrieval}
\rightarrow m_t
\rightarrow z'_t
\rightarrow \text{Actor}
\rightarrow a_t
\rightarrow \text{Env}
\rightarrow r_t
\]

最终希望形成两个相互优化的模块：

- **Action Policy**
  \[
  \pi_A(a_t|z_t,m_t)
  \]
- **Memory Policy**
  \[
  \pi_M=
  \{\pi_{write},\pi_{retrieve},\pi_{consolidate}\}
  \]

其中：

- \(\pi_{write}\)：决定什么经验值得写入 Memory；
- \(\pi_{retrieve}\)：决定当前状态应该检索什么 Memory；
- \(\pi_{consolidate}\)：决定哪些短期经验应该被总结为长期 Skill/Event。

---

# 2. 总体研究路线

当前建议采用逐步消融，而不是一开始直接实现完整的 RL-trained Hierarchical Memory。

```text
Step 0 / Step 1
RLT Baseline
      │
      ▼
Step 2A
Oracle / Fixed Short-term History
      │
      ▼
Step 2B
Similarity-based STM Retrieval
      │
      ▼
Step 3
STM → LTM Skill Consolidation
      │
      ▼
Step 4
RLT + STM + LTM
      │
      ▼
Step 5
RL-trained Memory Retrieval
      │
      ▼
Step 6
RL-trained Memory Write
      │
      ▼
Step 7
RL-trained Consolidation
      │
      ▼
完整 Hierarchical RL Memory
```

目前 **RLT baseline 已经完成**，因此下一步重点是 **Step 2**。

---

# 3. 一个关键问题：为什么短期经验可能有用？

这个问题非常重要：

> RLT 本身已经通过 Online RL 学习成功轨迹，为什么还需要把短期经验重新给 Actor？

答案不是“成功轨迹本身包含更多信息”，而是：

> **Policy 参数和 Episodic Memory 对经验的表示方式不同。**

## 3.1 Policy 是经验的参数化压缩

RLT Online RL 大致执行：

\[
\theta
\leftarrow
\theta+
\eta
\nabla_\theta
\log\pi_\theta(a_t|z_t)A_t
\]

过去的经验最终被压缩进 Policy 参数 \(\theta\)。

因此 Policy 更像是在学习：

> “遇到类似状态时，一般应该采取什么动作。”

---

## 3.2 Episodic Memory 保存具体经验

Memory 则显式保存：

\[
(z_1,a_1,r_1),
(z_2,a_2,r_2),
\ldots
\]

因此 Memory 更像是在告诉 Policy：

> “我之前在一个具体的、类似的状态下做过什么，以及结果怎么样。”

两者并不完全等价：

| Policy | Episodic Memory |
|---|---|
| 参数化知识 | 显式经验 |
| 长期压缩 | 短期/事件级保存 |
| 更新慢 | 可以立即访问 |
| General knowledge | Specific experience |
| 容易遗忘具体轨迹 | 可以保留具体事件 |

因此 Memory 的核心价值之一是：

\[
\text{slow parametric learning}
\quad+\quad
\text{fast episodic access}
\]

---

# 4. Memory 真正应该解决的问题：History Dependency

如果当前 observation 完全决定最优动作：

\[
a_t^*=f(o_t)
\]

那么 STM 很可能没有必要。

但是如果任务具有时间依赖：

\[
a_t^*=f(o_t,H_t)
\]

其中：

\[
H_t=(o_{t-K},a_{t-K},r_{t-K},\ldots,o_{t-1},a_{t-1},r_{t-1})
\]

那么仅仅使用当前 RLT token：

\[
z_t=f(o_t)
\]

可能丢失历史信息。

此时：

\[
\pi(a_t|z_t,H_t)
>
\pi(a_t|z_t)
\]

就有可能成立。

---

# 5. 因此 Step 2 的真正科学问题

Step 2 不应该直接假设：

> “Memory 一定有效。”

而应该验证：

> **Does explicit recent experience contain information that is not fully represented by RLT's current token?**

可以形式化为：

\[
I(H_t;A_t|z_t)>0
\]

如果：

\[
I(H_t;A_t|z_t)\approx0
\]

说明当前 RLT token 已经包含了足够的信息，STM 可能没有必要。

如果：

\[
I(H_t;A_t|z_t)>0
\]

则说明显式历史经验包含当前 token 没有编码的信息，Memory 有存在价值。

---

# 6. Step 2A：Fixed Short-term Memory

第一步不要直接做复杂 Retriever。

先实现一个最简单的：

> **Fixed Recent History / Oracle STM**

目的只有一个：

> 验证“显式短期历史”本身是否有用。

---

## 6.1 不修改 RLT representation

建议第一版严格遵循 RLT 的 representation-policy 分离思路：

- Frozen VLM / RLT representation
- Frozen RL token encoder
- 只新增 Memory Encoder
- Actor/Critic 使用 Memory 后的 token

即：

\[
o_t
\xrightarrow{\text{Frozen RLT}}
z_t
\]

然后：

\[
z_t
\xrightarrow{\text{STM}}
m_t
\]

最后：

\[
(z_t,m_t)
\xrightarrow{\text{Actor/Critic}}
a_t,V_t
\]

这样可以避免：

> Memory + Representation + RL 同时变化

导致实验无法解释。

---

# 7. Step 2A 的 STM 数据结构

第一版建议每个 Memory Entry 保存：

\[
e_t=(z_t,a_t,r_t)
\]

即：

```text
MemoryEntry:
    token: z_t
    action: a_t
    reward: r_t
```

而不是只保存：

\[
z_t
\]

原因是：

```text
z_t
→ 当前看到了什么

a_t
→ 当时做了什么

r_t
→ 结果怎么样
```

这实际上形成：

> **reward-conditioned experience**

也就是：

\[
\text{State}
\rightarrow
\text{Action}
\rightarrow
\text{Outcome}
\]

---

## 7.1 第一版不要保存 Advantage

不建议第一版直接存：

\[
(z_t,a_t,r_t,A_t)
\]

因为 \(A_t\) 通常需要 rollout 完成后才能计算，会让在线数据流更加复杂。

第一版使用：

\[
(z_t,a_t,r_t)
\]

即可。

未来做 RL-trained Memory 时，再考虑加入：

- advantage
- return
- TD-error
- success/failure label

等信号。

---

# 8. Step 2A：Fixed Window

设置一个固定历史长度：

\[
K=4/8/16
\]

例如：

\[
H_t=
\{e_{t-8},...,e_{t-1}\}
\]

严格规定：

\[
M_t
\subseteq
\{e_0,\ldots,e_{t-1}\}
\]

即：

> 当前 timestep 不能读取当前 action 之后的信息。

必须避免：

- future action
- future reward
- future success
- future trajectory

造成 **future leakage**。

---

# 9. STM Encoder

第一版推荐使用 GRU：

\[
h_i=
GRU(e_i,h_{i-1})
\]

最终：

\[
m_t=h_{t-1}
\]

即：

```text
e(t-K)
   ↓
 GRU
   ↓
 h1
   ↓
 GRU
   ↓
 h2
   ↓
 ...
   ↓
 GRU
   ↓
 m_t
```

相比直接上 Transformer，GRU 第一版更容易解释：

- 输入是短期 trajectory
- 输出是一个 history embedding
- 参数量小
- 对 Online RL 比较友好

---

# 10. 一个更简单的 Sanity Check

在 GRU 之前，可以先做一个更简单的 baseline：

\[
m_t=
\frac{1}{K}
\sum_{i=t-K}^{t-1}
\phi(e_i)
\]

也就是：

> Mean Pooling History

如果连简单的 history pooling 都完全没有收益，那么没有必要马上引入复杂 Memory Encoder。

---

# 11. Memory 与 RLT Token 的融合

推荐第一版使用 Residual Fusion：

\[
z'_t=z_t+W_m m_t
\]

然后：

\[
\pi(a_t|z'_t)
\]

优点是：

当：

\[
W_m\rightarrow0
\]

系统自然退化成：

\[
z'_t\approx z_t
\]

即原始 RLT。

因此 Memory 不会强迫 Policy 一定使用历史。

---

# 12. 更进一步：Gated Fusion

后续可以使用：

\[
z'_t=
z_t+
g_t\odot W_m m_t
\]

其中：

\[
g_t=\sigma(W_g[z_t,m_t])
\]

这样模型可以自己决定：

> “当前这个状态到底需不需要 Memory？”

例如：

```text
当前状态很明确
      ↓
Memory 不重要
      ↓
g ≈ 0

当前状态存在歧义
      ↓
Memory 很重要
      ↓
g ≈ 1
```

不过第一版建议先用 Residual，不要增加太多自由度。

---

# 13. Actor 和 Critic 是否都应该使用 Memory？

建议：

> **Actor 和 Critic 都使用 Memory。**

即：

\[
\pi(a_t|z_t,m_t)
\]

以及：

\[
V(z_t,m_t)
\]

而不是：

```text
Actor 看到 Memory
Critic 不看到 Memory
```

原因是如果 Actor 的决策依赖历史，而 Critic 完全不知道历史，那么 Value estimation 会变得更困难。

因此：

```text
               z_t
                │
                ▼
              STM
                │
                ▼
               m_t
                │
        ┌───────┴───────┐
        ▼               ▼
      Actor           Critic
    (z_t,m_t)        (z_t,m_t)
        │               │
        ▼               ▼
      action           value
```

---

# 14. Step 2A 完整数据流

```text
                  Observation
                       │
                       ▼
                ┌─────────────┐
                │ Frozen RLT  │
                └─────────────┘
                       │
                       ▼
                      z_t
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
      STM Write                 STM Read
          │                         │
      (z,a,r)                H(t-K:t-1)
                                    │
                                    ▼
                              Memory Encoder
                                    │
                                    ▼
                                   m_t
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                       Actor                Critic
                     (z_t,m_t)            (z_t,m_t)
                         │                     │
                         ▼                     ▼
                       action                value
                         │
                         ▼
                      Environment
                         │
                    ┌────┴────┐
                    ▼         ▼
                  reward    next obs
                    │
                    ▼
                 STM Write
```

---

# 15. Step 2A 的关键点：不是 Behavior Cloning

非常重要：

> 不要把历史成功 action 直接当成当前 action 的 label。

错误的思路是：

\[
H_t\rightarrow a_{t-1}\rightarrow a_t
\]

这很容易变成 imitation learning / behavior cloning。

我们真正要做的是：

\[
H_t
\rightarrow
m_t
\]

然后：

\[
\pi(a_t|z_t,m_t)
\]

Memory 是：

> **context**

而不是：

> **action label**

---

# 16. Step 2A 的训练方式

第一版建议：

- Environment Online interaction
- RLT representation frozen
- STM fixed write
- STM fixed recent window
- Memory Encoder learnable
- Actor learnable
- Critic learnable
- 使用原来的 Online RL objective

因此 Memory Encoder 本身通过 RL gradient 获得学习：

\[
\nabla_\phi J
\]

其中：

\[
m_t=f_\phi(H_t)
\]

最终：

\[
\pi_\theta(a_t|z_t,f_\phi(H_t))
\]

这已经实现了：

> **Memory helps Policy**

但还没有实现：

> **RL learns what to write / retrieve**

这是后面的 Step 5–7。

---

# 17. Step 2A 实验矩阵

建议至少做以下实验：

| 实验 | Current Token | History | Retrieval | 目的 |
|---|---:|---:|---|---|
| A | ✓ | ✗ | ✗ | RLT baseline |
| B | ✓ | Recent | ✗ | STM sanity check |
| C | ✓ | Random | ✗ | 排除单纯增加参数 |
| D | ✓ | Shuffled | ✗ | 验证时间顺序 |
| E | ✓ | `(z,a,r)` | ✗ | 验证 reward-conditioned experience |
| F | ✓ | Full STM | Similarity | 验证相关检索 |
| G | ✓ | Full STM | RL | 后续 RL retrieval |

---

# 18. 每个实验分别回答什么？

## A vs B

\[
RLT+STM
\quad vs \quad
RLT
\]

回答：

> 短期历史是否有价值？

---

## B vs C

如果：

\[
B>C
\]

说明提升不是简单因为增加参数或额外网络。

---

## B vs D

如果：

\[
B>D
\]

说明：

> temporal order 本身有信息。

否则可能只是：

> 增加了一些 context。

---

## B vs E

比较：

```text
(z history)
```

和：

```text
(z, action, reward history)
```

如果 E 更好，说明：

> reward-conditioned experience 比单纯 state history 更有价值。

---

## E vs F

比较：

```text
Recent History
```

和：

```text
Relevant History Retrieval
```

回答：

> Memory 是否需要真正的 retrieval？

---

## F vs G

最终回答：

> 人工设计的 retrieval 是否可以进一步被 RL 学习？

---

# 19. Step 2B：Similarity-based STM Retrieval

Step 2A 成功之后，再进入：

> **Learnable representation + similarity retrieval**

这里仍然不要马上用 RL 学 Retriever。

先做一个 deterministic retriever。

---

## 19.1 Query

当前 token：

\[
q_t=f_q(z_t)
\]

---

## 19.2 Memory Key

每条 Memory：

\[
k_i=f_k(e_i)
\]

---

## 19.3 Similarity

例如：

\[
score_i=q_t^\top k_i
\]

或者：

\[
score_i=
\frac{q_t^\top k_i}
{\|q_t\|\|k_i\|}
\]

即 cosine similarity。

---

## 19.4 Top-K

\[
I_t=
TopK_i(score_i)
\]

然后：

\[
M_t=
\{e_i|i\in I_t\}
\]

再送入 Memory Encoder。

---

# 20. 为什么 Step 2B 不直接做 RL Retrieval？

因为如果直接做：

\[
\pi_M(i|z_t)
\]

同时训练：

- Actor
- Critic
- Memory Encoder
- Retriever

那么最终即使性能变化，也很难知道：

> 到底是 Memory 有效，还是 Retriever 的随机性造成的？

所以建议严格拆开：

```text
Step 2A
证明 Memory 有用

        ↓

Step 2B
证明 Relevant Memory 比 Recent Memory 更好

        ↓

Step 5
证明 RL 可以学会更好的 Retrieval
```

这样论文中的因果链条会非常清楚。

---

# 21. Step 5：RL-trained Memory Retrieval

当 Step 2B 成功以后，再让 Memory Retrieval 本身成为一个 RL policy：

\[
i_t
\sim
\pi_{M_s}(i|z_t)
\]

其中 \(i_t\) 表示：

> 当前应该读取哪个 Memory。

可以定义：

\[
L_{M_s}
=
-A_t
\log
\pi_{M_s}(i_t|z_t)
\]

让高 advantage 的 trajectory 对应的 Memory selection 获得更高概率。

最终形成：

\[
\pi_A(a_t|z_t,m_t)
\]

和：

\[
\pi_{M_s}(i_t|z_t)
\]

联合优化。

---

# 22. Step 6：RL-trained Memory Write

下一步才学习：

> 什么经验值得写入 STM？

定义：

\[
w_t
\sim
\pi_{write}(w|e_t)
\]

其中：

\[
w_t\in\{0,1\}
\]

或者：

\[
w_t\in\{discard,store\}
\]

可以使用：

- reward
- advantage
- TD error
- novelty
- surprise
- future utility

作为输入特征。

最终：

\[
\pi_{write}
\]

学习：

> 什么经验值得保留？

---

# 23. Step 7：Hierarchical Consolidation

最终把：

\[
STM
\rightarrow
LTM
\]

看成一个 consolidation policy：

\[
\pi_{consolidate}
\]

它负责：

```text
Short-term episodes
        │
        ▼
   select segment
        │
        ▼
   summarize event
        │
        ▼
     Skill/Event
        │
        ▼
       LTM
```

例如：

\[
\tau_{t:t+k}
\rightarrow
s_j
\]

其中 \(s_j\) 可以代表：

> “Approach object → grasp → lift successfully”

而不是保存每一个 timestep。

---

# 24. 最终 Hierarchical Memory

完整系统可以写成：

\[
o_t
\rightarrow
z_t
\]

\[
z_t
\rightarrow
\pi_{retrieve}^{STM}
\rightarrow
m_t^{STM}
\]

\[
m_t^{STM}
\rightarrow
\pi_{consolidate}
\rightarrow
m^{LTM}
\]

然后：

\[
z'_t
=
Fusion(
z_t,
m_t^{STM},
m_t^{LTM}
)
\]

最终：

\[
a_t
\sim
\pi_A(a_t|z'_t)
\]

整个系统变成：

\[
\boxed{
\text{Memory Policy}
+
\text{Action Policy}
}
\]

---

# 25. 建议的研究假设

可以把整个项目拆成几个明确 Hypothesis。

## H1：Explicit Memory

\[
RLT+STM > RLT
\]

短期显式经验能够补充 RLT 当前 token。

---

## H2：Relevant Memory

\[
RLT+RetrievedSTM
>
RLT+RecentSTM
\]

相关经验比单纯最近经验更有价值。

---

## H3：Long-term Memory

\[
RLT+STM+LTM
>
RLT+STM
\]

长期 Skill/Event 可以跨 episode 复用经验。

---

## H4：Learnable Retrieval

\[
RL\text{-}RetrievedMemory
>
Similarity\text{-}RetrievedMemory
\]

RL 可以学会更 task-relevant 的 Memory retrieval。

---

## H5：Learnable Write

\[
RL\text{-}Write
>
FixedWrite
\]

Memory 的写入策略本身可以被优化。

---

## H6：Hierarchical Memory

最终：

\[
RLT+STM+LTM+RLMemory
\]

在 sample efficiency、long-horizon task 和 history-dependent task 上取得最佳表现。

---

# 26. 最重要的实验场景：History-dependent Task

一个潜在问题是：

> 普通 ManiSkill 任务可能本身就是 fully observable。

如果：

\[
a_t^*=f(o_t)
\]

那么：

\[
H_t
\]

确实可能没有额外价值。

这不是 Memory 方法失败，而是：

> **任务本身不需要 Memory。**

因此建议增加专门的 history-dependent / partially observable benchmark。

---

# 27. 推荐的 History-dependent Task

可以构造一个隐藏 task mode：

\[
m\in\{A,B\}
\]

当前 observation 相同：

\[
o_t^A\approx o_t^B
\]

但最优动作不同：

\[
a_t^{*,A}
\neq
a_t^{*,B}
\]

mode 信息不能从当前 observation 直接获得，只能通过历史 interaction 推断。

例如：

```text
Episode 开始
     ↓
隐藏 Mode A / B
     ↓
执行若干 interaction
     ↓
出现相同视觉状态
     ↓
需要选择不同 action
```

此时：

\[
a_t^*=f(o_t,H_t)
\]

理论上应该出现：

\[
RLT+STM
\gg
RLT
\]

这是验证 Memory 是否真正解决 temporal dependency 的关键实验。

---

# 28. 评价指标

不要只看最终 Success Rate。

建议至少记录：

## 28.1 Final Success

\[
Success_{final}
\]

---

## 28.2 Episode Return

\[
R_{episode}
=
\sum_t\gamma^tr_t
\]

---

## 28.3 Sample Efficiency

例如：

> 达到 80% success 所需要的 environment steps。

记作：

\[
N_{80}
\]

比较：

\[
N_{80}^{RLT}
\]

和：

\[
N_{80}^{RLT+Memory}
\]

如果：

\[
N_{80}^{Memory}
<
N_{80}^{RLT}
\]

说明 Memory 提高了 sample efficiency。

---

## 28.4 Episode Length

判断 Memory 是否减少：

- 重复尝试
- 无效动作
- 重复失败

---

## 28.5 Memory-specific Metrics

后续可以记录：

- Write rate
- Retrieval frequency
- Retrieval diversity
- Memory hit rate
- Success-conditioned retrieval
- Failure repetition rate
- Memory utilization

这些指标在 Step 5–7 尤其重要。

---

# 29. 推荐的最小实现版本

现在真正开始写代码时，不建议直接实现完整系统。

第一版只做：

```text
Frozen RLT
   +
Fixed STM
   +
GRU Memory Encoder
   +
Residual Fusion
   +
Online RL
```

即：

\[
z_t
\rightarrow
STM(H_t)
\rightarrow
GRU
\rightarrow
m_t
\]

\[
z'_t=z_t+W_mm_t
\]

\[
\pi(a_t|z'_t)
\]

这是整个项目最重要的 **Step 2A MVP**。

---

# 30. 推荐参数

第一版可以从非常保守的配置开始：

```yaml
memory:
    enabled: true

    type: stm

    window_size: 8

    entry:
        token: true
        action: true
        reward: true

    encoder:
        type: gru
        hidden_dim: 256

    fusion:
        type: residual
        memory_proj_dim: 256
```

实际维度需要根据 RLT token dimension 调整。

建议做：

\[
K\in\{4,8,16\}
\]

而不是一开始把 K 做得非常大。

---

# 31. 实现时最重要的工程原则

## 原则 1：Memory 不应该修改原始 RLT

Baseline：

\[
z_t
\rightarrow
Actor
\]

Memory：

\[
z_t
\rightarrow
Memory
\rightarrow
z'_t
\rightarrow
Actor
\]

保证可以随时：

\[
W_m=0
\]

退化回 baseline。

---

## 原则 2：严格避免 Future Leakage

必须保证：

\[
M_t=
\{e_0,\ldots,e_{t-1}\}
\]

不能出现：

\[
e_t
\]

更不能出现：

\[
e_{t+1},e_{t+2},...
\]

---

## 原则 3：Memory Entry 必须有明确生命周期

推荐：

```text
step t
   ↓
生成 z_t
   ↓
读取过去 STM
   ↓
生成 action
   ↓
Environment
   ↓
得到 reward
   ↓
写入 (z_t, a_t, r_t)
```

即：

> **Read first, Act, then Write**

而不是：

> Write current experience → Read → Act

后者很容易造成 current-action leakage。

---

# 32. 推荐的代码模块

在 RLT baseline 上，可以新增一个独立的 Memory module：

```text
memory/
├── __init__.py
├── stm.py
├── encoder.py
├── fusion.py
└── buffer.py
```

逻辑上分成：

```python
STMBuffer
    ├── write()
    ├── get_recent()
    └── reset()

MemoryEncoder
    └── forward(history)

MemoryFusion
    └── forward(z, memory)

MemoryModule
    ├── read()
    └── write()
```

这样以后 Step 2B、Step 5、Step 6 都可以替换内部实现，而不需要重写整个 Actor。

---

# 33. 推荐的接口设计

理想情况下：

```python
memory_state = memory.reset()

memory_context = memory.read(
    current_token=z_t,
    state=memory_state,
)

z_fused = fusion(
    z_t,
    memory_context,
)

action = actor(z_fused)

next_obs, reward, done, info = env.step(action)

memory_state = memory.write(
    token=z_t,
    action=action,
    reward=reward,
    state=memory_state,
)
```

未来只需要替换：

```python
memory.read()
```

就可以从：

```text
Recent History
```

变成：

```text
Similarity Retrieval
```

再变成：

```text
RL Retrieval
```

而：

```python
actor()
critic()
```

基本不需要改变。

---

# 34. 最终论文故事

整个研究最清晰的故事不是：

> “我们设计了一个复杂的 Memory。”

而应该是：

> **RLT's current token provides a compact representation of the current state, but may not retain task-relevant episodic information from recent interaction history. We therefore introduce hierarchical episodic memory and progressively learn how to retrieve, write, and consolidate experience through online reinforcement learning.**

整个逻辑链：

```text
RLT
 │
 │ 当前 token 可能丢失历史
 ▼
STM
 │
 │ 显式保存近期 experience
 ▼
Relevant Retrieval
 │
 │ 不应该所有 history 都读取
 ▼
LTM
 │
 │ 跨 episode 复用 skill
 ▼
RL Memory
 │
 │ Memory 本身也应该学习
 ▼
Hierarchical RL Memory
```

最终核心贡献可以概括为：

\[
\boxed{
\text{Policy learns from experience}
}
\]

进一步：

\[
\boxed{
\text{Memory learns which experience matters}
}
\]

最终：

\[
\boxed{
\text{Memory and Policy co-adapt through Online RL}
}
\]

---

# 35. 当前下一步：Step 2A

既然 RLT baseline 已经完成，现在不要直接做 LTM，也不要直接做 RL Retriever。

建议严格按照下面顺序：

### Step 2A.1

复制 RLT baseline。

### Step 2A.2

冻结 RLT token representation。

### Step 2A.3

增加：

```text
STMBuffer
```

保存：

\[
(z_t,a_t,r_t)
\]

### Step 2A.4

实现：

```text
get_recent(K)
```

### Step 2A.5

增加：

```text
GRU Memory Encoder
```

### Step 2A.6

实现：

\[
z'_t=z_t+W_mm_t
\]

### Step 2A.7

Actor 和 Critic 都使用：

\[
(z_t,m_t)
\]

### Step 2A.8

继续使用原来的 Online RL loss。

### Step 2A.9

完成：

```text
RLT
vs
RLT + Fixed STM
```

### Step 2A.10

如果确认 STM 有收益，再进入：

```text
Similarity Retrieval
```

---

# 36. 最终结论

当前最合理的实验路线是：

\[
\boxed{
RLT
\rightarrow
RLT+FixedSTM
\rightarrow
RLT+SimilaritySTM
\rightarrow
RLT+STM+LTM
\rightarrow
RL\text{-}Retrieval
\rightarrow
RL\text{-}Write
\rightarrow
RL\text{-}Consolidation
}
\]

而当前真正需要实现的只是：

\[
\boxed{
z_t
+
\underbrace{
GRU(z_{t-K:t-1},a_{t-K:t-1},r_{t-K:t-1})
}_{STM}
\rightarrow
z'_t
\rightarrow
Actor/Critic
}
\]

**不要把 Memory 当成 imitation label。**

Memory 的角色应该是：

\[
\boxed{
\text{Context for Policy}
}
\]

而不是：

\[
\boxed{
\text{Action Target}
}
\]

Step 2A 的成功标准也不是“STM 一定提升所有任务”，而是回答一个更基础、也更科学的问题：

\[
\boxed{
\text{Does explicit recent experience provide decision-relevant information beyond the current RLT token?}
}
\]

如果答案是 yes，才有充分理由继续研究 Retrieval、LTM，以及 RL-trained Memory。
