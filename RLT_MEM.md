# RLT 历史经验记忆扩展：Actor Memory 与 Critic Memory

## 1. 研究动机

原始 RLT（RL Token）通过将 VLA 的高维内部表征压缩为紧凑的 RL token，使轻量 Actor-Critic 能够利用 VLA 的知识进行在线强化学习。

原始 RLT 在当前决策时主要依赖当前 chunk 的信息：

$$
z_t = E_{\mathrm{VLA}}(o_t)
$$

Actor 根据当前 RL token 和 VLA reference action 产生 action chunk：

$$
a_t =
\pi_\theta(z_t,a_t^{ref},s_t)
$$

Critic 根据当前状态和 action 估计价值：

$$
Q_\phi(z_t,a_t,s_t)
$$

这种设计对于短时程任务有效，但对于长时程任务存在一个根本问题：

> 当前 chunk 的决策可能依赖过去多个 chunk 中已经发生的行为和反馈，而当前 RL token 本身并不包含这些历史经验。

因此，仅仅将历史观测直接拼接到当前输入，或者将历史信息强行压缩到原有 \(z_t\) 中，并不能保证 Actor-Critic 真正使用历史信息。

核心原因是：

* Actor 需要的是**过去什么行为有效**；
* Critic 需要的是**过去行为产生了什么结果以及反馈模式如何**；
* 两者需要的历史信息具有不同的功能。

因此提出双通路历史经验记忆：

$$
\boxed{
\text{Actor Memory}
+
\text{Critic Memory}
}
$$

两者均采用与 RL token 类似的低维 latent 表征作为信息载体，但分别服务于 Actor 和 Critic。

---

# 2. 总体框架

在原始 RLT 的基础上增加两个历史经验 token：

$$
m_t^A
$$

表示 Actor Memory；

$$
m_t^C
$$

表示 Critic Memory。

当前 RL token 保持：

$$
z_t
$$

最终形成：

$$
\boxed{
z_t + m_t^A + m_t^C
}
$$

但三者承担不同职责。

### Actor

$$
\boxed{
a_t =
\pi_\theta
(z_t,m_t^A,a_t^{ref},s_t)
}
$$

其中：

* \(z_t\)：当前 VLA 的 RL token；
* \(m_t^A\)：历史行为经验；
* \(a_t^{ref}\)：VLA reference action chunk；
* \(s_t\)：机器人 proprioception；
* \(a_t\)：RL Actor 输出的 action chunk。

### Critic

$$
\boxed{
Q_t =
Q_\phi
(z_t,m_t^C,a_t,s_t)
}
$$

其中：

* \(z_t\)：当前 RL token；
* \(m_t^C\)：历史反馈经验；
* \(a_t\)：当前候选 action chunk；
* \(s_t\)：当前机器人状态。

因此整体结构为：

```text
                         Current Observation
                                │
                                ▼
                              VLA
                                │
                                ▼
                              z_t
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
       Actor Memory                         Critic Memory
           m_A                                  m_C
              │                                   │
              ▼                                   ▼
       ┌──────────────┐                    ┌──────────────┐
       │    Actor     │                    │    Critic    │
       │ z + m_A + ref│                    │ z + m_C + a  │
       └──────┬───────┘                    └──────┬───────┘
              │                                   │
              ▼                                   ▼
         Action Chunk                         Q Value
              │                                   │
              └───────────────┬───────────────────┘
                              ▼
                         Environment
                              │
                         reward / next state
                              │
                              ▼
                         Experience
```

---

# 3. Actor Memory

## 3.1 定义

Actor Memory 用于保存和编码过去的**行为经验**。

核心问题是：

> 在过去类似状态下，我采取了什么行为？这个行为相对于 VLA 做了什么调整？最终是否有效？

因此 Actor Memory 不应该只是历史观测，而应该是：

$$
\boxed{
\text{State}
+
\text{Action}
+
\text{Outcome}
}
$$

即历史交互经验。

---

## 3.2 Actor Memory 的历史单元

对于过去第 \(i\) 个 chunk，可以定义：

$$
e_i^A =
[
z_i,
a_i^{ref},
a_i,
r_i
]
$$

其中：

* \(z_i\)：过去时刻的 RL token；
* \(a_i^{ref}\)：VLA reference action；
* \(a_i\)：实际执行的 action；
* \(r_i\)：该 chunk 获得的 reward。

由于 RLT 本身具有 reference action，因此可以进一步使用 action correction：

$$
\Delta a_i
=
a_i-a_i^{ref}
$$

于是更紧凑的经验表示可以写成：

$$
\boxed{
e_i^A =
[
z_i,
\Delta a_i,
r_i
]
}
$$

相比直接保存 action，\(\Delta a_i\) 更直接描述：

> RL 相对于原始 VLA 做了什么修正。

---

## 3.3 Actor Memory Encoder

过去 \(K\) 个 chunk 构成：

$$
\mathcal M_t^A
=
\{
e_{t-K}^A,
\dots,
e_{t-1}^A
\}
$$

通过 Actor Memory Encoder：

$$
m_t^A
=
E_A(
e_{t-K}^A,\dots,e_{t-1}^A
)
$$

得到固定维度的 memory token：

$$
\boxed{
m_t^A\in\mathbb R^d
}
$$

其中 \(d\) 与 RLT token 的维度保持一致或处于相同数量级。

第一版可以使用非常轻量的 Transformer、MLP + pooling 或其他简单序列编码器，不需要引入复杂 memory bank。

---

## 3.4 Actor Memory 学习什么？

Actor Memory 的核心目标不是记住完整历史，而是提取：

$$
\boxed{
\text{过去哪些行为在类似情况下有效}
}
$$

例如：

```text
过去状态 z_i
      ↓
VLA 给出 action
      ↓
RL 对 action 向右修正
      ↓
任务成功
      ↓
形成正向经验
```

当未来出现相似的：

$$
z_t\approx z_i
$$

Actor 可以通过：

$$
m_t^A
$$

获得过去的行为经验，并倾向于复用有效的 action correction。

因此：

$$
\boxed{
m_t^A
=
\text{Behavioral Experience}
}
$$

---

# 4. Actor 的输入与优化

Actor 最终变为：

$$
\boxed{
a_t=
\pi_\theta
(z_t,m_t^A,a_t^{ref},s_t)
}
$$

相比原始 RLT：

$$
a_t=
\pi_\theta(z_t,a_t^{ref},s_t)
$$

唯一新增的是：

$$
m_t^A
$$

因此它并不改变 RLT 的基本 action refinement 机制。

---

## 4.1 Actor Loss

保持 RLT 原有的 RL objective：

$$
\mathcal L_A
=
-Q_\phi(z_t,m_t^C,a_t,s_t)
+
\beta
\left\|
a_t-a_t^{ref}
\right\|^2
$$

其中：

第一项：

$$
-Q_\phi
$$

鼓励 Actor 选择高价值 action。

第二项：

$$
\left\|
a_t-a_t^{ref}
\right\|^2
$$

使 RL policy 保持在 VLA 行为先验附近。

因此 Actor Memory 并不替换 reference action，而是：

$$
\boxed{
\text{利用历史经验帮助 Actor 决定如何修正 reference action}
}
$$

---

# 5. Critic Memory

## 5.1 定义

Critic Memory 与 Actor Memory 的目的不同。

Critic 并不主要需要知道：

> “过去应该怎么做？”

而需要知道：

> “过去发生了什么，以及这些行为产生了什么反馈？”

因此 Critic Memory 表示：

$$
\boxed{
\text{Historical Feedback Experience}
}
$$

---

## 5.2 Critic Memory 的历史单元

对于过去第 \(i\) 个 chunk，可以定义：

$$
e_i^C=
[
z_i,
a_i,
r_i,
z_{i+1},
d_i
]
$$

其中：

* \(z_i\)：当前状态 RL token；
* \(a_i\)：实际执行的 action；
* \(r_i\)：reward；
* \(z_{i+1}\)：下一状态 RL token；
* \(d_i\)：episode 是否结束。

因此 Critic Memory 比 Actor Memory 更强调：

$$
\boxed{
(state,action,reward,next\ state)
}
$$

即完整的 value transition。

---

## 5.3 Critic Memory Encoder

历史反馈序列：

$$
\mathcal M_t^C
=
\{
e_{t-K}^C,
\dots,
e_{t-1}^C
\}
$$

通过独立的 Critic Memory Encoder：

$$
m_t^C
=
E_C(
e_{t-K}^C,\dots,e_{t-1}^C
)
$$

得到：

$$
\boxed{
m_t^C\in\mathbb R^d
}
$$

这里：

$$
E_A\neq E_C
$$

即 Actor Memory 和 Critic Memory 使用不同的 encoder。

这样可以避免将不同功能的历史信息强行压缩到同一个 latent。

---

# 6. Critic 的输入

Critic 变为：

$$
\boxed{
Q_t
=
Q_\phi(
z_t,m_t^C,a_t,s_t
)
}
$$

相比原始 RLT：

$$
Q_t=Q_\phi(z_t,a_t,s_t)
$$

新增：

$$
m_t^C
$$

因此 Critic 可以利用过去的反馈经验判断当前 action 的价值。

---

# 7. Critic Target

Critic 使用 TD learning。

当前 TD target：

$$
y_t
=
r_t+
\gamma
Q_{\bar\phi}
(
z_{t+1},
m_{t+1}^C,
a_{t+1},
s_{t+1}
)
$$

因此：

$$
\boxed{
\mathcal L_C
=
\left[
Q_\phi(z_t,m_t^C,a_t,s_t)
-
y_t
\right]^2
}
$$

一个非常重要的设计原则是：

$$
\boxed{
Q_t\rightarrow m_t^C
}
$$

而 target value：

$$
\boxed{
Q_{t+1}\rightarrow m_{t+1}^C
}
$$

不能简单地在整个 trajectory 中复用同一个 memory token，否则会产生时间错位。

---

# 8. Actor Memory 与 Critic Memory 的区别

| 维度      | Actor Memory              | Critic Memory               |
| ------- | ------------------------- | --------------------------- |
| 核心问题    | 过去什么行为有效？                 | 过去发生了什么？                    |
| 服务对象    | Actor                     | Critic                      |
| 核心信息    | 行为经验                      | 反馈经验                        |
| 历史单元    | \(z_i,a_i^{ref},a_i,r_i\) | \(z_i,a_i,r_i,z_{i+1},d_i\) |
| 推荐压缩    | \(z_i,\Delta a_i,r_i\)    | \(z_i,a_i,r_i,z_{i+1},d_i\) |
| Encoder | \(E_A\)                   | \(E_C\)                     |
| 输出      | \(m_A\)                   | \(m_C\)                     |
| 主要作用    | 改善 action refinement      | 改善 value estimation         |
| 关注重点    | Policy experience         | Value experience            |
| 是否共享    | 不建议第一版共享                  | 不建议第一版共享                    |

可以概括为：

$$
\boxed{
m_A=\text{What worked?}
}
$$

$$
\boxed{
m_C=\text{What happened?}
}
$$

---

# 9. 为什么不直接共享一个 Memory？

不建议第一版设计：

$$
m=f(E_{history})
$$

然后：

$$
Actor(z,m)
$$

和：

$$
Critic(z,m,a)
$$

共同使用。

原因是 Actor 和 Critic 对历史信息的需求不同。

Actor 更关注：

$$
(z_i,\Delta a_i,r_i)
$$

即：

> 什么行为在过去产生了好的结果。

Critic 更关注：

$$
(z_i,a_i,r_i,z_{i+1})
$$

即：

> 一个状态-动作转移产生了什么反馈。

如果强行共享，很容易重新出现原来的问题：

> 历史信息虽然进入网络，但没有形成针对 Actor 或 Critic 的有效功能表征。

因此第一版建议：

$$
\boxed{
E_A\neq E_C
}
$$

但：

$$
\boxed{
m_A,m_C\in\mathbb R^d
}
$$

保持与 RL token 相近的维度。

---

# 10. 与原始 RLT 的关系

这个方法不改变 RLT 的核心框架。

原始 RLT：

$$
z_t
\rightarrow
Actor/Critic
\rightarrow
a_t
$$

扩展后：

$$
\boxed{
z_t+
m_t^A+
m_t^C
\rightarrow
Actor/Critic
}
$$

其中：

$$
z_t
$$

仍然负责当前 VLA 表征；

$$
m_t^A
$$

负责历史行为经验；

$$
m_t^C
$$

负责历史反馈经验。

因此该方法不是重新设计一个新的 RL policy，而是：

> **在 RLT 的 RL interface 上增加历史经验表征。**

---

# 11. 推荐的最小可行版本

第一版不要加入复杂机制。

推荐：

### Current RL Token

$$
z_t
$$

保持原始 RLT 不变。

### Actor Memory

$$
e_i^A=
[
z_i,
\Delta a_i,
r_i
]
$$

$$
m_t^A
=
E_A(e_{t-K}^A,\dots,e_{t-1}^A)
$$

### Critic Memory

$$
e_i^C=
[
z_i,
a_i,
r_i,
z_{i+1},
d_i
]
$$

$$
m_t^C
=
E_C(e_{t-K}^C,\dots,e_{t-1}^C)
$$

### Actor

$$
a_t=
\pi_\theta
(z_t,m_t^A,a_t^{ref},s_t)
$$

### Critic

$$
Q_t=
Q_\phi
(z_t,m_t^C,a_t,s_t)
$$

---

# 12. 实验设计

建议首先在 ManiSkill 上验证，不立即引入长时程任务。

设置历史长度：

$$
K\in\{0,1,2,4,8\}
$$

比较：

| 方法        | Actor Memory | Critic Memory |
| --------- | -----------: | ------------: |
| RLT       |            × |             × |
| History-z |            ✓ |             ✓ |
| RLT-A     |            ✓ |             × |
| RLT-C     |            × |             ✓ |
| RLT-AC    |            ✓ |             ✓ |

其中：

### RLT

原始方法：

$$
\pi(z_t,a_t^{ref})
$$

### History-z

将历史直接压缩进当前表示：

$$
z_t'=E(o_{t-K:t})
$$

用于验证：

> 简单增加历史输入是否足够。

### RLT-A

只增加：

$$
m_A
$$

用于验证：

> 历史行为经验是否真正帮助 Actor。

### RLT-C

只增加：

$$
m_C
$$

用于验证：

> 历史反馈经验是否真正帮助 Critic。

### RLT-AC

同时使用：

$$
m_A+m_C
$$

用于验证：

> Actor/Critic 双通路是否产生互补作用。

---

# 13. 最重要的验证指标

不能只看最终 success rate。

还应该验证 Memory 是否真的被使用。

## 13.1 Actor Memory Effect

在相似当前状态：

$$
z_t\approx z_j
$$

下，比较不同历史：

$$
m_A^{good}
$$

和：

$$
m_A^{bad}
$$

是否导致不同 action：

$$
\pi(z,m_A^{good})
\neq
\pi(z,m_A^{bad})
$$

如果几乎完全相同，说明 Actor 没有使用 Memory。

---

## 13.2 Critic Memory Effect

固定：

$$
z_t,a_t
$$

改变历史反馈：

$$
m_C^{good}
$$

与：

$$
m_C^{bad}
$$

观察：

$$
Q(z,m_C^{good},a)
$$

与：

$$
Q(z,m_C^{bad},a)
$$

是否产生系统性差异。

如果：

$$
Q(z,m_C^{good},a)
\approx
Q(z,m_C^{bad},a)
$$

说明 Critic 同样忽略了 Memory。

---

# 14. 最终研究假设

整个方法可以归纳为两个核心假设。

### 假设一：Actor Memory

历史经验能够帮助 Actor 学习：

$$
\boxed{
\text{过去有效的行为修正模式}
}
$$

从而：

$$
\pi(a_t|z_t,m_t^A)
$$

比：

$$
\pi(a_t|z_t)
$$

更适合存在跨 chunk 行为依赖的任务。

---

### 假设二：Critic Memory

历史反馈能够帮助 Critic 学习：

$$
\boxed{
\text{当前决策所处的历史价值上下文}
}
$$

从而：

$$
Q(z_t,m_t^C,a_t)
$$

比：

$$
Q(z_t,a_t)
$$

能够更准确地估计长时程任务中的 action value。

---

# 15. 核心思想总结

最终可以将整个方法浓缩成：

$$
\boxed{
\text{Current State}
+
\text{Behavioral Experience}
+
\text{Feedback Experience}
}
$$

分别对应：

$$
\boxed{
z_t
}
$$

$$
\boxed{
m_t^A
}
$$

$$
\boxed{
m_t^C
}
$$

并形成：

$$
\boxed{
a_t=
\pi_\theta(z_t,m_t^A,a_t^{ref},s_t)
}
$$

$$
\boxed{
Q_t=
Q_\phi(z_t,m_t^C,a_t,s_t)
}
$$

其中：

> **Actor Memory 记住“过去什么行为有效”，Critic Memory 记住“过去行为产生了什么反馈”。**

最终目标不是让 RLT 获得一个更大的历史输入，而是让历史经验成为**能够真正改变 Actor 行为和 Critic 价值判断的 RL latent representation**。

因此整个扩展的核心可以概括为：

$$
\boxed{
\text{RLT}
+
\text{Actor Behavioral Memory}
+
\text{Critic Feedback Memory}
}
$$

而不是：

$$
\text{RLT}+\text{History Concatenation}
$$

也不是：

$$
\text{RLT}+\text{History-enhanced }z
$$

这一区别是该方法设计的核心。
