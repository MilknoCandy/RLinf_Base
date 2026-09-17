# C：RLT + Dynamic
## 流式动态表示增强的在线强化学习方案

> **版本**：v1.0  
> **定位**：在不维护显式历史帧缓存的前提下，为 RLT 增加一个固定维度、逐步更新的动态状态，使 RL Token 同时包含当前视觉语义与历史演化信息。  
> **核心关键词**：RLT、Streaming Dynamic State、Predictive Representation、Online RL、ManiSkill

---

## 1. 背景与动机

RLT（RL Tokens）将 VLA 内部视觉/多模态表征压缩成紧凑的 RL Token，并使用轻量 Actor/Critic 进行在线强化学习。其核心思想是：冻结 VLA，只让 RL 模块在低维表示空间中进行高效策略优化，从而避免直接对大型 VLA 进行在线 RL。

对于当前方案，推理过程可以抽象为：

```text
Observation_t
      ↓
Frozen VLA
      ↓
Visual Representation F_t
      ↓
RLT Encoder
      ↓
RL Token z_t
      ↓
Actor / Critic
      ↓
Action_t
```

这一结构主要依赖当前时刻的视觉表示。

在 ManiSkill 实验中，如果仅使用当前图像作为输入，也可以获得接近完整状态输入的性能。这说明问题未必是当前视觉表征缺少足够的空间/语义信息，而更可能在于：

> **单帧视觉表征缺少对“状态正在如何变化”的显式表示。**

此前尝试直接拼接历史 `z`、检索历史信息或将历史动作显式输入 RL 模块，并没有带来稳定的收敛收益，同时增加了计算和实现复杂度。

因此，本方案不再把重点放在“保存更多历史”，而是将历史信息在线压缩成一个**动态状态（Dynamic State）**。

---

# 2. 核心思想

## 2.1 不保存历史帧，而是递归更新动态状态

方案不维护：

```text
[I_{t-k}, ..., I_{t-1}, I_t]
```

也不维护：

```text
[z_{t-k}, ..., z_{t-1}, z_t]
```

而是只维护一个固定大小的状态：

```text
h_t
```

其更新形式为：

\[
h_t = U_\theta(h_{t-1}, F_t)
\]

其中：

- \(F_t\)：当前时刻冻结 VLA 提取的视觉表示；
- \(h_{t-1}\)：上一时刻的动态状态；
- \(U_\theta\)：可学习的动态状态更新器；
- \(h_t\)：当前时刻新的动态表示。

因此：

\[
h_0 \rightarrow h_1 \rightarrow h_2 \rightarrow \cdots \rightarrow h_t
\]

历史信息随着环境交互不断被压缩进当前状态。

这里的 `h_t` 不是一个“历史缓存”，而是一个**递归状态**。

---

# 3. C方案整体结构

完整模型：

```text
                         ┌────────────────────┐
Observation_t ─────────► │    Frozen VLA      │
                         └─────────┬──────────┘
                                   │
                                   ▼
                              F_t
                         VLA visual feature
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
                  ▼                                 ▼
          ┌───────────────┐                 ┌─────────────────┐
          │ RLT Encoder   │                 │ Dynamic Encoder │
          └───────┬───────┘                 └────────┬────────┘
                  │                                  │
                  ▼                                  ▼
              z_app,t                              h_t
          appearance token                    dynamic state
                  │                                  │
                  └────────────────┬─────────────────┘
                                   ▼
                           Gated Fusion
                                   │
                                   ▼
                              z_RL,t
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                      Actor                Critic
                         │                   │
                         └─────────┬─────────┘
                                   ▼
                                Action_t
                                   │
                                   ▼
                              Environment
                                   │
                                   ▼
                              Observation
                                   │
                                   └──────────► t+1
```

最终 RL 输入为：

\[
z_t^{RL}
=
Fusion(z_t^{app},h_t)
\]

其中：

- \(z_t^{app}\)：原始 RLT 当前视觉 Token；
- \(h_t\)：流式动态表示；
- \(z_t^{RL}\)：最终提供给 Actor/Critic 的 RL Token。

---

# 4. 为什么不是直接拼接历史 z

直接历史拼接的形式：

\[
z_t^{RL} = [z_t,z_{t-1},...,z_{t-k}]
\]

存在几个问题：

1. 输入维度随历史长度增长；
2. 历史信息未经任务相关变换；
3. Actor/Critic 需要自己学习“哪些历史值得保留”；
4. 历史帧之间存在大量冗余；
5. 每一步都需要处理多个历史 token；
6. 难以形成固定计算量的在线推理结构。

C方案改成：

\[
h_t=U(h_{t-1},F_t)
\]

因此无论运行了 10 步还是 1000 步，RL 模块看到的状态维度始终不变。

---

# 5. Dynamic Encoder

## 5.1 输入

冻结 VLA 输出：

\[
F_t \in \mathbb{R}^{N\times D_F}
\]

首先使用线性层投影：

\[
X_t=W_FF_t
\]

得到：

\[
X_t\in\mathbb{R}^{N\times D}
\]

推荐第一版：

```text
D = 256
Dynamic Tokens = 16
```

因此动态状态：

\[
h_t\in\mathbb{R}^{16\times256}
\]

最终只保留一个固定大小的动态状态。

---

# 6. Dynamic Update

第一版建议不要直接上复杂的视频模型，而是实现一个轻量的 Transformer/Cross-Attention 更新器。

定义：

\[
Q_t=W_QX_t
\]

\[
K_t=W_Kh_{t-1}
\]

\[
V_t=W_Vh_{t-1}
\]

然后：

\[
C_t=Attention(Q_t,K_t,V_t)
\]

将当前视觉输入和历史动态状态进行融合：

\[
M_t=MLP([X_t,C_t])
\]

最后更新：

\[
\tilde h_t=TransformerBlock(h_{t-1},M_t)
\]

得到：

\[
h_t=\tilde h_t
\]

---

# 7. 第一版更简单的 Sanity Baseline

在实现完整 Dynamic Encoder 之前，可以先实现 GRU 版本。

\[
h_t=GRU(X_t,h_{t-1})
\]

目的不是作为最终方案，而是回答一个非常关键的问题：

> **仅仅加入递归动态状态，是否能够改善 RLT 的在线 RL？**

因此建议：

```text
A: RLT baseline
B: RLT + GRU Dynamic
C: RLT + Transformer Dynamic
```

如果 B 已经有效，说明“时间递归状态”本身就有价值。

如果 B 无效而 C 有效，则说明简单递归容量不足。

---

# 8. Gated Fusion

动态状态不应该强制覆盖原始 RLT Token。

因此采用残差式门控融合：

\[
p_t=Pool(h_t)
\]

\[
g_t=\sigma(MLP([z_t^{app},p_t]))
\]

然后：

\[
z_t^{RL}
=
z_t^{app}
+
g_tW_hp_t
\]

其中：

- \(z_t^{app}\)：原始 RLT Token；
- \(p_t\)：动态状态池化结果；
- \(g_t\)：动态信息注入比例；
- \(W_h\)：将动态表示投影到 RL Token 空间。

这样模型可以自行学习：

```text
静态场景：
g_t → 小
→ 主要依赖当前视觉

动态场景：
g_t → 大
→ 更多使用动态信息
```

因此不会强迫所有任务都使用动态表示。

---

# 9. 动态状态的训练目标

如果只训练 RL loss：

\[
L=L_{RL}
\]

Dynamic Encoder 很容易退化成一个普通的特征融合模块。

因此给动态状态加入**未来预测任务**。

核心假设：

> 一个好的动态状态应该不仅描述“现在是什么”，还应该包含足够的信息来预测“接下来会发生什么”。

---

# 10. Future Prediction

动态状态：

\[
h_t
\]

预测未来 RLT Token：

\[
\hat z_{t+k}=P_k(h_t)
\]

推荐预测多个时间尺度：

\[
k\in\{1,4,8\}
\]

即：

```text
h_t
 ├──► z_hat(t+1)
 ├──► z_hat(t+4)
 └──► z_hat(t+8)
```

注意：

> 这里的 1/4/8 是预测 horizon，不是推理时需要读取的历史帧数量。

在线推理仍然只读取：

```text
I_t
```

然后：

```text
h_t = Update(h_{t-1}, F_t)
```

---

# 11. Prediction Loss

使用未来真实 RLT Token 作为 target：

\[
z_{t+k}=RLT(F_{t+k})
\]

预测损失：

\[
L_{pred}
=
\lambda_1D(\hat z_{t+1},sg(z_{t+1}))
+
\lambda_4D(\hat z_{t+4},sg(z_{t+4}))
+
\lambda_8D(\hat z_{t+8},sg(z_{t+8}))
\]

其中：

- `sg`：stop-gradient；
- \(D\)：建议首先使用 cosine distance。

即：

\[
D(a,b)=1-\cos(a,b)
\]

推荐初始权重：

```text
λ1 = 1.0
λ4 = 0.5
λ8 = 0.25
```

总损失：

\[
L=L_{RL}+\lambda_{pred}L_{pred}
\]

第一组实验：

```text
λpred ∈ {0.03, 0.1, 0.3}
```

建议从：

```text
λpred = 0.1
```

开始。

---

# 12. 为什么预测 RLT Token，而不是预测像素

本方案不建议第一版做：

\[
h_t\rightarrow I_{t+1}
\]

原因是像素预测容易把模型容量消耗在：

- 背景变化；
- 光照；
- 纹理；
- 相机噪声；
- 与控制无关的视觉细节。

而 RLT 本身已经提供了任务相关的视觉压缩表示。

因此让 Dynamic Encoder 预测：

\[
z_{t+k}
\]

更接近：

> **未来对控制有意义的视觉状态。**

这使 Dynamic State 更容易成为 RL representation，而不是一个普通视频预测模型。

---

# 13. Online RL 推理流程

每一个环境 timestep 只处理一张新的图像。

```python
obs = env.get_observation()

F = frozen_vla.encode_image(obs)

z_app = rlt_encoder(F)

h = dynamic_encoder.update(h, F)

z_rl = gated_fusion(z_app, h)

action = actor(z_rl, ref_action)

obs, reward, done = env.step(action)

if done:
    h = dynamic_encoder.reset()
```

核心特征：

```text
没有 frame buffer
没有 z history
没有 image cache
没有 retrieval
```

只存在：

```text
h
```

因此计算复杂度不会随着 episode 长度增长。

---

# 14. Episode Reset

每个 episode 开始：

\[
h_0=0
\]

或者使用一个可学习初始状态：

\[
h_0=h_{init}
\]

推荐第一版直接使用：

```python
h = torch.zeros(...)
```

避免额外引入参数。

episode 结束：

```python
h = dynamic_encoder.reset()
```

---

# 15. Replay Buffer 的处理

虽然在线推理不保存历史帧，但训练 Dynamic Encoder 时需要进行时间展开。

因此 replay buffer 不需要保存一个长期历史缓存。

推荐保存：

```text
F_t
action_t
reward_t
done_t
reference_action_t
next information
```

其中：

```text
F_t = frozen VLA image representation
```

可以直接保存冻结 VLA 的视觉 embedding。

这样训练 Dynamic Encoder 时：

```text
F_t
F_{t+1}
F_{t+2}
...
F_{t+T}
```

从 replay 中取连续 trajectory segment。

---

# 16. Sequence Training

例如：

```text
sequence length T = 16
```

训练时：

```python
h = h0

for t in range(T):
    F_t = batch_F[:, t]

    z_app_t = rlt_encoder(F_t)

    h = dynamic_encoder.update(h, F_t)

    z_rl_t = gated_fusion(z_app_t, h)

    action_t = actor(z_rl_t)

    critic_t = critic(z_rl_t, action_t)

    # future prediction
    z_hat_1 = predictor_1(h)
    z_hat_4 = predictor_4(h)
    z_hat_8 = predictor_8(h)
```

这样可以进行 BPTT。

需要注意：

> `T=16` 是训练时的 unroll length，不代表在线推理需要缓存 16 帧。

---

# 17. Replay 数据结构

推荐：

```python
Transition(
    visual_feature=F_t,
    action=a_t,
    reward=r_t,
    done=d_t,
    reference_action=ref_a_t,
)
```

sequence sampler：

```python
Sequence(
    F[t:t+T],
    action[t:t+T],
    reward[t:t+T],
    done[t:t+T],
    reference_action[t:t+T],
)
```

优先存储 `F_t` 而不是重新调用大型 VLA。

原因：

1. VLA 冻结；
2. VLA 特征不会变化；
3. 大幅减少 Dynamic Encoder 训练成本；
4. replay sampling 更快；
5. 可以将实验重点放在 Dynamic Encoder 本身。

---

# 18. 与原始 RLT 的兼容方式

原始 RLT 的 Actor/Critic 尽量不要修改。

原始：

```python
action = actor(z_app, ref_action)
q = critic(z_app, action)
```

改成：

```python
z_rl = gated_fusion(z_app, h)

action = actor(z_rl, ref_action)
q = critic(z_rl, action)
```

因此整个修改只发生在：

```text
VLA feature
      ↓
Dynamic Encoder
      ↓
Fusion
      ↓
Actor/Critic
```

这有利于进行严格 ablation。

---

# 19. 推荐网络结构

第一版：

```text
Frozen VLA
    │
    ├── F_t
    │
    ├── RLT Encoder
    │      └── z_app
    │
    └── Dynamic Encoder
           │
           ├── Linear Projection
           ├── Cross Attention
           ├── MLP
           ├── Transformer Block
           └── h_t
                    │
                    ▼
               Gated Fusion
                    │
                    ▼
                  z_RL
```

推荐参数：

| 模块 | 初始设置 |
|---|---|
| Feature dim | 256 |
| Dynamic tokens | 16 |
| Dynamic layers | 2 |
| Attention heads | 4 |
| MLP ratio | 4 |
| Prediction horizons | 1 / 4 / 8 |
| Prediction loss | Cosine |
| λpred | 0.1 |
| Unroll length | 16 |

这些参数首先用于验证机制，不应视为最终最优配置。

---

# 20. PyTorch 结构示意

```python
class DynamicEncoder(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        num_tokens=16,
        num_heads=4,
    ):
        super().__init__()

        self.feature_proj = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.q_proj = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.k_proj = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.v_proj = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.update = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            batch_first=True,
        )

        self.num_tokens = num_tokens
        self.feature_dim = feature_dim

    def init_state(self, batch_size, device):
        return torch.zeros(
            batch_size,
            self.num_tokens,
            self.feature_dim,
            device=device,
        )

    def forward(self, F_t, h_prev):

        x = self.feature_proj(F_t)

        q = self.q_proj(x)
        k = self.k_proj(h_prev)
        v = self.v_proj(h_prev)

        context, _ = self.attn(
            q,
            k,
            v,
        )

        # aggregate current observation
        m = context.mean(dim=1, keepdim=True)

        h = h_prev + m

        h = self.update(h)

        return h
```

这是结构原型。

正式实验时建议将 `m` 投影到 `num_tokens`，而不是简单复制/广播，以提高动态状态更新能力。

---

# 21. Gated Fusion 实现

```python
class GatedFusion(nn.Module):
    def __init__(
        self,
        z_dim,
        dynamic_dim,
    ):
        super().__init__()

        self.dynamic_proj = nn.Linear(
            dynamic_dim,
            z_dim,
        )

        self.gate = nn.Sequential(
            nn.Linear(
                z_dim + dynamic_dim,
                z_dim,
            ),
            nn.GELU(),
            nn.Linear(z_dim, 1),
        )

    def forward(self, z_app, h):

        h_pool = h.mean(dim=1)

        dynamic = self.dynamic_proj(h_pool)

        gate_input = torch.cat(
            [z_app, h_pool],
            dim=-1,
        )

        g = torch.sigmoid(
            self.gate(gate_input)
        )

        z_rl = z_app + g * dynamic

        return z_rl
```

---

# 22. Future Predictor

```python
class FuturePredictor(nn.Module):
    def __init__(
        self,
        hidden_dim,
        z_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, h):
        h = h.mean(dim=1)
        return self.net(h)
```

分别建立：

```python
predictor_1
predictor_4
predictor_8
```

---

# 23. Loss 实现

```python
def cosine_loss(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    return 1.0 - (pred * target).sum(dim=-1).mean()
```

训练：

```python
loss_pred = (
    1.0 * cosine_loss(
        z_hat_1,
        z_future_1.detach(),
    )
    +
    0.5 * cosine_loss(
        z_hat_4,
        z_future_4.detach(),
    )
    +
    0.25 * cosine_loss(
        z_hat_8,
        z_future_8.detach(),
    )
)

loss = loss_rl + 0.1 * loss_pred
```

---

# 24. 训练阶段

建议分两个阶段。

## Stage 1：Dynamic Pretraining

冻结：

```text
VLA
RLT Encoder
Actor
Critic
```

只训练：

```text
Dynamic Encoder
Future Predictors
Fusion
```

目标：

\[
L=L_{pred}
\]

目的：

让 `h_t` 首先学会表示视觉状态的时间演化。

---

## Stage 2：Online RL

然后加入：

\[
L_{RL}
\]

训练：

```text
Dynamic Encoder
Fusion
Actor
Critic
```

VLA 继续冻结。

总损失：

\[
L=L_{RL}+\lambda_{pred}L_{pred}
\]

---

# 25. 是否必须进行 Stage 1

不是必须。

为了快速验证，可以直接：

```text
RLT + Dynamic
        ↓
Online RL
        ↓
L_RL + λ L_pred
```

如果训练成本允许，建议比较：

```text
C1: RL only
C2: RL + Future Prediction
C3: Future Prediction → RL
```

这样可以判断预测任务到底是在：

- 帮助初始化；
- 提供持续表示约束；
- 还是本身没有贡献。

---

# 26. 实验矩阵

第一轮实验建议严格控制变量。

| 实验 | 模型 | Dynamic | Future Prediction |
|---|---|---:|---:|
| A | Original RLT | ✗ | ✗ |
| B | RLT + GRU | ✓ | ✗ |
| C | RLT + Dynamic | ✓ | ✓ |
| D | RLT + Dynamic，仅 SSL | ✓ | ✓ |
| E | History-z Fusion | ✗ | ✗ |

其中：

### A：Baseline

原始 RLT。

### B：Recurrence Baseline

只加入 GRU：

\[
h_t=GRU(F_t,h_{t-1})
\]

不加入预测 loss。

用于证明：

> 性能提升是否仅来自 recurrent state。

### C：完整方案

\[
h_t=U(h_{t-1},F_t)
\]

并：

\[
h_t\rightarrow z_{t+1},z_{t+4},z_{t+8}
\]

同时用于 Actor/Critic。

### D：Representation Learning

只训练动态预测任务。

用于分析：

> Dynamic State 是否真的学习到了未来状态信息。

### E：历史拼接

作为此前方案的负对照：

\[
z_t^{RL}=[z_t,z_{t-1},...,z_{t-k}]
\]

---

# 27. 最重要的 Ablation

最重要的不是比较很多网络，而是验证下面这个假设：

```text
History Storage
      vs
Streaming Dynamic Representation
```

即：

```text
历史 z 拼接
        ↓
直接提供历史信息

Streaming Dynamic
        ↓
历史信息
   ↓
动态压缩
   ↓
预测相关表示
   ↓
RL Token
```

如果结果表现为：

```text
History-z:
    无明显提升 / 计算增加

Streaming Dynamic:
    收敛更快 / 最终性能提高
```

那么实验就能够支持一个明确的结论：

> **在线 RL 所需要的并不是更多历史观测，而是经过动态建模后的历史信息。**

---

# 28. 收敛速度作为主要指标

由于该方案的主要目标是改善 online RL 的学习效率，不应只比较最终 success rate。

建议至少记录：

### 1. Final Success Rate

训练结束后的最终成功率。

### 2. Environment Steps to Threshold

例如：

```text
Steps@50%
Steps@70%
Steps@80%
```

即达到指定成功率所需要的 environment steps。

### 3. Area Under Learning Curve

比较整个训练过程中的：

\[
AUC=\int Success(step)d(step)
\]

### 4. Training Throughput

记录：

```text
env steps / second
gradient steps / second
```

### 5. Inference Latency

比较：

```text
RLT
RLT + History
RLT + Dynamic
```

---

# 29. 计算复杂度目标

C方案必须满足：

\[
Cost_t \approx constant
\]

而不是：

\[
Cost_t\propto HistoryLength
\]

因为：

```text
t = 10
t = 100
t = 1000
```

都只需要：

```text
F_t
h_{t-1}
```

因此动态状态不会产生不断增长的计算量。

---

# 30. 失败模式

## 30.1 Dynamic State 退化成当前帧特征

如果：

\[
h_t\approx f(F_t)
\]

说明模型没有真正利用历史。

检查方法：

- 打乱 trajectory 顺序；
- 将 `h_{t-1}` 替换为零；
- 比较 prediction loss；
- 比较 Actor/Critic 性能。

---

## 30.2 Dynamic State 过度依赖历史

可能导致：

```text
环境突然变化
        ↓
h_t 更新太慢
        ↓
策略反应滞后
```

因此需要 gated update。

可以增加：

\[
h_t=(1-\alpha_t)h_{t-1}
+\alpha_t\tilde h_t
\]

其中：

\[
\alpha_t=\sigma(g(F_t,h_{t-1}))
\]

这样当当前观测发生重大变化时，模型可以快速刷新动态状态。

---

# 31. Optional：Adaptive Update Gate

第二版可以加入：

```python
alpha = torch.sigmoid(
    update_gate(
        torch.cat(
            [
                current_feature,
                previous_state
            ],
            dim=-1
        )
    )
)

h = (
    (1 - alpha) * h_prev
    + alpha * h_candidate
)
```

直观上：

```text
静态环境
    ↓
alpha 小
    ↓
保留已有动态状态

状态突然变化
    ↓
alpha 大
    ↓
快速更新
```

该机制尤其适合 ManiSkill 中存在明显接触、碰撞、物体运动或阶段切换的任务。

---

# 32. 后续版本：Codec-style Dynamic

如果 C 方案验证有效，可以进一步向 codec-style representation 演进。

核心思想不是直接计算：

\[
I_t-I_{t-1}
\]

而是学习：

```text
Previous Dynamic State
        +
Current Feature
        ↓
Motion / Change Representation
        ↓
Prediction
        +
Residual
        ↓
New Dynamic State
```

形式：

\[
v_t=M_\theta(h_{t-1},F_t)
\]

预测当前 latent：

\[
\hat F_t=Warp(h_{t-1},v_t)
\]

残差：

\[
R_t=F_t-\hat F_t
\]

然后：

\[
h_t=U(h_{t-1},v_t,R_t)
\]

最终压缩成固定大小：

\[
h_t\in\mathbb{R}^{16\times256}
\]

这里的 `v_t` 不要求是传统 optical flow，而是：

> **对控制任务有意义的 latent motion representation。**

这一版本属于 C 方案之后的 V2，不建议第一轮实验直接加入。

---

# 33. 为什么先做 C，再做 Codec

C 方案更适合作为第一阶段验证，因为它只回答一个核心问题：

> **流式动态表示能否帮助 RLT online RL？**

如果 C 有收益，再继续研究：

```text
Dynamic
   ↓
Motion decomposition
   ↓
Motion + Residual
   ↓
Codec-style Dynamic Representation
```

否则同时加入 motion field、warp、residual 等结构，会很难判断性能变化究竟来自哪里。

---

# 34. 推荐最终实现路线

```text
Step 1
Original RLT
    ↓
建立稳定 baseline

Step 2
RLT + GRU
    ↓
验证 recurrent state 是否有效

Step 3
RLT + Transformer Dynamic
    ↓
固定大小 h_t

Step 4
加入 Future Prediction
    ↓
z(t+1), z(t+4), z(t+8)

Step 5
加入 Gated Fusion
    ↓
z_RL = z_app + g · dynamic

Step 6
完整 Online RL
    ↓
比较 convergence

Step 7
Codec-style Dynamic
    ↓
Motion + Residual
```

---

# 35. 最终模型定义

完整 C 方案可以总结为：

### Visual Representation

\[
F_t=E_{VLA}(I_t)
\]

### Appearance RL Token

\[
z_t^{app}=E_{RLT}(F_t)
\]

### Streaming Dynamic State

\[
h_t=U_\theta(h_{t-1},F_t)
\]

### Future Prediction

\[
\hat z_{t+k}=P_k(h_t),
\quad k\in\{1,4,8\}
\]

### Gated Fusion

\[
z_t^{RL}
=
z_t^{app}
+
g_tW_hPool(h_t)
\]

### RL

\[
a_t\sim\pi_\phi(a|z_t^{RL})
\]

### Optimization

\[
L
=
L_{RL}
+
\lambda_{pred}L_{pred}
\]

---

# 36. 一句话总结

C方案不是给 RLT 增加一个“历史缓存”，而是给 RLT 增加一个：

> **逐时刻读取当前视觉、递归压缩历史变化、并通过未来预测约束的固定维度动态状态。**

因此：

```text
过去：
I1 → I2 → I3 → I4 → ...
         ↓
     大量历史信息

C方案：
I1 → I2 → I3 → I4 → ...
 ↓    ↓    ↓    ↓
h1 → h2 → h3 → h4 → ...
              ↓
        Dynamic State
              ↓
        RL Token
              ↓
        Actor / Critic
```

整个在线过程始终只需要：

```text
当前观测 I_t
上一动态状态 h_{t-1}
```

而不需要读取或保存一个不断增长的历史窗口。

---

## 37. 实现时的最小可行版本（MVP）

如果目标是尽快在 ManiSkill 上验证，应先只实现以下部分：

```text
Frozen VLA
    ↓
F_t
 ┌──┴──────────────┐
 ↓                 ↓
RLT Encoder      GRU
 ↓                 ↓
z_app             h_t
 └───────┬─────────┘
         ↓
    Gated Fusion
         ↓
       z_RL
         ↓
    Actor / Critic
```

第一轮不要加入：

```text
× optical flow
× image reconstruction
× explicit frame buffer
× long-term memory
× retrieval
× motion field
```

先回答最关键的实验问题：

> **一个固定大小、无显式历史缓存的 Streaming Dynamic State，能否让 RLT 的 online RL 收敛更快？**

如果答案为肯定，再继续增加 Future Prediction 和 Codec-style Dynamic。
