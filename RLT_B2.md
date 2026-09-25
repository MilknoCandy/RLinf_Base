# B2 及后续方案：实验记录与问题

本文记录 RLT Stage-2 在 Peg Insertion 上尝试的 **B2（encoder 当记忆）** 以及之后的 dump/SFT、在线 Loop RL、progress critic，以及各自遇到的问题。设计原稿见 `RLT_MEM.md`。**B2 与 progress 相关代码已撤回**，当前主路径回到原版 RLT-AC。

---

## 1. 结论

B2 想解决的问题是：原版 Stage-2 每步只压缩当前帧，关键相 residual 短视。做法是不另建 memory bank，把 **prefix encoder 沿 chunk 循环**，并用自然语言反馈进下一步 VLM。

实践上这条线没有超过原版 RLT：

- Dump 先用裸 VLA、后又把反馈句塞进已训练 RLT，两次都不是同分布轨迹；Loop SFT 用距离/成败头当监督，教的是预报进度而不是压缩历史，且 \(I_t\) 里根本没有反馈句。
- 在线 Loop RL 比原版更重，且把 VLM / encoder 推离 Stage-1 分布；Peg Insertion 关键段短，单帧 \(z\) 往往够用。
- 反馈句进 VLM 后语言 hidden 被丢掉，语义进不了 loop，也进不了 actor。
- 把距离 / 成功拼进 critic 状态，是把答案写进 \(s\)，SAC 的 RL 项空转。

Actor-critic 不会像 Decision Transformer 那样自动用记忆。若再做历史，状态必须定义在 \(h\) 上，\(\pi\) 与 \(Q\) 都吃 \(h\)，稠密信号放在奖励（势函数）而不是状态里。

---

## 2. B2 要做什么

原版 Stage-2：冻住 Stage-1 的 VLA + RLT encoder，每步得到当前帧 \(z\) 与 `ref_chunk`，小 MLP 只在关键相做残差微调。历史不进同一套压缩。

B2 的核心判断：记忆模块就是现成的 prefix encoder，不另建 bank / Loop Transformer。被迁移的是 encoder 如何把当前视觉与上一时刻表征压在一起；换任务只换 instruction。

每步路径：

1. 当前图 + instruction + **上一 chunk 的自然语言反馈** 进冻结 VLM。
2. 用最后一层 **文本→图像注意力 TopK 50%** 筛 image token \(I_t\)。
3. Encoder：\(z_t=\mathrm{Enc}([I_t,\,z_{t-1}])\)（\(t=0\) / done 用原来的可学习 RL token）。
4. MLP 用 \(z_t\) 在关键相对 `ref_chunk` 做精细调整。
5. 环境给出成败、距离变化，写成短句，供下一步 VLM 使用。

职责拆分：

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| VLM | instruction 与反馈的语言理解；用最后一层文本注意力筛选 image | 不直接出精细动作 |
| Prefix encoder | 把筛选后的 \(I_t\) 与 \(z_{t-1}\) 压成 \(z_t\) | 不是 memory bank |
| MLP 策略头 | 用 \(z_t\) 微调 `ref_chunk` | 不替代 VLM 的场景理解 |

\(z\) 不是独立存储的 memory content，只是下一轮 prefix 里那个 RL token。

Peg Insertion 先验证能否不崩、是否更准；CALVIN 再验证同场景换 instruction。

---

## 3. 之后实际试过的解法

按时间顺序，不是按最终推荐顺序。

### 3.1 B2-0 / B2-1：推理 Loop

接线：反馈句生成、TopK、\(z\) 回注、`done` 时重置为 \(z_{\mathrm{init}}\)。Encoder 仍冻着，只有 MLP 在训。目标是先确认 Peg Insertion 上 Loop 不崩。

### 3.2 Dump + Loop SFT（B2-2）

动机：冻结的 Stage-1 encoder 从没见过 \(z_{t-1}\)，直接在 Stage-2 里开 Loop **不可能**有历史语义。必须先用 SFT 教会 encoder「怎么把当前视觉和上一步 \(z\) 压在一起」，RL 再教「怎么用这份 \(z\) 做关键相微调」。因此 B2-2 是两段离线工序，**不是**原来的 Stage-1 重建 SFT，也**不要**改 `OpenPiPytorchSFTActionModel` 的 MSE。

三条线不要混：已训好的是**原版 RLT**；Dump 只沿这条轨迹采数据；Loop 只发生在离线 encoder 上。

| | Dump | Loop SFT | 之后的 B2 RL |
| --- | --- | --- | --- |
| 谁在跑 | 已训练 Stage-1 + Stage-2 | 新 encoder + 读出头 | encoder + MLP |
| VLM | 有，**只有 instruction** | 无 | 有，instruction + 反馈句 |
| Loop \(z_{t-1}\) | **无** | 有，离线截断 BPTT | 有，在线 \(z_{t-1}\) detach |
| MLP / `ref_chunk` | 有，用来在环境里执行 | 无 | 有，用来学习 |
| 写出什么 | 每条 episode 的 \(I_t,d,s\) | encoder ckpt | 策略 ckpt |

顺序只能是 dump → Loop SFT → B2 RL。反馈句**只在 B2 RL** 才进 VLM。

#### Dump：一次、eval、不训练

目的：沿**已经训好的原版 RLT 轨迹**，把下一步 Loop SFT 要用的 encoder 输入 \(I_t\) 存下来。不改策略、不加反馈句、不做 Loop。LeRobot 帧没有 `peg_head_hole_*`，不能当 dump 源；必须在 `ManiskillRLTEnv` 里跑。

每一步的计算路径（修正后的口径）：

```
instruction-only VLM prefix（与 Stage-1/2 同分布）
    ├─ Encoder(静态 z_init) → z          } 执行：完整 RLT
    ├─ VLA → ref_chunk                   }
    └─ 另取 TopK 50% image tokens → I_t    只落盘，不改变上面两条

MLP(ref_chunk, z, proprio) → 残差
rlt_route：非关键段执行 ref_chunk，关键段执行 MLP
env.step → 特权 d_t、s_t
落盘：I_t, mask, d_t, s_t
```

`.pt` 里**只有** TopK 后的 image token \(I_t\)、mask、peg–hole 距离、成败。不存 \(z\)、动作、`ref_chunk`、prompt。训练时 \(z\) 会重算；\(z\) 若落盘只作 sanity check。

两件明确不能做：

1. **不能只跑裸 VLA。** Stage-1 的 VLA 前向**不吃** RL token。若 dump 只加载 Stage-1、用未条件于 \(z\) 的 VLA 去 step，存下的不是「RLT 每个时刻的输出」：没有 \(z\)、没有 MLP residual，轨迹也不是 RLT 策略走出来的。必须同时加载 Stage-1 feature model **和** Stage-2 MLP，走 `rlt_route`。
2. **不能为了 dump 把反馈句拼进 VLM，也不能在 dump 里开 Loop。** 已训练的 Stage-1/2 从未见过反馈句和 \(z_{t-1}\)。硬塞会让 `ref_chunk` 和 MLP 都偏出分布，轨迹变差，后面的 \(I_t\) 也废。Loop 是 SFT 阶段才在 encoder 上滚的。

Dump 之后不再跑 VLM、不再进仿真。配置曾为 `maniskill_rlt_b2_dump.yaml`。

#### Loop SFT：反复 epoch，只训 encoder

目的：让 encoder 学会 \(z_t=\mathrm{Enc}([I_t,\,z_{t-1}])\)。不再跑 VLM，不仿真，不动 MLP / VLA。从 Stage-1 ckpt 初始化 encoder（含 `rl_token_embed`），关掉重建 decoder，加上 \(z\to\hat d\)、\(z\to\hat p_{\mathrm{success}}\) 两个读出头。

每个 chunk **只 encoder 一次**，不是同一帧空转 \(K\) 次。「迭代」是沿时间把 \(z\) 回注到下一步。一条 dump 轨迹：

```
z_0 = Enc([I_0, z_init])                 # 尚无历史，不算压缩损失
for t = 1 .. T-1:
    z_t = Enc([I_t, z_{t-1}])            # 第一次完整 Loop 从 t=1 开始
    L += λ_d ||d̂_t - d_t||² + λ_s BCE(p̂_t, s_t)
```

- \(t=0\) 没有回注，损失从 \(t\ge 1\) 逐步算，不是整局只在终点算一次。
- 反传截断 \(K=2\sim 4\) 个 chunk，不拉满整条 episode。
- 读出头只是教压缩的探针，**不是** Stage-2 策略。SFT 留下的是新 encoder；策略头仍是原来的 MLP。

配置曾为 `examples/sft/config/maniskill_rlt_b2_sft.yaml`，入口 `examples/sft/train_rlt_b2.py`。

这里有一个设计裂缝，后来也没补上：dump 的 \(I_t\) 来自 **instruction-only** 的 TopK，SFT 学的是「当前视觉 + 上一步 \(z\)」，**还不是**「带反馈句调制后的视觉」。反馈进 VLM 被留到 B2 RL。因此即便 dump 分布改对了，Loop SFT 也没在 B2 真正会遇到的输入上训过 encoder。

#### 实际结果

先做错了 dump（裸 VLA，见 §4.1），改对执行路径后又曾把反馈句塞进 dump（同样偏分布）。读出头本身也教错了目标（§4.2）。这条路径没有产出可用的 encoder ckpt。随后改为不要 dump，在在线 RL 里直接接着训 Stage-1 encoder。

### 3.3 在线 Loop RL（B2-3，一度作为主路径）

去掉 dump。VLM / VLA 冻结。把 Stage-1 encoder 挂到 MLP 的 `rlt_loop`，与从头初始化的 MLP 一起做 SAC。Replay 存 \(I_t\)、\(z_{\mathrm{prev}}\)、`proprio`、`ref_chunk`；actor 侧 \(z_t=\mathrm{Enc}(I_t,\,z_{\mathrm{prev}}.\mathrm{detach}())\)。反馈句只当固定提示进 VLM，不单独训语言。

配置曾为 `maniskill_rlt_stage2_b2_loop.yaml`。在 Peg Insertion 上 **未比原版 RLT 更高效**。

### 3.4 Progress 进 Critic

承认视觉 Loop 伤原版通路后，退回冻 encoder，只给 critic 一份进度向量，不把反馈句塞进 VLM：

\[
m_t=\big[\,\tanh(d_t/\mathrm{scale}),\;\tanh(\Delta d_t),\;\mathbb{1}_{\mathrm{success}},\;\mathbb{1}_{\mathrm{has}}\big]
\]

\(Q\) 输入改为 \(\mathrm{cat}(z,\,\mathrm{proprio},\,m)\)；actor 第一版仍不看 \(m\)。奖励仍是 `only_success`，不用 \(\Delta d\) 改环境奖励。

能接到现有 SAC / replay，但 RL 上是错的（§4.5）。100 epoch 时 `progress_d` 掉不到 0.8 以下，起初被当成没训够，实际是特征饱和 + 抄标签。

### 3.5 文献对照与撤回

对照 RATE 等 memory RL：RATE 用的是 **Decision Transformer**，损失直接预测 \(a\mid\) 历史，记忆在计算图上。Actor-critic 没有这条路。随后撤回全部 B2 与 progress 文件，主路径回到原版 Stage-2。

---

## 4. 遇到的问题

### 4.1 Dump 两次采错分布

第一刀（致命）：dump 只挂了 Stage-1。Stage-1 的 VLA **不吃** RL token，前向就是基础 Pi0。于是环境里走的是裸 VLA，落盘的 \(I_t\) 也不是 RLT 条件 prefix 上筛出来的：没有 \(z\)、没有 MLP residual，更谈不上「RLT 每个时刻的输出」。

第二刀：执行改成 Stage-1+Stage-2 之后，又把反馈句拼进 dump 的 VLM。已训练模型没见过这句话，`ref_chunk` 和 MLP 一起 OOD，轨迹变差，\(I_t\) 同样废。

正确口径见 §3.2：dump 必须像原版 Stage-1/2 那样跑（instruction-only、静态 \(z_{\mathrm{init}}\)、完整 `rlt_route`），TopK 只从这次同分布 prefix 另取一份落盘。

### 4.2 Loop SFT 的监督不是「会压缩历史」

读出头是 \(z_t\to d_t\)、\(z_t\to s_t\)。encoder 最短的解是当进度预报器：当前 \(I_t\) 里 peg–hole 几何已经够预测 \(\hat d\)，不必真正使用 \(z_{t-1}\)。要教的是「压缩历史给下一步精细调整」，探针却在考「这一帧还多远、成没成」。

即便读出头换成别的，dump 的 \(I_t\) 仍是 instruction-only TopK，SFT 的 Loop **从未**在「反馈句已调制过的 image」上训练。B2 推理时 VLM 突然多一句反馈，encoder 输入再次 OOD。

所以这条 SFT 既可能学成忽略历史，又和后续 RL 的输入对不上。没有产出可用 encoder ckpt。

### 4.3 语义进不了 loop，也进不了 action

反馈句进 VLM 后，语言 hidden states 被丢掉，只留下 TopK 图像。Loop 没有语义能力；反馈最多改「看哪块图」，**改不了 actor 输出**。硬把文本特征拼进 encoder 也没有真正的语言理解。

因此「环境状态写成 NL 再进 VLM」没有把成败 / 远近送到策略头。

### 4.4 在线 B2 比原版更差是预期

原版 Stage-2 冻住一份已经对准的 \(z\)，只训小 MLP 去残差调 in-distribution 的 `ref_chunk`。B2 同时做了三件会伤这条通路的事：

1. **VLA / encoder 都偏出 Stage-1 分布。** 反馈句 Stage-1 没见过，却进了冻结 VLM，`ref_chunk` 变差。Encoder 输入从整段 prefix（原版 `rlt_image_only: False`）改成 TopK 50% 图像，再叠从未训练过的 \(z_{t-1}\) Loop。起步 \(z\) 就不如冻住的 Stage-1 encoder。
2. **历史很难帮 Peg Insertion。** 关键段很短，当前帧 \(z\) 已经够用；\(z_{t-1}.\mathrm{detach}()\) 没有跨步 BPTT，历史压缩几乎没有收益。再加上 §4.3，loop 学不到 closer / succeeded。
3. **优化更难、每步更重。** 原版只训小 MLP。B2 是 MLP 从头训再加可训 encoder，还要存 \(I_t\)、每次重算 encode、钩最后一层 attn。同样的 SAC 日程（高 BC warmup）会把策略拉向更差的 `ref_chunk`。墙钟和样本效率都会差一截。

所以当时的 B2 不是「原版 RLT + 历史」，而是「OOD prompt + 残缺视觉 \(z\) + 更重的训练」。在同一个 ManiSkill Stage-2 设定上不赢，不是偶然没训够。

### 4.5 Progress 进 critic：把答案写进状态

`train/critic/progress_d` 是 replay 整批 \(\tanh(d/0.05)\) 的均值，不是损失，不该当判据往 0 压。\(d=\sqrt{y^2+z^2}-x_{\mathrm{hole}}\)。临界相门外未插入时 \(x_{\mathrm{hole}}\) 常为负，即使 yz=0 也有约 16 cm，\(\tanh(0.16/0.05)\approx 0.997\)。均值要低于 0.8，需要大批样本已经插到 \(d<5.5\,\mathrm{cm}\)。100 epoch 大致刚过 warmup，buffer 里主要是抓取、靠近，再加 `scale=0.05`，均值会钉在 0.8 以上。

更严重的是 RL 缺陷。当前

\[
m_t=\big[\tanh(d/0.05),\;\tanh(\Delta d),\;\mathrm{success},\;\mathrm{has}\big]
\]

奖励又是 `only_success`。于是 \(Q(z,\mathrm{proprio},m,a)\) 可以学成：success=1 则 \(Q\) 高，其余步 \(\tanh(d)\) 饱和在约 1，等于常数。Bellman 目标在这种输入上可抄：下一状态若带 success，target 就高。critic 不必让 \(Q\) 依赖 \(a\)，也不必依赖 \(z\)。SAC 的 actor 目标是 \(\max_a Q(s,a)-\mathrm{BC}\)；\(Q\) 不看 \(a\) 时 \(\nabla_a Q\approx 0\)，**RL 项空转，只剩跟 `ref_chunk`**。`q_success_gap>0` 是抄标签，不是学到了进度。

非对称 critic 的特权应是马尔可夫状态（位姿、接触），用来补全 POMDP，不是把结果本身（成功位、已饱和的到目标距离）喂进去。结果在 \(s\) 里，价值函数退化成对结果的查找表。先训 critic 再给 actor 修不好：\(Q\) 已经不给动作排序。

尺度问题（`scale=0.05`）只是同一缺陷的症状：未插入的典型 \(d\) 全部顶格，\(m\) 几乎没有「更近 / 更远」的可分坐标。

### 4.6 Actor-critic 不会自动用记忆

RATE 一类工作的记忆有效，是因为 **Decision Transformer** 的损失就是 \(\mathcal{L}(\hat a, a\mid R, o_{\le t}, M)\)，记忆 token 在计算图上，动作监督会直接训练「该记什么」。

SAC 的梯度来自 TD 和 \(-Q\)。若 \(s\) 仍是 \((z_t,\,\mathrm{proprio},\,\mathrm{ref\_chunk})\) 且各步独立采样，没有 \(h\)，记忆既不进价值也不进策略。若只把 \(h\) 拼给 actor、critic 仍看单帧，\(-Q\) 按单帧排序，记忆无用。若把 success / \(d\) 写进 \(s\)，\(Q\) 抄标签，记忆同样无用。

单步 i.i.d. replay 还会把 \(h\) 的时间结构拆掉。

### 4.7 实现层面的次要风险（设计稿里已写、实践中碰到）

- 钩 Gemma 最后一层 attn 必须只开 B2，避免打断现有 eager prefix cache。
- 反馈句拉长 prompt，会改变 KV cache 与 `max_token_len` 预算，句子必须短。
- Stage-2 encoder 挂在 MLP 的 `rlt_loop` 上（不要叫 `encoder`，以免进 critic 优化器），经现有 `hf_model` weight_sync 同步到 rollout。

---

## 5. 撤回时留下的判断

Peg Insertion 上不要再把 success / \(\tanh d\) 当记忆，也不要先上视觉 Loop。

若还要做历史，Actor-critic 侧成立的条件是：

\[
h_{t+1}=f(h_t,o_t,a_t),\quad
\pi(a_t\mid h_t,o_t),\quad
Q(h_t,a_t)
\]

具体约束：

1. \(\pi\) 和 \(Q\) **都吃** \(h\)。非对称只允许特权位姿进 critic，禁止 success 进 \(s\)。Peg 上更干净的是对称：两边都只看 \(h\)。
2. TD 必须沿记忆走：\(Q(h_t,a_t)\approx r_t+\gamma Q(h_{t+1},a_{t+1})\)，episode / scene `done` 时切断并重置 \(h\)。
3. 稀疏 `only_success` 几乎训不动 \(f\)。不要把 \(d\) 放进 \(h\)，用势函数只改奖励：

   \[
   r'_t=r_t+\gamma\Phi(s_{t+1})-\Phi(s_t),\quad \Phi=-d
   \]

   这是 Ng 塑形，最优策略对原 \(r\) 不变；哪次更新让 \(d\) 变小，TD 立刻有信号。
4. Replay 用短段（Peg：2–8 个 action chunk）或 burn-in，不要单步 i.i.d.。\(f\) 用门控写，encoder / \(f\) 单独更小的学习率。写入 replay 的 \(h_t\) 对 \(f\) detach，反传只走本段 unroll。
5. 可选辅助（下一帧 / proprio 预测）可以让 \(h\) 在尚未成功时先有内容，但辅助梯度要小，并且 stop-grad 进 \(Q\) 的支路，避免 \(Q\) 抄预测头。

冻结 VLM，可训的只有 \(f\)（loop / GRU / 门控写）和 Stage-2 MLP。`ref_chunk` 仍留给 BC。这样：积累在 \(h_{t+1}=f(\cdot)\)，利用在 \(Q(h,a)\) 与 \(\pi(a\mid h)\)，训练信号是塑形后的 TD，不是 oracle 标签。

---

## 6. 与现有文档、代码的关系

| 文档 / 代码 | 现状 |
| --- | --- |
| `RLT_MEM.md` | B2 设计原稿（目标、公式、接线、风险），不是实验结论 |
| `RLT_STAGE2.md` | 原版 Stage-2 AC 架构 |
| B2 / progress 实现 | 已撤回（`b2_*.py`、`progress.py`、`rlt_b2_*`、对应 yaml / 单测） |
| 当前训练配置 | `examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml`（原版 RLT-AC） |
