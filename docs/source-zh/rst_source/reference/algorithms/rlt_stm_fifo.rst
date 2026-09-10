RLT + 短期记忆（STM-FIFO）
=========================

1. 简介
-------

RLT + STM-FIFO 是 RLT 第二阶段 actor-critic 微调的带记忆变体。RLT 已经将多模态观测压缩为紧凑的 RL token :math:`z_t`，并在其上学习一个小型 actor-critic 头。STM-FIFO 进一步为每个环境维护一段最近的 RL-token 经验，使 actor 和 critic 不再只依赖当前观测，而能显式利用近期轨迹上下文。

这是层次化记忆研究路线中的第一个消融：

.. code-block:: text

   RLT
   -> RLT + STM-FIFO
   -> RLT + STM-FIFO + LTM
   -> RLT + RL-trained STM
   -> RLT + RL-trained STM + LTM

2. 方法
-------

记忆单元定义为

.. math::

   e_t = (z_t, a_t, r_t, z_{t+1}),

其中 :math:`z_t` 是 RL token，:math:`a_t` 是已执行的动作块，:math:`r_t` 是块级奖励，:math:`z_{t+1}` 是下一状态的 RL token。短期记忆保存最近 :math:`K` 条经验的 FIFO：

.. math::

   S_t = \{e_{t-K+1}, \dots, e_t\}.

在 :math:`t` 时刻，记忆读取器输出固定维度特征 :math:`m_t`。Phase 1 使用确定性的均匀读取器：

.. math::

   m_t = [\mathrm{mean}(z_i), \mathrm{mean}(r_i), r_{\mathrm{last}}],

均值在当前 :math:`S_{t-1}` 中的经验上计算。记忆向量与当前 RL token 拼接，从而保持 RLT 的核心设计，即 RL token 仍是决策中间表示：

.. math::

   z'_t = [z_t, m_t].

第二阶段策略随后使用：

- actor 输入：:math:`[\mathrm{ref\_chunk}, z_t, m_t, \mathrm{proprio}]`
- critic 输入：:math:`[z_t, m_t, \mathrm{proprio}]`

3. 因果性与优势
---------------

记忆更新在 rollout worker 中完成，确保策略预测 :math:`a_t` 时已经能够使用 :math:`z_{t+1}`：

.. code-block:: text

   env worker -> rollout worker:
       obs_t + (last_reward, last_done)

   rollout worker:
       z_t = extract_rlt_obs(obs_t)
       用 z_t 补全 e_{t-1}
       m_t = read(S_{t-1})
       a_t = policy([z_t, m_t])
       将 (z_t, a_t, actor_switch_t) 记为 pending

优势 :math:`A_t` 无法在 rollout 时因果地获得，因此 Phase 1 使用奖励作为 reward-conditioned 信号；优势条件化记忆留待后续 RL 训练记忆阶段。

4. 实现
-------

主要组件如下：

- ``rlinf/algorithms/rlt/stm.py``
  实现 :class:`RLTSTMFIFO`，提供每个环境的 FIFO，以及 ``complete_and_retrieve`` 和 ``remember_current``。
- ``rlinf/algorithms/rlt/rollout.py``
  在第二阶段推理前将 ``stm_memory`` 挂到 RL-token 观测上，并写入 transition observation。
- ``rlinf/workers/rollout/hf/huggingface_worker.py``
  持有独立的 train/eval STM 实例，并传递环境 worker 返回的逐步 reward/done 上下文。
- ``rlinf/workers/env/env_worker.py``
  发送上一动作块的 ``last_reward`` 与 ``last_done``。
- ``rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py``
  接收 ``stm_memory_dim``，并将记忆特征拼接到 actor 与 critic 状态。

只有 RLT actor 实际控制动作的 transition（``actor_switch=True``）才会写入记忆，从而避免 ManiSkill warmup 或 base-policy 动作块污染短期记忆。

5. 配置
-------

ManiSkill 示例为
``examples/embodiment/config/maniskill_rlt_stm_fifo_stage2_ac_mlp.yaml``。算法块开启记忆：

.. code-block:: yaml

   algorithm:
     rlt_stm:
       enable: True
       capacity: 4

第二阶段模型设置记忆维度：

.. code-block:: yaml

   actor:
     model:
       model_type: "rlt_mlp_policy"
       z_dim: 2048
       # z_dim(2048) + mean_reward(1) + last_reward(1)
       stm_memory_dim: 2050

``rollout.model.stm_memory_dim`` 与 ``actor.model.stm_memory_dim`` 保持一致。

6. 当前范围
-----------

- 目前只在 ManiSkill RLT Stage-2 上实现。
- 记忆读取器固定且确定（对 RL token 和奖励标量做均值池化）。
- episode 终止时，以及环境 worker 在未携带 reward/done 上下文的 rollout 首个观测上重置记忆。
- eval 与训练使用相同的 STM 机制，保证训练/评测观测分布一致。
