# RLT + Short-Term Memory (STM-FIFO)

## 1. Introduction

RLT + STM-FIFO is a memory-augmented variant of RLT's Stage-2 actor-critic
fine-tuning. RLT already compresses multimodal observations into a compact RL
token $z_t$ and learns a small actor-critic head on top of it. STM-FIFO adds a
per-environment short-term memory over the most recent RL-token experiences, so
the actor and critic can condition on the immediate trajectory context instead
of only on the current observation.

This is the first ablation in the hierarchical-memory roadmap:

```text
RLT
-> RLT + STM-FIFO
-> RLT + STM-FIFO + LTM
-> RLT + RL-trained STM
-> RLT + RL-trained STM + LTM
```

## 2. Method

A memory unit is defined as:

$$e_t = (z_t, a_t, r_t, z_{t+1}),$$

where $z_t$ is the RL token, $a_t$ is the executed action chunk, $r_t$ is the
chunk reward, and $z_{t+1}$ is the next RL token. The short-term memory is a
FIFO of the most recent $K$ experiences:

$$S_t = \{e_{t-K+1}, \dots, e_t\}.$$

At time $t$, the memory reader produces a fixed-size feature $m_t$. Phase 1
uses a deterministic, uniform reader:

$$m_t = [\mathrm{mean}(z_i), \mathrm{mean}(r_i), r_{\mathrm{last}}],$$

where the means are taken over the experiences currently in $S_{t-1}$. The
memory vector is concatenated with the current RL token, preserving RLT's core
design that the RL token remains the decision representation:

$$z'_t = [z_t, m_t].$$

The Stage-2 policy then uses:

- actor input: $[\mathrm{ref\_chunk}, z_t, m_t, \mathrm{proprio}]$
- critic input: $[z_t, m_t, \mathrm{proprio}]$

## 3. Causality and Advantage

The memory update runs in the rollout worker so that $z_{t+1}$ is available
before the policy predicts $a_t$:

```text
env worker -> rollout worker:
    obs_t + (last_reward, last_done)

rollout worker:
    z_t = extract_rlt_obs(obs_t)
    complete e_{t-1} with z_t
    m_t = read(S_{t-1})
    a_t = policy([z_t, m_t])
    remember (z_t, a_t, actor_switch_t) as pending
```

Advantage $A_t$ is not available causally at rollout time, so Phase 1 uses
reward as the reward-conditioned signal. Advantage-conditioned memory is left
for a later RL-trained memory phase.

## 4. Implementation

The main components are:

- `rlinf/algorithms/rlt/stm.py`  
  Implements `RLTSTMFIFO`, a per-environment FIFO with `complete_and_retrieve`
  and `remember_current`.
- `rlinf/algorithms/rlt/rollout.py`  
  Attaches `stm_memory` to the RL-token observation before Stage-2 inference
  and records it in transition observations.
- `rlinf/workers/rollout/hf/huggingface_worker.py`  
  Owns separate train/eval STM instances and passes the per-step reward/done
  context returned by the environment worker.
- `rlinf/workers/env/env_worker.py`  
  Sends `last_reward` and `last_done` for the previous action chunk.
- `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py`  
  Accepts `stm_memory_dim` and appends the memory feature to actor and critic
  states.

Only transitions where the RLT actor actually controlled the action
(`actor_switch=True`) are written to memory. This keeps ManiSkill warmup and
base-policy chunks from contaminating the short-term memory.

## 5. Configuration

The ManiSkill example is
`examples/embodiment/config/maniskill_rlt_stm_fifo_stage2_ac_mlp.yaml`. The
algorithm block enables the memory:

```yaml
algorithm:
  rlt_stm:
    enable: True
    capacity: 4
```

The Stage-2 model sets the memory dimension:

```yaml
actor:
  model:
    model_type: "rlt_mlp_policy"
    z_dim: 2048
    # z_dim(2048) + mean_reward(1) + last_reward(1)
    stm_memory_dim: 2050
```

`rollout.model.stm_memory_dim` is kept identical to
`actor.model.stm_memory_dim`.

## 6. Current Scope

- Implemented for ManiSkill RLT Stage-2 only.
- Memory reader is fixed and deterministic (mean pooling over RL tokens and
  reward scalars).
- Memory is reset on episode termination and at the first observation of a new
  rollout, when the environment worker sends a reset observation without
  reward/done context.
- Eval uses the same STM mechanism as training for a consistent train/eval
  observation distribution.
