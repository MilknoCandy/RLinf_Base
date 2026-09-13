# RLT + Fixed Short-term Memory (Step 2A)

## Scope

This documents the implemented **Step 2A** of `RLT_Hierarchical_RL_Memory_Step2_Design.md`.

Step 2A keeps the VLM / RLT representation and the RL-token encoder frozen and only
adds a trainable short-term memory head:

```text
o_t -> frozen RLT -> z_t -> STM read -> m_t -> z'_t = z_t + W_m m_t -> Actor/Critic
```

The memory entry is `e_t = (z_t, a_t, r_t)`. The reader is a fixed recent window of
size `K`; the encoder is a GRU; the fusion is residual. Because `W_m` is
zero-initialized, the model degenerates exactly to the RLT baseline.

## Entry points

RLinf does not have a `train_sync.py` entry. RLT Stage 2 has two official embodied
entries:

- `examples/embodiment/train_async.py` — async / disaggregated pipeline. Rollout and
  training overlap, which is the efficient mode for RLT Stage 2.
- `examples/embodiment/train_embodied_agent.py` — sync entry. Rollout and training run
  serially. Both entries dispatch to off-policy SAC (`loss_type: rlt_ac`) with the same
  replay buffer, so neither is on-policy online PPO.

The commands below use the async entry for throughput. The sync configs
(`maniskill_rlt_stage2_ac_mlp.yaml`, `maniskill_rlt_stage2_stm_ac_mlp.yaml`) remain
available for parity with the sync RLT baseline.

## Implementation layout

- `rlinf/algorithms/rlt/memory/buffer.py` — `STMBuffer` (read / write / reset, recent /
  shuffled / random retrieval, `(z, a, r)` entry composition).
- `rlinf/algorithms/rlt/memory/encoder.py` — `GRUMemoryEncoder`.
- `rlinf/algorithms/rlt/memory/fusion.py` — `ResidualMemoryFusion`.
- `rlinf/algorithms/rlt/memory/module.py` — `RLTSTMModule` and `build_memory_module`.
- `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` — Actor and Critic both use the
  fused token `z'_t`.
- `rlinf/workers/rollout/hf/huggingface_worker.py` and
  `rlinf/workers/env/env_worker.py` — per-env STM buffer lifecycle during rollout/eval.
- `examples/embodiment/config/maniskill_rlt_stage2_stm_ac_mlp_async.yaml` — runnable
  async config; `maniskill_rlt_stage2_stm_ac_mlp.yaml` is the sync equivalent.

## Run

Baseline (RLT without memory):

```bash
python examples/embodiment/train_async.py \
  --config-name maniskill_rlt_stage2_ac_mlp_async
```

Step 2A main (RLT + fixed STM, recent window, `K=8`, `(z,a,r)` entries):

```bash
python examples/embodiment/train_async.py \
  --config-name maniskill_rlt_stage2_stm_ac_mlp_async
```

## Progressive ablation matrix

The same config supports the Step 2A matrix via Hydra overrides.

| Experiment | Overrides | Question |
|---|---|---|
| A. RLT baseline | `maniskill_rlt_stage2_ac_mlp_async` | reference |
| B. RLT + recent STM | `maniskill_rlt_stage2_stm_ac_mlp_async` | does recent history help? |
| B-K4 / B-K16 | `algorithm.memory.window_size=4` / `=16` | window sensitivity |
| C. random context | `algorithm.memory.retrieval=random` | rules out added parameters/context |
| D. shuffled context | `algorithm.memory.retrieval=shuffled` | does temporal order matter? |
| E. state-only entry | `algorithm.memory.entry.action=False algorithm.memory.entry.reward=False` | is `(a,r)` needed? |

For example:

```bash
python examples/embodiment/train_async.py \
  --config-name maniskill_rlt_stage2_stm_ac_mlp_async \
  algorithm.memory.window_size=4
```

## Causality

The rollout path follows `read -> act -> write`:

1. complete the previous pending `(z_{t-1}, a_{t-1})` with the previous chunk reward,
2. read `M_t = {e_0, ..., e_{t-1}}`,
3. compute the actor/critic action from `z'_t`,
4. store `(z_t, a_t)` as the next pending entry.

Episode boundaries clear the buffer, so memory never leaks across episodes or into the
current action's reward.
