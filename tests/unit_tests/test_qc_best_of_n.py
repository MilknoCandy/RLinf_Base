# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from rlinf.algorithms.qc import (
    QC_LOSS_TYPE,
    build_qc_critic_candidates,
    flatten_chunk_actions,
    is_rlt_stage2_loss,
    repeat_obs,
    select_best_of_n_actions,
    validate_qc_n_step,
)


def test_repeat_obs_repeats_batch_and_keeps_row_alignment():
    obs = {"z_rl": torch.arange(4).reshape(2, 2).float()}
    repeated = repeat_obs(obs, 3)
    assert repeated["z_rl"].shape == (6, 2)
    assert torch.equal(repeated["z_rl"][0], obs["z_rl"][0])
    assert torch.equal(repeated["z_rl"][1], obs["z_rl"][0])
    assert torch.equal(repeated["z_rl"][3], obs["z_rl"][1])


def test_flatten_chunk_actions_accepts_sequence_and_flat():
    chunk = torch.arange(2 * 5 * 7).reshape(2, 5, 7).float()
    flat = flatten_chunk_actions(chunk, chunk_len=5, action_dim=7)
    assert flat.shape == (2, 35)
    assert torch.equal(
        flatten_chunk_actions(flat, chunk_len=5, action_dim=7),
        flat,
    )


def test_build_qc_critic_candidates_keeps_source_order():
    ref_chunk = torch.zeros(2, 4)
    actor_mean = torch.ones(2, 4)
    actor_samples = torch.full((2, 3, 4), 2.0)
    stacked = build_qc_critic_candidates(
        actor_samples=actor_samples,
        ref_chunk=ref_chunk,
        actor_mean=actor_mean,
    )
    assert stacked.shape == (2, 5, 4)
    assert torch.equal(stacked[:, 0], ref_chunk)
    assert torch.equal(stacked[:, 1], actor_mean)
    assert torch.equal(stacked[:, 2:], actor_samples)


def test_select_best_of_n_actions_picks_highest_q():
    candidates = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            [[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]],
        ]
    )
    q_values = torch.tensor([[0.1, 0.9, 0.2], [0.3, 0.2, 0.8]])
    best, indices = select_best_of_n_actions(q_values, candidates)
    assert indices.tolist() == [1, 2]
    assert torch.allclose(best[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(best[1], torch.tensor([7.0, 7.0]))


def test_validate_qc_n_step_rejects_biased_multi_chunk_backup():
    validate_qc_n_step(1)
    with pytest.raises(ValueError, match="n_step=2"):
        validate_qc_n_step(2)


def test_rlt_qc_is_treated_as_stage2_loss():
    assert QC_LOSS_TYPE == "rlt_qc"
    assert is_rlt_stage2_loss("rlt_qc")
    assert is_rlt_stage2_loss("rlt_ac")
    assert not is_rlt_stage2_loss("embodied_sac")


class _FakeTwinQPolicy:
    """Score Q as the mean action so the larger candidate wins."""

    def __init__(self, ref_chunk: torch.Tensor, actor_chunk: torch.Tensor):
        self.ref_chunk = ref_chunk
        self.actor_chunk = actor_chunk

    def __call__(self, forward_type, obs, actions=None, deterministic=False, **kwargs):
        from rlinf.models.embodiment.base_policy import ForwardType

        batch = obs["z_rl"].shape[0]
        if forward_type == ForwardType.SAC:
            chunk = self.ref_chunk if deterministic else self.actor_chunk
            if chunk.shape[0] == 1 and batch > 1:
                chunk = chunk.expand(batch, -1)
            return chunk, torch.zeros(batch, 1), None
        if forward_type == ForwardType.SAC_Q:
            q = actions.mean(dim=-1, keepdim=True)
            return torch.cat([q, q], dim=-1)
        raise AssertionError(f"unexpected forward_type={forward_type}")


def test_qc_critic_mixin_selects_higher_q_candidate():
    from omegaconf import OmegaConf

    from rlinf.workers.actor.fsdp_qc_policy_worker import QCCriticMixin

    class _Harness(QCCriticMixin):
        def __init__(self):
            self.cfg = OmegaConf.create(
                {
                    "algorithm": {
                        "qc_num_samples": 0,
                        "qc_include_ref_chunk": True,
                        "qc_include_actor_mean": True,
                    },
                    "actor": {"model": {"num_action_chunks": 2, "action_dim": 2}},
                }
            )
            self.model = _FakeTwinQPolicy(
                ref_chunk=torch.zeros(1, 4),
                actor_chunk=torch.ones(1, 4),
            )

        def _chunk_shape(self):
            return 2, 2

        def _flatten_chunk(self, tensor):
            return tensor.reshape(tensor.shape[0], -1)

        def _ref_chunk(self, obs):
            return obs["ref_chunk"].reshape(obs["ref_chunk"].shape[0], -1)

        def _min_twin_q(self, all_q_values):
            return torch.minimum(all_q_values[..., 0:1], all_q_values[..., 1:2])

    mixin = _Harness()
    next_obs = {
        "z_rl": torch.zeros(1, 3),
        "proprio": torch.zeros(1, 2),
        "ref_chunk": torch.zeros(1, 2, 2),
    }
    best, _, _ = mixin._next_actions_for_critic_target(next_obs)
    assert torch.allclose(best, torch.ones(1, 4))
    assert mixin._last_qc_metrics["qc/selected_mean_rate"] == 1.0
    assert mixin._last_qc_metrics["qc/num_candidates"] == 2.0
