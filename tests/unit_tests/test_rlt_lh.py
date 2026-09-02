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

import torch

from rlinf.algorithms.rlt_lh.history import (
    RLTHistoryBuffer,
    build_history_summary,
    extract_rlt_lh_phase,
    extract_rlt_lh_subtask_success,
    history_summary_dim,
)
from rlinf.models.embodiment.mlp_policy.rlt_lh_mlp_policy import RLTLHMLPPolicy
from rlinf.models.embodiment.modules.rlt_history_encoder import RLTHistoryEncoder


def test_history_encoder_shape():
    encoder = RLTHistoryEncoder(input_dim=5, hidden_dim=8, num_layers=2)
    history = torch.randn(3, 4, 5)
    out = encoder(history)
    assert out.shape == (3, 8)


def test_history_summary_dim_and_buffer():
    summary_dim = history_summary_dim(
        z_dim=8, proprio_dim=4, action_dim=2, num_action_chunks=3
    )
    assert summary_dim == 8 + 4 + 2 * 3 + 3

    buffer = RLTHistoryBuffer(num_envs=3, window=4, summary_dim=summary_dim)
    summary = build_history_summary(
        z_rl=torch.randn(3, 8),
        proprio=torch.randn(3, 4),
        action=torch.randn(3, 3, 2),
        reward=torch.randn(3, 1),
        phase_idx=torch.tensor([[0.0], [1.0], [2.0]]),
        subtask_success=torch.tensor([[0.0], [1.0], [0.0]]),
    )
    assert summary.shape == (3, summary_dim)
    window = buffer.push(summary)
    assert window.shape == (3, 4, summary_dim)
    done = torch.tensor([False, True, False])
    buffer.push(torch.zeros_like(summary), done_mask=done)
    assert torch.all(buffer.buffer[1] == 0)


def test_history_progress_extraction_for_calvin():
    env_infos = {"episode": {"avg_len": torch.tensor([0.0, 1.0, 2.0])}}
    phase = extract_rlt_lh_phase(
        env_type="calvin",
        env_infos=env_infos,
        rlt_switch_flags=None,
        batch_size=3,
    )
    assert phase.shape == (3, 1)

    success = extract_rlt_lh_subtask_success(
        env_type="calvin",
        env_infos=env_infos,
        dones=None,
        batch_size=3,
        phase_idx=phase,
        prev_phase=torch.tensor([[0.0], [0.0], [1.0]]),
    )
    assert success.shape == (3, 1)
    assert success[0].item() == 0.0
    assert success[1].item() == 1.0
    assert success[2].item() == 1.0


def test_rlt_lh_policy_forward():
    z_dim = 8
    proprio_dim = 4
    action_dim = 2
    num_action_chunks = 3
    summary_dim = history_summary_dim(
        z_dim=z_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        num_action_chunks=num_action_chunks,
    )
    policy = RLTLHMLPPolicy(
        z_dim=z_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        num_action_chunks=num_action_chunks,
        history_input_dim=summary_dim,
        history_hidden_dim=16,
    )
    batch = 5
    obs = {
        "z_rl": torch.randn(batch, z_dim),
        "proprio": torch.randn(batch, proprio_dim),
        "ref_chunk": torch.randn(batch, num_action_chunks, action_dim),
        "history": torch.randn(batch, 4, summary_dim),
    }
    actions, result = policy.predict_action_batch(
        obs, mode="eval", return_obs=True
    )
    assert actions.shape == (batch, num_action_chunks, action_dim)
    assert result["forward_inputs"]["history"].shape == obs["history"].shape
    param_names = [name for name, _ in policy.named_parameters()]
    assert any("history_gru" in name for name in param_names)

    q_values = policy.sac_q_forward(obs, actions)
    assert q_values.shape == (batch, 2)
