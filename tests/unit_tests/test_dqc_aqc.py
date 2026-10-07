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

from rlinf.algorithms.qc.adaptive import (
    discount_normalized_advantage,
    select_adaptive_chunk,
    zscore,
)
from rlinf.algorithms.qc.expectile import expectile_loss, expectile_weight
from rlinf.algorithms.qc.windows import pack_dqc_windows
from rlinf.data.schema.embodied_types import Trajectory


def test_expectile_weights_high_kappa_emphasize_overprediction_residual():
    residual = torch.tensor([-1.0, 1.0])
    weight = expectile_weight(residual, 0.9)
    assert float(weight[0]) == 0.1
    assert float(weight[1]) == 0.9
    pred = torch.zeros(2)
    target = torch.tensor([-1.0, 1.0])
    loss = expectile_loss(pred, target, 0.9)
    assert float(loss) > 0.0


def test_advantage_selector_picks_best_scale_after_zscore():
    # [batch, N, |K|] with a unique maximum at sample 1, horizon 0.
    scores = torch.tensor(
        [
            [
                [0.1, -0.2],
                [1.5, 0.3],
                [0.0, 0.4],
            ]
        ]
    )
    sample_idx, horizon_idx = select_adaptive_chunk(scores)
    assert sample_idx.tolist() == [1]
    assert horizon_idx.tolist() == [0]
    q = torch.tensor([[1.0, 2.0, 0.5]])
    v = torch.zeros(1, 1)
    short = discount_normalized_advantage(q, v, gamma=0.99, horizon=1)
    assert float(short[0, 1]) > float(short[0, 0])
    assert zscore(short, dim=-1).shape == short.shape


def test_pack_dqc_windows_concatenates_two_chunks():
    rows = []
    for i in range(2):
        row = Trajectory(max_episode_length=1)
        row.actions = torch.ones(1, 1, 2, 3) * (i + 1)
        row.rewards = torch.ones(1, 1, 2) * 0.5
        row.dones = torch.zeros(1, 1, 2, dtype=torch.bool)
        row.next_obs = {
            "z_rl": torch.zeros(1, 1, 4),
            "proprio": torch.zeros(1, 1, 2),
            "ref_chunk": torch.zeros(1, 1, 6),
        }
        rows.append(row)
    packed = pack_dqc_windows(
        rows,
        chunk_len=2,
        action_dim=3,
        backup_chunks=2,
        gamma=0.99,
        z_dim=4,
        proprio_dim=2,
    )
    assert packed[0].dqc_valid_chunks.item() == 2
    assert packed[0].dqc_chunk_actions.numel() == 12
    assert packed[1].dqc_valid_chunks.item() == 1
    assert float(packed[0].dqc_n_step_returns) > float(packed[1].dqc_n_step_returns)
