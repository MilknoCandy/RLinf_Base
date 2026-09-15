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

from rlinf.algorithms.rlt.a1_stm import A1STMConfig, A1ShortTermMemory


def _make_stm(**overrides) -> A1ShortTermMemory:
    cfg = A1STMConfig(
        enable=True,
        window_size=8,
        pos_capacity=32,
        neg_capacity=32,
        top_k=4,
        temperature=0.1,
        gate=0.5,
        only_critical=True,
        fail_tail_steps=2,
        z_dim=8,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return A1ShortTermMemory(cfg)


def test_a1_stm_enhance_is_identity_without_memory():
    stm = _make_stm()
    z = torch.randn(3, 8)
    enhanced, metrics = stm.enhance(z)
    torch.testing.assert_close(enhanced, z)
    assert metrics["a1_stm/memory_size"] == 0.0
    assert metrics["a1_stm/retrieve_used"] == 0.0


def test_a1_stm_writes_success_and_failure_banks():
    stm = _make_stm()
    z0 = torch.randn(2, 8)
    stm.set_pending(
        z_rl=z0,
        critical_mask=torch.tensor([True, True]),
    )
    # Next step provides feedback for the pending write.
    stm.finalize_pending(
        dones=torch.tensor([False, False]),
        rewards=None,
        success=None,
        allow_write=True,
    )

    z1 = torch.randn(2, 8)
    stm.set_pending(
        z_rl=z1,
        critical_mask=torch.tensor([True, True]),
    )
    stm.finalize_pending(
        dones=torch.tensor([True, True]),
        rewards=None,
        success=torch.tensor([True, False]),
        allow_write=True,
    )

    assert stm.pos_bank.size >= 1
    assert stm.neg_bank.size >= 1

    query = z1[:1]
    enhanced, metrics = stm.enhance(query)
    assert enhanced.shape == query.shape
    assert metrics["a1_stm/retrieve_used"] == 1.0
    assert metrics["a1_stm/memory_size"] >= 1.0
    assert not torch.allclose(enhanced, query)


def test_a1_stm_skips_non_critical_when_only_critical():
    stm = _make_stm(only_critical=True)
    z = torch.randn(1, 8)
    stm.set_pending(z_rl=z, critical_mask=torch.tensor([False]))
    stm.finalize_pending(
        dones=torch.tensor([True]),
        rewards=None,
        success=torch.tensor([True]),
        allow_write=True,
    )
    assert stm.pos_bank.size == 0
    assert stm.neg_bank.size == 0


def test_a1_stm_pop_logged_metrics_averages_enhance_stats():
    stm = _make_stm()
    z = torch.randn(1, 8)
    stm.enhance(z)
    stm.enhance(z)
    metrics = stm.pop_logged_metrics()
    assert metrics["a1_stm/enhance_count"] == 2.0
    assert "a1_stm/retrieve_used" in metrics
    # Second pop should reset step accumulators.
    metrics_again = stm.pop_logged_metrics()
    assert metrics_again["a1_stm/enhance_count"] == 0.0
    assert metrics_again["a1_stm/write_pos_count"] == 0.0
