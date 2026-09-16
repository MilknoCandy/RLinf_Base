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

from rlinf.algorithms.rlt.a23_memory import (
    A23MemoryBank,
    A23MemoryConfig,
    mix_a23_bootstrap_q,
)


def test_a23_commit_success_and_failure_banks():
    cfg = A23MemoryConfig(
        enable=True,
        pos_capacity=64,
        neg_capacity=64,
        top_k=4,
        gamma=1.0,
        k_pre=2,
        k_post=0,
        failure_beta=0.25,
        z_dim=4,
        action_dim=2,
    )
    bank = A23MemoryBank(cfg)
    t = 5
    z = torch.randn(t, 4)
    a = torch.randn(t, 2)
    r_ok = torch.zeros(t)
    r_ok[-1] = 1.0
    r_fail = torch.zeros(t)
    n_pos = bank.commit_episode(
        z_seq=z,
        a_seq=a,
        r_seq=r_ok,
        critical_mask=torch.ones(t, dtype=torch.bool),
        success=True,
    )
    n_neg = bank.commit_episode(
        z_seq=z,
        a_seq=a,
        r_seq=r_fail,
        critical_mask=torch.ones(t, dtype=torch.bool),
        success=False,
    )
    assert n_pos == t
    assert n_neg == t
    assert bank.pos_bank.size == t
    assert bank.neg_bank.size == t
    assert bank.memory_size == 2 * t


def test_a23_estimate_q_m_action_conditioned_and_failure_beta():
    cfg = A23MemoryConfig(
        enable=True,
        pos_capacity=32,
        neg_capacity=32,
        top_k=2,
        temperature=0.07,
        gamma=1.0,
        k_pre=8,
        k_post=0,
        failure_beta=0.25,
        z_dim=4,
        action_dim=2,
        alpha=0.1,
    )
    bank = A23MemoryBank(cfg)

    # Distinct (z,a) for success with R=1 and failure with R=0.
    z_pos = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    a_pos = torch.tensor([[1.0, 0.0]])
    z_neg = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    a_neg = torch.tensor([[0.0, 1.0]])
    bank.commit_episode(
        z_seq=z_pos,
        a_seq=a_pos,
        r_seq=torch.tensor([1.0]),
        critical_mask=torch.tensor([True]),
        success=True,
    )
    bank.commit_episode(
        z_seq=z_neg,
        a_seq=a_neg,
        r_seq=torch.tensor([0.0]),
        critical_mask=torch.tensor([True]),
        success=False,
    )

    q_pos, metrics = bank.estimate_q_m(z_pos, a_pos)
    assert metrics["a23_memory/used"] == 1.0
    assert float(q_pos.item()) > 0.5

    q_neg, _ = bank.estimate_q_m(z_neg, a_neg)
    # Failure R≈0, so Q_M near 0 even with beta scaling.
    assert abs(float(q_neg.item())) < 1e-5


def test_mix_a23_bootstrap_q():
    q_next = torch.tensor([[1.0], [3.0]])
    q_m = torch.tensor([[5.0], [7.0]])
    mixed, metrics = mix_a23_bootstrap_q(q_next=q_next, q_m=q_m, alpha=0.25)
    expected = 0.75 * q_next + 0.25 * q_m
    assert torch.allclose(mixed, expected)
    assert metrics["a23_memory/alpha"] == 0.25
