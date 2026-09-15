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

from rlinf.algorithms.rlt.a21_context import A21ContextBuffer, A21ContextConfig
from rlinf.algorithms.rlt.a22_memory import (
    A22MemoryBank,
    A22MemoryConfig,
    compute_a22_memory_loss,
    compute_monte_carlo_rtg,
)
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def test_monte_carlo_rtg():
    rewards = torch.tensor([0.0, 0.0, 1.0])
    rtg = compute_monte_carlo_rtg(rewards, gamma=0.5)
    assert torch.allclose(rtg, torch.tensor([0.25, 0.5, 1.0]))


def test_a21_context_buffer_push_and_reset():
    cfg = A21ContextConfig(enable=True, ctx_len=3, z_dim=4, action_dim=2)
    buf = A21ContextBuffer(cfg)
    ctx0 = buf.get(2, torch.device("cpu"))
    assert float(ctx0["ctx_mask"].sum()) == 0.0

    z = torch.randn(2, 4)
    a = torch.randn(2, 2)
    ctx1 = buf.push(z_rl=z, actions=a)
    assert float(ctx1["ctx_mask"][:, -1].sum()) == 2.0 or float(ctx1["ctx_mask"].sum()) == 2.0

    buf.reset(torch.tensor([True, False]))
    ctx2 = buf.get(2, torch.device("cpu"))
    assert float(ctx2["ctx_mask"][0].sum()) == 0.0
    assert float(ctx2["ctx_mask"][1].sum()) == 1.0


def test_a21_policy_input_dim_includes_context():
    cfg = A21ContextConfig(enable=True, ctx_len=2, z_dim=8, action_dim=8)
    policy = RLTMLPPolicy(
        z_dim=8,
        proprio_dim=4,
        action_dim=2,
        num_action_chunks=4,
        a21_context_config=cfg,
    )
    assert policy.context_dim == 2 * (8 + 8)
    obs = {
        "z_rl": torch.randn(2, 8),
        "proprio": torch.randn(2, 4),
        "ref_chunk": torch.randn(2, 4, 2),
        "ctx_z": torch.randn(2, 2, 8),
        "ctx_a": torch.randn(2, 2, 8),
        "ctx_mask": torch.ones(2, 2),
    }
    action, _, _ = policy.sac_forward(obs)
    assert action.shape[0] == 2
    q = policy.sac_q_forward(obs, action)
    assert q.shape[0] == 2


def test_a22_commit_episode_fills_rtg_and_e3_window():
    cfg = A22MemoryConfig(
        enable=True,
        pos_capacity=64,
        neg_capacity=64,
        top_k=4,
        gamma=1.0,
        k_pre=2,
        k_post=1,
        z_dim=4,
        action_dim=2,
    )
    bank = A22MemoryBank(cfg)
    t = 6
    z = torch.randn(t, 4)
    a = torch.randn(t, 2)
    r = torch.zeros(t)
    r[-1] = 1.0
    critical = torch.zeros(t, dtype=torch.bool)
    critical[1:3] = True
    n = bank.commit_episode(
        z_seq=z,
        a_seq=a,
        r_seq=r,
        critical_mask=critical,
        success=True,
    )
    # E1: indices 1,2; E2 around T=5 with k_pre=2,k_post=1 -> 3,4,5 (and maybe 6 clipped)
    # union at least {1,2,3,4,5}
    assert n >= 5
    assert bank.pos_bank.size == n
    _, _, _, R = bank.pos_bank.tensors()
    assert float(R.max()) == 1.0


def test_a22_memory_loss_positive_adv_only():
    cfg = A22MemoryConfig(
        enable=True,
        pos_capacity=32,
        neg_capacity=32,
        top_k=4,
        gamma=1.0,
        k_pre=2,
        k_post=0,
        z_dim=8,
        action_dim=8,
        lambda_m=0.1,
        adv_threshold=0.0,
    )
    bank = A22MemoryBank(cfg)
    z = torch.randn(5, 8)
    a = torch.tanh(torch.randn(5, 8))
    r = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    bank.commit_episode(
        z_seq=z,
        a_seq=a,
        r_seq=r,
        critical_mask=torch.ones(5, dtype=torch.bool),
        success=True,
    )
    policy = RLTMLPPolicy(
        z_dim=8,
        proprio_dim=4,
        action_dim=2,
        num_action_chunks=4,
    )
    obs = {
        "z_rl": torch.randn(2, 8),
        "proprio": torch.randn(2, 4),
        "ref_chunk": torch.randn(2, 4, 2),
    }
    # Low baseline -> positive advantages when R~1
    loss, metrics = compute_a22_memory_loss(
        policy_model=policy,
        obs=obs,
        bank=bank,
        q_baseline=torch.zeros(2, 1),
        fixed_std=0.002,
    )
    assert metrics["a22_memory/retrieve_k"] > 0
    assert loss.ndim == 0
    loss.backward()
    assert policy.actor_mean.weight.grad is not None
