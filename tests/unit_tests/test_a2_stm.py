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

from rlinf.algorithms.rlt.a2_stm import A2MemoryBank, A2STMConfig, A2STMEncoder
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def _cfg(**kwargs) -> A2STMConfig:
    defaults = dict(
        enable=True,
        window_size=8,
        pos_capacity=32,
        neg_capacity=32,
        top_k=4,
        temperature=0.1,
        only_critical=True,
        fail_tail_steps=2,
        z_dim=8,
        action_dim=4,
        hidden_dim=16,
        gate_init_bias=-2.0,
    )
    defaults.update(kwargs)
    return A2STMConfig(**defaults)


def test_a2_encoder_empty_memory_is_identity():
    cfg = _cfg()
    enc = A2STMEncoder(cfg)
    z = torch.randn(3, 8, requires_grad=True)
    enhanced, metrics = enc.enhance(
        z,
        torch.zeros(0, 8),
        torch.zeros(0, 4),
    )
    assert torch.allclose(enhanced, z)
    assert metrics["a2_stm/retrieve_used"] == 0.0
    loss = enhanced.sum()
    loss.backward()
    assert z.grad is not None


def test_a2_encoder_retrieve_and_grad_to_gate():
    cfg = _cfg()
    enc = A2STMEncoder(cfg)
    z = torch.randn(2, 8)
    mem_z = torch.randn(5, 8)
    mem_a = torch.randn(5, 4)
    enhanced, metrics = enc.enhance(z, mem_z, mem_a)
    assert enhanced.shape == z.shape
    assert metrics["a2_stm/retrieve_used"] == 1.0
    assert metrics["a2_stm/enhance_delta_norm"] >= 0.0
    loss = enhanced.pow(2).mean()
    loss.backward()
    assert enc.gate[-1].weight.grad is not None
    assert enc.query.weight.grad is not None
    assert enc.value.weight.grad is not None


def test_a2_bank_write_on_success_and_fail():
    bank = A2MemoryBank(_cfg())
    z0 = torch.randn(2, 8)
    a0 = torch.randn(2, 4)
    bank.set_pending(
        z_rl=z0, actions=a0, critical_mask=torch.tensor([True, True])
    )
    bank.finalize_pending(
        dones=torch.tensor([False, False]),
        rewards=None,
        success=None,
        allow_write=True,
    )
    z1 = torch.randn(2, 8)
    a1 = torch.randn(2, 4)
    bank.set_pending(
        z_rl=z1, actions=a1, critical_mask=torch.tensor([True, False])
    )
    bank.finalize_pending(
        dones=torch.tensor([True, True]),
        rewards=torch.tensor([1.0, 0.0]),
        success=torch.tensor([True, False]),
        allow_write=True,
    )
    assert bank.pos_bank.size >= 1
    assert bank.neg_bank.size >= 1
    mem_z, mem_a = bank.gather_memory(torch.device("cpu"))
    assert mem_z.shape[0] == bank.memory_size
    assert mem_a.shape == (mem_z.shape[0], 4)


def test_a2_bank_batch_size_mismatch_drops_pending():
    bank = A2MemoryBank(_cfg())
    bank.set_pending(
        z_rl=torch.randn(3, 8),
        actions=torch.randn(3, 4),
        critical_mask=torch.ones(3, dtype=torch.bool),
    )
    bank.finalize_pending(
        dones=torch.zeros(2, dtype=torch.bool),
        rewards=None,
        success=None,
        allow_write=True,
    )
    assert bank.memory_size == 0


def test_rlt_mlp_policy_recomputes_z_with_grad():
    cfg = _cfg(z_dim=8, action_dim=8)
    policy = RLTMLPPolicy(
        z_dim=8,
        proprio_dim=4,
        action_dim=2,
        num_action_chunks=4,
        a2_stm_config=cfg,
    )
    bank = A2MemoryBank(cfg)
    bank.remember_batch(
        z_rl=torch.randn(6, 8),
        actions=torch.randn(6, 8),
        success=torch.tensor([True, True, False, False, True, False]),
    )
    policy.a2_stm_bank = bank

    obs = {
        "z_rl": torch.randn(2, 8, requires_grad=True),
        "proprio": torch.randn(2, 4),
        "ref_chunk": torch.randn(2, 4, 2),
    }
    action, _, _ = policy.sac_forward(obs)
    loss = action.pow(2).mean()
    loss.backward()
    assert policy.a2_stm_encoder.query.weight.grad is not None
    metrics = policy.pop_a2_stm_metrics()
    assert "a2_stm/gate_mean" in metrics
    assert metrics["a2_stm/memory_size"] > 0
