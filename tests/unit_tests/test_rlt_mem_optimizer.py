# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke tests: mem loop_encoder must be trained by the critic optimizer under FSDP."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.modules.rlt_mem_write import RLTLoopEncoder


CRITIC_FILTERS = ["encoders", "encoder", "loop_encoder", "q_head", "state_proj"]


def _matches_critic(name: str) -> bool:
    return any(token in name for token in CRITIC_FILTERS)


def test_critic_filters_match_loop_encoder_param_names():
    """use_orig_params=True keeps names like loop_encoder.action_proj.weight."""
    assert _matches_critic("loop_encoder.action_proj.weight")
    assert _matches_critic("loop_encoder.pred_ref.0.weight")
    assert _matches_critic("_fsdp_wrapped_module.q_head.qs.0.net.0.weight")
    assert not _matches_critic("_fsdp_wrapped_module.backbone.0.weight")


def test_write_accepts_reshape_views_without_inplace_error():
    """Replay often feeds reshape views; write must not trip view-inplace guards."""
    enc = RLTLoopEncoder(z_dim=32, action_dim=16, prefix_len=8, num_layers=1, num_heads=4)
    enc.train()
    batch = 3
    prefix = torch.randn(batch, 1, 8, 32).reshape(batch, 8, 32)
    mask = torch.ones(batch, 8, dtype=torch.bool)
    z_prev = torch.randn(batch, 1, 32).reshape(batch, 32)
    prev_action = torch.randn(batch, 2, 8).reshape(batch, 16)
    prev_reward = torch.randn(batch, 1, 1).reshape(batch, 1)
    z = enc.write(prefix, mask, z_prev, prev_action, prev_reward)
    ref = torch.randn(batch, 16)
    pred_ref, pred_r = enc.predict(z)
    loss = F.mse_loss(pred_ref, ref) + F.mse_loss(pred_r, torch.zeros(batch, 1))
    loss.backward()
    assert z.shape == (batch, 32)


def _tiny_mem_policy() -> RLTMLPPolicy:
    return RLTMLPPolicy(
        z_dim=32,
        proprio_dim=4,
        action_dim=8,
        num_action_chunks=2,
        use_mem=True,
        loop_prefix_len=8,
        loop_num_heads=4,
        loop_num_layers=1,
    )


def _init_single_process_group() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29572")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)


def _split_actor_critic_params(model: torch.nn.Module):
    actor_params = []
    critic_params = []
    critic_names = []
    actor_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _matches_critic(name):
            critic_params.append(param)
            critic_names.append(name)
        else:
            actor_params.append(param)
            actor_names.append(name)
    return actor_params, critic_params, actor_names, critic_names


@pytest.fixture(scope="module")
def fsdp_mem_model():
    _init_single_process_group()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = _tiny_mem_policy().to(device)
    fsdp_cfg = OmegaConf.create({"disable": False, "wrap_policy": {}})
    wrap_policy = get_fsdp_wrap_policy(
        module=policy,
        config=fsdp_cfg,
        is_lora=False,
        model_type="rlt_mlp_policy",
    )
    # Match mem YAML: use_orig_params=True so loop_encoder.* stays critic-owned
    # without nesting FSDP around custom write().
    model = FSDP(
        policy,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        use_orig_params=True,
        device_id=torch.cuda.current_device() if device.type == "cuda" else None,
    )
    yield model, device


def test_loop_encoder_params_assigned_to_critic_under_fsdp(fsdp_mem_model):
    model, _device = fsdp_mem_model
    _, critic_params, actor_names, critic_names = _split_actor_critic_params(model)
    assert critic_params, f"critic group empty; names={actor_names + critic_names}"
    assert any("loop_encoder" in name for name in critic_names), (
        f"loop_encoder missing from critic param names: {critic_names}"
    )
    assert not any("loop_encoder" in name for name in actor_names), (
        f"loop_encoder incorrectly in actor param names: {actor_names}"
    )


def test_critic_step_updates_loop_encoder_weights(fsdp_mem_model):
    """End-to-end smoke: critic Adam step must move loop_encoder / pred_ref."""
    model, device = fsdp_mem_model
    if device.type != "cuda":
        pytest.skip("FSDP smoke requires CUDA in this environment")

    _, critic_params, _, critic_names = _split_actor_critic_params(model)
    assert any("loop_encoder" in name for name in critic_names)

    qf_optimizer = torch.optim.Adam(critic_params, lr=1e-3)
    actor_params, _, _, _ = _split_actor_critic_params(model)
    actor_optimizer = torch.optim.Adam(actor_params, lr=1e-3)

    # Hold Parameter object refs: after an FSDP step, named_parameters() can
    # temporarily omit loop_encoder until the next forward.
    loop_param_refs = [
        (name, param)
        for name, param in zip(critic_names, critic_params)
        if "loop_encoder" in name
    ]
    assert loop_param_refs, "expected loop_encoder parameters in critic group"
    before = {
        name: param.detach().float().cpu().clone() for name, param in loop_param_refs
    }

    batch = 4
    obs = {
        "prefix_embs": torch.randn(batch, 8, 32, device=device),
        "prefix_mask": torch.ones(batch, 8, dtype=torch.bool, device=device),
        "z_prev": torch.randn(batch, 32, device=device),
        "prev_action": torch.randn(batch, 16, device=device),
        "prev_reward": torch.randn(batch, 1, device=device),
        "ref_chunk": torch.randn(batch, 2, 8, device=device),
        "proprio": torch.randn(batch, 4, device=device),
    }
    actions = torch.tanh(torch.randn(batch, 16, device=device))
    ref = obs["ref_chunk"].reshape(batch, 16)

    model.train()
    qf_optimizer.zero_grad(set_to_none=True)
    # Pred must stay inside the FSDP-wrapped SAC_Q call (not model.predictive_loss
    # via getattr), or use_orig_params writeback fails on the next forward.
    q_values, pred_ref_loss, pred_r_loss, _ = model(
        forward_type=ForwardType.SAC_Q,
        obs=obs,
        actions=actions,
        detach_encoder=False,
        pred_ref_chunk=ref,
        pred_reward_target=torch.zeros(batch, 1, device=device),
    )
    loss = F.mse_loss(q_values, torch.zeros_like(q_values)) + pred_ref_loss + pred_r_loss
    loss.backward()
    qf_optimizer.step()
    actor_optimizer.zero_grad(set_to_none=True)

    after = {
        name: param.detach().float().cpu().clone() for name, param in loop_param_refs
    }
    changed = {
        k: float((after[k] - before[k]).abs().max()) for k in before if k in after
    }
    assert any(delta > 0 for delta in changed.values()), (
        f"loop_encoder weights unchanged after critic step; deltas={changed}"
    )
    pred_keys = [k for k in changed if "pred_ref" in k]
    assert pred_keys, f"no pred_ref keys; deltas={list(changed)}"
    assert any(changed[k] > 0 for k in pred_keys), (
        f"pred_ref weights unchanged; deltas={ {k: changed[k] for k in pred_keys} }"
    )

    # Second critic step: catches FSDP _pre_forward writeback corruption.
    qf_optimizer.zero_grad(set_to_none=True)
    q_values2, pred_ref_loss2, pred_r_loss2, _ = model(
        forward_type=ForwardType.SAC_Q,
        obs=obs,
        actions=actions,
        detach_encoder=False,
        pred_ref_chunk=ref,
        pred_reward_target=torch.zeros(batch, 1, device=device),
    )
    loss2 = (
        F.mse_loss(q_values2, torch.zeros_like(q_values2))
        + pred_ref_loss2
        + pred_r_loss2
    )
    loss2.backward()
    qf_optimizer.step()
