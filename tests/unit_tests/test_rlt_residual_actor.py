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

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def _obs(batch=2, chunk_len=5, action_dim=3, z_dim=8, proprio_dim=4):
    torch.manual_seed(0)
    return {
        "z_rl": torch.randn(batch, z_dim),
        "proprio": torch.randn(batch, proprio_dim),
        "ref_chunk": torch.linspace(-0.8, 0.8, batch * chunk_len * action_dim).reshape(
            batch, chunk_len, action_dim
        ),
    }


def test_zero_init_residual_actor_executes_the_vla_reference():
    policy = RLTMLPPolicy(
        z_dim=8,
        proprio_dim=4,
        action_dim=3,
        num_action_chunks=5,
        frozen_action_dims=[2],
        residual_actor=True,
    )
    with torch.no_grad():
        policy.actor_mean.weight.zero_()
        policy.actor_mean.bias.zero_()
    obs = _obs()
    action, _, _ = policy.sac_forward(obs, deterministic=True)
    ref = policy._get_ref_chunk(obs)
    assert torch.allclose(action, ref, atol=1e-5)


def test_nonzero_residual_edits_pose_and_holds_gripper():
    policy = RLTMLPPolicy(
        z_dim=8,
        proprio_dim=4,
        action_dim=3,
        num_action_chunks=5,
        frozen_action_dims=[2],
    )
    with torch.no_grad():
        policy.actor_mean.bias.fill_(1.5)
    obs = _obs()
    action, _, _ = policy.sac_forward(obs, deterministic=True)
    ref = policy._get_ref_chunk(obs)
    delta = action - ref
    pose_delta = delta.reshape(2, 5, 3)[..., :2]
    gripper = action.reshape(2, 5, 3)[..., 2]
    ref_grip = ref.reshape(2, 5, 3)[..., 2]
    assert pose_delta.abs().mean() > 0.1
    assert torch.allclose(gripper, ref_grip, atol=1e-5)
    assert action.min() >= -1.0
    assert action.max() <= 1.0


def test_bc_ref_scale_zero_does_not_clone_weak_proposal():
    from omegaconf import OmegaConf

    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin

    class _BC(RLTACLossMixin):
        def __init__(self, scale: float):
            self.cfg = OmegaConf.create(
                {
                    "actor": {"model": {"num_action_chunks": 2, "action_dim": 3}},
                    "algorithm": {"bc_ref_scale": scale},
                }
            )

    pi = torch.zeros(2, 2, 3)
    actions = torch.ones(2, 2, 3)
    ref = torch.full((2, 2, 3), 0.5)
    loss_off, _ = _BC(0.0)._bc_metrics(pi, actions, ref, None)
    loss_on, _ = _BC(1.0)._bc_metrics(pi, actions, ref, None)
    assert float(loss_off) == 0.0
    assert float(loss_on) > 0.0

    human = torch.ones(2, 2, 3, dtype=torch.bool)
    loss_human, metrics = _BC(0.0)._bc_metrics(pi, actions, ref, human)
    assert float(loss_human) > 0.0
    assert metrics["bc_human_loss"] > 0.0
