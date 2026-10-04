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

from rlinf.algorithms.rlt.tsac_prefix import pack_tsac_prefix_windows
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.modules.tsac_transformer_critic import (
    MultiTSACTransformerQHead,
    discounted_prefix_returns,
    prefix_horizon_mask,
    sample_tsac_horizon,
)


def test_tsac_transformer_emits_prefix_q_and_is_causal():
    torch.manual_seed(0)
    critic = MultiTSACTransformerQHead(
        state_dim=8,
        action_dim=3,
        d_model=32,
        num_layers=2,
        num_heads=4,
        max_action_len=10,
        num_q_heads=2,
    )
    state = torch.randn(2, 8)
    actions = torch.randn(2, 6, 3)
    q = critic(state, actions)
    assert q.shape == (2, 6, 2)

    # Changing a future action must not change earlier prefix Qs.
    actions_alt = actions.clone()
    actions_alt[:, -1] += 1.0
    q_alt = critic(state, actions_alt)
    assert torch.allclose(q[:, :-1], q_alt[:, :-1], atol=1e-5)
    assert not torch.allclose(q[:, -1], q_alt[:, -1], atol=1e-5)


def test_sample_horizon_is_from_candidates_and_respects_valid_steps():
    horizons = [5, 10, 15, 20, 25]
    valid = torch.tensor([5, 12, 25, 3])
    sampled = sample_tsac_horizon(horizons, valid)
    assert sampled.tolist()[0] == 5
    assert sampled.tolist()[1] in (5, 10)
    assert sampled.tolist()[2] in horizons
    assert int(sampled.tolist()[3]) == 3  # below min horizon: fallback


def test_prefix_horizon_mask_and_discounted_returns():
    horizons = [5, 10, 15]
    sampled = torch.tensor([10, 15, 5])
    mask = prefix_horizon_mask(sampled, horizons)
    assert mask.tolist() == [
        [True, True, False],
        [True, True, True],
        [True, False, False],
    ]

    rewards = torch.ones(2, 15)
    returns = discounted_prefix_returns(rewards, gamma=0.5, horizons=horizons)
    # sum_{j=0}^{4} 0.5^j = (1 - 0.5^5) / (1 - 0.5) = 1.9375
    assert abs(float(returns[0, 0]) - 1.9375) < 1e-5


def _make_transition(
    *,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    z: torch.Tensor,
    proprio: torch.Tensor,
    ref: torch.Tensor,
    done: bool,
) -> Trajectory:
    traj = Trajectory(max_episode_length=1, model_weights_id="test")
    traj.actions = actions.view(1, 1, *actions.shape)
    traj.rewards = rewards.view(1, 1, *rewards.shape)
    traj.dones = torch.tensor([[[done]]], dtype=torch.bool)
    traj.terminations = torch.tensor([[[done]]], dtype=torch.bool)
    traj.truncations = torch.tensor([[[False]]], dtype=torch.bool)
    traj.next_obs = {
        "z_rl": z.view(1, 1, -1),
        "proprio": proprio.view(1, 1, -1),
        "ref_chunk": ref.view(1, 1, -1),
    }
    traj.curr_obs = {
        "z_rl": z.view(1, 1, -1),
        "proprio": proprio.view(1, 1, -1),
        "ref_chunk": ref.view(1, 1, -1),
    }
    return traj


def test_pack_tsac_prefix_windows_looks_ahead_until_done():
    chunk_len = 5
    action_dim = 2
    z_dim = 4
    proprio_dim = 3
    ref_dim = chunk_len * action_dim
    transitions = []
    for idx in range(3):
        transitions.append(
            _make_transition(
                actions=torch.full((chunk_len, action_dim), float(idx + 1)),
                rewards=torch.full((chunk_len,), float(idx + 1)),
                z=torch.full((z_dim,), float(idx + 10)),
                proprio=torch.full((proprio_dim,), float(idx + 20)),
                ref=torch.full((ref_dim,), float(idx + 30)),
                done=(idx == 1),
            )
        )

    packed = pack_tsac_prefix_windows(
        transitions,
        chunk_len=chunk_len,
        action_dim=action_dim,
        max_chunks=5,
        z_dim=z_dim,
        proprio_dim=proprio_dim,
        ref_dim=ref_dim,
    )
    assert int(packed[0].tsac_valid_chunks.item()) == 2
    assert packed[0].tsac_prefix_actions.shape == (1, 1, 25, 2)
    # First two chunks filled, remaining zeros.
    assert torch.allclose(
        packed[0].tsac_prefix_actions[0, 0, :5], torch.ones(5, 2)
    )
    assert torch.allclose(
        packed[0].tsac_prefix_actions[0, 0, 5:10], torch.full((5, 2), 2.0)
    )
    assert torch.allclose(
        packed[0].tsac_prefix_actions[0, 0, 10:], torch.zeros(15, 2)
    )
    assert bool(packed[0].tsac_chunk_done[0, 0, 1].item())


def test_rlt_mlp_policy_tsac_q_last_position():
    policy = RLTMLPPolicy(
        z_dim=8,
        proprio_dim=4,
        action_dim=3,
        num_action_chunks=5,
        ref_num_action_chunks=5,
        q_head_type="tsac_transformer",
        tsac_d_model=32,
        tsac_num_layers=1,
        tsac_num_heads=4,
        tsac_max_action_len=15,
    )
    obs = {
        "z_rl": torch.randn(2, 8),
        "proprio": torch.randn(2, 4),
        "ref_chunk": torch.randn(2, 5, 3),
    }
    actions = torch.randn(2, 5, 3)
    q_last = policy.sac_q_forward(obs, actions)
    q_seq = policy.sac_q_forward(obs, actions, return_sequence=True)
    assert q_last.shape == (2, 2)
    assert q_seq.shape == (2, 5, 2)
    assert torch.allclose(q_last, q_seq[:, -1, :])
