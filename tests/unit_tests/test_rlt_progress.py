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

from rlinf.algorithms.rlt.progress import ProgressMemoryState
from rlinf.algorithms.rlt.transition import extract_rlt_obs_from_forward_inputs
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def _infos(distance_x, abs_y, abs_z, success):
    return {
        "peg_head_hole_x": torch.tensor(distance_x),
        "peg_head_hole_abs_y": torch.tensor(abs_y),
        "peg_head_hole_abs_z": torch.tensor(abs_z),
        "success_current": torch.tensor(success),
    }


def test_progress_memory_zeros_on_first_step_then_tracks_delta():
    memory = ProgressMemoryState()
    device = torch.device("cpu")
    first = memory.update(
        batch_size=2,
        dones=None,
        env_infos=_infos([0.0, 0.0], [0.04, 0.04], [0.0, 0.0], [False, False]),
        device=device,
        distance_scale=0.05,
    )
    assert first.shape == (2, 4)
    assert float(first[0, 1]) == 0.0
    assert float(first[0, 3]) == 0.0

    closer = memory.update(
        batch_size=2,
        dones=torch.tensor([False, False]),
        env_infos=_infos([0.02, 0.0], [0.01, 0.04], [0.0, 0.0], [False, False]),
        device=device,
        distance_scale=0.05,
    )
    assert float(closer[0, 1]) < 0.0
    assert float(closer[0, 3]) == 1.0
    assert float(closer[1, 1]) == 0.0


def test_progress_memory_resets_delta_on_done():
    memory = ProgressMemoryState()
    device = torch.device("cpu")
    memory.update(
        batch_size=1,
        dones=None,
        env_infos=_infos([0.0], [0.04], [0.0], [False]),
        device=device,
    )
    reset = memory.update(
        batch_size=1,
        dones=torch.tensor([True]),
        env_infos=_infos([0.0], [0.08], [0.0], [False]),
        device=device,
    )
    assert float(reset[0, 1]) == 0.0
    assert float(reset[0, 3]) == 0.0


def test_critic_concatenates_progress_actor_does_not():
    policy = RLTMLPPolicy(
        z_dim=4,
        proprio_dim=3,
        action_dim=2,
        num_action_chunks=2,
        progress_dim=4,
        progress_in_actor=False,
    )
    obs_zero = {
        "z_rl": torch.zeros(2, 4),
        "proprio": torch.zeros(2, 3),
        "ref_chunk": torch.zeros(2, 4),
        "rlt_progress": torch.zeros(2, 4),
    }
    obs_one = dict(obs_zero)
    obs_one["rlt_progress"] = torch.ones(2, 4)
    assert torch.equal(policy._actor_state(obs_zero), policy._actor_state(obs_one))
    assert not torch.equal(
        policy._critic_state(obs_zero), policy._critic_state(obs_one)
    )
    assert policy._critic_state(obs_one).shape[-1] == 4 + 3 + 4
    assert policy._actor_state(obs_one).shape[-1] == 4 + 3 + 4


def test_extract_rlt_obs_keeps_progress_in_replay_keys():
    forward_inputs = {
        "z_rl": torch.zeros(2, 4),
        "proprio": torch.zeros(2, 3),
        "ref_chunk": torch.zeros(2, 8),
        "rlt_progress": torch.ones(2, 4),
        "rlt_transition_z_rl": torch.zeros(2, 4),
        "rlt_transition_proprio": torch.zeros(2, 3),
        "rlt_transition_ref_chunk": torch.zeros(2, 8),
        "rlt_transition_rlt_progress": torch.full((2, 4), 2.0),
    }
    current = extract_rlt_obs_from_forward_inputs(forward_inputs)
    nxt = extract_rlt_obs_from_forward_inputs(forward_inputs, transition=True)
    assert "rlt_progress" in current
    assert float(nxt["rlt_progress"].mean()) == 2.0
