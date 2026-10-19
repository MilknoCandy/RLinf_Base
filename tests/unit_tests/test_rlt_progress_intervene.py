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
# See the License for the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from rlinf.envs.libero.utils import (
    bowl_place_progress_from_obs,
    count_new_episode_dones,
    merge_expert_intervene_flags,
    update_progress_intervene_state,
)


def _obs(*, ee, bowl, plate, grip=(0.04, 0.04)):
    return {
        "robot0_eef_pos": np.asarray(ee, dtype=np.float64),
        "akita_black_bowl_1_pos": np.asarray(bowl, dtype=np.float64),
        "plate_1_pos": np.asarray(plate, dtype=np.float64),
        "robot0_gripper_qpos": np.asarray(grip, dtype=np.float64),
    }


def _step_state(
    best,
    stall,
    burst,
    initialized,
    *,
    progress,
    valid=True,
    success=False,
    done=False,
    min_delta=0.008,
    stall_steps=3,
    burst_steps=2,
):
    return update_progress_intervene_state(
        best=best,
        stall=stall,
        burst_left=burst,
        initialized=initialized,
        progress=np.asarray(progress, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
        success=np.asarray(success, dtype=bool),
        done=np.asarray(done, dtype=bool),
        min_delta=min_delta,
        stall_steps=stall_steps,
        burst_steps=burst_steps,
    )


def test_approach_then_grasp_jumps_progress():
    open_grip = (0.08, 0.08)
    far = bowl_place_progress_from_obs(
        _obs(
            ee=(1.0, 0.0, 0.2),
            bowl=(0.0, 0.0, 0.05),
            plate=(0.2, 0.0, 0.02),
            grip=open_grip,
        )
    )
    near = bowl_place_progress_from_obs(
        _obs(
            ee=(0.02, 0.0, 0.06),
            bowl=(0.0, 0.0, 0.05),
            plate=(0.2, 0.0, 0.02),
            grip=open_grip,
        )
    )
    grasped = bowl_place_progress_from_obs(
        _obs(
            ee=(0.01, 0.0, 0.06),
            bowl=(0.0, 0.0, 0.05),
            plate=(0.2, 0.0, 0.02),
            grip=(0.02, 0.02),
        )
    )
    assert far[1] and near[1] and grasped[1]
    assert near[0] > far[0]
    assert grasped[0] > 0.5
    assert grasped[0] > near[0]


def test_missing_object_is_invalid():
    score, valid = bowl_place_progress_from_obs(
        {"robot0_eef_pos": np.zeros(3), "robot0_gripper_qpos": np.zeros(2)}
    )
    assert not valid
    assert score == 0.0


def test_stall_triggers_short_burst_then_releases():
    n = 1
    best = np.full(n, -1.0e6)
    stall = np.zeros(n, dtype=np.int32)
    burst = np.zeros(n, dtype=np.int32)
    initialized = np.zeros(n, dtype=bool)
    flags = []
    for _ in range(8):
        best, stall, burst, initialized, intervene = _step_state(
            best, stall, burst, initialized, progress=[0.0]
        )
        flags.append(bool(intervene[0]))
    # first step initializes; next 3 stall; burst is 2 steps.
    assert flags[:1] == [False]
    assert flags[1:3] == [False, False]
    assert flags[3] is True
    assert True in flags[3:6]
    assert flags[6] is False
    assert flags[7] is False


def test_progress_resets_stall():
    n = 1
    best = np.full(n, -1.0e6)
    stall = np.zeros(n, dtype=np.int32)
    burst = np.zeros(n, dtype=np.int32)
    initialized = np.zeros(n, dtype=bool)
    best, stall, burst, initialized, _ = _step_state(
        best, stall, burst, initialized, progress=[0.0]
    )
    best, stall, burst, initialized, _ = _step_state(
        best, stall, burst, initialized, progress=[0.0]
    )
    assert int(stall[0]) == 1
    best, stall, burst, initialized, intervene = _step_state(
        best, stall, burst, initialized, progress=[0.05]
    )
    assert int(stall[0]) == 0
    assert not bool(intervene[0])


def test_invalid_progress_still_triggers_burst():
    n = 1
    best = np.full(n, -1.0e6)
    stall = np.zeros(n, dtype=np.int32)
    burst = np.zeros(n, dtype=np.int32)
    initialized = np.zeros(n, dtype=bool)
    flags = []
    for _ in range(5):
        best, stall, burst, initialized, intervene = _step_state(
            best,
            stall,
            burst,
            initialized,
            progress=[0.0],
            valid=False,
        )
        flags.append(bool(intervene[0]))
    assert True in flags


def test_tiny_jitter_does_not_reset_large_min_delta():
    n = 1
    best = np.full(n, -1.0e6)
    stall = np.zeros(n, dtype=np.int32)
    burst = np.zeros(n, dtype=np.int32)
    initialized = np.zeros(n, dtype=bool)
    progress = 0.0
    triggered = False
    for step in range(6):
        progress += 0.008
        best, stall, burst, initialized, intervene = _step_state(
            best,
            stall,
            burst,
            initialized,
            progress=[progress],
            min_delta=0.04,
            stall_steps=3,
        )
        triggered = triggered or bool(intervene[0])
        if step == 2:
            assert int(stall[0]) >= 2
    assert triggered


def test_always_expert_owns_train_collect():
    off = merge_expert_intervene_flags(3, seed_active=False, always=False)
    assert not off.any()
    seed = merge_expert_intervene_flags(3, seed_active=True, always=False)
    assert seed.all()
    always = merge_expert_intervene_flags(3, seed_active=False, always=True)
    assert always.all()
    progress = np.array([True, False, False])
    mixed = merge_expert_intervene_flags(
        3, seed_active=False, progress_flags=progress, always=False
    )
    assert mixed.tolist() == [True, False, False]
    forced = merge_expert_intervene_flags(
        3, seed_active=False, progress_flags=progress, always=True
    )
    assert forced.all()


def test_count_new_episode_dones_ignores_sticky_done():
    prev = np.array([False, False])
    cur = np.array([True, False])
    prev, n = count_new_episode_dones(prev, cur)
    assert n == 1
    prev, n = count_new_episode_dones(prev, cur)
    assert n == 0
    cur = np.array([True, True])
    _, n = count_new_episode_dones(prev, cur)
    assert n == 1


def test_object_perturbation_mug_counts_as_bowl():
    score, valid = bowl_place_progress_from_obs(
        {
            "robot0_eef_pos": np.array([0.0, 0.0, 0.2]),
            "red_mug_1_pos": np.array([0.0, 0.0, 0.05]),
            "white_plate_1_pos": np.array([0.2, 0.0, 0.02]),
            "robot0_gripper_qpos": np.array([0.08, 0.08]),
        }
    )
    assert valid
    assert score < 0.0


def test_success_and_done_clear_burst():
    n = 1
    best = np.array([0.0])
    stall = np.array([10], dtype=np.int32)
    burst = np.array([5], dtype=np.int32)
    initialized = np.array([True])
    _, stall, burst, initialized, intervene = _step_state(
        best,
        stall,
        burst,
        initialized,
        progress=[0.0],
        success=True,
    )
    assert int(burst[0]) == 0
    assert not bool(intervene[0])

    best = np.array([0.0])
    stall = np.array([10], dtype=np.int32)
    burst = np.array([5], dtype=np.int32)
    initialized = np.array([True])
    _, _, burst, initialized, intervene = _step_state(
        best,
        stall,
        burst,
        initialized,
        progress=[0.0],
        done=True,
    )
    assert int(burst[0]) == 0
    assert not bool(initialized[0])
    assert not bool(intervene[0])
