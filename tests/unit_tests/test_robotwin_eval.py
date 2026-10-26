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

"""Unit tests for RoboTwin eval seed fallback and RLT task screening."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rlinf.envs.sim.robotwin.seed_utils import load_task_success_seeds

REPO = Path(__file__).resolve().parents[2]
EVAL_ROBOTWIN = REPO / "evaluations" / "robotwin"
sys.path.insert(0, str(EVAL_ROBOTWIN))

from robotwin_tasks import (  # noqa: E402
    ALL_TASKS,
    SEEDED_TASKS,
    UNSUPPORTED_TASKS,
    classify_rlt_band,
    classify_rlt_phases,
    eval_seeds_path,
    list_tasks,
    step_lim,
)
from summarize_robotwin_eval import format_filter_table, parse_log  # noqa: E402


def test_load_task_success_seeds_missing_task_returns_none(tmp_path):
    seeds = tmp_path / "eval_seeds.json"
    seeds.write_text(
        json.dumps({"adjust_bottle": {"success_seeds": [11, 12]}}),
        encoding="utf-8",
    )
    assert load_task_success_seeds(str(seeds), "adjust_bottle") == [11, 12]
    assert load_task_success_seeds(str(seeds), "stamp_seal") is None
    assert load_task_success_seeds(None, "adjust_bottle") is None
    assert (
        load_task_success_seeds(str(tmp_path / "absent.json"), "adjust_bottle") is None
    )


def test_catalog_matches_official_suite_and_in_repo_seeds():
    from rlinf.envs.sim.robotwin.critical_phase import GATES, has_critical_gate

    assert len(ALL_TASKS) == 50
    assert len(set(ALL_TASKS)) == 50
    assert len(SEEDED_TASKS) == 22
    assert len(UNSUPPORTED_TASKS) == 4
    assert set(SEEDED_TASKS) <= set(ALL_TASKS)
    assert set(UNSUPPORTED_TASKS) <= set(ALL_TASKS)
    assert set(GATES) == set(ALL_TASKS)
    assert all(has_critical_gate(name) for name in ALL_TASKS)
    assert list_tasks("all") == list(ALL_TASKS)
    assert list_tasks("gated") == list(ALL_TASKS)
    assert "place_fan" in list_tasks("all")
    assert list_tasks("supported") == [
        name for name in ALL_TASKS if name not in UNSUPPORTED_TASKS
    ]
    assert "place_fan" not in list_tasks("supported")
    assert list_tasks("handover_block,stamp_seal") == [
        "handover_block",
        "stamp_seal",
    ]
    with pytest.raises(ValueError, match="unknown"):
        list_tasks("not_a_task")


def test_eval_seeds_json_keys_match_seeded_catalog():
    data = json.loads(eval_seeds_path().read_text(encoding="utf-8"))
    assert set(data) == set(SEEDED_TASKS)


def test_step_lim_is_chunk_aligned():
    assert step_lim("adjust_bottle") == 200
    assert step_lim("handover_block") == 400
    assert step_lim("stamp_seal") == 600
    for name in ALL_TASKS:
        assert step_lim(name) % 50 == 0


def test_pi05_eval_yaml_matches_jax_agilex_recipe():
    text = (EVAL_ROBOTWIN / "robotwin_openpi_pi05_eval.yaml").read_text(
        encoding="utf-8"
    )
    assert "adapt_to_pi: false" in text
    assert "extra_delta_transform: true" in text
    assert "physical-intelligence/robotwin/norm_stats.json" in text
    assert 'config_name: "pi05_aloha_robotwin"' in text


def test_rlt_bands():
    assert classify_rlt_band(0.96) == "saturated"
    assert classify_rlt_band(0.85) == "borderline_high"
    assert classify_rlt_band(0.80) == "keep"
    assert classify_rlt_band(0.55) == "keep"
    assert classify_rlt_band(0.50) == "borderline_low"
    assert classify_rlt_band(0.20) == "too_hard"


def test_rlt_phases_need_reach_not_episode_success():
    assert classify_rlt_phases(0.63, None) == "no_gate"
    assert classify_rlt_phases(0.63, 0.63, has_gate=False) == "no_gate"
    assert classify_rlt_phases(0.60, 0.50) == "coarse_fail"
    assert classify_rlt_phases(0.72, 0.75) == "saturated"
    assert classify_rlt_phases(0.72, 0.80) == "low_headroom"
    assert classify_rlt_phases(0.55, 0.80) == "keep"


def test_summarize_parse_and_keep_list():
    parsed = parse_log(
        "Evaluation\nsuccess_once=0.55 reach_critical_once=0.80 "
        "fail_in_critical=0.25 critical_gate=1.0 num_trajectories=100\n"
    )
    assert parsed["success_rate"] == pytest.approx(0.55)
    assert parsed["reach_rate"] == pytest.approx(0.80)
    assert parsed["fail_in_critical"] == pytest.approx(0.25)
    assert parsed["num_trajectories"] == 100
    table = format_filter_table(
        [
            {
                "task": "handover_mic",
                "setting": "clean",
                "success_rate": 0.55,
                "reach_rate": 0.80,
                "fail_in_critical": 0.25,
                "critical_gate": 1.0,
                "num_trajectories": 100,
            },
            {
                "task": "adjust_bottle",
                "setting": "clean",
                "success_rate": 0.99,
                "num_trajectories": 100,
            },
        ]
    )
    assert "keep" in table
    keep_line = table.split("keep for RLT Stage-2:")[1]
    assert "handover_mic" in keep_line
    assert "adjust_bottle" not in keep_line
    assert "episode" in table


class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FailingTask:
    def setup_demo(self, **kwargs):
        raise RuntimeError("setup always fails")

    def get_info(self):
        return {}

    def close_env(self, clear_cache=True):
        return None


def test_setup_task_retry_limit_raises_instead_of_hanging():
    from rlinf.envs.sim.robotwin.eval_compat import patch_setup_task_retry_limit

    class SubEnv:
        def close(self):
            return None

    calls = {"n": 0}

    def class_decorator(_name):
        calls["n"] += 1
        return _FailingTask()

    patch_setup_task_retry_limit(SubEnv, class_decorator=class_decorator, max_trials=3)
    env = SubEnv()
    env.task_name = "place_fan"
    env.env_seed = 0
    env.args = {}
    env.global_lock = _DummyLock()
    env.lock = _DummyLock()
    with pytest.raises(RuntimeError, match="setup_demo failed after 3 seeds"):
        env.setup_task()
    assert calls["n"] == 1 + 3  # close() replacement + 3 trials


def test_ensure_robotwin_assets_links_or_stubs(tmp_path, monkeypatch):
    from rlinf.envs.sim.robotwin.eval_compat import (
        EMPTY_OBJAVERSE_LIST,
        ensure_robotwin_assets,
        objaverse_list_path,
    )

    shared = tmp_path / "dataroot" / "RoboTwin" / "assets"
    target = shared / "objects" / "objaverse" / "list.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("RLinf_DATAROOT", str(tmp_path / "dataroot"))
    clone = tmp_path / "RoboTwin"
    clone.mkdir()
    got = ensure_robotwin_assets(clone)
    assert got.is_file()
    assert got == objaverse_list_path(clone)

    clone2 = tmp_path / "RoboTwin2"
    clone2.mkdir()
    (clone2 / "assets" / "embodiments" / "aloha-agilex").mkdir(parents=True)
    monkeypatch.delenv("RLinf_DATAROOT", raising=False)
    monkeypatch.setenv("RLinf_DATAROOT", str(tmp_path / "missing"))
    got2 = ensure_robotwin_assets(clone2)
    assert json.loads(got2.read_text(encoding="utf-8")) == EMPTY_OBJAVERSE_LIST


def test_dense_reward_constructor_accepts_task_kwargs():
    from rlinf.envs.sim.robotwin.eval_compat import _PATCHED_ATTR

    class Reward:
        pass

    assert Reward.__init__ is object.__init__

    def __init__(self, *args, **kwargs):
        self.subtasks = kwargs.get("subtasks", args[0] if args else [])
        self.transition_rewards = kwargs.get("transition_rewards", [])

    Reward.__init__ = __init__
    setattr(Reward, _PATCHED_ATTR, True)
    built = Reward(subtasks=["pick"], transition_rewards=[1])
    assert built.subtasks == ["pick"]
    assert built.transition_rewards == [1]


class _Pose:
    def __init__(self, x, y, z):
        self.p = [x, y, z]


class _Actor:
    def __init__(self, x, y, z):
        self._pose = _Pose(x, y, z)

    def get_pose(self):
        return self._pose

    def get_functional_point(self, *args):
        if args and args[-1] == "pose":
            return self._pose
        return self._pose.p


class _UnknownTask:
    pass


class handover_mic:  # noqa: N801 — matches RoboTwin task class name
    def __init__(self, z, contacts):
        self.microphone = _Actor(0.1, 0.0, z)
        self._contacts = contacts

    def get_gripper_actor_contact_position(self, name):
        assert name == "018_microphone"
        return self._contacts


class stack_bowls_three:  # noqa: N801
    def __init__(self, bowls):
        self.bowl1 = _Actor(*bowls[0])
        self.bowl2 = _Actor(*bowls[1])
        self.bowl3 = _Actor(*bowls[2])


class adjust_bottle:  # noqa: N801
    def __init__(self, x, z, contacts):
        self.bottle = _Actor(x, -0.10, z)
        self._contacts = contacts

    def get_gripper_actor_contact_position(self, name):
        assert name == "001_bottle"
        return self._contacts


class beat_block_hammer:  # noqa: N801
    def __init__(self, hammer_xyz, block_xyz, contacts):
        self.hammer = _Actor(*hammer_xyz)
        self.block = _Actor(*block_xyz)
        self._contacts = contacts

    def get_gripper_actor_contact_position(self, name):
        assert name == "020_hammer"
        return self._contacts


class blocks_ranking_rgb:  # noqa: N801
    def __init__(self, blocks):
        self.block1 = _Actor(*blocks[0])
        self.block2 = _Actor(*blocks[1])
        self.block3 = _Actor(*blocks[2])


def test_critical_phase_gate_and_stack_handover():
    from rlinf.envs.sim.robotwin.critical_phase import (
        has_critical_gate,
        in_critical,
        progress_into_critical,
    )

    assert has_critical_gate("handover_mic")
    assert has_critical_gate("adjust_bottle")
    assert in_critical(_UnknownTask()) == (False, False)
    assert progress_into_critical(_UnknownTask()) == (False, False)

    lifted = handover_mic(z=0.90, contacts=[[0.0, 0.0, 0.9]])
    missed = handover_mic(z=0.80, contacts=[])
    assert in_critical(lifted) == (True, True)
    assert in_critical(missed) == (False, True)

    two_stacked = stack_bowls_three(
        ((0.0, -0.1, 0.74), (0.01, -0.1, 0.77), (0.2, 0.1, 0.74))
    )
    none_stacked = stack_bowls_three(
        ((0.0, -0.1, 0.74), (0.2, 0.1, 0.74), (-0.2, 0.1, 0.74))
    )
    assert in_critical(two_stacked) == (True, True)
    assert in_critical(none_stacked) == (False, True)


def test_spawn_true_gate_is_not_last_skill_entry():
    from rlinf.envs.sim.robotwin.critical_phase import (
        in_critical,
        progress_into_critical,
    )

    already_up = handover_mic(z=0.90, contacts=[[0.0, 0.0, 0.9]])
    assert in_critical(already_up) == (True, True)
    assert progress_into_critical(already_up) == (False, True)
    assert progress_into_critical(already_up) == (False, True)

    later = handover_mic(z=0.80, contacts=[])
    assert progress_into_critical(later) == (False, True)
    later.microphone = _Actor(0.1, 0.0, 0.90)
    later._contacts = [[0.0, 0.0, 0.9]]
    assert progress_into_critical(later) == (True, True)


def test_last_skill_gates_reject_start_pose():
    from rlinf.envs.sim.robotwin.critical_phase import in_critical

    spawn_bottle = adjust_bottle(x=0.10, z=0.752, contacts=[])
    assert in_critical(spawn_bottle) == (False, True)
    held = adjust_bottle(x=0.10, z=0.85, contacts=[[0.1, -0.1, 0.85]])
    assert in_critical(held) == (True, True)

    near_block = beat_block_hammer(
        hammer_xyz=(0.0, -0.06, 0.783),
        block_xyz=(0.06, -0.05, 0.76),
        contacts=[],
    )
    assert in_critical(near_block) == (False, True)
    picked = beat_block_hammer(
        hammer_xyz=(0.0, -0.06, 0.85),
        block_xyz=(0.20, 0.10, 0.76),
        contacts=[[0.0, -0.06, 0.85]],
    )
    assert in_critical(picked) == (True, True)

    unordered_close = blocks_ranking_rgb(
        ((0.05, 0.0, 0.765), (0.0, 0.01, 0.765), (0.25, -0.08, 0.765))
    )
    assert in_critical(unordered_close) == (False, True)
    two_ranked = blocks_ranking_rgb(
        ((-0.08, -0.15, 0.74), (0.0, -0.15, 0.74), (0.25, 0.04, 0.765))
    )
    assert in_critical(two_ranked) == (True, True)


def test_stage1_yaml_matches_maniskill_rlt_and_aloha_recipe():
    sft = REPO / "examples" / "sft" / "config"
    text = (sft / "robotwin_rlt_stage1_sft_openpi_pi05.yaml").read_text(
        encoding="utf-8"
    )
    assert "model_type: \"openpi\"" in text
    assert "use_rlt: True" in text
    assert "train_expert_only: False" in text
    assert 'config_name: "pi05_aloha_robotwin"' in text
    assert "action_dim: 14" in text
    assert "num_action_chunks: 50" in text
    assert "num_images_in_input: 3" in text
    assert "adapt_to_pi: false" in text
    assert "extra_delta_transform: true" in text
    assert "module_names: [rlt_module]" in text
    assert "ROBOTWIN_TASK_PROMPT" not in text
    assert "default_prompt:" not in text
    assert (sft / "robotwin_rlt_stage1_sft_openpi_pi05_exp.yaml").is_file()
    assert (sft / "robotwin_rlt_stage1_sft_openpi_pi05_mtp.yaml").is_file()


def test_robotwin_hdf5_episode_to_openpi_frames(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from rlinf.data.datasets.openpi.robotwin_hdf5 import (
        find_episode_files,
        frame_to_lerobot,
        iter_episode_frames,
        task_prompt,
    )

    assert task_prompt("adjust_bottle") == "adjust bottle"
    demo = tmp_path / "demo_clean" / "adjust_bottle" / "aloha-agilex" / "data"
    demo.mkdir(parents=True)
    path = demo / "episode0.hdf5"
    t, h, w = 4, 8, 10
    with h5py.File(path, "w") as root:
        root.create_dataset(
            "joint_action/vector",
            data=np.arange(t * 14, dtype=np.float32).reshape(t, 14),
        )
        for name, fill in (
            ("head_camera", 10),
            ("left_camera", 20),
            ("right_camera", 30),
        ):
            root.create_dataset(
                f"observation/{name}/rgb",
                data=np.full((t, h, w, 3), fill, dtype=np.uint8),
            )
    files = find_episode_files(tmp_path / "demo_clean" / "adjust_bottle")
    assert files == [path]
    frames = list(iter_episode_frames(path, task_name="adjust_bottle"))
    assert len(frames) == 3
    assert frames[0].state.tolist() == list(range(14))
    assert frames[0].action.tolist() == list(range(14, 28))
    assert frames[0].cam_high[0, 0, 0] == 10
    assert frames[0].cam_left_wrist[0, 0, 0] == 20
    packed = frame_to_lerobot(frames[0])
    assert packed["observation.state"].shape == (14,)
    assert packed["action"].shape == (14,)
    assert packed["task"] == "adjust bottle"
    assert "observation.images.cam_high" in packed


def test_robotwin_sidecar_instruction(tmp_path):
    from rlinf.data.datasets.openpi.robotwin_hdf5 import load_sidecar_instruction

    hdf5 = tmp_path / "aloha_agilex" / "data" / "episode_0000000.hdf5"
    hdf5.parent.mkdir(parents=True)
    hdf5.write_bytes(b"")
    inst = tmp_path / "aloha_agilex" / "instruction" / "episode_0000000.json"
    inst.parent.mkdir(parents=True)
    inst.write_text('{"instruction": "adjust the bottle"}', encoding="utf-8")
    assert load_sidecar_instruction(hdf5) == "adjust the bottle"


def test_convert_robotwin_demo_writes_lerobot(tmp_path):
    pytest.importorskip("lerobot")
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from toolkits.lerobot.convert_robotwin_demo_to_lerobot import (
        convert_demo,
        is_lerobot_dataset,
    )

    demo = tmp_path / "demo_clean" / "adjust_bottle" / "aloha-agilex" / "data"
    demo.mkdir(parents=True)
    path = demo / "episode0.hdf5"
    t, h, w = 3, 4, 6
    with h5py.File(path, "w") as root:
        root.create_dataset(
            "joint_action/vector",
            data=np.arange(t * 14, dtype=np.float32).reshape(t, 14),
        )
        for name in ("head_camera", "left_camera", "right_camera"):
            root.create_dataset(
                f"observation/{name}/rgb",
                data=np.full((t, h, w, 3), 7, dtype=np.uint8),
            )
    output = tmp_path / "lerobot" / "adjust_bottle"
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}", encoding="utf-8")
    convert_demo(
        task="adjust_bottle",
        demo_root=tmp_path / "demo_clean" / "adjust_bottle",
        output=output,
        fps=20,
        prompt="adjust bottle",
        norm_stats=stats,
        overwrite=True,
    )
    assert is_lerobot_dataset(output)
    assert (output / "norm_stats.json").is_file()
    assert (output / "rlinf_robotwin_convert.json").is_file()


def test_find_lerobot_dataset_roots_mixed_layout(tmp_path):
    import importlib.util

    from rlinf.data.storage.lerobot.paths import find_lerobot_dataset_roots

    spec = importlib.util.spec_from_file_location(
        "download_lerobot_v21",
        REPO / "scripts" / "robotwin" / "download_lerobot_v21.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    mixed = tmp_path / "v21" / "robotwin_demo_clean_aloha_agilex"
    (mixed / "meta").mkdir(parents=True)
    (mixed / "meta" / "info.json").write_text(
        '{"total_episodes": 2500}', encoding="utf-8"
    )
    roots = find_lerobot_dataset_roots(tmp_path / "v21")
    assert roots == [mixed.resolve()]
    assert module.resolve_mixed_root(tmp_path / "v21") == mixed.resolve()
