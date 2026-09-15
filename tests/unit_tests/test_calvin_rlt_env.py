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

"""Unit tests for CalvinRLTEnv switch helpers (no CALVIN runtime required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_calvin_rlt_env_module():
    """Load calvin_rlt_env.py with a stub CalvinEnv base class."""
    stub = ModuleType("rlinf.envs.calvin.calvin_gym_env")

    class CalvinEnv:  # noqa: D101
        pass

    stub.CalvinEnv = CalvinEnv
    sys.modules["rlinf.envs.calvin.calvin_gym_env"] = stub

    module_path = (
        Path(__file__).resolve().parents[2]
        / "rlinf"
        / "envs"
        / "calvin"
        / "calvin_rlt_env.py"
    )
    # Force reload so edits are picked up when tests re-run in-process.
    sys.modules.pop("calvin_rlt_env_under_test", None)
    spec = importlib.util.spec_from_file_location(
        "calvin_rlt_env_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_calvin_rlt_env = _load_calvin_rlt_env_module()
CalvinRLTEnv = _calvin_rlt_env.CalvinRLTEnv


def _make_calvin_rlt_env(
    *,
    num_envs: int = 4,
    enable: bool = True,
    task_mode: str = "full_task",
    trigger_mode: str = "always_on",
    latch_until_done: bool = True,
    min_task_idx: int = 2,
    current_task_idx: list[int] | None = None,
) -> CalvinRLTEnv:
    cfg = SimpleNamespace(
        rlt_policy_switch={
            "enable": enable,
            "task_mode": task_mode,
            "trigger_mode": trigger_mode,
            "latch_until_done": latch_until_done,
            "auto_gate": {"min_task_idx": min_task_idx},
            "expert_takeover": {"enable": False},
        }
    )
    env = CalvinRLTEnv.__new__(CalvinRLTEnv)
    env.num_envs = num_envs
    env.cfg = cfg
    env.record_metrics = True
    env.current_task_idx = (
        list(current_task_idx)
        if current_task_idx is not None
        else [0] * num_envs
    )
    env._rlt_switch_cfg = cfg.rlt_policy_switch
    env._rlt_switch_state = None
    env._init_rlt_switch()
    return env


def test_calvin_rlt_always_on_exports_switch_flags():
    env = _make_calvin_rlt_env(trigger_mode="always_on")
    info = env._export_rlt_switch_info()
    assert info["rlt_switch_flags"].shape == (4, 1)
    assert bool(info["rlt_switch_flags"].all())
    assert info["intervene_flag"].shape == (4, 1)
    assert not bool(info["intervene_flag"].any())
    assert torch.equal(
        info["current_task_idx"],
        torch.zeros((4, 1), dtype=torch.long),
    )


def test_calvin_rlt_always_on_reset_restores_active():
    env = _make_calvin_rlt_env(trigger_mode="always_on")
    env._rlt_switch_state["rlt_switch_flags"].fill_(False)
    env._reset_rlt_switch([1, 3])
    flags = env._rlt_switch_state["rlt_switch_flags"]
    assert not bool(flags[0])
    assert bool(flags[1])
    assert not bool(flags[2])
    assert bool(flags[3])
    env._reset_rlt_switch()
    assert bool(env._rlt_switch_state["rlt_switch_flags"].all())


def test_calvin_rlt_disabled_exports_false_flags():
    env = _make_calvin_rlt_env(enable=False)
    assert env._rlt_switch_state is None
    info = env._export_rlt_switch_info()
    assert not bool(info["rlt_switch_flags"].any())


def test_calvin_rlt_rejects_unsupported_trigger_mode():
    with pytest.raises(ValueError, match="always_on' or 'auto"):
        _make_calvin_rlt_env(trigger_mode="stalled_progress")


def test_calvin_rlt_auto_inactive_before_min_task_idx():
    env = _make_calvin_rlt_env(
        trigger_mode="auto",
        min_task_idx=2,
        current_task_idx=[0, 1, 0, 1],
    )
    env._update_rlt_switch()
    assert not bool(env._rlt_switch_state["rlt_switch_flags"].any())


def test_calvin_rlt_auto_enters_at_min_task_idx():
    env = _make_calvin_rlt_env(
        trigger_mode="auto",
        min_task_idx=2,
        current_task_idx=[0, 2, 1, 3],
    )
    env._update_rlt_switch()
    flags = env._rlt_switch_state["rlt_switch_flags"]
    assert list(flags.tolist()) == [False, True, False, True]


def test_calvin_rlt_auto_latches_until_reset():
    env = _make_calvin_rlt_env(
        trigger_mode="auto",
        min_task_idx=2,
        latch_until_done=True,
        current_task_idx=[2, 2, 0, 0],
    )
    env._update_rlt_switch()
    # Drop below threshold; latch should keep previously entered envs on.
    env.current_task_idx = [0, 0, 0, 0]
    env._update_rlt_switch()
    flags = env._rlt_switch_state["rlt_switch_flags"]
    assert list(flags.tolist()) == [True, True, False, False]

    env._reset_rlt_switch([0])
    env.current_task_idx = [0, 0, 0, 0]
    env._update_rlt_switch()
    flags = env._rlt_switch_state["rlt_switch_flags"]
    assert list(flags.tolist()) == [False, True, False, False]


def test_calvin_rlt_auto_without_latch_follows_idx():
    env = _make_calvin_rlt_env(
        trigger_mode="auto",
        min_task_idx=2,
        latch_until_done=False,
        current_task_idx=[3, 1],
        num_envs=2,
    )
    env._update_rlt_switch()
    assert list(env._rlt_switch_state["rlt_switch_flags"].tolist()) == [True, False]
    env.current_task_idx = [1, 4]
    env._update_rlt_switch()
    assert list(env._rlt_switch_state["rlt_switch_flags"].tolist()) == [False, True]


def test_calvin_rlt_attach_updates_infos():
    env = _make_calvin_rlt_env(num_envs=2, trigger_mode="always_on")
    infos = {}
    env._attach_rlt_switch_info(infos)
    assert torch.equal(
        infos["rlt_switch_flags"],
        torch.ones((2, 1), dtype=torch.bool),
    )
