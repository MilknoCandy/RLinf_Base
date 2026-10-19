# Copyright 2025 The RLinf Authors.
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

"""Task-instruction override for ManiSkill peg eval."""

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_peg_variants_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root / "rlinf" / "envs" / "maniskill" / "peg_insertion_side_variants.py"
    )
    fake_package = types.ModuleType("rlinf.envs.maniskill")
    fake_package.__path__ = [str(module_path.parent)]
    monkeypatch.setitem(sys.modules, "rlinf.envs.maniskill", fake_package)
    monkeypatch.delitem(
        sys.modules, "rlinf.envs.maniskill.peg_insertion_side_variants", raising=False
    )
    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.maniskill.peg_insertion_side_variants", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _dummy_raw_obs(num_envs: int = 2) -> dict:
    return {
        "sensor_data": {
            "3rd_view_camera": {"rgb": torch.zeros(num_envs, 4, 4, 3)},
            "wide_hand_camera": {"rgb": torch.zeros(num_envs, 4, 4, 3)},
        },
        "sensor_param": {},
        "agent": {"qpos": torch.zeros(num_envs, 9)},
    }


def test_cfg_task_instruction_empty(monkeypatch):
    module = _load_peg_variants_module(monkeypatch)
    assert module.cfg_task_instruction(None) is None
    assert module.cfg_task_instruction({}) is None
    assert module.cfg_task_instruction({"task_instruction": None}) is None
    assert module.cfg_task_instruction({"task_instruction": ""}) is None
    assert module.cfg_task_instruction({"task_instruction": "   "}) is None


def test_cfg_task_instruction_keeps_legacy_into(monkeypatch):
    module = _load_peg_variants_module(monkeypatch)
    text = module.cfg_task_instruction(
        {"task_instruction": module._RLT_LEGACY_PEG_PROMPT}
    )
    assert text == module._RLT_LEGACY_PEG_PROMPT
    assert module.normalize_peg_instruction(text) == module._RLT_DEFAULT_PROMPT
    assert text != module.normalize_peg_instruction(text)


def test_expand_task_instruction(monkeypatch):
    module = _load_peg_variants_module(monkeypatch)
    assert module.expand_task_instruction("pick up the cube", num_envs=3) == [
        "pick up the cube",
        "pick up the cube",
        "pick up the cube",
    ]


def test_wrap_force_skips_infos_prompt(monkeypatch):
    module = _load_peg_variants_module(monkeypatch)
    forced = ["insert the peg into the hole", "insert the peg into the hole"]
    obs = module.wrap_rlt_openpi_joint_obs(
        _dummy_raw_obs(),
        infos={"prompt": ["pick up the cube", "pick up the cube"]},
        task_descriptions=forced,
        num_envs=2,
        device=torch.device("cpu"),
        is_peg_insertion_side=True,
        force_task_descriptions=True,
    )
    assert obs["task_descriptions"] == forced


def test_wrap_infos_prompt_rewrites_legacy_into(monkeypatch):
    module = _load_peg_variants_module(monkeypatch)
    obs = module.wrap_rlt_openpi_joint_obs(
        _dummy_raw_obs(),
        infos={
            "prompt": [
                module._RLT_LEGACY_PEG_PROMPT,
                module._RLT_LEGACY_PEG_PROMPT,
            ]
        },
        task_descriptions=["unused", "unused"],
        num_envs=2,
        device=torch.device("cpu"),
        is_peg_insertion_side=True,
        force_task_descriptions=False,
    )
    assert obs["task_descriptions"] == [
        module._RLT_DEFAULT_PROMPT,
        module._RLT_DEFAULT_PROMPT,
    ]
