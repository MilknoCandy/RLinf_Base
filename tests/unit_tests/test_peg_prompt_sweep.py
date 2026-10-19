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

"""ManiSkill peg prompt-sensitivity catalog."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_PY = REPO_ROOT / "evaluations" / "maniskill" / "peg_prompt_sweep.py"
SUMMARIZE_PY = REPO_ROOT / "evaluations" / "maniskill" / "summarize_maniskill_eval.py"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fine_sweep_is_one_change_per_axis():
    sweep = _load(SWEEP_PY)
    rows = sweep.load_prompt_rows("fine")
    tags = [tag for tag, _ in rows]
    assert tags[0] == "train"
    assert rows[0][1] == "insert the peg in the hole"
    assert len(rows) == 12
    assert "lex_into" in tags
    assert "para_put" in tags
    assert "spat_align" in tags
    assert "trunc_obj" in tags
    assert "neg_cube" in tags


def test_mini_and_all_sizes():
    sweep = _load(SWEEP_PY)
    mini_tags = [tag for tag, _ in sweep.load_prompt_rows("mini")]
    assert mini_tags == ["para_put", "trunc_obj", "neg_cube"]
    assert "train" not in mini_tags
    assert "lex_into" not in mini_tags
    all_rows = sweep.load_prompt_rows("all")
    assert len(all_rows) >= 24
    assert len({tag for tag, _ in all_rows}) == len(all_rows)


def test_comma_groups_and_unknown_sweep():
    sweep = _load(SWEEP_PY)
    tags = [tag for tag, _ in sweep.load_prompt_rows("lexical,negative")]
    assert "train" in tags
    assert "lex_into" in tags
    assert "neg_cube" in tags
    assert "para_put" not in tags
    with pytest.raises(ValueError, match="unknown"):
        sweep.load_prompt_rows("not_a_sweep")


def test_prompt_compare_delta_vs_train():
    summarize = _load(SUMMARIZE_PY)
    text = summarize.format_prompt_compare(
        [
            ("train", "insert the peg in the hole", 0.70, 0.0, 1),
            ("neg_cube", "pick up the cube", 0.40, 0.0, 1),
            ("lex_into", "insert the peg into the hole", 0.68, 0.0, 1),
        ]
    )
    assert "neg_cube" in text
    assert "-30.0pp" in text
    assert "train" in text
