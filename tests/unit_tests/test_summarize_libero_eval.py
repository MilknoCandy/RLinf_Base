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

import importlib.util
from pathlib import Path

from rlinf.envs.libero.utils import perturbation_label_from_folder

_SUMMARIZE = (
    Path(__file__).resolve().parents[2]
    / "evaluations"
    / "libero"
    / "summarize_libero_eval.py"
)


def _load_summarize():
    spec = importlib.util.spec_from_file_location("summarize_libero_eval", _SUMMARIZE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_perturbation_label_from_folder():
    assert perturbation_label_from_folder("libero_spatial_lan") == "lan"
    assert perturbation_label_from_folder("libero_10_swap") == "swap"
    assert perturbation_label_from_folder("libero_spatial_object") == "object"
    assert perturbation_label_from_folder("libero_spatial") == "orig"
    assert perturbation_label_from_folder(None) == "orig"


def test_parse_log_splits_success_by_pert_and_task():
    summarize = _load_summarize()
    text = """
[libero eval] task_id=0, trial_id=0, pert=lan, success=True, task='bowl between plate and ramekin'
[libero eval] task_id=0, trial_id=1, pert=lan, success=False, task='bowl between plate and ramekin'
[libero eval] task_id=1, trial_id=0, pert=object, success=True
[libero eval] task_id=1, trial_id=1, pert=object, success=True
success_once=0.75
num_trajectories=4
"""
    parsed = summarize.parse_log(text)
    assert parsed["by_pert_task"]["lan"][0] == [1.0, 0.0]
    assert parsed["by_pert_task"]["object"][1] == [1.0, 1.0]
    assert parsed["task_names"][0] == "bowl between plate and ramekin"
    summary = summarize.format_summary(parsed, title="unit")
    assert "perturbation: lan" in summary
    assert "perturbation: object" in summary
    assert "bowl between plate and ramekin" in summary
    assert "combined (all perturbations in this run)" in summary


def test_parse_run_dir_reads_eval_log_not_just_metrics(tmp_path):
    summarize = _load_summarize()
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.log").write_text("success_once=0.5\nnum_trajectories=2\n")
    (run / "eval_embodiment.log").write_text(
        "[libero eval] task_id=3, trial_id=0, pert=lan, success=True\n"
        "[libero eval] task_id=3, trial_id=1, pert=lan, success=False\n"
    )
    parsed = summarize.parse_run_dir(run / "metrics.log")
    assert parsed["by_task"][3] == [1.0, 0.0]
    assert parsed["agg_success_once"] == 0.5
    summary = summarize.format_summary(parsed)
    assert "     3" in summary


def test_parse_log_fills_agg_from_episodes_when_metrics_table_missing():
    summarize = _load_summarize()
    text = """
[libero eval] task_id=9, trial_id=0, pert=task, success=True, task='bowl'
[libero eval] task_id=9, trial_id=1, pert=task, success=False, task='bowl'
"""
    parsed = summarize.parse_log(text)
    assert parsed["agg_success_once"] == 0.5
    assert parsed["num_trajectories"] == 2
    seed_summary = summarize.format_seed_aggregate(
        [("seed0", parsed["agg_success_once"], parsed["num_trajectories"])],
        title="unit multi-seed",
        parsed_runs=[parsed],
    )
    assert "0.5000" in seed_summary
    assert "n_seeds=1" in seed_summary


def test_parse_log_legacy_lines_without_pert():
    summarize = _load_summarize()
    text = """
[libero eval] task_id=2, trial_id=0, success=True
[libero eval] task_id=2, trial_id=1, success=False
success_once=0.5
"""
    parsed = summarize.parse_log(text)
    assert parsed["by_task"][2] == [1.0, 0.0]
    summary = summarize.format_summary(parsed)
    assert "perturbation:" not in summary
