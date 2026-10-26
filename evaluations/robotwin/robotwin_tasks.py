#!/usr/bin/env python3
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

"""RoboTwin 2.0 task catalog for RLinf π₀.₅ eval sweeps.

The suite is 50 tasks. Default ``TASKS=all`` runs every name; each one has a
critical-phase detector in ``rlinf.envs.sim.robotwin.critical_phase``. The
four names in ``NO_DENSE_REWARD_TASKS`` only lack a dense reward for RL
*training*; eval still judges their last-skill entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

TaskSet = Literal["seeded", "supported", "all", "gated"]

# Dense reward is missing for RL training. Eval still runs these 50-task names.
NO_DENSE_REWARD_TASKS = (
    "open_laptop",
    "place_fan",
    "place_object_scale",
    "put_object_cabinet",
)
UNSUPPORTED_TASKS = NO_DENSE_REWARD_TASKS

# Official Aloha-AgileX eval seeds shipped in-repo.
SEEDED_TASKS = (
    "adjust_bottle",
    "place_phone_stand",
    "click_bell",
    "place_a2b_left",
    "move_can_pot",
    "handover_mic",
    "place_a2b_right",
    "pick_dual_bottles",
    "lift_pot",
    "put_bottles_dustbin",
    "stack_blocks_two",
    "stack_bowls_two",
    "handover_block",
    "shake_bottle",
    "place_empty_cup",
    "place_container_plate",
    "move_stapler_pad",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "move_pillbottle_pad",
    "place_mouse_pad",
    "place_shoe",
)

# 50 RoboTwin 2.0 tasks from upstream ``envs/*.py``.
ALL_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)

# Keep 200/400 from existing RLinf env YAMLs; longer-horizon tasks use 600.
_STEP_LIM_200 = frozenset(
    {
        "adjust_bottle",
        "beat_block_hammer",
        "lift_pot",
        "move_can_pot",
        "pick_dual_bottles",
        "place_container_plate",
        "place_empty_cup",
    }
)
_STEP_LIM_600 = frozenset(
    {
        "blocks_ranking_rgb",
        "blocks_ranking_size",
        "hanging_mug",
        "open_laptop",
        "open_microwave",
        "put_bottles_dustbin",
        "put_object_cabinet",
        "scan_object",
        "stack_blocks_three",
        "stack_bowls_three",
        "stamp_seal",
    }
)

DEFAULT_KEEP_LOW = 0.55
DEFAULT_KEEP_HIGH = 0.80
DEFAULT_SATURATED = 0.90
DEFAULT_TOO_HARD = 0.40
DEFAULT_KEEP_REACH = 0.70
DEFAULT_SATURATED_GIVEN_REACH = 0.90
DEFAULT_MIN_FAIL_IN_CRITICAL = 0.10


def eval_seeds_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "rlinf"
        / "envs"
        / "sim"
        / "robotwin"
        / "seeds"
        / "eval_seeds.json"
    )


def step_lim(task_name: str) -> int:
    """Return the eval horizon for ``task_name`` (divisible by action chunk 50)."""
    if task_name in _STEP_LIM_200:
        return 200
    if task_name in _STEP_LIM_600:
        return 600
    return 400


def is_rlinf_supported(task_name: str) -> bool:
    return task_name not in UNSUPPORTED_TASKS


def has_eval_seeds(task_name: str) -> bool:
    return task_name in SEEDED_TASKS


def classify_rlt_band(
    success_rate: float,
    *,
    keep_low: float = DEFAULT_KEEP_LOW,
    keep_high: float = DEFAULT_KEEP_HIGH,
    saturated: float = DEFAULT_SATURATED,
    too_hard: float = DEFAULT_TOO_HARD,
) -> str:
    """Band episode success only. Prefer ``classify_rlt_phases`` when reach exists."""
    if success_rate > saturated:
        return "saturated"
    if success_rate > keep_high:
        return "borderline_high"
    if success_rate >= keep_low:
        return "keep"
    if success_rate >= too_hard:
        return "borderline_low"
    return "too_hard"


def classify_rlt_phases(
    success_rate: float,
    reach_rate: float | None,
    *,
    keep_reach: float = DEFAULT_KEEP_REACH,
    saturated_given_reach: float = DEFAULT_SATURATED_GIVEN_REACH,
    min_fail_in_critical: float = DEFAULT_MIN_FAIL_IN_CRITICAL,
    has_gate: bool | None = None,
) -> str:
    """Band RLT tasks by critical-phase entry, not episode success alone."""
    if has_gate is False or reach_rate is None:
        return "no_gate"
    fail_in_critical = max(0.0, float(reach_rate) - float(success_rate))
    succ_given = (
        float(success_rate) / float(reach_rate) if float(reach_rate) > 0.0 else 0.0
    )
    if float(reach_rate) < keep_reach:
        return "coarse_fail"
    if succ_given > saturated_given_reach:
        return "saturated"
    if fail_in_critical < min_fail_in_critical:
        return "low_headroom"
    return "keep"


def gated_tasks() -> list[str]:
    """Tasks that have a critical-phase detector, in ``ALL_TASKS`` order."""
    from rlinf.envs.sim.robotwin.critical_phase import has_critical_gate

    return [name for name in ALL_TASKS if has_critical_gate(name)]


def list_tasks(task_set: str) -> list[str]:
    """Return task names for ``gated``, ``seeded``, ``supported``, ``all``, or a CSV list."""
    wanted = task_set.strip().lower()
    if wanted == "gated":
        return gated_tasks()
    if wanted == "seeded":
        return list(SEEDED_TASKS)
    if wanted == "supported":
        return [name for name in ALL_TASKS if is_rlinf_supported(name)]
    if wanted == "all":
        return list(ALL_TASKS)
    names = [part.strip() for part in task_set.split(",") if part.strip()]
    unknown = [name for name in names if name not in ALL_TASKS]
    if unknown:
        raise ValueError(
            f"unknown RoboTwin task(s) {unknown}; expected names from ALL_TASKS"
        )
    return names


def task_prompt(task_name: str) -> str:
    """Language prompt for Stage-1 SFT when the episode has no instruction."""
    if task_name not in ALL_TASKS:
        raise ValueError(f"unknown RoboTwin task: {task_name}")
    return task_name.replace("_", " ")


def task_record(task_name: str) -> dict[str, object]:
    from rlinf.envs.sim.robotwin.critical_phase import gate_spec

    spec = gate_spec(task_name)
    return {
        "task_name": task_name,
        "step_lim": step_lim(task_name),
        "rlinf_supported": is_rlinf_supported(task_name),
        "has_eval_seeds": has_eval_seeds(task_name),
        "has_critical_gate": spec is not None,
        "last_skill": None if spec is None else spec.last_skill,
        "enter_when": None if spec is None else spec.enter_when,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        dest="task_set",
        default="all",
        help="all (50, default) | gated (same 50, every task has a detector) | "
        "supported (46) | seeded (22) | csv",
    )
    parser.add_argument(
        "--gate",
        metavar="TASK",
        default=None,
        help="print last_skill and enter_when for one task and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print one JSON object per task instead of the name",
    )
    parser.add_argument(
        "--step-lim",
        metavar="TASK",
        default=None,
        help="print the eval horizon for one task and exit",
    )
    args = parser.parse_args(argv)
    if args.gate is not None:
        rec = task_record(args.gate) if args.gate in ALL_TASKS else None
        if rec is None:
            raise SystemExit(f"unknown RoboTwin task: {args.gate}")
        if not rec["has_critical_gate"]:
            print(f"{args.gate}\tno_gate")
            return 0
        print(f"{args.gate}\t{rec['last_skill']}\t{rec['enter_when']}")
        return 0
    if args.step_lim is not None:
        if args.step_lim not in ALL_TASKS:
            raise SystemExit(f"unknown RoboTwin task: {args.step_lim}")
        print(step_lim(args.step_lim))
        return 0
    for name in list_tasks(args.task_set):
        if args.json:
            print(json.dumps(task_record(name)))
        else:
            print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
