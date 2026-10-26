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

"""Aggregate RoboTwin eval metrics and band tasks for RLT screening."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robotwin_tasks import classify_rlt_band, classify_rlt_phases  # noqa: E402

EVAL_SPLIT_RE = re.compile(r"Evaluation", re.I)
SUCCESS_ONCE_RE = re.compile(r"\bsuccess_once=([0-9.eE+-]+)")
SUCCESS_STD_RE = re.compile(r"\bsuccess_once_std=([0-9.eE+-]+)")
REACH_RE = re.compile(r"\breach_critical_once=([0-9.eE+-]+)")
FAIL_CRIT_RE = re.compile(r"\bfail_in_critical=([0-9.eE+-]+)")
GATE_RE = re.compile(r"\bcritical_gate=([0-9.eE+-]+)")
NUM_TRAJ_RE = re.compile(r"\bnum_trajectories=([0-9]+)")
TASK_DIR_RE = re.compile(
    r"(?:robotwin_openpi_pi05_eval-)?(?P<task>[a-z0-9_]+)"
    r"(?:-(?P<setting>clean|randomized))?(?:-seed(?P<seed>-?\d+))?$"
)


def _mean_std(xs: list[float], *, sample: bool = False) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return mean, 0.0
    denom = len(xs) - 1 if sample else len(xs)
    var = sum((x - mean) ** 2 for x in xs) / denom
    return mean, math.sqrt(var)


def _eval_block(text: str) -> str:
    parts = EVAL_SPLIT_RE.split(text)
    if len(parts) >= 2:
        return parts[-1]
    return text


def parse_log(text: str) -> dict:
    block = _eval_block(text)
    once = SUCCESS_ONCE_RE.findall(block)
    stds = SUCCESS_STD_RE.findall(block)
    reach = REACH_RE.findall(block)
    fail_c = FAIL_CRIT_RE.findall(block)
    gate = GATE_RE.findall(block)
    ns = NUM_TRAJ_RE.findall(block)
    return {
        "success_rate": float(once[-1]) if once else None,
        "success_std": float(stds[-1]) if stds else None,
        "reach_rate": float(reach[-1]) if reach else None,
        "fail_in_critical": float(fail_c[-1]) if fail_c else None,
        "critical_gate": float(gate[-1]) if gate else None,
        "num_trajectories": int(ns[-1]) if ns else None,
    }


def parse_run_dir(path: Path) -> dict:
    if path.is_file() and path.name in {"metrics.log", "eval_embodiment.log"}:
        path = path.parent
    if path.is_dir():
        texts: list[str] = []
        for name in ("eval_embodiment.log", "metrics.log"):
            cand = path / name
            if cand.is_file():
                texts.append(cand.read_text(errors="ignore"))
        if not texts:
            raise FileNotFoundError(f"no metrics/log under {path}")
        parsed = parse_log("\n".join(texts))
        parsed["run_dir"] = str(path)
        parsed.update(_task_from_dirname(path.name))
        return parsed
    parsed = parse_log(path.read_text(errors="ignore"))
    parsed["run_dir"] = str(path)
    parsed.update(_task_from_dirname(path.stem))
    return parsed


def _task_from_dirname(name: str) -> dict[str, str | None]:
    match = TASK_DIR_RE.search(name)
    if not match:
        return {"task": None, "setting": None, "seed": None}
    return {
        "task": match.group("task"),
        "setting": match.group("setting"),
        "seed": match.group("seed"),
    }


def load_runs_tsv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            run_dir = Path(raw["run_dir"])
            parsed = parse_run_dir(run_dir)
            parsed["task"] = raw.get("task") or parsed.get("task")
            parsed["setting"] = raw.get("setting") or parsed.get("setting")
            parsed["seed"] = raw.get("seed") or parsed.get("seed")
            parsed["rc"] = raw.get("rc")
            rows.append(parsed)
    return rows


def format_filter_table(
    rows: list[dict],
    title: str = "RoboTwin π0.5 RLT screen",
) -> str:
    ranked = sorted(
        rows,
        key=lambda item: (
            -(item["success_rate"] if item["success_rate"] is not None else -1.0),
            item.get("task") or "",
        ),
    )
    lines = [
        "",
        "=" * 96,
        title,
        "=" * 96,
        f"{'task':<28}  {'set':<11}  {'SR':>7}  {'Reach':>7}  {'S|R':>7}  "
        f"{'failC':>7}  {'n':>5}  {'band':<14}",
        "-" * 96,
    ]
    keep: list[str] = []
    for row in ranked:
        sr = row["success_rate"]
        reach = row.get("reach_rate")
        fail_c = row.get("fail_in_critical")
        gate = row.get("critical_gate")
        has_gate = None if gate is None else bool(gate > 0.5)
        if sr is None:
            band = "missing"
            sr_s = "    n/a"
        elif has_gate is False:
            band = "no_gate"
            sr_s = f"{sr:7.4f}"
        elif reach is None:
            band = classify_rlt_band(sr) + "/episode"
            sr_s = f"{sr:7.4f}"
        else:
            band = classify_rlt_phases(sr, reach, has_gate=True)
            sr_s = f"{sr:7.4f}"
        if band == "keep":
            keep.append(str(row.get("task") or row["run_dir"]))
        succ_given = None
        if sr is not None and reach is not None and reach > 0:
            succ_given = sr / reach
        if fail_c is None and sr is not None and reach is not None:
            fail_c = max(0.0, reach - sr)
        n = row.get("num_trajectories")
        n_s = str(n) if n is not None else "-"
        reach_s = f"{reach:7.4f}" if reach is not None else "    n/a"
        given_s = f"{succ_given:7.4f}" if succ_given is not None else "    n/a"
        fail_s = f"{fail_c:7.4f}" if fail_c is not None else "    n/a"
        lines.append(
            f"{str(row.get('task') or '-'):<28}  "
            f"{str(row.get('setting') or '-'):<11}  "
            f"{sr_s}  {reach_s}  {given_s}  {fail_s}  {n_s:>5}  {band:<14}"
        )
    lines.append("-" * 96)
    lines.append(
        "RLT keep: Reach>=0.70 and S|R<=0.90 and failC>=0.10. "
        "coarse_fail=never enters critical; saturated=critical already solved; "
        "no_gate/episode=old success_once-only logs, not a reach rate."
    )
    lines.append("keep for RLT Stage-2: " + (", ".join(keep) if keep else "(none)"))
    lines.extend(["=" * 96, ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Eval run directories, log files, or a sweep runs.tsv",
    )
    parser.add_argument(
        "--runs-tsv",
        type=Path,
        default=None,
        help="TSV with columns task, setting, seed, run_dir[, rc]",
    )
    parser.add_argument("--title", default="RoboTwin π0.5 RLT screen")
    parser.add_argument("--append", type=Path, default=None)
    args = parser.parse_args(argv)

    rows: list[dict] = []
    if args.runs_tsv is not None:
        rows.extend(load_runs_tsv(args.runs_tsv))
    for path in args.paths:
        if (
            path.is_file()
            and path.suffix in {".tsv", ".txt"}
            and path.name == "runs.tsv"
        ):
            rows.extend(load_runs_tsv(path))
            continue
        rows.append(parse_run_dir(path))

    if not rows:
        parser.error("pass run directories or --runs-tsv")

    text = format_filter_table(rows, title=args.title)
    sys.stdout.write(text)
    if args.append is not None:
        args.append.parent.mkdir(parents=True, exist_ok=True)
        with args.append.open("a", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
