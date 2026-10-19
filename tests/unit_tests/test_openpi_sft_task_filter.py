# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for OpenPI SFT single-task episode filtering."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
    collect_sft_task_filter,
    normalize_sft_task_name,
    resolve_lerobot_task_filter_episodes,
)

_WOODEN_CABINET = (
    "pick up the black bowl on the wooden cabinet and place it on the plate"
)
_WOODEN_CABINET_LIBERO = (
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
)


def _write_episodes(root: Path, rows: list[dict]) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}", encoding="utf-8")
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_normalize_sft_task_name_equates_underscores_and_spaces():
    assert normalize_sft_task_name(_WOODEN_CABINET) == normalize_sft_task_name(
        _WOODEN_CABINET_LIBERO
    )


def test_resolve_task_filter_keeps_matching_episodes(tmp_path: Path):
    _write_episodes(
        tmp_path,
        [
            {"episode_index": 10, "tasks": ["put the bowl on the plate"], "length": 3},
            {"episode_index": 11, "tasks": [_WOODEN_CABINET], "length": 5},
            {"episode_index": 12, "tasks": [_WOODEN_CABINET], "length": 4},
            {
                "episode_index": 13,
                "tasks": [
                    "pick up the black bowl in the top drawer of the wooden "
                    "cabinet and place it on the plate"
                ],
                "length": 6,
            },
        ],
    )
    got = resolve_lerobot_task_filter_episodes(tmp_path, [_WOODEN_CABINET_LIBERO])
    assert got == [11, 12]


def test_resolve_task_filter_missing_prompt_raises(tmp_path: Path):
    _write_episodes(
        tmp_path,
        [{"episode_index": 0, "tasks": ["put the bowl on the plate"], "length": 2}],
    )
    with pytest.raises(ValueError, match="matched 0 episodes"):
        resolve_lerobot_task_filter_episodes(tmp_path, [_WOODEN_CABINET])


def test_collect_sft_task_filter_prefers_path_override():
    cfg = OmegaConf.create({"data": {"task_filter": ["from-data"]}})
    paths = [{"dataset_path": "/tmp/ds", "task_filter": [_WOODEN_CABINET]}]
    assert collect_sft_task_filter(cfg, paths) == [_WOODEN_CABINET]


def test_collect_sft_task_filter_from_data_key():
    cfg = OmegaConf.create({"data": {"task_filter": [_WOODEN_CABINET]}})
    assert collect_sft_task_filter(cfg, [{"dataset_path": "/tmp/ds"}]) == [
        _WOODEN_CABINET
    ]
