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

"""Backfill ``meta/episodes_stats.jsonl`` for a LeRobot v2 dataset.

Some LeRobot datasets, such as ``InternRobotics/InternData-Calvin_ABC``, ship
``meta/episodes.jsonl`` but not ``meta/episodes_stats.jsonl``. OpenPI's LeRobot
loader requires the latter file at dataset-open time even though neither
training nor ``toolkits/lerobot/calculate_norm_stats.py`` consumes its values.

This script reconstructs a valid LeRobot v2 ``episodes_stats.jsonl`` without
decoding videos:

  # Fast path, uses dataset-level stats already present in meta/stats.json:
  python toolkits/lerobot/backfill_episodes_stats.py \
      --dataset-root /path/to/InternData-Calvin_ABC

  # Exact path, recomputes per-episode stats from data/*.parquet:
  python toolkits/lerobot/backfill_episodes_stats.py \
      --dataset-root /path/to/InternData-Calvin_ABC --mode from-data

BEHAVIOR/RLinf's ``openpi_rlinf`` BEHAVIOR loader expects the same values nested
under a single ``stats`` field. Use ``--stats-format nested-stats`` for that
path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


_RESERVED_KEYS = {"episode_index", "length", "frame_index", "timestamp", "task_index"}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonlines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size") or 1000)
    data_template = info.get("data_path")
    if not data_template:
        raise KeyError("meta/info.json is missing data_path")
    rel = data_template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return root / rel


def _scalar_features(info: dict[str, Any]) -> list[str]:
    features = info.get("features") or {}
    return [
        key
        for key, feature in features.items()
        if key not in _RESERVED_KEYS
        and str(feature.get("dtype", "")).lower() not in {"video", "image"}
    ]


def _episode_length(
    root: Path, info: dict[str, Any], episode_index: int, record: dict[str, Any]
) -> int:
    length = record.get("length")
    if isinstance(length, int) and length >= 0:
        return length
    path = _episode_path(root, info, episode_index)
    return int(pq.ParquetFile(path).metadata.num_rows)


def _column_to_2d_array(table: Any, column_name: str) -> np.ndarray:
    values = table.column(column_name).to_pylist()
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            f"{column_name} must be a 1D/2D numeric column, got {arr.shape}"
        )
    return arr


def _feature_stats(arr: np.ndarray) -> dict[str, Any]:
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D array, got {arr.shape}")
    return {
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "count": int(arr.shape[0]),
    }


def _make_record(
    episode_index: int,
    length: int,
    feature_stats: dict[str, Any],
    stats_format: str,
) -> dict[str, Any]:
    if stats_format == "nested-stats":
        return {
            "episode_index": episode_index,
            "length": length,
            "stats": feature_stats,
        }
    record: dict[str, Any] = {"episode_index": episode_index, "length": length}
    record.update(feature_stats)
    return record


def _build_from_stats(
    root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    stats_format: str,
) -> list[dict[str, Any]]:
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"{stats_path} not found; use --mode from-data to recompute stats."
        )
    dataset_stats = _load_json(stats_path)
    scalar_features = _scalar_features(info)

    records: list[dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = _episode_length(root, info, episode_index, episode)
        feature_stats: dict[str, Any] = {}
        for feature_key in scalar_features:
            if feature_key in _RESERVED_KEYS:
                continue
            stats = dataset_stats.get(feature_key)
            if not isinstance(stats, dict):
                continue
            if not {"min", "max", "mean", "std"}.issubset(stats):
                continue
            feature_stats[feature_key] = {
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
                "count": length,
            }
        records.append(
            _make_record(episode_index, length, feature_stats, stats_format)
        )
    return records


def _build_from_data(
    root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    stats_format: str,
) -> list[dict[str, Any]]:
    scalar_features = _scalar_features(info)
    records: list[dict[str, Any]] = []

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        parquet_path = _episode_path(root, info, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Episode data not found: {parquet_path}")

        schema_names = set(pq.read_schema(parquet_path).names)
        columns = [key for key in scalar_features if key in schema_names]
        table = pq.read_table(parquet_path, columns=columns) if columns else None
        length = int(pq.ParquetFile(parquet_path).metadata.num_rows)

        feature_stats: dict[str, Any] = {}
        if table is None:
            records.append(
                _make_record(episode_index, length, feature_stats, stats_format)
            )
            continue
        for feature_key in columns:
            if feature_key in _RESERVED_KEYS:
                continue
            arr = _column_to_2d_array(table, feature_key)
            feature_stats[feature_key] = _feature_stats(arr)
        records.append(
            _make_record(episode_index, length, feature_stats, stats_format)
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill meta/episodes_stats.jsonl for a LeRobot v2 dataset."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("auto", "from-stats", "from-data"),
        default="auto",
        help="auto uses meta/stats.json when present, otherwise recomputes from data.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional limit on the number of episodes to backfill.",
    )
    parser.add_argument(
        "--stats-format",
        choices=("lerobot", "nested-stats"),
        default="lerobot",
        help=(
            "lerobot writes feature stats at the top level; nested-stats writes "
            "them under a single 'stats' key for RLinf's BEHAVIOR loader."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing episodes_stats.jsonl.",
    )
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    meta_dir = root / "meta"
    info = _load_json(meta_dir / "info.json")
    episodes = _load_jsonlines(meta_dir / "episodes.jsonl")
    if not episodes:
        raise ValueError(f"No episodes found in {meta_dir / 'episodes.jsonl'}")
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    mode = args.mode
    if mode == "auto":
        mode = "from-stats" if (meta_dir / "stats.json").is_file() else "from-data"

    if mode == "from-stats":
        records = _build_from_stats(root, info, episodes, args.stats_format)
    else:
        records = _build_from_data(root, info, episodes, args.stats_format)

    output_path = meta_dir / "episodes_stats.jsonl"
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"{output_path} already exists; pass --force to overwrite."
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
