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

"""Download official RoboTwin mixed LeRobot v2.1 for RLT Stage-1 SFT.

Stage-1 is 50-task co-training. The matching dump is::

    hf://datasets/TianxingChen/RoboTwin2.0/lerobot_dataset/RoboTwin_lerobot_v21.zip

Do **not** use ``RoboTwin_lerobot_v30.zip``: this env's LeRobot is v2.1 and
``pi05_aloha_robotwin`` reads v2.1 parquet.

    python scripts/robotwin/download_lerobot_v21.py \\
      --output /path/to/robotwin_lerobot_v21
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rlinf.data.storage.lerobot.paths import find_lerobot_dataset_roots  # noqa: E402

HF_REPO_ID = "TianxingChen/RoboTwin2.0"
HF_FILENAME = "lerobot_dataset/RoboTwin_lerobot_v21.zip"
MARKER_NAME = "rlinf_lerobot_root.txt"


def _episode_count(root: Path) -> int:
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return int(info.get("total_episodes") or info.get("total_frames") or 0)


def resolve_mixed_root(extract_dir: Path) -> Path:
    """Pick the mixed Stage-1 dataset under an extracted zip."""
    marker = extract_dir / MARKER_NAME
    if marker.is_file():
        marked = Path(marker.read_text(encoding="utf-8").strip())
        if (marked / "meta" / "info.json").is_file():
            return marked
    roots = find_lerobot_dataset_roots(extract_dir)
    if not roots:
        raise FileNotFoundError(
            f"no LeRobot meta/info.json under {extract_dir}. "
            "Expected the official RoboTwin_lerobot_v21.zip layout."
        )
    if len(roots) == 1:
        return roots[0]
    mixed = extract_dir / "mixed"
    if (mixed / "meta" / "info.json").is_file():
        return mixed.resolve()
    from toolkits.lerobot.merge_lerobot_datasets import merge_lerobot_datasets

    print(
        f"[robotwin-lerobot-v21] zip has {len(roots)} datasets; merging for mix SFT",
        file=sys.stderr,
    )
    n_ep = merge_lerobot_datasets([extract_dir], mixed)
    if n_ep < 1 or not (mixed / "meta" / "info.json").is_file():
        raise RuntimeError(f"merge of {len(roots)} LeRobot datasets wrote 0 episodes")
    return mixed.resolve()


def download_v21(
    *,
    output: Path,
    cache_dir: Path,
    revision: str,
    force: bool,
) -> Path:
    if not force:
        try:
            root = resolve_mixed_root(output)
        except FileNotFoundError:
            root = None
        else:
            print(f"[robotwin-lerobot-v21] reuse {root}", file=sys.stderr)
            (output / MARKER_NAME).write_text(str(root), encoding="utf-8")
            return root

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "huggingface_hub is required to download RoboTwin_lerobot_v21.zip"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=revision,
            filename=HF_FILENAME,
            local_dir=str(cache_dir),
            force_download=force,
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    print(f"[robotwin-lerobot-v21] extract {archive} -> {output}", file=sys.stderr)
    with ZipFile(archive) as zipped:
        zipped.extractall(output)
    root = resolve_mixed_root(output)
    (output / MARKER_NAME).write_text(str(root), encoding="utf-8")
    print(
        f"[robotwin-lerobot-v21] dataset {root} "
        f"(episodes={_episode_count(root)})",
        file=sys.stderr,
    )
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--print-root",
        action="store_true",
        help="print the resolved dataset root on stdout",
    )
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser()
    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else output.parent / "robotwin_data" / "download_cache"
    )
    root = download_v21(
        output=output,
        cache_dir=cache_dir,
        revision=args.revision,
        force=args.force,
    )
    if args.print_root:
        print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
