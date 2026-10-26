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

"""Download RoboTwin ``demo_clean`` HDF5 from Hugging Face.

Official archive::

    hf://datasets/TianxingChen/RoboTwin2.0/dataset/<task>/demo_clean.zip

Extracts to::

    <data-home>/demo_clean/<task>/aloha_agilex/data/episode_*.hdf5

    python scripts/robotwin/download_demo_clean.py --task adjust_bottle \\
      --data-home /path/to/robotwin_data
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rlinf.data.datasets.openpi.robotwin_hdf5 import find_episode_files  # noqa: E402

HF_REPO_ID = "TianxingChen/RoboTwin2.0"
HF_ARCHIVE_NAME = "demo_clean.zip"
EMBODIMENT = "aloha_agilex"


def demo_ready(data_home: Path, task: str) -> Path | None:
    """Return the demo root when HDF5 already exists."""
    for candidate in (
        data_home / "demo_clean" / task,
        data_home / "demo_clean" / task / EMBODIMENT,
        data_home / "demo_clean" / task / "aloha-agilex",
        data_home / task,
    ):
        try:
            find_episode_files(candidate)
        except FileNotFoundError:
            continue
        return candidate
    return None


def _locate_payload(extract_root: Path, task: str) -> Path:
    matches = sorted(
        path
        for path in extract_root.rglob(EMBODIMENT)
        if path.is_dir() and (path / "data").is_dir()
    )
    if not matches:
        matches = sorted(
            path
            for path in extract_root.rglob("aloha-agilex")
            if path.is_dir() and (path / "data").is_dir()
        )
    if not matches:
        raise FileNotFoundError(
            f"demo_clean.zip for {task} has no {EMBODIMENT}/data payload"
        )
    preferred = [path for path in matches if path.parent.name == task]
    return (preferred or matches)[0]


def download_task(
    *,
    task: str,
    data_home: Path,
    cache_dir: Path,
    revision: str,
    force: bool,
) -> Path:
    existing = demo_ready(data_home, task)
    if existing is not None and not force:
        print(f"[robotwin-demo] reuse {existing}")
        return existing

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "huggingface_hub is required to download RoboTwin demos"
        ) from exc

    filename = f"dataset/{task}/{HF_ARCHIVE_NAME}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=str(cache_dir),
            force_download=force,
        )
    )
    dest = data_home / "demo_clean" / task / EMBODIMENT
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.parent / f".{task}-extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        with ZipFile(archive) as zipped:
            zipped.extractall(tmp)
        payload = _locate_payload(tmp, task)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(payload, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if demo_ready(data_home, task) is None:
        raise FileNotFoundError(
            f"download finished but no episode*.hdf5 under {dest}"
        )
    print(f"[robotwin-demo] wrote {dest} from {filename}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", dest="tasks", required=True)
    parser.add_argument("--data-home", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    data_home = Path(args.data_home).expanduser()
    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else data_home / "download_cache"
    )
    for task in args.tasks:
        download_task(
            task=task,
            data_home=data_home,
            cache_dir=cache_dir,
            revision=args.revision,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
