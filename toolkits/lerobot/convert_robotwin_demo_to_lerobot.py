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

"""Convert RoboTwin ``demo_clean`` HDF5 to the LeRobot layout OpenPI SFT loads.

Official episodes::

    data/demo_clean/<task>/aloha-agilex/data/episode0.hdf5

Output keys match ``LeRobotAlohaDataConfig`` (``cam_high`` / wrists / 14-d
state+action). Copies ``--norm-stats`` when given so Stage-1 keeps the same
quantile stats as the π₀.₅ co-train.

    python toolkits/lerobot/convert_robotwin_demo_to_lerobot.py \
      --task adjust_bottle \
      --demo-root /path/to/demo_clean/adjust_bottle \
      --output /path/to/lerobot/adjust_bottle \
      --norm-stats /path/to/pi05_openpi/physical-intelligence/robotwin/norm_stats.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rlinf.data.datasets.openpi.robotwin_hdf5 import (  # noqa: E402
    ACTION_DIM,
    find_episode_files,
    frame_to_lerobot,
    iter_episode_frames,
    task_prompt,
)
from rlinf.data.storage.lerobot.compat import add_frame_to_dataset  # noqa: E402


def _lerobot_cls():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def _image_feature(shape: tuple[int, int, int]) -> dict:
    return {
        "dtype": "image",
        "shape": list(shape),
        "names": ["height", "width", "channel"],
    }


def convert_demo(
    *,
    task: str,
    demo_root: Path,
    output: Path,
    fps: int,
    prompt: str | None,
    norm_stats: Path | None,
    overwrite: bool,
) -> Path:
    episodes = find_episode_files(demo_root)
    language = prompt or task_prompt(task)
    first = next(iter_episode_frames(episodes[0], task_name=task, prompt=language))
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    create_kwargs = {
        "repo_id": output.name,
        "robot_type": "aloha",
        "fps": fps,
        "features": {
            "observation.images.cam_high": _image_feature(first.cam_high.shape),
            "observation.images.cam_left_wrist": _image_feature(
                first.cam_left_wrist.shape
            ),
            "observation.images.cam_right_wrist": _image_feature(
                first.cam_right_wrist.shape
            ),
            "observation.state": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["action"],
            },
        },
        "root": output,
        "use_videos": False,
    }
    create = _lerobot_cls().create
    accepted = set(inspect.signature(create).parameters)
    dataset = create(
        **{key: value for key, value in create_kwargs.items() if key in accepted}
    )
    n_frames = 0
    for episode in episodes:
        for frame in iter_episode_frames(episode, task_name=task, prompt=language):
            add_frame_to_dataset(dataset, frame_to_lerobot(frame))
            n_frames += 1
        dataset.save_episode()
    if n_frames < 1:
        raise RuntimeError(f"converted 0 frames from {demo_root}")

    if norm_stats is not None:
        src = Path(norm_stats).expanduser()
        if src.is_dir():
            src = src / "norm_stats.json"
        if not src.is_file():
            raise FileNotFoundError(f"norm_stats not found: {src}")
        dest = output / "meta" / "norm_stats.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        shutil.copy2(src, output / "norm_stats.json")

    meta = {
        "task": task,
        "prompt": language,
        "n_episodes": len(episodes),
        "n_frames": n_frames,
        "action_dim": ACTION_DIM,
        "source": str(demo_root),
    }
    (output / "rlinf_robotwin_convert.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return output


def is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() or (
        path / "meta" / "episodes.jsonl"
    ).is_file()


def resolve_demo_root(task: str, demo_root: str | None, data_home: str | None) -> Path:
    if demo_root:
        return Path(demo_root).expanduser()
    if not data_home:
        raise ValueError("pass --demo-root or --data-home")
    home = Path(data_home).expanduser()
    for candidate in (
        home / "demo_clean" / task,
        home / "demo_clean" / task / "aloha-agilex",
        home / "demo_clean" / task / "aloha_agilex",
        home / task,
        home / task / "aloha-agilex",
    ):
        try:
            find_episode_files(candidate)
        except FileNotFoundError:
            continue
        return candidate
    raise FileNotFoundError(
        f"no HDF5 for {task} under {home}. Download official clean demos:\n"
        f"  bash scripts/download_xpolicylab_data.sh {task}\n"
        "and point --data-home at the RoboTwin data root that contains demo_clean/."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="RoboTwin task name")
    parser.add_argument("--demo-root", default=None)
    parser.add_argument("--data-home", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="exit 0 when --output is already a LeRobot dataset",
    )
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser()
    if args.skip_if_exists and is_lerobot_dataset(output):
        print(f"[robotwin-sft] reuse {output}")
        return 0
    demo_root = resolve_demo_root(args.task, args.demo_root, args.data_home)
    convert_demo(
        task=args.task,
        demo_root=demo_root,
        output=output,
        fps=args.fps,
        prompt=args.prompt,
        norm_stats=Path(args.norm_stats) if args.norm_stats else None,
        overwrite=args.overwrite or args.skip_if_exists,
    )
    print(f"[robotwin-sft] wrote {output} from {demo_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
