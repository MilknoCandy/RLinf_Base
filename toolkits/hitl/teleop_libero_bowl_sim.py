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

"""Standalone LIBERO Bowl teleop through the Jupyter mailbox. No Ray / VLA.

1. Notebook: ``run_console("/tmp/rlinf_hitl")``
2. Terminal::

    bash toolkits/hitl/run_sim_libero_bowl.sh --mode hitl --episodes 1

Toggle Take over and drive the arm. Without hold, the script sends a hover
action (zeros + open gripper) so the scene still updates.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rlinf.envs.libero.human_mailbox import HumanMailbox
from rlinf.envs.libero.utils import get_libero_image, get_libero_wrist_image


def _make_env(task_suite_name: str, task_id: int, seed: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    bddl = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    init_states = task_suite.get_task_init_states(task_id)
    return env, task.language, init_states[0]


_DEFAULT_ROOT = str(pathlib.Path(__file__).resolve().parent / "live")


def run_teleop(
    root: str = _DEFAULT_ROOT,
    *,
    task_suite_name: str = "libero_spatial",
    task_id: int = 9,
    seed: int = 0,
    max_steps: int = 240,
    episodes: int = 3,
) -> None:
    mailbox = HumanMailbox(root, action_dim=7, block=True, timeout_s=120.0)
    env, language, init_state = _make_env(task_suite_name, task_id, seed)
    hover = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    print(f"[hitl-teleop] task={language!r} mailbox={mailbox.root}")
    print("[hitl-teleop] Console must already be running. Take over to move.")
    try:
        for episode in range(episodes):
            env.reset()
            obs = env.set_init_state(init_state)
            success = False
            for step in range(max_steps):
                main = get_libero_image(obs)
                wrist = get_libero_wrist_image(obs)
                seq = mailbox.publish(
                    main,
                    wrist,
                    extra={
                        "elapsed": step,
                        "success": bool(success),
                        "episode": episode,
                        "language": language,
                    },
                )
                cmd = mailbox.wait_command(seq)
                action = cmd.action if cmd.hold else hover
                obs, _reward, done, info = env.step(action.tolist())
                success = bool(info.get("success", False) or done and _reward > 0)
                if step % 10 == 0:
                    print(
                        f"[hitl-teleop] ep={episode} step={step} hold={cmd.hold} "
                        f"success={success} action={np.round(action, 2).tolist()}"
                    )
                if success:
                    print(f"[hitl-teleop] episode {episode} success at step {step}")
                    break
            mailbox.publish(
                get_libero_image(obs),
                get_libero_wrist_image(obs),
                extra={"elapsed": max_steps, "success": success, "episode": episode},
            )
    finally:
        env.close()
    print("[hitl-teleop] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone LIBERO HITL teleop")
    parser.add_argument("--root", default=_DEFAULT_ROOT)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    run_teleop(
        args.root,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        seed=args.seed,
        max_steps=args.max_steps,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
