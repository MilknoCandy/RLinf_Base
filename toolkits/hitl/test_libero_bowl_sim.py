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

"""LIBERO-Pro Bowl OffScreenRenderEnv test. No Ray / VLA / training.

Launch through the generic (or exp) EGL wrapper::

    bash toolkits/hitl/run_sim_libero_bowl.sh --env-file toolkits/hitl/env.local.sh
    bash toolkits/hitl/run_sim_libero_bowl.sh --mode hitl --episodes 1
    bash toolkits/hitl/run_sim_libero_bowl_exp.sh --mode hitl --episodes 1
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_ROOT = str(pathlib.Path(__file__).resolve().parent / "live")


def _install_egl_shim() -> None:
    from rlinf.utils.robosuite_compat import install_robosuite_egl_device_shim

    install_robosuite_egl_device_shim()


def _image(obs: dict, key: str) -> np.ndarray:
    return np.asarray(obs[key])[::-1, ::-1]


def _make_pro_env(task_suite_name: str, task_id: int, seed: int):
    """Match ``LiberoEnv.get_env_fns``: remap ``libero`` → ``liberopro``."""
    os.environ.setdefault("LIBERO_TYPE", "pro")
    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]

    real_pkg = importlib.import_module("liberopro")
    real_core = importlib.import_module("liberopro.liberopro")
    real_bench = importlib.import_module("liberopro.liberopro.benchmark")
    real_envs = importlib.import_module("liberopro.liberopro.envs")
    sys.modules["libero"] = real_pkg
    sys.modules["libero.libero"] = real_core
    sys.modules["libero.libero.benchmark"] = real_bench
    sys.modules["libero.libero.envs"] = real_envs

    loaded = os.path.dirname(real_core.__file__)
    os.environ["LIBERO_ASSET_ROOT"] = os.path.join(loaded, "assets")
    os.environ["LIBERO_BDDL_PATH"] = os.path.join(loaded, "bddl_files")
    os.environ["LIBERO_INIT_STATES_PATH"] = os.path.join(loaded, "init_files")

    get_libero_path = real_core.get_libero_path
    task_suite = real_bench.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    bddl = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    print(f"[hitl-sim] language={task.language!r}")
    print(f"[hitl-sim] bddl={bddl}")
    print(f"[hitl-sim] assets={os.environ['LIBERO_ASSET_ROOT']}")
    env = real_envs.OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    init_states = task_suite.get_task_init_states(task_id)
    return env, task.language, init_states[0]


def _publish(mailbox, obs, extra: dict) -> int:
    return mailbox.publish(
        _image(obs, "agentview_image"),
        _image(obs, "robot0_eye_in_hand_image"),
        extra=extra,
    )


def run_smoke(root: str, env, language: str, init_state, steps: int) -> None:
    from rlinf.envs.libero.human_mailbox import HumanMailbox

    mailbox = HumanMailbox(root, action_dim=7, block=False, timeout_s=0.1)
    env.reset()
    obs = env.set_init_state(init_state)
    hover = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    print(
        f"[hitl-sim] smoke steps={steps} main={obs['agentview_image'].shape} "
        f"wrist={obs['robot0_eye_in_hand_image'].shape} mailbox={mailbox.root}"
    )
    for step in range(int(steps)):
        seq = _publish(
            mailbox,
            obs,
            {"elapsed": step, "success": False, "mode": "sim-smoke", "language": language},
        )
        obs, reward, done, info = env.step(hover.tolist())
        print(
            f"[hitl-sim] step={step} seq={seq} reward={reward} done={done} "
            f"success={info.get('success', False)}"
        )
    _publish(
        mailbox,
        obs,
        {"elapsed": steps, "success": False, "mode": "sim-smoke", "language": language},
    )
    print("[hitl-sim] smoke ok — open the notebook to see Bowl frames")


def run_hitl(
    root: str,
    env,
    language: str,
    init_state,
    *,
    max_steps: int,
    episodes: int,
) -> None:
    from rlinf.envs.libero.human_mailbox import HumanMailbox

    mailbox = HumanMailbox(root, action_dim=7, block=True, timeout_s=120.0)
    hover = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    print(f"[hitl-sim] hitl mailbox={mailbox.root}  Space/set_cmd to take over")
    for episode in range(int(episodes)):
        env.reset()
        obs = env.set_init_state(init_state)
        success = False
        for step in range(int(max_steps)):
            seq = _publish(
                mailbox,
                obs,
                {
                    "elapsed": step,
                    "success": success,
                    "episode": episode,
                    "language": language,
                    "mode": "sim-hitl",
                },
            )
            cmd = mailbox.wait_command(seq)
            action = cmd.action if cmd.hold else hover
            obs, reward, done, info = env.step(action.tolist())
            success = bool(info.get("success", False) or done and reward > 0)
            if step % 5 == 0:
                print(
                    f"[hitl-sim] ep={episode} step={step} hold={cmd.hold} "
                    f"success={success} action={np.round(action, 2).tolist()}"
                )
            if success:
                print(f"[hitl-sim] episode {episode} success at step {step}")
                break
        _publish(
            mailbox,
            obs,
            {"elapsed": max_steps, "success": success, "episode": episode, "mode": "sim-hitl"},
        )
    print("[hitl-sim] hitl done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp EGL LIBERO-Pro Bowl sim test")
    parser.add_argument("--mode", choices=("smoke", "hitl"), default="smoke")
    parser.add_argument("--root", default=_DEFAULT_ROOT)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    print(
        f"[hitl-sim] MUJOCO_GL={os.environ.get('MUJOCO_GL')} "
        f"__EGL_VENDOR_LIBRARY_DIRS={os.environ.get('__EGL_VENDOR_LIBRARY_DIRS')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')}"
    )
    _install_egl_shim()
    env, language, init_state = _make_pro_env(
        args.task_suite_name, args.task_id, args.seed
    )
    try:
        if args.mode == "smoke":
            run_smoke(args.root, env, language, init_state, args.steps)
        else:
            run_hitl(
                args.root,
                env,
                language,
                init_state,
                max_steps=args.max_steps,
                episodes=args.episodes,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
