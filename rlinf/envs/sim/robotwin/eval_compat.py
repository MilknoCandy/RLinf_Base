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

"""Eval-only patches for RoboTwin VectorEnv / task files.

Policy eval never calls ``play_once``, and ``SubEnv.setup_task`` swallows every
``setup_demo`` exception in an unbounded retry loop. That combination hangs or
crashes a 50-task sweep on a few tasks; these patches keep sparse-success eval
moving.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from rlinf.utils.logging import get_logger

logger = get_logger()

_PATCHED_ATTR = "_rlinf_eval_patched"
DEFAULT_SETUP_DEMO_MAX_TRIALS = 25

EMPTY_OBJAVERSE_LIST = {
    "item_names": [],
    "list_of_items": {},
    "z_max": {},
    "radius": {},
    "z_offset": {},
}


def objaverse_list_path(assets_path: str | os.PathLike[str]) -> Path:
    """``{assets_path}/assets/objects/objaverse/list.json`` (ASSETS_PATH is the RoboTwin root)."""
    return (
        Path(assets_path).expanduser()
        / "assets"
        / "objects"
        / "objaverse"
        / "list.json"
    )


def _shared_asset_candidates() -> list[Path]:
    roots: list[Path] = []
    dataroot = os.environ.get("RLinf_DATAROOT")
    if dataroot:
        roots.append(Path(dataroot) / "RoboTwin" / "assets")
    roots.extend(
        [
            Path(
                "/opt/huawei/dataset/recaption60M/Datasets_for_Intern/"
                "VLA_datasets/RoboTwin/assets"
            ),
            Path(
                "/home/ma-user/work/dataset/recaption60M/Datasets_for_Intern/"
                "VLA_datasets/RoboTwin/assets"
            ),
        ]
    )
    seen: set[str] = set()
    unique: list[Path] = []
    for path in roots:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def ensure_robotwin_assets(assets_path: str | os.PathLike[str]) -> Path:
    """Make ``objaverse/list.json`` importable before VectorEnv loads.

    MTP clones often have the RoboTwin code without the shared ``assets``
    symlink. Clean eval still imports ``rand_create_cluttered_actor``, which
    reads this file at module import.
    """
    root = Path(assets_path).expanduser()
    list_json = objaverse_list_path(root)
    if list_json.is_file():
        return list_json

    assets_dir = root / "assets"
    if not assets_dir.exists():
        for shared in _shared_asset_candidates():
            if (shared / "objects" / "objaverse" / "list.json").is_file():
                assets_dir.symlink_to(shared)
                logger.info(
                    "robotwin eval: linked %s -> %s", shared, assets_dir
                )
                return list_json

    if list_json.is_file():
        return list_json

    if (root / "assets" / "embodiments" / "aloha-agilex").exists():
        list_json.parent.mkdir(parents=True, exist_ok=True)
        list_json.write_text(
            json.dumps(EMPTY_OBJAVERSE_LIST), encoding="utf-8"
        )
        same_json = root / "assets" / "objects" / "same.json"
        if not same_json.is_file():
            same_json.write_text("{}", encoding="utf-8")
        logger.warning(
            "robotwin eval: wrote empty %s; cluttered_table eval has no "
            "objaverse objects. Link shared assets for Hard.",
            list_json,
        )
        return list_json

    tried = ", ".join(str(p) for p in _shared_asset_candidates())
    raise FileNotFoundError(
        f"RoboTwin assets missing {list_json}. MTP clones need the shared "
        f"assets symlink: ln -sfn <dataroot>/RoboTwin/assets {assets_dir}. "
        f"Looked in: {tried}."
    )


def apply_robotwin_eval_patches(*, setup_demo_max_trials: int = DEFAULT_SETUP_DEMO_MAX_TRIALS) -> None:
    """Patch RoboTwin after its modules are importable, before VectorEnv()."""
    import robotwin.envs.vector_env as vector_env

    patch_setup_task_retry_limit(
        vector_env.SubEnv,
        class_decorator=vector_env.class_decorator,
        max_trials=setup_demo_max_trials,
    )
    patch_dense_reward_constructor()
    patch_policy_eval_check_success()
    patch_critical_phase_info()


def patch_dense_reward_constructor() -> None:
    """Accept ``Reward(subtasks=..., transition_rewards=...)``.

    RLinf_support's ``Reward`` is a factory with no ``__init__``. ``place_fan``
    (and similar) still call that constructor; VectorEnv then retries forever.
    Sparse eval does not read ``self.reward``.
    """
    from envs.reward import Reward

    if getattr(Reward, _PATCHED_ATTR, False):
        return
    if Reward.__init__ is not object.__init__:
        return

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.subtasks = kwargs.get("subtasks", args[0] if args else [])
        self.transition_rewards = kwargs.get("transition_rewards", [])

    Reward.__init__ = __init__  # type: ignore[method-assign]
    setattr(Reward, _PATCHED_ATTR, True)
    logger.info("robotwin eval: Reward() constructor patched for dense-tree tasks")


def patch_setup_task_retry_limit(
    sub_env_cls: type,
    *,
    class_decorator: Callable[[str], Any],
    max_trials: int = DEFAULT_SETUP_DEMO_MAX_TRIALS,
) -> None:
    """Replace unbounded ``setup_demo`` retries with a hard cap and a log."""
    existing = getattr(sub_env_cls, "setup_task", None)
    if existing is not None and getattr(existing, _PATCHED_ATTR, False):
        return

    def setup_task(self) -> None:
        self.close()
        self.task = class_decorator(self.task_name)

        with self.global_lock:
            with self.lock:
                trial_seed = self.env_seed
                last_error: BaseException | None = None
                for trial in range(max_trials):
                    task = class_decorator(self.task_name)
                    try:
                        task.setup_demo(
                            now_ep_num=trial_seed,
                            seed=trial_seed,
                            is_test=True,
                            **self.args,
                        )
                        episode_info = task.get_info()
                    except Exception as exc:
                        last_error = exc
                        logger.warning(
                            "robotwin eval: %s setup_demo seed=%s trial=%s/%s failed: %s",
                            self.task_name,
                            trial_seed,
                            trial + 1,
                            max_trials,
                            exc,
                        )
                        task.close_env()
                        trial_seed += 1
                        continue
                    task.close_env()
                    self.episode_info_list = [episode_info]
                    return
                raise RuntimeError(
                    f"RoboTwin {self.task_name} setup_demo failed after "
                    f"{max_trials} seeds (last={trial_seed - 1}): {last_error!r}"
                ) from last_error

    setattr(setup_task, _PATCHED_ATTR, True)
    sub_env_cls.setup_task = setup_task


def patch_critical_phase_info() -> None:
    """Put last-skill *progress* (not the spawn-true raw gate) on SubEnv info."""
    from robotwin.envs.vector_env import SubEnv

    from rlinf.envs.sim.robotwin.critical_phase import progress_into_critical

    def _annotate(result: dict[str, Any], task: Any) -> dict[str, Any]:
        info = result.get("info")
        if not isinstance(info, dict):
            return result
        entered, gate = progress_into_critical(task)
        info["in_critical"] = bool(entered)
        info["critical_gate"] = bool(gate)
        return result

    orig_step = SubEnv.step
    if not getattr(orig_step, _PATCHED_ATTR, False):

        def step(self, actions):
            return _annotate(orig_step(self, actions), self.task)

        setattr(step, _PATCHED_ATTR, True)
        SubEnv.step = step

    orig_full = SubEnv.step_with_full_obs
    if not getattr(orig_full, _PATCHED_ATTR, False):

        def step_with_full_obs(self, actions):
            return _annotate(orig_full(self, actions), self.task)

        setattr(step_with_full_obs, _PATCHED_ATTR, True)
        SubEnv.step_with_full_obs = step_with_full_obs


def patch_policy_eval_check_success() -> None:
    """Fill ``arm_tag`` / ``origin_z`` that expert ``play_once`` would have set."""
    _wrap_open_laptop()
    _wrap_actor_x_arm_tag("place_object_scale", "object")
    _wrap_actor_x_arm_tag("put_object_cabinet", "object", origin_z_attr="origin_z")


def _wrap_check_success(cls: type, ensure: Callable[[Any], None]) -> None:
    orig = cls.check_success
    if getattr(orig, _PATCHED_ATTR, False):
        return

    def check_success(self, *args: Any, **kwargs: Any):
        ensure(self)
        return orig(self, *args, **kwargs)

    setattr(check_success, _PATCHED_ATTR, True)
    cls.check_success = check_success


def _wrap_open_laptop() -> None:
    from envs.open_laptop import open_laptop
    from envs.utils import ArmTag, get_face_prod

    def ensure(task: Any) -> None:
        if getattr(task, "arm_tag", None) is not None:
            return
        face_prod = get_face_prod(task.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        task.arm_tag = ArmTag("left" if face_prod > 0 else "right")

    _wrap_check_success(open_laptop, ensure)


def _wrap_actor_x_arm_tag(
    module_name: str, actor_attr: str, *, origin_z_attr: str | None = None
) -> None:
    module = __import__(f"envs.{module_name}", fromlist=[module_name])
    cls = getattr(module, module_name)
    from envs.utils import ArmTag

    def ensure(task: Any) -> None:
        actor = getattr(task, actor_attr)
        if getattr(task, "arm_tag", None) is None:
            task.arm_tag = ArmTag(
                "right" if actor.get_pose().p[0] > 0 else "left"
            )
        if origin_z_attr is not None and not hasattr(task, origin_z_attr):
            setattr(task, origin_z_attr, actor.get_pose().p[2])

    _wrap_check_success(cls, ensure)
