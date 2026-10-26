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

"""Read official RoboTwin ``demo_clean`` HDF5 into Aloha OpenPI frames.

``pi05_aloha_robotwin`` expects LeRobot keys ``observation.images.cam_*``,
``observation.state`` (14-d), and ``action`` (14-d). Official episodes store
cameras as ``head_camera`` / ``left_camera`` / ``right_camera`` and joints as
``joint_action/{left,right}_{arm,gripper}`` or ``joint_action/vector``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

CAMERA_ALIASES: dict[str, tuple[str, ...]] = {
    "cam_high": ("cam_high", "head_camera", "head", "front_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left"),
    "cam_right_wrist": ("cam_right_wrist", "right_camera", "right"),
}
ACTION_DIM = 14


@dataclass(frozen=True)
class RobotwinDemoFrame:
    """One OpenPI Aloha training frame from a RoboTwin episode."""

    cam_high: np.ndarray
    cam_left_wrist: np.ndarray
    cam_right_wrist: np.ndarray
    state: np.ndarray
    action: np.ndarray
    prompt: str


def task_prompt(task_name: str) -> str:
    """Language prompt used when the HDF5 episode has no instruction."""
    return str(task_name).replace("_", " ").strip()


def find_episode_files(demo_root: str | Path) -> list[Path]:
    """Return ``episode*.hdf5`` under a RoboTwin demo directory."""
    root = Path(demo_root).expanduser()
    if root.is_file() and root.suffix == ".hdf5":
        return [root]
    candidates = (
        root / "data",
        root / "aloha-agilex" / "data",
        root / "aloha_agilex" / "data",
        root,
    )
    found: list[Path] = []
    seen: set[Path] = set()
    for folder in candidates:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("episode*.hdf5")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    if not found:
        raise FileNotFoundError(
            f"no episode*.hdf5 under {root}. Expected official "
            "demo_clean/<task>/aloha-agilex/data/episode0.hdf5"
        )
    return found


def _as_uint8_image(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected HxWx3 RGB, got shape {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _decode_jpeg_batch(raw: np.ndarray) -> np.ndarray:
    import cv2

    frames: list[np.ndarray] = []
    for buf in np.asarray(raw).reshape(-1):
        if isinstance(buf, (bytes, bytearray)):
            encoded = np.frombuffer(buf, dtype=np.uint8)
        else:
            encoded = np.asarray(buf)
            if encoded.dtype != np.uint8:
                encoded = encoded.astype(np.uint8)
            encoded = encoded.reshape(-1)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode failed on a stored camera buffer")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def _load_camera_seq(group: Any, name: str) -> np.ndarray:
    node = group[name]
    if hasattr(node, "keys") and "rgb" in node:
        node = node["rgb"]
    data = node[()]
    array = np.asarray(data)
    if array.dtype == object or array.ndim == 1:
        return _decode_jpeg_batch(array)
    if array.ndim == 4:
        return np.stack([_as_uint8_image(frame) for frame in array], axis=0)
    if array.ndim == 3:
        return _as_uint8_image(array)[None]
    raise ValueError(f"unsupported camera array shape {array.shape} for {name}")


def _first_present(mapping: Any, names: tuple[str, ...]) -> str | None:
    keys = set(mapping.keys())
    for name in names:
        if name in keys:
            return name
    return None


def _observation_group(root: Any) -> Any:
    if "observation" in root:
        return root["observation"]
    if "observations" in root:
        return root["observations"]
    return root


def load_joint_vector(root: Any) -> np.ndarray:
    """Return ``[T, 14]`` left-arm / left-gripper / right-arm / right-gripper."""
    joints = root["joint_action"] if "joint_action" in root else root
    if "vector" in joints:
        vector = np.asarray(joints["vector"][()], dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"joint_action/vector last dim must be {ACTION_DIM}, got {vector.shape}"
            )
        return vector
    required = ("left_arm", "left_gripper", "right_arm", "right_gripper")
    missing = [name for name in required if name not in joints]
    if missing:
        raise KeyError(
            "RoboTwin HDF5 needs joint_action/vector or "
            f"{required}; missing {missing}"
        )
    left_arm = np.asarray(joints["left_arm"][()], dtype=np.float32)
    right_arm = np.asarray(joints["right_arm"][()], dtype=np.float32)
    left_g = np.asarray(joints["left_gripper"][()], dtype=np.float32).reshape(-1, 1)
    right_g = np.asarray(joints["right_gripper"][()], dtype=np.float32).reshape(-1, 1)
    if left_arm.ndim == 1:
        left_arm = left_arm.reshape(1, -1)
        right_arm = right_arm.reshape(1, -1)
    return np.concatenate([left_arm, left_g, right_arm, right_g], axis=1)


def load_camera_sequences(root: Any) -> dict[str, np.ndarray]:
    """Map OpenPI camera names to ``[T, H, W, 3]`` RGB."""
    obs = _observation_group(root)
    if "images" in obs:
        obs = obs["images"]
    loaded: dict[str, np.ndarray] = {}
    for dest, aliases in CAMERA_ALIASES.items():
        name = _first_present(obs, aliases)
        if name is None:
            raise KeyError(
                f"camera {dest} not in HDF5; tried {aliases}, have {list(obs.keys())}"
            )
        loaded[dest] = _load_camera_seq(obs, name)
    return loaded


def load_sidecar_instruction(hdf5_path: str | Path) -> str | None:
    """Read ``instruction/episode_*.json`` next to an official HDF5 episode."""
    path = Path(hdf5_path)
    stem = path.stem
    for folder in (
        path.parent.parent / "instruction",
        path.parent.parent / "instructions",
        path.parent / "instruction",
    ):
        for name in (f"{stem}.json", stem.replace("episode", "episode_") + ".json"):
            candidate = folder / name
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, str) and payload.strip():
                return payload.strip()
            if isinstance(payload, dict):
                for key in ("instruction", "prompt", "language", "task", "text"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    return None


def episode_prompt(root: Any, *, task_name: str, fallback: str | None = None) -> str:
    """Prefer an instruction stored on the episode, else the task language."""
    for key in ("prompt", "instruction", "language_instruction", "task"):
        if key not in root.attrs:
            continue
        value = root.attrs[key]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        text = str(value).strip()
        if text:
            return text
    if fallback:
        return fallback
    return task_prompt(task_name)


def iter_episode_frames(
    hdf5_path: str | Path,
    *,
    task_name: str,
    prompt: str | None = None,
) -> Iterator[RobotwinDemoFrame]:
    """Yield OpenPI frames. Action at ``t`` is the next joint vector."""
    import h5py

    path = Path(hdf5_path)
    with h5py.File(path, "r") as root:
        joints = load_joint_vector(root)
        cameras = load_camera_sequences(root)
        language = episode_prompt(
            root,
            task_name=task_name,
            fallback=prompt or load_sidecar_instruction(path),
        )
    lengths = [joints.shape[0], *(seq.shape[0] for seq in cameras.values())]
    horizon = min(lengths)
    if horizon < 2:
        raise ValueError(f"{path} has {horizon} frames; need at least 2 for action[t+1]")
    for index in range(horizon - 1):
        yield RobotwinDemoFrame(
            cam_high=cameras["cam_high"][index],
            cam_left_wrist=cameras["cam_left_wrist"][index],
            cam_right_wrist=cameras["cam_right_wrist"][index],
            state=joints[index].astype(np.float32),
            action=joints[index + 1].astype(np.float32),
            prompt=language,
        )


def frame_to_lerobot(frame: RobotwinDemoFrame) -> dict[str, Any]:
    """Keys consumed by ``LeRobotAlohaDataConfig`` / OpenPI SFT."""
    return {
        "observation.images.cam_high": frame.cam_high,
        "observation.images.cam_left_wrist": frame.cam_left_wrist,
        "observation.images.cam_right_wrist": frame.cam_right_wrist,
        "observation.state": frame.state,
        "action": frame.action,
        "task": frame.prompt,
    }
