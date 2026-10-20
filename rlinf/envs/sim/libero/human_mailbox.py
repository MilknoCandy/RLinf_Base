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

"""File mailbox so a Jupyter notebook can take over a headless LIBERO env.

The env writes ``state.json`` + JPEG frames. The notebook writes ``cmd.json``.
No SSH tunnel or display on the GPU node is required: Jupyter already has the
browser. Enable with ``env.train.human_mailbox.enable: True`` and
``total_num_envs: 1``. ``cmd.json`` ``controller`` is ``human`` (keyboard
pose: xy/z/rpy) or ``expert`` (hold latches ZAP on the next action chunk).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

_STATE_NAME = "state.json"
_CMD_NAME = "cmd.json"
_MAIN_NAME = "frame_main.jpg"
_WRIST_NAME = "frame_wrist.jpg"
_EPISODES_NAME = "episodes.jsonl"
_NUMBERED_FRAME_RE = re.compile(r"^frame_(main|wrist)_(\d+)\.jpg$")
_KEEP_NUMBERED_FRAMES = 4


def _atomic_write_bytes(path: str, data: bytes) -> None:
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _atomic_write_text(path: str, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _numbered_name(kind: str, seq: int) -> str:
    return f"frame_{kind}_{int(seq)}.jpg"


def _cleanup_numbered_frames(
    root: str, seq: int, keep: int = _KEEP_NUMBERED_FRAMES
) -> None:
    cutoff = int(seq) - int(keep)
    if cutoff <= 0:
        return
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        match = _NUMBERED_FRAME_RE.match(name)
        if match is None:
            continue
        if int(match.group(2)) <= cutoff:
            try:
                os.remove(os.path.join(root, name))
            except OSError:
                pass


def _encode_jpeg(image: np.ndarray) -> bytes:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 3:
        pass
    elif array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("human_mailbox needs Pillow to write JPEG frames") from exc
    import io

    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


_POSE_IDLE_ABS = 0.02


def apply_human_override(
    policy_actions: np.ndarray,
    human: np.ndarray,
    *,
    hold: bool,
    freeze_gripper: bool = True,
    inherit_gripper: bool = True,
    already_holding: bool = False,
    gripper_latch: float = -1.0,
) -> tuple[np.ndarray | None, bool, float]:
    """Apply keyboard/SpaceMouse takeover to one env step.

    Pressed pose axes replace the policy (not added). Axes below
    ``_POSE_IDLE_ABS`` inherit the current π/actor command so an unpressed
    keyboard axis does not write 0 and freeze that DoF. All-idle pose skips
    the override. Bowl (``freeze_gripper=True``) keeps the gripper on the
    policy/VLA command (bowl HITL freezes gripper).
    """
    if not hold:
        return None, False, float(gripper_latch)
    actions = np.asarray(policy_actions, dtype=np.float32).copy()
    human = np.asarray(human, dtype=np.float32).copy()
    pose = human[:-1] if human.size > 1 else human
    if pose.size and float(np.max(np.abs(pose))) < _POSE_IDLE_ABS:
        return None, False, float(gripper_latch)
    if actions.ndim == 1:
        policy_grip = float(actions[-1]) if actions.size else -1.0
        pose_dim = max(int(min(actions.size, human.size) - 1), 0)
        merged = actions.copy()
        if pose_dim > 0:
            take = np.abs(human[:pose_dim]) >= _POSE_IDLE_ABS
            merged[:pose_dim] = np.where(take, human[:pose_dim], actions[:pose_dim])
        merged[-1] = human[-1] if human.size else policy_grip
    else:
        policy_grip = float(actions[0, -1]) if actions.size else -1.0
        pose_dim = max(int(min(actions.shape[-1], human.size) - 1), 0)
        merged = actions.copy()
        if pose_dim > 0:
            take = np.abs(human[:pose_dim]) >= _POSE_IDLE_ABS
            merged[0, :pose_dim] = np.where(
                take, human[:pose_dim], actions[0, :pose_dim]
            )
        if human.size:
            merged[0, -1] = human[-1]
    latch = float(gripper_latch)
    if freeze_gripper:
        if merged.ndim == 1:
            merged[-1] = policy_grip
        else:
            merged[0, -1] = policy_grip
        latch = policy_grip
    else:
        if not already_holding:
            latch = float(gripper_latch)
        if inherit_gripper:
            if merged.ndim == 1:
                merged[-1] = latch
            else:
                merged[0, -1] = latch
        else:
            latch = float(merged[-1] if merged.ndim == 1 else merged[0, -1])
    return merged, True, latch


@dataclass
class HumanCommand:
    seq: int
    hold: bool
    action: np.ndarray
    # When True, env keeps the latched gripper instead of action[-1].
    # Default keyboard UI leaves gripper at -1; Space/WASD must not force open.
    inherit_gripper: bool = True
    # One-shot give-up: UI increments skip_id and keeps skip=True until
    # state.json reports skip_id_consumed >= skip_id.
    skip: bool = False
    skip_id: int = 0
    # Sticky: env wait_command does not return until pause is cleared.
    pause: bool = False
    # HITL source: ``human`` (keyboard pose) or ``expert`` (ZAP on hold).
    controller: str = "human"

    @classmethod
    def idle(cls, action_dim: int, seq: int = -1) -> HumanCommand:
        return cls(
            seq=int(seq),
            hold=False,
            action=np.zeros((int(action_dim),), dtype=np.float32),
            inherit_gripper=True,
            skip=False,
            skip_id=0,
            pause=False,
            controller="human",
        )


def normalize_hitl_controller(raw: Any) -> str:
    """Map UI / cmd.json controller names to ``human`` or ``expert``."""
    text = str(raw or "human").strip().lower()
    if text in {"expert", "zap", "teacher"}:
        return "expert"
    return "human"


def mailbox_expert_hold(cmd: HumanCommand | None) -> bool:
    """True when the operator latched ZAP takeover (not keyboard pose)."""
    return (
        cmd is not None
        and normalize_hitl_controller(cmd.controller) == "expert"
        and bool(cmd.hold)
    )


def mailbox_applies_wasd(cmd: HumanCommand | None) -> bool:
    """True when the operator is driving pose with the keyboard / sliders."""
    return (
        cmd is not None
        and normalize_hitl_controller(cmd.controller) == "human"
        and bool(cmd.hold)
    )


def _cmd_blocks_step(cmd: HumanCommand | None) -> bool:
    """Pause holds the env; skip still goes through so the operator can leave."""
    return cmd is not None and bool(cmd.pause) and not bool(cmd.skip)


def consume_human_skip(
    cmd: HumanCommand | None, consumed_id: int
) -> tuple[bool, int]:
    """Apply skip once per rising ``skip_id`` so a sticky cmd.json cannot chain-skip."""
    if cmd is None or not cmd.skip:
        return False, int(consumed_id)
    skip_id = int(cmd.skip_id)
    if skip_id <= int(consumed_id):
        return False, int(consumed_id)
    return True, skip_id


class HumanMailbox:
    """Shared-directory protocol between the env process and a Jupyter cell."""

    def __init__(
        self,
        root: str,
        *,
        action_dim: int = 7,
        block: bool = True,
        timeout_s: float = 180.0,
        idle_timeout_s: float | None = None,
        poll_s: float = 0.05,
    ):
        self.root = os.path.abspath(root)
        self.action_dim = int(action_dim)
        self.block = bool(block)
        self.timeout_s = float(timeout_s)
        self.idle_timeout_s = (
            self.timeout_s if idle_timeout_s is None else float(idle_timeout_s)
        )
        self.poll_s = float(poll_s)
        os.makedirs(self.root, exist_ok=True)
        self.seq = 0
        self.last_cmd = HumanCommand.idle(self.action_dim)

    @classmethod
    def from_cfg(cls, cfg: Any | None) -> HumanMailbox | None:
        if cfg is None:
            return None
        enable = bool(cfg.get("enable", False)) if hasattr(cfg, "get") else False
        if not enable:
            return None
        root = str(cfg.get("root", "/tmp/rlinf_hitl"))
        idle = cfg.get("idle_timeout_s", None)
        return cls(
            root,
            action_dim=int(cfg.get("action_dim", 7)),
            block=bool(cfg.get("block", True)),
            timeout_s=float(cfg.get("timeout_s", 180.0)),
            idle_timeout_s=None if idle is None else float(idle),
            poll_s=float(cfg.get("poll_s", 0.05)),
        )

    @property
    def state_path(self) -> str:
        return os.path.join(self.root, _STATE_NAME)

    @property
    def cmd_path(self) -> str:
        return os.path.join(self.root, _CMD_NAME)

    def publish(
        self,
        main_image: np.ndarray | None,
        wrist_image: np.ndarray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> int:
        self.seq += 1
        payload = {"seq": self.seq, "action_dim": self.action_dim}
        if extra:
            payload.update(extra)
        if main_image is not None:
            jpeg = _encode_jpeg(main_image)
            numbered = _numbered_name("main", self.seq)
            _atomic_write_bytes(os.path.join(self.root, _MAIN_NAME), jpeg)
            _atomic_write_bytes(os.path.join(self.root, numbered), jpeg)
            payload["main"] = numbered
            payload["main_bytes"] = len(jpeg)
        if wrist_image is not None:
            jpeg = _encode_jpeg(wrist_image)
            numbered = _numbered_name("wrist", self.seq)
            _atomic_write_bytes(os.path.join(self.root, _WRIST_NAME), jpeg)
            _atomic_write_bytes(os.path.join(self.root, numbered), jpeg)
            payload["wrist"] = numbered
            payload["wrist_bytes"] = len(jpeg)
        _atomic_write_text(self.state_path, json.dumps(payload))
        _cleanup_numbered_frames(self.root, self.seq)
        return self.seq

    def read_command(self) -> HumanCommand | None:
        if not os.path.isfile(self.cmd_path):
            return None
        try:
            with open(self.cmd_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        action = np.asarray(data.get("action", []), dtype=np.float32).reshape(-1)
        if action.size < self.action_dim:
            padded = np.zeros((self.action_dim,), dtype=np.float32)
            padded[: action.size] = action
            action = padded
        else:
            action = action[: self.action_dim]
        # Missing key → inherit (safe default: UI gripper=-1 must not open).
        inherit = data.get("inherit_gripper", True)
        return HumanCommand(
            seq=int(data.get("seq", -1)),
            hold=bool(data.get("hold", False)),
            action=action,
            inherit_gripper=bool(inherit),
            skip=bool(data.get("skip", False)),
            skip_id=int(data.get("skip_id", 0) or 0),
            pause=bool(data.get("pause", False)),
            controller=normalize_hitl_controller(data.get("controller", "human")),
        )

    def _patch_state(self, **fields: Any) -> None:
        if not os.path.isfile(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        payload.update(fields)
        _atomic_write_text(self.state_path, json.dumps(payload))

    def wait_command(self, seq: int | None = None) -> HumanCommand:
        target = self.seq if seq is None else int(seq)
        published_at = time.time()
        idle_deadline = published_at + self.idle_timeout_s
        deadline = published_at + self.timeout_s
        paused = bool(self.last_cmd.pause)
        while True:
            now = time.time()
            cmd = self.read_command()
            if cmd is not None:
                try:
                    mtime = os.path.getmtime(self.cmd_path)
                except OSError:
                    mtime = 0.0
                # Jupyter often writes a stale seq. Accept a file rewritten
                # after this publish, or an explicit ack of ``target``.
                if cmd.seq >= target or mtime >= published_at - 0.25:
                    self.last_cmd = cmd
                    if _cmd_blocks_step(cmd):
                        if not paused:
                            self._patch_state(paused=True)
                            paused = True
                        time.sleep(self.poll_s)
                        continue
                    if paused or cmd.skip:
                        cmd.pause = False
                        self._patch_state(paused=False)
                    return cmd
            if _cmd_blocks_step(self.last_cmd):
                if not paused:
                    self._patch_state(paused=True)
                    paused = True
                time.sleep(self.poll_s)
                continue
            if now >= deadline:
                break
            # Online RL should keep stepping when the notebook is idle.
            # If the human is already holding, wait out timeout_s so a slow
            # Jupyter PUT does not drop the intervention.
            if now >= idle_deadline and not self.last_cmd.hold:
                self.last_cmd = HumanCommand.idle(self.action_dim, seq=target)
                return self.last_cmd
            time.sleep(self.poll_s)
        cmd = self.read_command()
        if cmd is not None and not _cmd_blocks_step(cmd):
            self.last_cmd = cmd
            return cmd
        self.last_cmd = HumanCommand.idle(self.action_dim, seq=target)
        return self.last_cmd

    def write_command(
        self,
        *,
        hold: bool,
        action: np.ndarray | list[float],
        seq: int,
        inherit_gripper: bool = True,
        skip: bool = False,
        skip_id: int = 0,
        pause: bool = False,
        controller: str = "human",
    ):
        action_list = np.asarray(action, dtype=np.float32).reshape(-1).tolist()
        _atomic_write_text(
            self.cmd_path,
            json.dumps(
                {
                    "seq": int(seq),
                    "hold": bool(hold),
                    "action": action_list,
                    "inherit_gripper": bool(inherit_gripper),
                    "skip": bool(skip),
                    "skip_id": int(skip_id),
                    "pause": bool(pause),
                    "controller": normalize_hitl_controller(controller),
                }
            ),
        )

    def append_episode(self, outcome: dict[str, Any]) -> None:
        """Append one finished-episode record for the Jupyter console / offline audit."""
        path = os.path.join(self.root, _EPISODES_NAME)
        line = json.dumps(outcome, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def first_env_obs(raw: Any) -> dict | None:
    """Return env0's raw dict from a list, object-ndarray, or dict obs."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        if len(raw) == 0:
            return None
        obs0 = raw[0]
    except (TypeError, IndexError, KeyError):
        return None
    return obs0 if isinstance(obs0, dict) else None


def _finite_round(value: float, ndigits: int = 3) -> float | None:
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, ndigits)


def env0_action_chunk(value: Any, chunk_len: int, action_dim: int) -> np.ndarray:
    """Take env0's first ``chunk_len × action_dim`` from a batched action/ref tensor."""
    chunk_len = int(chunk_len)
    action_dim = int(action_dim)
    out = np.full((chunk_len, action_dim), np.nan, dtype=np.float32)
    if value is None:
        return out
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return out
    if array.ndim >= 3:
        array = array[0]
    if array.ndim == 2 and array.shape[-1] == action_dim:
        rows = min(chunk_len, int(array.shape[0]))
        out[:rows] = array[:rows, :action_dim]
        return out
    if array.ndim == 2:
        array = array[0]
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    n = min(int(flat.size), chunk_len * action_dim)
    out.reshape(-1)[:n] = flat[:n]
    return out


def _as_chunk_rows(array: np.ndarray | None, chunk_len: int, action_dim: int) -> np.ndarray:
    if array is None:
        return np.full((chunk_len, action_dim), np.nan, dtype=np.float32)
    out = np.asarray(array, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    padded = np.full((chunk_len, action_dim), np.nan, dtype=np.float32)
    rows = min(chunk_len, out.shape[0])
    cols = min(action_dim, out.shape[1])
    padded[:rows, :cols] = out[:rows, :cols]
    return padded


def _mean_abs_l1(left: np.ndarray, right: np.ndarray) -> float | None:
    if not bool(np.isfinite(left).all() and np.isfinite(right).all()):
        return None
    return round(float(np.mean(np.abs(left - right))), 3)


def _pad_bool_flags(flags: np.ndarray | bool | None, chunk_len: int) -> np.ndarray:
    if flags is None:
        return np.zeros((chunk_len,), dtype=bool)
    if np.isscalar(flags) or isinstance(flags, (bool, np.bool_)):
        return np.full((chunk_len,), bool(flags), dtype=bool)
    out = np.asarray(flags, dtype=bool).reshape(-1)
    if out.size < chunk_len:
        padded = np.zeros((chunk_len,), dtype=bool)
        padded[: out.size] = out
        return padded
    return out[:chunk_len]


def format_hitl_chunk(
    actor: np.ndarray | None,
    executed: np.ndarray | None,
    flags: np.ndarray | None,
    *,
    chunk_id: int,
    step_i: int,
    cmd: np.ndarray | None = None,
    ref: np.ndarray | None = None,
    expert: np.ndarray | bool | None = None,
) -> dict | None:
    """JSON payload: π=VLA, a=actor, exec=env, cmd=keyboard, zap=expert."""
    if actor is None:
        return None
    actor_arr = np.asarray(actor, dtype=np.float32)
    if actor_arr.ndim == 1:
        actor_arr = actor_arr.reshape(1, -1)
    chunk_len, action_dim = actor_arr.shape
    ref_arr = _as_chunk_rows(ref, chunk_len, action_dim)
    exec_arr = _as_chunk_rows(executed, chunk_len, action_dim)
    cmd_arr = _as_chunk_rows(cmd, chunk_len, action_dim)
    human = _pad_bool_flags(flags, chunk_len)
    expert_arr = _pad_bool_flags(expert, chunk_len)
    step_i = int(max(0, min(int(step_i), chunk_len)))
    l1_pi_a: list[float | None] = []
    l1_a_exec: list[float | None] = []
    for idx in range(chunk_len):
        if idx >= step_i:
            l1_pi_a.append(None)
            l1_a_exec.append(None)
            continue
        l1_pi_a.append(_mean_abs_l1(ref_arr[idx], actor_arr[idx]))
        l1_a_exec.append(_mean_abs_l1(actor_arr[idx], exec_arr[idx]))
    return {
        "id": int(chunk_id),
        "len": int(chunk_len),
        "step": step_i,
        "human_steps": int(np.sum(human[:step_i])),
        "expert_steps": int(np.sum(expert_arr[:step_i])),
        "ref": [[_finite_round(x) for x in row] for row in ref_arr],
        "actor": [[_finite_round(x) for x in row] for row in actor_arr],
        "policy": [[_finite_round(x) for x in row] for row in actor_arr],
        "cmd": [[_finite_round(x) for x in row] for row in cmd_arr],
        "executed": [[_finite_round(x) for x in row] for row in exec_arr],
        "human": [bool(flag) for flag in human.tolist()],
        "expert": [bool(flag) for flag in expert_arr.tolist()],
        "l1": l1_pi_a,
        "l1_pi_a": l1_pi_a,
        "l1_a_exec": l1_a_exec,
    }


def _cmd_row_has_values(row: list) -> bool:
    return any(value is not None for value in row)


def format_hitl_chunk_text(chunk: dict | None) -> str:
    """π=VLA ref_chunk, a=actor residual, exec=env. H=keyboard, E=ZAP."""
    if not chunk:
        return "chunk: (waiting)"
    expert_steps = int(chunk.get("expert_steps") or 0)
    lines = [
        (
            f"chunk #{chunk['id']}  {chunk['step']}/{chunk['len']}  "
            f"human_steps={chunk['human_steps']}  expert_steps={expert_steps}"
        ),
        "π=VLA ref_chunk  a=actor  exec=env  L1(π,a)=residual vs VLA",
        "H=keyboard cmd  E=ZAP (next chunk if zap line has no numbers yet)",
        "         dx     dy     dz  droll dpitch   dyaw   grip",
    ]

    def _row(vals: list) -> str:
        parts = []
        for value in vals:
            if value is None:
                parts.append("   n/a")
            else:
                parts.append(f"{float(value):6.2f}")
        return " ".join(parts)

    ref_rows = chunk.get("ref") or []
    actor_rows = chunk.get("actor") or chunk.get("policy") or []
    cmd_rows = chunk.get("cmd") or []
    expert_flags = chunk.get("expert") or [False] * int(chunk["len"])
    l1_pi_a = chunk.get("l1_pi_a") or chunk.get("l1") or []
    l1_a_exec = chunk.get("l1_a_exec") or []
    for idx in range(int(chunk["len"])):
        is_expert = bool(expert_flags[idx]) if idx < len(expert_flags) else False
        is_human = bool(chunk["human"][idx]) and not is_expert
        if is_expert:
            mark = "E"
        elif is_human:
            mark = "H"
        elif idx < int(chunk["step"]):
            mark = "P"
        else:
            mark = "."
        cursor = ">" if idx == int(chunk["step"]) - 1 else " "
        extra = ""
        if idx < len(l1_pi_a) and l1_pi_a[idx] is not None:
            extra += f"  L1(π,a)={float(l1_pi_a[idx]):.3f}"
        if idx < len(l1_a_exec) and l1_a_exec[idx] is not None:
            extra += f"  L1(a,exec)={float(l1_a_exec[idx]):.3f}"
            if is_human and float(l1_a_exec[idx]) < 0.02:
                extra += "  << no override"
        pi_row = ref_rows[idx] if idx < len(ref_rows) else [None] * 7
        a_row = actor_rows[idx] if idx < len(actor_rows) else [None] * 7
        lines.append(f"{cursor}{mark} π    {_row(pi_row)}")
        lines.append(f"    a    {_row(a_row)}{extra}")
        if is_expert:
            zap_row = cmd_rows[idx] if idx < len(cmd_rows) else []
            if _cmd_row_has_values(zap_row):
                lines.append(f"    zap  {_row(zap_row)}")
            else:
                lines.append("    zap  (next chunk)")
        elif is_human and idx < len(cmd_rows):
            lines.append(f"    cmd  {_row(cmd_rows[idx])}")
        lines.append(f"    exec {_row(chunk['executed'][idx])}")
    return "\n".join(lines)


def is_chunk_wide_intervene_action(
    action: Any,
    num_envs: int,
    chunk_len: int,
    action_dim: int,
) -> bool:
    """True when ``action`` is already ``[B, T * A]`` for ``update_last_actions``."""
    if action is None:
        return False
    try:
        tensor = torch.as_tensor(action)
    except (TypeError, ValueError):
        return False
    if tensor.numel() == 0:
        return False
    return int(tensor.reshape(int(num_envs), -1).shape[-1]) == int(chunk_len) * int(
        action_dim
    )


def stack_chunk_intervene(infos_list: list[dict], num_envs: int) -> dict | None:
    """Stack per-step human labels to the chunk layout ``update_last_actions`` expects.

    Each step writes ``intervene_action`` as ``[B, A]`` and ``intervene_flag`` as
    ``[B, 1]``. The trajectory builder needs ``[B, T * A]`` and ``[B, T]``.
    Steps that only have a flag (ZAP request) keep executed actions: zeros +
    False, so a mixed human/expert chunk still stacks to ``T * A``.
    """
    if not infos_list:
        return None
    action_rows: list[torch.Tensor | None] = []
    flag_rows: list[torch.Tensor] = []
    template: torch.Tensor | None = None
    for infos in infos_list:
        action = infos.get("intervene_action")
        flag = infos.get("intervene_flag")
        if action is not None and hasattr(action, "reshape"):
            row = torch.as_tensor(action).reshape(num_envs, -1)
            template = row
            action_rows.append(row)
        else:
            action_rows.append(None)
        if flag is None:
            flag_rows.append(torch.zeros((num_envs, 1), dtype=torch.bool))
        else:
            flag_rows.append(
                torch.as_tensor(flag).reshape(num_envs, -1)[:, -1:].to(torch.bool)
            )
    if template is None:
        return None
    filled_actions = []
    filled_flags = []
    for row, flag in zip(action_rows, flag_rows, strict=True):
        if row is None:
            filled_actions.append(torch.zeros_like(template))
            filled_flags.append(torch.zeros((num_envs, 1), dtype=torch.bool))
        else:
            filled_actions.append(row)
            filled_flags.append(flag)
    stacked_actions = torch.stack(filled_actions, dim=1)
    stacked_flags = torch.stack(filled_flags, dim=1).reshape(num_envs, -1)
    return {
        "intervene_action": stacked_actions.reshape(num_envs, -1),
        "intervene_flag": stacked_flags,
    }
