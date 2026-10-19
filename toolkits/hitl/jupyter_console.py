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

"""Widget-free HITL viewer for Jupyter.

ipywidgets hangs on this JupyterLab when the kernel env does not match the
server. This module only uses ``IPython.display`` and a JSON panel file.

    from toolkits.hitl.jupyter_console import watch, set_panel
    set_panel(hold=True, dx=0.2)
    watch("/tmp/rlinf_hitl", seconds=90)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from IPython.display import Image, clear_output, display

_PANEL_DEFAULT = Path(__file__).resolve().parent / "panel.json"
_KEYS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _read_panel(panel_path: str) -> dict:
    if not os.path.isfile(panel_path):
        if _PANEL_DEFAULT.is_file():
            panel_path = str(_PANEL_DEFAULT)
        else:
            return {"hold": False, **{key: 0.0 for key in _KEYS[:-1]}, "gripper": -1.0}
    with open(panel_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    hold = bool(data.get("hold", False))
    action = [float(data.get(key, -1.0 if key == "gripper" else 0.0)) for key in _KEYS]
    return {"hold": hold, "action": action, "raw": data}


def set_panel(
    panel_path: str | None = None,
    *,
    hold: bool | None = None,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
    droll: float | None = None,
    dpitch: float | None = None,
    dyaw: float | None = None,
    gripper: float | None = None,
) -> dict:
    """Update ``toolkits/hitl/panel.json``. ``watch()`` reads it every tick."""
    path = panel_path or str(_PANEL_DEFAULT)
    data = _read_panel(path)["raw"] if os.path.isfile(path) else {
        "hold": False,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
        "droll": 0.0,
        "dpitch": 0.0,
        "dyaw": 0.0,
        "gripper": -1.0,
    }
    if hold is not None:
        data["hold"] = bool(hold)
    updates = {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "droll": droll,
        "dpitch": dpitch,
        "dyaw": dyaw,
        "gripper": gripper,
    }
    for key, value in updates.items():
        if value is not None:
            data[key] = float(value)
    _atomic_write_text(path, json.dumps(data, indent=2))
    return data


def _read_seq(root: str) -> int:
    path = os.path.join(root, "state.json")
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(json.load(handle).get("seq", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def _write_cmd(
    root: str,
    *,
    hold: bool,
    action: list[float],
    seq: int,
    controller: str = "human",
) -> None:
    name = str(controller or "human").strip().lower()
    if name not in {"expert", "zap", "teacher"}:
        name = "human"
    else:
        name = "expert"
    _atomic_write_text(
        os.path.join(root, "cmd.json"),
        json.dumps(
            {
                "seq": int(seq),
                "hold": bool(hold),
                "action": action,
                "controller": name,
            }
        ),
    )


def watch(
    root: str = "/tmp/rlinf_hitl",
    *,
    seconds: float = 90.0,
    hz: float = 4.0,
    panel_path: str | None = None,
) -> None:
    """Blocking live view. Interrupt the cell to stop.

    Edit ``toolkits/hitl/panel.json`` (or call ``set_panel``) to take over.
    """
    os.makedirs(root, exist_ok=True)
    panel_path = panel_path or str(_PANEL_DEFAULT)
    deadline = time.time() + float(seconds)
    print(f"mailbox={root}")
    print(f"panel={panel_path}  (set hold=true to take over)")
    while time.time() < deadline:
        panel = _read_panel(panel_path)
        seq = _read_seq(root)
        raw = panel.get("raw") if isinstance(panel.get("raw"), dict) else {}
        _write_cmd(
            root,
            hold=panel["hold"],
            action=panel["action"],
            seq=seq,
            controller=raw.get("controller", "human"),
        )
        clear_output(wait=True)
        print(
            f"seq={seq} hold={panel['hold']} action={[round(x, 2) for x in panel['action']]}"
        )
        print("Interrupt this cell to stop. Use set_panel(...) in another notebook.")
        shown = False
        for name, title in (
            ("frame_main.jpg", "agentview"),
            ("frame_wrist.jpg", "wrist"),
        ):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                print(title)
                display(Image(filename=path))
                shown = True
        if not shown:
            print("No frames yet. Start smoke_mailbox.py or teleop_libero_bowl.py.")
        time.sleep(1.0 / max(float(hz), 1.0))
    print("watch() finished")


def run_console(root: str = "/tmp/rlinf_hitl", **kwargs):
    """Back-compat: same as ``watch()``. No ipywidgets."""
    seconds = float(kwargs.get("seconds", 90.0))
    hz = float(kwargs.get("refresh_hz", kwargs.get("hz", 4.0)))
    watch(root, seconds=seconds, hz=hz)
