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

"""Mailbox smoke test: no Ray, no VLA, no LIBERO.

Run the Jupyter console first, then this process. You should see a moving
color bar in the notebook. Toggle Take over and move sliders; this script
prints the command it received.

    python toolkits/hitl/smoke_mailbox.py --seconds 90
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlinf.envs.libero.human_mailbox import HumanMailbox


def _dummy_frames(step: int) -> tuple[np.ndarray, np.ndarray]:
    main = np.zeros((240, 320, 3), dtype=np.uint8)
    wrist = np.zeros((240, 320, 3), dtype=np.uint8)
    col = (step * 6) % 256
    main[:, :, 0] = 40
    main[:, col % 320, :] = 255
    main[20:60, 20:80, 1] = 200
    wrist[:, :, 2] = 40
    wrist[col % 240, :, 1] = 220
    return main, wrist


_DEFAULT_ROOT = str(
    Path(__file__).resolve().parent / "live"
)


def run_smoke(
    root: str = _DEFAULT_ROOT,
    *,
    seconds: float = 90.0,
    hz: float = 5.0,
) -> None:
    mailbox = HumanMailbox(root, action_dim=7, block=True, timeout_s=2.0)
    print(f"[hitl-smoke] mailbox={mailbox.root}")
    print("[hitl-smoke] Open the Jupyter console first, then watch the frames.")
    deadline = time.time() + float(seconds)
    step = 0
    last_print = 0.0
    while time.time() < deadline:
        main, wrist = _dummy_frames(step)
        seq = mailbox.publish(
            main,
            wrist,
            extra={"elapsed": step, "success": False, "mode": "smoke"},
        )
        cmd = mailbox.wait_command(seq)
        now = time.time()
        if now - last_print > 0.8:
            print(
                f"[hitl-smoke] seq={seq} hold={cmd.hold} "
                f"action={np.round(cmd.action, 3).tolist()}"
            )
            last_print = now
        step += 1
        time.sleep(max(0.0, 1.0 / hz - 0.01))
    print("[hitl-smoke] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="HITL mailbox smoke test")
    parser.add_argument("--root", default=_DEFAULT_ROOT)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--hz", type=float, default=5.0)
    args = parser.parse_args()
    run_smoke(args.root, seconds=args.seconds, hz=args.hz)


if __name__ == "__main__":
    main()
