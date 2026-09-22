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

"""Write sequential B2 encoder inputs ``I_t`` plus privileged labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from rlinf.utils.logging import get_logger

logger = get_logger()


def _done_mask(dones: torch.Tensor | None, batch_size: int) -> torch.Tensor:
    if dones is None:
        return torch.zeros(batch_size, dtype=torch.bool)
    mask = dones.detach().cpu()
    if mask.dim() > 1:
        mask = mask.reshape(batch_size, -1)[:, -1]
    return mask.reshape(batch_size).to(dtype=torch.bool)


class B2DumpWriter:
    """Accumulate per-env ``(I, mask, d, s)`` and write one ``.pt`` per episode.

    When ``auto_reset`` is True, ``done`` means the current observation already
    belongs to the next episode, so the previous buffer is flushed first.
    When False, the current step is the last frame of the finished episode.
    """

    def __init__(
        self,
        dump_dir: str | Path,
        *,
        rank: int = 0,
        auto_reset: bool = True,
        min_steps: int = 2,
    ):
        self.dump_dir = Path(dump_dir).expanduser()
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.auto_reset = bool(auto_reset)
        self.min_steps = int(min_steps)
        self._buffers: list[list[dict[str, Any]]] | None = None
        self._episode_id = 0
        self.num_written = 0

    def append(
        self,
        *,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor | None,
        distance: torch.Tensor | None,
        success: torch.Tensor,
        dones: torch.Tensor | None,
    ) -> None:
        batch_size = int(image_tokens.shape[0])
        if self._buffers is None:
            self._buffers = [[] for _ in range(batch_size)]
        elif len(self._buffers) != batch_size:
            self.close()
            self._buffers = [[] for _ in range(batch_size)]

        done = _done_mask(dones, batch_size)
        tokens_cpu = image_tokens.detach().to(device="cpu", dtype=torch.float16)
        if image_mask is None:
            mask_cpu = torch.ones(tokens_cpu.shape[:2], dtype=torch.bool)
        else:
            mask_cpu = image_mask.detach().to(device="cpu", dtype=torch.bool)
        if distance is None:
            dist_cpu = torch.zeros(batch_size, dtype=torch.float32)
        else:
            dist_cpu = distance.detach().to(device="cpu", dtype=torch.float32).reshape(
                batch_size
            )
        success_cpu = success.detach().to(device="cpu").reshape(batch_size).to(
            dtype=torch.bool
        )

        for index in range(batch_size):
            if bool(done[index]) and self.auto_reset and self._buffers[index]:
                self._flush(index)
            self._buffers[index].append(
                {
                    "image_tokens": tokens_cpu[index].contiguous(),
                    "image_mask": mask_cpu[index].contiguous(),
                    "distance": float(dist_cpu[index]),
                    "success": bool(success_cpu[index]),
                }
            )
            if bool(done[index]) and not self.auto_reset:
                self._flush(index)

    def close(self) -> None:
        if self._buffers is None:
            return
        for index in range(len(self._buffers)):
            if self._buffers[index]:
                self._flush(index)

    def _flush(self, index: int) -> None:
        steps = self._buffers[index] if self._buffers is not None else []
        if self._buffers is not None:
            self._buffers[index] = []
        if len(steps) < self.min_steps:
            return
        episode = {
            "image_tokens": torch.stack(
                [step["image_tokens"] for step in steps], dim=0
            ),
            "image_mask": torch.stack(
                [step["image_mask"] for step in steps], dim=0
            ),
            "distance": torch.tensor(
                [step["distance"] for step in steps], dtype=torch.float32
            ),
            "success": torch.tensor(
                [step["success"] for step in steps], dtype=torch.bool
            ),
        }
        path = self.dump_dir / f"rank{self.rank:03d}_ep{self._episode_id:06d}.pt"
        torch.save(episode, path)
        self._episode_id += 1
        self.num_written += 1
        if self.num_written == 1 or self.num_written % 20 == 0:
            logger.info(
                "B2 dump: wrote %s (%d episodes, T=%d)",
                path,
                self.num_written,
                int(episode["distance"].numel()),
            )
