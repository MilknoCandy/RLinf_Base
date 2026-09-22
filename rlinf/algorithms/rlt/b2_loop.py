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

"""Per-env B2 loop state: previous z as the next RL token, plus feedback text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rlinf.algorithms.rlt.b2_feedback import (
    distance_from_env_infos,
    format_peg_insertion_feedback,
    success_from_env_infos,
)

_OFF_AXIS_THRESHOLD = 0.01


def _done_mask(
    dones: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if dones is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    mask = dones.to(device=device)
    if mask.dim() > 1:
        mask = mask.reshape(batch_size, -1)[:, -1]
    return mask.reshape(batch_size).to(dtype=torch.bool)


@dataclass
class B2LoopState:
    """Rollout-side state for the B2 encoder loop.

    ``z`` is not a memory bank. It is the previous encoder output that becomes
    the next RL token. Episode/scene reset (``done``) restores the learned
    initial token; a same-scene task switch does not.
    """

    z: torch.Tensor | None = None
    prev_distance: torch.Tensor | None = None
    has_history: torch.Tensor | None = None
    last_feedback: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.z = None
        self.prev_distance = None
        self.has_history = None
        self.last_feedback = []

    def build_rl_token(
        self,
        *,
        init_token: torch.Tensor,
        batch_size: int,
        dones: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[B, 1, D]`` RL tokens, using ``z_init`` on reset/first step."""
        init = init_token.to(device=device, dtype=dtype)
        if init.dim() == 1:
            init = init.unsqueeze(0)
        if init.dim() == 2:
            init = init.unsqueeze(1)
        init = init.expand(batch_size, 1, -1).clone()
        if self.z is None or self.z.shape[0] != batch_size:
            return init
        previous = self.z.to(device=device, dtype=dtype).reshape(batch_size, 1, -1)
        done = _done_mask(dones, batch_size, device).view(batch_size, 1, 1)
        return torch.where(done, init, previous)

    def feedback_sentences(
        self,
        *,
        batch_size: int,
        dones: torch.Tensor | None,
        env_infos: dict[str, Any] | None,
        device: torch.device,
        off_axis_threshold: float = _OFF_AXIS_THRESHOLD,
    ) -> list[str]:
        """Build per-env feedback for the current VLM prefix.

        Feedback is empty on the first step of an episode. After that it is
        written from the env infos bundled with the current observation.
        """
        empty = [""] * batch_size
        done = _done_mask(dones, batch_size, device)
        if env_infos is None:
            self.last_feedback = empty
            return empty

        distance = distance_from_env_infos(env_infos, batch_size, device)
        success = success_from_env_infos(env_infos, batch_size, device)
        if distance is None:
            self.last_feedback = empty
            return empty

        history = self.has_history
        if (
            self.prev_distance is None
            or history is None
            or self.prev_distance.shape[0] != batch_size
            or history.shape[0] != batch_size
        ):
            self.prev_distance = distance.detach()
            self.last_feedback = empty
            return empty

        delta = distance - self.prev_distance.to(device=device, dtype=distance.dtype)
        off_axis = None
        abs_y = env_infos.get("peg_head_hole_abs_y")
        abs_z = env_infos.get("peg_head_hole_abs_z")
        if abs_y is not None and abs_z is not None:
            y_tensor = (
                abs_y if torch.is_tensor(abs_y) else torch.as_tensor(abs_y, device=device)
            )
            z_tensor = (
                abs_z if torch.is_tensor(abs_z) else torch.as_tensor(abs_z, device=device)
            )
            off_axis = (
                torch.sqrt(
                    y_tensor.float().reshape(-1) ** 2 + z_tensor.float().reshape(-1) ** 2
                )
                > off_axis_threshold
            )

        sentences = format_peg_insertion_feedback(
            success=success,
            delta_distance=delta,
            off_axis=off_axis,
        )
        use_feedback = history.to(device=device) & (~done)
        self.prev_distance = distance.detach()
        self.last_feedback = [
            sentence if bool(use_feedback[index]) else ""
            for index, sentence in enumerate(sentences)
        ]
        return self.last_feedback

    def commit_z(self, z: torch.Tensor) -> None:
        """Store the current encoder output as the next RL token."""
        stored = z.detach()
        if stored.dim() == 3:
            stored = stored.reshape(stored.shape[0], -1)
        self.z = stored
        self.has_history = torch.ones(
            stored.shape[0], dtype=torch.bool, device=stored.device
        )
