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

"""Privileged insertion-progress memory for the Stage 2 critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rlinf.algorithms.rlt.b2_feedback import (
    distance_from_env_infos,
    success_from_env_infos,
)

PROGRESS_DIM = 4
PROGRESS_OBS_KEY = "rlt_progress"
DEFAULT_DISTANCE_SCALE = 0.05


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
class ProgressMemoryState:
    """Per-env progress features: distance, delta, success, has_history.

    These are POMDP coordinates for the critic. They are not fed to the frozen
    VLM. Episode/scene ``done`` zeros the delta for that step, then the reset
    observation becomes the next step's previous distance.
    """

    prev_distance: torch.Tensor | None = None
    has_history: torch.Tensor | None = None

    def reset(self) -> None:
        self.prev_distance = None
        self.has_history = None

    def update(
        self,
        *,
        batch_size: int,
        dones: torch.Tensor | None,
        env_infos: dict[str, Any] | None,
        device: torch.device,
        distance_scale: float = DEFAULT_DISTANCE_SCALE,
    ) -> torch.Tensor:
        """Return ``[B, 4]`` float32 features for the current observation."""
        zeros = torch.zeros(batch_size, dtype=torch.float32, device=device)
        done = _done_mask(dones, batch_size, device)
        distance = distance_from_env_infos(env_infos, batch_size, device)
        success = success_from_env_infos(env_infos, batch_size, device)
        if distance is None:
            return torch.stack((zeros, zeros, zeros, zeros), dim=-1)

        if (
            self.prev_distance is None
            or self.has_history is None
            or self.prev_distance.shape[0] != batch_size
            or self.has_history.shape[0] != batch_size
        ):
            has = torch.zeros(batch_size, dtype=torch.bool, device=device)
            delta = zeros
        else:
            has = self.has_history.to(device=device) & (~done)
            delta = distance - self.prev_distance.to(
                device=device, dtype=distance.dtype
            )
            delta = torch.where(has, delta, zeros)

        self.prev_distance = distance.detach()
        self.has_history = torch.ones(
            batch_size, dtype=torch.bool, device=device
        )

        scale = max(float(distance_scale), 1.0e-6)
        features = torch.stack(
            (
                torch.tanh(distance.float() / scale),
                torch.tanh(delta.float() / scale),
                success.to(dtype=torch.float32),
                has.to(dtype=torch.float32),
            ),
            dim=-1,
        )
        return features
