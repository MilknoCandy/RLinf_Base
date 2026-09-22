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

"""SFT readout heads: predict distance and success from the current z."""

from __future__ import annotations

import torch
import torch.nn as nn


class B2ReadoutHeads(nn.Module):
    """Map encoder output ``z`` to a distance scalar and a success logit.

    These heads exist only to teach the encoder how to compress history during
    SFT. They are not used as the Stage 2 policy.
    """

    def __init__(self, z_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else min(512, int(z_dim))
        self.dist_head = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.success_head = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(distance, success_logit)`` with the leading dims of ``z``."""
        if z.shape[-1] == 0:
            raise ValueError("z must have a non-empty feature dim.")
        flat = z.reshape(-1, z.shape[-1])
        distance = self.dist_head(flat).squeeze(-1).reshape(z.shape[:-1])
        success_logit = self.success_head(flat).squeeze(-1).reshape(z.shape[:-1])
        return distance, success_logit
