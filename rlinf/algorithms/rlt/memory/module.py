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

"""Trainable short-term memory module for the RLT Stage-2 policy."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.algorithms.rlt.memory.buffer import memory_entry_dim
from rlinf.algorithms.rlt.memory.encoder import GRUMemoryEncoder
from rlinf.algorithms.rlt.memory.fusion import ResidualMemoryFusion


class RLTSTMModule(torch.nn.Module):
    """Encode recent history and fuse it with the current RLT token."""

    def __init__(self, *, z_dim: int, entry_dim: int, hidden_dim: int):
        super().__init__()
        self.z_dim = int(z_dim)
        self.entry_dim = int(entry_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = GRUMemoryEncoder(
            entry_dim=self.entry_dim,
            hidden_dim=self.hidden_dim,
        )
        self.fusion = ResidualMemoryFusion(
            z_dim=self.z_dim,
            memory_dim=self.hidden_dim,
        )

    def forward(
        self,
        z_t: torch.Tensor,
        history: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if history is None or mask is None:
            return z_t
        memory = self.encoder(history, mask)
        return self.fusion(z_t, memory)


def build_memory_module(
    memory_cfg: Any,
    *,
    z_dim: int,
    action_dim: int,
) -> RLTSTMModule | None:
    """Build the Step 2A STM module, or ``None`` when memory is disabled."""
    if memory_cfg is None or not bool(memory_cfg.get("enabled", False)):
        return None
    encoder_cfg = memory_cfg.get("encoder", {}) or {}
    hidden_dim = int(encoder_cfg.get("hidden_dim", 256))
    fusion_cfg = memory_cfg.get("fusion", {}) or {}
    fusion_memory_dim = fusion_cfg.get("memory_proj_dim", None)
    if fusion_memory_dim is not None and int(fusion_memory_dim) != hidden_dim:
        raise ValueError(
            "memory.fusion.memory_proj_dim must match memory.encoder.hidden_dim, "
            f"got {int(fusion_memory_dim)} != {hidden_dim}."
        )
    entry_dim = memory_entry_dim(
        memory_cfg,
        z_dim=z_dim,
        action_dim=action_dim,
    )
    return RLTSTMModule(
        z_dim=z_dim,
        entry_dim=entry_dim,
        hidden_dim=hidden_dim,
    )
