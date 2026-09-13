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

"""Residual fusion between the current RLT token and memory embedding."""

from __future__ import annotations

import torch


class ResidualMemoryFusion(torch.nn.Module):
    """Implement ``z'_t = z_t + W_m * m_t``.

    The projection is zero-initialized so the model exactly degenerates to the
    RLT baseline when memory is disabled or its contribution is not useful.
    """

    def __init__(self, *, z_dim: int, memory_dim: int):
        super().__init__()
        self.z_dim = int(z_dim)
        self.memory_dim = int(memory_dim)
        self.memory_proj = torch.nn.Linear(self.memory_dim, self.z_dim, bias=False)
        torch.nn.init.zeros_(self.memory_proj.weight)

    def forward(self, z_t: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return z_t + self.memory_proj(memory)
