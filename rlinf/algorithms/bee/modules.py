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

"""Correction Model and state-dependent Lagrange multiplier for Bee."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden_dim: int, num_hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(int(num_hidden_layers)):
        layers.append(nn.Linear(dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        dim = int(hidden_dim)
    return nn.Sequential(*layers)


class BeeCorrectionModel(nn.Module):
    """Proposal-conditioned diagonal Gaussian over human correction residuals.

    Predicts ``μ_φ(s, ã)`` and per-dimension ``σ_φ`` for
    ``Δ^H = a^H - ã`` (Bee Eq. 5). Input is the flat concatenation of the
    RL state ``s = [z ‖ p]`` and the VLA proposal chunk ``ã``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 512,
        num_hidden_layers: int = 3,
        variance_floor: float = 0.02,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.variance_floor = float(variance_floor)
        self.trunk = _mlp(
            int(state_dim) + int(action_dim), int(hidden_dim), int(num_hidden_layers)
        )
        self.mean_head = nn.Linear(int(hidden_dim), int(action_dim))
        self.log_std_head = nn.Linear(int(hidden_dim), int(action_dim))
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, 0.0)

    def forward(
        self, state: torch.Tensor, proposal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, sigma)`` with ``sigma >= variance_floor``."""
        feat = self.trunk(torch.cat([state, proposal], dim=-1))
        mu = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        # Softplus keeps σ positive; floor matches Table III.
        sigma = F.softplus(log_std) + self.variance_floor
        return mu, sigma


class BeeMultiplier(nn.Module):
    """State-dependent Lagrange multiplier ``λ_ω(s) >= 0`` (Bee Eq. 7–9)."""

    def __init__(
        self,
        state_dim: int,
        *,
        hidden_dim: int = 512,
        num_hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        self.trunk = _mlp(int(state_dim), int(hidden_dim), int(num_hidden_layers))
        self.head = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return softplus-activated multiplier ``[B, 1]``."""
        feat = self.trunk(state)
        return F.softplus(self.head(feat))
