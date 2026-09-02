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

"""Temporal history encoder for Stage-2 long-horizon RLT.

The encoder consumes a window of chunk summaries produced by
:mod:`rlinf.algorithms.rlt_lh.history`. It is intentionally small and is
trained only in Stage 2 so it can adapt to the online policy distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RLTHistoryEncoder"]


class RLTHistoryEncoder(nn.Module):
    """Encode a sliding window of chunk summaries into one history vector.

    Input shape: ``[batch, window, summary_dim]``.
    Output shape: ``[batch, hidden_dim]``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim!r}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}.")

        self.input_proj = (
            nn.Linear(self.input_dim, self.hidden_dim)
            if self.input_dim != self.hidden_dim
            else nn.Identity()
        )
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return the last GRU hidden state for each batch element."""
        if history.dim() != 3:
            raise ValueError(
                "RLTHistoryEncoder expects history with shape "
                f"[batch, window, summary_dim], got {tuple(history.shape)}."
            )
        if history.shape[-1] != self.input_dim:
            raise ValueError(
                "History summary dimension mismatch: expected "
                f"{self.input_dim}, got {history.shape[-1]}."
            )
        projected = self.input_proj(history)
        _, hidden = self.gru(projected)
        return hidden[-1]
