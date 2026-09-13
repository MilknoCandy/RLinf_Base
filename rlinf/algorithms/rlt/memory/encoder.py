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

"""GRU encoder for short-term memory histories."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pack_padded_sequence


class GRUMemoryEncoder(torch.nn.Module):
    """Encode a padded recent-history sequence into one memory embedding."""

    def __init__(self, *, entry_dim: int, hidden_dim: int):
        super().__init__()
        self.entry_dim = int(entry_dim)
        self.hidden_dim = int(hidden_dim)
        self.gru = torch.nn.GRU(
            input_size=self.entry_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

    def forward(
        self,
        history: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if history.shape[-1] != self.entry_dim:
            raise ValueError(
                f"STM history entry_dim mismatch: expected {self.entry_dim}, "
                f"got {history.shape[-1]}."
            )
        lengths = mask.to(torch.long).sum(dim=-1).clamp(min=1)
        packed = pack_padded_sequence(
            history,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        memory = h_n[-1]
        valid = (mask.sum(dim=-1) > 0).to(
            dtype=memory.dtype, device=memory.device
        )
        return memory * valid.unsqueeze(-1)
