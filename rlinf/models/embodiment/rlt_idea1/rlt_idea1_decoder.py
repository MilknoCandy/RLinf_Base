# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decoder-only module for RLT Idea1.

Idea1 has no post-hoc encoder. A learnable token is injected into the VLM
prefix, and the VLM's attention layers produce the compressed feature at the
token's output position. This module reuses the standard ``RLTTokenDecoder``
to reconstruct the prefix from that feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenDecoder,
    sinusoidal_pe_init,
)


class RltIdea1Decoder(nn.Module):
    """Learner-side RLT Idea1 module.

    Attributes:
        rl_token_embed: learnable token injected into the VLM prefix, shape
            ``[1, input_dim]`` (input_dim equals the VLM prefix hidden width).
        z_proj: optional projection from the VLM token output width to the
            decoder token width (``embed_dim``).
        decoder: the same ``RLTTokenDecoder`` used by the standard RLT.
    """

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        embed_dim: int = 2048,
        prefix_seq_len: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.prefix_seq_len = int(prefix_seq_len)

        # Injected token must match the VLM prefix hidden width.
        self.rl_token_embed = nn.Parameter(
            sinusoidal_pe_init(1, self.input_dim)
        )

        # VLM token output width -> decoder token width.
        self.z_proj = (
            nn.Linear(self.input_dim, self.embed_dim)
            if self.input_dim != self.embed_dim
            else nn.Identity()
        )

        self.decoder = RLTTokenDecoder(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            prefix_seq_len=self.prefix_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
        )

    @property
    def z_dim(self) -> int:
        return self.embed_dim

    def get_rl_token(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return the learnable token ``[B, 1, input_dim]`` for VLM injection."""
        return (
            self.rl_token_embed.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

    def encode_z(self, z_rl_raw: torch.Tensor) -> torch.Tensor:
        """Project the VLM token output to the decoder token width.

        Args:
            z_rl_raw: ``[B, input_dim]`` hidden state at the learnable token
                position in the VLM output.

        Returns:
            ``[B, embed_dim]`` feature consumed by the decoder and Stage2.
        """
        return self.z_proj(z_rl_raw)

    def loss(
        self,
        z_rl: torch.Tensor,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Reconstruction MSE between the decoded prefix and the target prefix.

        Args:
            z_rl: ``[B, input_dim]`` VLM hidden state at the learnable token
                position (the compressed feature).
            prefix_embs: ``[B, S, input_dim]`` target prefix embeddings.
            mask: optional ``[B, S]`` valid-token mask.

        Returns:
            ``(mse, {"mse": mse, "z_rl": projected_z_rl})``.
        """
        projected = self.encode_z(z_rl)  # [B, embed_dim]
        rl_tokens = projected.unsqueeze(1)  # [B, 1, embed_dim]
        reconstructed = self.decoder(rl_tokens, prefix_embs, mask)

        target = prefix_embs.detach().to(dtype=torch.float32)
        reconstructed = reconstructed.to(dtype=torch.float32)
        sq_error = torch.square(reconstructed - target)

        if mask is not None:
            mask_expanded = mask.to(device=sq_error.device, dtype=sq_error.dtype)[
                ..., None
            ]
            sq_error = sq_error * mask_expanded
            denom = torch.clamp(
                mask_expanded.sum() * prefix_embs.shape[-1], min=1.0
            )
            mse = sq_error.sum() / denom
        else:
            mse = sq_error.mean()

        return mse, {"mse": mse, "z_rl": projected}

    def forward(
        self,
        z_rl: torch.Tensor,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss(z_rl, prefix_embs, mask)