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

"""Bottleneck-free decoder for the official-OpenPI RLT Idea1 variant.

Idea1 injects a single learnable token into the official OpenPI VLM prefix and
reads the hidden state at that token position as the RL feature ``z_rl``.
Unlike :class:`rlinf.models.embodiment.rlt_idea1.rlt_idea1_decoder.RltIdea1Decoder`,
this module does **not** compress ``z_rl`` through a low-dimensional bottleneck
(e.g. 2048 -> 512 -> 2048). ``z_rl`` keeps the decoder token width and is
reconstructed directly by the shared ``RLTTokenDecoder``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenDecoder,
    sinusoidal_pe_init,
)


class RltIdea1OpenPiDecoder(nn.Module):
    """Learner-side RLT Idea1 module without a bottleneck MLP compression.

    Attributes:
        rl_token_embed: learnable token injected into the VLM prefix, shape
            ``[1, input_dim]`` (``input_dim`` equals the VLM prefix hidden
            width).
        z_proj: optional projection from the VLM token output width to the
            decoder token width (``embed_dim``).
        z_out_norm: optional per-sample LayerNorm over the full ``embed_dim``
            feature. It does not change dimensionality; it only standardizes
            the feature handed to Stage2 and the decoder.
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
        z_norm: bool = False,
        z_l2_weight: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.prefix_seq_len = int(prefix_seq_len)
        self.z_l2_weight = float(z_l2_weight)

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

        # Full-width standardization of z_rl. This is not a bottleneck: the
        # feature stays ``embed_dim``-dimensional.
        self.z_out_norm = (
            nn.LayerNorm(self.embed_dim, elementwise_affine=False)
            if z_norm
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
        """Project the VLM token output to the full-width RL feature.

        Args:
            z_rl_raw: ``[B, input_dim]`` hidden state at the learnable token
                position in the VLM output.

        Returns:
            ``[B, embed_dim]`` feature consumed by Stage2 and the decoder.
        """
        projected = self.z_proj(z_rl_raw)
        return self.z_out_norm(projected)

    def loss(
        self,
        z_rl: torch.Tensor,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Reconstruction MSE between the decoded prefix and the target prefix.

        Args:
            z_rl: ``[B, input_dim]`` VLM hidden state at the learnable token
                position.
            prefix_embs: ``[B, S, input_dim]`` target prefix embeddings.
            mask: optional ``[B, S]`` valid-token mask.

        Returns:
            ``(mse, {"mse": mse, "z_rl": z})``.
        """
        z = self.encode_z(z_rl)  # [B, embed_dim]
        rl_tokens = z.unsqueeze(1)  # [B, 1, embed_dim]
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

        metrics = {"mse": mse, "z_rl": z}
        if self.z_l2_weight > 0.0:
            z_f32 = z.to(dtype=torch.float32)
            z_l2 = torch.mean(torch.square(z_f32))
            loss = mse + self.z_l2_weight * z_l2
            metrics["z_l2"] = z_l2
        else:
            loss = mse

        return loss, metrics

    def forward(
        self,
        z_rl: torch.Tensor,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss(z_rl, prefix_embs, mask)
