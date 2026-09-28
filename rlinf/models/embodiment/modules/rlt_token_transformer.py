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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sinusoidal_pe_init(seq_len: int, embed_dim: int) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32)
        * -(math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(seq_len, embed_dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class GeGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(inputs).chunk(2, dim=-1)
        return x * F.gelu(gate)


class RLTSelfAttentionLayer(nn.Module):
    """Self-attention transformer block for RLT token modules."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)
        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.Dropout(dropout_rate),
            GeGLU(mlp_dim),
            nn.Linear(mlp_dim, embed_dim),
        )

    @staticmethod
    def _key_padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        return ~mask.to(dtype=torch.bool)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_weight = self.self_norm.weight
        x = x.to(device=norm_weight.device, dtype=norm_weight.dtype)
        key_padding_mask = self._key_padding_mask(mask)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=x.device)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=x.device)

        residual = x
        x_norm = self.self_norm(x)
        x = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = residual + x

        return x + self.mlp(self.mlp_norm(x))


class RLTCrossAttentionLayer(nn.Module):
    """Pre-LN transformer block with ``z`` as the query and tokens as memory.

    The block matches the RLT encoder layer: residual attention, then a
    residual GeGLU feed-forward. Query length is one, so the attention
    sublayer is cross-attention rather than self-attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.Dropout(dropout_rate),
            GeGLU(mlp_dim),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one Pre-LN cross-attention block. ``query`` is ``[B, D]``."""
        residual = query.unsqueeze(1)
        attn_out = self.attn(
            self.q_norm(residual),
            self.kv_norm(tokens),
            self.kv_norm(tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        hidden = residual + attn_out
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden.squeeze(1)


class RLTTokenEncoder(nn.Module):
    """Compress VLA prefix embeddings into a single RL token."""

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

        self.input_proj = (
            nn.Linear(self.input_dim, self.embed_dim)
            if self.input_dim != self.embed_dim
            else nn.Identity()
        )
        self.rl_token_embed = nn.Parameter(sinusoidal_pe_init(1, self.embed_dim))
        self.prefix_pos_enc = nn.Parameter(
            sinusoidal_pe_init(self.prefix_seq_len, self.embed_dim)
        )
        self.rl_token_pos_enc = nn.Parameter(sinusoidal_pe_init(1, self.embed_dim))
        self.extra_pos_enc = nn.Parameter(sinusoidal_pe_init(4, self.embed_dim))
        self.layers = nn.ModuleList(
            [
                RLTSelfAttentionLayer(
                    self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
        rl_token: torch.Tensor | None = None,
        extra_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prefix_embs = self.input_proj(prefix_embs)
        seq_len = prefix_embs.shape[-2]
        if seq_len > self.prefix_seq_len:
            raise ValueError(
                f"prefix sequence length {seq_len} exceeds configured "
                f"prefix_seq_len {self.prefix_seq_len}."
            )

        prefix_pos = self.prefix_pos_enc[:seq_len].to(
            device=prefix_embs.device, dtype=prefix_embs.dtype
        )
        prefix_tokens = prefix_embs + prefix_pos
        batch_size = prefix_embs.shape[0]
        parts = [prefix_tokens]
        extra_len = 0
        if extra_tokens is not None:
            extra_len = int(extra_tokens.shape[1])
            if extra_len > self.extra_pos_enc.shape[0]:
                raise ValueError(
                    f"extra_tokens length {extra_len} exceeds extra_pos_enc "
                    f"{self.extra_pos_enc.shape[0]}."
                )
            extra_pos = self.extra_pos_enc[:extra_len].to(
                device=prefix_embs.device, dtype=prefix_embs.dtype
            )
            parts.append(extra_tokens.to(dtype=prefix_embs.dtype) + extra_pos)
        if rl_token is None:
            rl_tokens = (
                self.rl_token_embed.to(
                    device=prefix_embs.device, dtype=prefix_embs.dtype
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
        else:
            rl_tokens = rl_token.to(
                device=prefix_embs.device, dtype=prefix_embs.dtype
            ).reshape(batch_size, 1, self.embed_dim)
        rl_pos = self.rl_token_pos_enc.to(
            device=prefix_embs.device, dtype=prefix_embs.dtype
        )
        rl_tokens = rl_tokens + rl_pos
        parts.append(rl_tokens)
        x = torch.cat(parts, dim=1)

        if mask is not None:
            mask = mask.to(device=prefix_embs.device, dtype=torch.bool)
            extra_ones = torch.ones(
                batch_size,
                extra_len + 1,
                device=prefix_embs.device,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, extra_ones], dim=1)

        for layer in self.layers:
            x = layer(x, mask=mask)
        return x[:, -1:]


class RLTCrossAttentionEncoder(nn.Module):
    """Compress a token window with a Pre-LN cross-attention transformer.

    ``z + mean(tokens)`` is the input of the first block only. Every block is
    the same residual cross-attention plus feed-forward. Later blocks read
    the residual ``z`` and do not add the mean again. Token positions stay
    fixed as the attention memory.
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
        heads = int(num_heads)
        if self.embed_dim % heads != 0:
            raise ValueError(
                f"embed_dim {self.embed_dim} must be divisible by num_heads {heads}."
            )
        self.input_proj = (
            nn.Linear(self.input_dim, self.embed_dim)
            if self.input_dim != self.embed_dim
            else nn.Identity()
        )
        self.prefix_pos_enc = nn.Parameter(
            sinusoidal_pe_init(self.prefix_seq_len, self.embed_dim)
        )
        self.z_init = nn.Parameter(torch.zeros(self.embed_dim))
        self.layers = nn.ModuleList(
            [
                RLTCrossAttentionLayer(
                    self.embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.out_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
        rl_token: torch.Tensor | None = None,
        extra_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del extra_tokens
        tokens = self.input_proj(prefix_embs)
        seq_len = tokens.shape[1]
        if seq_len > self.prefix_seq_len:
            raise ValueError(
                f"prefix sequence length {seq_len} exceeds configured "
                f"prefix_seq_len {self.prefix_seq_len}."
            )
        pos = self.prefix_pos_enc[:seq_len].to(
            device=tokens.device, dtype=tokens.dtype
        )
        tokens = tokens + pos
        batch = tokens.shape[0]
        if rl_token is None:
            z = self.z_init.to(device=tokens.device, dtype=tokens.dtype).expand(batch, -1)
        else:
            z = rl_token.to(device=tokens.device, dtype=tokens.dtype).reshape(batch, -1)
        if mask is None:
            pooled = tokens.mean(dim=1)
            key_padding = None
        else:
            valid = mask.to(device=tokens.device, dtype=torch.bool)
            empty = ~valid.any(dim=-1)
            if empty.any():
                valid = valid.clone()
                valid[empty, 0] = True
            weight = valid.to(dtype=tokens.dtype).unsqueeze(-1)
            pooled = (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
            key_padding = ~valid
        z = z + pooled
        for layer in self.layers:
            z = layer(z, tokens, key_padding)
        return self.out_norm(z).unsqueeze(1)


class RLTTokenDecoder(nn.Module):
    """Autoregressively reconstruct VLA prefix embeddings."""

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

        self.teacher_input_proj = (
            nn.Linear(self.input_dim, self.embed_dim)
            if self.input_dim != self.embed_dim
            else nn.Identity()
        )
        self.decoder_pos_enc = nn.Parameter(
            sinusoidal_pe_init(self.prefix_seq_len, self.embed_dim)
        )
        self.layers = nn.ModuleList(
            [
                RLTSelfAttentionLayer(
                    self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(self.embed_dim, self.input_dim)

    def forward(
        self,
        rl_tokens: torch.Tensor,
        target_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_seq_len = target_embeddings.shape[1]
        if target_seq_len > self.prefix_seq_len:
            raise ValueError(
                f"target sequence length {target_seq_len} exceeds configured "
                f"prefix_seq_len {self.prefix_seq_len}."
            )

        frozen_targets = target_embeddings.detach().to(
            device=rl_tokens.device,
            dtype=rl_tokens.dtype,
        )
        shifted_targets = frozen_targets[:, :-1]
        if mask is not None:
            shifted_targets = shifted_targets.masked_fill(
                ~mask[:, :-1]
                .to(device=shifted_targets.device, dtype=torch.bool)
                .unsqueeze(-1),
                0,
            )

        shifted_targets = self.teacher_input_proj(shifted_targets)
        decoder_inputs = torch.cat([rl_tokens, shifted_targets], dim=1)
        decoder_inputs = decoder_inputs + self.decoder_pos_enc[:target_seq_len].to(
            device=decoder_inputs.device,
            dtype=decoder_inputs.dtype,
        )

        # Boolean MultiheadAttention masks use True for disallowed positions.
        causal_mask = torch.triu(
            torch.ones(
                target_seq_len,
                target_seq_len,
                device=decoder_inputs.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        decoder_input_mask = None
        if mask is not None:
            start_mask = torch.ones(
                mask.shape[0],
                1,
                device=decoder_inputs.device,
                dtype=torch.bool,
            )
            decoder_input_mask = torch.cat(
                [
                    start_mask,
                    mask[:, :-1].to(
                        device=decoder_inputs.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )

        x = decoder_inputs
        for layer in self.layers:
            x = layer(x, mask=decoder_input_mask, attn_mask=causal_mask)
        return self.output_proj(x)


class RLTTokenTransformer(nn.Module):
    """Stage-1 loop: a multi-layer cross-attention stack compresses tokens into ``z``.

    The stack has the same depth and MLP as the RLT encoder. Training carries
    ``z`` across a same-episode window and reconstructs only the last chunk, so
    the decoder runs once. Stage 2 reuses this module online and does not
    reconstruct.
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

        self.encoder = RLTCrossAttentionEncoder(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            prefix_seq_len=self.prefix_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
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

    def encode(
        self,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
        rl_token: torch.Tensor | None = None,
        extra_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(
            prefix_embs,
            mask,
            rl_token=rl_token,
            extra_tokens=extra_tokens,
        )

    def encode_flat(
        self,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
        rl_token: torch.Tensor | None = None,
        extra_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(
            prefix_embs,
            mask,
            rl_token=rl_token,
            extra_tokens=extra_tokens,
        ).reshape(prefix_embs.shape[0], -1)

    def decode(
        self,
        rl_tokens: torch.Tensor,
        target_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(rl_tokens, target_embeddings, mask)

    def reconstruct(
        self, prefix_embs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frozen_prefix = prefix_embs.detach()
        rl_tokens = self.encode(frozen_prefix, mask)
        reconstructed = self.decode(rl_tokens, frozen_prefix, mask)
        return reconstructed, rl_tokens

    def loss(
        self,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
        recon_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Reconstruct ``prefix_embs``. ``recon_mask`` limits the supervised tokens.

        ``mask`` is what the encoder and decoder may read. ``recon_mask`` is the
        bottleneck target, typically the text-attention top-k image tokens.
        """
        reconstructed, rl_tokens = self.reconstruct(prefix_embs, mask)
        target = prefix_embs.detach().to(dtype=torch.float32)
        reconstructed = reconstructed.to(dtype=torch.float32)
        sq_error = torch.square(reconstructed - target)
        loss_mask = mask if recon_mask is None else recon_mask
        if mask is not None and recon_mask is not None:
            loss_mask = mask.to(dtype=torch.bool) & recon_mask.to(dtype=torch.bool)

        if loss_mask is not None:
            mask_expanded = loss_mask.to(device=sq_error.device, dtype=sq_error.dtype)[
                ..., None
            ]
            sq_error = sq_error * mask_expanded
            denom = torch.clamp(mask_expanded.sum() * prefix_embs.shape[-1], min=1.0)
            mse = sq_error.sum() / denom
        else:
            mse = sq_error.mean()

        return mse, {
            "mse": mse,
            "z_rl": rl_tokens.reshape(prefix_embs.shape[0], -1),
        }

    def _crop_window_valid(
        self,
        hist_valid: torch.Tensor,
        mem_len_min: int | None,
        mem_len_max: int | None,
    ) -> torch.Tensor:
        """Keep a suffix of each row. Training draws its length uniformly."""
        valid = hist_valid.to(dtype=torch.bool)
        if valid.dim() != 2:
            raise ValueError(
                "hist_valid must have shape [batch, steps]; "
                f"got {tuple(valid.shape)}."
            )
        if mem_len_min is None or mem_len_max is None:
            return valid
        steps = int(valid.shape[1])
        low = int(mem_len_min)
        high = min(int(mem_len_max), steps)
        if low < 0 or high < 0 or low > int(mem_len_max):
            raise ValueError(
                "mem window length must satisfy 0 <= min <= max, got "
                f"min={mem_len_min} max={mem_len_max}."
            )
        low = min(low, high)
        positions = torch.arange(steps, device=valid.device)
        if self.training and low < high:
            lengths = torch.randint(
                low, high + 1, (valid.shape[0],), device=valid.device
            )
            keep = positions.unsqueeze(0) >= (steps - lengths.unsqueeze(1))
        else:
            keep = positions >= (steps - high)
            keep = keep.unsqueeze(0)
        return valid & keep

    def encode_window(
        self,
        hist_prefix: torch.Tensor,
        hist_mask: torch.Tensor | None,
        hist_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``z`` through a window. The first valid step starts from ``z_init``.

        Args:
            hist_prefix: ``[batch, steps, seq, dim]`` token sequences, oldest first.
            hist_mask: optional ``[batch, steps, seq]`` encoder mask.
            hist_valid: ``[batch, steps]`` bool, true where that chunk is in-episode.

        Returns:
            Final ``z`` of shape ``[batch, 1, dim]`` and per-step ``z`` of shape
            ``[batch, steps, dim]``. Invalid steps leave that row's ``z`` unchanged.
        """
        if hist_prefix.dim() != 4:
            raise ValueError(
                "hist_prefix must have shape [batch, steps, seq, dim]; "
                f"got {tuple(hist_prefix.shape)}."
            )
        batch, steps, _, _ = hist_prefix.shape
        valid = hist_valid.to(device=hist_prefix.device, dtype=torch.bool).reshape(
            batch, steps
        )
        z_init = self.encoder.z_init.to(
            device=hist_prefix.device, dtype=hist_prefix.dtype
        )
        z_state = z_init.view(1, 1, -1).expand(batch, 1, -1)
        z_seq: list[torch.Tensor] = []
        for step in range(steps):
            step_valid = valid[:, step]
            if bool(step_valid.any()):
                mask_t = None if hist_mask is None else hist_mask[:, step]
                prev = z_state.reshape(batch, -1)
                tokens = hist_prefix[:, step]
                if self.training and torch.is_grad_enabled():
                    if mask_t is None:
                        z_new = checkpoint(
                            lambda tok, prev_z: self.encode(tok, None, rl_token=prev_z),
                            tokens,
                            prev,
                            use_reentrant=False,
                        )
                    else:
                        z_new = checkpoint(
                            lambda tok, msk, prev_z: self.encode(
                                tok, msk, rl_token=prev_z
                            ),
                            tokens,
                            mask_t,
                            prev,
                            use_reentrant=False,
                        )
                else:
                    z_new = self.encode(tokens, mask_t, rl_token=prev)
                if z_state.dtype != z_new.dtype or z_state.device != z_new.device:
                    z_state = z_state.to(device=z_new.device, dtype=z_new.dtype)
                take = step_valid.view(batch, 1, 1)
                z_state = torch.where(take, z_new, z_state)
            z_seq.append(z_state.reshape(batch, -1))
        return z_state, torch.stack(z_seq, dim=1)

    def _reconstruction_mse(
        self,
        rl_tokens: torch.Tensor,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None,
        recon_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        reconstructed = self.decode(rl_tokens, prefix_embs, mask)
        target = prefix_embs.detach().to(dtype=torch.float32)
        sq_error = torch.square(reconstructed.to(dtype=torch.float32) - target)
        loss_mask = mask if recon_mask is None else recon_mask
        if mask is not None and recon_mask is not None:
            loss_mask = mask.to(dtype=torch.bool) & recon_mask.to(dtype=torch.bool)
        if loss_mask is None:
            return sq_error.mean()
        mask_expanded = loss_mask.to(device=sq_error.device, dtype=sq_error.dtype)[
            ..., None
        ]
        denom = torch.clamp(mask_expanded.sum() * prefix_embs.shape[-1], min=1.0)
        return (sq_error * mask_expanded).sum() / denom

    def step_memory(
        self,
        prefix_embs: torch.Tensor,
        mask: torch.Tensor | None,
        z_prev: torch.Tensor | None,
        reset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write one online chunk. Reset rows start from ``z_init``."""
        batch = prefix_embs.shape[0]
        z_init = self.encoder.z_init.to(
            device=prefix_embs.device, dtype=prefix_embs.dtype
        )
        if z_prev is None or int(z_prev.shape[0]) != batch:
            prev = z_init.expand(batch, -1)
        else:
            prev = z_prev.reshape(batch, -1).to(
                device=prefix_embs.device, dtype=prefix_embs.dtype
            )
        if reset is not None:
            reset = reset.reshape(batch).to(device=prefix_embs.device, dtype=torch.bool)
            prev = torch.where(reset.unsqueeze(-1), z_init.expand(batch, -1), prev)
        return self.encode_flat(prefix_embs, mask, rl_token=prev)

    def sequence_loss(
        self,
        hist_prefix: torch.Tensor,
        hist_mask: torch.Tensor | None,
        hist_valid: torch.Tensor,
        hist_recon_mask: torch.Tensor | None = None,
        *,
        mem_len_min: int | None = None,
        mem_len_max: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Carry ``z`` across the window and reconstruct only the last chunk.

        Earlier chunks still update ``z``. The decoder runs once, on the final
        ``z`` and that chunk's top-k target.
        """
        hist_prefix = hist_prefix.detach()
        valid = self._crop_window_valid(hist_valid, mem_len_min, mem_len_max)
        valid = valid.to(device=hist_prefix.device)
        row_len = valid.sum(dim=-1)
        active = row_len > 0
        if not bool(active.any()):
            return self._empty_window_loss(int(valid.shape[0]), hist_prefix.device)
        z_final, _ = self.encode_window(hist_prefix, hist_mask, valid)
        last = valid.shape[1] - 1 - valid.flip(dims=(-1,)).to(dtype=torch.long).argmax(
            dim=-1
        )
        rows = torch.arange(valid.shape[0], device=valid.device)
        rows = rows[active]
        last = last[active]
        mask_t = None if hist_mask is None else hist_mask[rows, last]
        recon_t = None if hist_recon_mask is None else hist_recon_mask[rows, last]
        mse = self._reconstruction_mse(
            z_final[active],
            hist_prefix[rows, last],
            mask_t,
            recon_t,
        )
        loop_len = valid.to(dtype=mse.dtype).sum(dim=-1).mean()
        return mse, {
            "mse": mse,
            "mem_loop_len": loop_len,
            "z_rl": z_final.reshape(hist_prefix.shape[0], -1),
        }

    def _empty_window_loss(
        self, batch: int, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Length 0 has no chunk to reconstruct, but still touches parameters."""
        zero = torch.zeros((), device=device, dtype=torch.float32)
        for param in self.parameters():
            zero = zero + param.reshape(-1)[:1].float().sum() * 0
        z = self.encoder.z_init.to(device=device, dtype=torch.float32)
        z = z.reshape(1, -1).expand(batch, -1)
        return zero, {
            "mse": zero.detach(),
            "mem_loop_len": zero.detach(),
            "z_rl": z.detach(),
        }

    def forward(
        self, prefix_embs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss(prefix_embs, mask)
