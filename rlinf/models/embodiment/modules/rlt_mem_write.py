# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory compressor for RLT.

Each chunk adds ``mean(I_t)`` to ``z`` once, then runs ``num_layers`` Pre-LN
blocks. Scheme 1 is a cross-attention stack: ``z`` queries ``I_t``. Scheme 2
alternates one cross-attention block and one self-attention block. The
self-attention block may concatenate ``I_t`` with ``z``, and the next block
receives only ``z``. A final LayerNorm is the carried state. ``ref_chunk`` is
an actor input. The decoder reconstructs the text-attention top-k image tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTCrossAttentionLayer,
    RLTSelfAttentionLayer,
    RLTTokenDecoder,
    sinusoidal_pe_init,
)


def pool_rlt_prefix(
    prefix_embs: torch.Tensor,
    prefix_mask: torch.Tensor | None,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool prefix tokens to ``length``. The memory path stores top-k tokens instead."""
    length = int(length)
    if prefix_embs.shape[1] == length:
        if prefix_mask is None:
            prefix_mask = torch.ones(
                prefix_embs.shape[0],
                length,
                device=prefix_embs.device,
                dtype=torch.bool,
            )
        return prefix_embs, prefix_mask.to(dtype=torch.bool)
    pooled = F.adaptive_avg_pool1d(prefix_embs.transpose(1, 2), length).transpose(1, 2)
    mask = torch.ones(
        prefix_embs.shape[0],
        length,
        device=prefix_embs.device,
        dtype=torch.bool,
    )
    return pooled, mask


def text_image_attention_scores(
    prefix: torch.Tensor,
    image_len: int,
    prefix_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score image tokens by how much the text tokens attend to them.

    ``prefix`` is the VLM last-layer hidden state, laid out as image tokens
    followed by text tokens. The score is a masked softmax of text queries
    against image keys, averaged over the text positions.
    """
    image_len = int(image_len)
    image = prefix[:, :image_len].float()
    text = prefix[:, image_len:].float()
    if text.shape[1] == 0:
        return torch.ones(
            prefix.shape[0], image_len, device=prefix.device, dtype=torch.float32
        )
    scale = prefix.shape[-1] ** -0.5
    logits = torch.matmul(text, image.transpose(-1, -2)) * scale
    if prefix_mask is not None:
        text_mask = prefix_mask[:, image_len:].to(device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(~text_mask.unsqueeze(-1), -1.0e4)
    else:
        text_mask = torch.ones(
            text.shape[:2], device=logits.device, dtype=torch.bool
        )
    weights = torch.softmax(logits, dim=-1) * text_mask.unsqueeze(-1).to(logits.dtype)
    denom = text_mask.sum(dim=-1).clamp(min=1).unsqueeze(-1).to(logits.dtype)
    return weights.sum(dim=1) / denom


def topk_mask_from_scores(
    scores: torch.Tensor,
    valid: torch.Tensor | None,
    ratio: float,
) -> torch.Tensor:
    """Keep the top ``ratio`` fraction of valid positions in each row."""
    ratio = float(ratio)
    if valid is None:
        valid = torch.ones(scores.shape, device=scores.device, dtype=torch.bool)
    else:
        valid = valid.to(device=scores.device, dtype=torch.bool)
    if ratio >= 1.0:
        return valid
    masked_scores = scores.float().masked_fill(~valid, -1.0e4)
    counts = valid.sum(dim=-1)
    keep = torch.zeros_like(valid)
    for row in range(scores.shape[0]):
        count = int(counts[row].item())
        if count <= 0:
            continue
        k = max(1, int(round(ratio * count)))
        k = min(k, count)
        chosen = torch.topk(masked_scores[row], k=k).indices
        keep[row, chosen] = True
    return keep


def pack_masked_tokens(
    tokens: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack selected tokens to the front, preserving their order."""
    mask = mask.to(device=tokens.device, dtype=torch.bool)
    width = max(int(mask.sum(dim=-1).max().item()), 1)
    packed = tokens.new_zeros(tokens.shape[0], width, tokens.shape[-1])
    packed_mask = torch.zeros(
        tokens.shape[0], width, device=tokens.device, dtype=torch.bool
    )
    position = mask.long().cumsum(dim=-1) - 1
    batch_index = torch.arange(tokens.shape[0], device=tokens.device)[:, None]
    batch_index = batch_index.expand_as(position)
    packed[batch_index[mask], position[mask]] = tokens[mask]
    packed_mask[batch_index[mask], position[mask]] = True
    return packed, packed_mask


def full_image_tokens_and_recon_mask(
    prefix: torch.Tensor,
    prefix_mask: torch.Tensor | None,
    lang_len: int,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Keep every image token, and mark the text-attention top-k for reconstruction.

    The compressor reads the full image span. ``recon_mask`` is the only place
    the top-k selection is applied.
    """
    lang_len = int(lang_len)
    image_len = int(prefix.shape[1] if lang_len <= 0 else prefix.shape[1] - lang_len)
    image = prefix[:, :image_len]
    if prefix_mask is None:
        image_mask = torch.ones(
            prefix.shape[0],
            image_len,
            device=prefix.device,
            dtype=torch.bool,
        )
    else:
        image_mask = prefix_mask[:, :image_len].to(dtype=torch.bool)
    ratio = float(ratio)
    if ratio >= 1.0 or lang_len <= 0 or image_len <= 0:
        return image, image_mask, None
    scores = text_image_attention_scores(prefix, image_len, prefix_mask)
    recon_mask = topk_mask_from_scores(scores, image_mask, ratio)
    return image, image_mask, recon_mask


def bottleneck_token_masks(
    prefix_mask: torch.Tensor,
    scores: torch.Tensor | None,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Full prefix for the encoder, text-attention top-k image tokens to rebuild.

    The VLM already saw the complete image and text. ``prefix_mask`` stays that
    full sequence. The reconstruction mask keeps the highest-scoring image
    tokens and leaves text positions out of the decoder target.
    """
    ratio = float(ratio)
    if scores is None or ratio >= 1.0:
        return prefix_mask, None
    image_len = int(scores.shape[-1])
    if image_len > prefix_mask.shape[1]:
        return prefix_mask, None
    image_valid = prefix_mask[:, :image_len].to(dtype=torch.bool)
    topk = topk_mask_from_scores(scores, image_valid, ratio)
    recon_mask = torch.zeros_like(prefix_mask)
    recon_mask[:, :image_len] = topk
    return prefix_mask, recon_mask


class RLTLoopEncoder(nn.Module):
    """Compress a variable-length image-token window into ``z``."""

    def __init__(
        self,
        z_dim: int,
        action_dim: int = 0,
        prefix_len: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        mem_scheme: int = 1,
    ):
        super().__init__()
        del action_dim
        z_dim = int(z_dim)
        self.z_dim = z_dim
        self.prefix_len = int(prefix_len)
        self.mem_scheme = int(mem_scheme)
        heads = int(num_heads)
        depth = int(num_layers)
        if z_dim % heads != 0:
            raise ValueError(f"z_dim {z_dim} must be divisible by num_heads {heads}.")
        if self.mem_scheme not in (1, 2):
            raise ValueError(f"mem_scheme must be 1 or 2, got {self.mem_scheme}.")
        if self.mem_scheme == 1:
            self.block_kinds = tuple("cross" for _ in range(depth))
        else:
            self.block_kinds = tuple(
                "cross" if index % 2 == 0 else "self" for index in range(depth)
            )
        n_cross = self.block_kinds.count("cross")
        n_self = self.block_kinds.count("self")
        self.prefix_pos_enc = nn.Parameter(sinusoidal_pe_init(self.prefix_len, z_dim))
        if n_self > 0:
            self.z_pos_enc = nn.Parameter(sinusoidal_pe_init(1, z_dim))
        self.cross_layers = nn.ModuleList(
            [
                RLTCrossAttentionLayer(z_dim, num_heads=heads)
                for _ in range(n_cross)
            ]
        )
        self.self_layers = nn.ModuleList(
            [
                RLTSelfAttentionLayer(z_dim, num_heads=heads)
                for _ in range(n_self)
            ]
        )
        self.layers = self.cross_layers
        self.out_norm = nn.LayerNorm(z_dim)
        self.mem_init = nn.Parameter(torch.zeros(z_dim))
        self.decoder = RLTTokenDecoder(
            input_dim=z_dim,
            embed_dim=z_dim,
            prefix_seq_len=self.prefix_len,
            num_layers=int(num_layers),
            num_heads=heads,
        )

    def initial_mem(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return self.mem_init.to(device=device, dtype=dtype).unsqueeze(0).expand(
            batch_size, -1
        )

    @staticmethod
    def _masked_mean(
        tokens: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        weight = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

    def write(
        self,
        prefix_embs: torch.Tensor,
        prefix_mask: torch.Tensor | None,
        z_prev: torch.Tensor,
        *,
        detach_prev: bool = True,
    ) -> torch.Tensor:
        """Add the token mean once, then run the residual transformer stack."""
        if detach_prev:
            z_prev = z_prev.detach()
        compute = self.out_norm.weight
        tokens = prefix_embs.to(device=compute.device, dtype=compute.dtype)
        z_prev = z_prev.to(device=compute.device, dtype=compute.dtype)
        seq_len = tokens.shape[1]
        if seq_len > self.prefix_len:
            raise ValueError(
                f"prefix sequence length {seq_len} exceeds the position table "
                f"loop_prefix_len {self.prefix_len}."
            )
        pos = self.prefix_pos_enc[:seq_len].to(
            device=tokens.device, dtype=tokens.dtype
        )
        tokens = tokens + pos
        mask = None if prefix_mask is None else prefix_mask.to(device=tokens.device)
        if mask is not None:
            empty = ~mask.reshape(tokens.shape[0], -1).any(dim=-1)
            if empty.any():
                mask = mask.clone()
                mask[empty, 0] = True
        else:
            empty = None
        key_padding = None if mask is None else ~mask.to(dtype=torch.bool)
        z = z_prev + self._masked_mean(tokens, mask)
        cross_index = 0
        self_index = 0
        for kind in self.block_kinds:
            if kind == "cross":
                z = self.cross_layers[cross_index](z, tokens, key_padding)
                cross_index += 1
            else:
                z = self._self_ffn(
                    z, tokens, mask, self.self_layers[self_index]
                )
                self_index += 1
        z = self.out_norm(z.to(dtype=self.out_norm.weight.dtype))
        if empty is not None and empty.any():
            z = torch.where(empty.unsqueeze(-1), z_prev, z)
        return z

    def _self_ffn(
        self,
        z: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        layer: nn.Module,
    ) -> torch.Tensor:
        """Self-attend ``[tokens; z]`` and return only the ``z`` position."""
        weight = layer.self_norm.weight
        tokens = tokens.to(device=weight.device, dtype=weight.dtype)
        z = z.to(device=weight.device, dtype=weight.dtype)
        z_tok = z + self.z_pos_enc.to(device=z.device, dtype=z.dtype)
        sequence = torch.cat([tokens, z_tok.unsqueeze(1)], dim=1)
        if token_mask is None:
            attn_mask = None
        else:
            z_valid = torch.ones(
                token_mask.shape[0],
                1,
                device=token_mask.device,
                dtype=torch.bool,
            )
            attn_mask = torch.cat(
                [token_mask.to(dtype=torch.bool), z_valid], dim=1
            )
        sequence = layer(sequence, mask=attn_mask)
        return sequence[:, -1]

    def unroll(
        self,
        hist_prefix: torch.Tensor,
        hist_mask: torch.Tensor | None,
        hist_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write ``z`` through a same-episode window. Gradients flow across steps."""
        if hist_prefix.dim() == 5:
            hist_prefix = hist_prefix.reshape(
                hist_prefix.shape[0], -1, hist_prefix.shape[-2], hist_prefix.shape[-1]
            )
        batch, steps, _, _ = hist_prefix.shape
        device, dtype = hist_prefix.device, hist_prefix.dtype
        valid = hist_valid.to(device=device).reshape(batch, steps, 1).to(dtype=dtype)
        z = self.initial_mem(batch, device, dtype)
        z_seq = []
        for step in range(steps):
            mask_t = None if hist_mask is None else hist_mask[:, step]
            z_new = self.write(
                hist_prefix[:, step],
                mask_t,
                z,
                detach_prev=False,
            )
            z = z_new * valid[:, step] + z * (1.0 - valid[:, step])
            z_seq.append(z)
        return z, torch.stack(z_seq, dim=1), hist_valid.to(device=device).reshape(
            batch, steps
        )

    def reconstruction_loss(
        self,
        z: torch.Tensor,
        prefix_embs: torch.Tensor,
        prefix_mask: torch.Tensor | None = None,
        recon_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive decoder loss on the selected tokens only."""
        tokens = prefix_embs.to(dtype=z.dtype)
        mask = prefix_mask
        if recon_mask is not None:
            recon_mask = recon_mask.to(device=tokens.device, dtype=torch.bool)
            mask = recon_mask if mask is None else mask.to(dtype=torch.bool) & recon_mask
        reconstructed = self.decoder(z.reshape(z.shape[0], 1, -1), tokens, mask)
        sq_error = torch.square(reconstructed.float() - tokens.detach().float())
        if mask is None:
            return sq_error.mean()
        weight = mask.to(device=sq_error.device, dtype=sq_error.dtype).unsqueeze(-1)
        denom = torch.clamp(weight.sum() * tokens.shape[-1], min=1.0)
        return (sq_error * weight).sum() / denom

    def sequence_reconstruction_loss(
        self,
        z_seq: torch.Tensor,
        hist_prefix: torch.Tensor,
        hist_mask: torch.Tensor | None,
        hist_valid: torch.Tensor,
        hist_recon_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruction at every valid step of a history window."""
        steps = z_seq.shape[1]
        total = z_seq.new_zeros(())
        counted = z_seq.new_zeros(())
        valid = hist_valid.to(device=z_seq.device).reshape(z_seq.shape[0], steps)
        for step in range(steps):
            step_valid = valid[:, step]
            if not bool(step_valid.any()):
                continue
            mask_t = None if hist_mask is None else hist_mask[:, step]
            recon_t = None if hist_recon_mask is None else hist_recon_mask[:, step]
            loss = self.reconstruction_loss(
                z_seq[:, step],
                hist_prefix[:, step],
                mask_t,
                recon_t,
            )
            weight = step_valid.to(dtype=loss.dtype).mean()
            total = total + loss * weight
            counted = counted + weight
        return total / torch.clamp(counted, min=1.0)
