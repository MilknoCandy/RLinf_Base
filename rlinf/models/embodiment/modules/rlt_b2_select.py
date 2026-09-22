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

"""Select image tokens with last-layer text-to-image attention for B2."""

from __future__ import annotations

import torch


def text_to_image_attention_scores(
    attn_probs: torch.Tensor,
    *,
    num_image_tokens: int,
    lang_len: int,
) -> torch.Tensor:
    """Pool last-layer attention into per-image-token scores.

    ``attn_probs`` is either already ``[B, S]`` scores, or full last-layer
    probabilities ``[B, kv_heads, groups, T, S]`` / ``[B, heads, T, S]``.
    """
    if attn_probs.dim() == 2:
        scores = attn_probs
    elif attn_probs.dim() == 4:
        scores = attn_probs.mean(dim=1)
    elif attn_probs.dim() == 5:
        scores = attn_probs.mean(dim=(1, 2))
    else:
        raise ValueError(
            "attn_probs must have shape [B, S], [B, H, T, S], or "
            f"[B, K, G, T, S]; got {tuple(attn_probs.shape)}."
        )

    seq_len = scores.shape[-1]
    if num_image_tokens < 0 or num_image_tokens > seq_len:
        raise ValueError(
            f"num_image_tokens={num_image_tokens} is outside prefix length "
            f"{seq_len}."
        )
    if lang_len < 0 or num_image_tokens + lang_len > seq_len:
        raise ValueError(
            f"lang_len={lang_len} with {num_image_tokens} image tokens exceeds "
            f"prefix length {seq_len}."
        )
    if scores.dim() == 2:
        return scores[:, :num_image_tokens]

    query_start = num_image_tokens
    query_end = num_image_tokens + lang_len if lang_len > 0 else scores.shape[-2]
    if query_end <= query_start:
        return scores.mean(dim=-2)[:, :num_image_tokens]
    text_queries = scores[:, query_start:query_end, :num_image_tokens]
    return text_queries.mean(dim=-2)


def select_topk_image_tokens(
    prefix_output: torch.Tensor,
    prefix_mask: torch.Tensor | None,
    *,
    attn_probs: torch.Tensor,
    num_image_tokens: int,
    keep_ratio: float = 0.5,
    lang_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep a fixed fraction of image tokens ranked by text attention.

    Language positions are dropped. Dummy/padded image tokens with
    ``prefix_mask=False`` are never selected. Returns packed tokens and a
    dense True mask of the kept length (the same K for every batch item).
    """
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}.")
    if prefix_output.shape[1] < num_image_tokens:
        raise ValueError(
            f"prefix length {prefix_output.shape[1]} is smaller than "
            f"num_image_tokens={num_image_tokens}."
        )

    image_tokens = prefix_output[:, :num_image_tokens]
    if prefix_mask is None:
        image_mask = torch.ones(
            image_tokens.shape[:2],
            dtype=torch.bool,
            device=image_tokens.device,
        )
    else:
        image_mask = prefix_mask[:, :num_image_tokens].to(dtype=torch.bool)

    scores = text_to_image_attention_scores(
        attn_probs,
        num_image_tokens=num_image_tokens,
        lang_len=lang_len,
    ).to(device=image_tokens.device, dtype=torch.float32)
    if scores.shape != image_mask.shape:
        raise ValueError(
            f"attention scores shape {tuple(scores.shape)} does not match "
            f"image token shape {tuple(image_mask.shape)}."
        )
    scores = scores.masked_fill(~image_mask, torch.finfo(scores.dtype).min)

    valid_counts = image_mask.sum(dim=-1)
    if torch.any(valid_counts <= 0):
        raise ValueError("each batch item must have at least one valid image token.")
    keep = torch.clamp(
        torch.round(valid_counts.float() * float(keep_ratio)).to(dtype=torch.long),
        min=1,
    )
    keep_len = int(keep.max().item())
    _, ranked = torch.topk(scores, k=keep_len, dim=-1)
    gather_index = ranked.unsqueeze(-1).expand(-1, -1, image_tokens.shape[-1])
    selected = torch.gather(image_tokens, dim=1, index=gather_index)
    selected_mask = torch.arange(keep_len, device=image_tokens.device).unsqueeze(
        0
    ) < keep.unsqueeze(1)
    return selected, selected_mask
