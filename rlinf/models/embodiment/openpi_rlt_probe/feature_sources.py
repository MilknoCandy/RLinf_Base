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

"""Feature-source selection for the Stage 2 RLT probe.

Stage 2 keeps the Stage 1 VLM frozen and only changes how ``z_rl`` is produced.
Each feature source maps an intermediate VLM tensor to a ``(B, S, D)`` sequence
plus a boolean mask; the probe then either runs the frozen RLT token
transformer on that sequence (token mode) or mean-pools it (no-token mode).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Tuple

import torch


@dataclasses.dataclass
class RLTFeatureBundle:
    """Intermediate tensors a feature source may consume.

    ``prefix_output``/``prefix_mask`` follow the PaliGemma prefix layout: image
    tokens first, then language tokens (``prompt_tokens`` length ``L``).
    """

    prefix_output: torch.Tensor | None = None
    prefix_mask: torch.Tensor | None = None
    prompt_tokens: torch.Tensor | None = None
    action_output: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None


FeatureSourceFn = Callable[
    [RLTFeatureBundle],
    Tuple[torch.Tensor, torch.Tensor],
]

_FEATURE_SOURCES: dict[str, FeatureSourceFn] = {}


def register_feature_source(name: str) -> Callable[[FeatureSourceFn], FeatureSourceFn]:
    """Register a feature-source extractor under ``name``."""

    def _decorator(fn: FeatureSourceFn) -> FeatureSourceFn:
        _FEATURE_SOURCES[name] = fn
        return fn

    return _decorator


def available_feature_sources() -> tuple[str, ...]:
    """Return the registered feature-source names, insertion-ordered."""
    return tuple(_FEATURE_SOURCES)


def get_feature_source(name: str) -> FeatureSourceFn:
    try:
        return _FEATURE_SOURCES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown rlt_feature_source={name!r}; available sources: "
            f"{sorted(_FEATURE_SOURCES)}."
        ) from exc


def _require_prefix(bundle: RLTFeatureBundle) -> tuple[torch.Tensor, torch.Tensor]:
    if bundle.prefix_output is None or bundle.prefix_mask is None:
        raise ValueError(
            "This feature source requires the PaliGemma prefix hidden states."
        )
    return bundle.prefix_output, bundle.prefix_mask


def _prompt_len(bundle: RLTFeatureBundle) -> int:
    if bundle.prompt_tokens is None:
        return 0
    return int(bundle.prompt_tokens.shape[1])


def _slice_prefix(
    bundle: RLTFeatureBundle,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix, mask = _require_prefix(bundle)
    return prefix[:, start:stop].contiguous(), mask[:, start:stop].contiguous()


@register_feature_source("all")
def _all(bundle: RLTFeatureBundle) -> tuple[torch.Tensor, torch.Tensor]:
    """Full PaliGemma prefix: images + language [+ point cloud]."""
    return _require_prefix(bundle)


@register_feature_source("image")
def _image(bundle: RLTFeatureBundle) -> tuple[torch.Tensor, torch.Tensor]:
    """Image-token prefix hidden states only."""
    prompt_len = _prompt_len(bundle)
    num_image_tokens = (
        bundle.prefix_output.shape[1] - prompt_len
        if prompt_len
        else bundle.prefix_output.shape[1]
    )
    if num_image_tokens <= 0:
        raise ValueError("image feature source requires image tokens in the prefix.")
    return _slice_prefix(bundle, 0, num_image_tokens)


@register_feature_source("language")
def _language(bundle: RLTFeatureBundle) -> tuple[torch.Tensor, torch.Tensor]:
    """Language/text prompt prefix hidden states only."""
    prompt_len = _prompt_len(bundle)
    if prompt_len == 0:
        raise ValueError(
            "language feature source requires tokenized prompt tokens; "
            "prompt_tokens was None or empty."
        )
    num_image_tokens = bundle.prefix_output.shape[1] - prompt_len
    return _slice_prefix(
        bundle, num_image_tokens, num_image_tokens + prompt_len
    )


@register_feature_source("text")
def _text(bundle: RLTFeatureBundle) -> tuple[torch.Tensor, torch.Tensor]:
    """Alias for the language/text prompt source."""
    return _language(bundle)


@register_feature_source("action")
def _action(bundle: RLTFeatureBundle) -> tuple[torch.Tensor, torch.Tensor]:
    """Action-expert suffix hidden states for the action chunk."""
    if bundle.action_output is None or bundle.action_mask is None:
        raise ValueError(
            "action feature source requires action-expert suffix features; "
            "pass action_output/action_mask in the RLTFeatureBundle."
        )
    return bundle.action_output, bundle.action_mask


def select_features(
    source: str,
    bundle: RLTFeatureBundle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the token sequence produced by ``source``."""
    features, mask = get_feature_source(source)(bundle)
    if mask is not None and mask.shape[:2] != features.shape[:2]:
        raise ValueError(
            f"feature mask shape {tuple(mask.shape)} does not match feature "
            f"shape {tuple(features.shape)} for source {source!r}."
        )
    return features, mask


def mean_pool_features(
    features: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean-pool a ``(B, S, D)`` feature sequence to ``(B, D)`` using ``mask``."""
    if mask is None:
        return features.mean(dim=1)
    weights = mask.to(dtype=features.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (features * weights).sum(dim=1) / denom
