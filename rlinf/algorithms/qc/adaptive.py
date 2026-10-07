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

"""AQC discount-normalized advantage selector (Li et al. / Gireesh et al.)."""

from __future__ import annotations

import torch


def discount_normalized_advantage(
    q_values: torch.Tensor,
    baselines: torch.Tensor,
    *,
    gamma: float,
    horizon: int,
) -> torch.Tensor:
    """``(Q^k - V^k) / gamma^k`` from AQC Eq. 10."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    scale = float(gamma) ** int(horizon)
    return (q_values - baselines) / scale


def zscore(values: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Z-score along ``dim`` as in AQC Eq. 16."""
    mean = values.mean(dim=dim, keepdim=True)
    std = values.std(dim=dim, keepdim=True, unbiased=False)
    return (values - mean) / (std + eps)


def select_adaptive_chunk(
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick ``(sample_index, horizon_index)`` of the max score per batch row.

    Args:
        scores: ``[batch, num_samples, num_horizons]``.

    Returns:
        ``(sample_idx, horizon_idx)`` each ``[batch]``.
    """
    if scores.dim() != 3:
        raise ValueError(
            f"scores must be [batch, N, |K|], got {tuple(scores.shape)}."
        )
    batch, num_samples, num_horizons = scores.shape
    flat = scores.reshape(batch, num_samples * num_horizons)
    flat_idx = torch.argmax(flat, dim=-1)
    sample_idx = torch.div(flat_idx, num_horizons, rounding_mode="floor")
    horizon_idx = flat_idx % num_horizons
    return sample_idx, horizon_idx
