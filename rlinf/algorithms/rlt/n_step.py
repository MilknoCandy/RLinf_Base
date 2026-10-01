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

"""N-step chunk returns for RLT transition replay.

One replay row is one action chunk. An ``n_step`` of ``k`` looks ahead at most
``k`` chunks along the same episode, folds their within-chunk discounted
rewards, and bootstraps from the observation after the last included chunk.
"""

from __future__ import annotations

import torch


def discounted_chunk_reward(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discount per-control-step rewards inside one chunk into a scalar."""
    flat = rewards.detach().float().reshape(-1)
    if flat.numel() == 0:
        return torch.zeros((), dtype=torch.float32)
    discounts = torch.pow(
        torch.as_tensor(gamma, dtype=flat.dtype),
        torch.arange(flat.numel(), dtype=flat.dtype),
    )
    return torch.sum(flat * discounts)


def compute_n_step_chunk_target(
    chunk_rewards: list[torch.Tensor],
    chunk_dones: list[bool],
    *,
    gamma: float,
    n_step: int,
) -> tuple[torch.Tensor, float, int, bool]:
    """Fold up to ``n_step`` ordered chunks into an n-step return.

    Args:
        chunk_rewards: Per-chunk reward tensors aligned with the episode order,
            starting at the query chunk.
        chunk_dones: Whether each chunk ends the episode.
        gamma: Discount applied between control steps and between chunks.
        n_step: Maximum number of chunks to include (``>= 1``).

    Returns:
        ``(n_step_return, bootstrap_discount, bootstrap_offset, bootstrapped)``
        where ``bootstrap_offset`` indexes the chunk whose ``next_obs`` is used
        for bootstrapping, and ``bootstrapped`` is False when the window hits a
        terminal chunk (no Q bootstrap).
    """
    if n_step < 1:
        raise ValueError(f"n_step must be >= 1, got {n_step}.")
    if not chunk_rewards:
        raise ValueError("chunk_rewards must be non-empty.")
    if len(chunk_rewards) != len(chunk_dones):
        raise ValueError("chunk_rewards and chunk_dones must have the same length.")

    n_step_return = torch.zeros((), dtype=torch.float32)
    discount = 1.0
    bootstrap_offset = 0
    bootstrapped = True
    limit = min(int(n_step), len(chunk_rewards))
    for offset in range(limit):
        reward = discounted_chunk_reward(chunk_rewards[offset], gamma)
        horizon = max(int(chunk_rewards[offset].detach().reshape(-1).numel()), 1)
        n_step_return = n_step_return + discount * reward
        bootstrap_offset = offset
        if chunk_dones[offset]:
            bootstrapped = False
            break
        discount *= float(gamma) ** horizon
    bootstrap_discount = discount if bootstrapped else 0.0
    return n_step_return, bootstrap_discount, bootstrap_offset, bootstrapped
