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

"""Best-of-N helpers for Q-chunking.

Official QC (``agents/acfql.py``) uses Best-of-N both to interact with the
environment and to choose ``a_next`` in the chunk TD backup. Samples come from
the behavior policy (here: residual actor + frozen VLA ``ref_chunk``).
"""

from __future__ import annotations

from typing import Any

import torch


def flatten_chunk_actions(
    actions: torch.Tensor,
    *,
    chunk_len: int,
    action_dim: int,
) -> torch.Tensor:
    """Flatten a chunk to ``[batch, chunk_len * action_dim]``."""
    if actions.dim() <= 2:
        expected = chunk_len * action_dim
        if actions.shape[-1] != expected:
            raise ValueError(
                "Flat action dim mismatch: expected "
                f"{expected} from chunk_len={chunk_len} action_dim={action_dim}, "
                f"got {tuple(actions.shape)}."
            )
        return actions
    if actions.shape[-2] < chunk_len or actions.shape[-1] != action_dim:
        raise ValueError(
            "Chunk action shape mismatch: expected "
            f"[..., {chunk_len}, {action_dim}], got {tuple(actions.shape)}."
        )
    return actions[..., :chunk_len, :].reshape(actions.shape[0], chunk_len * action_dim)


def repeat_obs(obs: dict[str, Any], num_repeats: int) -> dict[str, Any]:
    """Repeat each tensor in ``obs`` along the batch axis."""
    if num_repeats < 1:
        raise ValueError(f"num_repeats must be >= 1, got {num_repeats}.")
    if num_repeats == 1:
        return obs

    repeated: dict[str, Any] = {}
    for key, value in obs.items():
        if not torch.is_tensor(value):
            repeated[key] = value
            continue
        expanded = value.unsqueeze(1).expand(
            value.shape[0], num_repeats, *value.shape[1:]
        )
        repeated[key] = expanded.reshape(value.shape[0] * num_repeats, *value.shape[1:])
    return repeated


def stack_action_candidates(candidates: list[torch.Tensor]) -> torch.Tensor:
    """Stack ``[B, A]`` or ``[B, K, A]`` candidate tensors to ``[B, K_total, A]``."""
    if not candidates:
        raise ValueError("candidates must be non-empty.")

    parts: list[torch.Tensor] = []
    for candidate in candidates:
        if candidate.dim() == 2:
            parts.append(candidate.unsqueeze(1))
        elif candidate.dim() == 3:
            parts.append(candidate)
        else:
            raise ValueError(
                "Each candidate must be [batch, action] or "
                f"[batch, num_samples, action], got {tuple(candidate.shape)}."
            )
    return torch.cat(parts, dim=1)


def build_qc_critic_candidates(
    *,
    actor_samples: torch.Tensor | None = None,
    ref_chunk: torch.Tensor | None = None,
    actor_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble critic-target candidates.

    Order is ``ref_chunk``, then ``actor_mean``, then residual ``actor_samples``.
    That keeps source indices stable for logging.
    """
    parts: list[torch.Tensor] = []
    if ref_chunk is not None:
        parts.append(ref_chunk)
    if actor_mean is not None:
        parts.append(actor_mean)
    if actor_samples is not None:
        parts.append(actor_samples)
    if not parts:
        raise ValueError(
            "Need at least one of actor_samples, ref_chunk, or actor_mean."
        )
    return stack_action_candidates(parts)


def select_best_of_n_actions(
    q_values: torch.Tensor,
    candidate_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the highest-Q candidate in each batch row.

    Args:
        q_values: ``[batch, num_candidates]`` scores.
        candidate_actions: ``[batch, num_candidates, action]`` chunks.

    Returns:
        ``(best_actions, indices)`` with shapes ``[batch, action]`` and
        ``[batch]``.
    """
    if q_values.dim() != 2:
        raise ValueError(f"q_values must be [batch, N], got {tuple(q_values.shape)}.")
    if candidate_actions.dim() != 3:
        raise ValueError(
            "candidate_actions must be [batch, N, action], got "
            f"{tuple(candidate_actions.shape)}."
        )
    if q_values.shape[:2] != candidate_actions.shape[:2]:
        raise ValueError(
            "q_values/candidate batch mismatch: "
            f"{tuple(q_values.shape)} vs {tuple(candidate_actions.shape)}."
        )

    indices = torch.argmax(q_values, dim=-1)
    batch = torch.arange(indices.shape[0], device=indices.device)
    return candidate_actions[batch, indices], indices
