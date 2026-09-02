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

"""Sliding-window chunk history for long-horizon RLT Stage 2."""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "RLTHistoryBuffer",
    "build_history_summary",
    "extract_rlt_lh_phase",
    "extract_rlt_lh_subtask_success",
    "history_summary_dim",
]


def _flatten_last(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() <= 2:
        return tensor.reshape(tensor.shape[0], -1)
    return tensor.reshape(tensor.shape[0], -1)


def _scalar(value: torch.Tensor | None, batch_size: int) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, 1, dtype=torch.float32)
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.numel() == 1:
        return value.reshape(1, 1).expand(batch_size, 1).contiguous()
    value = value.reshape(batch_size, -1)
    if value.shape[1] == 1:
        return value
    return value[:, -1:]


def history_summary_dim(
    *,
    z_dim: int,
    proprio_dim: int,
    action_dim: int,
    num_action_chunks: int,
) -> int:
    """Return the canonical summary dimension used by :func:`build_history_summary`."""
    return (
        int(z_dim)
        + int(proprio_dim)
        + int(action_dim) * int(num_action_chunks)
        + 3
    )


def extract_rlt_lh_phase(
    *,
    env_type: str,
    env_infos: dict[str, Any] | None,
    rlt_switch_flags: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor:
    """Return the current long-horizon phase index as ``[batch, 1]``.

    ManiSkill RLT exposes an explicit ``rlt_switch_flags`` binary phase.
    CALVIN exposes the current subtask index through its episode metrics, which
    is a much stronger long-horizon phase signal than the raw task-level done.
    """
    if rlt_switch_flags is not None:
        flags = torch.as_tensor(rlt_switch_flags, dtype=torch.float32).reshape(
            batch_size, -1
        )
        return flags[:, -1:]

    env_type = str(env_type)
    if env_type == "calvin" and isinstance(env_infos, dict):
        episode = env_infos.get("episode")
        if isinstance(episode, dict):
            value = episode.get("avg_len", episode.get("current_task_idx"))
            if value is not None:
                return torch.as_tensor(value, dtype=torch.float32).reshape(
                    batch_size, -1
                )[:, -1:]

    return torch.zeros(batch_size, 1, dtype=torch.float32)


def extract_rlt_lh_subtask_success(
    *,
    env_type: str,
    env_infos: dict[str, Any] | None,
    dones: torch.Tensor | None,
    batch_size: int,
    phase_idx: torch.Tensor | None = None,
    prev_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-chunk subtask success as ``[batch, 1]``.

    Prefer an explicit environment success signal when available, then a phase
    increase for environments whose only observable progress is the phase index
    (CALVIN), then finally the chunk-level ``dones`` as a conservative fallback.
    """
    if isinstance(env_infos, dict):
        episode = env_infos.get("episode")
        if isinstance(episode, dict):
            for key in ("success_event", "success"):
                value = episode.get(key, env_infos.get(key))
                if value is not None:
                    return _scalar(torch.as_tensor(value), batch_size)

    if phase_idx is not None and prev_phase is not None:
        phase = torch.as_tensor(phase_idx, dtype=torch.float32).reshape(batch_size, -1)
        previous = torch.as_tensor(prev_phase, dtype=torch.float32).reshape(
            batch_size, -1
        )
        return (phase[:, -1:] > previous[:, -1:]).to(dtype=torch.float32)

    return _scalar(torch.as_tensor(dones) if dones is not None else None, batch_size)


def build_history_summary(
    *,
    z_rl: torch.Tensor,
    proprio: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    phase_idx: torch.Tensor | None = None,
    subtask_success: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a per-chunk history summary vector.

    Layout: ``[z_rl, proprio, previous_action, reward, phase_idx, subtask_success]``.
    ``action`` is the action executed by the previous chunk.
    """
    batch_size = z_rl.shape[0]
    z = _flatten_last(z_rl).to(dtype=torch.float32)
    proprio_flat = _flatten_last(proprio).to(dtype=torch.float32)
    action_flat = _flatten_last(action).to(dtype=torch.float32)
    reward_flat = _scalar(reward, batch_size).to(device=z.device)
    phase_flat = _scalar(phase_idx, batch_size).to(device=z.device)
    success_flat = _scalar(subtask_success, batch_size).to(device=z.device)
    return torch.cat(
        [z, proprio_flat, action_flat, reward_flat, phase_flat, success_flat],
        dim=-1,
    )


class RLTHistoryBuffer:
    """Maintain a per-env sliding window of chunk summaries.

    The buffer stores raw summaries, not encoded features, so the Stage-2
    history encoder can be trained through replay with the current weights.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        window: int,
        summary_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_envs = int(num_envs)
        self.window = int(window)
        self.summary_dim = int(summary_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        if self.window <= 0:
            raise ValueError(f"window must be positive, got {window!r}.")
        self._buffer = torch.zeros(
            self.num_envs,
            self.window,
            self.summary_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        if env_idx is None:
            self._buffer.zero_()
            return
        self._buffer[env_idx] = 0.0

    def push(
        self,
        summary: torch.Tensor,
        done_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append ``summary`` and return the updated window tensor.

        For environments marked done, the history is cleared after this chunk so
        the next episode starts with an empty context.
        """
        summary = torch.as_tensor(summary, dtype=self.dtype, device=self.device)
        if summary.dim() == 1:
            summary = summary.unsqueeze(0)
        if summary.shape[0] != self.num_envs:
            raise ValueError(
                f"summary batch mismatch: expected {self.num_envs}, "
                f"got {summary.shape[0]}."
            )
        if summary.shape[-1] != self.summary_dim:
            raise ValueError(
                f"summary dimension mismatch: expected {self.summary_dim}, "
                f"got {summary.shape[-1]}."
            )

        self._buffer[:, :-1] = self._buffer[:, 1:]
        self._buffer[:, -1] = summary

        if done_mask is not None:
            done_mask = torch.as_tensor(done_mask, dtype=torch.bool, device=self.device)
            done_mask = done_mask.reshape(-1)
            if done_mask.numel() > self.num_envs:
                done_mask = done_mask[: self.num_envs]
            elif done_mask.numel() < self.num_envs:
                padded = torch.zeros(
                    self.num_envs - done_mask.numel(),
                    dtype=torch.bool,
                    device=self.device,
                )
                done_mask = torch.cat([done_mask, padded])
            done_mask = done_mask.reshape(self.num_envs)
            if done_mask.any():
                self._buffer[done_mask] = 0.0

        return self._buffer.clone()

    @property
    def buffer(self) -> torch.Tensor:
        return self._buffer

    def get(self, env_idx: Any | None = None) -> torch.Tensor:
        if env_idx is None:
            return self._buffer
        return self._buffer[env_idx]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_envs={self.num_envs}, "
            f"window={self.window}, summary_dim={self.summary_dim})"
        )
