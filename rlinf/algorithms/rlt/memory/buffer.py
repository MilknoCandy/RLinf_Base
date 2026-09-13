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

"""Fixed-window short-term memory buffer for RLT Step 2A.

The buffer is intentionally deterministic and causal: it stores completed
experiences ``e_t = (z_t, a_t, r_t)`` and reads only the recent ``K`` entries
that precede the current decision. A pending ``(z_t, a_t)`` is completed with
the reward of the previous step on the next rollout call, which implements the
``read -> act -> write`` lifecycle without leaking the current action or reward.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import torch


def memory_entry_dim(
    memory_cfg: Any,
    *,
    z_dim: int,
    action_dim: int,
) -> int:
    """Return the flattened dimension of one STM entry."""
    if memory_cfg is None:
        return int(z_dim) + int(action_dim) + 1
    entry_cfg = memory_cfg.get("entry", {}) or {}
    dim = 0
    if entry_cfg.get("token", True):
        dim += int(z_dim)
    if entry_cfg.get("action", True):
        dim += int(action_dim)
    if entry_cfg.get("reward", True):
        dim += 1
    if dim <= 0:
        raise ValueError("memory.entry enables no component; entry_dim would be 0.")
    return int(dim)


def _to_batch_vector(
    value: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, device=device, dtype=dtype)
    value = torch.as_tensor(value, device=device)
    if value.numel() == 0:
        return torch.zeros(batch_size, device=device, dtype=dtype)
    if value.numel() == 1:
        return value.reshape(1).to(device=device, dtype=dtype).expand(batch_size)
    return value.reshape(-1).to(device=device, dtype=dtype)[:batch_size]


class STMBuffer:
    """Per-env fixed-window episodic short-term memory.

    Each completed entry is a concatenation of the enabled components
    ``(z_t, a_t, r_t)``. The reader supports three ablations: ``recent`` keeps
    chronological order, ``shuffled`` destroys temporal order within the recent
    window, and ``random`` samples from the whole buffer. All modes pad the
    older side with zeros and return a boolean mask for the GRU encoder.
    """

    def __init__(
        self,
        *,
        window_size: int,
        z_dim: int,
        action_dim: int,
        entry_cfg: dict[str, Any] | None = None,
        retrieval: str = "recent",
        seed: int | None = None,
    ):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        self.window_size = int(window_size)
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)

        entry_cfg = dict(entry_cfg or {})
        self.use_token = bool(entry_cfg.get("token", True))
        self.use_action = bool(entry_cfg.get("action", True))
        self.use_reward = bool(entry_cfg.get("reward", True))
        self.entry_dim = memory_entry_dim(
            {"entry": entry_cfg},
            z_dim=self.z_dim,
            action_dim=self.action_dim,
        )

        if retrieval not in {"recent", "shuffled", "random"}:
            raise ValueError(
                f"retrieval must be one of 'recent', 'shuffled', 'random', "
                f"got {retrieval}."
            )
        self.retrieval = retrieval
        self._rng = random.Random(seed)

        self._num_envs = 0
        self._fifo: list[deque[torch.Tensor]] = []
        self._pending_z: list[torch.Tensor | None] = []
        self._pending_action: list[torch.Tensor | None] = []

    def _ensure_envs(self, batch_size: int) -> None:
        if self._num_envs == batch_size:
            return
        self._num_envs = batch_size
        self._fifo = [deque(maxlen=None) for _ in range(batch_size)]
        self._pending_z = [None] * batch_size
        self._pending_action = [None] * batch_size

    def reset(self) -> None:
        """Clear all per-env buffers and pending experiences."""
        for env_idx in range(self._num_envs):
            self._fifo[env_idx].clear()
            self._pending_z[env_idx] = None
            self._pending_action[env_idx] = None

    def _build_entry(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.use_token:
            parts.append(z.reshape(-1)[: self.z_dim].to(torch.float32))
        if self.use_action:
            parts.append(action.reshape(-1)[: self.action_dim].to(torch.float32))
        if self.use_reward:
            parts.append(
                torch.as_tensor([float(reward)], dtype=torch.float32)
            )
        return torch.cat(parts, dim=0).contiguous()

    def _select_entries(self, entries: list[torch.Tensor]) -> list[torch.Tensor]:
        window = self.window_size
        if self.retrieval == "recent":
            return list(entries[-window:])
        if self.retrieval == "shuffled":
            selected = list(entries[-window:])
            self._rng.shuffle(selected)
            return selected
        if self.retrieval == "random":
            return self._rng.sample(entries, min(window, len(entries)))
        raise ValueError(f"Unsupported retrieval mode: {self.retrieval}.")

    def complete_and_read(
        self,
        z_t: torch.Tensor,
        last_reward: torch.Tensor | None = None,
        last_done: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete the previous pending entry, then read recent history.

        ``last_reward`` / ``last_done`` describe the result of the previous
        action chunk. On episode boundaries the corresponding env buffer is
        cleared so the next episode starts with empty memory.
        """
        z_t = torch.as_tensor(z_t).detach().float()
        batch_size = int(z_t.shape[0])
        self._ensure_envs(batch_size)

        reward = _to_batch_vector(
            last_reward, batch_size, torch.device("cpu"), torch.float32
        )
        done = _to_batch_vector(
            last_done, batch_size, torch.device("cpu"), torch.bool
        ).bool()

        for env_idx in range(batch_size):
            if bool(done[env_idx].item()):
                self._fifo[env_idx].clear()
                self._pending_z[env_idx] = None
                self._pending_action[env_idx] = None
                continue
            pending_z = self._pending_z[env_idx]
            if pending_z is not None:
                self._fifo[env_idx].append(
                    self._build_entry(
                        pending_z,
                        self._pending_action[env_idx],
                        reward[env_idx],
                    )
                )
            self._pending_z[env_idx] = None
            self._pending_action[env_idx] = None

        return self.read(batch_size, device=z_t.device)

    def read(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(history, mask)`` with shapes ``(B, K, D)`` and ``(B, K)``."""
        self._ensure_envs(batch_size)
        history = torch.zeros(
            (batch_size, self.window_size, self.entry_dim),
            dtype=torch.float32,
        )
        mask = torch.zeros((batch_size, self.window_size), dtype=torch.bool)

        for env_idx in range(batch_size):
            entries = list(self._fifo[env_idx])
            if not entries:
                continue
            selected = self._select_entries(entries)
            start = self.window_size - len(selected)
            for offset, entry in enumerate(selected):
                history[env_idx, start + offset] = entry
                mask[env_idx, start + offset] = True

        return history.to(device=device, dtype=dtype), mask.to(device=device)

    def remember_current(
        self,
        z_t: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """Store the current state/action as the pending entry for the next step."""
        z_t = torch.as_tensor(z_t).detach().float()
        batch_size = int(z_t.shape[0])
        self._ensure_envs(batch_size)

        z_flat = z_t.reshape(batch_size, -1)[:, : self.z_dim].contiguous().cpu()
        actions = torch.as_tensor(actions).detach().float()
        action_flat = (
            actions.reshape(batch_size, -1)[:, : self.action_dim].contiguous().cpu()
        )

        for env_idx in range(batch_size):
            self._pending_z[env_idx] = z_flat[env_idx].clone()
            self._pending_action[env_idx] = action_flat[env_idx].clone()
