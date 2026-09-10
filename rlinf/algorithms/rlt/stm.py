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

"""FIFO short-term memory for the RLT Stage-2 policy."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch

from rlinf.algorithms.rlt.transition import RLT_STM_KEY


def _to_batch_vector(value: Any, batch_size: int, device: torch.device, dtype) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, device=device, dtype=dtype)
    value = torch.as_tensor(value, device=device)
    if value.numel() == 0:
        return torch.zeros(batch_size, device=device, dtype=dtype)
    if value.numel() == 1:
        return value.reshape(1).to(device=device, dtype=dtype).expand(batch_size)
    return value.reshape(-1).to(device=device, dtype=dtype)[:batch_size]


class RLTSTMFIFO:
    """Per-env FIFO memory for the RLT actor.

    The memory unit is ``(z_t, a_t, r_t, z_{t+1})``. Phase 1 uses a fixed,
    deterministic reader that pools recent ``z`` and reward scalars and
    concatenates the result to the current RL token. Advantage is not available
    causally at rollout time and is intentionally omitted from this Phase 1
    feature.
    """

    def __init__(self, *, capacity: int, z_dim: int, action_dim: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}.")
        self.capacity = int(capacity)
        self.z_dim = int(z_dim)
        self.flat_action_dim = int(action_dim)
        self.memory_dim = self.z_dim + 2

        self._num_envs = 0
        self._fifo: list[deque[dict[str, Any]]] = []
        self._pending_z: list[torch.Tensor | None] = []
        self._pending_action: list[torch.Tensor | None] = []
        self._pending_actor_switch: list[bool] = []

    def _ensure_envs(self, batch_size: int) -> None:
        if self._num_envs == batch_size:
            return
        self._num_envs = batch_size
        self._fifo = [deque(maxlen=self.capacity) for _ in range(batch_size)]
        self._pending_z = [None] * batch_size
        self._pending_action = [None] * batch_size
        self._pending_actor_switch = [False] * batch_size

    def reset(self) -> None:
        """Clear all per-env memories and pending experiences."""
        for env_idx in range(self._num_envs):
            self._fifo[env_idx].clear()
            self._pending_z[env_idx] = None
            self._pending_action[env_idx] = None
            self._pending_actor_switch[env_idx] = False

    @staticmethod
    def _flat_cpu(tensor: Any, batch_size: int, dim: int) -> torch.Tensor:
        tensor = torch.as_tensor(tensor).detach().float().cpu()
        tensor = tensor.reshape(batch_size, -1)
        if tensor.shape[1] < dim:
            raise ValueError(
                f"Expected at least {dim} columns, got {tensor.shape[1]}."
            )
        return tensor[:, :dim].contiguous()

    def complete_and_retrieve(
        self,
        z_t: torch.Tensor,
        last_reward: torch.Tensor | None = None,
        last_done: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Complete the previous experience with ``z_t`` and return ``m_t``."""
        z_t = torch.as_tensor(z_t).detach().float()
        batch_size = int(z_t.shape[0])
        z_flat = z_t.reshape(batch_size, -1)
        if z_flat.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected z_rl dim {self.z_dim}, got {z_flat.shape[1]}."
            )

        self._ensure_envs(batch_size)
        device = z_t.device
        reward = _to_batch_vector(
            last_reward, batch_size, torch.device("cpu"), torch.float32
        )
        done = _to_batch_vector(
            last_done, batch_size, torch.device("cpu"), torch.bool
        ).bool()

        memory = z_flat.new_zeros((batch_size, self.memory_dim))
        for env_idx in range(batch_size):
            if bool(done[env_idx].item()):
                self._fifo[env_idx].clear()
                self._pending_z[env_idx] = None
                self._pending_action[env_idx] = None
                self._pending_actor_switch[env_idx] = False
                continue

            pending_z = self._pending_z[env_idx]
            if pending_z is not None and self._pending_actor_switch[env_idx]:
                self._fifo[env_idx].append(
                    {
                        "z": pending_z.to(device=device, dtype=z_flat.dtype),
                        "action": self._pending_action[env_idx].to(
                            device=device, dtype=z_flat.dtype
                        ),
                        "reward": float(reward[env_idx].item()),
                        "next_z": z_flat[env_idx].clone(),
                    }
                )

            self._pending_z[env_idx] = None
            self._pending_action[env_idx] = None
            self._pending_actor_switch[env_idx] = False

            buffer = self._fifo[env_idx]
            if not buffer:
                continue
            z_mean = torch.stack([entry["z"] for entry in buffer], dim=0).mean(
                dim=0
            )
            reward_values = torch.stack(
                [
                    torch.as_tensor(
                        entry["reward"], device=device, dtype=z_flat.dtype
                    )
                    for entry in buffer
                ],
                dim=0,
            )
            memory_vector = torch.cat(
                [
                    z_mean,
                    reward_values.mean().reshape(1),
                    reward_values[-1].reshape(1),
                ],
                dim=0,
            )
            memory[env_idx] = memory_vector

        return memory

    def remember_current(
        self,
        z_t: torch.Tensor,
        actions: torch.Tensor,
        actor_switch: torch.Tensor | None,
    ) -> None:
        """Remember the current action/state as the next pending experience."""
        z_t = torch.as_tensor(z_t).detach().float()
        batch_size = int(z_t.shape[0])
        self._ensure_envs(batch_size)

        z_flat = self._flat_cpu(z_t, batch_size, self.z_dim)
        action_flat = self._flat_cpu(actions, batch_size, self.flat_action_dim)
        switch = _to_batch_vector(
            actor_switch, batch_size, torch.device("cpu"), torch.bool
        ).bool()

        for env_idx in range(batch_size):
            if bool(switch[env_idx].item()):
                self._pending_z[env_idx] = z_flat[env_idx].clone()
                self._pending_action[env_idx] = action_flat[env_idx].clone()
                self._pending_actor_switch[env_idx] = True
            else:
                self._pending_z[env_idx] = None
                self._pending_action[env_idx] = None
                self._pending_actor_switch[env_idx] = False

    @property
    def stm_key(self) -> str:
        return RLT_STM_KEY
