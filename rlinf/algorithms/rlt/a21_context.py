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

"""A21: in-episode temporal context for RLT Stage-2.

C1: the actor and critic both observe the last ``K_ctx`` steps of ``(z, a)``
from the *current* episode (zero-padded at the start). No cross-episode bank
and no memory policy loss — this isolates whether extra context helps Stage-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


RLT_CONTEXT_KEYS = ("ctx_z", "ctx_a", "ctx_mask")


@dataclass
class A21ContextConfig:
    """Config for A21 in-episode context."""

    enable: bool = False
    ctx_len: int = 8
    z_dim: int = 2048
    action_dim: int = 80  # flat action chunk


def build_a21_context_config(cfg: Any) -> A21ContextConfig:
    """Build A21 config from ``algorithm.a21_context``."""
    from omegaconf import OmegaConf

    block = OmegaConf.select(cfg, "algorithm.a21_context", default=None)
    if block is None:
        return A21ContextConfig(enable=False)
    action_dim = int(OmegaConf.select(cfg, "actor.model.action_dim", default=8))
    num_chunks = int(OmegaConf.select(cfg, "actor.model.num_action_chunks", default=10))
    z_dim = int(
        OmegaConf.select(
            cfg,
            "actor.model.z_dim",
            default=OmegaConf.select(block, "z_dim", default=2048),
        )
    )
    flat_action = int(
        OmegaConf.select(block, "action_dim", default=action_dim * num_chunks)
    )
    return A21ContextConfig(
        enable=bool(OmegaConf.select(block, "enable", default=False)),
        ctx_len=int(OmegaConf.select(block, "ctx_len", default=8)),
        z_dim=z_dim,
        action_dim=flat_action,
    )


def empty_context_batch(
    batch_size: int,
    config: A21ContextConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Return zero-padded empty context tensors."""
    k = int(config.ctx_len)
    return {
        "ctx_z": torch.zeros(
            batch_size, k, config.z_dim, device=device, dtype=dtype
        ),
        "ctx_a": torch.zeros(
            batch_size, k, config.action_dim, device=device, dtype=dtype
        ),
        "ctx_mask": torch.zeros(batch_size, k, device=device, dtype=dtype),
    }


def flatten_context(obs: dict[str, Any]) -> torch.Tensor:
    """Flatten ``ctx_z/ctx_a`` (masked) into a single feature vector."""
    ctx_z = obs["ctx_z"]
    ctx_a = obs["ctx_a"]
    ctx_mask = obs.get("ctx_mask")
    if ctx_z.ndim == 2:
        # Already flat; keep for robustness.
        return ctx_z
    mask = (
        ctx_mask.to(dtype=ctx_z.dtype).unsqueeze(-1)
        if ctx_mask is not None
        else torch.ones(
            ctx_z.shape[0], ctx_z.shape[1], 1, device=ctx_z.device, dtype=ctx_z.dtype
        )
    )
    z_flat = (ctx_z * mask).reshape(ctx_z.shape[0], -1)
    a_flat = (ctx_a * mask).reshape(ctx_a.shape[0], -1)
    return torch.cat([z_flat, a_flat], dim=-1)


class A21ContextBuffer:
    """Per-env ring of recent ``(z, a)`` pairs for A21."""

    def __init__(self, config: A21ContextConfig):
        self.config = config
        self._num_envs = 0
        self._z: torch.Tensor | None = None
        self._a: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        self._len: torch.Tensor | None = None
        self._ptr: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enable)

    def _ensure(self, num_envs: int, device: torch.device) -> None:
        if self._num_envs == num_envs and self._z is not None:
            if self._z.device == device:
                return
            self._z = self._z.to(device=device)
            self._a = self._a.to(device=device)
            self._mask = self._mask.to(device=device)
            self._len = self._len.to(device=device)
            self._ptr = self._ptr.to(device=device)
            return
        self._num_envs = int(num_envs)
        k = self.config.ctx_len
        self._z = torch.zeros(
            self._num_envs, k, self.config.z_dim, device=device, dtype=torch.float32
        )
        self._a = torch.zeros(
            self._num_envs,
            k,
            self.config.action_dim,
            device=device,
            dtype=torch.float32,
        )
        self._mask = torch.zeros(
            self._num_envs, k, device=device, dtype=torch.float32
        )
        self._len = torch.zeros(self._num_envs, dtype=torch.long, device=device)
        self._ptr = torch.zeros(self._num_envs, dtype=torch.long, device=device)

    def get(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return empty_context_batch(batch_size, self.config, device)
        self._ensure(batch_size, device)
        assert self._z is not None and self._a is not None and self._mask is not None
        # Return chronologically ordered windows.
        ordered_z = torch.zeros_like(self._z)
        ordered_a = torch.zeros_like(self._a)
        ordered_m = torch.zeros_like(self._mask)
        for env_id in range(self._num_envs):
            length = int(self._len[env_id].item())
            if length <= 0:
                continue
            if length < self.config.ctx_len:
                ordered_z[env_id, :length] = self._z[env_id, :length]
                ordered_a[env_id, :length] = self._a[env_id, :length]
                ordered_m[env_id, :length] = 1.0
            else:
                ptr = int(self._ptr[env_id].item())
                ordered_z[env_id] = torch.cat(
                    [self._z[env_id, ptr:], self._z[env_id, :ptr]], dim=0
                )
                ordered_a[env_id] = torch.cat(
                    [self._a[env_id, ptr:], self._a[env_id, :ptr]], dim=0
                )
                ordered_m[env_id] = 1.0
        return {"ctx_z": ordered_z, "ctx_a": ordered_a, "ctx_mask": ordered_m}

    def push(
        self,
        *,
        z_rl: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Append ``(z, a)`` and return the *updated* context (for next_obs)."""
        z = z_rl.detach().float()
        a = actions.detach().float()
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1)
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
        if a.shape[-1] != self.config.action_dim:
            flat = torch.zeros(a.shape[0], self.config.action_dim, dtype=a.dtype)
            n = min(self.config.action_dim, a.shape[-1])
            flat[:, :n] = a[:, :n]
            a = flat
        self._ensure(z.shape[0], z.device)
        assert self._z is not None and self._a is not None
        assert self._len is not None and self._ptr is not None
        for env_id in range(z.shape[0]):
            slot = int(self._ptr[env_id].item())
            self._z[env_id, slot] = z[env_id].to(device=self._z.device)
            self._a[env_id, slot] = a[env_id].to(device=self._a.device)
            self._ptr[env_id] = (slot + 1) % self.config.ctx_len
            self._len[env_id] = min(
                int(self._len[env_id].item()) + 1, self.config.ctx_len
            )
        return self.get(z.shape[0], self._z.device)

    def reset(self, done_mask: torch.Tensor | None) -> None:
        if not self.enabled or self._len is None or done_mask is None:
            return
        mask = done_mask.to(device=self._len.device)
        if mask.ndim > 1:
            mask = mask.reshape(mask.shape[0], -1).any(dim=-1)
        mask = mask.to(dtype=torch.bool).reshape(-1)
        if mask.numel() != self._num_envs:
            return
        self._len[mask] = 0
        self._ptr[mask] = 0
        if self._z is not None:
            self._z[mask] = 0
            self._a[mask] = 0
            self._mask[mask] = 0
