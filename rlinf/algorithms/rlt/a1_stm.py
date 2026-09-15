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

"""A1 Short-Term Memory for RLT Stage-2 STM ablation.

A1 is the first STM verification drop:
- rule-based write (critical-phase success / failure banks)
- non-trainable cosine retrieval + fixed gated residual fusion
- enhances ``z_rl`` before the Stage-2 policy head

This module is intentionally non-trainable so PegInsertion can isolate whether
explicit short-term memory improves sample efficiency over plain RLT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class A1STMConfig:
    """Config for the A1 STM verifier."""

    enable: bool = False
    window_size: int = 64
    pos_capacity: int = 2048
    neg_capacity: int = 2048
    top_k: int = 16
    temperature: float = 0.07
    gate: float = 0.5
    only_critical: bool = True
    fail_tail_steps: int = 16
    write_on_eval: bool = False
    z_dim: int = 2048


def build_a1_stm_config(cfg: Any) -> A1STMConfig:
    """Build A1 STM config from ``algorithm.a1_stm``."""
    from omegaconf import OmegaConf

    block = OmegaConf.select(cfg, "algorithm.a1_stm", default=None)
    if block is None:
        return A1STMConfig(enable=False)
    z_dim = int(
        OmegaConf.select(
            cfg,
            "actor.model.z_dim",
            default=OmegaConf.select(block, "z_dim", default=2048),
        )
    )
    return A1STMConfig(
        enable=bool(OmegaConf.select(block, "enable", default=False)),
        window_size=int(OmegaConf.select(block, "window_size", default=64)),
        pos_capacity=int(OmegaConf.select(block, "pos_capacity", default=2048)),
        neg_capacity=int(OmegaConf.select(block, "neg_capacity", default=2048)),
        top_k=int(OmegaConf.select(block, "top_k", default=16)),
        temperature=float(OmegaConf.select(block, "temperature", default=0.07)),
        gate=float(OmegaConf.select(block, "gate", default=0.5)),
        only_critical=bool(OmegaConf.select(block, "only_critical", default=True)),
        fail_tail_steps=int(OmegaConf.select(block, "fail_tail_steps", default=16)),
        write_on_eval=bool(OmegaConf.select(block, "write_on_eval", default=False)),
        z_dim=z_dim,
    )


def extract_a1_stm_success(
    *,
    rewards: torch.Tensor | None,
    success: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve per-env success flags for A1 write consolidation."""
    if success is not None:
        flag = success.to(device=device)
        if flag.ndim > 1:
            flag = flag.reshape(flag.shape[0], -1).any(dim=-1)
        return flag.to(dtype=torch.bool).reshape(batch_size)

    if rewards is not None:
        reward = rewards.to(device=device)
        if reward.ndim > 1:
            reward = reward.reshape(reward.shape[0], -1).sum(dim=-1)
        return (reward > 0).to(dtype=torch.bool).reshape(batch_size)

    return torch.zeros(batch_size, dtype=torch.bool, device=device)


class _RingBank:
    """Fixed-capacity ring buffer of ``z`` vectors."""

    def __init__(self, capacity: int, z_dim: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self.capacity = int(capacity)
        self.z_dim = int(z_dim)
        self.z = torch.zeros(self.capacity, self.z_dim, dtype=torch.float32)
        self.size = 0
        self.ptr = 0

    def append(self, entries: torch.Tensor) -> None:
        if entries.numel() == 0:
            return
        entries = entries.detach().float().reshape(-1, self.z_dim).cpu()
        for row in entries:
            self.z[self.ptr] = row
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def tensor(self) -> torch.Tensor:
        if self.size == 0:
            return self.z[:0]
        if self.size < self.capacity:
            return self.z[: self.size]
        return self.z


class A1ShortTermMemory:
    """Rule-based STM used for the A1 RLT PegInsertion verification."""

    def __init__(self, config: A1STMConfig):
        self.config = config
        self.pos_bank = _RingBank(config.pos_capacity, config.z_dim)
        self.neg_bank = _RingBank(config.neg_capacity, config.z_dim)
        self._num_envs = 0
        self._window_z: torch.Tensor | None = None
        self._window_len: torch.Tensor | None = None
        self._window_ptr: torch.Tensor | None = None
        self._pending_z: torch.Tensor | None = None
        self._pending_critical: torch.Tensor | None = None
        self._pending_valid: torch.Tensor | None = None
        self._metric_count = 0
        self._metric_sums: dict[str, float] = {}
        self._write_pos_count = 0
        self._write_neg_count = 0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enable)

    def stats(self) -> dict[str, float]:
        return {
            "a1_stm/pos_size": float(self.pos_bank.size),
            "a1_stm/neg_size": float(self.neg_bank.size),
            "a1_stm/num_envs": float(self._num_envs),
        }

    def _record_step_metrics(self, metrics: dict[str, float]) -> None:
        """Accumulate per-enhance scalars until the next ``pop_logged_metrics``."""
        self._metric_count += 1
        for key, value in metrics.items():
            self._metric_sums[key] = self._metric_sums.get(key, 0.0) + float(value)

    def pop_logged_metrics(self) -> dict[str, float]:
        """Return window averages + current gauges, then reset step accumulators.

        Bank sizes are gauges (latest snapshot). Retrieve stats are means over
        all ``enhance`` calls since the previous pop. Write counts are totals
        since the previous pop.
        """
        out = self.stats()
        if self._metric_count > 0:
            count = float(self._metric_count)
            for key, total in self._metric_sums.items():
                # Prefer averaged step metrics over a one-shot gauge snapshot.
                out[key] = total / count
        out["a1_stm/enhance_count"] = float(self._metric_count)
        out["a1_stm/write_pos_count"] = float(self._write_pos_count)
        out["a1_stm/write_neg_count"] = float(self._write_neg_count)
        self._metric_count = 0
        self._metric_sums = {}
        self._write_pos_count = 0
        self._write_neg_count = 0
        return out

    def _ensure_num_envs(self, num_envs: int, device: torch.device) -> None:
        if self._num_envs == num_envs and self._window_z is not None:
            return
        self._num_envs = int(num_envs)
        self._window_z = torch.zeros(
            self._num_envs,
            self.config.window_size,
            self.config.z_dim,
            dtype=torch.float32,
            device=device,
        )
        self._window_len = torch.zeros(
            self._num_envs, dtype=torch.long, device=device
        )
        self._window_ptr = torch.zeros(
            self._num_envs, dtype=torch.long, device=device
        )
        self._pending_z = torch.zeros(
            self._num_envs, self.config.z_dim, dtype=torch.float32, device=device
        )
        self._pending_critical = torch.zeros(
            self._num_envs, dtype=torch.bool, device=device
        )
        self._pending_valid = torch.zeros(
            self._num_envs, dtype=torch.bool, device=device
        )

    def _as_bool_mask(
        self,
        mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        out = mask.to(device=device)
        if out.ndim > 1:
            out = out.reshape(out.shape[0], -1).any(dim=-1)
        return out.to(dtype=torch.bool).reshape(batch_size)

    def finalize_pending(
        self,
        *,
        dones: torch.Tensor | None,
        rewards: torch.Tensor | None,
        success: torch.Tensor | None,
        allow_write: bool,
    ) -> None:
        """Commit the previous-step pending entries, then consolidate finished envs."""
        if not self.enabled or self._pending_valid is None or self._window_z is None:
            return
        if not bool(self._pending_valid.any()):
            # Still honor done consolidation if pending was empty.
            if dones is not None and allow_write:
                done_mask = self._as_bool_mask(
                    dones, self._num_envs, self._window_z.device
                )
                success_mask = extract_a1_stm_success(
                    rewards=rewards,
                    success=success,
                    batch_size=self._num_envs,
                    device=self._window_z.device,
                )
                self._consolidate_done(done_mask, success_mask)
            return

        device = self._window_z.device
        pending_idx = torch.where(self._pending_valid)[0]
        write_critical = self._pending_critical[pending_idx]
        if self.config.only_critical:
            write_idx = pending_idx[write_critical]
        else:
            write_idx = pending_idx

        if allow_write and write_idx.numel() > 0:
            assert self._window_ptr is not None
            for env_id in write_idx.tolist():
                length = int(self._window_len[env_id].item())
                slot = int(self._window_ptr[env_id].item())
                self._window_z[env_id, slot] = self._pending_z[env_id]
                self._window_ptr[env_id] = (slot + 1) % self.config.window_size
                self._window_len[env_id] = min(length + 1, self.config.window_size)

        self._pending_valid.zero_()

        if dones is None or not allow_write:
            return
        done_mask = self._as_bool_mask(dones, self._num_envs, device)
        success_mask = extract_a1_stm_success(
            rewards=rewards,
            success=success,
            batch_size=self._num_envs,
            device=device,
        )
        self._consolidate_done(done_mask, success_mask)

    def _episode_entries(self, env_id: int) -> torch.Tensor:
        assert self._window_z is not None
        assert self._window_len is not None
        assert self._window_ptr is not None
        length = int(self._window_len[env_id].item())
        if length <= 0:
            return self._window_z.new_zeros((0, self.config.z_dim))
        if length < self.config.window_size:
            return self._window_z[env_id, :length].detach().cpu()
        ptr = int(self._window_ptr[env_id].item())
        ordered = torch.cat(
            [
                self._window_z[env_id, ptr:],
                self._window_z[env_id, :ptr],
            ],
            dim=0,
        )
        return ordered.detach().cpu()

    def _consolidate_done(
        self, done_mask: torch.Tensor, success_mask: torch.Tensor
    ) -> None:
        assert self._window_len is not None
        assert self._window_ptr is not None
        done_ids = torch.where(done_mask)[0].tolist()
        for env_id in done_ids:
            entries = self._episode_entries(env_id)
            if entries.numel() > 0:
                if bool(success_mask[env_id].item()):
                    self.pos_bank.append(entries)
                    self._write_pos_count += 1
                else:
                    tail = max(1, int(self.config.fail_tail_steps))
                    self.neg_bank.append(entries[-tail:])
                    self._write_neg_count += 1
            self._window_len[env_id] = 0
            self._window_ptr[env_id] = 0

    def set_pending(
        self,
        *,
        z_rl: torch.Tensor,
        critical_mask: torch.Tensor | None,
    ) -> None:
        """Cache the current step so the next env feedback can write it."""
        if not self.enabled:
            return
        z = z_rl.detach().float()
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1)
        self._ensure_num_envs(z.shape[0], z.device)
        assert self._pending_z is not None
        assert self._pending_critical is not None
        assert self._pending_valid is not None
        self._pending_z.copy_(z.to(device=self._pending_z.device))
        self._pending_critical.copy_(
            self._as_bool_mask(critical_mask, z.shape[0], self._pending_z.device)
        )
        self._pending_valid.fill_(True)

    def enhance(self, z_rl: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Retrieve STM context and return gated residual ``z'``."""
        if not self.enabled:
            return z_rl, {}

        z = z_rl.float()
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1)
        self._ensure_num_envs(z.shape[0], z.device)
        assert self._window_z is not None and self._window_len is not None

        memory = self._gather_memory(z.device)
        metrics = self.stats()
        metrics["a1_stm/memory_size"] = float(memory.shape[0])
        if memory.numel() == 0:
            metrics["a1_stm/retrieve_used"] = 0.0
            self._record_step_metrics(metrics)
            return z_rl, metrics

        # Cosine attention over the shared memory bank.
        q = F.normalize(z, dim=-1)
        k = F.normalize(memory.to(device=z.device, dtype=z.dtype), dim=-1)
        scores = torch.matmul(q, k.transpose(0, 1)) / max(self.config.temperature, 1e-6)
        top_k = min(int(self.config.top_k), k.shape[0])
        top_scores, top_idx = torch.topk(scores, k=top_k, dim=-1)
        weights = torch.softmax(top_scores, dim=-1)
        gathered = k[top_idx]  # [B, top_k, D]
        retrieved = torch.sum(gathered * weights.unsqueeze(-1), dim=1)
        gate = float(self.config.gate)
        enhanced = z + gate * retrieved
        metrics["a1_stm/retrieve_used"] = 1.0
        metrics["a1_stm/top_score_mean"] = float(top_scores[:, 0].mean().item())
        self._record_step_metrics(metrics)
        return enhanced.to(dtype=z_rl.dtype), metrics

    def _gather_memory(self, device: torch.device) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for env_id in range(self._num_envs):
            entries = self._episode_entries(env_id)
            if entries.numel() > 0:
                chunks.append(entries.to(device=device))
        pos = self.pos_bank.tensor()
        neg = self.neg_bank.tensor()
        if pos.numel() > 0:
            chunks.append(pos.to(device=device))
        if neg.numel() > 0:
            chunks.append(neg.to(device=device))
        if not chunks:
            return torch.zeros(0, self.config.z_dim, device=device)
        return torch.cat(chunks, dim=0)
