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

"""A2 learnable Short-Term Memory for RLT Stage-2.

Design intent
-------------
RL post-training should *learn how to use* historical experience, not apply a
fixed residual on frozen tokens.

A2 therefore separates:

* **Learnable encoder** (``A2STMEncoder``): query/key/value + gate. These
  parameters live on the Stage-2 policy and are optimized by the SAC/RLT-AC
  objectives when ``z'`` is recomputed during actor updates.
* **Experience bank** (``A2MemoryBank``): stores ``(z, a)`` entries. Banks are
  *not* model parameters (not synced by weight sync); each rollout/actor worker
  keeps its own bank.

Metrics to monitor for benefit
------------------------------
Primary task metrics (compare enable on/off):

* ``eval/success`` (or env success rate) vs ``rlt/update_step``
* sample efficiency: updates / wall-time to reach a target success

A2-specific process metrics (logged under ``a2_stm/``):

* ``a2_stm/pos_size``, ``a2_stm/neg_size``: bank growth (should rise after
  critical successes/failures)
* ``a2_stm/retrieve_used``: fraction of steps with non-empty retrieval
* ``a2_stm/gate_mean``: mean gate magnitude (learning to use memory → often
  moves away from init; collapse to 0 means memory ignored)
* ``a2_stm/top_score_mean``: retrieval peak similarity
* ``a2_stm/enhance_delta_norm``: ``||z'-z||`` (should be >0 when gate/retrieval
  fire; near-0 means no effective enhancement)

If task metrics improve while ``gate_mean`` / ``enhance_delta_norm`` stay near
zero, gains are probably not from memory. If banks stay empty, fix write path
before claiming A2 failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class A2STMConfig:
    """Config for learnable A2 STM."""

    enable: bool = False
    window_size: int = 64
    pos_capacity: int = 2048
    neg_capacity: int = 2048
    top_k: int = 16
    temperature: float = 0.07
    only_critical: bool = True
    fail_tail_steps: int = 16
    write_on_eval: bool = False
    share_train_bank_on_eval: bool = True
    z_dim: int = 2048
    action_dim: int = 80  # flat action chunk, e.g. 10*8
    hidden_dim: int = 256
    gate_init_bias: float = -2.0  # sigmoid(-2)~0.12, start conservative


def build_a2_stm_config(cfg: Any) -> A2STMConfig:
    """Build A2 STM config from ``algorithm.a2_stm``."""
    from omegaconf import OmegaConf

    block = OmegaConf.select(cfg, "algorithm.a2_stm", default=None)
    if block is None:
        return A2STMConfig(enable=False)
    z_dim = int(
        OmegaConf.select(
            cfg,
            "actor.model.z_dim",
            default=OmegaConf.select(block, "z_dim", default=2048),
        )
    )
    action_dim = int(OmegaConf.select(cfg, "actor.model.action_dim", default=8))
    num_chunks = int(OmegaConf.select(cfg, "actor.model.num_action_chunks", default=10))
    flat_action = int(
        OmegaConf.select(block, "action_dim", default=action_dim * num_chunks)
    )
    return A2STMConfig(
        enable=bool(OmegaConf.select(block, "enable", default=False)),
        window_size=int(OmegaConf.select(block, "window_size", default=64)),
        pos_capacity=int(OmegaConf.select(block, "pos_capacity", default=2048)),
        neg_capacity=int(OmegaConf.select(block, "neg_capacity", default=2048)),
        top_k=int(OmegaConf.select(block, "top_k", default=16)),
        temperature=float(OmegaConf.select(block, "temperature", default=0.07)),
        only_critical=bool(OmegaConf.select(block, "only_critical", default=True)),
        fail_tail_steps=int(OmegaConf.select(block, "fail_tail_steps", default=16)),
        write_on_eval=bool(OmegaConf.select(block, "write_on_eval", default=False)),
        share_train_bank_on_eval=bool(
            OmegaConf.select(block, "share_train_bank_on_eval", default=True)
        ),
        z_dim=z_dim,
        action_dim=flat_action,
        hidden_dim=int(OmegaConf.select(block, "hidden_dim", default=256)),
        gate_init_bias=float(OmegaConf.select(block, "gate_init_bias", default=-2.0)),
    )


def extract_a2_stm_success(
    *,
    rewards: torch.Tensor | None,
    success: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve per-env success flags for A2 write consolidation."""
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


class _PairRingBank:
    """Fixed-capacity ring buffer of ``(z, a)`` pairs."""

    def __init__(self, capacity: int, z_dim: int, action_dim: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self.capacity = int(capacity)
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        self.z = torch.zeros(self.capacity, self.z_dim, dtype=torch.float32)
        self.a = torch.zeros(self.capacity, self.action_dim, dtype=torch.float32)
        self.size = 0
        self.ptr = 0

    def append(self, z_entries: torch.Tensor, a_entries: torch.Tensor) -> None:
        if z_entries.numel() == 0:
            return
        z_entries = z_entries.detach().float().reshape(-1, self.z_dim).cpu()
        a_entries = a_entries.detach().float().reshape(-1, self.action_dim).cpu()
        if a_entries.shape[0] != z_entries.shape[0]:
            raise ValueError(
                f"z/a batch mismatch: {z_entries.shape[0]} vs {a_entries.shape[0]}"
            )
        for z_row, a_row in zip(z_entries, a_entries):
            self.z[self.ptr] = z_row
            self.a[self.ptr] = a_row
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0:
            return self.z[:0], self.a[:0]
        if self.size < self.capacity:
            return self.z[: self.size], self.a[: self.size]
        return self.z, self.a


class A2MemoryBank:
    """Non-parameter experience bank used by A2 retrieval."""

    def __init__(self, config: A2STMConfig):
        self.config = config
        self.pos_bank = _PairRingBank(
            config.pos_capacity, config.z_dim, config.action_dim
        )
        self.neg_bank = _PairRingBank(
            config.neg_capacity, config.z_dim, config.action_dim
        )
        self._num_envs = 0
        self._window_z: torch.Tensor | None = None
        self._window_a: torch.Tensor | None = None
        self._window_len: torch.Tensor | None = None
        self._window_ptr: torch.Tensor | None = None
        self._pending_z: torch.Tensor | None = None
        self._pending_a: torch.Tensor | None = None
        self._pending_critical: torch.Tensor | None = None
        self._pending_valid: torch.Tensor | None = None
        self._metric_count = 0
        self._metric_sums: dict[str, float] = {}
        self._write_pos_count = 0
        self._write_neg_count = 0

    def stats(self) -> dict[str, float]:
        return {
            "a2_stm/pos_size": float(self.pos_bank.size),
            "a2_stm/neg_size": float(self.neg_bank.size),
            "a2_stm/num_envs": float(self._num_envs),
            "a2_stm/memory_size": float(self.memory_size),
        }

    @property
    def memory_size(self) -> int:
        return int(self.pos_bank.size + self.neg_bank.size)

    def record_step_metrics(self, metrics: dict[str, float]) -> None:
        self._metric_count += 1
        for key, value in metrics.items():
            self._metric_sums[key] = self._metric_sums.get(key, 0.0) + float(value)

    def pop_logged_metrics(self) -> dict[str, float]:
        out = self.stats()
        if self._metric_count > 0:
            count = float(self._metric_count)
            for key, total in self._metric_sums.items():
                out[key] = total / count
        out["a2_stm/enhance_count"] = float(self._metric_count)
        out["a2_stm/write_pos_count"] = float(self._write_pos_count)
        out["a2_stm/write_neg_count"] = float(self._write_neg_count)
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
        self._window_a = torch.zeros(
            self._num_envs,
            self.config.window_size,
            self.config.action_dim,
            dtype=torch.float32,
            device=device,
        )
        self._window_len = torch.zeros(self._num_envs, dtype=torch.long, device=device)
        self._window_ptr = torch.zeros(self._num_envs, dtype=torch.long, device=device)
        self._pending_z = torch.zeros(
            self._num_envs, self.config.z_dim, dtype=torch.float32, device=device
        )
        self._pending_a = torch.zeros(
            self._num_envs, self.config.action_dim, dtype=torch.float32, device=device
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

    def _batch_size_of(self, mask: torch.Tensor | None, fallback: int) -> int:
        if mask is None:
            return int(fallback)
        return int(mask.shape[0])

    def set_pending(
        self,
        *,
        z_rl: torch.Tensor,
        actions: torch.Tensor,
        critical_mask: torch.Tensor | None,
    ) -> None:
        z = z_rl.detach().float()
        a = actions.detach().float()
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1)
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
        if a.shape[-1] != self.config.action_dim:
            # Pad/truncate to configured flat action dim.
            flat = torch.zeros(a.shape[0], self.config.action_dim, dtype=a.dtype)
            n = min(self.config.action_dim, a.shape[-1])
            flat[:, :n] = a[:, :n]
            a = flat
        self._ensure_num_envs(z.shape[0], z.device)
        assert self._pending_z is not None
        assert self._pending_a is not None
        assert self._pending_critical is not None
        assert self._pending_valid is not None
        self._pending_z.copy_(z.to(device=self._pending_z.device))
        self._pending_a.copy_(a.to(device=self._pending_a.device))
        self._pending_critical.copy_(
            self._as_bool_mask(critical_mask, z.shape[0], self._pending_z.device)
        )
        self._pending_valid.fill_(True)

    def finalize_pending(
        self,
        *,
        dones: torch.Tensor | None,
        rewards: torch.Tensor | None,
        success: torch.Tensor | None,
        allow_write: bool,
    ) -> None:
        if self._pending_valid is None or self._window_z is None:
            return
        feedback_batch = self._batch_size_of(dones, self._num_envs)
        if feedback_batch != self._num_envs:
            self._pending_valid.zero_()
            return

        if not bool(self._pending_valid.any()):
            if dones is not None and allow_write:
                done_mask = self._as_bool_mask(
                    dones, self._num_envs, self._window_z.device
                )
                success_mask = extract_a2_stm_success(
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
        write_idx = (
            pending_idx[write_critical]
            if self.config.only_critical
            else pending_idx
        )

        if allow_write and write_idx.numel() > 0:
            assert self._window_ptr is not None and self._window_a is not None
            for env_id in write_idx.tolist():
                length = int(self._window_len[env_id].item())
                slot = int(self._window_ptr[env_id].item())
                self._window_z[env_id, slot] = self._pending_z[env_id]
                self._window_a[env_id, slot] = self._pending_a[env_id]
                self._window_ptr[env_id] = (slot + 1) % self.config.window_size
                self._window_len[env_id] = min(length + 1, self.config.window_size)

        self._pending_valid.zero_()
        if dones is None or not allow_write:
            return
        done_mask = self._as_bool_mask(dones, self._num_envs, device)
        success_mask = extract_a2_stm_success(
            rewards=rewards,
            success=success,
            batch_size=self._num_envs,
            device=device,
        )
        self._consolidate_done(done_mask, success_mask)

    def _episode_entries(self, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._window_z is not None
        assert self._window_a is not None
        assert self._window_len is not None
        assert self._window_ptr is not None
        length = int(self._window_len[env_id].item())
        if length <= 0:
            empty_z = self._window_z.new_zeros((0, self.config.z_dim))
            empty_a = self._window_a.new_zeros((0, self.config.action_dim))
            return empty_z, empty_a
        if length < self.config.window_size:
            return (
                self._window_z[env_id, :length].detach().cpu(),
                self._window_a[env_id, :length].detach().cpu(),
            )
        ptr = int(self._window_ptr[env_id].item())
        z_ord = torch.cat(
            [self._window_z[env_id, ptr:], self._window_z[env_id, :ptr]], dim=0
        )
        a_ord = torch.cat(
            [self._window_a[env_id, ptr:], self._window_a[env_id, :ptr]], dim=0
        )
        return z_ord.detach().cpu(), a_ord.detach().cpu()

    def _consolidate_done(
        self, done_mask: torch.Tensor, success_mask: torch.Tensor
    ) -> None:
        assert self._window_len is not None and self._window_ptr is not None
        for env_id in torch.where(done_mask)[0].tolist():
            z_entries, a_entries = self._episode_entries(env_id)
            if z_entries.numel() > 0:
                if bool(success_mask[env_id].item()):
                    self.pos_bank.append(z_entries, a_entries)
                    self._write_pos_count += 1
                else:
                    tail = max(1, int(self.config.fail_tail_steps))
                    self.neg_bank.append(z_entries[-tail:], a_entries[-tail:])
                    self._write_neg_count += 1
            self._window_len[env_id] = 0
            self._window_ptr[env_id] = 0

    def remember_batch(
        self,
        *,
        z_rl: torch.Tensor,
        actions: torch.Tensor,
        success: torch.Tensor | None,
    ) -> None:
        """Direct bank write used by the actor when ingesting transitions."""
        z = z_rl.detach().float().reshape(-1, self.config.z_dim).cpu()
        a = actions.detach().float().reshape(z.shape[0], -1).cpu()
        if a.shape[-1] != self.config.action_dim:
            flat = torch.zeros(a.shape[0], self.config.action_dim)
            n = min(self.config.action_dim, a.shape[-1])
            flat[:, :n] = a[:, :n]
            a = flat
        if success is None:
            self.pos_bank.append(z, a)
            self._write_pos_count += int(z.shape[0])
            return
        flag = success.detach().reshape(-1).to(dtype=torch.bool).cpu()
        if flag.numel() != z.shape[0]:
            flag = flag[:1].repeat(z.shape[0])
        if bool(flag.any()):
            self.pos_bank.append(z[flag], a[flag])
            self._write_pos_count += int(flag.sum().item())
        if bool((~flag).any()):
            self.neg_bank.append(z[~flag], a[~flag])
            self._write_neg_count += int((~flag).sum().item())

    def gather_memory(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_chunks: list[torch.Tensor] = []
        a_chunks: list[torch.Tensor] = []
        pos_z, pos_a = self.pos_bank.tensors()
        neg_z, neg_a = self.neg_bank.tensors()
        if pos_z.numel() > 0:
            z_chunks.append(pos_z.to(device=device))
            a_chunks.append(pos_a.to(device=device))
        if neg_z.numel() > 0:
            z_chunks.append(neg_z.to(device=device))
            a_chunks.append(neg_a.to(device=device))
        if not z_chunks:
            return (
                torch.zeros(0, self.config.z_dim, device=device),
                torch.zeros(0, self.config.action_dim, device=device),
            )
        return torch.cat(z_chunks, dim=0), torch.cat(a_chunks, dim=0)

    def snapshot_for_share(self) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU tensors for read-only eval sharing."""
        return self.gather_memory(torch.device("cpu"))


class A2STMEncoder(nn.Module):
    """Learnable retrieve + gate producing ``z' = z + g ⊙ m``."""

    def __init__(self, config: A2STMConfig):
        super().__init__()
        self.config = config
        z_dim = config.z_dim
        hidden = config.hidden_dim
        self.query = nn.Linear(z_dim, hidden)
        self.key = nn.Linear(z_dim, hidden)
        self.value = nn.Linear(z_dim + config.action_dim, z_dim)
        self.gate = nn.Sequential(
            nn.Linear(z_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )
        nn.init.constant_(self.gate[-1].bias, float(config.gate_init_bias))

    def enhance(
        self,
        z_rl: torch.Tensor,
        memory_z: torch.Tensor,
        memory_a: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        z = z_rl.float()
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1)
        metrics = {
            "a2_stm/memory_size": float(memory_z.shape[0]),
            "a2_stm/retrieve_used": 0.0,
            "a2_stm/gate_mean": 0.0,
            "a2_stm/top_score_mean": 0.0,
            "a2_stm/enhance_delta_norm": 0.0,
        }
        if memory_z.numel() == 0:
            return z_rl, metrics

        mem_z = memory_z.to(device=z.device, dtype=z.dtype)
        mem_a = memory_a.to(device=z.device, dtype=z.dtype)
        q = F.normalize(self.query(z), dim=-1)
        k = F.normalize(self.key(mem_z), dim=-1)
        scores = torch.matmul(q, k.transpose(0, 1)) / max(
            self.config.temperature, 1e-6
        )
        top_k = min(int(self.config.top_k), k.shape[0])
        top_scores, top_idx = torch.topk(scores, k=top_k, dim=-1)
        weights = torch.softmax(top_scores, dim=-1)
        gathered_z = mem_z[top_idx]
        gathered_a = mem_a[top_idx]
        value_in = torch.cat([gathered_z, gathered_a], dim=-1)
        values = self.value(value_in)
        retrieved = torch.sum(values * weights.unsqueeze(-1), dim=1)
        gate = torch.sigmoid(self.gate(torch.cat([z, retrieved], dim=-1)))
        enhanced = z + gate * retrieved
        metrics["a2_stm/retrieve_used"] = 1.0
        metrics["a2_stm/gate_mean"] = float(gate.detach().mean().item())
        metrics["a2_stm/top_score_mean"] = float(top_scores[:, 0].detach().mean().item())
        metrics["a2_stm/enhance_delta_norm"] = float(
            (enhanced - z).detach().norm(dim=-1).mean().item()
        )
        return enhanced.to(dtype=z_rl.dtype), metrics
