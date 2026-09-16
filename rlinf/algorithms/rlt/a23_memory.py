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

"""A23: episodic memory as non-local value evidence for the critic only.

Frozen A23-v1 contract
----------------------
* Memory entry: ``e_i = (z_i, a_i, R_i, outcome)`` with MC return-to-go
  filled only after episode end (no bootstrap leakage).
* Retrieval is **action-conditioned**:
  ``I_t = TopK sim((z_t, a_t), (z_i, a_i))``
* Memory Q:
  ``Q_M(z, a) = sum_i w_i R_i`` with success weight 1 and failure weight
  ``beta`` (default 0.25).
* Memory only modifies the **critic bootstrap / target**. Actor loss stays
  exactly the original RLT objective.

Aligned with RLT Stage-2 critic::

    y = r_chunk + 1_{not done} * gamma^H * q_boot
    q_boot = (1 - alpha) * Q_bar(z', a') + alpha * Q_M(z', a')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rlinf.algorithms.rlt.a22_memory import (
    collapse_chunk_rewards,
    compute_monte_carlo_rtg,
)


@dataclass
class A23MemoryConfig:
    """Config for A23 episodic value memory."""

    enable: bool = False
    pos_capacity: int = 4096
    neg_capacity: int = 4096
    top_k: int = 16
    temperature: float = 0.07
    gamma: float = 0.99
    k_pre: int = 16
    k_post: int = 4
    # Mix weight on Q_M inside the RLT bootstrap term.
    alpha: float = 0.1
    # Relative weight of failure-bank evidence inside Q_M.
    failure_beta: float = 0.25
    z_dim: int = 2048
    action_dim: int = 80


def build_a23_memory_config(cfg: Any) -> A23MemoryConfig:
    """Build A23 config from ``algorithm.a23_memory``."""
    from omegaconf import OmegaConf

    block = OmegaConf.select(cfg, "algorithm.a23_memory", default=None)
    if block is None:
        return A23MemoryConfig(enable=False)
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
    gamma = float(
        OmegaConf.select(
            block,
            "gamma",
            default=OmegaConf.select(cfg, "algorithm.gamma", default=0.99),
        )
    )
    return A23MemoryConfig(
        enable=bool(OmegaConf.select(block, "enable", default=False)),
        pos_capacity=int(OmegaConf.select(block, "pos_capacity", default=4096)),
        neg_capacity=int(OmegaConf.select(block, "neg_capacity", default=4096)),
        top_k=int(OmegaConf.select(block, "top_k", default=16)),
        temperature=float(OmegaConf.select(block, "temperature", default=0.07)),
        gamma=gamma,
        k_pre=int(OmegaConf.select(block, "k_pre", default=16)),
        k_post=int(OmegaConf.select(block, "k_post", default=4)),
        alpha=float(OmegaConf.select(block, "alpha", default=0.1)),
        failure_beta=float(OmegaConf.select(block, "failure_beta", default=0.25)),
        z_dim=z_dim,
        action_dim=flat_action,
    )


class _OutcomeRingBank:
    """Fixed-capacity ring of ``(z, a, R, outcome)`` with outcome in {0,1}."""

    def __init__(self, capacity: int, z_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        self.z = torch.zeros(self.capacity, self.z_dim, dtype=torch.float32)
        self.a = torch.zeros(self.capacity, self.action_dim, dtype=torch.float32)
        self.R = torch.zeros(self.capacity, dtype=torch.float32)
        self.outcome = torch.zeros(self.capacity, dtype=torch.float32)
        self.size = 0
        self.ptr = 0

    def append(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        R: torch.Tensor,
        outcome: float,
    ) -> int:
        if z.numel() == 0:
            return 0
        z = z.detach().float().reshape(-1, self.z_dim).cpu()
        a = a.detach().float().reshape(-1, self.action_dim).cpu()
        R = R.detach().float().reshape(-1).cpu()
        n = z.shape[0]
        out_val = float(outcome)
        for i in range(n):
            self.z[self.ptr] = z[i]
            self.a[self.ptr] = a[i]
            self.R[self.ptr] = R[i]
            self.outcome[self.ptr] = out_val
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        return n

    def tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            return self.z[:0], self.a[:0], self.R[:0], self.outcome[:0]
        if self.size < self.capacity:
            return (
                self.z[: self.size],
                self.a[: self.size],
                self.R[: self.size],
                self.outcome[: self.size],
            )
        return self.z, self.a, self.R, self.outcome


class A23MemoryBank:
    """Episodic success/failure banks providing action-conditioned Q_M."""

    def __init__(self, config: A23MemoryConfig):
        self.config = config
        self.pos_bank = _OutcomeRingBank(
            config.pos_capacity, config.z_dim, config.action_dim
        )
        self.neg_bank = _OutcomeRingBank(
            config.neg_capacity, config.z_dim, config.action_dim
        )
        self._write_pos = 0
        self._write_neg = 0

    @property
    def memory_size(self) -> int:
        return int(self.pos_bank.size + self.neg_bank.size)

    def stats(self) -> dict[str, float]:
        return {
            "a23_memory/pos_size": float(self.pos_bank.size),
            "a23_memory/neg_size": float(self.neg_bank.size),
            "a23_memory/memory_size": float(self.memory_size),
            "a23_memory/write_pos_count": float(self._write_pos),
            "a23_memory/write_neg_count": float(self._write_neg),
        }

    def pop_write_metrics(self) -> dict[str, float]:
        out = self.stats()
        self._write_pos = 0
        self._write_neg = 0
        return out

    def _flatten_action(self, actions: torch.Tensor) -> torch.Tensor:
        a = actions.detach().float()
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
        if a.shape[-1] == self.config.action_dim:
            return a
        flat = torch.zeros(a.shape[0], self.config.action_dim, dtype=torch.float32)
        n = min(self.config.action_dim, a.shape[-1])
        flat[:, :n] = a[:, :n]
        return flat

    def commit_episode(
        self,
        *,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        r_seq: torch.Tensor,
        critical_mask: torch.Tensor | None,
        success: bool,
    ) -> int:
        """Commit E1∪E2 indices after episode end with MC-RTG filled."""
        z_seq = z_seq.detach().float().reshape(-1, self.config.z_dim)
        a_seq = self._flatten_action(a_seq)
        r_seq = collapse_chunk_rewards(r_seq, self.config.gamma)
        t = min(int(z_seq.shape[0]), int(a_seq.shape[0]), int(r_seq.shape[0]))
        if t == 0:
            return 0
        z_seq = z_seq[:t]
        a_seq = a_seq[:t]
        r_seq = r_seq[:t]

        rtg = compute_monte_carlo_rtg(r_seq, self.config.gamma)
        select = torch.zeros(t, dtype=torch.bool)
        if critical_mask is not None:
            crit = critical_mask.detach().reshape(-1).to(dtype=torch.bool)
            if crit.numel() == t:
                select |= crit
        t_end = t - 1
        lo = max(0, t_end - int(self.config.k_pre))
        hi = min(t, t_end + int(self.config.k_post) + 1)
        select[lo:hi] = True
        if not bool(select.any()):
            return 0

        z = z_seq[select]
        a = a_seq[select]
        R = rtg[select]
        outcome = 1.0 if success else 0.0
        if success:
            n = self.pos_bank.append(z, a, R, outcome=outcome)
            self._write_pos += n
        else:
            n = self.neg_bank.append(z, a, R, outcome=outcome)
            self._write_neg += n
        return n

    def gather_all(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunks_z: list[torch.Tensor] = []
        chunks_a: list[torch.Tensor] = []
        chunks_R: list[torch.Tensor] = []
        chunks_o: list[torch.Tensor] = []
        for bank in (self.pos_bank, self.neg_bank):
            z, a, R, outcome = bank.tensors()
            if z.numel() == 0:
                continue
            chunks_z.append(z.to(device=device))
            chunks_a.append(a.to(device=device))
            chunks_R.append(R.to(device=device))
            chunks_o.append(outcome.to(device=device))
        if not chunks_z:
            return (
                torch.zeros(0, self.config.z_dim, device=device),
                torch.zeros(0, self.config.action_dim, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, device=device),
            )
        return (
            torch.cat(chunks_z, dim=0),
            torch.cat(chunks_a, dim=0),
            torch.cat(chunks_R, dim=0),
            torch.cat(chunks_o, dim=0),
        )

    def estimate_q_m(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Action-conditioned episodic Q_M for a batch of ``(z, a)``."""
        z = z.detach().float()
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1)
        a = self._flatten_action(actions).to(device=z.device, dtype=z.dtype)
        bsz = z.shape[0]
        device = z.device
        metrics = {
            "a23_memory/retrieve_k": 0.0,
            "a23_memory/q_m_mean": 0.0,
            "a23_memory/used": 0.0,
        }
        if self.memory_size == 0:
            return torch.zeros(bsz, 1, device=device, dtype=z.dtype), metrics

        mem_z, mem_a, mem_R, mem_o = self.gather_all(device)
        k = min(int(self.config.top_k), int(mem_z.shape[0]))
        if k <= 0:
            return torch.zeros(bsz, 1, device=device, dtype=z.dtype), metrics

        query = F.normalize(torch.cat([z, a], dim=-1), dim=-1)
        key = F.normalize(torch.cat([mem_z, mem_a], dim=-1), dim=-1)
        scores = torch.matmul(query, key.transpose(0, 1)) / max(
            self.config.temperature, 1e-6
        )
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        weights = torch.softmax(top_scores, dim=-1)
        gathered_R = mem_R[top_idx]
        gathered_o = mem_o[top_idx]
        # Success weight 1, failure weight beta.
        outcome_scale = torch.where(
            gathered_o > 0.5,
            torch.ones_like(gathered_o),
            torch.full_like(gathered_o, float(self.config.failure_beta)),
        )
        q_m = (weights * outcome_scale * gathered_R).sum(dim=-1, keepdim=True)
        metrics["a23_memory/retrieve_k"] = float(k)
        metrics["a23_memory/q_m_mean"] = float(q_m.mean().item())
        metrics["a23_memory/used"] = 1.0
        return q_m.to(dtype=z.dtype), metrics


def mix_a23_bootstrap_q(
    *,
    q_next: torch.Tensor,
    q_m: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mix TD bootstrap Q with episodic Q_M: ``(1-α)Q + α Q_M``."""
    alpha = float(max(0.0, min(1.0, alpha)))
    if q_m.numel() == 0:
        return q_next, {"a23_memory/alpha": alpha, "a23_memory/td_gap": 0.0}
    q_m = q_m.to(device=q_next.device, dtype=q_next.dtype)
    if q_m.shape != q_next.shape:
        q_m = q_m.reshape_as(q_next)
    mixed = (1.0 - alpha) * q_next + alpha * q_m
    gap = float((q_m - q_next).mean().item())
    return mixed, {"a23_memory/alpha": alpha, "a23_memory/td_gap": gap}
