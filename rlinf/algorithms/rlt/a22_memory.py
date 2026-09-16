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

"""A22: episodic memory bank with MC return-to-go and advantage-weighted L_M.

Write protocol (E3 = E1 + E2)
-----------------------------
* Buffer transitions during an episode (critical steps for E1; always keep full
  episode tensors so E2 can cut a terminal window).
* On episode end only: compute Monte-Carlo return-to-go

      R_i = sum_{k=i}^{T} gamma^{k-i} r_k

  then commit the union of E1 (critical indices) and E2
  (``[T-K_pre, T+K_post]`` clipped) entries ``(z, a, r, R)`` into success /
  failure banks. No bootstrap, no future leakage.

Policy loss
-----------
Retrieve from both banks by ``sim(z_t, z_i)``. With detached critic baseline:

      A_i^M = R_i - stopgrad(Q_theta(z_t, a ~ pi))

Only ``A_i^M > 0`` enters:

      L_M = - sum_i w_i A_i^M log pi_theta(a_i | z_t, ctx)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal


@dataclass
class A22MemoryConfig:
    """Config for A22 memory bank + L_M."""

    enable: bool = False
    pos_capacity: int = 4096
    neg_capacity: int = 4096
    top_k: int = 16
    temperature: float = 0.07
    gamma: float = 0.99
    k_pre: int = 16
    k_post: int = 4
    lambda_m: float = 0.1
    adv_threshold: float = 0.0
    z_dim: int = 2048
    action_dim: int = 80


def build_a22_memory_config(cfg: Any) -> A22MemoryConfig:
    """Build A22 config from ``algorithm.a22_memory``."""
    from omegaconf import OmegaConf

    block = OmegaConf.select(cfg, "algorithm.a22_memory", default=None)
    if block is None:
        return A22MemoryConfig(enable=False)
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
    return A22MemoryConfig(
        enable=bool(OmegaConf.select(block, "enable", default=False)),
        pos_capacity=int(OmegaConf.select(block, "pos_capacity", default=4096)),
        neg_capacity=int(OmegaConf.select(block, "neg_capacity", default=4096)),
        top_k=int(OmegaConf.select(block, "top_k", default=16)),
        temperature=float(OmegaConf.select(block, "temperature", default=0.07)),
        gamma=gamma,
        k_pre=int(OmegaConf.select(block, "k_pre", default=16)),
        k_post=int(OmegaConf.select(block, "k_post", default=4)),
        lambda_m=float(OmegaConf.select(block, "lambda_m", default=0.1)),
        adv_threshold=float(OmegaConf.select(block, "adv_threshold", default=0.0)),
        z_dim=z_dim,
        action_dim=flat_action,
    )


def compute_monte_carlo_rtg(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute ``R_i = sum_{k=i}^T gamma^{k-i} r_k`` for a 1D reward sequence."""
    rewards = rewards.detach().float().reshape(-1)
    t = rewards.numel()
    rtg = torch.zeros(t, dtype=torch.float32)
    running = 0.0
    for i in range(t - 1, -1, -1):
        running = float(rewards[i].item()) + float(gamma) * running
        rtg[i] = running
    return rtg


def collapse_chunk_rewards(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Collapse per-substep chunk rewards ``[T, C]`` into one scalar per chunk.

    Matches RLT critic ``_discounted_chunk_rewards``::

        r_t = sum_{c=0}^{C-1} gamma^c * r_{t,c}

    Already-1D rewards are returned unchanged.
    """
    rewards = rewards.detach().float()
    if rewards.ndim <= 1:
        return rewards.reshape(-1)
    flat = rewards.reshape(rewards.shape[0], -1)
    chunk_len = int(flat.shape[-1])
    discounts = torch.pow(
        torch.as_tensor(gamma, dtype=flat.dtype),
        torch.arange(chunk_len, dtype=flat.dtype),
    )
    return torch.sum(flat * discounts, dim=-1)


class _EntryRingBank:
    """Fixed-capacity ring of ``(z, a, r, R)``."""

    def __init__(self, capacity: int, z_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        self.z = torch.zeros(self.capacity, self.z_dim, dtype=torch.float32)
        self.a = torch.zeros(self.capacity, self.action_dim, dtype=torch.float32)
        self.r = torch.zeros(self.capacity, dtype=torch.float32)
        self.R = torch.zeros(self.capacity, dtype=torch.float32)
        self.size = 0
        self.ptr = 0

    def append(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        R: torch.Tensor,
    ) -> int:
        if z.numel() == 0:
            return 0
        z = z.detach().float().reshape(-1, self.z_dim).cpu()
        a = a.detach().float().reshape(-1, self.action_dim).cpu()
        r = r.detach().float().reshape(-1).cpu()
        R = R.detach().float().reshape(-1).cpu()
        n = z.shape[0]
        for i in range(n):
            self.z[self.ptr] = z[i]
            self.a[self.ptr] = a[i]
            self.r[self.ptr] = r[i]
            self.R[self.ptr] = R[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        return n

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            return (
                self.z[:0],
                self.a[:0],
                self.r[:0],
                self.R[:0],
            )
        if self.size < self.capacity:
            return (
                self.z[: self.size],
                self.a[: self.size],
                self.r[: self.size],
                self.R[: self.size],
            )
        return self.z, self.a, self.r, self.R


class A22MemoryBank:
    """Success/failure memory banks for A22."""

    def __init__(self, config: A22MemoryConfig):
        self.config = config
        self.pos_bank = _EntryRingBank(
            config.pos_capacity, config.z_dim, config.action_dim
        )
        self.neg_bank = _EntryRingBank(
            config.neg_capacity, config.z_dim, config.action_dim
        )
        self._write_pos = 0
        self._write_neg = 0

    @property
    def memory_size(self) -> int:
        return int(self.pos_bank.size + self.neg_bank.size)

    def stats(self) -> dict[str, float]:
        return {
            "a22_memory/pos_size": float(self.pos_bank.size),
            "a22_memory/neg_size": float(self.neg_bank.size),
            "a22_memory/memory_size": float(self.memory_size),
            "a22_memory/write_pos_count": float(self._write_pos),
            "a22_memory/write_neg_count": float(self._write_neg),
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

        # E1: critical-phase steps.
        if critical_mask is not None:
            crit = critical_mask.detach().reshape(-1).to(dtype=torch.bool)
            if crit.numel() == t:
                select |= crit

        # E2: terminal window [T-K_pre, T+K_post] clipped to [0, T).
        t_end = t - 1
        lo = max(0, t_end - int(self.config.k_pre))
        hi = min(t, t_end + int(self.config.k_post) + 1)
        select[lo:hi] = True

        if not bool(select.any()):
            return 0

        z = z_seq[select]
        a = a_seq[select]
        r = r_seq[select]
        R = rtg[select]
        if success:
            n = self.pos_bank.append(z, a, r, R)
            self._write_pos += n
        else:
            n = self.neg_bank.append(z, a, r, R)
            self._write_neg += n
        return n

    def gather_all(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(mem_z, mem_a, mem_R)`` from both banks."""
        chunks_z: list[torch.Tensor] = []
        chunks_a: list[torch.Tensor] = []
        chunks_R: list[torch.Tensor] = []
        for bank in (self.pos_bank, self.neg_bank):
            z, a, _, R = bank.tensors()
            if z.numel() == 0:
                continue
            chunks_z.append(z.to(device=device))
            chunks_a.append(a.to(device=device))
            chunks_R.append(R.to(device=device))
        if not chunks_z:
            return (
                torch.zeros(0, self.config.z_dim, device=device),
                torch.zeros(0, self.config.action_dim, device=device),
                torch.zeros(0, device=device),
            )
        return (
            torch.cat(chunks_z, dim=0),
            torch.cat(chunks_a, dim=0),
            torch.cat(chunks_R, dim=0),
        )

    def retrieve(
        self, z_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-k retrieve. Returns ``(mem_z, mem_a, mem_R, weights)`` shaped [B,K,*]."""
        z = z_t.detach().float()
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1)
        device = z.device
        mem_z, mem_a, mem_R = self.gather_all(device)
        bsz = z.shape[0]
        k = min(int(self.config.top_k), int(mem_z.shape[0]))
        if k <= 0:
            return (
                torch.zeros(bsz, 0, self.config.z_dim, device=device),
                torch.zeros(bsz, 0, self.config.action_dim, device=device),
                torch.zeros(bsz, 0, device=device),
                torch.zeros(bsz, 0, device=device),
            )
        q = F.normalize(z, dim=-1)
        key = F.normalize(mem_z, dim=-1)
        scores = torch.matmul(q, key.transpose(0, 1)) / max(
            self.config.temperature, 1e-6
        )
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        weights = torch.softmax(top_scores, dim=-1)
        return (
            mem_z[top_idx],
            mem_a[top_idx],
            mem_R[top_idx],
            weights,
        )


def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def gaussian_tanh_log_prob(
    *,
    action_mean: torch.Tensor,
    action_std: float,
    actions_tanh: torch.Tensor,
) -> torch.Tensor:
    """Log-prob of tanh-squashed actions under diagonal Gaussian + tanh."""
    pre = _atanh(actions_tanh)
    dist = Normal(action_mean, action_std)
    log_prob = dist.log_prob(pre)
    # Tanh correction: log|det da/du| = sum log(1 - tanh(u)^2)
    log_prob = log_prob - torch.log(1.0 - actions_tanh.pow(2) + 1e-6)
    return log_prob.sum(dim=-1)


def compute_a22_memory_loss(
    *,
    policy_model: Any,
    obs: dict[str, Any],
    bank: A22MemoryBank,
    q_baseline: torch.Tensor,
    fixed_std: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Advantage-weighted memory BC with detached Q baseline.

    ``q_baseline`` must already be detached and broadcastable to ``[B]``.
    """
    cfg = bank.config
    model = policy_model.module if hasattr(policy_model, "module") else policy_model
    device = next(model.parameters()).device
    z_t = obs["z_rl"].to(device=device)
    if z_t.ndim > 2:
        z_t = z_t.reshape(z_t.shape[0], -1)
    bsz = z_t.shape[0]

    mem_z, mem_a, mem_R, weights = bank.retrieve(z_t)
    metrics = {
        "a22_memory/retrieve_k": float(mem_a.shape[1]),
        "a22_memory/positive_frac": 0.0,
        "a22_memory/adv_mean": 0.0,
        "a22_memory/loss": 0.0,
    }
    if mem_a.shape[1] == 0:
        return z_t.new_zeros(()), metrics

    baseline = q_baseline.detach().float().reshape(bsz, 1).to(device=device)
    adv = mem_R.to(device=device) - baseline
    pos = adv > float(cfg.adv_threshold)
    metrics["a22_memory/positive_frac"] = float(pos.float().mean().item())
    if not bool(pos.any()):
        return z_t.new_zeros(()), metrics

    # Policy mean for current obs (grads into policy; baseline already detached).
    actor_state = model._actor_state(obs)
    action_mean = model.actor_mean(model.backbone(actor_state))

    k = mem_a.shape[1]
    mean_exp = action_mean.unsqueeze(1).expand(-1, k, -1)
    flat_mean = mean_exp.reshape(bsz * k, -1)
    flat_act = mem_a.reshape(bsz * k, -1).to(device=device, dtype=action_mean.dtype)
    logp = gaussian_tanh_log_prob(
        action_mean=flat_mean,
        action_std=float(fixed_std),
        actions_tanh=flat_act,
    ).reshape(bsz, k)

    weighted = weights.to(device=device) * adv.clamp(min=0.0) * logp
    weighted = torch.where(pos, weighted, torch.zeros_like(weighted))
    per_sample = weighted.sum(dim=-1)
    loss = -per_sample.mean()
    metrics["a22_memory/adv_mean"] = (
        float(adv[pos].mean().item()) if bool(pos.any()) else 0.0
    )
    metrics["a22_memory/loss"] = float(loss.detach().item())
    return loss, metrics
