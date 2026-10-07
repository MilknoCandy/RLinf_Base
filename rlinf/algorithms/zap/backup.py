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

"""ZAP path backup: metric-gated n-step plus neighbor Q borrow.

A path is a sequence of atoms $$(z_k, a_k, r_k, done_k)$$. The critic still
evaluates $$Q(z, a)$$ on a single atom. Continuation uses $$(z, a)$$ kernels;
truncation bootstrap optionally mixes neighbor $$Q$$ values.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from rlinf.algorithms.rlt.n_step import discounted_chunk_reward
from rlinf.algorithms.zap.kernels import (
    continuation_factor,
    cosine_kernel_z,
    rbf_kernel_a,
)


def accumulate_product_and_truncate(
    continuations: torch.Tensor,
    *,
    truncate_eps: float,
) -> tuple[int, torch.Tensor]:
    """Return how many *extra* atoms after the query to include, and $$C_k$$.

    ``continuations`` is $$c_1,\\ldots,c_T$$ for atoms after the query.
    Truncation cuts *before* the first atom whose cumulative product is
    below ``truncate_eps``.
    """
    if truncate_eps <= 0 or truncate_eps > 1:
        raise ValueError(f"truncate_eps must be in (0, 1], got {truncate_eps}.")
    c = torch.as_tensor(continuations, dtype=torch.float32).reshape(-1)
    if c.numel() == 0:
        return 0, c
    products = torch.cumprod(c, dim=0)
    below = products < float(truncate_eps)
    if bool(below.any()):
        first = int(below.nonzero(as_tuple=False)[0].item())
        return first, products
    return int(c.numel()), products


def path_nstep_return(
    rewards: torch.Tensor,
    *,
    gamma: float,
    horizons: torch.Tensor | None = None,
    num_steps: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Discounted sum of the first ``num_steps`` atom rewards.

    Each ``rewards[k]`` should already fold within-atom control-step discounts.
    ``horizons[k]`` is the control-step length of atom ``k`` and advances the
    between-atom discount.
    """
    r = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
    if num_steps is None:
        num_steps = int(r.numel())
    if num_steps < 0:
        raise ValueError(f"num_steps must be >= 0, got {num_steps}.")
    num_steps = min(int(num_steps), int(r.numel()))
    if horizons is None:
        horizons = torch.ones(r.numel(), dtype=torch.float32)
    else:
        horizons = torch.as_tensor(horizons, dtype=torch.float32).reshape(-1)
        if horizons.numel() < r.numel():
            raise ValueError("horizons must cover every reward atom.")
    total = torch.zeros((), dtype=torch.float32)
    discount = 1.0
    for k in range(num_steps):
        total = total + discount * r[k]
        discount *= float(gamma) ** float(max(horizons[k].item(), 1.0))
    return total, discount


def neighbor_weights(
    z_query: torch.Tensor,
    a_pi_query: torch.Tensor,
    neighbor_z: torch.Tensor,
    neighbor_a: torch.Tensor,
    *,
    tau_z: float,
    tau_a: float,
    top_k: int | None = None,
    uniform_entropy_ratio: float = 0.95,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Softmax weights $$w_j \\propto k_z k_a$$ over neighbor rows.

    Returns an all-zero weight vector when the z-kernel is nearly uniform so
    the caller falls back to the local bootstrap Q.
    """
    z_q = z_query.reshape(1, -1)
    a_q = a_pi_query.reshape(1, -1)
    z_n = neighbor_z.reshape(neighbor_z.shape[0], -1)
    a_n = neighbor_a.reshape(neighbor_a.shape[0], -1)
    n = int(z_n.shape[0])
    metrics = {
        "zap/neighbor_count": float(n),
        "zap/neighbor_fallback": 0.0,
        "zap/z_weight_entropy": 0.0,
    }
    if n == 0:
        metrics["zap/neighbor_fallback"] = 1.0
        return torch.ones(0, dtype=torch.float32), metrics

    k_z = cosine_kernel_z(z_q.expand(n, -1), z_n, temperature=tau_z)
    k_a = rbf_kernel_a(a_q.expand(n, -1), a_n, temperature=tau_a)
    scores = torch.log(k_z.clamp_min(1e-12)) + torch.log(k_a.clamp_min(1e-12))

    if top_k is not None and 0 < int(top_k) < n:
        top_k = int(top_k)
        vals, idx = torch.topk(scores, k=top_k)
        mask_scores = torch.full_like(scores, fill_value=-1e9)
        mask_scores[idx] = vals
        scores = mask_scores

    weights = torch.softmax(scores, dim=0)
    z_only = torch.softmax(torch.log(k_z.clamp_min(1e-12)), dim=0)
    entropy = float((-(z_only * torch.log(z_only.clamp_min(1e-12))).sum()).item())
    uniform = math.log(n)
    metrics["zap/z_weight_entropy"] = entropy
    if entropy >= uniform * float(uniform_entropy_ratio):
        metrics["zap/neighbor_fallback"] = 1.0
        return torch.zeros(n, dtype=torch.float32), metrics
    return weights, metrics


def select_zap_include(
    *,
    path_z: torch.Tensor,
    path_a_buf: torch.Tensor,
    path_a_pi: torch.Tensor,
    path_dones: torch.Tensor,
    tau_z: float,
    tau_a: float,
    truncate_eps: float,
) -> tuple[int, float]:
    """Return ``(include, mean_c)`` for a path under current $$a^\\pi$$."""
    z = path_z.reshape(path_z.shape[0], -1).float()
    a_buf = path_a_buf.reshape(path_a_buf.shape[0], -1).float()
    a_pi = path_a_pi.reshape(path_a_pi.shape[0], -1).float()
    dones = path_dones.reshape(-1).to(torch.bool)
    t_path = int(z.shape[0])
    if t_path < 1:
        raise ValueError("path must contain the query atom.")
    include = 1
    mean_c = 0.0
    if t_path > 1:
        c_list = []
        for k in range(1, t_path):
            if bool(dones[k - 1]):
                break
            c_k = continuation_factor(
                z[0:1],
                z[k : k + 1],
                a_pi[k : k + 1],
                a_buf[k : k + 1],
                tau_z=tau_z,
                tau_a=tau_a,
            )
            c_list.append(c_k.reshape(()))
        if c_list:
            continuations = torch.stack(c_list)
            mean_c = float(continuations.mean().item())
            tau_extra, _ = accumulate_product_and_truncate(
                continuations, truncate_eps=truncate_eps
            )
            include = 1 + tau_extra
            for k in range(include):
                if bool(dones[k]):
                    include = k + 1
                    break
    return include, mean_c


def compute_zap_path_target(
    *,
    path_z: torch.Tensor,
    path_a_buf: torch.Tensor,
    path_a_pi: torch.Tensor,
    path_rewards: torch.Tensor,
    path_dones: torch.Tensor,
    path_horizons: torch.Tensor | None = None,
    gamma: float,
    tau_z: float,
    tau_a: float,
    truncate_eps: float,
    neighbor_z: torch.Tensor | None = None,
    neighbor_a: torch.Tensor | None = None,
    neighbor_q: torch.Tensor | None = None,
    local_bootstrap_q: torch.Tensor | None = None,
    neighbor_top_k: int | None = 32,
    uniform_entropy_ratio: float = 0.95,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build one ZAP target along a stored path.

    Index 0 is the query atom. Continuations $$c_k$$ for $$k \\ge 1$$ compare
    $$z_0$$ to $$z_k$$ and $$a^\\pi_k$$ to $$a^{buf}_k$$. The query reward is
    always included.
    """
    z = path_z.reshape(path_z.shape[0], -1).float()
    a_pi = path_a_pi.reshape(path_a_pi.shape[0], -1).float()
    rewards = path_rewards.reshape(-1).float()
    dones = path_dones.reshape(-1).to(torch.bool)
    t_path = int(rewards.numel())
    if z.shape[0] != t_path or a_pi.shape[0] != t_path:
        raise ValueError("path tensors must share the same length.")

    include, mean_c = select_zap_include(
        path_z=path_z,
        path_a_buf=path_a_buf,
        path_a_pi=path_a_pi,
        path_dones=path_dones,
        tau_z=tau_z,
        tau_a=tau_a,
        truncate_eps=truncate_eps,
    )
    metrics: dict[str, float] = {
        "zap/path_len": float(t_path),
        "zap/tau": float(include),
        "zap/mean_c": mean_c,
        "zap/bootstrap": 0.0,
        "zap/neighbor_mix": 0.0,
    }
    ret, bootstrap_discount = path_nstep_return(
        rewards,
        gamma=gamma,
        horizons=path_horizons,
        num_steps=include,
    )

    ended = bool(dones[include - 1])
    bootstrapped = (not ended) and bootstrap_discount > 0
    if not bootstrapped:
        return ret.reshape(1), metrics

    if include < t_path:
        trunc_z = z[include]
        trunc_a_pi = a_pi[include]
    else:
        trunc_z = z[include - 1]
        trunc_a_pi = a_pi[include - 1]

    local_q = (
        torch.as_tensor(local_bootstrap_q, dtype=torch.float32).reshape(())
        if local_bootstrap_q is not None
        else torch.zeros((), dtype=torch.float32)
    )
    bootstrap_q = local_q
    if (
        neighbor_z is not None
        and neighbor_a is not None
        and neighbor_q is not None
        and neighbor_z.shape[0] > 0
    ):
        weights, w_metrics = neighbor_weights(
            trunc_z,
            trunc_a_pi,
            neighbor_z,
            neighbor_a,
            tau_z=tau_z,
            tau_a=tau_a,
            top_k=neighbor_top_k,
            uniform_entropy_ratio=uniform_entropy_ratio,
        )
        metrics.update(w_metrics)
        if weights.numel() > 0 and float(weights.sum().item()) > 0:
            q_n = neighbor_q.reshape(-1).float()
            if q_n.shape[0] != weights.shape[0]:
                raise ValueError("neighbor_q must align with neighbor rows.")
            bootstrap_q = (weights * q_n).sum()
            metrics["zap/neighbor_mix"] = 1.0

    metrics["zap/bootstrap"] = 1.0
    y = ret + float(bootstrap_discount) * bootstrap_q
    return y.reshape(1), metrics


def stack_path_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    if not metric_list:
        return {}
    keys = set()
    for metrics in metric_list:
        keys.update(metrics.keys())
    out: dict[str, float] = {}
    for key in sorted(keys):
        vals = [float(m.get(key, 0.0)) for m in metric_list]
        out[key] = float(sum(vals) / max(len(vals), 1))
    return out


def attach_episode_zap_paths(
    transitions: list[Any],
    *,
    max_atoms: int,
    gamma: float,
    z_key: str = "z_rl",
) -> None:
    """Write padded forward path fields onto each transition of one episode."""
    n = len(transitions)
    max_atoms = max(int(max_atoms), 1)
    if n == 0:
        return
    z_dim = int(transitions[0].curr_obs[z_key].detach().reshape(-1).numel())
    a_dim = int(transitions[0].actions.detach().reshape(-1).numel())
    for start in range(n):
        end = min(n, start + max_atoms)
        length = end - start
        z_pad = torch.zeros(max_atoms, z_dim, dtype=torch.float32)
        a_pad = torch.zeros(max_atoms, a_dim, dtype=torch.float32)
        r_pad = torch.zeros(max_atoms, dtype=torch.float32)
        h_pad = torch.ones(max_atoms, dtype=torch.float32)
        d_pad = torch.ones(max_atoms, dtype=torch.bool)
        for offset, idx in enumerate(range(start, end)):
            tr = transitions[idx]
            if z_key not in tr.curr_obs:
                raise KeyError(f"ZAP path packing requires curr_obs['{z_key}'].")
            z_pad[offset] = tr.curr_obs[z_key].detach().float().reshape(-1)
            if not isinstance(tr.actions, torch.Tensor):
                raise ValueError("ZAP paths require actions on every transition.")
            a_pad[offset] = tr.actions.detach().float().reshape(-1)
            if not isinstance(tr.rewards, torch.Tensor):
                raise ValueError("ZAP paths require rewards on every transition.")
            rewards = tr.rewards.detach().float()
            r_pad[offset] = discounted_chunk_reward(rewards, gamma)
            h_pad[offset] = float(max(int(rewards.reshape(-1).numel()), 1))
            d_pad[offset] = (
                isinstance(tr.dones, torch.Tensor)
                and tr.dones.reshape(-1).to(torch.bool).any()
            )
        # Shape [T=1, B=1, max_atoms, ...] so replay concat on dim=1 works.
        transitions[start].zap_path_z = z_pad.unsqueeze(0).unsqueeze(0)
        transitions[start].zap_path_a = a_pad.unsqueeze(0).unsqueeze(0)
        transitions[start].zap_path_r = r_pad.view(1, 1, max_atoms)
        transitions[start].zap_path_horizon = h_pad.view(1, 1, max_atoms)
        transitions[start].zap_path_done = d_pad.view(1, 1, max_atoms)
        transitions[start].zap_path_len = torch.tensor(
            [[[length]]], dtype=torch.long
        )
