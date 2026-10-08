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
from rlinf.algorithms.zap.kernels import cosine_kernel_query_keys, rbf_kernel_query_keys


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
    extra, products = accumulate_product_and_truncate_batch(
        torch.as_tensor(continuations, dtype=torch.float32).reshape(1, -1),
        truncate_eps=truncate_eps,
    )
    return int(extra.reshape(())), products.reshape(-1)


def accumulate_product_and_truncate_batch(
    continuations: torch.Tensor,
    *,
    truncate_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched product truncation.

    ``continuations`` is ``[B, T]``. Returns extra counts ``[B]`` and products
    ``[B, T]``.
    """
    if truncate_eps <= 0 or truncate_eps > 1:
        raise ValueError(f"truncate_eps must be in (0, 1], got {truncate_eps}.")
    c = torch.as_tensor(continuations, dtype=torch.float32)
    if c.dim() != 2:
        raise ValueError(f"continuations must be 2D, got {tuple(c.shape)}.")
    batch, length = c.shape
    if length == 0:
        return torch.zeros(batch, dtype=torch.long, device=c.device), c
    products = torch.cumprod(c, dim=-1)
    below = products < float(truncate_eps)
    none = ~below.any(dim=-1)
    # First True index; unused rows get ``length`` then swapped to full extra.
    first = below.float().argmax(dim=-1)
    extra = torch.where(
        none,
        torch.full_like(first, length),
        first,
    )
    return extra.to(torch.long), products


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
    r = torch.as_tensor(rewards, dtype=torch.float32).reshape(1, -1)
    if num_steps is None:
        include = torch.tensor([r.shape[-1]], dtype=torch.long)
    else:
        if num_steps < 0:
            raise ValueError(f"num_steps must be >= 0, got {num_steps}.")
        include = torch.tensor([min(int(num_steps), int(r.shape[-1]))], dtype=torch.long)
    if horizons is None:
        h = torch.ones_like(r)
    else:
        h = torch.as_tensor(horizons, dtype=torch.float32).reshape(1, -1)
        if h.shape[-1] < r.shape[-1]:
            raise ValueError("horizons must cover every reward atom.")
        h = h[:, : r.shape[-1]]
    total, discount = path_nstep_return_batch(r, h, include, gamma)
    return total.reshape(()), float(discount.reshape(()))


def path_nstep_return_batch(
    rewards: torch.Tensor,
    horizons: torch.Tensor,
    include: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched n-step returns.

    ``rewards`` / ``horizons`` are ``[B, L]``; ``include`` is ``[B]``. Returns
    folded rewards ``[B]`` and bootstrap discounts ``[B]``.
    """
    r = torch.as_tensor(rewards, dtype=torch.float32)
    h = torch.as_tensor(horizons, dtype=torch.float32).clamp_min(1.0)
    if r.shape != h.shape or r.dim() != 2:
        raise ValueError("rewards and horizons must share shape [B, L].")
    batch, length = r.shape
    steps = torch.arange(length, device=r.device)
    mask = steps.unsqueeze(0) < include.reshape(batch, 1).to(device=r.device)
    h_exc = torch.zeros_like(h)
    if length > 1:
        h_exc[:, 1:] = torch.cumsum(h[:, :-1], dim=-1)
    disc = torch.pow(
        torch.as_tensor(gamma, dtype=r.dtype, device=r.device),
        h_exc,
    )
    total = (r * disc * mask.to(dtype=r.dtype)).sum(dim=-1)
    boot_power = (h * mask.to(dtype=h.dtype)).sum(dim=-1)
    discount = torch.pow(
        torch.as_tensor(gamma, dtype=r.dtype, device=r.device),
        boot_power,
    )
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
    weights, metrics = neighbor_weights_batch(
        z_query.reshape(1, -1),
        a_pi_query.reshape(1, -1),
        neighbor_z,
        neighbor_a,
        tau_z=tau_z,
        tau_a=tau_a,
        top_k=top_k,
        uniform_entropy_ratio=uniform_entropy_ratio,
    )
    return weights.reshape(-1), {
        key: float(value.reshape(())) for key, value in metrics.items()
    }


def neighbor_weights_batch(
    z_query: torch.Tensor,
    a_pi_query: torch.Tensor,
    neighbor_z: torch.Tensor,
    neighbor_a: torch.Tensor,
    *,
    tau_z: float,
    tau_a: float,
    top_k: int | None = None,
    uniform_entropy_ratio: float = 0.95,
    exclude_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Batched neighbor softmax.

    Queries are ``[B, D]``, neighbors ``[N, D]``. ``exclude_mask`` is optional
    ``[B, N]`` True entries (typically the query's own row).
    """
    z_q = z_query.reshape(z_query.shape[0], -1).float()
    a_q = a_pi_query.reshape(a_pi_query.shape[0], -1).float()
    batch = int(z_q.shape[0])
    if neighbor_z is None or neighbor_a is None or neighbor_z.numel() == 0:
        zeros = torch.zeros(batch, 0, dtype=torch.float32, device=z_q.device)
        metrics = {
            "zap/neighbor_count": torch.zeros(batch, dtype=torch.float32),
            "zap/neighbor_fallback": torch.ones(batch, dtype=torch.float32),
            "zap/z_weight_entropy": torch.zeros(batch, dtype=torch.float32),
        }
        return zeros, metrics

    z_n = neighbor_z.reshape(neighbor_z.shape[0], -1).float()
    a_n = neighbor_a.reshape(neighbor_a.shape[0], -1).float()
    n_neighbors = int(z_n.shape[0])
    k_z = cosine_kernel_query_keys(z_q, z_n, temperature=tau_z)
    k_a = rbf_kernel_query_keys(a_q, a_n, temperature=tau_a)
    scores = torch.log(k_z.clamp_min(1e-12)) + torch.log(k_a.clamp_min(1e-12))
    z_scores = torch.log(k_z.clamp_min(1e-12))
    if exclude_mask is not None:
        exclude = exclude_mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(exclude, -1e9)
        z_scores = z_scores.masked_fill(exclude, -1e9)
    else:
        exclude = torch.zeros_like(scores, dtype=torch.bool)

    valid = ~exclude
    n_eff = valid.sum(dim=-1).clamp(min=1)
    if top_k is not None and 0 < int(top_k) < n_neighbors:
        k = min(int(top_k), n_neighbors)
        vals, idx = torch.topk(scores, k=k, dim=-1)
        masked = torch.full_like(scores, fill_value=-1e9)
        masked.scatter_(dim=-1, index=idx, src=vals)
        scores = masked

    weights = torch.softmax(scores, dim=-1)
    z_only = torch.softmax(z_scores, dim=-1)
    entropy = -(z_only * torch.log(z_only.clamp_min(1e-12))).sum(dim=-1)
    uniform = torch.log(n_eff.float())
    fallback = entropy >= uniform * float(uniform_entropy_ratio)
    weights = torch.where(fallback.unsqueeze(-1), torch.zeros_like(weights), weights)
    metrics = {
        "zap/neighbor_count": n_eff.float(),
        "zap/neighbor_fallback": fallback.to(dtype=torch.float32),
        "zap/z_weight_entropy": entropy,
    }
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
    include, mean_c = select_zap_include_batch(
        path_z=path_z.unsqueeze(0),
        path_a_buf=path_a_buf.unsqueeze(0),
        path_a_pi=path_a_pi.unsqueeze(0),
        path_dones=path_dones.unsqueeze(0),
        path_len=torch.tensor([path_z.shape[0]], dtype=torch.long),
        tau_z=tau_z,
        tau_a=tau_a,
        truncate_eps=truncate_eps,
    )
    return int(include.reshape(())), float(mean_c.reshape(()))


def select_zap_include_batch(
    *,
    path_z: torch.Tensor,
    path_a_buf: torch.Tensor,
    path_a_pi: torch.Tensor,
    path_dones: torch.Tensor,
    path_len: torch.Tensor,
    tau_z: float,
    tau_a: float,
    truncate_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched include lengths.

    Path tensors are ``[B, L, ...]``. Returns include ``[B]`` and mean $$c$$
    ``[B]``.
    """
    z = path_z.float()
    batch, length = z.shape[0], z.shape[1]
    include = torch.ones(batch, dtype=torch.long, device=z.device)
    mean_c = torch.zeros(batch, dtype=torch.float32, device=z.device)
    if length < 2:
        return include, mean_c

    k_z = cosine_kernel_query_keys(z[:, 0], z[:, 1:], temperature=tau_z)
    a_pi_rest = path_a_pi[:, 1:].reshape(batch, length - 1, -1).float()
    a_buf_rest = path_a_buf[:, 1:].reshape(batch, length - 1, -1).float()
    k_a = torch.exp(-((a_pi_rest - a_buf_rest) ** 2).sum(dim=-1) / float(tau_a))
    continuations = k_z * k_a
    steps = torch.arange(length - 1, device=z.device)
    valid = (steps.unsqueeze(0) + 1) < path_len.reshape(batch, 1).to(device=z.device)
    dones = path_dones.reshape(batch, length).to(torch.bool)
    valid = valid & ~dones[:, :-1]
    continuations = continuations.masked_fill(~valid, 0.0)
    extra, _ = accumulate_product_and_truncate_batch(
        continuations, truncate_eps=truncate_eps
    )
    include = (1 + extra).clamp(min=1)
    include = torch.minimum(include, path_len.to(device=z.device, dtype=torch.long))
    done_pos = torch.where(
        dones,
        torch.arange(length, device=z.device).unsqueeze(0).expand(batch, -1),
        torch.full((batch, length), length, device=z.device, dtype=torch.long),
    )
    first_done = done_pos.min(dim=-1).values
    include = torch.minimum(include, first_done + 1)
    valid_count = valid.to(dtype=torch.float32).sum(dim=-1).clamp_min(1.0)
    mean_c = (continuations * valid.to(dtype=continuations.dtype)).sum(
        dim=-1
    ) / valid_count
    mean_c = torch.where(valid.any(dim=-1), mean_c, torch.zeros_like(mean_c))
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
    length = int(path_rewards.reshape(-1).numel())
    if path_horizons is None:
        horizons = torch.ones(length, dtype=torch.float32)
    else:
        horizons = path_horizons
    local_q = (
        torch.as_tensor(local_bootstrap_q, dtype=torch.float32).reshape(1)
        if local_bootstrap_q is not None
        else torch.zeros(1, dtype=torch.float32)
    )
    y, metrics = compute_zap_path_targets_batch(
        path_z=path_z.unsqueeze(0),
        path_a_buf=path_a_buf.unsqueeze(0),
        path_a_pi=path_a_pi.unsqueeze(0),
        path_rewards=path_rewards.reshape(1, -1),
        path_dones=path_dones.reshape(1, -1),
        path_horizons=horizons.reshape(1, -1),
        path_len=torch.tensor([length], dtype=torch.long),
        gamma=gamma,
        tau_z=tau_z,
        tau_a=tau_a,
        truncate_eps=truncate_eps,
        local_bootstrap_q=local_q,
        neighbor_z=neighbor_z,
        neighbor_a=neighbor_a,
        neighbor_q=neighbor_q,
        neighbor_top_k=neighbor_top_k,
        uniform_entropy_ratio=uniform_entropy_ratio,
    )
    return y.reshape(1), {key: float(value.mean()) for key, value in metrics.items()}


def compute_zap_path_targets_batch(
    *,
    path_z: torch.Tensor,
    path_a_buf: torch.Tensor,
    path_a_pi: torch.Tensor,
    path_rewards: torch.Tensor,
    path_dones: torch.Tensor,
    path_horizons: torch.Tensor,
    path_len: torch.Tensor,
    gamma: float,
    tau_z: float,
    tau_a: float,
    truncate_eps: float,
    local_bootstrap_q: torch.Tensor,
    neighbor_z: torch.Tensor | None = None,
    neighbor_a: torch.Tensor | None = None,
    neighbor_q: torch.Tensor | None = None,
    exclude_mask: torch.Tensor | None = None,
    neighbor_top_k: int | None = 32,
    uniform_entropy_ratio: float = 0.95,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build ZAP targets for a batch of stored paths."""
    z = path_z.float()
    a_pi = path_a_pi.float()
    rewards = path_rewards.reshape(path_rewards.shape[0], -1).float()
    dones = path_dones.reshape(path_dones.shape[0], -1).to(torch.bool)
    horizons = path_horizons.reshape(path_horizons.shape[0], -1).float()
    batch, length = rewards.shape
    include, mean_c = select_zap_include_batch(
        path_z=z,
        path_a_buf=path_a_buf,
        path_a_pi=a_pi,
        path_dones=dones,
        path_len=path_len,
        tau_z=tau_z,
        tau_a=tau_a,
        truncate_eps=truncate_eps,
    )
    ret, bootstrap_discount = path_nstep_return_batch(
        rewards, horizons, include, gamma
    )
    boot_pos = (include - 1).clamp(min=0)
    ended = dones.gather(1, boot_pos.unsqueeze(1)).reshape(-1)
    bootstrapped = (~ended) & (bootstrap_discount > 0)
    use_next = include < path_len.to(device=include.device)
    trunc_idx = torch.where(use_next, include, (include - 1).clamp(min=0)).clamp(
        max=length - 1
    )
    gather_z = trunc_idx.view(batch, 1, 1).expand(batch, 1, z.shape[-1])
    gather_a = trunc_idx.view(batch, 1, 1).expand(batch, 1, a_pi.shape[-1])
    trunc_z = z.gather(1, gather_z).reshape(batch, -1)
    trunc_a_pi = a_pi.gather(1, gather_a).reshape(batch, -1)

    local_q = torch.as_tensor(
        local_bootstrap_q, dtype=torch.float32, device=z.device
    ).reshape(batch)
    bootstrap_q = local_q
    mix = torch.zeros(batch, dtype=torch.float32, device=z.device)
    w_metrics: dict[str, torch.Tensor] = {
        "zap/neighbor_count": torch.zeros(batch, dtype=torch.float32, device=z.device),
        "zap/neighbor_fallback": torch.ones(batch, dtype=torch.float32, device=z.device),
        "zap/z_weight_entropy": torch.zeros(batch, dtype=torch.float32, device=z.device),
    }
    if (
        neighbor_z is not None
        and neighbor_a is not None
        and neighbor_q is not None
        and neighbor_z.shape[0] > 0
    ):
        weights, w_metrics = neighbor_weights_batch(
            trunc_z,
            trunc_a_pi,
            neighbor_z,
            neighbor_a,
            tau_z=tau_z,
            tau_a=tau_a,
            top_k=neighbor_top_k,
            uniform_entropy_ratio=uniform_entropy_ratio,
            exclude_mask=exclude_mask,
        )
        q_n = neighbor_q.reshape(-1).float()
        if q_n.shape[0] != weights.shape[1]:
            raise ValueError("neighbor_q must align with neighbor rows.")
        mixed = weights.matmul(q_n)
        used = weights.sum(dim=-1) > 0
        bootstrap_q = torch.where(used, mixed, local_q)
        mix = used.to(dtype=torch.float32)

    y = ret + bootstrap_discount * bootstrap_q
    y = torch.where(bootstrapped, y, ret)
    metrics = {
        "zap/path_len": path_len.to(dtype=torch.float32, device=z.device),
        "zap/tau": include.to(dtype=torch.float32),
        "zap/mean_c": mean_c,
        "zap/bootstrap": bootstrapped.to(dtype=torch.float32),
        "zap/neighbor_mix": mix * bootstrapped.to(dtype=torch.float32),
    }
    metrics.update(w_metrics)
    return y, metrics


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


def reduce_zap_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Mean-reduce batched ZAP diagnostics to logger scalars."""
    return {key: float(value.float().mean().item()) for key, value in metrics.items()}


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
    if z_key not in transitions[0].curr_obs:
        raise KeyError(f"ZAP path packing requires curr_obs['{z_key}'].")
    z_rows = []
    a_rows = []
    r_rows = []
    h_rows = []
    d_rows = []
    for tr in transitions:
        if z_key not in tr.curr_obs:
            raise KeyError(f"ZAP path packing requires curr_obs['{z_key}'].")
        if not isinstance(tr.actions, torch.Tensor):
            raise ValueError("ZAP paths require actions on every transition.")
        if not isinstance(tr.rewards, torch.Tensor):
            raise ValueError("ZAP paths require rewards on every transition.")
        z_rows.append(tr.curr_obs[z_key].detach().float().reshape(-1))
        a_rows.append(tr.actions.detach().float().reshape(-1))
        rewards = tr.rewards.detach().float()
        r_rows.append(discounted_chunk_reward(rewards, gamma).reshape(()))
        h_rows.append(
            torch.tensor(float(max(int(rewards.reshape(-1).numel()), 1)))
        )
        d_rows.append(
            torch.tensor(
                isinstance(tr.dones, torch.Tensor)
                and bool(tr.dones.reshape(-1).to(torch.bool).any())
            )
        )
    z = torch.stack(z_rows, dim=0)
    a = torch.stack(a_rows, dim=0)
    r = torch.stack(r_rows, dim=0)
    h = torch.stack(h_rows, dim=0)
    d = torch.stack(d_rows, dim=0)
    z_win = _forward_windows(z, max_atoms, pad_value=0.0)
    a_win = _forward_windows(a, max_atoms, pad_value=0.0)
    r_win = _forward_windows(r, max_atoms, pad_value=0.0)
    h_win = _forward_windows(h, max_atoms, pad_value=1.0)
    d_win = _forward_windows(d.to(torch.float32), max_atoms, pad_value=1.0).to(
        torch.bool
    )
    lengths = torch.clamp(
        torch.arange(n, 0, -1, dtype=torch.long), max=max_atoms
    )
    for start in range(n):
        # Shape [T=1, B=1, max_atoms, ...] so replay concat on dim=1 works.
        transitions[start].zap_path_z = z_win[start].unsqueeze(0).unsqueeze(0)
        transitions[start].zap_path_a = a_win[start].unsqueeze(0).unsqueeze(0)
        transitions[start].zap_path_r = r_win[start].view(1, 1, max_atoms)
        transitions[start].zap_path_horizon = h_win[start].view(1, 1, max_atoms)
        transitions[start].zap_path_done = d_win[start].view(1, 1, max_atoms)
        transitions[start].zap_path_len = lengths[start].view(1, 1, 1)


def _forward_windows(
    rows: torch.Tensor, max_atoms: int, *, pad_value: float
) -> torch.Tensor:
    """Sliding windows ``[n, max_atoms, ...]`` along the episode axis."""
    if rows.dim() == 1:
        rows = rows.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    pad = torch.full(
        (max_atoms - 1, *rows.shape[1:]),
        fill_value=pad_value,
        dtype=rows.dtype,
        device=rows.device,
    )
    padded = torch.cat([rows, pad], dim=0)
    windows = padded.unfold(dimension=0, size=max_atoms, step=1)
    # unfold puts the window axis last: [n, feat..., max_atoms]
    perm = (0, windows.dim() - 1) + tuple(range(1, windows.dim() - 1))
    windows = windows.permute(*perm).contiguous()
    if squeeze:
        windows = windows.squeeze(-1)
    return windows[: rows.shape[0]]
