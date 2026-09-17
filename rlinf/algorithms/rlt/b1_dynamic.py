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

"""B1: Transformer streaming dynamic state for RLT (ManiSkill).

Computes a fixed-size recurrent state ``h_t`` from the appearance RL token
``z_app`` (image-only RLT readout), fuses it into ``z_rl`` via a gate, and
optionally trains future-token predictors ``P_k(h_t) -> z_app_{t+k}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class B1DynamicConfig:
    """Hyper-parameters for B1 dynamic encoder / fusion / prediction."""

    enable: bool = False
    z_dim: int = 2048
    feature_dim: int = 256
    num_tokens: int = 16
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    only_critical: bool = True
    lambda_pred: float = 0.1
    pred_horizons: tuple[int, ...] = (1, 4, 8)
    pred_weights: tuple[float, ...] = (1.0, 0.5, 0.25)


def build_b1_dynamic_config(cfg: Any) -> B1DynamicConfig:
    """Build config from ``algorithm.b1_dynamic`` or ``actor.model.b1_dynamic``."""
    from omegaconf import OmegaConf

    block = OmegaConf.select(cfg, "algorithm.b1_dynamic", default=None)
    if block is None:
        block = OmegaConf.select(cfg, "actor.model.b1_dynamic", default=None)
    if block is None:
        return B1DynamicConfig(enable=False)

    horizons = OmegaConf.select(block, "pred_horizons", default=[1, 4, 8])
    weights = OmegaConf.select(block, "pred_weights", default=[1.0, 0.5, 0.25])
    horizons_t = tuple(int(x) for x in list(horizons))
    weights_t = tuple(float(x) for x in list(weights))
    if len(weights_t) != len(horizons_t):
        raise ValueError(
            "b1_dynamic.pred_weights must match pred_horizons length, got "
            f"{len(weights_t)} vs {len(horizons_t)}."
        )
    return B1DynamicConfig(
        enable=bool(OmegaConf.select(block, "enable", default=False)),
        z_dim=int(OmegaConf.select(block, "z_dim", default=2048)),
        feature_dim=int(OmegaConf.select(block, "feature_dim", default=256)),
        num_tokens=int(OmegaConf.select(block, "num_tokens", default=16)),
        num_layers=int(OmegaConf.select(block, "num_layers", default=2)),
        num_heads=int(OmegaConf.select(block, "num_heads", default=4)),
        mlp_ratio=float(OmegaConf.select(block, "mlp_ratio", default=4.0)),
        dropout=float(OmegaConf.select(block, "dropout", default=0.0)),
        only_critical=bool(OmegaConf.select(block, "only_critical", default=True)),
        lambda_pred=float(OmegaConf.select(block, "lambda_pred", default=0.1)),
        pred_horizons=horizons_t,
        pred_weights=weights_t,
    )


class B1DynamicEncoder(nn.Module):
    """Cross-attention + Transformer update: ``h_t = U(h_{t-1}, z_app)``."""

    def __init__(
        self,
        *,
        z_dim: int = 2048,
        feature_dim: int = 256,
        num_tokens: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.feature_dim = int(feature_dim)
        self.num_tokens = int(num_tokens)

        self.z_proj = nn.Linear(self.z_dim, self.feature_dim)
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.context_to_tokens = nn.Linear(self.feature_dim, self.num_tokens * self.feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.feature_dim * float(mlp_ratio)),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.update = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        dtype = torch.float32 if dtype is None else dtype
        return torch.zeros(
            batch_size,
            self.num_tokens,
            self.feature_dim,
            device=device,
            dtype=dtype,
        )

    def forward(self, z_app: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # z_app: [B, z_dim], h_prev: [B, T, D]
        x = self.z_proj(z_app).unsqueeze(1)  # [B, 1, D]
        q = self.q_proj(x)
        k = self.k_proj(h_prev)
        v = self.v_proj(h_prev)
        context, _ = self.attn(q, k, v, need_weights=False)  # [B, 1, D]
        delta = self.context_to_tokens(context.squeeze(1))
        delta = delta.view(z_app.shape[0], self.num_tokens, self.feature_dim)
        h = h_prev + delta
        return self.update(h)


class B1GatedFusion(nn.Module):
    """``z_rl = z_app + g * W pool(h)``."""

    def __init__(self, *, z_dim: int, feature_dim: int):
        super().__init__()
        self.dynamic_proj = nn.Linear(int(feature_dim), int(z_dim))
        self.gate = nn.Sequential(
            nn.Linear(int(z_dim) + int(feature_dim), int(z_dim)),
            nn.GELU(),
            nn.Linear(int(z_dim), 1),
        )

    def forward(
        self, z_app: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_pool = h.mean(dim=1)
        g = torch.sigmoid(self.gate(torch.cat([z_app, h_pool], dim=-1)))
        z_rl = z_app + g * self.dynamic_proj(h_pool)
        return z_rl, g


class B1FuturePredictors(nn.Module):
    """Predict future appearance tokens from pooled dynamic state."""

    def __init__(
        self,
        *,
        feature_dim: int,
        z_dim: int,
        horizons: Sequence[int],
    ):
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        self.predictors = nn.ModuleDict(
            {
                str(k): nn.Sequential(
                    nn.Linear(int(feature_dim), int(feature_dim)),
                    nn.GELU(),
                    nn.Linear(int(feature_dim), int(z_dim)),
                )
                for k in self.horizons
            }
        )

    def forward(self, h: torch.Tensor) -> dict[int, torch.Tensor]:
        h_pool = h.mean(dim=1)
        return {k: self.predictors[str(k)](h_pool) for k in self.horizons}


class B1DynamicModule(nn.Module):
    """Bundled B1 encoder + fusion + future predictors."""

    def __init__(self, config: B1DynamicConfig):
        super().__init__()
        self.config = config
        self.encoder = B1DynamicEncoder(
            z_dim=config.z_dim,
            feature_dim=config.feature_dim,
            num_tokens=config.num_tokens,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
        self.fusion = B1GatedFusion(
            z_dim=config.z_dim,
            feature_dim=config.feature_dim,
        )
        self.predictors = B1FuturePredictors(
            feature_dim=config.feature_dim,
            z_dim=config.z_dim,
            horizons=config.pred_horizons,
        )

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return self.encoder.init_state(batch_size, device, dtype=dtype)

    def update(
        self, z_app: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(z_app, h_prev)
        z_rl, gate = self.fusion(z_app, h)
        return h, z_rl, gate


@dataclass
class B1DynamicRuntime:
    """Per-env recurrent state for rollout (not a history buffer of z)."""

    config: B1DynamicConfig
    h: torch.Tensor | None = None
    _device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def ensure_state(self, batch_size: int, device: torch.device, module: B1DynamicModule) -> None:
        self._device = device
        if (
            self.h is None
            or self.h.shape[0] != batch_size
            or self.h.device != device
        ):
            self.h = module.init_state(batch_size, device)

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        if self.h is None:
            return
        if env_idx is None:
            self.h.zero_()
            return
        idx = env_idx.reshape(-1).to(device=self.h.device, dtype=torch.long)
        if idx.numel() == 0:
            return
        self.h[idx] = 0

    def step(
        self,
        *,
        module: B1DynamicModule,
        z_app: torch.Tensor,
        critical_mask: torch.Tensor | None,
        dones: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Update h on critical envs; return tensors to store in the transition."""
        batch_size = int(z_app.shape[0])
        device = z_app.device
        self.ensure_state(batch_size, device, module)

        if dones is not None:
            done = torch.as_tensor(dones, device=device, dtype=torch.bool).reshape(
                batch_size, -1
            )
            done = done.any(dim=-1)
            if done.any():
                self.reset(done.nonzero(as_tuple=False).reshape(-1))

        h_prev = self.h.detach()
        if critical_mask is None:
            critical = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            critical = torch.as_tensor(
                critical_mask, device=device, dtype=torch.bool
            ).reshape(batch_size, -1)
            critical = critical.any(dim=-1)

        if self.config.only_critical:
            update_mask = critical
        else:
            update_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        h_new, z_rl, gate = module.update(z_app, h_prev)
        # Non-critical: keep previous h and appearance token as z_rl.
        h_out = torch.where(update_mask[:, None, None], h_new, h_prev)
        z_out = torch.where(update_mask[:, None], z_rl, z_app)
        gate_out = torch.where(update_mask[:, None], gate, torch.zeros_like(gate))
        self.h = h_out.detach()

        return {
            "z_app": z_app,
            "z_rl": z_out,
            "b1_h": h_out.detach(),
            "b1_h_prev": h_prev,
            "b1_gate": gate_out.detach(),
            # Keep a trailing dim so episode stack [T, B, 1] survives
            # ReplayBuffer._flatten_trajectory (requires tensor.dim() >= 2).
            "b1_updated": update_mask.unsqueeze(-1),
        }


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


def compute_b1_pred_loss(
    module: B1DynamicModule,
    *,
    z_app: torch.Tensor,
    h_prev: torch.Tensor,
    future_z: dict[int, torch.Tensor],
    future_mask: dict[int, torch.Tensor],
    config: B1DynamicConfig,
    update_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Recompute ``h_t`` from stored ``(h_prev, z_app)`` and apply ``L_pred``."""
    h_t, _, gate = module.update(z_app, h_prev.detach())
    preds = module.predictors(h_t)
    loss = z_app.new_zeros(())
    metrics: dict[str, float] = {
        "b1_dynamic/gate_mean": float(gate.detach().float().mean().item()),
    }
    if update_mask is not None:
        active = update_mask.reshape(-1).to(dtype=torch.bool)
        metrics["b1_dynamic/update_frac"] = float(active.float().mean().item())
    else:
        active = None

    total_weight = 0.0
    for horizon, weight in zip(config.pred_horizons, config.pred_weights):
        if horizon not in future_z or horizon not in future_mask:
            continue
        target = future_z[horizon]
        mask = future_mask[horizon].reshape(-1).to(dtype=torch.bool)
        if active is not None:
            mask = mask & active
        if not bool(mask.any()):
            metrics[f"b1_dynamic/pred_valid_frac_{horizon}"] = 0.0
            continue
        pred = preds[horizon]
        dist = cosine_distance(pred[mask], target[mask].detach())
        loss = loss + float(weight) * dist.mean()
        total_weight += float(weight)
        metrics[f"b1_dynamic/pred_dist_{horizon}"] = float(dist.mean().item())
        metrics[f"b1_dynamic/pred_valid_frac_{horizon}"] = float(
            mask.float().mean().item()
        )

    if total_weight > 0:
        loss = loss / total_weight
    metrics["b1_dynamic/pred_loss"] = float(loss.detach().item())
    return loss, metrics


def ensure_b1_obs_schema(
    obs: dict[str, torch.Tensor],
    *,
    horizons: Sequence[int],
    z_dim: int = 2048,
    feature_dim: int = 256,
    num_tokens: int = 16,
) -> None:
    """Pad missing B1 / future keys so TrajectoryCache schemas stay aligned.

    ``next_obs`` must carry the same key set as ``curr_obs``: terminal rows share
    ``curr_obs`` (with futures), while non-terminal ``next_obs`` previously omitted
    them and froze the cache without ``z_app_future_*``.
    """
    device = torch.device("cpu")
    dtype = torch.float32
    for value in obs.values():
        if isinstance(value, torch.Tensor):
            device = value.device
            dtype = value.dtype
            break

    z_rl = obs.get("z_rl")
    z_app = obs.get("z_app")
    if isinstance(z_app, torch.Tensor):
        ref = z_app
    elif isinstance(z_rl, torch.Tensor):
        ref = z_rl
    else:
        ref = torch.zeros(1, 1, int(z_dim), device=device, dtype=dtype)

    if not isinstance(obs.get("z_app"), torch.Tensor):
        obs["z_app"] = torch.zeros_like(ref)
    if not isinstance(obs.get("b1_h"), torch.Tensor):
        obs["b1_h"] = torch.zeros(
            1, 1, int(num_tokens), int(feature_dim), device=device, dtype=ref.dtype
        )
    if not isinstance(obs.get("b1_h_prev"), torch.Tensor):
        obs["b1_h_prev"] = torch.zeros_like(obs["b1_h"])
    if not isinstance(obs.get("b1_gate"), torch.Tensor):
        obs["b1_gate"] = torch.zeros(1, 1, 1, device=device, dtype=ref.dtype)
    if not isinstance(obs.get("b1_updated"), torch.Tensor):
        obs["b1_updated"] = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)

    z_app = obs["z_app"]
    false_mask = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)
    for horizon in horizons:
        z_key = f"z_app_future_{horizon}"
        m_key = f"z_app_future_mask_{horizon}"
        if not isinstance(obs.get(z_key), torch.Tensor):
            obs[z_key] = torch.zeros_like(z_app)
        if not isinstance(obs.get(m_key), torch.Tensor):
            obs[m_key] = false_mask.clone()


def attach_b1_future_targets(
    curr_obs: dict[str, torch.Tensor],
    *,
    flat_curr_obs: dict[str, Any],
    t: int,
    env_idx: int,
    traj_len: int,
    bsz: int,
    horizons: Sequence[int],
    z_dim: int = 2048,
    feature_dim: int = 256,
    num_tokens: int = 16,
) -> None:
    """Write a fixed B1 curr_obs schema and fill future appearance targets.

    TrajectoryCache freezes keys on the first ``put``, so every transition must
    carry the same B1 / future / mask keys. Masks use shape ``[1, 1, 1]`` so they
    survive ``ReplayBuffer._flatten_trajectory`` (keeps only ``tensor.dim() >= 2``).
    """
    ensure_b1_obs_schema(
        curr_obs,
        horizons=horizons,
        z_dim=z_dim,
        feature_dim=feature_dim,
        num_tokens=num_tokens,
    )

    device = curr_obs["z_app"].device
    z_app = curr_obs["z_app"]
    z_app_all = flat_curr_obs.get("z_app")
    has_lookup = isinstance(z_app_all, torch.Tensor)

    def _mask(valid: bool) -> torch.Tensor:
        return torch.full(
            (1, 1, 1),
            fill_value=bool(valid),
            dtype=torch.bool,
            device=device,
        )

    for horizon in horizons:
        future_t = t + int(horizon)
        z_key = f"z_app_future_{horizon}"
        m_key = f"z_app_future_mask_{horizon}"
        if not has_lookup or future_t >= traj_len:
            curr_obs[z_key] = torch.zeros_like(z_app)
            curr_obs[m_key] = _mask(False)
            continue
        future_idx = future_t * int(bsz) + int(env_idx)
        if future_idx >= int(z_app_all.shape[0]):
            curr_obs[z_key] = torch.zeros_like(z_app)
            curr_obs[m_key] = _mask(False)
            continue
        curr_obs[z_key] = z_app_all[future_idx : future_idx + 1].reshape_as(z_app)
        curr_obs[m_key] = _mask(True)


def maybe_build_b1_module(cfg: Any) -> B1DynamicModule | None:
    config = build_b1_dynamic_config(cfg)
    if not config.enable:
        return None
    return B1DynamicModule(config)


__all__ = [
    "B1DynamicConfig",
    "B1DynamicEncoder",
    "B1DynamicModule",
    "B1DynamicRuntime",
    "B1FuturePredictors",
    "B1GatedFusion",
    "attach_b1_future_targets",
    "build_b1_dynamic_config",
    "compute_b1_pred_loss",
    "cosine_distance",
    "ensure_b1_obs_schema",
    "maybe_build_b1_module",
]
