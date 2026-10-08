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

"""Bee policy / correction / multiplier losses (arXiv 2609.27450 Eqs. 3–9)."""

from __future__ import annotations

import math

import torch


def mahalanobis_rho(
    policy_action: torch.Tensor,
    corrected_action: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Normalized per-sample Mahalanobis deviation (Bee Eq. 3).

    ``ρ = (1/D) (a_θ - â^H)^T Σ^{-1} (a_θ - â^H)`` with diagonal ``Σ = diag(σ²)``.
    Returns shape ``[B]``.
    """
    diff = policy_action - corrected_action
    inv_var = 1.0 / torch.square(sigma).clamp_min(1e-8)
    return torch.mean(diff * diff * inv_var, dim=-1)


def correction_nll_loss(
    residual: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Diagonal Gaussian NLL of human residual ``Δ^H`` (Bee Eq. 6)."""
    var = torch.square(sigma).clamp_min(1e-8)
    nll = 0.5 * (
        torch.log(var) + torch.square(residual - mu) / var + math.log(2.0 * math.pi)
    )
    return torch.mean(nll)


def bee_actor_loss(
    q_values: torch.Tensor,
    policy_action: torch.Tensor,
    proposal: torch.Tensor,
    rho: torch.Tensor,
    lam: torch.Tensor,
    *,
    anchor_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normalized primal actor objective (Bee Eq. 8).

    ``L_π = E[ (-Q + η‖a-ã‖² + λ ρ) / (1+λ) ]``. The constant ``-λ ε`` term
    is omitted for the actor step.
    """
    if q_values.ndim > 1:
        q = q_values.reshape(q_values.shape[0], -1).mean(dim=-1)
    else:
        q = q_values
    anchor = torch.mean(torch.square(policy_action - proposal), dim=-1)
    lam = lam.reshape(-1)
    rho = rho.reshape(-1)
    numer = -q + float(anchor_weight) * anchor + lam * rho
    denom = 1.0 + lam
    loss = torch.mean(numer / denom)
    metrics = {
        "bee/q_pi": float(q.mean().detach().item()),
        "bee/anchor": float(anchor.mean().detach().item()),
        "bee/rho": float(rho.mean().detach().item()),
        "bee/lambda": float(lam.mean().detach().item()),
        "bee/actor_loss": float(loss.detach().item()),
    }
    return loss, metrics


def bee_multiplier_loss(
    lam: torch.Tensor,
    rho: torch.Tensor,
    *,
    epsilon: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dual ascent as minimization of ``-λ(ρ - ε)`` (Bee Eq. 9)."""
    lam = lam.reshape(-1)
    rho = rho.reshape(-1)
    slack = rho - float(epsilon)
    loss = -torch.mean(lam * slack)
    violation = (slack > 0).to(dtype=lam.dtype)
    metrics = {
        "bee/multiplier_loss": float(loss.detach().item()),
        "bee/constraint_slack": float(slack.mean().detach().item()),
        "bee/violation_rate": float(violation.mean().detach().item()),
    }
    return loss, metrics
