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

"""Kernels for ZAP: (Z, A) Path continuation and neighbor weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten_rows(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 0:
        raise ValueError(f"{name} must have a batch dimension, got scalar.")
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    return tensor.reshape(tensor.shape[0], -1)


def cosine_kernel_z(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """$$k_z = \\exp((\\cos(z, z') - 1) / \\tau_z)$$ in (0, 1]."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    left = F.normalize(_flatten_rows(left, "left").float(), dim=-1, eps=1e-6)
    right = F.normalize(_flatten_rows(right, "right").float(), dim=-1, eps=1e-6)
    if left.shape != right.shape:
        raise ValueError(
            f"z tensors must match, got {tuple(left.shape)} vs {tuple(right.shape)}."
        )
    cos = (left * right).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.exp((cos - 1.0) / float(temperature))


def rbf_kernel_a(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """$$k_a = \\exp(-\\|a - a'\\|^2 / \\tau_a)$$ in (0, 1]."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    left = _flatten_rows(left, "left").float()
    right = _flatten_rows(right, "right").float()
    if left.shape != right.shape:
        raise ValueError(
            f"action tensors must match, got {tuple(left.shape)} vs "
            f"{tuple(right.shape)}."
        )
    sq = ((left - right) ** 2).sum(dim=-1)
    return torch.exp(-sq / float(temperature))


def continuation_factor(
    z_anchor: torch.Tensor,
    z_step: torch.Tensor,
    a_pi: torch.Tensor,
    a_buf: torch.Tensor,
    *,
    tau_z: float,
    tau_a: float,
) -> torch.Tensor:
    """$$c = k_z(z_t, z) \\cdot k_a(a^\\pi, a^{buf})$$."""
    return cosine_kernel_z(z_anchor, z_step, temperature=tau_z) * rbf_kernel_a(
        a_pi, a_buf, temperature=tau_a
    )
