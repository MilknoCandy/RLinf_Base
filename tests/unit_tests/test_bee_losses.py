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

"""Unit tests for Bee Mahalanobis constraint and Correction Model schedule."""

from __future__ import annotations

import torch

from rlinf.algorithms.bee import (
    BeeCorrectionModel,
    BeeMultiplier,
    bee_actor_loss,
    bee_multiplier_loss,
    correction_nll_loss,
    correction_update_steps,
    mahalanobis_rho,
    should_update_correction,
)


def test_mahalanobis_weights_low_variance_dims_more():
    policy = torch.tensor([[1.0, 1.0]])
    center = torch.tensor([[0.0, 0.0]])
    # Dim 0 is consistent (small σ); dim 1 is noisy.
    sigma = torch.tensor([[0.1, 1.0]])
    rho = mahalanobis_rho(policy, center, sigma)
    # Equal squared errors; dim0 contributes 100x more after Σ^{-1}.
    assert rho.shape == (1,)
    assert float(rho.item()) > 50.0


def test_correction_nll_and_schedule():
    assert should_update_correction(99, update_interval_m=100) is False
    assert should_update_correction(100, update_interval_m=100) is True
    assert correction_update_steps(update_interval_m=100, update_to_data_n=0.12) == 12

    model = BeeCorrectionModel(state_dim=8, action_dim=4, hidden_dim=32, num_hidden_layers=2)
    state = torch.randn(5, 8)
    proposal = torch.randn(5, 4)
    residual = torch.randn(5, 4) * 0.1
    mu, sigma = model(state, proposal)
    assert mu.shape == (5, 4)
    assert sigma.shape == (5, 4)
    assert torch.all(sigma >= 0.02)
    loss = correction_nll_loss(residual, mu, sigma)
    assert loss.ndim == 0
    loss.backward()


def test_actor_and_multiplier_objectives():
    q = torch.tensor([1.0, 2.0])
    policy = torch.randn(2, 6)
    proposal = torch.randn(2, 6)
    rho = torch.tensor([0.1, 0.5])
    lam = torch.tensor([0.2, 0.8], requires_grad=True)
    actor_loss, actor_metrics = bee_actor_loss(
        q, policy, proposal, rho, lam.detach(), anchor_weight=1.0
    )
    assert "bee/actor_loss" in actor_metrics
    assert actor_loss.ndim == 0

    mult = BeeMultiplier(state_dim=8, hidden_dim=16, num_hidden_layers=1)
    state = torch.randn(2, 8)
    lam2 = mult(state)
    mult_loss, mult_metrics = bee_multiplier_loss(lam2, rho, epsilon=0.3)
    assert "bee/violation_rate" in mult_metrics
    mult_loss.backward()
    assert any(p.grad is not None for p in mult.parameters())
