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

"""Unit tests for B1 dynamic encoder (no full RLinf runtime required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_b1_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "rlinf"
        / "algorithms"
        / "rlt"
        / "b1_dynamic.py"
    )
    sys.modules.pop("b1_dynamic_under_test", None)
    spec = importlib.util.spec_from_file_location("b1_dynamic_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["b1_dynamic_under_test"] = module
    spec.loader.exec_module(module)
    return module


_b1 = _load_b1_module()
B1DynamicConfig = _b1.B1DynamicConfig
B1DynamicModule = _b1.B1DynamicModule
B1DynamicRuntime = _b1.B1DynamicRuntime
compute_b1_pred_loss = _b1.compute_b1_pred_loss


def _cfg(**overrides):
    cfg = B1DynamicConfig(
        enable=True,
        z_dim=32,
        feature_dim=16,
        num_tokens=4,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
        only_critical=True,
        lambda_pred=0.1,
        pred_horizons=(1, 4, 8),
        pred_weights=(1.0, 0.5, 0.25),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_b1_update_shapes_and_critical_gate():
    cfg = _cfg()
    module = B1DynamicModule(cfg)
    runtime = B1DynamicRuntime(config=cfg)
    z = torch.randn(3, cfg.z_dim)
    critical = torch.tensor([True, False, True])
    out = runtime.step(
        module=module,
        z_app=z,
        critical_mask=critical,
        dones=None,
    )
    assert out["z_rl"].shape == z.shape
    assert out["b1_h"].shape == (3, cfg.num_tokens, cfg.feature_dim)
    assert torch.allclose(out["z_rl"][1], z[1])
    assert out["b1_updated"].tolist() == [True, False, True]


def test_b1_reset_on_done():
    cfg = _cfg(only_critical=False)
    module = B1DynamicModule(cfg)
    runtime = B1DynamicRuntime(config=cfg)
    z = torch.randn(2, cfg.z_dim)
    runtime.step(module=module, z_app=z, critical_mask=None, dones=None)
    assert runtime.h is not None
    prev = runtime.h.clone()
    out = runtime.step(
        module=module,
        z_app=z,
        critical_mask=None,
        dones=torch.tensor([True, False]),
    )
    # Done env is cleared before the new-episode update, so h_prev is zero.
    assert torch.allclose(out["b1_h_prev"][0], torch.zeros_like(out["b1_h_prev"][0]))
    # Non-done env keeps carrying previous state into h_prev.
    assert torch.allclose(out["b1_h_prev"][1], prev[1])
    assert not torch.allclose(out["b1_h"][0], torch.zeros_like(out["b1_h"][0]))


def test_b1_pred_loss_backprops_to_encoder():
    cfg = _cfg()
    module = B1DynamicModule(cfg)
    z_app = torch.randn(4, cfg.z_dim)
    h_prev = module.init_state(4, z_app.device)
    future_z = {
        1: torch.randn(4, cfg.z_dim),
        4: torch.randn(4, cfg.z_dim),
        8: torch.randn(4, cfg.z_dim),
    }
    future_mask = {
        1: torch.tensor([True, True, False, True]),
        4: torch.tensor([True, False, False, True]),
        8: torch.tensor([False, False, False, True]),
    }
    loss, metrics = compute_b1_pred_loss(
        module,
        z_app=z_app,
        h_prev=h_prev,
        future_z=future_z,
        future_mask=future_mask,
        config=cfg,
        update_mask=torch.ones(4, dtype=torch.bool),
    )
    assert loss.ndim == 0
    assert "b1_dynamic/pred_loss" in metrics
    loss.backward()
    grads = [p.grad for p in module.encoder.parameters() if p.grad is not None]
    assert len(grads) > 0
