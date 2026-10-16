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

import copy
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from rlinf.algorithms.expert import build_expert_model_config
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import resolve_full_weights
from rlinf.utils.logging import get_logger

__all__ = [
    "ResidualExpertPolicy",
    "build_expert_model_config",
    "build_residual_expert",
    "is_residual_expert_config",
    "load_residual_expert_weights",
    "predict_expert_actions",
]


def is_residual_expert_config(expert_cfg: Any) -> bool:
    """True when ``rollout.expert_model`` is an RLT residual head, not a VLA."""
    if expert_cfg is None:
        return False
    model_type = OmegaConf.select(expert_cfg, "model_type", default=None)
    return str(model_type or "").strip().lower() == "rlt_mlp_policy"


def load_residual_expert_weights(residual: torch.nn.Module, model_path: Any) -> None:
    """Load an FSDP ``full_weights.pt`` residual actor into ``residual``."""
    weights_path = resolve_full_weights(model_path)
    if weights_path is None:
        raise FileNotFoundError(
            f"residual expert checkpoint not found under {model_path}"
        )
    loaded = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(
            f"residual expert checkpoint {weights_path} is {type(loaded)}, expected dict"
        )
    state_dict = {
        key: value for key, value in loaded.items() if torch.is_tensor(value)
    }
    if not any(key.startswith(("backbone.", "actor_mean.")) for key in state_dict):
        raise ValueError(
            f"{weights_path} has no residual actor tensors "
            "(expected backbone.* / actor_mean.*). This is not an OpenPI VLA."
        )
    incompatible = residual.load_state_dict(state_dict, strict=False)
    missing_actor = [
        key
        for key in incompatible.missing_keys
        if key.startswith(("backbone.", "actor_mean."))
    ]
    if missing_actor:
        raise RuntimeError(
            f"residual expert {weights_path} missed actor weights: {missing_actor[:8]}"
        )
    get_logger().info(
        "Loaded residual expert weights from %s (missing=%d unexpected=%d)",
        weights_path,
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )


def _move_rlt_obs(rlt_obs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in rlt_obs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=False)
        else:
            moved[key] = value
    return moved


def _call_without_compile(fn, *args, **kwargs):
    """Run OpenPI extract in eager mode.

    Pi0 marks RoPE / GLU with ``@torch.compile``. On CPU that hits inductor,
    which needs ``distutils`` (absent on Python 3.12) and otherwise looks hung.
    """
    dynamo = getattr(torch, "_dynamo", None)
    disable = getattr(dynamo, "disable", None)
    if callable(disable):
        return disable(fn)(*args, **kwargs)
    return fn(*args, **kwargs)


class ResidualExpertPolicy:
    """Frozen Stage-2 residual teacher: feature VLA encodes z, residual acts.

    ZAP / AC residual weights are trained on a specific Stage-1 z. The
    student feature model (e.g. 1010) must not be reused for that encode.

    Feature encode stays on the residual device (GPU). Extract is eager so
    Pi0 ``@torch.compile`` does not hit the CPU inductor. Set
    ``offload_feature=True`` only if VRAM cannot hold both VLAs.
    """

    def __init__(
        self,
        residual: torch.nn.Module,
        feature_model: Any,
        *,
        offload_feature: bool = False,
    ):
        if not hasattr(feature_model, "extract_rlt_obs"):
            raise TypeError("residual expert feature_model must implement extract_rlt_obs")
        self.residual = residual
        self.feature_model = feature_model
        self.offload_feature = bool(offload_feature)
        self.residual.eval()
        self.residual.requires_grad_(False)
        if hasattr(self.feature_model, "eval"):
            self.feature_model.eval()
        if hasattr(self.feature_model, "requires_grad_"):
            self.feature_model.requires_grad_(False)
        if self.offload_feature and hasattr(self.feature_model, "cpu"):
            self.feature_model.cpu()

    def _extract_feature(self, env_obs) -> dict[str, Any]:
        residual_device = next(self.residual.parameters()).device
        moved = False
        if (
            self.offload_feature
            and residual_device.type == "cuda"
            and hasattr(self.feature_model, "to")
        ):
            self.feature_model.to(residual_device)
            moved = True
        try:
            extract = self.feature_model.extract_rlt_obs
            # CUDA ``@torch.compile`` is fine; CPU inductor is not.
            if residual_device.type != "cuda":
                return _call_without_compile(extract, env_obs)
            return extract(env_obs)
        finally:
            if moved and hasattr(self.feature_model, "cpu"):
                self.feature_model.cpu()

    def predict_action_batch(self, env_obs, mode="eval", **kwargs):
        rlt_obs = self._extract_feature(env_obs)
        residual_device = next(self.residual.parameters()).device
        if rlt_obs:
            sample = next(iter(rlt_obs.values()))
            if torch.is_tensor(sample) and sample.device != residual_device:
                rlt_obs = _move_rlt_obs(rlt_obs, residual_device)
        return self.residual.predict_action_batch(
            env_obs=rlt_obs,
            mode=mode,
            **kwargs,
        )

    def eval(self):
        self.residual.eval()
        if hasattr(self.feature_model, "eval"):
            self.feature_model.eval()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        self.residual.requires_grad_(requires_grad)
        if hasattr(self.feature_model, "requires_grad_"):
            self.feature_model.requires_grad_(requires_grad)
        return self

    def to(self, device):
        self.residual.to(device)
        if self.offload_feature:
            if hasattr(self.feature_model, "cpu"):
                self.feature_model.cpu()
        elif hasattr(self.feature_model, "to"):
            self.feature_model.to(device)
        return self


def build_residual_expert(cfg: Any) -> ResidualExpertPolicy:
    """Build a ZAP/AC residual teacher plus the Stage-1 that produced its z."""
    from rlinf.models import get_model

    expert_cfg = cfg.rollout.expert_model
    if not is_residual_expert_config(expert_cfg):
        raise ValueError("build_residual_expert requires expert_model.model_type=rlt_mlp_policy")
    rollout_cfg = cfg.rollout
    feature_over = OmegaConf.select(
        rollout_cfg, "expert_feature_model", default=None
    )
    if feature_over is None:
        raise ValueError(
            "residual expert requires rollout.expert_feature_model "
            "(the Stage-1 whose z the residual was trained on)."
        )
    feature_base = OmegaConf.select(rollout_cfg, "rlt_feature_model", default=None)
    if feature_base is None:
        raise ValueError("residual expert requires rollout.rlt_feature_model as the architecture base")

    residual = get_model(copy.deepcopy(expert_cfg))
    load_residual_expert_weights(residual, expert_cfg.model_path)

    feature_cfg = copy.deepcopy(feature_base)
    offload_feature = bool(rollout_cfg.get("expert_feature_offload", False))
    with open_dict(feature_cfg):
        for key, value in feature_over.items():
            feature_cfg[key] = value
        if offload_feature:
            feature_cfg.load_to_device = False
        else:
            feature_cfg.load_to_device = True
    feature_model = get_model(feature_cfg)
    if offload_feature and hasattr(feature_model, "cpu"):
        feature_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ResidualExpertPolicy(
        residual, feature_model, offload_feature=offload_feature
    )


def predict_expert_actions(
    expert_model: Any,
    env_obs: dict[str, Any],
    *,
    chunk_len: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad():
        expert_actions, _ = expert_model.predict_action_batch(
            env_obs=env_obs,
            mode="eval",
            compute_values=False,
        )
    if isinstance(expert_actions, np.ndarray):
        expert_actions = torch.from_numpy(expert_actions)
    if expert_actions.dim() == 2:
        expert_actions = expert_actions.reshape(expert_actions.shape[0], -1, action_dim)
    return expert_actions[:, :chunk_len, :action_dim].to(device=device, dtype=dtype)
