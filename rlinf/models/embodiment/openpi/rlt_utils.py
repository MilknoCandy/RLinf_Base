# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLT configuration and checkpoint helpers for OpenPI_RLinf."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

from rlinf.utils.logging import get_logger

logger = get_logger()

FULL_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)

_FSDP_WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_orig_mod.",
    "module.",
)

_BARE_PI0_PREFIXES = (
    "llm.",
    "img.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
    "state_proj.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
    "pointnet.",
)

_OLD_OPENPI_PREFIX = "paligemma_with_expert."


_ORIGINAL_ENCODER_TYPES = frozenset(
    {
        "append_self_attention",
        "token",
        "original",
        "self_attention",
    }
)
_CROSS_ENCODER_TYPES = frozenset(
    {
        "cross_attention",
        "cross",
        "memory",
        "mem",
    }
)
_OLD_ENCODER_MARKERS = (
    "encoder.rl_token_embed",
    "encoder.rl_token_pos_enc",
)
_NEW_ENCODER_MARKERS = (
    "encoder.z_init",
    "encoder.out_norm",
)


@dataclasses.dataclass(frozen=True)
class OpenPiPytorchRLTConfig:
    """RLT-token knobs shared by the SFT and eval wrappers."""

    use_rlt: bool = False
    rlt_alpha: float = 1.0
    rlt_input_dim: int = 2048
    rlt_embed_dim: int = 2048
    rlt_prefix_seq_len: int = 768
    rlt_num_layers: int = 2
    rlt_num_heads: int = 8
    rlt_mlp_ratio: float = 4.0
    rlt_image_only: bool = True
    rlt_use_mask: bool = False
    rlt_recon_topk_ratio: float = 1.0
    rlt_mem_len_min: int = 1
    rlt_mem_len_max: int = 1
    rlt_mem_scheme: int = 1
    rlt_encoder_type: str = ""
    rlt_return_prefix: bool = False


def _collect_state_dict_keys(obj: Any, *, depth: int = 0) -> list[str]:
    if not isinstance(obj, dict) or depth > 4:
        return []
    keys = [_normalize_wrapper_key(str(key)) for key in obj]
    for value in obj.values():
        if isinstance(value, dict):
            keys.extend(_collect_state_dict_keys(value, depth=depth + 1))
    return keys


def infer_rlt_encoder_kind(state_dict: Any) -> str | None:
    """Return ``token``, ``cross``, or ``None`` from checkpoint parameter names.

    Pre-memory Stage-1 (ManiSkill 20260809) stores ``encoder.rl_token_embed``.
    The later cross-attention encoder stores ``encoder.z_init`` / ``out_norm``.
    """
    keys = _collect_state_dict_keys(state_dict)
    if not keys:
        return None

    def _has(marker: str) -> bool:
        return any(marker in key for key in keys)

    has_old = any(_has(marker) for marker in _OLD_ENCODER_MARKERS)
    has_new = any(_has(marker) for marker in _NEW_ENCODER_MARKERS)
    if has_old and not has_new:
        return "token"
    if has_new and not has_old:
        return "cross"
    return None


def resolve_rlt_mem_scheme(
    *,
    encoder_type: str = "",
    mem_scheme: int = 1,
    mem_len_max: int = 1,
    return_prefix: bool = False,
    state_dict: Any = None,
) -> int:
    """Pick ``0`` (original token encoder) or ``1``/``2`` (cross-attention).

    Checkpoint keys win so ManiSkill Stage-1 loads the old encoder even when
    YAML still says ``append_self_attention``, and bowl Stage-1 stays on the
    cross encoder even though that same unused YAML key is present.
    """
    yaml_scheme = int(mem_scheme)
    kind = infer_rlt_encoder_kind(state_dict)
    if kind == "token":
        return 0
    if kind == "cross":
        return yaml_scheme if yaml_scheme in (1, 2) else 1

    encoder_type = str(encoder_type or "").strip().lower()
    if int(mem_len_max) > 1 or bool(return_prefix):
        return yaml_scheme if yaml_scheme in (1, 2) else 1
    if encoder_type in _CROSS_ENCODER_TYPES:
        return yaml_scheme if yaml_scheme in (1, 2) else 1
    if encoder_type in _ORIGINAL_ENCODER_TYPES or yaml_scheme == 0:
        return 0
    return yaml_scheme if yaml_scheme in (0, 1, 2) else 1


def bind_rlt_mem_scheme(config_obj: Any, state_dict: Any = None) -> int:
    """Write the resolved scheme onto a config object and return it."""
    raw_scheme = getattr(config_obj, "rlt_mem_scheme", 1)
    raw_mem_len = getattr(config_obj, "rlt_mem_len_max", 1)
    scheme = resolve_rlt_mem_scheme(
        encoder_type=getattr(config_obj, "rlt_encoder_type", "") or "",
        mem_scheme=1 if raw_scheme is None else int(raw_scheme),
        mem_len_max=1 if raw_mem_len is None else int(raw_mem_len),
        return_prefix=bool(getattr(config_obj, "rlt_return_prefix", False)),
        state_dict=state_dict,
    )
    encoder_type = "append_self_attention" if scheme == 0 else "cross_attention"
    try:
        config_obj.__dict__["rlt_mem_scheme"] = scheme
        config_obj.__dict__["rlt_encoder_type"] = encoder_type
    except Exception:
        setattr(config_obj, "rlt_mem_scheme", scheme)
        setattr(config_obj, "rlt_encoder_type", encoder_type)
    return scheme


def build_rlt_config(model_cfg: Any, state_dict: Any = None) -> OpenPiPytorchRLTConfig:
    """Build optional RLT-token config from ``actor.model.openpi``."""
    from omegaconf import OmegaConf

    encoder_type = str(
        OmegaConf.select(model_cfg, "rlt_encoder_type", default="") or ""
    )
    mem_scheme = resolve_rlt_mem_scheme(
        encoder_type=encoder_type,
        mem_scheme=int(OmegaConf.select(model_cfg, "rlt_mem_scheme", default=1)),
        mem_len_max=int(OmegaConf.select(model_cfg, "rlt_mem_len_max", default=1)),
        return_prefix=bool(
            OmegaConf.select(model_cfg, "rlt_return_prefix", default=False)
        ),
        state_dict=state_dict,
    )
    return OpenPiPytorchRLTConfig(
        use_rlt=bool(OmegaConf.select(model_cfg, "use_rlt", default=False)),
        rlt_alpha=float(OmegaConf.select(model_cfg, "rlt_alpha", default=1.0)),
        rlt_input_dim=int(OmegaConf.select(model_cfg, "rlt_input_dim", default=2048)),
        rlt_embed_dim=int(OmegaConf.select(model_cfg, "rlt_embed_dim", default=2048)),
        rlt_prefix_seq_len=int(
            OmegaConf.select(model_cfg, "rlt_prefix_seq_len", default=768)
        ),
        rlt_num_layers=int(OmegaConf.select(model_cfg, "rlt_num_layers", default=2)),
        rlt_num_heads=int(OmegaConf.select(model_cfg, "rlt_num_heads", default=8)),
        rlt_mlp_ratio=float(OmegaConf.select(model_cfg, "rlt_mlp_ratio", default=4.0)),
        rlt_image_only=bool(
            OmegaConf.select(model_cfg, "rlt_image_only", default=True)
        ),
        rlt_use_mask=bool(OmegaConf.select(model_cfg, "rlt_use_mask", default=False)),
        rlt_recon_topk_ratio=float(
            OmegaConf.select(model_cfg, "rlt_recon_topk_ratio", default=1.0)
        ),
        rlt_mem_len_min=int(
            OmegaConf.select(model_cfg, "rlt_mem_len_min", default=1)
        ),
        rlt_mem_len_max=int(
            OmegaConf.select(model_cfg, "rlt_mem_len_max", default=1)
        ),
        rlt_mem_scheme=mem_scheme,
        rlt_encoder_type=(
            "append_self_attention" if mem_scheme == 0 else "cross_attention"
        ),
        rlt_return_prefix=bool(
            OmegaConf.select(model_cfg, "rlt_return_prefix", default=False)
        ),
    )


def resolve_model_safetensors(model_path: Any) -> pathlib.Path | None:
    """Resolve a base ``model.safetensors`` checkpoint path."""
    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".safetensors"):
        return path
    weights_path = path / "model.safetensors"
    return weights_path if weights_path.exists() else None


def resolve_full_weights(model_path: Any) -> pathlib.Path | None:
    """Resolve an RLinf FSDP ``full_weights.pt`` checkpoint path."""
    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".pt"):
        return path
    for rel_path in FULL_WEIGHTS_CANDIDATES:
        candidate = path / rel_path
        if candidate.exists():
            return candidate
    return None


def _normalize_wrapper_key(key: str) -> str:
    while True:
        for prefix in _FSDP_WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        else:
            return key


def _normalize_wrapper_state_dict(state_dict):
    normalized = {}
    for key, tensor in state_dict.items():
        key = _normalize_wrapper_key(key)
        if key in normalized:
            raise ValueError(
                f"Duplicate checkpoint key after prefix normalization: {key!r}."
            )
        normalized[key] = tensor

    has_wrapper_key = any(
        key.startswith("model.")
        or key.startswith("rlt_module.")
        or key.startswith("mem.")
        for key in normalized
    )
    if has_wrapper_key:
        return normalized

    if any(key.startswith(_BARE_PI0_PREFIXES) for key in normalized):
        return {f"model.{key}": tensor for key, tensor in normalized.items()}
    return normalized


def load_full_wrapper_weights(
    wrapper, weights_path, *, expect_rlt: bool, loaded: Any = None
) -> None:
    """Load an RLinf full-wrapper checkpoint into an OpenPI_RLinf wrapper."""
    import torch

    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    if loaded is None:
        loaded = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = _normalize_wrapper_state_dict(as_state_dict(loaded))
    has_rlt = any(key.startswith("rlt_module.") for key in state_dict)
    has_mem = any(key.startswith("mem.") for key in state_dict)
    if expect_rlt and not has_rlt and not has_mem:
        raise ValueError(
            "openpi_rlinf RLT checkpoint has no rlt_module.* or mem.* weights. "
            "Stage2 must consume a Stage1 checkpoint trained with openpi.use_rlt=True."
        )

    incompatible = wrapper.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    matched = len(state_dict) - len(unexpected)
    if matched <= 0:
        raise RuntimeError(
            f"No tensors from {weights_path} matched the openpi_rlinf wrapper. "
            "This usually means the checkpoint is still in the legacy official "
            "OpenPI PyTorch key layout."
        )
    if expect_rlt and has_rlt and any(key.startswith("rlt_module.") for key in missing):
        raise RuntimeError(
            f"RLT checkpoint {weights_path} did not load all rlt_module weights; "
            f"missing={missing[:8]}"
        )

    if missing or unexpected:
        logger.warning(
            "openpi_rlinf: loaded wrapper checkpoint %s with strict=False "
            "(matched=%d missing=%d unexpected=%d)",
            weights_path,
            matched,
            len(missing),
            len(unexpected),
        )
    else:
        logger.info(
            "openpi_rlinf: loaded full wrapper checkpoint from %s", weights_path
        )


def load_base_safetensors(model, safetensors_path) -> None:
    """Load a base checkpoint, accepting new and legacy OpenPI layouts."""
    import safetensors.torch

    state_dict = safetensors.torch.load_file(str(safetensors_path), device="cpu")
    if any(key.startswith(_OLD_OPENPI_PREFIX) for key in state_dict):
        from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi_rlinf import (
            old_to_new_state_dict,
        )

        state_dict = old_to_new_state_dict(state_dict)
        logger.info(
            "openpi_rlinf: converted OpenPI PyTorch checkpoint keys from %s in memory",
            safetensors_path,
        )
    model.load_state_dict(state_dict, strict=True)
