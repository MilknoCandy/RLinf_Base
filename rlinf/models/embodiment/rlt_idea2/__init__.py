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

"""Factory for RLT Idea2 models (image-only learnable-token-in-VLM).

Usage in YAML::

    model:
      model_type: rlt_idea2
      model_path: /path/to/base/checkpoint
      rlt_idea2:
        task: sft           # or eval for Stage2 feature extraction
        use_rlt: true
        rlt_alpha: 1.0
        config_name: pi05_maniskill
        model_action_dim: 32
        paligemma_variant: gemma_2b
        action_expert_variant: gemma_300m
"""

from __future__ import annotations

from typing import Any

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    load_base_safetensors,
    load_full_wrapper_weights,
    resolve_full_weights,
    resolve_model_safetensors,
)
from rlinf.models.embodiment.rlt_idea1.rlt_idea1_config import (
    build_rlt_idea1_config,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


def _resolve_data_kwargs(cfg: Any) -> Any:
    from omegaconf import OmegaConf

    data_kwargs = OmegaConf.select(cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    return data_kwargs


def _build_eval_model(
    cfg: Any,
    model_cfg: Any,
    model: Any,
    *,
    num_steps: int,
    action_chunk: int,
    action_env_dim: int,
) -> Any:
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_rlinf.transforms_pipeline import (
        build_openpi_transforms,
    )
    from rlinf.models.embodiment.rlt_idea2.rlt_idea2_eval_model import (
        RltIdea2EvalModel,
    )

    config_name = str(OmegaConf.select(model_cfg, "config_name", default=""))
    if not config_name:
        raise ValueError(
            "actor.model.rlt_idea2.config_name is required for task='eval'."
        )

    input_transforms, output_transforms = build_openpi_transforms(
        cfg.model_path, config_name, data_kwargs=_resolve_data_kwargs(cfg)
    )

    eval_model = RltIdea2EvalModel(
        model,
        num_steps=num_steps,
        action_env_dim=action_env_dim,
        action_chunk=action_chunk,
        config_name=config_name,
        state_indices=OmegaConf.select(model_cfg, "state_indices", default=None),
        rlt_cfg=build_rlt_idea1_config(model_cfg),
    )
    eval_model.setup_wrappers(input_transforms, output_transforms)
    return eval_model


def _build_sft_model(
    model_cfg: Any,
    model: Any,
    *,
    num_steps: int,
    action_env_dim: int,
) -> Any:
    from rlinf.models.embodiment.rlt_idea2.rlt_idea2_sft_model import (
        RltIdea2SFTModel,
    )

    return RltIdea2SFTModel(
        model,
        num_steps=num_steps,
        action_env_dim=action_env_dim,
        rlt_cfg=build_rlt_idea1_config(model_cfg),
    )


def get_model(cfg: Any, torch_dtype: Any = None) -> Any:
    """Build a Pi0/Pi0.5 model with the RLT Idea2 objective."""
    import pathlib

    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config

    model_cfg = cfg.rlt_idea2
    pi05 = bool(OmegaConf.select(cfg, "pi05", default=True))
    target_dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )

    model_path = pathlib.Path(cfg.model_path).expanduser()
    safetensors_path = resolve_model_safetensors(model_path)
    full_weights_path = resolve_full_weights(model_path)
    if safetensors_path is None and full_weights_path is None:
        raise FileNotFoundError(
            f"rlt_idea2 checkpoint not found at {model_path}."
        )

    pi0_kwargs = {
        "pi05": pi05,
        "action_horizon": int(cfg.num_action_chunks),
        "action_dim": int(model_cfg.model_action_dim),
        "paligemma_variant": str(model_cfg.paligemma_variant),
        "action_expert_variant": str(model_cfg.action_expert_variant),
        "dtype": "bfloat16",
        "pcd": False,
    }
    discrete_state_input = OmegaConf.select(
        model_cfg, "discrete_state_input", default=None
    )
    if discrete_state_input is not None:
        pi0_kwargs["discrete_state_input"] = bool(discrete_state_input)
    max_token_len = OmegaConf.select(model_cfg, "max_token_len", default=None)
    if max_token_len is not None:
        pi0_kwargs["max_token_len"] = int(max_token_len)

    pi0_config = Pi0Config(**pi0_kwargs)
    model = pi0_config.create()
    if safetensors_path is not None and full_weights_path is None:
        load_base_safetensors(model, safetensors_path)
    n_params = sum(param.numel() for param in model.parameters())
    if target_dtype is not None:
        model = model.to(target_dtype)

    num_steps = int(cfg.num_steps)
    action_chunk = int(cfg.num_action_chunks)
    action_env_dim = int(cfg.action_dim)

    task = OmegaConf.select(model_cfg, "task", default=None)
    if task is None:
        raise ValueError(
            "actor.model.rlt_idea2.task is required: set to 'sft' or 'eval'."
        )
    task = str(task).lower()

    if task == "eval":
        wrapper = _build_eval_model(
            cfg,
            model_cfg,
            model,
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
        )
    elif task == "sft":
        wrapper = _build_sft_model(
            model_cfg,
            model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
        )
    else:
        raise ValueError(
            f"actor.model.rlt_idea2.task={task!r} not supported; "
            "use 'sft' or 'eval'."
        )

    if full_weights_path is not None:
        load_full_wrapper_weights(
            wrapper,
            full_weights_path,
            expect_rlt=bool(OmegaConf.select(model_cfg, "use_rlt", default=False)),
        )

    source = full_weights_path if full_weights_path is not None else safetensors_path
    logger.info(
        "rlt_idea2[%s]: loaded %s (%.2fB params) from %s",
        task,
        pi0_config,
        n_params / 1e9,
        source,
    )
    return wrapper
