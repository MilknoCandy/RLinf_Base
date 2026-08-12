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

"""Model builders for the VLM-internal RLT (openpi_rlinf_vlm) factory."""

from __future__ import annotations

import logging

from rlinf.models.embodiment.openpi_rlinf_vlm.utils.rlt_config import (
    build_rlt_config_vlm,
)

logger = logging.getLogger(__name__)


def _resolve_data_kwargs(cfg):
    from omegaconf import OmegaConf

    data_kwargs = OmegaConf.select(cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    return data_kwargs


def _build_eval_model_vlm(
    cfg,
    model_cfg,
    model,
    *,
    num_steps,
    action_chunk,
    action_env_dim,
):
    """Build the VLM-internal eval variant with openpi transforms pipeline."""
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_rlinf.transforms_pipeline import (
        build_openpi_transforms,
    )
    from rlinf.models.embodiment.openpi_rlinf_vlm.eval_action_model import (
        OpenPiPytorchEvalActionModelVLM,
    )

    config_name = str(OmegaConf.select(model_cfg, "config_name", default=""))
    if not config_name:
        raise ValueError(
            "actor.model.openpi_vlm.config_name is required for task='eval'."
        )

    input_transforms, output_transforms = build_openpi_transforms(
        cfg.model_path, config_name, data_kwargs=_resolve_data_kwargs(cfg)
    )

    eval_model = OpenPiPytorchEvalActionModelVLM(
        model,
        num_steps=num_steps,
        action_env_dim=action_env_dim,
        action_chunk=action_chunk,
        config_name=config_name,
        state_indices=OmegaConf.select(model_cfg, "state_indices", default=None),
        rlt_cfg=build_rlt_config_vlm(model_cfg),
    )
    eval_model.setup_wrappers(input_transforms, output_transforms)
    return eval_model


def _build_sft_model_vlm(
    model_cfg,
    model,
    *,
    num_steps,
    action_env_dim,
):
    """Build the VLM-internal SFT variant (no transforms pipeline)."""
    from rlinf.models.embodiment.openpi_rlinf_vlm.sft_action_model import (
        OpenPiPytorchSFTActionModelVLM,
    )

    return OpenPiPytorchSFTActionModelVLM(
        model,
        num_steps=num_steps,
        action_env_dim=action_env_dim,
        rlt_cfg=build_rlt_config_vlm(model_cfg),
    )