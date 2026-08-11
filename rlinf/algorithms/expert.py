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

from omegaconf import DictConfig, OmegaConf, open_dict


def build_expert_model_config(
    cfg: Any,
    model_cfg: Any,
    *,
    rlt_feature_model_config: Any | None = None,
):
    """Build a teacher/expert model config from rollout.expert_model overrides.

    Performs a *deep* merge for nested DictConfig values (e.g. ``openpi.*``)
    so that expert_model only needs to specify the keys it overrides; all
    other keys from the base config (rlt_feature_model or model_cfg) are
    preserved.
    """
    expert_cfg = cfg.rollout.expert_model
    expert_model_config = copy.deepcopy(
        rlt_feature_model_config if rlt_feature_model_config is not None else model_cfg
    )

    with open_dict(expert_model_config):
        for key, value in expert_cfg.items():
            if (
                key in expert_model_config
                and isinstance(expert_model_config[key], DictConfig)
                and isinstance(value, DictConfig)
            ):
                # Deep-merge nested configs: expert_model.openpi.use_rlt
                # overrides rlt_feature_model.openpi.use_rlt while keeping
                # config_name, num_images_in_input, etc.
                expert_model_config[key] = OmegaConf.merge(
                    expert_model_config[key], value
                )
            else:
                expert_model_config[key] = value

    return expert_model_config
