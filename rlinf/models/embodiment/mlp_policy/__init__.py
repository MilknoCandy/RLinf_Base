# Copyright 2025 The RLinf Authors.
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

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from rlinf.models.embodiment.mlp_policy.iql_mlp_policy import IQLMLPPolicy
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import RLTTD3MLPPolicy

    iql_config = cfg.get("iql_config", None)
    if cfg.model_type == "rlt_mlp_policy":
        model = RLTMLPPolicy(
            z_dim=cfg.z_dim,
            proprio_dim=cfg.proprio_dim,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            ref_num_action_chunks=cfg.get(
                "ref_num_action_chunks", cfg.num_action_chunks
            ),
            add_q_head=cfg.get("add_q_head", True),
            q_head_type=cfg.get("q_head_type", "default"),
            fixed_std=cfg.get("fixed_std", 0.002),
        )
        b1_block = cfg.get("b1_dynamic", None)
        if b1_block is not None and bool(b1_block.get("enable", False)):
            from rlinf.algorithms.rlt.b1_dynamic import (
                B1DynamicConfig,
                B1DynamicModule,
            )

            horizons = b1_block.get("pred_horizons", [1, 4, 8])
            weights = b1_block.get("pred_weights", [1.0, 0.5, 0.25])
            b1_cfg = B1DynamicConfig(
                enable=True,
                z_dim=int(b1_block.get("z_dim", cfg.z_dim)),
                feature_dim=int(b1_block.get("feature_dim", 256)),
                num_tokens=int(b1_block.get("num_tokens", 16)),
                num_layers=int(b1_block.get("num_layers", 2)),
                num_heads=int(b1_block.get("num_heads", 4)),
                mlp_ratio=float(b1_block.get("mlp_ratio", 4.0)),
                dropout=float(b1_block.get("dropout", 0.0)),
                only_critical=bool(b1_block.get("only_critical", True)),
                lambda_pred=float(b1_block.get("lambda_pred", 0.1)),
                pred_horizons=tuple(int(x) for x in list(horizons)),
                pred_weights=tuple(float(x) for x in list(weights)),
            )
            model.add_module("b1_dynamic", B1DynamicModule(b1_cfg))
    elif cfg.model_type == "rlt_td3_mlp_policy":
        model = RLTTD3MLPPolicy(
            z_dim=cfg.z_dim,
            proprio_dim=cfg.proprio_dim,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            ref_num_action_chunks=cfg.get(
                "ref_num_action_chunks", cfg.num_action_chunks
            ),
            add_q_head=cfg.get("add_q_head", True),
            q_head_type=cfg.get("q_head_type", "default"),
            mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
            mlp_num_hidden_layers=cfg.get("mlp_num_hidden_layers", 2),
            actor_noise_sigma=cfg.get("actor_noise_sigma", 0.1),
            ref_action_dropout=cfg.get("ref_action_dropout", 0.0),
        )
    elif iql_config is not None:
        model = IQLMLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
            add_q_head=cfg.get("add_q_head", False),
            q_head_type=cfg.get("q_head_type", "default"),
        )
        model.configure_iql(iql_config)
    else:
        model = MLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
            add_q_head=cfg.get("add_q_head", False),
            q_head_type=cfg.get("q_head_type", "default"),
        )

    return model
