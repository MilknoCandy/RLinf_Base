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
            mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
            mlp_num_hidden_layers=cfg.get("mlp_num_hidden_layers", 3),
            frozen_action_dims=cfg.get("frozen_action_dims", None),
            tsac_d_model=cfg.get("tsac_d_model", 512),
            tsac_num_layers=cfg.get("tsac_num_layers", 2),
            tsac_num_heads=cfg.get("tsac_num_heads", 8),
            tsac_max_action_len=cfg.get("tsac_max_action_len", 25),
            action_extract=cfg.get("action_extract", "sample"),
            qc_num_samples=cfg.get("qc_num_samples", 8),
            qc_include_ref_chunk=cfg.get("qc_include_ref_chunk", True),
            qc_include_actor_mean=cfg.get("qc_include_actor_mean", True),
            chunk_critic_steps=cfg.get("chunk_critic_steps", None),
            scale_critic_steps=cfg.get("scale_critic_steps", None),
            add_scale_value_heads=cfg.get("add_scale_value_heads", False),
            aqc_gamma=cfg.get("aqc_gamma", 0.99),
            residual_actor=cfg.get("residual_actor", False),
        )
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
            frozen_action_dims=cfg.get("frozen_action_dims", None),
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
