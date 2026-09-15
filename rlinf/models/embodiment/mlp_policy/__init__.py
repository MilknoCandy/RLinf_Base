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
    from rlinf.algorithms.rlt.a2_stm import A2STMConfig
    from rlinf.models.embodiment.mlp_policy.iql_mlp_policy import IQLMLPPolicy
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import RLTTD3MLPPolicy

    def _build_a2_stm_config_from_model_cfg(model_cfg: DictConfig) -> A2STMConfig | None:
        block = model_cfg.get("a2_stm", None)
        if block is None or not bool(block.get("enable", False)):
            return None
        action_dim = int(model_cfg.action_dim)
        num_chunks = int(model_cfg.num_action_chunks)
        flat_action = int(block.get("action_dim", action_dim * num_chunks))
        return A2STMConfig(
            enable=True,
            window_size=int(block.get("window_size", 64)),
            pos_capacity=int(block.get("pos_capacity", 2048)),
            neg_capacity=int(block.get("neg_capacity", 2048)),
            top_k=int(block.get("top_k", 16)),
            temperature=float(block.get("temperature", 0.07)),
            only_critical=bool(block.get("only_critical", True)),
            fail_tail_steps=int(block.get("fail_tail_steps", 16)),
            write_on_eval=bool(block.get("write_on_eval", False)),
            share_train_bank_on_eval=bool(block.get("share_train_bank_on_eval", True)),
            z_dim=int(model_cfg.z_dim),
            action_dim=flat_action,
            hidden_dim=int(block.get("hidden_dim", 256)),
            gate_init_bias=float(block.get("gate_init_bias", -2.0)),
        )

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
            a2_stm_config=_build_a2_stm_config_from_model_cfg(cfg),
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
