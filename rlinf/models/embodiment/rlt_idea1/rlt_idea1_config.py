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

"""Configuration for the RLT Idea1 (learnable-token-in-VLM) variant.

Idea1 removes the post-hoc transformer encoder of the standard RLT design.
Instead, a single learnable token is injected into the VLM prefix. The VLM's
own attention layers compress the observation into the token's output
position, and the same decoder as the standard RLT reconstructs the prefix.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class RltIdea1Config:
    """RLT Idea1 hyper-parameters shared by SFT and eval wrappers.

    Fields mirror ``OpenPiPytorchRLTConfig`` so existing RLT configs can be
    reused, but there is no encoder; ``rlt_input_dim`` is the VLM prefix hidden
    width and ``rlt_embed_dim`` is the decoder token width.
    """

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
    # Route B: keep the VLM trainable and use a joint
    # ``rlt_loss + rlt_alpha * vla_loss`` objective.
    freeze_vlm: bool = False
    # The learnable-token output is compressed into this low-dimensional,
    # normalized latent before it is (a) expanded for reconstruction and
    # (b) handed to the Stage2 RL head as ``z_rl``.
    rlt_latent_dim: int = 512
    rlt_z_norm: bool = True
    rlt_z_l2_weight: float = 0.0


def build_rlt_idea1_config(model_cfg: Any) -> RltIdea1Config:
    """Build ``RltIdea1Config`` from a Hydra ``actor.model.rlt_idea1`` block."""
    from omegaconf import OmegaConf

    return RltIdea1Config(
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
        freeze_vlm=bool(OmegaConf.select(model_cfg, "freeze_vlm", default=False)),
        rlt_latent_dim=int(
            OmegaConf.select(model_cfg, "rlt_latent_dim", default=512)
        ),
        rlt_z_norm=bool(OmegaConf.select(model_cfg, "rlt_z_norm", default=True)),
        rlt_z_l2_weight=float(
            OmegaConf.select(model_cfg, "rlt_z_l2_weight", default=0.0)
        ),
    )
