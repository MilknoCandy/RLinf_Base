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

"""SFT dataloader adapter for the RLT Idea1/Idea2 variants.

The RLT idea model configs keep their OpenPI settings under ``rlt_idea1`` /
``rlt_idea2`` instead of the legacy ``openpi`` block. This module exposes the
same official OpenPI SFT loader after mapping the idea block onto the fields
expected by the shared embodied-SFT worker.
"""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
    build_official_openpi_sft_dataloader,
)


def build_rlt_idea_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
    *,
    model_key: str = "rlt_idea1",
) -> tuple[Any, Any]:
    """Build the official OpenPI SFT loader for an RLT idea model config."""
    model_cfg = cfg.actor.model
    rlt_block = getattr(model_cfg, model_key)
    openpi_data = model_cfg.get("openpi_data", None)
    if openpi_data is not None:
        openpi_data = OmegaConf.to_container(openpi_data, resolve=True)

    adapter = OmegaConf.create(
        {
            "actor": {
                "micro_batch_size": cfg.actor.micro_batch_size,
                "eval_batch_size": cfg.actor.get(
                    "eval_batch_size", cfg.actor.micro_batch_size
                ),
                "model": {
                    "model_type": model_cfg.model_type,
                    "model_path": model_cfg.model_path,
                    "openpi": {
                        "config_name": rlt_block.config_name,
                    },
                    "openpi_data": openpi_data,
                },
            },
        }
    )
    return build_official_openpi_sft_dataloader(
        adapter, world_size, rank, data_paths, eval_dataset
    )
