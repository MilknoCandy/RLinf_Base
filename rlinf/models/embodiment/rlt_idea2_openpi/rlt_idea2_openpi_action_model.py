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

"""Official-OpenPI-backed RLT Idea2 (image-only RL token)."""

from __future__ import annotations

import torch

from rlinf.models.embodiment.rlt_idea1_openpi.rlt_idea1_openpi_action_model import (
    OpenPiIdea1ActionModel,
)


class OpenPiIdea2ActionModel(OpenPiIdea1ActionModel):
    """RL token attends only to image-token keys."""

    def _apply_rl_token_attention(
        self,
        att_2d_masks: torch.Tensor,
        prefix_len: int,
        num_image_tokens: int,
    ) -> torch.Tensor:
        rl_query_idx = prefix_len - 1
        key_is_image = torch.zeros(
            att_2d_masks.shape[2], dtype=torch.bool, device=att_2d_masks.device
        )
        key_is_image[:num_image_tokens] = True

        new_mask = att_2d_masks.clone()
        new_mask[:, rl_query_idx, :] = (
            att_2d_masks[:, rl_query_idx, :] & key_is_image.unsqueeze(0)
        )
        return new_mask

    def _select_decoder_target(
        self,
        prefix_output: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        lang_tokens,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_image_tokens = prefix_output.shape[1] - (
            lang_tokens.shape[1] if lang_tokens is not None else 0
        )
        return prefix_output[:, :num_image_tokens], prefix_pad_masks[:, :num_image_tokens]
