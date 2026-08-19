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

"""Base action model for the RLT Idea3 variant.

Idea3 builds on Idea1 (learnable token injected into the VLM prefix) but
constrains the RL token to attend only to text tokens. Image / point-cloud /
suffix tokens are invisible to the RL token, so the compressed feature z_rl
captures pure language information.
"""

from __future__ import annotations

import torch

from rlinf.models.embodiment.rlt_idea2.rlt_idea2_action_model import (
    RltIdea2ActionModel,
)


class RltIdea3ActionModel(RltIdea2ActionModel):
    """Wrapper around Pi0 with a text-only learnable-token RLT objective.

    Reuses Idea1 plumbing and the image-token counting helper from Idea2,
    changing only the attention mask for the RL token position.
    """

    @staticmethod
    def _apply_text_only_mask(
        attn_mask: torch.Tensor,
        prefix_len: int,
        num_image_tokens: int,
    ) -> torch.Tensor:
        """Restrict the RL token query to attend only to text token keys.

        Args:
            attn_mask: ``[B, N, N]`` bool attention mask where ``True`` means
                allowed attention. N is the full sequence length (prefix +
                optional suffix).
            prefix_len: number of prefix tokens including the injected RL
                token. The RL token is therefore at query index
                ``prefix_len - 1``.
            num_image_tokens: number of image tokens at the start of the
                prefix. Text tokens occupy the range
                ``[num_image_tokens, prefix_len - 1)``.

        Returns:
            A new attention mask with the RL token row restricted to text
            keys.
        """
        rl_query_idx = prefix_len - 1
        key_is_text = torch.zeros(
            attn_mask.shape[2], dtype=torch.bool, device=attn_mask.device
        )
        key_is_text[num_image_tokens : rl_query_idx] = True

        new_mask = attn_mask.clone()
        new_mask[:, rl_query_idx, :] = (
            attn_mask[:, rl_query_idx, :] & key_is_text.unsqueeze(0)
        )
        return new_mask

    def _select_rlt_text_prefix_embeddings(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
        num_image_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return only the text-token prefix embeddings as decoder targets."""
        return (
            prefix_output[:, num_image_tokens:],
            prefix_mask[:, num_image_tokens:],
        )
