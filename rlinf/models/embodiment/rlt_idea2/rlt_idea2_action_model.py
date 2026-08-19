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

"""Base action model for the RLT Idea2 variant.

Idea2 builds on Idea1 (learnable token injected into the VLM prefix) but
constrains the RL token to attend only to image tokens. Language / point-cloud
/ suffix tokens are invisible to the RL token, so the compressed feature z_rl
captures pure visual information.
"""

from __future__ import annotations

import torch

from rlinf.models.embodiment.rlt_idea1.rlt_idea1_action_model import (
    RltIdea1ActionModel,
)


class RltIdea2ActionModel(RltIdea1ActionModel):
    """Wrapper around Pi0 with an image-only learnable-token RLT objective.

    Reuses all Idea1 plumbing (decoder-only module, token injection, FSDP
    marks) and only changes the attention mask for the RL token position.
    """

    @staticmethod
    def _apply_image_only_mask(
        attn_mask: torch.Tensor,
        prefix_len: int,
        num_image_tokens: int,
    ) -> torch.Tensor:
        """Restrict the RL token query to attend only to image token keys.

        Args:
            attn_mask: ``[B, N, N]`` bool attention mask where ``True`` means
                allowed attention. N is the full sequence length (prefix +
                optional suffix).
            prefix_len: number of prefix tokens including the injected RL
                token. The RL token is therefore at query index
                ``prefix_len - 1``.
            num_image_tokens: number of image tokens at the start of the
                prefix (image tokens precede language tokens).

        Returns:
            A new attention mask with the RL token row restricted to image
            keys.
        """
        rl_query_idx = prefix_len - 1
        key_is_image = torch.zeros(
            attn_mask.shape[2], dtype=torch.bool, device=attn_mask.device
        )
        key_is_image[:num_image_tokens] = True

        new_mask = attn_mask.clone()
        new_mask[:, rl_query_idx, :] = (
            attn_mask[:, rl_query_idx, :] & key_is_image.unsqueeze(0)
        )
        return new_mask

    @staticmethod
    def _num_image_tokens(prefix_len_with_token: int, tokenized_prompt) -> int:
        """Return the number of image tokens in the prefix.

        Images precede language tokens, and the injected RL token is appended
        last. This assumes no point-cloud tokens follow the language tokens
        (pcd=False), matching the ManiSkill RLT configs.
        """
        lang_len = (
            tokenized_prompt.shape[1] if tokenized_prompt is not None else 0
        )
        return prefix_len_with_token - 1 - lang_len