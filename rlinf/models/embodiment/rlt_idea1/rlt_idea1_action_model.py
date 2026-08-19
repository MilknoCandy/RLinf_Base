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

"""Base action model for the RLT Idea1 variant.

Subclasses :class:`OpenPiPytorchActionModel` and replaces the standard
post-hoc ``RLTTokenTransformer`` with :class:`RltIdea1Decoder`. The learnable
token is injected into the VLM prefix by concatenation (no Pi0 core changes),
and the VLM output at the token position is treated as the RL feature z_rl.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rlinf.models.embodiment.openpi_rlinf.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.rlt_idea1.rlt_idea1_config import RltIdea1Config
from rlinf.models.embodiment.rlt_idea1.rlt_idea1_decoder import RltIdea1Decoder


class RltIdea1ActionModel(OpenPiPytorchActionModel):
    """Wrapper around Pi0 with a learnable-token-in-VLM RLT objective.

    The parent class setup is replicated with two differences:
      * ``self.rlt_cfg`` is a :class:`RltIdea1Config`.
      * ``self.rlt_module`` is a :class:`RltIdea1Decoder` (decoder only).
    All shared plumbing (device, FSDP marks, gradient checkpointing, prefix
    selection) is inherited unchanged.
    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        rlt_cfg: RltIdea1Config | None = None,
    ):
        nn.Module.__init__(self)
        self.model = pi0_model
        self.num_steps = num_steps
        self.action_env_dim = action_env_dim
        self.rlt_cfg = rlt_cfg or RltIdea1Config()

        if self.rlt_cfg.use_rlt:
            self.rlt_module = RltIdea1Decoder(
                input_dim=self.rlt_cfg.rlt_input_dim,
                embed_dim=self.rlt_cfg.rlt_embed_dim,
                prefix_seq_len=self.rlt_cfg.rlt_prefix_seq_len,
                num_layers=self.rlt_cfg.rlt_num_layers,
                num_heads=self.rlt_cfg.rlt_num_heads,
                mlp_ratio=self.rlt_cfg.rlt_mlp_ratio,
            ).to(dtype=next(self.model.parameters()).dtype)

        self._mark_fsdp_wrap_names()

    # --- RLT forward for the decoder-only path ---

    def _rlt_forward(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
        *,
        z_rl: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the Idea1 reconstruction loss.

        Args:
            prefix_output: ``[B, S, input_dim]`` decoder target embeddings
                (without the learnable token).
            prefix_mask: ``[B, S]`` target mask.
            z_rl: ``[B, input_dim]`` VLM hidden state at the learnable token
                position. Required for Idea1.

        Returns:
            ``(mse, metrics)``.
        """
        self._require_rlt()
        if z_rl is None:
            raise ValueError(
                "RltIdea1 requires z_rl from the VLM token position; "
                "pass z_rl=... to _rlt_forward()."
            )
        rlt_param = next(self.rlt_module.parameters())
        prefix_output = prefix_output.to(
            device=rlt_param.device, dtype=rlt_param.dtype
        )
        rlt_mask = prefix_mask if self.rlt_cfg.rlt_use_mask else None
        return self.rlt_module(z_rl, prefix_output, rlt_mask)

    def _encode_rlt_flat(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Idea1 has no encoder; z_rl comes from the VLM token position."""
        self._require_rlt()
        raise RuntimeError(
            "_encode_rlt_flat is not used by RltIdea1; z_rl is the VLM "
            "hidden state at the learnable token position."
        )

    def _get_vlm_rl_token(self, batch_size: int) -> torch.Tensor:
        """Return the learnable token ``[B, 1, input_dim]`` for VLM injection."""
        if not self.rlt_cfg.use_rlt:
            raise RuntimeError("_get_vlm_rl_token requires use_rlt=True.")
        return self.rlt_module.get_rl_token(
            batch_size, self.device, next(self.model.parameters()).dtype
        )

    @staticmethod
    def _inject_rl_token_into_prefix(
        prefix_tokens: torch.Tensor,
        prefix_mask: torch.Tensor,
        prefix_ar_mask: torch.Tensor,
        rl_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append a learnable token to the prefix embeddings.

        The token gets a valid mask and bidirectional attention (ar_mask
        entry False), matching the rest of the prefix.

        Args:
            prefix_tokens: ``[B, S, D]``.
            prefix_mask: ``[B, S]`` bool.
            prefix_ar_mask: ``[S]`` bool.
            rl_token: ``[B, 1, D]``.

        Returns:
            ``(tokens, mask, ar_mask)`` with the token appended.
        """
        tokens = torch.cat([prefix_tokens, rl_token], dim=1)
        extra_mask = torch.ones(
            rl_token.shape[0], 1, dtype=torch.bool, device=rl_token.device
        )
        mask = torch.cat([prefix_mask, extra_mask], dim=1)
        extra_ar = torch.tensor([False], device=prefix_ar_mask.device)
        ar_mask = torch.cat([prefix_ar_mask, extra_ar], dim=0)
        return tokens, mask, ar_mask