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

"""VLM-internal RLT variant: learnable token injected into the VLM prefix.

This module extends the standard ``OpenPiPytorchActionModel`` with a
learnable-token-in-VLM path. Instead of a post-hoc transformer encoder,
a single learnable token is concatenated to the VLM prefix embeddings.
The VLM's own attention layers compress the observation into this token's
hidden state, which is then decoded by a lightweight ``RLTTokenDecoderOnly``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rlinf.models.embodiment.openpi_rlinf.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf_vlm.utils.rlt_config import RLTConfigVLM


class OpenPiPytorchActionModelVLM(OpenPiPytorchActionModel):
    """VLM-internal RLT variant of the base action model.

    Differences from the standard (post_vlm) variant:
    - Uses ``RLTTokenDecoderOnly`` instead of ``RLTTokenTransformer``
    - No separate encoder; VLM attention layers do compression
    - Learnable token injected by concatenating to prefix embeddings
    - ``_rlt_forward`` accepts optional ``z_rl`` from VLM token position
    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        rlt_cfg: RLTConfigVLM | None = None,
    ):
        # Bypass parent's rlt_module creation; we create our own.
        # Call nn.Module.__init__ directly, then replicate parent setup.
        nn.Module.__init__(self)
        self.model = pi0_model
        self.num_steps = num_steps
        self.action_env_dim = action_env_dim
        self.rlt_cfg = rlt_cfg or RLTConfigVLM()

        if self.rlt_cfg.use_rlt:
            from rlinf.models.embodiment.modules.rlt_token_transformer import (
                RLTTokenDecoderOnly,
            )

            self.rlt_module = RLTTokenDecoderOnly(
                input_dim=self.rlt_cfg.rlt_input_dim,
                embed_dim=self.rlt_cfg.rlt_embed_dim,
                prefix_seq_len=self.rlt_cfg.rlt_prefix_seq_len,
                num_layers=self.rlt_cfg.rlt_num_layers,
                num_heads=self.rlt_cfg.rlt_num_heads,
                mlp_ratio=self.rlt_cfg.rlt_mlp_ratio,
            ).to(dtype=next(self.model.parameters()).dtype)

        self._mark_fsdp_wrap_names()

    # --- Override RLT forward to accept z_rl from VLM output ---

    def _rlt_forward(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
        *,
        z_rl: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """RLT reconstruction loss.

        For vlm_internal: ``z_rl`` must be provided (extracted from the
        learnable token's VLM output position). The decoder reconstructs
        ``prefix_output`` from ``z_rl``.

        For the post_vlm path (parent class default), ``z_rl`` is None and
        the full encoder-decoder runs.
        """
        self._require_rlt()
        rlt_param = next(self.rlt_module.parameters())
        prefix_output = prefix_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
        rlt_mask = prefix_mask if self.rlt_cfg.rlt_use_mask else None
        if z_rl is not None:
            return self.rlt_module(z_rl, prefix_output, rlt_mask)
        # Fallback to parent behaviour (full encoder-decoder)
        return self.rlt_module(prefix_output, rlt_mask)

    def _encode_rlt_flat(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Not used for vlm_internal.

        z_rl comes from the VLM token position directly, not from encoding.
        """
        self._require_rlt()
        raise RuntimeError(
            "_encode_rlt_flat is not used with openpi_rlinf_vlm; "
            "z_rl comes from the VLM token position directly."
        )

    def _get_vlm_rl_token(self, batch_size: int) -> torch.Tensor:
        """Return the learnable token [B, 1, embed_dim] for VLM injection."""
        if not self.rlt_cfg.use_rlt:
            raise RuntimeError(
                "_get_vlm_rl_token requires use_rlt=True."
            )
        return self.rlt_module.get_rl_token(
            batch_size, self.device, next(self.model.parameters()).dtype
        )

    # --- Helper: inject learnable token into prefix embeddings ---

    @staticmethod
    def _inject_rl_token_into_prefix(
        prefix_tokens: torch.Tensor,
        prefix_mask: torch.Tensor,
        prefix_ar_mask: torch.Tensor,
        rl_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append a learnable token to prefix embeddings.

        The extra token gets bidirectional attention (ar_mask=False) and a
        valid mask (True).

        Args:
            prefix_tokens: [B, S, D] original prefix embeddings
            prefix_mask: [B, S] bool mask
            prefix_ar_mask: [S] autoregressive mask
            rl_token: [B, 1, D] learnable token

        Returns:
            (tokens, mask, ar_mask) with the learnable token appended.
        """
        tokens = torch.cat([prefix_tokens, rl_token], dim=1)
        extra_mask = torch.ones(
            rl_token.shape[0], 1, dtype=torch.bool, device=rl_token.device
        )
        mask = torch.cat([prefix_mask, extra_mask], dim=1)
        extra_ar = torch.tensor([False], device=prefix_ar_mask.device)
        ar_mask = torch.cat([prefix_ar_mask, extra_ar], dim=0)
        return tokens, mask, ar_mask