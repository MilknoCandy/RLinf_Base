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

"""Eval action model for the VLM-internal RLT variant.

Extends the standard eval model to extract z_rl from the learnable token's
VLM output position (instead of from a post-hoc encoder).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from rlinf.models.embodiment.openpi_rlinf.eval_action_model import (
    OpenPiPytorchEvalActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf_vlm.openpi_action_model import (
    OpenPiPytorchActionModelVLM,
)
from rlinf.models.embodiment.openpi_rlinf_vlm.utils.rlt_config import RLTConfigVLM


class OpenPiPytorchEvalActionModelVLM(
    OpenPiPytorchActionModelVLM, OpenPiPytorchEvalActionModel
):
    """Eval variant: VLM-internal learnable token for feature extraction.

    Inherits the eval pipeline (openpi transforms, Euler sampler) from
    ``OpenPiPytorchEvalActionModel`` and overrides ``extract_rlt_obs``
    to extract z_rl from the VLM token position instead of the encoder.
    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
        rlt_cfg: RLTConfigVLM | None = None,
    ):
        # MRO: OpenPiPytorchActionModelVLM -> OpenPiPytorchActionModel -> nn.Module
        #      OpenPiPytorchEvalActionModel -> OpenPiPytorchActionModel -> nn.Module
        # Both inherit from OpenPiPytorchActionModel.__init__.
        # We need OpenPiPytorchActionModelVLM.__init__ for the rlt_module.
        # We also need OpenPiPytorchEvalActionModel's pipeline setup.
        OpenPiPytorchActionModelVLM.__init__(
            self,
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            rlt_cfg=rlt_cfg,
        )
        self.action_chunk = action_chunk
        self.config_name = config_name
        self.state_indices = list(state_indices) if state_indices else None
        self._input_transform_fn = None
        self._output_transform_fn = None

    # Inherit these from OpenPiPytorchEvalActionModel via MRO:
    #   setup_wrappers, _ensure_wrappers, _select_configured_state,
    #   _repack_env_obs, input_transform, output_transform,
    #   _observation_dict_to_device, predict_action_batch, _predict_eval,
    #   _sample_actions_from_prefix_cache

    # --- Override: extract z_rl from VLM token position ---

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract Stage1 features for Stage2 RLT head.

        Injects the learnable token into the VLM prefix, runs one forward
        pass, and extracts z_rl from the token's output position.
        """
        self._require_rlt()
        repacked = self._repack_env_obs(env_obs)
        processed = self.input_transform(repacked, transpose=False)
        observation = self._observation_dict_to_device(processed)

        prepared_observation = pi0_model_module.preprocess_observation(
            observation, train=False
        )

        # 1. Build prefix cache normally (no token yet)
        prefix_output, prefix_mask, kv_cache = self.model.build_prefix_cache(
            prepared_observation
        )
        batch_size = prefix_output.shape[0]

        # 2. Inject learnable token & re-run prefix pass with the token
        rl_token = self._get_vlm_rl_token(batch_size)

        # Re-embed with the learnable token concatenated
        prefix_tokens, prefix_mask_raw, prefix_ar_mask = (
            self.model.embed_prefix(prepared_observation)
        )
        prefix_tokens, prefix_mask_raw, prefix_ar_mask = (
            self._inject_rl_token_into_prefix(
                prefix_tokens, prefix_mask_raw, prefix_ar_mask, rl_token
            )
        )

        from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import make_attn_mask

        prefix_attn_mask = make_attn_mask(prefix_mask_raw, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask_raw.int(), dim=1) - 1
        outputs, kv_cache = self.model.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )
        prefix_output = outputs[0]
        prefix_mask = prefix_mask_raw  # includes learnable token

        # 3. Extract z_rl from learnable token position (last in prefix)
        z_rl = prefix_output[:, -1:, :].to(dtype=torch.float32)  # [B, embed_dim]

        # 4. Sample actions using the cache with token for reference
        model_actions = self._sample_actions_from_prefix_cache(
            prepared_observation,
            prefix_mask,
            kv_cache,
        )
        ref_chunk = self.output_transform(
            {"actions": model_actions, "state": observation.state}
        )["actions"]

        raw_proprio = self._select_configured_state(env_obs["states"])
        if "maniskill" in self.config_name.lower():
            state_dim = (
                raw_proprio.shape[-1]
                if hasattr(raw_proprio, "shape")
                else np.asarray(raw_proprio).shape[-1]
            )
            proprio = observation.state[..., :state_dim]
        else:
            proprio = raw_proprio
        if not torch.is_tensor(proprio):
            proprio = torch.as_tensor(proprio)

        return {
            "z_rl": z_rl,
            "proprio": proprio.to(device=z_rl.device, dtype=torch.float32),
            "ref_chunk": ref_chunk.to(device=z_rl.device, dtype=torch.float32),
        }