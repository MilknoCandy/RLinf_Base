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

"""Eval / Stage2 feature extraction model for the RLT Idea2 variant.

Mirrors :class:`RltIdea1EvalModel`, except the injected RL token is masked so
that it attends only to image-token keys in the VLM prefix pass.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from rlinf.models.embodiment.openpi_rlinf.eval_action_model import (
    OpenPiPytorchEvalActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0, make_attn_mask
from rlinf.models.embodiment.rlt_idea1.rlt_idea1_config import RltIdea1Config
from rlinf.models.embodiment.rlt_idea2.rlt_idea2_action_model import (
    RltIdea2ActionModel,
)


class RltIdea2EvalModel(RltIdea2ActionModel, OpenPiPytorchEvalActionModel):
    """Eval wrapper with the Idea2 image-only RL token feature extractor."""

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
        rlt_cfg: RltIdea1Config | None = None,
    ):
        RltIdea2ActionModel.__init__(
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

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract the frozen Stage1 features consumed by the Stage2 head.

        Idea2: inject the learnable token, run the VLM prefix pass with an
        image-only attention row for the token, and read z_rl from the token's
        output position.
        """
        self._require_rlt()
        repacked = self._repack_env_obs(env_obs)
        processed = self.input_transform(repacked, transpose=False)
        observation = self._observation_dict_to_device(processed)

        prepared_observation = pi0_model_module.preprocess_observation(
            observation, train=False
        )
        batch_size = prepared_observation.state.shape[0]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(
            prepared_observation
        )
        rl_token = self._get_vlm_rl_token(batch_size)
        prefix_tokens, prefix_mask, prefix_ar_mask = (
            self._inject_rl_token_into_prefix(
                prefix_tokens, prefix_mask, prefix_ar_mask, rl_token
            )
        )
        prefix_len = prefix_tokens.shape[1]

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        num_image_tokens = self._num_image_tokens(
            prefix_len, prepared_observation.tokenized_prompt
        )
        prefix_attn_mask = self._apply_image_only_mask(
            prefix_attn_mask, prefix_len, num_image_tokens
        )

        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
        outputs, kv_cache = self.model.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )
        prefix_output = outputs[0]

        z_rl_raw = prefix_output[:, -1:, :].squeeze(1)
        z_rl = self.rlt_module.encode_z(z_rl_raw).to(dtype=torch.float32)

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
