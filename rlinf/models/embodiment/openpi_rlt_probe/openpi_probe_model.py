# Copyright 2026 The RLinf Authors.
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

"""Stage 2 feature model for the RLT probe, built on the official OpenPI path.

This wrapper subclasses ``openpi.OpenPi0ForRLActionPrediction`` and only changes
how ``z_rl`` is produced in :meth:`extract_rlt_obs`. The Stage 1 SFT forward is
left untouched so the VLM part stays bit-for-bit identical to ``openpi``.

Two ablation modes:

* ``rlt_mode="token"``: run the frozen Stage 1 RLT token transformer on the
  selected ``image``/``text``/``action`` feature sequence.
* ``rlt_mode="none"``: skip the RLT token transformer and mean-pool the
  selected VLM feature sequence directly into ``z_rl``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from openpi.models import model as _model

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi_rlt_probe.feature_sources import (
    RLTFeatureBundle,
    available_feature_sources,
    mean_pool_features,
    select_features,
)

_DEFAULT_ACTION_FEATURE_DIM = 1024  # gemma_300m (action expert) width


class OpenPiProbeForRLActionPrediction(OpenPi0ForRLActionPrediction):
    """OpenPI wrapper that extracts a configurable VLM feature as ``z_rl``."""

    def __init__(
        self,
        config: OpenPi0Config,
        *,
        feature_source: str = "all",
        rlt_mode: str = "token",
    ):
        super().__init__(config)

        feature_source = str(feature_source).lower()
        rlt_mode = str(rlt_mode).lower()
        if feature_source not in available_feature_sources():
            raise ValueError(
                f"feature_source={feature_source!r} is not registered; "
                f"available sources: {available_feature_sources()}."
            )
        if rlt_mode not in ("token", "none"):
            raise ValueError(
                f"rlt_mode={rlt_mode!r} is not supported; use 'token' or 'none'."
            )
        if rlt_mode == "token" and not (
            config.use_rlt and hasattr(self, "rlt_module")
        ):
            raise ValueError(
                "rlt_mode='token' requires a Stage 1 RLT checkpoint "
                "(openpi.use_rlt=True and an rlt_module)."
            )

        self.feature_source = feature_source
        self.rlt_mode = rlt_mode

        feature_dim = (
            self._get_action_feature_dim()
            if feature_source == "action"
            else config.rlt_input_dim
        )
        self.feature_dim = feature_dim
        self.z_proj = None
        if feature_dim != config.rlt_embed_dim:
            self.z_proj = nn.Linear(
                feature_dim, config.rlt_embed_dim
            ).to(dtype=torch.bfloat16)

    def _get_action_feature_dim(self) -> int:
        """Return the action-expert hidden width, falling back to 1024."""
        try:
            expert = self.paligemma_with_expert.gemma_expert
            return int(expert.model.config.hidden_size)
        except (AttributeError, KeyError):
            return _DEFAULT_ACTION_FEATURE_DIM

    def _extract_action_features(self, state, prefix_pad_masks, past_key_values, x_0):
        """Action-expert suffix features for the clean sampled action chunk."""
        timestep = torch.zeros(
            (state.shape[0],), device=state.device, dtype=torch.float32
        )
        return self.get_suffix_out(
            state, prefix_pad_masks, past_key_values, x_0, timestep
        )

    def _project_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.z_proj is None:
            return features
        return self.z_proj(features.to(dtype=self.z_proj.weight.dtype))

    def _compute_z_rl(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected = self._project_features(features)
        if self.rlt_mode == "token":
            rlt_param = next(self.rlt_module.parameters())
            projected = projected.to(
                device=rlt_param.device, dtype=rlt_param.dtype
            )
            rlt_mask = feature_mask if self.config.rlt_use_mask else None
            return self.rlt_module.encode_flat(projected, rlt_mask).to(
                dtype=torch.float32
            )
        return mean_pool_features(projected, feature_mask).to(dtype=torch.float32)

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract ``z_rl`` / ``proprio`` / ``ref_chunk`` for Stage 2."""
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)

        prefix_output, prefix_pad_masks, past_key_values, lang_tokens, state = (
            self._build_rlt_prefix_cache(observation, train=False)
        )

        outputs = self._sample_actions_with_prefix_cache(
            state,
            prefix_output,
            prefix_pad_masks,
            past_key_values,
            mode="eval",
            compute_values=False,
        )
        model_actions = outputs["actions"]

        action_output = None
        action_mask = None
        if self.feature_source == "action":
            action_output = self._extract_action_features(
                state, prefix_pad_masks, past_key_values, model_actions
            )
            action_mask = torch.ones(
                (model_actions.shape[0], model_actions.shape[1]),
                dtype=torch.bool,
                device=model_actions.device,
            )

        bundle = RLTFeatureBundle(
            prefix_output=prefix_output,
            prefix_mask=prefix_pad_masks,
            prompt_tokens=lang_tokens,
            action_output=action_output,
            action_mask=action_mask,
        )
        features, feature_mask = select_features(self.feature_source, bundle)
        z_rl = self._compute_z_rl(features, feature_mask)

        ref_chunk = self.output_transform(
            {"actions": model_actions, "state": observation.state}
        )["actions"]
        raw_proprio = self._select_configured_state(env_obs["states"])
        if (
            isinstance(self.config.config_name, str)
            and "maniskill" in self.config.config_name.lower()
        ):
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
