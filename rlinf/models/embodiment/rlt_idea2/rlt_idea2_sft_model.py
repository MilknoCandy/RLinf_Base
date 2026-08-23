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

"""SFT action model for the RLT Idea2 variant.

Same as Idea1, except the injected RL token only attends to image tokens.
"""

from __future__ import annotations

from typing import Any

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0, make_attn_mask
from rlinf.models.embodiment.openpi_rlinf.sft_action_model import (
    OpenPiPytorchSFTActionModel,
)
from rlinf.models.embodiment.rlt_idea1.rlt_idea1_config import RltIdea1Config
from rlinf.models.embodiment.rlt_idea1.rlt_idea1_sft_model import (
    RltIdea1SFTModel,
)
from rlinf.models.embodiment.rlt_idea2.rlt_idea2_action_model import (
    RltIdea2ActionModel,
)


class RltIdea2SFTModel(RltIdea2ActionModel, RltIdea1SFTModel):
    """SFT wrapper with the Idea2 image-only RL token objective."""

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        rlt_cfg: RltIdea1Config | None = None,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            rlt_cfg=rlt_cfg,
        )

    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs):
        if forward_type != ForwardType.SFT:
            raise NotImplementedError(
                f"{type(self).__name__} only supports ForwardType.SFT; "
                f"got forward_type={forward_type!r}."
            )
        return self.sft_forward(**kwargs)

    def sft_forward(self, data: Any) -> torch.Tensor:
        observation, actions = OpenPiPytorchSFTActionModel._unpack_sft_batch(data)
        observation = OpenPiPytorchSFTActionModel._observation_to_device(
            self, observation
        )
        actions = OpenPiPytorchSFTActionModel._actions_to_device(self, actions)

        if not self.rlt_cfg.use_rlt:
            per_timestep_loss = self.model.compute_loss(
                observation, actions, train=True
            )
            return per_timestep_loss.mean()

        per_timestep_loss, decoder_target, decoder_mask, z_rl = (
            self._sft_forward_with_rlt_prefix(observation, actions)
        )
        vla_loss = per_timestep_loss.mean()
        rlt_loss, _ = self._rlt_forward(
            decoder_target, decoder_mask, z_rl=z_rl
        )
        return {
            "loss": rlt_loss + self.rlt_cfg.rlt_alpha * vla_loss,
            "vla_loss": vla_loss,
            "rlt_loss": rlt_loss,
        }

    def compute_loss(self, data: Any) -> torch.Tensor:
        return self.sft_forward(data)

    def _sft_forward_with_rlt_prefix(
        self,
        observation: Observation,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one VLA pass and extract z_rl from the image-only RL token."""
        batch_size = actions.shape[0]
        device = actions.device

        observation = pi0_model_module.preprocess_observation(observation, train=True)
        embed_dtype = self.model.embed_dtype
        observation = pi0_model_module._observation_to_dtype(observation, embed_dtype)
        actions = actions.to(dtype=embed_dtype)
        dtype = actions.dtype

        noise = torch.randn(actions.shape, device=device, dtype=dtype)
        time = (
            torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
            .sample((batch_size,))
            .to(device=device, dtype=dtype)
        )
        time = time * 0.999 + 0.001
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(
            observation
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = (
            self.model.embed_suffix(observation, x_t, time)
        )

        rl_token = self._get_vlm_rl_token(batch_size)
        prefix_tokens, prefix_mask, prefix_ar_mask = (
            self._inject_rl_token_into_prefix(
                prefix_tokens, prefix_mask, prefix_ar_mask, rl_token
            )
        )
        prefix_len = prefix_tokens.shape[1]

        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        num_image_tokens = self._num_image_tokens(
            prefix_len, observation.tokenized_prompt
        )
        attn_mask = self._apply_image_only_mask(
            attn_mask, prefix_len, num_image_tokens
        )

        positions = torch.cumsum(input_mask.int(), dim=1) - 1
        prefix_out, suffix_out = self.model.llm(
            [prefix_tokens, suffix_tokens],
            positions=positions,
            mask=attn_mask,
            adarms_cond=[None, adarms_cond],
        )[0]

        v_t = self.model.velocity_from_suffix(
            suffix_out[:, -self.model.action_horizon :]
        )
        loss = torch.mean(torch.square(v_t - u_t), dim=-1)

        z_rl = prefix_out[:, -1:, :].squeeze(1)

        decoder_target = prefix_out[:, :-1, :].detach()
        decoder_mask = prefix_mask[:, :-1]
        decoder_target, decoder_mask = self._select_rlt_prefix_embeddings(
            decoder_target, decoder_mask, observation.tokenized_prompt
        )

        return loss, decoder_target, decoder_mask, z_rl
