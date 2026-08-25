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

"""OpenPI-based RLT Idea1 action model.

Unlike :mod:`rlinf.models.embodiment.rlt_idea1` (which is built on the
``openpi_rlinf`` Pi0 reimplementation), this module subclasses the official
OpenPI wrapper :class:`OpenPi0ForRLActionPrediction`. A learnable token is
appended to the VLM prefix, and the hidden state at that token position is the
RL feature ``z_rl``. The prefix is reconstructed by the shared
:class:`RltIdea1OpenPiDecoder`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from torch.utils._pytree import tree_map

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.rlt_idea1.rlt_idea1_config import RltIdea1Config
from rlinf.models.embodiment.rlt_idea1_openpi.rlt_idea1_openpi_decoder import (
    RltIdea1OpenPiDecoder,
)
from rlinf.utils.pytree import register_pytree_dataclasses


class OpenPiIdea1ActionModel(OpenPi0ForRLActionPrediction):
    """Official-OpenPI wrapper with a learnable-token-in-VLM RLT objective."""

    def __init__(self, config, *, rlt_cfg: RltIdea1Config | None = None):
        self.rlt_cfg = rlt_cfg or RltIdea1Config()
        # The official wrapper's __init__ builds the standard encoder-decoder
        # RLTTokenTransformer whenever config.use_rlt is set. Idea1 replaces it
        # with a decoder-only token-injection module, so suppress that unused
        # allocation first and restore the flag right after.
        # OpenPi0Config is a frozen dataclass, so mutate its __dict__ the
        # same way the official openpi factory does instead of using
        # attribute assignment (which raises FrozenInstanceError).
        config.__dict__["use_rlt"] = False
        super().__init__(config)
        self.config.__dict__["use_rlt"] = bool(self.rlt_cfg.use_rlt)

        if self.rlt_cfg.use_rlt:
            # Decoder-only module used by the token-injection ideas.
            self.rlt_module = RltIdea1OpenPiDecoder(
                input_dim=self.rlt_cfg.rlt_input_dim,
                embed_dim=self.rlt_cfg.rlt_embed_dim,
                prefix_seq_len=self.rlt_cfg.rlt_prefix_seq_len,
                num_layers=self.rlt_cfg.rlt_num_layers,
                num_heads=self.rlt_cfg.rlt_num_heads,
                mlp_ratio=self.rlt_cfg.rlt_mlp_ratio,
                z_norm=self.rlt_cfg.rlt_z_norm,
                z_l2_weight=self.rlt_cfg.rlt_z_l2_weight,
            ).to(dtype=torch.bfloat16)
            if self.rlt_cfg.freeze_vlm:
                self.freeze_vlm()

    # ------------------------------------------------------------------
    # RL token plumbing
    # ------------------------------------------------------------------
    def _require_rlt(self) -> None:
        if not self.rlt_cfg.use_rlt or not hasattr(self, "rlt_module"):
            raise ValueError(
                "OpenPI Idea1 RLT operation requires openpi.use_rlt=True."
            )

    def _get_rl_token(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        self._require_rlt()
        return self.rlt_module.get_rl_token(batch_size, device, dtype)

    @staticmethod
    def _inject_rl_token_into_prefix(
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        rl_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append the learnable token to the prefix embeddings and masks."""
        batch_size = rl_token.shape[0]
        device = rl_token.device

        prefix_embs = torch.cat([prefix_embs, rl_token], dim=1)
        pad_extra = torch.ones(
            batch_size, 1, dtype=prefix_pad_masks.dtype, device=device
        )
        prefix_pad_masks = torch.cat([prefix_pad_masks, pad_extra], dim=1)
        # The token is a non-autoregressive prefix token, matching the other
        # prefix tokens (att_masks value 0 in the official make_att_2d_masks
        # cumulative-index formulation).
        att_extra = prefix_att_masks.new_zeros(batch_size, 1)
        prefix_att_masks = torch.cat([prefix_att_masks, att_extra], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _num_image_tokens(self, prefix_len_with_token: int, lang_tokens) -> int:
        lang_len = (
            lang_tokens.shape[1] if lang_tokens is not None else 0
        )
        return prefix_len_with_token - 1 - lang_len

    def _apply_rl_token_attention(
        self,
        att_2d_masks: torch.Tensor,
        prefix_len: int,
        num_image_tokens: int,
    ) -> torch.Tensor:
        """Hook for restricting which keys the RL token can attend to.

        Idea1 keeps the full bidirectional prefix attention, so this base
        implementation is a no-op.
        """
        del prefix_len, num_image_tokens
        return att_2d_masks

    def _select_decoder_target(
        self,
        prefix_output: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        lang_tokens,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the prefix embeddings used as the reconstruction target.

        Idea1 reconstructs the full prefix excluding the learnable token.
        """
        del lang_tokens
        return prefix_output, prefix_pad_masks

    def _rlt_forward(
        self,
        decoder_target: torch.Tensor,
        decoder_mask: torch.Tensor,
        *,
        z_rl: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._require_rlt()
        rlt_param = next(self.rlt_module.parameters())
        decoder_target = decoder_target.to(
            device=rlt_param.device, dtype=rlt_param.dtype
        )
        # The VLM prefix output is float32 while the decoder weights are
        # bf16; cast z_rl explicitly (gradients still flow through the cast).
        z_rl = z_rl.to(device=rlt_param.device, dtype=rlt_param.dtype)
        rlt_mask = decoder_mask if self.rlt_cfg.rlt_use_mask else None
        return self.rlt_module(z_rl, decoder_target, rlt_mask)

    # ------------------------------------------------------------------
    # SFT path
    # ------------------------------------------------------------------
    def sft_forward(self, data, use_action_chunk_loss: bool = False, **kwargs):
        if not self.rlt_cfg.use_rlt:
            return OpenPi0ForRLActionPrediction.sft_forward(
                self, data, use_action_chunk_loss=use_action_chunk_loss, **kwargs
            )

        if hasattr(self, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable()

        if isinstance(data, tuple):
            observation, actions = data
        else:
            observation = data["observation"]
            actions = data["actions"]

        device = next(self.parameters()).device
        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: (
                torch.as_tensor(x, device=device).contiguous().clone()
                if x is not None
                else x
            ),
            observation,
        )
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=device)
        else:
            actions = actions.to(device=device)
        actions = actions.to(dtype=torch.float32)

        loss, decoder_target, decoder_mask, z_rl = (
            self._sft_forward_with_rlt_prefix(observation, actions)
        )
        if use_action_chunk_loss:
            loss = loss[:, : self.config.action_chunk, : self.config.action_env_dim]
        vla_loss = loss.mean()
        rlt_loss, _ = self._rlt_forward(
            decoder_target, decoder_mask, z_rl=z_rl
        )
        return {
            "loss": rlt_loss + self.rlt_cfg.rlt_alpha * vla_loss,
            "vla_loss": vla_loss,
            "rlt_loss": rlt_loss,
        }

    def _sft_forward_with_rlt_prefix(self, observation, actions):
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=True)
        )

        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, time)
        )

        backbone_dtype = self.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        if prefix_embs.dtype != backbone_dtype:
            prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        if suffix_embs.dtype != backbone_dtype:
            suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        rl_token = self._get_rl_token(
            actions.shape[0], prefix_embs.device, prefix_embs.dtype
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self._inject_rl_token_into_prefix(
                prefix_embs, prefix_pad_masks, prefix_att_masks, rl_token
            )
        )
        prefix_len = prefix_embs.shape[1]

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        num_image_tokens = self._num_image_tokens(prefix_len, lang_tokens)
        att_2d_masks = self._apply_rl_token_attention(
            att_2d_masks, prefix_len, num_image_tokens
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(
            prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        ):
            (prefix_output, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_output, suffix_out

        prefix_output, suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        loss = F.mse_loss(u_t, v_t, reduction="none")

        # z_rl must retain gradients so the reconstruction objective teaches
        # the VLM and the injected token to compress the prefix.
        z_rl = prefix_output[:, -1, :]
        decoder_target, decoder_mask = self._select_decoder_target(
            prefix_output[:, :-1, :].detach(),
            prefix_pad_masks[:, :-1],
            lang_tokens,
        )
        return loss, decoder_target, decoder_mask, z_rl

    # ------------------------------------------------------------------
    # Stage2 feature extraction
    # ------------------------------------------------------------------
    def _build_rlt_prefix_cache(self, observation, *, train: bool):
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=train)
        )
        device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        if lang_tokens is not None:
            lang_tokens = lang_tokens.to(device)
        if lang_masks is not None:
            lang_masks = lang_masks.to(device)
        state = state.to(device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        rl_token = self._get_rl_token(
            prefix_embs.shape[0], prefix_embs.device, prefix_embs.dtype
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self._inject_rl_token_into_prefix(
                prefix_embs, prefix_pad_masks, prefix_att_masks, rl_token
            )
        )
        prefix_len = prefix_embs.shape[1]

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        num_image_tokens = self._num_image_tokens(prefix_len, lang_tokens)
        prefix_att_2d_masks = self._apply_rl_token_attention(
            prefix_att_2d_masks, prefix_len, num_image_tokens
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
            prefix_att_2d_masks
        )
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_output, prefix_pad_masks, past_key_values, lang_tokens, state

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs):
        self._require_rlt()
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)

        prefix_output, prefix_pad_masks, past_key_values, _, state = (
            self._build_rlt_prefix_cache(observation, train=False)
        )

        rlt_param = next(self.rlt_module.parameters())
        z_rl_raw = prefix_output[:, -1, :].to(
            device=rlt_param.device, dtype=rlt_param.dtype
        )
        z_rl = self.rlt_module.encode_z(z_rl_raw).to(dtype=torch.float32)

        outputs = self._sample_actions_with_prefix_cache(
            state,
            prefix_output,
            prefix_pad_masks,
            past_key_values,
            mode="eval",
            compute_values=False,
        )
        ref_chunk = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"]

        raw_proprio = self._select_configured_state(env_obs["states"])
        if "maniskill" in str(self.config.config_name).lower():
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
