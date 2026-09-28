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

from __future__ import annotations

from typing import Any

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_rlinf.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0, make_attn_mask
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    OpenPiPytorchRLTConfig,
)


def _index_time(value: Any, step: int) -> Any:
    """Slice the time axis of a collated history tree. Time is dimension 1."""
    if isinstance(value, dict):
        return {key: _index_time(child, step) for key, child in value.items()}
    if torch.is_tensor(value):
        return value[:, step]
    return value


class OpenPiPytorchSFTActionModel(OpenPiPytorchActionModel):
    """SFT variant of :class:`OpenPiPytorchActionModel`.

    With ``openpi.use_rlt=False`` this computes the ordinary flow-matching loss.
    With ``openpi.use_rlt=True`` and ``rlt_mem_len_max <= 1`` it adds the original
    single-frame reconstruction on ``rlt_module``. When ``rlt_mem_len_max > 1``
    the reconstruction module is ``mem``: it carries ``z`` across a random
    same-episode suffix and reconstructs only the last chunk. That path has
    no separate ``rlt_loss``:

    ``loss = mem_loss + rlt_alpha * vla_loss``.
    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        rlt_cfg: OpenPiPytorchRLTConfig | None = None,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            rlt_cfg=rlt_cfg,
        )

    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs):
        """Dispatch — SFT variant only supports :attr:`ForwardType.SFT`."""
        if forward_type != ForwardType.SFT:
            raise NotImplementedError(
                f"{type(self).__name__} only supports ForwardType.SFT; "
                f"got forward_type={forward_type!r}. "
                "Use the RL subclass (actor.model.openpi.task='rl') for PPO."
            )
        return self.sft_forward(**kwargs)

    def sft_forward(self, data: Any) -> torch.Tensor:
        """Compute the flow-matching SFT loss for one batch.

        ``data`` is either a ``(observation, actions)`` tuple or a dict with
        ``observation`` and ``actions`` keys. The data loader has already run
        the openpi transform pipeline, so ``actions`` arrive normalised and
        padded to the model action dim. Returns the scalar mean of the
        ``(B, action_horizon)`` per-timestep loss from :meth:`Pi0.compute_loss`
        (which samples the flow-matching noise/time internally).
        """
        observation, actions = self._unpack_sft_batch(data)
        hist, hist_valid = self._unpack_rlt_history(data)
        observation = self._observation_to_device(observation)
        actions = self._actions_to_device(actions)
        if not self.rlt_cfg.use_rlt:
            per_timestep_loss = self.model.compute_loss(
                observation, actions, train=True
            )
            return per_timestep_loss.mean()

        per_timestep_loss, prefix_output, prefix_mask, encode_mask, recon_mask = (
            self._sft_forward_with_rlt_prefix(observation, actions)
        )
        vla_loss = per_timestep_loss.mean()
        if getattr(self, "mem", None) is not None:
            if hist is None or hist_valid is None:
                raise ValueError(
                    "Stage-1 mem window requires rlt_hist and rlt_hist_valid "
                    "from the SFT loader. "
                    f"rlt_mem_len_max={self.rlt_cfg.rlt_mem_len_max}."
                )
            mem_loss, mem_metrics = self._window_mem_loss(hist, hist_valid)
            return {
                "loss": mem_loss + self.rlt_cfg.rlt_alpha * vla_loss,
                "vla_loss": vla_loss,
                "mem_loss": mem_loss,
                "mem_loop_len": mem_metrics["mem_loop_len"],
            }
        rlt_loss, _ = self._rlt_forward(
            prefix_output,
            encode_mask if encode_mask is not None else prefix_mask,
            recon_mask=recon_mask,
        )
        return {
            "loss": rlt_loss + self.rlt_cfg.rlt_alpha * vla_loss,
            "vla_loss": vla_loss,
            "rlt_loss": rlt_loss,
        }

    def compute_loss(self, data: Any) -> torch.Tensor:
        """Alias kept for interface parity with the old action model."""
        return self.sft_forward(data)

    @staticmethod
    def _unpack_sft_batch(data: Any) -> tuple[Any, Any]:
        if isinstance(data, (tuple, list)):
            if len(data) != 2:
                raise ValueError(
                    "SFT batch tuple must be (observation, actions); "
                    f"got length {len(data)}."
                )
            observation, actions = data
        elif isinstance(data, dict):
            if "observation" not in data or "actions" not in data:
                raise ValueError(
                    "SFT batch dict must contain 'observation' and 'actions'; "
                    f"got keys {sorted(data)}."
                )
            observation, actions = data["observation"], data["actions"]
        else:
            raise TypeError(f"Unsupported SFT batch type: {type(data)!r}.")
        if observation is None or actions is None:
            raise ValueError("SFT batch is missing observation or actions.")
        return observation, actions

    @staticmethod
    def _unpack_rlt_history(data: Any) -> tuple[Any, Any]:
        if isinstance(data, dict) and "rlt_hist" in data:
            return data.get("rlt_hist"), data.get("rlt_hist_valid")
        return None, None

    def _observation_to_device(self, observation: Any) -> Observation:
        observation = Observation.from_observation_like(observation)
        device = self.device

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        return Observation(
            images={k: _move(v) for k, v in observation.images.items()},
            image_masks={k: _move(v) for k, v in observation.image_masks.items()},
            state=_move(observation.state),
            tokenized_prompt=_move(observation.tokenized_prompt),
            tokenized_prompt_mask=_move(observation.tokenized_prompt_mask),
            token_ar_mask=_move(observation.token_ar_mask),
            token_loss_mask=_move(observation.token_loss_mask),
            pcd_xyz=_move(observation.pcd_xyz),
        )

    def _actions_to_device(self, actions: Any) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        model_action_dim = self.model.action_dim
        if actions.dim() != 3:
            raise ValueError(
                "SFT actions must have shape [B, action_horizon, D]; "
                f"got {tuple(actions.shape)}."
            )
        if actions.shape[-1] == model_action_dim:
            return actions.to(device=self.device, dtype=torch.float32)
        raise ValueError(
            "SFT actions must arrive normalized + padded to the model action "
            f"dim {model_action_dim} (the openpi_rlinf SFT data loader applies the "
            f"openpi transform pipeline before collation); got last dim "
            f"{actions.shape[-1]}."
        )

    def _sft_forward_with_rlt_prefix(
        self,
        observation: Observation,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute VLA loss while retaining the prefix hidden states for RLT."""
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

        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = torch.cumsum(input_mask.int(), dim=1) - 1

        lang_len = (
            0
            if observation.tokenized_prompt is None
            else int(observation.tokenized_prompt.shape[1])
        )
        prefix_len = int(prefix_tokens.shape[1])
        image_len = prefix_len - lang_len
        topk_ratio = float(getattr(self.rlt_cfg, "rlt_recon_topk_ratio", 1.0))
        capture = topk_ratio < 1.0 and image_len > 0 and lang_len > 0
        self.model.llm.capture_text_image_span = (
            (image_len, prefix_len) if capture else None
        )

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
        prefix_out, prefix_mask = self._select_rlt_prefix_embeddings(
            prefix_out.detach(), prefix_mask, observation.tokenized_prompt
        )
        self.model.llm.capture_text_image_span = None
        scores = getattr(self.model.llm.layers[-1].attn, "last_text_image_scores", None)
        from rlinf.models.embodiment.modules.rlt_mem_write import bottleneck_token_masks

        encode_mask, recon_mask = bottleneck_token_masks(
            prefix_mask,
            scores,
            topk_ratio,
        )
        return loss, prefix_out, prefix_mask, encode_mask, recon_mask

    def _rlt_prefix_from_observation(
        self, observation: Observation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Prefix hidden states for one observation, without the action expert.

        Dropout is off so the reconstruction target does not depend on a mask
        that the memory write cannot see. Image augmentation still follows
        ``self.training``.
        """
        train_images = self.training
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                observation = pi0_model_module.preprocess_observation(
                    observation, train=train_images
                )
                observation = pi0_model_module._observation_to_dtype(
                    observation, self.model.embed_dtype
                )
                prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(
                    observation
                )
                attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
                positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
                lang_len = (
                    0
                    if observation.tokenized_prompt is None
                    else int(observation.tokenized_prompt.shape[1])
                )
                prefix_len = int(prefix_tokens.shape[1])
                image_len = prefix_len - lang_len
                topk_ratio = float(getattr(self.rlt_cfg, "rlt_recon_topk_ratio", 1.0))
                capture = topk_ratio < 1.0 and image_len > 0 and lang_len > 0
                self.model.llm.capture_text_image_span = (
                    (image_len, prefix_len) if capture else None
                )
                prefix_out = self.model.llm(
                    [prefix_tokens, None],
                    positions=positions,
                    mask=attn_mask,
                    adarms_cond=[None, None],
                )[0][0]
                prefix_out, prefix_mask = self._select_rlt_prefix_embeddings(
                    prefix_out, prefix_mask, observation.tokenized_prompt
                )
                self.model.llm.capture_text_image_span = None
                scores = getattr(
                    self.model.llm.layers[-1].attn, "last_text_image_scores", None
                )
                from rlinf.models.embodiment.modules.rlt_mem_write import (
                    bottleneck_token_masks,
                )

                encode_mask, recon_mask = bottleneck_token_masks(
                    prefix_mask, scores, topk_ratio
                )
                return prefix_out, encode_mask, recon_mask
        finally:
            self.model.llm.capture_text_image_span = None
            self.model.train(was_training)

    def _window_mem_loss(
        self, hist: Any, hist_valid: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train ``mem`` on the sampled suffix and reconstruct its last chunk."""
        if not isinstance(hist, dict):
            raise TypeError(
                f"rlt_hist must be a dict of window tensors; got {type(hist)!r}."
            )
        if not torch.is_tensor(hist_valid):
            hist_valid = torch.as_tensor(hist_valid)
        hist_valid = hist_valid.to(device=self.device, dtype=torch.bool)
        if hist_valid.dim() != 2:
            raise ValueError(
                "rlt_hist_valid must have shape [batch, steps]; "
                f"got {tuple(hist_valid.shape)}."
            )
        hist_valid = self.mem._crop_window_valid(
            hist_valid,
            int(self.rlt_cfg.rlt_mem_len_min),
            int(self.rlt_cfg.rlt_mem_len_max),
        )
        batch, steps = hist_valid.shape
        prefixes = None
        masks = None
        recons = None
        for step in range(steps):
            if not bool(hist_valid[:, step].any()):
                continue
            observation = self._observation_to_device(_index_time(hist, step))
            prefix, mask, recon = self._rlt_prefix_from_observation(observation)
            if prefixes is None:
                prefixes = prefix.new_zeros(
                    batch, steps, prefix.shape[1], prefix.shape[-1]
                )
                masks = mask.new_zeros(batch, steps, mask.shape[-1])
                if recon is not None:
                    recons = torch.zeros(
                        batch,
                        steps,
                        recon.shape[-1],
                        device=prefix.device,
                        dtype=torch.bool,
                    )
            if prefix.shape[1] != prefixes.shape[2]:
                raise ValueError(
                    "RLT history steps produced different prefix lengths: "
                    f"{prefix.shape[1]} vs {prefixes.shape[2]}."
                )
            prefixes[:, step] = prefix
            masks[:, step] = mask
            if recons is not None:
                if recon is None:
                    recons[:, step] = True
                else:
                    recons[:, step] = recon.to(dtype=torch.bool)
        if prefixes is None:
            return self.mem._empty_window_loss(batch, self.device)
        if not self.rlt_cfg.rlt_use_mask:
            masks = None
            recons = None
        mem_param = next(self.mem.parameters())
        prefixes = prefixes.to(device=mem_param.device, dtype=mem_param.dtype)
        hist_valid = hist_valid.to(device=mem_param.device)
        if masks is not None:
            masks = masks.to(device=mem_param.device)
        if recons is not None:
            recons = recons.to(device=mem_param.device)
        return self.mem.sequence_loss(
            prefixes,
            masks,
            hist_valid,
            hist_recon_mask=recons,
        )
