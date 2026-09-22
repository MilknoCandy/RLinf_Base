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

"""Loop SFT: teach the RLT encoder to compress dumped history into z."""

from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from rlinf.models.embodiment.modules.rlt_b2_heads import B2ReadoutHeads
from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenEncoder


def load_encoder_weights(encoder: nn.Module, ckpt_path: str | pathlib.Path) -> None:
    """Load ``rlt_module.encoder.*`` (or bare ``encoder.*``) weights into ``encoder``."""
    from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
        _normalize_wrapper_state_dict,
        resolve_full_weights,
    )
    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    resolved = resolve_full_weights(ckpt_path)
    path = pathlib.Path(resolved if resolved is not None else ckpt_path).expanduser()
    loaded = torch.load(str(path), map_location="cpu", weights_only=False)
    if (
        isinstance(loaded, dict)
        and "encoder" in loaded
        and isinstance(loaded["encoder"], dict)
        and loaded["encoder"]
        and all(torch.is_tensor(value) for value in loaded["encoder"].values())
    ):
        encoder.load_state_dict(loaded["encoder"], strict=True)
        return

    state_dict = _normalize_wrapper_state_dict(as_state_dict(loaded))
    encoder_sd: dict[str, torch.Tensor] = {}
    prefixes = ("rlt_module.encoder.", "encoder.")
    for key, tensor in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                encoder_sd[key[len(prefix) :]] = tensor
                break
    if not encoder_sd:
        raise ValueError(
            f"No encoder weights found in {path}. Expected keys starting with "
            "rlt_module.encoder. or encoder."
        )
    encoder.load_state_dict(encoder_sd, strict=True)


def build_encoder(cfg: Any) -> RLTTokenEncoder:
    """Build an encoder whose shape matches the Stage 1 RLT checkpoint."""
    return RLTTokenEncoder(
        input_dim=int(cfg.rlt_input_dim),
        embed_dim=int(cfg.rlt_embed_dim),
        prefix_seq_len=int(cfg.rlt_prefix_seq_len),
        num_layers=int(cfg.rlt_num_layers),
        num_heads=int(cfg.rlt_num_heads),
        mlp_ratio=float(cfg.rlt_mlp_ratio),
        dropout_rate=float(getattr(cfg, "rlt_dropout_rate", 0.0)),
    )


class B2SFTModel(nn.Module):
    """Encoder + readout heads. VLM is not present; inputs are dumped ``I_t``."""

    def __init__(
        self,
        encoder: RLTTokenEncoder,
        *,
        lambda_dist: float = 1.0,
        lambda_success: float = 1.0,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.heads = B2ReadoutHeads(encoder.embed_dim, hidden_dim=hidden_dim)
        self.lambda_dist = float(lambda_dist)
        self.lambda_success = float(lambda_success)

    def unroll(
        self,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor | None,
        *,
        bptt_k: int,
    ) -> torch.Tensor:
        """Run the encoder loop. Returns ``z`` with shape ``[B, T, D]``.

        ``t=0`` uses the learned ``z_init``. Gradients through previous ``z``
        are truncated every ``bptt_k`` steps. ``bptt_k <= 0`` keeps full BPTT.
        """
        batch_size, time_len = image_tokens.shape[:2]
        z_prev: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []
        for time_index in range(time_len):
            step_mask = None if image_mask is None else image_mask[:, time_index]
            z_t = self.encoder(
                image_tokens[:, time_index],
                step_mask,
                rl_token=z_prev,
            )
            outputs.append(z_t)
            z_next = z_t
            if bptt_k > 0 and (time_index + 1) % bptt_k == 0:
                z_next = z_t.detach()
            z_prev = z_next
        stacked = torch.cat(outputs, dim=1)
        if stacked.shape[0] != batch_size:
            raise RuntimeError(
                f"Encoder loop batch mismatch: {stacked.shape[0]} vs {batch_size}."
            )
        return stacked

    def compute_loss(
        self,
        *,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor | None,
        distance: torch.Tensor,
        success: torch.Tensor,
        valid: torch.Tensor,
        bptt_k: int = 4,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Distance MSE + success BCE on ``t >= 1`` valid steps."""
        z = self.unroll(image_tokens, image_mask, bptt_k=bptt_k)
        pred_dist, pred_logit = self.heads(z)
        time_len = z.shape[1]
        time_index = torch.arange(time_len, device=z.device)
        step_mask = valid.to(device=z.device, dtype=torch.bool) & (
            time_index.view(1, -1) >= 1
        )
        denom = torch.clamp(step_mask.to(dtype=torch.float32).sum(), min=1.0)
        target_dist = distance.to(device=z.device, dtype=torch.float32)
        dist_err = torch.square(pred_dist.to(dtype=torch.float32) - target_dist)
        success_target = success.to(device=z.device, dtype=torch.float32)
        success_err = F.binary_cross_entropy_with_logits(
            pred_logit.to(dtype=torch.float32),
            success_target,
            reduction="none",
        )
        mask_f = step_mask.to(dtype=torch.float32)
        dist_loss = (dist_err * mask_f).sum() / denom
        success_loss = (success_err * mask_f).sum() / denom
        loss = self.lambda_dist * dist_loss + self.lambda_success * success_loss
        with torch.no_grad():
            pred_ok = (pred_logit >= 0).to(dtype=torch.float32)
            acc = ((pred_ok == success_target) * mask_f).sum() / denom
        metrics = {
            "loss": loss.detach(),
            "dist_loss": dist_loss.detach(),
            "success_loss": success_loss.detach(),
            "success_acc": acc,
            "num_steps": denom.detach(),
        }
        return loss, metrics


class B2DumpDataset(Dataset):
    """One dumped episode per ``.pt`` file."""

    def __init__(self, dump_dir: str | pathlib.Path):
        self.dump_dir = pathlib.Path(dump_dir).expanduser()
        self.paths = sorted(self.dump_dir.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(
                f"No B2 dump episodes found in {self.dump_dir}. "
                "Run the dump config first (rlt_b2_dump_dir)."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return torch.load(self.paths[index], map_location="cpu", weights_only=False)


def collate_b2_episodes(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length episodes to ``[B, T, N, D]``."""
    time_len = max(int(item["image_tokens"].shape[0]) for item in batch)
    token_len = max(int(item["image_tokens"].shape[1]) for item in batch)
    embed_dim = int(batch[0]["image_tokens"].shape[-1])
    batch_size = len(batch)
    image_tokens = torch.zeros(batch_size, time_len, token_len, embed_dim)
    image_mask = torch.zeros(batch_size, time_len, token_len, dtype=torch.bool)
    distance = torch.zeros(batch_size, time_len, dtype=torch.float32)
    success = torch.zeros(batch_size, time_len, dtype=torch.float32)
    valid = torch.zeros(batch_size, time_len, dtype=torch.bool)
    for index, item in enumerate(batch):
        tokens = item["image_tokens"].to(dtype=torch.float32)
        step_t, step_n = tokens.shape[:2]
        image_tokens[index, :step_t, :step_n] = tokens
        if "image_mask" in item and item["image_mask"] is not None:
            image_mask[index, :step_t, :step_n] = item["image_mask"].to(dtype=torch.bool)
        else:
            image_mask[index, :step_t, :step_n] = True
        distance[index, :step_t] = item["distance"].to(dtype=torch.float32).reshape(-1)
        success[index, :step_t] = item["success"].to(dtype=torch.float32).reshape(-1)
        valid[index, :step_t] = True
    return {
        "image_tokens": image_tokens,
        "image_mask": image_mask,
        "distance": distance,
        "success": success,
        "valid": valid,
    }
