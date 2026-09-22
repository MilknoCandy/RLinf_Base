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

"""Train the B2 RLT encoder on dumped I_t sequences (no live sim, no VLM)."""

from __future__ import annotations

import json
import logging
from itertools import cycle
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rlinf.algorithms.rlt.b2_sft import (
    B2DumpDataset,
    B2SFTModel,
    build_encoder,
    collate_b2_episodes,
    load_encoder_weights,
)

logger = logging.getLogger("rlt_b2_sft")


def _save_checkpoint(model: B2SFTModel, path: Path, step: int, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "encoder": model.encoder.state_dict(),
        "heads": model.heads.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(payload, path)
    logger.info("Saved B2 SFT checkpoint to %s", path)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_rlt_b2_sft",
)
def main(cfg) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(cfg.model)
    if cfg.model.get("encoder_ckpt"):
        load_encoder_weights(encoder, cfg.model.encoder_ckpt)
    model = B2SFTModel(
        encoder,
        lambda_dist=float(cfg.loss.lambda_dist),
        lambda_success=float(cfg.loss.lambda_success),
        hidden_dim=cfg.model.get("head_hidden_dim"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=(float(cfg.optim.adam_beta1), float(cfg.optim.adam_beta2)),
        eps=float(cfg.optim.adam_eps),
    )

    dataset = B2DumpDataset(cfg.data.dump_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        collate_fn=collate_b2_episodes,
        drop_last=False,
    )
    batches = cycle(loader)
    save_dir = Path(cfg.runner.save_dir).expanduser()
    clip_grad = float(cfg.optim.clip_grad)
    bptt_k = int(cfg.loss.bptt_k)
    log_interval = int(cfg.runner.log_interval)
    save_interval = int(cfg.runner.save_interval)
    max_steps = int(cfg.runner.max_steps)

    model.train()
    for step in range(1, max_steps + 1):
        batch = next(batches)
        image_tokens = batch["image_tokens"].to(device=device, non_blocking=True)
        image_mask = batch["image_mask"].to(device=device, non_blocking=True)
        distance = batch["distance"].to(device=device, non_blocking=True)
        success = batch["success"].to(device=device, non_blocking=True)
        valid = batch["valid"].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.compute_loss(
            image_tokens=image_tokens,
            image_mask=image_mask,
            distance=distance,
            success=success,
            valid=valid,
            bptt_k=bptt_k,
        )
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if step == 1 or step % log_interval == 0:
            logger.info(
                "B2 SFT step %d loss=%.4f dist=%.4f success=%.4f acc=%.3f n=%.0f",
                step,
                float(metrics["loss"]),
                float(metrics["dist_loss"]),
                float(metrics["success_loss"]),
                float(metrics["success_acc"]),
                float(metrics["num_steps"]),
            )
        if save_interval > 0 and step % save_interval == 0:
            _save_checkpoint(
                model, save_dir / f"step_{step:06d}.pt", step, cfg
            )

    _save_checkpoint(model, save_dir / "final.pt", max_steps, cfg)


if __name__ == "__main__":
    main()
