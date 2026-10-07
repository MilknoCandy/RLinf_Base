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

"""Optional VLM yes/no latch for the RLT precision phase.

Enable with ``algorithm.rlt_phase_gate.mode: vlm``. Short single-task runs
(e.g. bowl) should use ``mode: none`` so the residual actor trains on the
full episode.
"""

from __future__ import annotations

from typing import Any

import torch

DEFAULT_VLM_PHASE_QUESTION = (
    "answer en Looking at the images, is the robot currently in the precise "
    "manipulation phase (grasping, contacting, inserting, or placing) of the "
    "task '{task}', rather than still approaching?"
)


def _episode_done(
    dones: torch.Tensor | None, batch_size: int, device: torch.device
) -> torch.Tensor:
    if dones is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    done = torch.as_tensor(dones, device=device)
    if done.numel() == 1:
        return torch.full(
            (batch_size,),
            bool(done.reshape(-1)[0].item()),
            dtype=torch.bool,
            device=device,
        )
    return done.reshape(batch_size, -1).to(torch.bool).any(dim=-1)


class VLMPhaseGate:
    """Per-stage latch over a frozen VLM's yes/no phase decision."""

    needs_vlm_score = True

    def __init__(
        self,
        *,
        yes_threshold: float = 0.55,
        min_chunks_before_enter: int = 1,
        latch_until_done: bool = True,
        question: str = DEFAULT_VLM_PHASE_QUESTION,
        force_enter_after_chunks: int = 0,
    ):
        if not 0.0 <= yes_threshold <= 1.0:
            raise ValueError(f"yes_threshold must be in [0, 1], got {yes_threshold}.")
        if min_chunks_before_enter < 0:
            raise ValueError(
                "min_chunks_before_enter must be >= 0, "
                f"got {min_chunks_before_enter}."
            )
        self.yes_threshold = float(yes_threshold)
        self.min_chunks_before_enter = int(min_chunks_before_enter)
        self.latch_until_done = bool(latch_until_done)
        self.question = str(question)
        self.force_enter_after_chunks = int(force_enter_after_chunks)
        self._states: dict[int, dict[str, torch.Tensor]] = {}
        self.last_metrics: dict[str, float] = {}

    @classmethod
    def from_cfg(cls, cfg: Any | None) -> VLMPhaseGate:
        cfg = cfg or {}
        return cls(
            yes_threshold=float(cfg.get("yes_threshold", 0.55)),
            min_chunks_before_enter=int(cfg.get("min_chunks_before_enter", 1)),
            latch_until_done=bool(cfg.get("latch_until_done", True)),
            question=str(cfg.get("question", DEFAULT_VLM_PHASE_QUESTION)),
            force_enter_after_chunks=int(cfg.get("force_enter_after_chunks", 0)),
        )

    def _state(
        self, stage_id: int, batch_size: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        state = self._states.get(stage_id)
        if (
            state is None
            or state["latched"].shape[0] != batch_size
            or state["latched"].device != device
        ):
            state = {
                "latched": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "chunk_count": torch.zeros(
                    batch_size, dtype=torch.long, device=device
                ),
            }
            self._states[stage_id] = state
        return state

    def all_latched(self, stage_id: int, batch_size: int, device: torch.device) -> bool:
        state = self._states.get(int(stage_id))
        if state is None or state["latched"].shape[0] != batch_size:
            return False
        return bool(state["latched"].to(device=device).all().item())

    def update(
        self,
        p_yes: torch.Tensor,
        *,
        dones: torch.Tensor | None = None,
        stage_id: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        p_yes = torch.as_tensor(p_yes, dtype=torch.float32)
        if p_yes.dim() > 1:
            p_yes = p_yes.reshape(p_yes.shape[0], -1)[:, 0]
        batch_size = int(p_yes.shape[0])
        device = p_yes.device
        state = self._state(int(stage_id), batch_size, device)
        p_yes = p_yes.to(device=device)
        done = _episode_done(dones, batch_size, device)

        matured = state["chunk_count"] >= self.min_chunks_before_enter
        vlm_enter = matured & (p_yes >= self.yes_threshold)
        if self.force_enter_after_chunks > 0:
            vlm_enter = vlm_enter | (
                state["chunk_count"] >= self.force_enter_after_chunks
            )
        if self.latch_until_done:
            flags = state["latched"] | vlm_enter
        else:
            flags = vlm_enter
        flags = flags & matured

        state["latched"] = flags
        state["chunk_count"] = state["chunk_count"] + 1
        if bool(done.any()):
            state["latched"] = state["latched"] & (~done)
            state["chunk_count"] = torch.where(
                done, torch.zeros_like(state["chunk_count"]), state["chunk_count"]
            )

        metrics = {
            "vlm_phase/enter_rate": float(flags.float().mean().item()),
            "vlm_phase/mean_p_yes": float(p_yes.mean().item()),
            "vlm_phase/mean_chunk_count": float(
                state["chunk_count"].float().mean().item()
            ),
        }
        self.last_metrics = metrics
        return flags, metrics
