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

"""CALVIN RLT environment with actor/reference switching.

Supports:
- ``trigger_mode: always_on`` — RLT actor for the full episode (ablation/smoke).
- ``trigger_mode: auto`` — enter the RLT actor once ``current_task_idx`` reaches
  ``auto_gate.min_task_idx`` (option A: index-gated critical phase), matching
  the original RLT design of refining only the hard phase while the frozen VLA
  handles earlier subtasks via ``ref_chunk``.

``latch_until_done`` keeps the actor switch on until episode reset.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from rlinf.envs.calvin.calvin_gym_env import CalvinEnv


class CalvinRLTEnv(CalvinEnv):
    """CALVIN env exposing RLT switch flags for Stage 2 rollout."""

    _RLT_FULL_TASK = "full_task"
    _RLT_CRITICAL_PHASE = "critical_phase"
    _RLT_ALWAYS_ON_TRIGGER = "always_on"
    _RLT_AUTO_TRIGGER = "auto"

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )
        self.record_metrics = record_metrics
        self._rlt_switch_cfg = getattr(cfg, "rlt_policy_switch", None)
        self._rlt_switch_state: dict[str, torch.Tensor] | None = None
        self._init_rlt_switch()

    def _rlt_switch_enabled(self) -> bool:
        return self._rlt_switch_cfg is not None and bool(
            self._rlt_switch_cfg.get("enable", False)
        )

    def _init_rlt_switch(self) -> None:
        if not self._rlt_switch_enabled():
            return

        task_mode = str(self._rlt_switch_cfg.get("task_mode", self._RLT_FULL_TASK))
        trigger_mode = str(
            self._rlt_switch_cfg.get("trigger_mode", self._RLT_AUTO_TRIGGER)
        )
        if task_mode not in {self._RLT_FULL_TASK, self._RLT_CRITICAL_PHASE}:
            raise ValueError(
                "CALVIN RLT task_mode must be 'full_task' or 'critical_phase', "
                f"got {task_mode!r}."
            )
        if trigger_mode not in {
            self._RLT_ALWAYS_ON_TRIGGER,
            self._RLT_AUTO_TRIGGER,
        }:
            raise ValueError(
                "CALVIN RLT trigger_mode must be 'always_on' or 'auto', "
                f"got {trigger_mode!r}."
            )
        if trigger_mode == self._RLT_AUTO_TRIGGER:
            auto_gate = self._rlt_switch_cfg.get("auto_gate", {}) or {}
            min_task_idx = int(auto_gate.get("min_task_idx", 2))
            if min_task_idx < 0:
                raise ValueError(
                    "CALVIN RLT auto_gate.min_task_idx must be >= 0, "
                    f"got {min_task_idx}."
                )

        self._rlt_switch_state = self._init_rlt_switch_state(self.num_envs)

    def _init_rlt_switch_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        task_mode = str(self._rlt_switch_cfg.get("task_mode", self._RLT_FULL_TASK))
        trigger_mode = str(
            self._rlt_switch_cfg.get("trigger_mode", self._RLT_AUTO_TRIGGER)
        )
        start_active = (
            task_mode == self._RLT_CRITICAL_PHASE
            or trigger_mode == self._RLT_ALWAYS_ON_TRIGGER
        )
        return {
            "rlt_switch_flags": torch.full(
                (batch_size,),
                start_active,
                dtype=torch.bool,
            ),
        }

    def _reset_rlt_switch(self, env_idx: Optional[Any] = None) -> None:
        if self._rlt_switch_state is None:
            return
        new_state = self._init_rlt_switch_state(self.num_envs)
        if env_idx is None:
            self._rlt_switch_state = new_state
            return
        indices = np.asarray(env_idx).reshape(-1)
        if indices.size == 0:
            return
        for key, value in new_state.items():
            self._rlt_switch_state[key][indices] = value[indices]

    def _current_task_idx_tensor(self) -> torch.Tensor:
        return torch.as_tensor(self.current_task_idx, dtype=torch.long)

    def _rlt_auto_enter_actor(self) -> torch.Tensor:
        auto_gate = self._rlt_switch_cfg.get("auto_gate", {}) or {}
        min_task_idx = int(auto_gate.get("min_task_idx", 2))
        task_idx = self._current_task_idx_tensor()
        # idx == 5 means the 5-subtask chain finished; keep latch semantics via
        # previous flags rather than re-entering from a completed episode.
        return (task_idx >= min_task_idx) & (task_idx <= 4)

    def _update_rlt_switch(self) -> None:
        if self._rlt_switch_state is None:
            return

        task_mode = str(self._rlt_switch_cfg.get("task_mode", self._RLT_FULL_TASK))
        trigger_mode = str(
            self._rlt_switch_cfg.get("trigger_mode", self._RLT_AUTO_TRIGGER)
        )
        if (
            task_mode == self._RLT_CRITICAL_PHASE
            or trigger_mode == self._RLT_ALWAYS_ON_TRIGGER
        ):
            enter_actor = torch.ones(self.num_envs, dtype=torch.bool)
        elif (
            task_mode == self._RLT_FULL_TASK and trigger_mode == self._RLT_AUTO_TRIGGER
        ):
            enter_actor = self._rlt_auto_enter_actor()
        else:
            raise ValueError(
                "rlt_policy_switch supports task_mode in "
                f"{self._RLT_FULL_TASK, self._RLT_CRITICAL_PHASE} and trigger_mode in "
                f"{self._RLT_AUTO_TRIGGER, self._RLT_ALWAYS_ON_TRIGGER}, got "
                f"{task_mode=} {trigger_mode=}."
            )

        previous = self._rlt_switch_state["rlt_switch_flags"]
        latch_until_done = bool(self._rlt_switch_cfg.get("latch_until_done", True))
        if latch_until_done:
            self._rlt_switch_state["rlt_switch_flags"] = previous | enter_actor
        else:
            self._rlt_switch_state["rlt_switch_flags"] = enter_actor

    def _export_rlt_switch_info(self) -> dict[str, torch.Tensor]:
        batch_size = self.num_envs
        task_idx = self._current_task_idx_tensor()
        if self._rlt_switch_state is None:
            rlt_switch_flags = torch.zeros(batch_size, dtype=torch.bool)
        else:
            rlt_switch_flags = self._rlt_switch_state["rlt_switch_flags"]
        return {
            "rlt_switch_flags": rlt_switch_flags.reshape(batch_size, 1),
            "intervene_flag": torch.zeros(batch_size, 1, dtype=torch.bool),
            "current_task_idx": task_idx.reshape(batch_size, 1),
        }

    def _attach_rlt_switch_info(self, infos: dict[str, Any]) -> None:
        if isinstance(infos, dict):
            infos.update(self._export_rlt_switch_info())

    def reset(
        self,
        env_idx: Optional[Any] = None,
        reset_state_ids=None,
    ):
        self._reset_rlt_switch(env_idx)
        obs, infos = super().reset(
            env_idx=env_idx,
            reset_state_ids=reset_state_ids,
        )
        self._update_rlt_switch()
        self._attach_rlt_switch_info(infos)
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        obs, step_reward, terminations, truncations, infos = super().step(
            actions, auto_reset=auto_reset
        )
        self._update_rlt_switch()
        self._attach_rlt_switch_info(infos)
        return obs, step_reward, terminations, truncations, infos


__all__ = ["CalvinRLTEnv"]
