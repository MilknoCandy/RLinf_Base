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

"""IsaacLab RLT environment with actor/reference switching."""

from __future__ import annotations

from typing import Any, Optional

import torch

from rlinf.envs.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv


class IsaaclabRLTEnv(IsaaclabStackCubeEnv):
    """IsaacLab env exposing RLT switch flags for Stage 2 rollout.

    The initial IsaacLab RLT adapter deliberately keeps switching simple: it
    supports ``trigger_mode: always_on`` (and both ``full_task`` /
    ``critical_phase`` task modes) so the actor can take over once the RLT
    schedule reaches ``ready_for_online``. A task-specific auto gate for the
    stack-cube critical phase can be added later by overriding
    ``_export_rlt_switch_info``.
    """

    _RLT_FULL_TASK = "full_task"
    _RLT_CRITICAL_PHASE = "critical_phase"
    _RLT_ALWAYS_ON_TRIGGER = "always_on"

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
            self._rlt_switch_cfg.get("trigger_mode", self._RLT_ALWAYS_ON_TRIGGER)
        )
        if task_mode not in {self._RLT_FULL_TASK, self._RLT_CRITICAL_PHASE}:
            raise ValueError(
                "IsaacLab RLT task_mode must be 'full_task' or 'critical_phase', "
                f"got {task_mode!r}."
            )
        if trigger_mode != self._RLT_ALWAYS_ON_TRIGGER:
            raise ValueError(
                "IsaacLab RLT currently supports only trigger_mode='always_on'; "
                f"got {trigger_mode!r}."
            )

        self._rlt_switch_state = {
            "rlt_switch_flags": torch.full(
                (self.num_envs,),
                True,
                dtype=torch.bool,
                device=self.device,
            ),
        }

    def _reset_rlt_switch(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if self._rlt_switch_state is None:
            return
        if env_ids is None:
            self._rlt_switch_state["rlt_switch_flags"].fill_(True)
            return
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        self._rlt_switch_state["rlt_switch_flags"][env_ids] = True

    def _export_rlt_switch_info(self) -> dict[str, torch.Tensor]:
        batch_size = self.num_envs
        if self._rlt_switch_state is None:
            rlt_switch_flags = torch.zeros(
                batch_size, dtype=torch.bool, device=self.device
            )
        else:
            rlt_switch_flags = self._rlt_switch_state["rlt_switch_flags"]
        return {
            "rlt_switch_flags": rlt_switch_flags[:, None],
            "intervene_flag": torch.zeros(
                batch_size, 1, dtype=torch.bool, device=self.device
            ),
        }

    def _attach_rlt_switch_info(self, infos: dict[str, Any]) -> None:
        infos.update(self._export_rlt_switch_info())

    def reset(self, seed: Optional[int] = None, env_ids: Optional[torch.Tensor] = None):
        obs, infos = super().reset(seed=seed, env_ids=env_ids)
        self._reset_rlt_switch(env_ids)
        infos = dict(infos or {})
        self._attach_rlt_switch_info(infos)
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        obs, step_reward, terminations, truncations, infos = super().step(
            actions, auto_reset=auto_reset
        )
        infos = dict(infos or {})
        self._attach_rlt_switch_info(infos)
        return obs, step_reward, terminations, truncations, infos


__all__ = ["IsaaclabRLTEnv"]
