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

The initial CALVIN RLT adapter keeps switching simple: it uses an
``always_on`` RLT actor switch. This is enough to validate the RLT
actor-critic path on CALVIN without adding per-subtask phase detection.
``current_task_idx`` / ``subtask_success`` are already available inside
``CalvinEnv`` and can be used later to implement task-aware switching.

For the history-injection experiment, ``history_len > 0`` appends the
previous ``history_len`` pairs of ``(proprio, action)`` to the ``states``
vector. Because the ``openpi`` RLT feature model returns the raw configured
state as ``proprio``, the Stage2 actor sees this enlarged state without
changing the VLA prefix/action code path.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from rlinf.envs.calvin.calvin_gym_env import CalvinEnv


class CalvinRLTEnv(CalvinEnv):
    """CALVIN env exposing RLT switch flags and optional history state."""

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
        self.history_len = int(getattr(cfg, "history_len", 0) or 0)
        self._rlt_switch_cfg = getattr(cfg, "rlt_policy_switch", None)

        self._rlt_switch_state: dict[str, torch.Tensor] | None = None
        self._history_proprio: torch.Tensor | None = None
        self._history_actions: torch.Tensor | None = None
        self._history_pos: torch.Tensor | None = None

        self._init_rlt_switch()
        self._init_history()

    # ------------------------------------------------------------------
    # RLT switching
    # ------------------------------------------------------------------
    def _rlt_switch_enabled(self) -> bool:
        return self._rlt_switch_cfg is not None and bool(
            self._rlt_switch_cfg.get("enable", False)
        )

    def _init_rlt_switch(self) -> None:
        if not self._rlt_switch_enabled():
            return
        self._rlt_switch_state = {
            "rlt_switch_flags": torch.ones(
                (self.num_envs,), dtype=torch.bool
            ),
        }

    def _export_rlt_switch_info(self) -> dict[str, torch.Tensor]:
        batch_size = self.num_envs
        if self._rlt_switch_state is None:
            rlt_switch_flags = torch.zeros(
                batch_size, dtype=torch.bool
            )
        else:
            rlt_switch_flags = self._rlt_switch_state["rlt_switch_flags"]
        return {
            "rlt_switch_flags": rlt_switch_flags.reshape(batch_size, 1),
            "intervene_flag": torch.zeros(
                batch_size, 1, dtype=torch.bool
            ),
        }

    def _attach_rlt_switch_info(self, infos: dict[str, Any]) -> None:
        if isinstance(infos, dict):
            infos.update(self._export_rlt_switch_info())

    # ------------------------------------------------------------------
    # History injection
    # ------------------------------------------------------------------
    def _init_history(self) -> None:
        if self.history_len <= 0:
            return
        self._history_proprio = torch.zeros(
            (self.num_envs, self.history_len, 7), dtype=torch.float32
        )
        self._history_actions = torch.zeros(
            (self.num_envs, self.history_len, 7), dtype=torch.float32
        )
        self._history_pos = torch.zeros(
            self.num_envs, dtype=torch.long
        )

    def _reset_history(self, env_idx: Optional[Any] = None) -> None:
        if self.history_len <= 0:
            return
        if env_idx is None:
            self._history_proprio.zero_()
            self._history_actions.zero_()
            self._history_pos.zero_()
            return

        indices = np.asarray(env_idx).reshape(-1)
        if indices.size == 0:
            return
        self._history_proprio[indices].zero_()
        self._history_actions[indices].zero_()
        self._history_pos[indices] = 0

    def _append_history(
        self,
        proprio: Any,
        actions: Any,
        dones: Optional[Any],
    ) -> None:
        if self.history_len <= 0:
            return
        proprio = torch.as_tensor(proprio, dtype=torch.float32).reshape(
            self.num_envs, -1
        )
        actions = torch.as_tensor(actions, dtype=torch.float32).reshape(
            self.num_envs, -1
        )

        write_pos = self._history_pos
        rows = torch.arange(self.num_envs)
        self._history_proprio[rows, write_pos, :] = proprio[:, :7]
        self._history_actions[rows, write_pos, :] = actions[:, :7]
        self._history_pos = (write_pos + 1) % self.history_len

        if dones is not None:
            done = torch.as_tensor(dones, dtype=torch.bool).reshape(
                self.num_envs
            )
            if done.any():
                done_idx = done.nonzero(as_tuple=False).reshape(-1)
                self._reset_history(done_idx.numpy())

    def _append_history_to_states(self, states: Any) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=torch.float32)
        if self.history_len <= 0:
            return states
        history = torch.cat(
            [self._history_proprio, self._history_actions], dim=-1
        ).reshape(self.num_envs, -1)
        return torch.cat([states, history], dim=-1)

    def _wrap_obs(self, obs_list):
        obs = super()._wrap_obs(obs_list)
        if self.history_len > 0:
            obs["states"] = self._append_history_to_states(obs["states"])
        return obs

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(
        self,
        env_idx: Optional[Any] = None,
        reset_state_ids=None,
    ):
        self._reset_history(env_idx)
        obs, infos = super().reset(
            env_idx=env_idx,
            reset_state_ids=reset_state_ids,
        )
        self._attach_rlt_switch_info(infos)
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        obs, step_reward, terminations, truncations, infos = super().step(
            actions, auto_reset=auto_reset
        )
        if self.history_len > 0:
            current_states = obs["states"][..., :7]
            dones = torch.logical_or(terminations, truncations)
            self._append_history(
                current_states,
                actions,
                dones,
            )
        self._attach_rlt_switch_info(infos)
        return obs, step_reward, terminations, truncations, infos


__all__ = ["CalvinRLTEnv"]
