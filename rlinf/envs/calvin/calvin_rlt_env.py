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
            "expert_takeover_active": torch.zeros(
                (self.num_envs,), dtype=torch.bool
            ),
            "stalled_chunks": torch.zeros(
                (self.num_envs,), dtype=torch.float32
            ),
        }

    # ------------------------------------------------------------------
    # Expert takeover
    # ------------------------------------------------------------------
    def _expert_takeover_cfg(self) -> dict[str, Any]:
        if self._rlt_switch_cfg is None:
            return {}
        return self._rlt_switch_cfg.get("expert_takeover", {}) or {}

    def _expert_takeover_enabled(self) -> bool:
        return bool(self._expert_takeover_cfg().get("enable", False))

    def _expert_takeover_trigger_mode(self) -> str:
        return str(
            self._expert_takeover_cfg().get(
                "trigger_mode", "stalled_progress"
            )
        )

    def _reset_expert_takeover(self, env_idx: Optional[Any] = None) -> None:
        if self._rlt_switch_state is None:
            return
        if env_idx is None:
            self._rlt_switch_state["expert_takeover_active"].zero_()
            self._rlt_switch_state["stalled_chunks"].zero_()
            return

        indices = np.asarray(env_idx).reshape(-1)
        if indices.size == 0:
            return
        self._rlt_switch_state["expert_takeover_active"][indices] = False
        self._rlt_switch_state["stalled_chunks"][indices] = 0.0

    def _update_expert_takeover_chunk(
        self,
        start_task_idx: Any,
        terminations: Any,
        truncations: Any,
    ) -> None:
        if self._rlt_switch_state is None:
            return
        state = self._rlt_switch_state
        if not self._expert_takeover_enabled() or self.cfg.is_eval:
            state["expert_takeover_active"].zero_()
            state["stalled_chunks"].zero_()
            return

        trigger_mode = self._expert_takeover_trigger_mode()
        if trigger_mode != "stalled_progress":
            # always_on / critical_phase are computed on the fly in
            # _intervene_flag and need no latch state here.
            state["expert_takeover_active"].zero_()
            state["stalled_chunks"].zero_()
            return

        gate_cfg = self._expert_takeover_cfg().get("gate", {}) or {}
        end_task_idx = torch.as_tensor(
            np.asarray(self.current_task_idx, dtype=np.int64)
        )
        start_task_idx = torch.as_tensor(
            np.asarray(start_task_idx, dtype=np.int64)
        )
        done = (
            torch.as_tensor(terminations, dtype=torch.bool)
            | torch.as_tensor(truncations, dtype=torch.bool)
        )
        if done.dim() > 1:
            done = done.any(dim=1)
        done = done.reshape(self.num_envs)

        # CALVIN is long-horizon: progress means a subtask completed and
        # current_task_idx advanced; the episode is complete at idx == 5.
        in_actor_phase = state["rlt_switch_flags"].clone()
        progressed = end_task_idx > start_task_idx
        stalled = in_actor_phase & (~done) & (~progressed)

        stuck_before_takeover = max(
            1,
            int(gate_cfg.get("stuck_chunks_before_takeover", 3)),
        )
        state["stalled_chunks"] = torch.where(
            stalled,
            state["stalled_chunks"] + 1.0,
            torch.zeros_like(state["stalled_chunks"]),
        )
        trigger_now = stalled & (
            state["stalled_chunks"] >= float(stuck_before_takeover)
        )

        active_before = state["expert_takeover_active"] & (~done)
        state["expert_takeover_active"] = active_before | trigger_now

    def _intervene_flag(self) -> torch.Tensor:
        if self._rlt_switch_state is None:
            return torch.zeros(self.num_envs, dtype=torch.bool)
        if not self._expert_takeover_enabled() or self.cfg.is_eval:
            return torch.zeros(self.num_envs, dtype=torch.bool)

        trigger_mode = self._expert_takeover_trigger_mode()
        if trigger_mode == "always_on":
            return torch.ones(self.num_envs, dtype=torch.bool)
        if trigger_mode == "critical_phase":
            return self._rlt_switch_state["rlt_switch_flags"].clone()
        if trigger_mode == "stalled_progress":
            return self._rlt_switch_state["expert_takeover_active"].clone()
        raise ValueError(
            "rlt_policy_switch.expert_takeover.trigger_mode supports "
            "'stalled_progress', 'critical_phase', and 'always_on', got "
            f"{trigger_mode!r}."
        )

    def _export_rlt_switch_info(self) -> dict[str, torch.Tensor]:
        batch_size = self.num_envs
        if self._rlt_switch_state is None:
            rlt_switch_flags = torch.zeros(
                batch_size, dtype=torch.bool
            )
            intervene_flag = torch.zeros(
                batch_size, dtype=torch.bool
            )
        else:
            rlt_switch_flags = self._rlt_switch_state["rlt_switch_flags"]
            intervene_flag = self._intervene_flag()
        return {
            "rlt_switch_flags": rlt_switch_flags.reshape(batch_size, 1),
            "intervene_flag": intervene_flag.reshape(batch_size, 1),
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
        self._reset_expert_takeover(env_idx)
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

    def chunk_step(self, chunk_actions):
        # CALVIN long-horizon expert takeover is evaluated once per action
        # chunk (matching ManiSkill's stuck_chunks_before_takeover semantics),
        # not once per env step.
        start_task_idx = np.asarray(
            self.current_task_idx, dtype=np.int64
        ).copy()
        (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        ) = super().chunk_step(chunk_actions)
        self._update_expert_takeover_chunk(
            start_task_idx,
            chunk_terminations,
            chunk_truncations,
        )
        if infos_list:
            self._attach_rlt_switch_info(infos_list[-1])
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )


__all__ = ["CalvinRLTEnv"]
