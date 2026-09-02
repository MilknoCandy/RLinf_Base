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

"""Transition helpers for long-horizon RLT Stage 2.

This is a separate long-horizon variant of ``rlinf.algorithms.rlt.transition``.
It adds the raw history window to the RLT transition state while keeping the
same replay structure.
"""

from __future__ import annotations

from typing import Any

from rlinf.utils.nested_dict_process import copy_dict_tensor

RLT_LH_OBS_KEYS = ("z_rl", "proprio", "ref_chunk", "history")
RLT_LH_TRANSITION_PREFIX = "rlt_lh_transition_"

_LH_ENV_TYPES = {"calvin"}


def use_lh_simulator_transition_replay(cfg: Any) -> bool:
    """Return True when the train env should store one replay row per chunk.

    The long-horizon path is opt-in via ``env.train.rlt_long_horizon`` or is
    selected for known long-horizon RLT environment types. ``env.eval`` is
    also honored so eval-only long-horizon runs still attach history.
    """

    def _is_lh_env(env_cfg: Any) -> bool:
        if env_cfg is None:
            return False
        if bool(env_cfg.get("rlt_long_horizon", False)):
            return True
        return str(env_cfg.get("env_type", "")) in _LH_ENV_TYPES

    return _is_lh_env(cfg.env.get("train", None)) or _is_lh_env(
        cfg.env.get("eval", None)
    )


def extract_rlt_lh_obs_from_forward_inputs(
    forward_inputs: dict[str, Any],
    *,
    transition: bool = False,
) -> dict[str, Any]:
    prefix = RLT_LH_TRANSITION_PREFIX if transition else ""
    missing = [
        f"{prefix}{key}"
        for key in RLT_LH_OBS_KEYS
        if f"{prefix}{key}" not in forward_inputs
    ]
    if missing:
        raise ValueError(
            f"Missing long-horizon RLT forward_inputs keys: {missing}. Ensure "
            "the rollout worker populates history with predict_rlt_lh_actions."
        )
    return copy_dict_tensor(
        {key: forward_inputs[f"{prefix}{key}"] for key in RLT_LH_OBS_KEYS}
    )


def update_rlt_lh_transitions(
    stage_id: int,
    pending_obs: list[dict[str, Any] | None],
    rollout_results: list[Any],
    rollout_result: Any,
    *,
    cache_current: bool,
) -> None:
    if pending_obs[stage_id] is not None:
        next_obs = extract_rlt_lh_obs_from_forward_inputs(
            rollout_result.forward_inputs,
            transition=True,
        )
        rollout_results[stage_id].append_transitions(
            pending_obs[stage_id],
            next_obs,
        )
        pending_obs[stage_id] = None

    if cache_current:
        pending_obs[stage_id] = extract_rlt_lh_obs_from_forward_inputs(
            rollout_result.forward_inputs
        )
