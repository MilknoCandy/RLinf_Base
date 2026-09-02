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

"""Rollout prediction for the long-horizon RLT Stage-2 path."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from rlinf.algorithms.rlt.route import RLTRouteContext
from rlinf.algorithms.rlt_lh.route import build_rlt_lh_route
from rlinf.algorithms.rlt_lh.transition import (
    RLT_LH_OBS_KEYS,
    RLT_LH_TRANSITION_PREFIX,
)

__all__ = ["predict_rlt_lh_actions"]


def _append_rlt_lh_transition_obs(
    *,
    feature_model: Any,
    result: dict[str, Any],
    rlt_obs: dict[str, torch.Tensor],
    final_obs: dict[str, Any] | None,
) -> None:
    transition_obs = rlt_obs
    if final_obs is not None:
        transition_obs = feature_model.extract_rlt_obs(final_obs)
        if "history" in rlt_obs:
            transition_obs = dict(transition_obs)
            transition_obs["history"] = rlt_obs["history"]
    for key in RLT_LH_OBS_KEYS:
        result["forward_inputs"][f"{RLT_LH_TRANSITION_PREFIX}{key}"] = transition_obs[key]


def predict_rlt_lh_actions(
    *,
    policy_model: Any,
    feature_model: Any,
    env_obs: dict[str, Any],
    final_obs: dict[str, Any] | None,
    mode: Literal["train", "eval"],
    version: int = 0,
    rlt_switch_flags: torch.Tensor | None = None,
    intervene_requested: torch.Tensor | None = None,
    expert_model: Any | None = None,
    cfg: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Predict and route a long-horizon RLT action chunk.

    ``env_obs`` must contain ``history``, either as a raw window
    ``[batch, window, summary_dim]`` or a pre-encoded vector. The route mirrors
    the original RLT simulator route, and transition obs are written with the
    long-horizon key prefix.
    """
    if cfg is None:
        raise ValueError("predict_rlt_lh_actions requires cfg to build the route.")
    rlt_route = build_rlt_lh_route(cfg)

    with torch.no_grad():
        rlt_obs = feature_model.extract_rlt_obs(env_obs)
        history = env_obs.get("history")
        if history is None:
            raise ValueError(
                "Long-horizon RLT rollout requires `history` in env_obs. "
                "Populate it with RLTHistoryBuffer before calling this function."
            )
        if isinstance(history, np.ndarray):
            history = torch.from_numpy(history)
        rlt_obs["history"] = history.to(
            device=next(policy_model.parameters()).device,
            dtype=torch.float32,
        )

        actions, result = policy_model.predict_action_batch(
            env_obs=rlt_obs,
            mode=mode,
            return_obs=True,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        route_output = rlt_route.route(
            RLTRouteContext(
                env_obs=env_obs,
                rlt_obs=rlt_obs,
                student_actions=actions,
                result=result,
                mode=mode,
                rlt_switch_flags=rlt_switch_flags,
                intervene_requested=intervene_requested,
                expert_model=expert_model,
                version=version,
            )
        )
        actions = route_output.actions
        result = route_output.result

        _append_rlt_lh_transition_obs(
            feature_model=feature_model,
            result=result,
            rlt_obs=rlt_obs,
            final_obs=final_obs,
        )

    return actions, result
