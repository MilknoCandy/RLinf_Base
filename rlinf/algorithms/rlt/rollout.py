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

from typing import Any, Literal

import numpy as np
import torch

from rlinf.algorithms.rlt.b1_dynamic import B1DynamicModule, B1DynamicRuntime
from rlinf.algorithms.rlt.route import RLTRoute, RLTRouteContext
from rlinf.algorithms.rlt.transition import (
    RLT_B1_OPTIONAL_KEYS,
    RLT_OBS_KEYS,
    RLT_TRANSITION_PREFIX,
)


def _append_rlt_transition_obs(
    *,
    feature_model: Any,
    result: dict[str, Any],
    rlt_obs: dict[str, torch.Tensor],
    final_obs: dict[str, Any] | None,
) -> None:
    transition_obs = rlt_obs
    if final_obs is not None:
        transition_obs = feature_model.extract_rlt_obs(final_obs)
        # Keep B1 fields from the current-step fused obs when final_obs re-extracts
        # appearance features without the recurrent state.
        for key in RLT_B1_OPTIONAL_KEYS:
            if key in rlt_obs and key not in transition_obs:
                transition_obs[key] = rlt_obs[key]
    for key in RLT_OBS_KEYS:
        result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[key]
    for key in RLT_B1_OPTIONAL_KEYS:
        if key in transition_obs:
            result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[
                key
            ]


def _apply_b1_dynamic(
    *,
    policy_model: Any,
    rlt_obs: dict[str, torch.Tensor],
    b1_runtime: B1DynamicRuntime | None,
    rlt_switch_flags: torch.Tensor | None,
    dones: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    module = getattr(policy_model, "b1_dynamic", None)
    if module is None or b1_runtime is None or not b1_runtime.config.enable:
        return rlt_obs

    z_app = rlt_obs["z_rl"]
    extras = b1_runtime.step(
        module=module,
        z_app=z_app,
        critical_mask=rlt_switch_flags,
        dones=dones,
    )
    out = dict(rlt_obs)
    out["z_app"] = extras["z_app"]
    out["z_rl"] = extras["z_rl"]
    out["b1_h"] = extras["b1_h"]
    out["b1_h_prev"] = extras["b1_h_prev"]
    out["b1_gate"] = extras["b1_gate"]
    out["b1_updated"] = extras["b1_updated"]
    return out


def predict_rlt_actions(
    *,
    policy_model: Any,
    feature_model: Any,
    rlt_route: RLTRoute,
    env_obs: dict[str, Any],
    final_obs: dict[str, Any] | None,
    mode: Literal["train", "eval"],
    version: int = 0,
    rlt_switch_flags: torch.Tensor | None = None,
    intervene_requested: torch.Tensor | None = None,
    expert_model: Any | None = None,
    dones: torch.Tensor | None = None,
    rewards: torch.Tensor | None = None,
    success: torch.Tensor | None = None,
    b1_runtime: B1DynamicRuntime | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    del rewards, success  # Reserved for future memory write hooks.
    with torch.no_grad():
        rlt_obs = feature_model.extract_rlt_obs(env_obs)
        rlt_obs = _apply_b1_dynamic(
            policy_model=policy_model,
            rlt_obs=rlt_obs,
            b1_runtime=b1_runtime,
            rlt_switch_flags=rlt_switch_flags,
            dones=dones,
        )

        actions, result = policy_model.predict_action_batch(
            env_obs=rlt_obs,
            mode=mode,
            return_obs=True,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        if rlt_switch_flags is not None and isinstance(
            result.get("forward_inputs"), dict
        ):
            result["forward_inputs"]["rlt_switch_flags"] = rlt_switch_flags

        # Ensure B1 tensors are present in forward_inputs for transition replay.
        if isinstance(result.get("forward_inputs"), dict):
            for key in ("z_rl", "proprio", "ref_chunk", *RLT_B1_OPTIONAL_KEYS):
                if key in rlt_obs:
                    result["forward_inputs"][key] = rlt_obs[key]

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

        _append_rlt_transition_obs(
            feature_model=feature_model,
            result=result,
            rlt_obs=rlt_obs,
            final_obs=final_obs,
        )

    return actions, result
