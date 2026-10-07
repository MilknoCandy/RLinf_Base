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

from rlinf.algorithms.rlt.route import RLTRoute, RLTRouteContext
from rlinf.algorithms.rlt.transition import RLT_OBS_KEYS, RLT_TRANSITION_PREFIX


def _maybe_score_vlm_phase(
    *,
    feature_model: Any,
    rlt_route: RLTRoute,
    env_obs: dict[str, Any],
    rlt_obs: dict[str, torch.Tensor],
    rlt_switch_flags: torch.Tensor | None,
    stage_id: int,
) -> torch.Tensor | None:
    gate = getattr(rlt_route, "phase_gate", None) or getattr(rlt_route, "z_phase", None)
    if rlt_switch_flags is not None:
        return None
    if gate is None or not getattr(gate, "needs_vlm_score", False):
        return None
    batch_size = int(rlt_obs["z_rl"].shape[0])
    device = rlt_obs["z_rl"].device
    if gate.all_latched(stage_id, batch_size, device):
        return torch.ones(batch_size, device=device, dtype=torch.float32)
    if not hasattr(feature_model, "score_rlt_phase_yes"):
        raise RuntimeError(
            "VLM phase gate requires feature_model.score_rlt_phase_yes."
        )
    return feature_model.score_rlt_phase_yes(env_obs, question=gate.question)


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
    for key in RLT_OBS_KEYS:
        result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[key]


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
    stage_id: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        rlt_obs = feature_model.extract_rlt_obs(env_obs)
        if "ref_chunk" not in rlt_obs or "z_rl" not in rlt_obs:
            raise ValueError(
                "RLT extract must return both ref_chunk (VLA execution) and "
                "z_rl (RL token). The MLP residual-adjusts ref_chunk using z."
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

        vlm_p_yes = _maybe_score_vlm_phase(
            feature_model=feature_model,
            rlt_route=rlt_route,
            env_obs=env_obs,
            rlt_obs=rlt_obs,
            rlt_switch_flags=rlt_switch_flags,
            stage_id=stage_id,
        )

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
                dones=dones,
                stage_id=stage_id,
                vlm_p_yes=vlm_p_yes,
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
