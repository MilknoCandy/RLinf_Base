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

from rlinf.algorithms.rlt.a1_stm import A1ShortTermMemory
from rlinf.algorithms.rlt.a21_context import A21ContextBuffer, RLT_CONTEXT_KEYS
from rlinf.algorithms.rlt.route import RLTRoute, RLTRouteContext
from rlinf.algorithms.rlt.transition import RLT_OBS_KEYS, RLT_TRANSITION_PREFIX


def _append_rlt_transition_obs(
    *,
    feature_model: Any,
    result: dict[str, Any],
    rlt_obs: dict[str, torch.Tensor],
    final_obs: dict[str, Any] | None,
    next_context: dict[str, torch.Tensor] | None = None,
) -> None:
    transition_obs = rlt_obs
    if final_obs is not None:
        transition_obs = feature_model.extract_rlt_obs(final_obs)
    for key in RLT_OBS_KEYS:
        result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[key]
    # A21: next state's context is the buffer *after* pushing current (z, a).
    if next_context is not None:
        for key in RLT_CONTEXT_KEYS:
            if key in next_context:
                result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = next_context[
                    key
                ]


def _apply_a1_stm(
    *,
    rlt_obs: dict[str, torch.Tensor],
    stm: A1ShortTermMemory | None,
    mode: Literal["train", "eval"],
    rlt_switch_flags: torch.Tensor | None,
    dones: torch.Tensor | None,
    rewards: torch.Tensor | None,
    success: torch.Tensor | None,
) -> dict[str, float]:
    if stm is None or not stm.enabled:
        return {}

    allow_write = mode == "train" or bool(stm.config.write_on_eval)
    stm.finalize_pending(
        dones=dones,
        rewards=rewards,
        success=success,
        allow_write=allow_write,
    )
    raw_z = rlt_obs["z_rl"]
    enhanced_z, metrics = stm.enhance(raw_z)
    rlt_obs["z_rl"] = enhanced_z
    rlt_obs["z_rl_raw"] = raw_z
    if allow_write:
        stm.set_pending(z_rl=raw_z, critical_mask=rlt_switch_flags)
    return metrics


def _flatten_actions(actions: torch.Tensor) -> torch.Tensor:
    flat = actions.detach().float()
    if flat.ndim > 2:
        flat = flat.reshape(flat.shape[0], -1)
    return flat


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
    a1_stm: A1ShortTermMemory | None = None,
    a21_context: A21ContextBuffer | None = None,
    dones: torch.Tensor | None = None,
    rewards: torch.Tensor | None = None,
    success: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        rlt_obs = feature_model.extract_rlt_obs(env_obs)
        _apply_a1_stm(
            rlt_obs=rlt_obs,
            stm=a1_stm,
            mode=mode,
            rlt_switch_flags=rlt_switch_flags,
            dones=dones,
            rewards=rewards,
            success=success,
        )

        if a21_context is not None and a21_context.enabled:
            # Reset completed envs before reading context for the new step.
            a21_context.reset(dones)
            batch_size = int(rlt_obs["z_rl"].shape[0])
            device = rlt_obs["z_rl"].device
            ctx = a21_context.get(batch_size, device)
            rlt_obs.update(ctx)

        actions, result = policy_model.predict_action_batch(
            env_obs=rlt_obs,
            mode=mode,
            return_obs=True,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        if rlt_switch_flags is not None and isinstance(result.get("forward_inputs"), dict):
            result["forward_inputs"]["rlt_switch_flags"] = rlt_switch_flags

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

        next_context = None
        if a21_context is not None and a21_context.enabled:
            # Push executed (routed) actions so context matches what the env saw.
            raw_z = rlt_obs.get("z_rl_raw", rlt_obs["z_rl"])
            next_context = a21_context.push(
                z_rl=raw_z, actions=_flatten_actions(actions)
            )

        _append_rlt_transition_obs(
            feature_model=feature_model,
            result=result,
            rlt_obs=rlt_obs,
            final_obs=final_obs,
            next_context=next_context,
        )

    return actions, result
