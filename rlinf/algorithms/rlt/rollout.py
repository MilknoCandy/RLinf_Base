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

from rlinf.algorithms.rlt.a2_stm import A2MemoryBank, A2STMEncoder
from rlinf.algorithms.rlt.route import RLTRoute, RLTRouteContext
from rlinf.algorithms.rlt.transition import RLT_OBS_KEYS, RLT_TRANSITION_PREFIX


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


def _flatten_actions_for_stm(
    actions: torch.Tensor, action_dim: int
) -> torch.Tensor:
    flat = actions.detach().float()
    if flat.ndim > 2:
        flat = flat.reshape(flat.shape[0], -1)
    if flat.shape[-1] == action_dim:
        return flat
    out = flat.new_zeros(flat.shape[0], action_dim)
    n = min(action_dim, flat.shape[-1])
    out[:, :n] = flat[:, :n]
    return out


def _apply_a2_stm_before_policy(
    *,
    rlt_obs: dict[str, torch.Tensor],
    write_bank: A2MemoryBank | None,
    encoder: A2STMEncoder | None,
    memory_z: torch.Tensor | None,
    memory_a: torch.Tensor | None,
    mode: Literal["train", "eval"],
    dones: torch.Tensor | None,
    rewards: torch.Tensor | None,
    success: torch.Tensor | None,
) -> dict[str, float]:
    if write_bank is None or encoder is None:
        return {}

    allow_write = mode == "train" or bool(write_bank.config.write_on_eval)
    write_bank.finalize_pending(
        dones=dones,
        rewards=rewards,
        success=success,
        allow_write=allow_write,
    )

    raw_z = rlt_obs["z_rl"]
    if memory_z is None or memory_a is None:
        memory_z, memory_a = write_bank.gather_memory(raw_z.device)
    enhanced_z, metrics = encoder.enhance(raw_z, memory_z, memory_a)
    rlt_obs["z_rl"] = enhanced_z
    rlt_obs["z_rl_raw"] = raw_z
    write_bank.record_step_metrics(metrics)
    return metrics


def _finalize_a2_stm_after_policy(
    *,
    rlt_obs: dict[str, torch.Tensor],
    result: dict[str, Any],
    actions: torch.Tensor,
    write_bank: A2MemoryBank | None,
    mode: Literal["train", "eval"],
    rlt_switch_flags: torch.Tensor | None,
) -> None:
    """Store raw ``z`` in replay tensors and arm pending ``(z, a)`` writes."""
    raw_z = rlt_obs.pop("z_rl_raw", None)
    if raw_z is None:
        return

    # Replay / transition must keep raw z so the actor can recompute z' w/ grad.
    rlt_obs["z_rl"] = raw_z
    if isinstance(result.get("forward_inputs"), dict):
        result["forward_inputs"]["z_rl"] = raw_z

    if write_bank is None:
        return
    allow_write = mode == "train" or bool(write_bank.config.write_on_eval)
    if not allow_write:
        return
    flat_a = _flatten_actions_for_stm(actions, write_bank.config.action_dim)
    write_bank.set_pending(
        z_rl=raw_z,
        actions=flat_a,
        critical_mask=rlt_switch_flags,
    )


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
    a2_stm_bank: A2MemoryBank | None = None,
    a2_stm_memory: tuple[torch.Tensor, torch.Tensor] | None = None,
    dones: torch.Tensor | None = None,
    rewards: torch.Tensor | None = None,
    success: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        rlt_obs = feature_model.extract_rlt_obs(env_obs)
        encoder = getattr(policy_model, "a2_stm_encoder", None)
        memory_z = None
        memory_a = None
        if a2_stm_memory is not None:
            memory_z, memory_a = a2_stm_memory
        _apply_a2_stm_before_policy(
            rlt_obs=rlt_obs,
            write_bank=a2_stm_bank,
            encoder=encoder,
            memory_z=memory_z,
            memory_a=memory_a,
            mode=mode,
            dones=dones,
            rewards=rewards,
            success=success,
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

        # After routing: store raw z for replay, and remember executed (z, a).
        _finalize_a2_stm_after_policy(
            rlt_obs=rlt_obs,
            result=result,
            actions=actions,
            write_bank=a2_stm_bank,
            mode=mode,
            rlt_switch_flags=rlt_switch_flags,
        )

        _append_rlt_transition_obs(
            feature_model=feature_model,
            result=result,
            rlt_obs=rlt_obs,
            final_obs=final_obs,
        )
        # Do not put STM metrics into forward_inputs: _split_policy_output
        # assumes every value is a batch tensor.

    return actions, result
