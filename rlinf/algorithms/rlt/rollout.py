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

from rlinf.algorithms.rlt.b2_feedback import (
    distance_from_env_infos,
    success_from_env_infos,
)
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
    env_infos: dict[str, Any] | None = None,
    b2_loop: Any | None = None,
    dump_writer: Any | None = None,
    dump_use_ref_chunk: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    del rewards, success
    with torch.no_grad():
        extract_kwargs: dict[str, Any] = {}
        rlt_cfg = getattr(feature_model, "rlt_cfg", None)
        use_b2 = bool(b2_loop is not None and getattr(rlt_cfg, "rlt_b2", False))
        batch_size = None
        device = None
        if use_b2:
            states = env_obs.get("states")
            if not torch.is_tensor(states):
                raise ValueError("B2 RLT extract requires batched env_obs['states'].")
            batch_size = int(states.shape[0])
            encoder = feature_model.rlt_module.encoder
            device = next(encoder.parameters()).device
            dtype = next(encoder.parameters()).dtype
            rl_token = b2_loop.build_rl_token(
                init_token=encoder.rl_token_embed,
                batch_size=batch_size,
                dones=dones,
                device=device,
                dtype=dtype,
            )
            feedback = b2_loop.feedback_sentences(
                batch_size=batch_size,
                dones=dones,
                env_infos=env_infos,
                device=device,
            )
            extract_kwargs = {
                "rl_token": rl_token.detach(),
                "feedback_sentences": feedback,
            }
        if dump_writer is not None:
            extract_kwargs["return_image_tokens"] = True

        rlt_obs = feature_model.extract_rlt_obs(env_obs, **extract_kwargs)
        if use_b2:
            b2_loop.commit_z(rlt_obs["z_rl"])
        if dump_writer is not None:
            if batch_size is None:
                tokens = rlt_obs["rlt_image_tokens"]
                batch_size = int(tokens.shape[0])
                device = tokens.device
            dump_writer.append(
                image_tokens=rlt_obs["rlt_image_tokens"],
                image_mask=rlt_obs.get("rlt_image_mask"),
                distance=distance_from_env_infos(env_infos, batch_size, device),
                success=success_from_env_infos(env_infos, batch_size, device),
                dones=dones,
            )

        if dump_use_ref_chunk:
            actions = _actions_from_ref_chunk(rlt_obs, policy_model)
            result = _result_from_ref_actions(rlt_obs, actions)
        else:
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

        if not dump_use_ref_chunk:
            _append_rlt_transition_obs(
                feature_model=feature_model,
                result=result,
                rlt_obs=rlt_obs,
                final_obs=final_obs,
            )

    return actions, result


def _actions_from_ref_chunk(
    rlt_obs: dict[str, torch.Tensor], policy_model: Any
) -> torch.Tensor:
    ref_chunk = rlt_obs["ref_chunk"]
    action_dim = int(getattr(policy_model, "action_dim", 0) or 0)
    num_chunks = int(getattr(policy_model, "num_action_chunks", 0) or 0)
    if ref_chunk.dim() == 3:
        chunk_len = num_chunks or ref_chunk.shape[1]
        dim = action_dim or ref_chunk.shape[2]
        return ref_chunk[:, :chunk_len, :dim].contiguous()
    if action_dim <= 0:
        raise ValueError(
            "B2 dump with a rank-2 ref_chunk requires policy_model.action_dim."
        )
    actions = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)
    if num_chunks:
        actions = actions[:, :num_chunks]
    return actions.contiguous()


def _result_from_ref_actions(
    rlt_obs: dict[str, torch.Tensor],
    actions: torch.Tensor,
) -> dict[str, Any]:
    batch_size = actions.shape[0]
    flat = actions.reshape(batch_size, -1).contiguous()
    logprobs = torch.zeros(
        batch_size, actions.shape[1], device=actions.device, dtype=actions.dtype
    )
    forward_inputs = {
        "action": flat,
        "model_action": flat,
        "z_rl": rlt_obs["z_rl"],
        "proprio": rlt_obs["proprio"],
        "ref_chunk": rlt_obs["ref_chunk"],
    }
    return {
        "prev_logprobs": logprobs,
        "prev_values": torch.zeros(
            batch_size, actions.shape[1], 1, device=actions.device, dtype=actions.dtype
        ),
        "forward_inputs": forward_inputs,
    }
