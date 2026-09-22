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
from rlinf.algorithms.rlt.transition import (
    RLT_B2_OBS_KEYS,
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
    for key in (*RLT_OBS_KEYS, *RLT_B2_OBS_KEYS):
        if key in transition_obs:
            result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[
                key
            ]


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
) -> tuple[torch.Tensor, dict[str, Any]]:
    del rewards, success
    with torch.no_grad():
        extract_kwargs: dict[str, Any] = {}
        rlt_cfg = getattr(feature_model, "rlt_cfg", None)
        dumping = dump_writer is not None
        # Dump keeps the trained Stage 1/2 distribution (no feedback, no Loop).
        # Online B2 RL opens Loop + templated feedback in the frozen VLM prompt.
        use_b2 = bool(
            b2_loop is not None
            and getattr(rlt_cfg, "rlt_b2", False)
            and not dumping
        )
        policy_encoder = getattr(policy_model, "rlt_loop", None)
        if use_b2:
            states = env_obs.get("states")
            if not torch.is_tensor(states):
                raise ValueError("B2 RLT extract requires batched env_obs['states'].")
            batch_size = int(states.shape[0])
            loop_encoder = policy_encoder or feature_model.rlt_module.encoder
            device = next(loop_encoder.parameters()).device
            dtype = next(loop_encoder.parameters()).dtype
            z_prev = b2_loop.build_rl_token(
                init_token=loop_encoder.rl_token_embed,
                batch_size=batch_size,
                dones=dones,
                device=device,
                dtype=dtype,
            ).detach()
            extract_kwargs = {
                "feedback_sentences": b2_loop.feedback_sentences(
                    batch_size=batch_size,
                    dones=dones,
                    env_infos=env_infos,
                    device=device,
                ),
                "return_image_tokens": True,
                "encode_z": policy_encoder is None,
            }
            if policy_encoder is None:
                extract_kwargs["rl_token"] = z_prev
        if dumping:
            extract_kwargs["return_image_tokens"] = True

        rlt_obs = feature_model.extract_rlt_obs(env_obs, **extract_kwargs)
        if use_b2:
            rlt_obs["z_prev"] = z_prev
            if policy_encoder is not None:
                rlt_obs["z_rl"] = policy_model.encode_rlt(
                    rlt_obs["rlt_image_tokens"],
                    rlt_obs.get("rlt_image_mask"),
                    z_prev,
                ).detach()
            tokens = rlt_obs["rlt_image_tokens"]
            rlt_obs["rlt_image_tokens"] = tokens.detach().to(dtype=torch.float16)
            if rlt_obs.get("rlt_image_mask") is not None:
                rlt_obs["rlt_image_mask"] = rlt_obs["rlt_image_mask"].detach()
        if "ref_chunk" not in rlt_obs or "z_rl" not in rlt_obs:
            raise ValueError(
                "RLT extract must return both ref_chunk (VLA execution) and "
                "z_rl (RL token). The MLP residual-adjusts ref_chunk using z."
            )
        if use_b2:
            b2_loop.commit_z(rlt_obs["z_rl"])

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

        if dump_writer is not None:
            tokens = rlt_obs["rlt_image_tokens"]
            batch_size = int(tokens.shape[0])
            dump_writer.append(
                image_tokens=tokens,
                image_mask=rlt_obs.get("rlt_image_mask"),
                distance=distance_from_env_infos(
                    env_infos, batch_size, tokens.device
                ),
                success=success_from_env_infos(
                    env_infos, batch_size, tokens.device
                ),
                dones=dones,
            )

        if dump_writer is None:
            _append_rlt_transition_obs(
                feature_model=feature_model,
                result=result,
                rlt_obs=rlt_obs,
                final_obs=None if use_b2 else final_obs,
            )

    return actions, result
