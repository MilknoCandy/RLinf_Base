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

"""Pack multi-chunk prefixes for T-SAC-style critic training on RLT transitions."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.data.schema.embodied_types import Trajectory


def _flatten_obs_tensor(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().float().reshape(-1)
    return flat


def _chunk_actions(transition: Trajectory, chunk_len: int, action_dim: int) -> torch.Tensor:
    if not isinstance(transition.actions, torch.Tensor):
        raise ValueError("TSAC prefix packing requires actions on every transition.")
    actions = transition.actions.detach().float().reshape(-1, action_dim)
    if actions.shape[0] < chunk_len:
        pad = torch.zeros(
            chunk_len - actions.shape[0],
            action_dim,
            dtype=actions.dtype,
        )
        actions = torch.cat([actions, pad], dim=0)
    return actions[:chunk_len].contiguous()


def _chunk_rewards(transition: Trajectory, chunk_len: int) -> torch.Tensor:
    if not isinstance(transition.rewards, torch.Tensor):
        raise ValueError("TSAC prefix packing requires rewards on every transition.")
    rewards = transition.rewards.detach().float().reshape(-1)
    if rewards.numel() < chunk_len:
        pad = torch.zeros(chunk_len - rewards.numel(), dtype=rewards.dtype)
        rewards = torch.cat([rewards, pad], dim=0)
    return rewards[:chunk_len].contiguous()


def _transition_done(transition: Trajectory) -> bool:
    return (
        isinstance(transition.dones, torch.Tensor)
        and transition.dones.reshape(-1).to(torch.bool).any()
    )


def _require_obs_key(obs: dict[str, Any], key: str) -> torch.Tensor:
    if key not in obs or not isinstance(obs[key], torch.Tensor):
        raise ValueError(
            f"TSAC prefix packing requires next_obs['{key}'] on every transition."
        )
    return _flatten_obs_tensor(obs[key])


def pack_tsac_prefix_windows(
    transitions: list[Trajectory],
    *,
    chunk_len: int,
    action_dim: int,
    max_chunks: int,
    z_dim: int,
    proprio_dim: int,
    ref_dim: int,
) -> list[Trajectory]:
    """Attach padded multi-chunk prefixes used by the TSAC Transformer critic.

    For each start transition, look ahead up to ``max_chunks`` realized chunks in
    the same episode and store:

    - ``tsac_prefix_actions``: ``[1, 1, max_steps, action_dim]``
    - ``tsac_prefix_rewards``: ``[1, 1, max_steps]``
    - ``tsac_bootstrap_z / proprio / ref``: boundary states after each chunk
    - ``tsac_chunk_done``: whether that chunk ended the episode
    - ``tsac_valid_chunks``: how many chunks are valid from this start
    """
    if chunk_len < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len}.")
    if max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1, got {max_chunks}.")
    if not transitions:
        return transitions

    max_steps = int(chunk_len * max_chunks)
    for start in range(len(transitions)):
        action_chunks: list[torch.Tensor] = []
        reward_chunks: list[torch.Tensor] = []
        bootstrap_z: list[torch.Tensor] = []
        bootstrap_proprio: list[torch.Tensor] = []
        bootstrap_ref: list[torch.Tensor] = []
        chunk_done: list[bool] = []

        for offset in range(max_chunks):
            idx = start + offset
            if idx >= len(transitions):
                break
            transition = transitions[idx]
            action_chunks.append(
                _chunk_actions(transition, chunk_len=chunk_len, action_dim=action_dim)
            )
            reward_chunks.append(_chunk_rewards(transition, chunk_len=chunk_len))
            next_obs = transition.next_obs
            if not isinstance(next_obs, dict):
                raise ValueError(
                    "TSAC prefix packing requires next_obs on every transition."
                )
            z = _require_obs_key(next_obs, "z_rl")
            proprio = _require_obs_key(next_obs, "proprio")
            ref = _require_obs_key(next_obs, "ref_chunk")
            if z.numel() != z_dim:
                raise ValueError(f"Expected z_dim={z_dim}, got {z.numel()}.")
            if proprio.numel() != proprio_dim:
                raise ValueError(
                    f"Expected proprio_dim={proprio_dim}, got {proprio.numel()}."
                )
            if ref.numel() < ref_dim:
                raise ValueError(f"Expected ref_dim>={ref_dim}, got {ref.numel()}.")
            bootstrap_z.append(z[:z_dim])
            bootstrap_proprio.append(proprio[:proprio_dim])
            bootstrap_ref.append(ref[:ref_dim])
            done = _transition_done(transition)
            chunk_done.append(done)
            if done:
                break

        valid_chunks = len(action_chunks)
        if valid_chunks < 1:
            raise ValueError("TSAC prefix packing produced an empty window.")

        prefix_actions = torch.zeros(max_steps, action_dim, dtype=torch.float32)
        prefix_rewards = torch.zeros(max_steps, dtype=torch.float32)
        for offset, (actions, rewards) in enumerate(zip(action_chunks, reward_chunks)):
            start_step = offset * chunk_len
            prefix_actions[start_step : start_step + chunk_len] = actions
            prefix_rewards[start_step : start_step + chunk_len] = rewards

        boot_z = torch.zeros(max_chunks, z_dim, dtype=torch.float32)
        boot_p = torch.zeros(max_chunks, proprio_dim, dtype=torch.float32)
        boot_r = torch.zeros(max_chunks, ref_dim, dtype=torch.float32)
        done_flags = torch.zeros(max_chunks, dtype=torch.bool)
        for offset in range(valid_chunks):
            boot_z[offset] = bootstrap_z[offset]
            boot_p[offset] = bootstrap_proprio[offset]
            boot_r[offset] = bootstrap_ref[offset]
            done_flags[offset] = bool(chunk_done[offset])

        transitions[start].tsac_prefix_actions = prefix_actions.view(
            1, 1, max_steps, action_dim
        ).contiguous()
        transitions[start].tsac_prefix_rewards = prefix_rewards.view(
            1, 1, max_steps
        ).contiguous()
        transitions[start].tsac_bootstrap_z = boot_z.view(
            1, 1, max_chunks, z_dim
        ).contiguous()
        transitions[start].tsac_bootstrap_proprio = boot_p.view(
            1, 1, max_chunks, proprio_dim
        ).contiguous()
        transitions[start].tsac_bootstrap_ref = boot_r.view(
            1, 1, max_chunks, ref_dim
        ).contiguous()
        transitions[start].tsac_chunk_done = done_flags.view(1, 1, max_chunks).contiguous()
        transitions[start].tsac_valid_chunks = torch.tensor(
            [[[valid_chunks]]], dtype=torch.long
        )

    return transitions
