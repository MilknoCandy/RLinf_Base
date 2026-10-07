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

"""Pack consecutive RLT chunks for a DQC long-horizon critic."""

from __future__ import annotations

import torch

from rlinf.algorithms.rlt.n_step import discounted_chunk_reward
from rlinf.data.schema.embodied_types import Trajectory


def _flatten_obs(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(-1)


def _chunk_actions(
    transition: Trajectory, chunk_len: int, action_dim: int
) -> torch.Tensor:
    if not isinstance(transition.actions, torch.Tensor):
        raise ValueError("DQC packing requires actions on every transition.")
    actions = transition.actions.detach().float().reshape(-1, action_dim)
    if actions.shape[0] < chunk_len:
        pad = torch.zeros(
            chunk_len - actions.shape[0], action_dim, dtype=actions.dtype
        )
        actions = torch.cat([actions, pad], dim=0)
    return actions[:chunk_len].contiguous()


def _chunk_rewards(transition: Trajectory, chunk_len: int) -> torch.Tensor:
    if not isinstance(transition.rewards, torch.Tensor):
        raise ValueError("DQC packing requires rewards on every transition.")
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


def pack_dqc_windows(
    transitions: list[Trajectory],
    *,
    chunk_len: int,
    action_dim: int,
    backup_chunks: int,
    gamma: float,
    z_dim: int,
    proprio_dim: int,
) -> list[Trajectory]:
    """Attach a concatenated action-chunk window for the DQC chunk critic.

    ``backup_chunks`` is the critic horizon in policy chunks (each of length
    ``chunk_len`` env steps). The distilled policy critic still consumes one
    policy chunk; this window is only for ``Q^h``.
    """
    if backup_chunks < 1:
        raise ValueError(f"backup_chunks must be >= 1, got {backup_chunks}.")
    if not transitions:
        return transitions

    backup_steps = int(chunk_len * backup_chunks)
    for start, start_transition in enumerate(transitions):
        action_parts: list[torch.Tensor] = []
        reward_parts: list[torch.Tensor] = []
        valid_chunks = 0
        done = False
        bootstrap_obs: dict[str, torch.Tensor] | None = None
        for offset in range(backup_chunks):
            idx = start + offset
            if idx >= len(transitions):
                break
            transition = transitions[idx]
            action_parts.append(
                _chunk_actions(transition, chunk_len=chunk_len, action_dim=action_dim)
            )
            reward_parts.append(_chunk_rewards(transition, chunk_len=chunk_len))
            valid_chunks += 1
            next_obs = transition.next_obs
            if not isinstance(next_obs, dict):
                raise ValueError("DQC packing requires next_obs on every transition.")
            bootstrap_obs = {
                "z_rl": _flatten_obs(next_obs["z_rl"])[:z_dim],
                "proprio": _flatten_obs(next_obs["proprio"])[:proprio_dim],
                "ref_chunk": _flatten_obs(next_obs["ref_chunk"]),
            }
            done = _transition_done(transition)
            if done:
                break

        if valid_chunks < 1 or bootstrap_obs is None:
            raise ValueError("DQC packing produced an empty window.")

        flat_actions = torch.zeros(backup_steps * action_dim, dtype=torch.float32)
        cursor = 0
        n_step_return = torch.zeros((), dtype=torch.float32)
        discount = 1.0
        for rewards, actions in zip(reward_parts, action_parts):
            n_step_return = n_step_return + discount * discounted_chunk_reward(
                rewards, gamma
            )
            horizon = max(int(rewards.numel()), 1)
            discount *= float(gamma) ** horizon
            numel = actions.numel()
            flat_actions[cursor : cursor + numel] = actions.reshape(-1)
            cursor += numel

        bootstrap_discount = 0.0 if done else float(discount)
        start_transition.dqc_chunk_actions = flat_actions.view(
            1, 1, backup_steps * action_dim
        ).contiguous()
        start_transition.dqc_n_step_returns = torch.tensor(
            [[[float(n_step_return)]]], dtype=torch.float32
        )
        start_transition.dqc_bootstrap_discount = torch.tensor(
            [[[bootstrap_discount]]], dtype=torch.float32
        )
        start_transition.dqc_valid_chunks = torch.tensor(
            [[[valid_chunks]]], dtype=torch.long
        )
        start_transition.dqc_bootstrap_z = bootstrap_obs["z_rl"].view(1, 1, z_dim)
        start_transition.dqc_bootstrap_proprio = bootstrap_obs["proprio"].view(
            1, 1, proprio_dim
        )
        ref = bootstrap_obs["ref_chunk"]
        start_transition.dqc_bootstrap_ref = ref.view(1, 1, int(ref.numel()))
        start_transition.dqc_done = torch.tensor([[[done]]], dtype=torch.bool)

    return transitions
