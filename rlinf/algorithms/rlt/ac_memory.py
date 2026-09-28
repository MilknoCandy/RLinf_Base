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

"""Action-space actor memory and return-space critic memory.

This is separate from ``use_mem``. That path loops history into one ``z``.
Here the actor bank adds a retrieved correction to the reference action, and
the critic bank mixes a retrieved return into the Bellman target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def discounted_chunk_reward(reward: torch.Tensor, gamma: float) -> float:
    """Discount rewards inside one chunk and sum them to a scalar."""
    flat = reward.detach().to(dtype=torch.float32).reshape(-1)
    if flat.numel() == 0:
        return 0.0
    powers = torch.arange(flat.numel(), dtype=torch.float32)
    discounts = torch.pow(torch.tensor(float(gamma), dtype=torch.float32), powers)
    return float((flat * discounts).sum())


def memory_confidence(
    std: torch.Tensor,
    count: torch.Tensor,
    lambda_max: float,
    sigma0: float,
) -> torch.Tensor:
    """Trust retrieved returns when neighbors agree. Empty reads stay at 0."""
    if float(sigma0) <= 0:
        raise ValueError(f"ac memory sigma0 must be positive, got {sigma0}.")
    confidence = float(lambda_max) * torch.exp(-std / float(sigma0))
    return torch.where(count > 0, confidence, torch.zeros_like(confidence))


def mix_q_values(
    q_values: torch.Tensor,
    retrieved_return: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """Mix parametric Q with a detached episodic return."""
    retrieved = retrieved_return.to(device=q_values.device, dtype=q_values.dtype)
    weight = confidence.to(device=q_values.device, dtype=q_values.dtype)
    while retrieved.dim() < q_values.dim():
        retrieved = retrieved.unsqueeze(-1)
        weight = weight.unsqueeze(-1)
    return (1.0 - weight) * q_values + weight * retrieved.detach()


@dataclass
class MemoryStep:
    """One recorded chunk before it is written into the banks."""

    traj_t: int
    z: torch.Tensor
    delta: torch.Tensor
    reward: float
    done: bool


@dataclass
class MemoryStamp:
    """Episode clock stored on a replay transition for leave-one-out reads."""

    episode_id: int
    time: int


class CriticReturnBank:
    """FIFO bank of completed chunks and their realized returns."""

    def __init__(self, capacity: int, z_dim: int, action_dim: int):
        if int(capacity) <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self.capacity = int(capacity)
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        self.z = torch.zeros(self.capacity, self.z_dim)
        self.delta = torch.zeros(self.capacity, self.action_dim)
        self.returns = torch.zeros(self.capacity)
        self.episode = torch.full((self.capacity,), -1, dtype=torch.long)
        self.time = torch.zeros(self.capacity, dtype=torch.long)
        self.valid = torch.zeros(self.capacity)
        self.cursor = 0
        self.size = 0

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        return {
            "z": self.z.clone(),
            "delta": self.delta.clone(),
            "returns": self.returns.clone(),
            "episode": self.episode.clone(),
            "time": self.time.clone(),
            "valid": self.valid.clone(),
            "cursor": int(self.cursor),
            "size": int(self.size),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | int]) -> None:
        self.z.copy_(state["z"])
        self.delta.copy_(state["delta"])
        self.returns.copy_(state["returns"])
        self.episode.copy_(state["episode"])
        self.time.copy_(state["time"])
        self.valid.copy_(state["valid"])
        self.cursor = int(state["cursor"])
        self.size = int(state["size"])

    def mean_return(self) -> float:
        mask = self.valid > 0.5
        if not bool(mask.any()):
            return 0.0
        return float(self.returns[mask].mean())

    def add_tail(
        self, episode_id: int, next_time: int, tail: float, discount: float
    ) -> None:
        """Add a later segment's return onto earlier slots of the same episode."""
        mask = (
            (self.valid > 0.5)
            & (self.episode == int(episode_id))
            & (self.time < int(next_time))
        )
        if not bool(mask.any()):
            return
        steps = (int(next_time) - self.time[mask]).to(dtype=self.returns.dtype)
        factor = torch.pow(
            torch.tensor(float(discount), dtype=self.returns.dtype),
            steps,
        )
        self.returns[mask] = self.returns[mask] + factor * float(tail)

    def write(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        realized_return: float,
        episode_id: int,
        time: int,
    ) -> None:
        slot = int(self.cursor % self.capacity)
        self.cursor += 1
        self.size = min(self.size + 1, self.capacity)
        self.z[slot].copy_(z.detach().to(dtype=self.z.dtype).reshape(-1))
        self.delta[slot].copy_(delta.detach().to(dtype=self.delta.dtype).reshape(-1))
        self.returns[slot] = float(realized_return)
        self.episode[slot] = int(episode_id)
        self.time[slot] = int(time)
        self.valid[slot] = 1.0


class ActorCorrectionBank:
    """Positive-advantage corrections. Full banks replace the smallest advantage."""

    def __init__(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        returns: torch.Tensor,
        advantage: torch.Tensor,
        episode: torch.Tensor,
        time: torch.Tensor,
        valid: torch.Tensor,
        topk: int,
        tau: float,
    ):
        if int(topk) <= 0:
            raise ValueError(f"topk must be positive, got {topk}.")
        if float(tau) <= 0:
            raise ValueError(f"tau must be positive, got {tau}.")
        self.z = z
        self.delta = delta
        self.returns = returns
        self.advantage = advantage
        self.episode = episode
        self.time = time
        self.valid = valid
        self.topk = int(topk)
        self.tau = float(tau)
        self.capacity = int(z.shape[0])
        self.z_dim = int(z.shape[-1])
        self.action_dim = int(delta.shape[-1])

    def apply_baseline(self, baseline: float) -> None:
        """Drop corrections that are no longer above the return baseline."""
        mask = self.valid > 0.5
        if not bool(mask.any()):
            return
        self.advantage[mask] = self.returns[mask] - float(baseline)
        keep = mask & (self.advantage > 0)
        self.valid.zero_()
        self.valid[keep] = 1.0

    def invalidate_key(self, episode_id: int, time: int) -> None:
        match = (
            (self.episode == int(episode_id))
            & (self.time == int(time))
            & (self.valid > 0.5)
        )
        if bool(match.any()):
            self.valid[match] = 0.0

    def upsert(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        realized_return: float,
        advantage: float,
        episode_id: int,
        time: int,
    ) -> None:
        if float(advantage) <= 0:
            self.invalidate_key(episode_id, time)
            return
        existing = (
            (self.episode == int(episode_id)) & (self.time == int(time))
        ).nonzero(as_tuple=False)
        if existing.numel() > 0:
            slot = int(existing[0])
        else:
            slot = self._free_slot(float(advantage))
            if slot is None:
                return
        self._fill(slot, z, delta, realized_return, advantage, episode_id, time)

    def retrieve_action(
        self,
        query_z: torch.Tensor,
        query_episode: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        """Similarity-weighted :math:`\\Delta a`. Future slots of this episode are hidden."""
        weighted, _, _ = _retrieve(
            query_z,
            query_episode,
            query_time,
            self.z,
            self.episode,
            self.time,
            self.valid,
            self.delta,
            self.topk,
            self.tau,
        )
        return weighted

    def _free_slot(self, advantage: float) -> int | None:
        empty = (self.valid <= 0.5).nonzero(as_tuple=False)
        if empty.numel() > 0:
            return int(empty[0])
        valid_idx = (self.valid > 0.5).nonzero(as_tuple=False).flatten()
        min_pos = int(torch.argmin(self.advantage[valid_idx]))
        slot = int(valid_idx[min_pos])
        if advantage <= float(self.advantage[slot]):
            return None
        return slot

    def _fill(
        self,
        slot: int,
        z: torch.Tensor,
        delta: torch.Tensor,
        realized_return: float,
        advantage: float,
        episode_id: int,
        time: int,
    ) -> None:
        z_row = z.detach().to(dtype=self.z.dtype, device=self.z.device).reshape(-1)
        delta_row = delta.detach().to(
            dtype=self.delta.dtype, device=self.delta.device
        ).reshape(-1)
        self.z[slot].copy_(z_row)
        self.delta[slot].copy_(delta_row)
        self.returns[slot] = float(realized_return)
        self.advantage[slot] = float(advantage)
        self.episode[slot] = int(episode_id)
        self.time[slot] = int(time)
        self.valid[slot] = 1.0


def _retrieve(
    query_z: torch.Tensor,
    query_episode: torch.Tensor,
    query_time: torch.Tensor,
    bank_z: torch.Tensor,
    bank_episode: torch.Tensor,
    bank_time: torch.Tensor,
    bank_valid: torch.Tensor,
    bank_values: torch.Tensor,
    topk: int,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted read with same-episode times ``>= query_time`` removed."""
    device = query_z.device
    dtype = query_z.dtype
    batch = int(query_z.shape[0])
    value_shape = bank_values.shape[1:]
    zeros = torch.zeros(batch, *value_shape, device=device, dtype=dtype)
    zero_std = torch.zeros(batch, device=device, dtype=dtype)
    zero_count = torch.zeros(batch, device=device, dtype=dtype)
    valid = bank_valid.detach().to(device=device) > 0.5
    if not bool(valid.any()):
        return zeros, zero_std, zero_count

    query = query_z.detach().to(device=device, dtype=torch.float32).reshape(batch, -1)
    keys = bank_z.detach().to(device=device, dtype=torch.float32)
    episode = query_episode.detach().to(device=device, dtype=torch.long).reshape(batch)
    time = query_time.detach().to(device=device, dtype=torch.long).reshape(batch)
    bank_episode = bank_episode.detach().to(device=device)
    bank_time = bank_time.detach().to(device=device)
    same_episode = bank_episode.unsqueeze(0) == episode.unsqueeze(1)
    too_new = bank_time.unsqueeze(0) >= time.unsqueeze(1)
    exclude = same_episode & too_new & (episode.unsqueeze(1) >= 0)
    row_valid = valid.unsqueeze(0) & ~exclude
    if not bool(row_valid.any()):
        return zeros, zero_std, zero_count

    query_n = torch.nan_to_num(F.normalize(query, dim=-1))
    key_n = torch.nan_to_num(F.normalize(keys, dim=-1))
    similarity = query_n @ key_n.T
    similarity = similarity.masked_fill(~row_valid, -1.0e4)
    k = min(int(topk), int(similarity.shape[-1]))
    chosen_sim, chosen_idx = torch.topk(similarity, k=k, dim=-1)
    chosen_valid = row_valid.gather(1, chosen_idx)
    scores = chosen_sim.masked_fill(~chosen_valid, -1.0e4)
    weights = torch.softmax(scores / float(tau), dim=-1)
    weights = weights * chosen_valid.to(dtype=weights.dtype)
    denom = weights.sum(dim=-1, keepdim=True)
    has_neighbor = denom.squeeze(-1) > 0
    weights = torch.where(denom > 0, weights / denom.clamp(min=1.0e-6), weights)

    flat_values = bank_values.detach().to(device=device, dtype=torch.float32).reshape(
        bank_values.shape[0], -1
    )
    gathered = flat_values[chosen_idx]
    weighted = (weights.unsqueeze(-1) * gathered).sum(dim=1)
    weighted = weighted.reshape(batch, *value_shape).to(device=device, dtype=dtype)
    weighted = torch.where(has_neighbor.reshape(batch, *([1] * len(value_shape))), weighted, zeros)

    centered = gathered - weighted.detach().to(dtype=torch.float32).reshape(batch, -1).unsqueeze(1)
    # Std is over the value's mean when values are vectors; callers that need a
    # scalar std pass a ``[N]`` or ``[N, 1]`` return vector.
    per_item = centered.square().mean(dim=-1)
    variance = (per_item * chosen_valid.to(dtype=per_item.dtype)).sum(dim=-1)
    variance = variance / chosen_valid.to(dtype=variance.dtype).sum(dim=-1).clamp(min=1.0)
    std = variance.sqrt().to(device=device, dtype=dtype)
    std = torch.where(has_neighbor, std, zero_std)
    count = chosen_valid.to(dtype=dtype).sum(dim=-1).to(device=device)
    return weighted, std, count


class ACMemoryWriter:
    """Write both banks from recorded chunks and keep episode clocks."""

    def __init__(
        self,
        critic_bank: CriticReturnBank,
        actor_bank: ActorCorrectionBank | None,
        discount: float,
    ):
        self.critic_bank = critic_bank
        self.actor_bank = actor_bank
        self.discount = float(discount)
        self.next_episode_id = 0
        self.open_episodes: dict[int, tuple[int, int]] = {}
        # Lagged mean. The chunk being written is compared with this value, so
        # the first positive return is not cancelled by its own average.
        self.baseline = 0.0

    def state_dict(self) -> dict:
        return {
            "bank": self.critic_bank.state_dict(),
            "next_episode_id": int(self.next_episode_id),
            "open_episodes": dict(self.open_episodes),
            "baseline": float(self.baseline),
        }

    def load_state_dict(self, state: dict) -> None:
        self.critic_bank.load_state_dict(state["bank"])
        self.next_episode_id = int(state["next_episode_id"])
        self.open_episodes = {
            int(env_idx): (int(episode_id), int(next_time))
            for env_idx, (episode_id, next_time) in state["open_episodes"].items()
        }
        self.baseline = float(state.get("baseline", self.critic_bank.mean_return()))

    def ingest_env(self, env_idx: int, steps: list[MemoryStep]) -> list[MemoryStamp]:
        """Write one env's recorded chunks. Stamps align with ``steps``."""
        stamps: list[MemoryStamp | None] = [None] * len(steps)
        if not steps:
            return []
        segments: list[tuple[list[int], bool]] = []
        buffer: list[int] = []
        for index, step in enumerate(steps):
            buffer.append(index)
            if step.done:
                segments.append((buffer, True))
                buffer = []
        if buffer:
            segments.append((buffer, False))

        first_segment = True
        for indices, ended in segments:
            continuing = first_segment and env_idx in self.open_episodes
            first_segment = False
            if continuing:
                episode_id, start_time = self.open_episodes[env_idx]
            else:
                episode_id = self.next_episode_id
                self.next_episode_id += 1
                start_time = 0
            times = _episode_times(steps, indices, start_time)
            returns = _segment_returns(steps, indices, times, self.discount)
            if continuing and start_time > 0:
                self.critic_bank.add_tail(
                    episode_id, start_time, returns[0], self.discount
                )
            for local, index in enumerate(indices):
                step = steps[index]
                self.critic_bank.write(
                    step.z,
                    step.delta,
                    returns[local],
                    episode_id,
                    times[local],
                )
                stamps[index] = MemoryStamp(episode_id, times[local])
            if ended:
                self.open_episodes.pop(env_idx, None)
            else:
                self.open_episodes[env_idx] = (episode_id, times[-1] + 1)
        self._sync_actor(self.baseline)
        self.baseline = self.critic_bank.mean_return()
        if any(stamp is None for stamp in stamps):
            raise RuntimeError("Every recorded chunk must receive a memory stamp.")
        return [stamp for stamp in stamps if stamp is not None]

    def _sync_actor(self, baseline: float) -> None:
        actor = self.actor_bank
        if actor is None:
            return
        actor.apply_baseline(baseline)
        critic = self.critic_bank
        mask = critic.valid > 0.5
        if not bool(mask.any()):
            return
        indices = mask.nonzero(as_tuple=False).flatten().tolist()
        for index in indices:
            advantage = float(critic.returns[index]) - baseline
            actor.upsert(
                critic.z[index],
                critic.delta[index],
                float(critic.returns[index]),
                advantage,
                int(critic.episode[index]),
                int(critic.time[index]),
            )


def _episode_times(
    steps: list[MemoryStep], indices: list[int], start_time: int
) -> list[int]:
    times: list[int] = []
    clock = int(start_time)
    previous_traj: int | None = None
    for index in indices:
        traj_t = int(steps[index].traj_t)
        if previous_traj is not None:
            clock += traj_t - previous_traj
        times.append(clock)
        previous_traj = traj_t
    return times


def _segment_returns(
    steps: list[MemoryStep],
    indices: list[int],
    times: list[int],
    discount: float,
) -> list[float]:
    returns = [0.0] * len(indices)
    for local in range(len(indices) - 1, -1, -1):
        realized = float(steps[indices[local]].reward)
        if not steps[indices[local]].done and local + 1 < len(indices):
            gap = int(times[local + 1] - times[local])
            realized = realized + (float(discount) ** gap) * returns[local + 1]
        returns[local] = realized
    return returns


def retrieve_returns(
    bank: CriticReturnBank,
    query_z: torch.Tensor,
    query_episode: torch.Tensor,
    query_time: torch.Tensor,
    topk: int,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read ``(mean, std, count)`` of neighbor returns."""
    mean, std, count = _retrieve(
        query_z,
        query_episode,
        query_time,
        bank.z,
        bank.episode,
        bank.time,
        bank.valid,
        bank.returns.unsqueeze(-1),
        topk,
        tau,
    )
    return mean.reshape(-1), std.reshape(-1), count.reshape(-1)
