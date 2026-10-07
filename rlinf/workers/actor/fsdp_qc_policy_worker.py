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

"""RLT Stage 2 worker with online Q-chunking.

Official QC (Li et al., NeurIPS 2025) trains a chunked critic and extracts
actions by Best-of-N from a behavior policy. Here the frozen Stage-1 VLA plus
the residual actor are that behavior prior: Best-of-N is used both for the TD
bootstrap and (via ``action_extract: best_of_n``) for environment interaction.
There is no offline critic pretrain.
"""

from __future__ import annotations

import torch

from rlinf.algorithms.qc import (
    build_qc_critic_candidates,
    flatten_chunk_actions,
    repeat_obs,
    select_best_of_n_actions,
    validate_qc_n_step,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    AsyncRLTACFSDPPolicy,
    RLTACFSDPPolicy,
)


class QCCriticMixin:
    """Override only the critic next-action used in the TD target."""

    _last_qc_metrics: dict[str, float]

    def _qc_num_samples(self) -> int:
        return int(self.cfg.algorithm.get("qc_num_samples", 8))

    def _qc_include_ref_chunk(self) -> bool:
        return bool(self.cfg.algorithm.get("qc_include_ref_chunk", True))

    def _qc_include_actor_mean(self) -> bool:
        return bool(self.cfg.algorithm.get("qc_include_actor_mean", True))

    def _qc_flat_actions(self, actions: torch.Tensor) -> torch.Tensor:
        chunk_len, action_dim = self._chunk_shape()
        return flatten_chunk_actions(
            self._flatten_chunk(actions),
            chunk_len=chunk_len,
            action_dim=action_dim,
        )

    def _qc_actor_actions(
        self,
        obs: dict[str, torch.Tensor],
        *,
        num_samples: int,
        deterministic: bool,
    ) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
        expanded_obs = repeat_obs(obs, num_samples)
        actions, _, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=expanded_obs,
            deterministic=deterministic,
        )
        flat = self._qc_flat_actions(actions)
        if num_samples == 1:
            return flat
        batch_size = next(
            int(value.shape[0])
            for value in obs.values()
            if torch.is_tensor(value)
        )
        return flat.reshape(batch_size, num_samples, -1)

    def _next_actions_for_critic_target(self, next_obs):
        """Select ``a_next`` with online Q, then let the parent bootstrap."""
        include_ref = self._qc_include_ref_chunk()
        include_mean = self._qc_include_actor_mean()
        num_samples = self._qc_num_samples()
        if not include_ref and not include_mean and num_samples < 1:
            return super()._next_actions_for_critic_target(next_obs)

        ref_chunk = self._ref_chunk(next_obs) if include_ref else None
        actor_mean = None
        if include_mean:
            actor_mean = self._qc_actor_actions(
                next_obs, num_samples=1, deterministic=True
            )
        actor_samples = None
        if num_samples > 0:
            actor_samples = self._qc_actor_actions(
                next_obs, num_samples=num_samples, deterministic=False
            )

        candidates = build_qc_critic_candidates(
            actor_samples=actor_samples,
            ref_chunk=ref_chunk,
            actor_mean=actor_mean,
        )
        batch_size, num_candidates, _ = candidates.shape
        expanded_obs = repeat_obs(next_obs, num_candidates)
        all_q_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=expanded_obs,
            actions=candidates.reshape(batch_size * num_candidates, -1),
        )
        q_values = self._min_twin_q(all_q_values).reshape(batch_size, num_candidates)
        best_actions, indices = select_best_of_n_actions(q_values, candidates)

        selected_ref = (
            torch.mean((indices == 0).to(torch.float32)).item() if include_ref else 0.0
        )
        mean_index = 1 if include_ref else 0
        selected_mean = (
            torch.mean((indices == mean_index).to(torch.float32)).item()
            if include_mean
            else 0.0
        )
        self._last_qc_metrics = {
            "qc/num_candidates": float(num_candidates),
            "qc/selected_ref_rate": selected_ref,
            "qc/selected_mean_rate": selected_mean,
            "qc/q_best_mean": q_values.max(dim=-1).values.mean().item(),
            "qc/q_candidate_std": q_values.std(dim=-1).mean().item(),
        }
        dummy_log_pi = torch.zeros(
            batch_size, 1, device=best_actions.device, dtype=best_actions.dtype
        )
        return best_actions, dummy_log_pi, None

    def forward_critic(self, batch):
        self._last_qc_metrics = {}
        critic_loss, critic_metrics = super().forward_critic(batch)
        critic_metrics.update(self._last_qc_metrics)
        return critic_loss, critic_metrics


class QCFSDPPolicy(QCCriticMixin, RLTACFSDPPolicy):
    """Synchronous RLT Stage 2 actor with a Q-chunking critic target."""

    def __init__(self, cfg):
        super().__init__(cfg)
        validate_qc_n_step(int(self.cfg.algorithm.get("n_step", 1)))
        self._last_qc_metrics = {}


class AsyncQCFSDPPolicy(QCCriticMixin, AsyncRLTACFSDPPolicy):
    """Async RLT Stage 2 actor with a Q-chunking critic target."""

    def __init__(self, cfg):
        super().__init__(cfg)
        validate_qc_n_step(int(self.cfg.algorithm.get("n_step", 1)))
        self._last_qc_metrics = {}
