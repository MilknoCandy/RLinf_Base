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

"""RLT Stage 2 worker with Adaptive Q-Chunking critics.

Online-only AQC: a long-horizon critic ``Q^h`` uses the QC Best-of-N backup;
partial critics ``Q^k`` are trained on prefixes bootstrapped from ``V^h``
(AQC Eq. 12-15). Rollout scores prefixes with the discount-normalized
advantage (Eq. 10) and z-score (Eq. 16). Env stepping stays one policy chunk
because RLinf vectorized LIBERO chunks are fixed-length; ``k*`` chooses which
candidate chunk to commit, not a receding prefix of a single sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.algorithms.qc import expectile_loss, validate_qc_n_step
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_qc_policy_worker import QCCriticMixin
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    AsyncRLTACFSDPPolicy,
    RLTACFSDPPolicy,
)


class AQCCriticMixin(QCCriticMixin):
    """QC full-chunk backup plus AQC partial critics and value baselines."""

    def _aqc_kappa_v(self) -> float:
        return float(self.cfg.algorithm.get("aqc_kappa_v", 0.9))

    def _aqc_horizons(self) -> tuple[int, ...]:
        chunk_len, _ = self._chunk_shape()
        raw = self.cfg.algorithm.get("aqc_horizons", None)
        if raw is None:
            raw = self.cfg.actor.model.get("scale_critic_steps", [chunk_len])
        horizons = tuple(sorted({int(step) for step in raw}))
        if chunk_len not in horizons:
            horizons = tuple(sorted(set(horizons) | {chunk_len}))
        return horizons

    def _prefix_return(self, rewards: torch.Tensor, horizon: int) -> torch.Tensor:
        rewards = rewards.reshape(rewards.shape[0], -1).to(self.torch_dtype)
        prefix = rewards[:, :horizon]
        discounts = torch.pow(
            torch.as_tensor(self.cfg.algorithm.gamma, device=prefix.device),
            torch.arange(horizon, device=prefix.device, dtype=prefix.dtype),
        )
        return torch.sum(prefix * discounts, dim=-1, keepdim=True)

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        self._last_qc_metrics = {}
        full_loss, metrics = super().forward_critic(batch)
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        horizons = self._aqc_horizons()
        chunk_len, action_dim = self._chunk_shape()
        kappa = self._aqc_kappa_v()
        gamma = float(self.cfg.algorithm.gamma)
        flat_actions = self._flatten_chunk(actions)

        with torch.no_grad():
            v_h_next = self.target_model.scale_v_forward(
                next_obs, horizon=chunk_len
            )
            if v_h_next.dim() == 1:
                v_h_next = v_h_next.unsqueeze(-1)
            q_h_data = self._min_twin_q(
                self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=curr_obs,
                    actions=actions,
                )
            )

        extra_loss = torch.zeros((), device=full_loss.device, dtype=full_loss.dtype)
        v_h = self.model.scale_v_forward(curr_obs, horizon=chunk_len)
        if v_h.dim() == 1:
            v_h = v_h.unsqueeze(-1)
        extra_loss = extra_loss + expectile_loss(v_h, q_h_data, kappa)
        metrics["aqc/v_h"] = v_h.mean().item()

        for horizon in horizons:
            if horizon == chunk_len:
                continue
            prefix = flat_actions[:, : horizon * action_dim]
            q_k = self.model.scale_q_forward(curr_obs, prefix, horizon)
            reward_k = self._prefix_return(rewards, horizon)
            target_k = reward_k + (gamma**horizon) * v_h_next
            target_k = target_k.to(dtype=q_k.dtype)
            extra_loss = extra_loss + F.mse_loss(
                q_k, target_k.expand_as(q_k)
            )
            v_k = self.model.scale_v_forward(curr_obs, horizon)
            if v_k.dim() == 1:
                v_k = v_k.unsqueeze(-1)
            extra_loss = extra_loss + expectile_loss(
                v_k, self._min_twin_q(q_k).detach(), kappa
            )
            metrics[f"aqc/q_k{horizon}"] = q_k.mean().item()

        metrics.update(self._last_qc_metrics)
        metrics["aqc/partial_loss"] = extra_loss.detach().item()
        return full_loss + extra_loss, metrics


class AQCFSDPPolicy(AQCCriticMixin, RLTACFSDPPolicy):
    """Synchronous RLT Stage 2 actor with AQC critics."""

    def __init__(self, cfg):
        super().__init__(cfg)
        validate_qc_n_step(int(self.cfg.algorithm.get("n_step", 1)))
        self._last_qc_metrics = {}


class AsyncAQCFSDPPolicy(AQCCriticMixin, AsyncRLTACFSDPPolicy):
    """Async RLT Stage 2 actor with AQC critics."""

    def __init__(self, cfg):
        super().__init__(cfg)
        validate_qc_n_step(int(self.cfg.algorithm.get("n_step", 1)))
        self._last_qc_metrics = {}
