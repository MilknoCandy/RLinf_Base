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

"""RLT Stage 2 worker with Decoupled Q-Chunking critics.

Online-only DQC: a long-horizon chunk critic ``Q^h`` is trained with a
multi-chunk unbiased backup, then a policy-chunk critic ``Q^k`` is distilled
from it by expectile (official ``agents/dqc.py``). Rollout uses Best-of-N on
``Q^k``. There is no offline pretrain; the frozen VLA is the behavior prior.
"""

from __future__ import annotations

import torch

from rlinf.algorithms.qc import expectile_loss, pack_dqc_windows, validate_qc_n_step
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_qc_policy_worker import QCCriticMixin
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    AsyncRLTACFSDPPolicy,
    RLTACFSDPPolicy,
)


class DQCCriticMixin(QCCriticMixin):
    """Long-chunk critic + distilled policy-chunk critic."""

    def _dqc_backup_chunks(self) -> int:
        chunk_len, _ = self._chunk_shape()
        raw_steps = self.cfg.algorithm.get("dqc_backup_horizon", None)
        if raw_steps is None:
            raw_steps = self.cfg.actor.model.get("chunk_critic_steps", chunk_len)
        backup_steps = int(raw_steps)
        if backup_steps % chunk_len != 0:
            raise ValueError(
                "dqc_backup_horizon must be a multiple of num_action_chunks="
                f"{chunk_len}, got {backup_steps}."
            )
        return max(backup_steps // chunk_len, 1)

    def _dqc_kappa_d(self) -> float:
        return float(self.cfg.algorithm.get("dqc_kappa_d", 0.9))

    def _dqc_kappa_b(self) -> float:
        return float(self.cfg.algorithm.get("dqc_kappa_b", 0.9))

    def _postprocess_env_transitions(self, env_transitions, *, n_step, gamma):
        del n_step
        chunk_len, action_dim = self._chunk_shape()
        return pack_dqc_windows(
            env_transitions,
            chunk_len=chunk_len,
            action_dim=action_dim,
            backup_chunks=self._dqc_backup_chunks(),
            gamma=gamma,
            z_dim=int(self.cfg.actor.model.z_dim),
            proprio_dim=int(self.cfg.actor.model.proprio_dim),
        )

    def _dqc_bootstrap_obs(self, batch) -> dict[str, torch.Tensor]:
        return {
            "z_rl": batch["dqc_bootstrap_z"],
            "proprio": batch["dqc_bootstrap_proprio"],
            "ref_chunk": batch["dqc_bootstrap_ref"],
        }

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        if not hasattr(self.model, "chunk_q_head"):
            raise RuntimeError(
                "DQC requires actor.model.chunk_critic_steps so RLTMLPPolicy "
                "builds chunk_q_head."
            )
        curr_obs = batch["curr_obs"]
        chunk_actions = batch["dqc_chunk_actions"]
        n_step_returns = batch["dqc_n_step_returns"].to(self.torch_dtype).reshape(-1, 1)
        bootstrap_discount = (
            batch["dqc_bootstrap_discount"].to(self.torch_dtype).reshape(-1, 1)
        )
        valid_chunks = batch["dqc_valid_chunks"].reshape(-1)
        required = int(self._dqc_backup_chunks())
        valid = (valid_chunks >= required).reshape(-1, 1).to(self.torch_dtype)
        done = batch["dqc_done"].reshape(-1, 1).to(dtype=torch.bool)
        bootstrap_obs = self._dqc_bootstrap_obs(batch)

        with torch.no_grad():
            next_v = self.target_model.scale_v_forward(
                bootstrap_obs, horizon=self.model.chunk_len
            )
            if next_v.dim() == 1:
                next_v = next_v.unsqueeze(-1)
            target_chunk = n_step_returns + (~done) * bootstrap_discount * next_v

        chunk_q = self.model.chunk_q_forward(curr_obs, chunk_actions)
        target_chunk = target_chunk.to(dtype=chunk_q.dtype)
        chunk_err = (chunk_q - target_chunk.expand_as(chunk_q)).square()
        valid_rows = valid.reshape(-1) > 0
        if bool(valid_rows.any()):
            chunk_loss = chunk_err[valid_rows].mean()
        else:
            chunk_loss = chunk_q.sum() * 0.0

        policy_q = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=batch["actions"],
        )
        policy_min = self._min_twin_q(policy_q)
        with torch.no_grad():
            distill_target = self._min_twin_q(chunk_q.detach())
        if bool(valid_rows.any()):
            distill_loss = expectile_loss(
                policy_min[valid_rows],
                distill_target[valid_rows],
                self._dqc_kappa_d(),
            )
        else:
            distill_loss = policy_q.sum() * 0.0

        with torch.no_grad():
            data_q = self._min_twin_q(
                self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=curr_obs,
                    actions=batch["actions"],
                )
            )
        v_pred = self.model.scale_v_forward(curr_obs, horizon=self.model.chunk_len)
        if v_pred.dim() == 1:
            v_pred = v_pred.unsqueeze(-1)
        value_loss = expectile_loss(v_pred, data_q, self._dqc_kappa_b())

        critic_loss = chunk_loss + distill_loss + value_loss
        metrics = {
            "q_data": policy_q.mean().item(),
            "dqc/chunk_q": chunk_q.mean().item(),
            "dqc/chunk_loss": chunk_loss.detach().item(),
            "dqc/distill_loss": distill_loss.detach().item(),
            "dqc/value_loss": value_loss.detach().item(),
            "dqc/valid_window_rate": float(valid.mean().item()),
            "dqc/backup_chunks": float(required),
        }
        return critic_loss, metrics


class DQCFSDPPolicy(DQCCriticMixin, RLTACFSDPPolicy):
    """Synchronous RLT Stage 2 actor with DQC critics."""

    def __init__(self, cfg):
        super().__init__(cfg)
        validate_qc_n_step(int(self.cfg.algorithm.get("n_step", 1)))
        self._last_qc_metrics = {}


class AsyncDQCFSDPPolicy(DQCCriticMixin, AsyncRLTACFSDPPolicy):
    """Async RLT Stage 2 actor with DQC critics."""

    def __init__(self, cfg):
        super().__init__(cfg)
        validate_qc_n_step(int(self.cfg.algorithm.get("n_step", 1)))
        self._last_qc_metrics = {}
