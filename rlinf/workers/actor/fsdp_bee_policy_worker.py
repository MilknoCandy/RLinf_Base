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

"""Bee Stage-2 worker: residual actor + Correction Model + Lagrange multiplier.

Implements arXiv 2609.27450 on top of the existing RLT replay / schedule stack.
Human (or simulated expert) corrections seed ``demo_buffer`` (B_H); online
transitions land in ``replay_buffer`` (B_R). Batches already mix 1:1 when
``demo_buffer`` is configured.
"""

from __future__ import annotations

import torch

from rlinf.algorithms.bee import (
    BeeCorrectionModel,
    BeeMultiplier,
    bee_actor_loss,
    bee_multiplier_loss,
    correction_nll_loss,
    correction_update_steps,
    mahalanobis_rho,
    should_update_correction,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    AsyncRLTACFSDPPolicy,
    RLTACFSDPPolicy,
)


class BeeLossMixin:
    """Actor / correction / multiplier objectives for Bee."""

    def _bee_cfg(self):
        return self.cfg.algorithm.get("bee", {}) or {}

    def _rl_state(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        z = self._flatten_chunk(obs["z_rl"])
        proprio = self._flatten_chunk(obs["proprio"])
        return torch.cat([z, proprio], dim=-1)

    def _setup_bee_modules(self) -> None:
        bee_cfg = self._bee_cfg()
        z_dim = int(self.cfg.actor.model.z_dim)
        proprio_dim = int(self.cfg.actor.model.proprio_dim)
        chunk_len, action_dim = self._chunk_shape()
        flat_action_dim = chunk_len * action_dim
        state_dim = z_dim + proprio_dim
        hidden = int(self.cfg.actor.model.get("mlp_hidden_dim", 512))
        layers = int(self.cfg.actor.model.get("mlp_num_hidden_layers", 3))
        variance_floor = float(bee_cfg.get("variance_floor", 0.02))

        self.bee_correction = BeeCorrectionModel(
            state_dim=state_dim,
            action_dim=flat_action_dim,
            hidden_dim=hidden,
            num_hidden_layers=layers,
            variance_floor=variance_floor,
        ).to(device=self.device, dtype=self.torch_dtype)
        self.bee_multiplier = BeeMultiplier(
            state_dim=state_dim,
            hidden_dim=hidden,
            num_hidden_layers=layers,
        ).to(device=self.device, dtype=self.torch_dtype)

        corr_lr = float(bee_cfg.get("correction_lr", self.cfg.actor.optim.lr))
        mult_lr = float(bee_cfg.get("multiplier_lr", 3.0e-6))
        betas = (
            float(self.cfg.actor.optim.get("adam_beta1", 0.9)),
            float(self.cfg.actor.optim.get("adam_beta2", 0.95)),
        )
        eps = float(self.cfg.actor.optim.get("adam_eps", 1.0e-8))
        self.correction_optimizer = torch.optim.Adam(
            self.bee_correction.parameters(), lr=corr_lr, betas=betas, eps=eps
        )
        self.multiplier_optimizer = torch.optim.Adam(
            self.bee_multiplier.parameters(), lr=mult_lr, betas=betas, eps=eps
        )

        self.bee_epsilon = float(bee_cfg.get("epsilon", 0.3))
        self.bee_anchor_weight = float(bee_cfg.get("anchor_weight", 1.0))
        self.bee_update_interval_m = int(bee_cfg.get("update_interval_m", 100))
        self.bee_update_to_data_n = float(bee_cfg.get("update_to_data_n", 0.12))
        self.bee_correction_batch_size = int(
            bee_cfg.get("correction_batch_size", self.cfg.actor.global_batch_size)
        )
        self.bee_pretrain_correction_steps = int(
            bee_cfg.get("pretrain_correction_steps", 0)
        )
        self._new_corrections_since_update = 0
        self._bee_correction_pretrained = False

    def _maybe_pretrain_correction(self) -> dict[str, float]:
        if self._bee_correction_pretrained:
            return {}
        if self.demo_buffer is None:
            self._bee_correction_pretrained = True
            return {}
        steps = self.bee_pretrain_correction_steps
        if steps <= 0:
            # One pass at NM scale when seed demos are already loaded.
            steps = correction_update_steps(
                update_interval_m=max(self.demo_buffer.total_samples, 1),
                update_to_data_n=self.bee_update_to_data_n,
            )
            steps = max(steps, correction_update_steps())
        if self.demo_buffer.total_samples < 1:
            return {}
        metrics = self._run_correction_updates(steps)
        self._bee_correction_pretrained = True
        return {f"bee/pretrain_{k}": v for k, v in metrics.items()}

    def _run_correction_updates(self, num_steps: int) -> dict[str, float]:
        if self.demo_buffer is None or num_steps <= 0:
            return {}
        batch_size = min(
            int(self.bee_correction_batch_size // max(self._world_size, 1)),
            max(int(self.demo_buffer.total_samples), 1),
        )
        losses = []
        for _ in range(int(num_steps)):
            batch = self.demo_buffer.sample(batch_size)
            batch = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            state = self._rl_state(batch["curr_obs"])
            proposal = self._ref_chunk(batch["curr_obs"])
            human = self._flatten_chunk(batch["actions"])
            residual = human - proposal
            mu, sigma = self.bee_correction(state, proposal)
            loss = correction_nll_loss(residual, mu, sigma)
            self.correction_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.bee_correction.parameters(),
                float(self.cfg.actor.optim.get("clip_grad", 1.0)),
            )
            self.correction_optimizer.step()
            losses.append(float(loss.detach().item()))
        return {
            "correction_nll": float(sum(losses) / max(len(losses), 1)),
            "correction_steps": float(len(losses)),
        }

    def _maybe_update_correction(self) -> dict[str, float]:
        if not should_update_correction(
            self._new_corrections_since_update,
            update_interval_m=self.bee_update_interval_m,
        ):
            return {}
        steps = correction_update_steps(
            update_interval_m=self.bee_update_interval_m,
            update_to_data_n=self.bee_update_to_data_n,
        )
        metrics = self._run_correction_updates(steps)
        self._new_corrections_since_update = 0
        return metrics

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        if getattr(self, "qf_optimizer", None) is not None:
            self.qf_optimizer.zero_grad(set_to_none=True)

        curr_obs = batch["curr_obs"]
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        pi, log_pi, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            apply_reference_dropout=reference_dropout_prob > 0.0,
            reference_dropout_prob=reference_dropout_prob,
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)

        all_qf_pi = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=pi,
            detach_encoder=True,
        )
        qf_pi = self._q1(all_qf_pi)

        state = self._rl_state(curr_obs)
        proposal = self._ref_chunk(curr_obs)
        with torch.no_grad():
            mu, sigma = self.bee_correction(state, proposal)
            corrected = proposal + mu
        rho = mahalanobis_rho(
            self._flatten_chunk(pi), corrected, sigma.detach()
        )
        lam = self.bee_multiplier(state.detach())

        actor_loss, bee_metrics = bee_actor_loss(
            qf_pi,
            self._flatten_chunk(pi),
            proposal,
            rho,
            lam.detach(),
            anchor_weight=self.bee_anchor_weight,
        )
        bee_metrics["bee/lambda"] = float(lam.mean().detach().item())
        bee_metrics["bee/rho"] = float(rho.mean().detach().item())
        bee_metrics["q_pi"] = qf_pi.mean().detach().item()
        bee_metrics["reference_dropout_prob"] = reference_dropout_prob
        entropy = -log_pi.mean()
        return actor_loss, entropy, bee_metrics

    def _step_multiplier(self, batch: dict) -> dict[str, float]:
        """Dual ascent on a fresh batch so the actor graph is not retained."""
        curr_obs = batch["curr_obs"]
        with torch.no_grad():
            pi, _, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=curr_obs,
                apply_reference_dropout=False,
                deterministic=True,
            )
            state = self._rl_state(curr_obs)
            proposal = self._ref_chunk(curr_obs)
            mu, sigma = self.bee_correction(state, proposal)
            corrected = proposal + mu
            rho = mahalanobis_rho(
                self._flatten_chunk(pi), corrected, sigma
            )
        lam = self.bee_multiplier(state.detach())
        mult_loss, mult_metrics = bee_multiplier_loss(
            lam, rho, epsilon=self.bee_epsilon
        )
        self.multiplier_optimizer.zero_grad(set_to_none=True)
        mult_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.bee_multiplier.parameters(),
            float(self.cfg.actor.optim.get("clip_grad", 1.0)),
        )
        self.multiplier_optimizer.step()
        return mult_metrics


class BeeFSDPPolicy(BeeLossMixin, RLTACFSDPPolicy):
    """Synchronous Bee worker for LIBERO-PRO / real-world RLT interfaces."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.bee_correction = None
        self.bee_multiplier = None
        self.correction_optimizer = None
        self.multiplier_optimizer = None

    def setup_sac_components(self):
        super().setup_sac_components()
        self._setup_bee_modules()
        # Seed B_H may already be loaded via demo_buffer.load_path.
        pretrain_metrics = self._maybe_pretrain_correction()
        if pretrain_metrics:
            self.log_info(f"Bee Correction Model pretrain: {pretrain_metrics}")

    def _ingest_rollout_trajectories(self, recv_list):
        before = (
            0 if self.demo_buffer is None else int(self.demo_buffer.total_samples)
        )
        added, completed = super()._ingest_rollout_trajectories(recv_list)
        after = (
            0 if self.demo_buffer is None else int(self.demo_buffer.total_samples)
        )
        self._new_corrections_since_update += max(after - before, 0)
        return added, completed

    def update_one_epoch(self, train_actor: bool = True):
        corr_metrics = self._maybe_update_correction()
        metrics = super().update_one_epoch(train_actor=train_actor)
        if train_actor and self.update_step % self.critic_actor_ratio == 0:
            global_batch_size_per_rank = (
                self.cfg.actor.global_batch_size // self._world_size
            )
            global_batch = next(self.buffer_dataloader_iter)
            micro_batches = split_dict_to_chunk(
                global_batch,
                max(global_batch_size_per_rank // self.cfg.actor.micro_batch_size, 1),
            )
            batch = put_tensor_device(micro_batches[0], device=self.device)
            mult_metrics = self._step_multiplier(batch)
            metrics.update({f"actor/{k}": v for k, v in mult_metrics.items()})
        if corr_metrics:
            metrics.update({f"bee/{k}": v for k, v in corr_metrics.items()})
        return metrics


class AsyncBeeFSDPPolicy(BeeFSDPPolicy, AsyncRLTACFSDPPolicy):
    """Async Bee worker sharing the RLT schedule path."""

    def __init__(self, cfg):
        AsyncRLTACFSDPPolicy.__init__(self, cfg)
        self.bee_correction = None
        self.bee_multiplier = None
        self.correction_optimizer = None
        self.multiplier_optimizer = None
