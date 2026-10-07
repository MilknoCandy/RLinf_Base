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

"""RLT Stage 2 worker with ZAP critic targets.

ZAP ((Z, A) Path) keeps $$Q(z, a)$$ on a single atom. Path continuation is
gated by $$(z, a)$$ kernels; truncation bootstraps from neighbor $$Q$$ values
under the current policy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.algorithms.zap import (
    attach_episode_zap_paths,
    compute_zap_path_target,
    stack_path_metrics,
)
from rlinf.algorithms.zap.backup import select_zap_include
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    AsyncRLTACFSDPPolicy,
    RLTACFSDPPolicy,
)


def _flatten_batch_obs(
    obs: dict[str, torch.Tensor], batch_size: int
) -> dict[str, torch.Tensor]:
    flat = {}
    for key, value in obs.items():
        if not torch.is_tensor(value):
            continue
        flat[key] = value.reshape(batch_size, *value.shape[2:])
    return flat


class ZAPCriticMixin:
    """Override critic targets with ZAP path backups."""

    _last_zap_metrics: dict[str, float]

    def _zap_tau_z(self) -> float:
        return float(self.cfg.algorithm.get("za_tau_z", 0.1))

    def _zap_tau_a(self) -> float:
        return float(self.cfg.algorithm.get("za_tau_a", 0.05))

    def _zap_truncate_eps(self) -> float:
        return float(self.cfg.algorithm.get("za_truncate_eps", 0.01))

    def _zap_max_atoms(self) -> int:
        return int(self.cfg.algorithm.get("za_max_horizon", 32))

    def _zap_neighbor_k(self) -> int:
        return int(self.cfg.algorithm.get("za_neighbor_k", 32))

    def _zap_uniform_entropy_ratio(self) -> float:
        return float(self.cfg.algorithm.get("za_uniform_entropy_ratio", 0.95))

    def _zap_enable_neighbors(self) -> bool:
        return bool(self.cfg.algorithm.get("za_enable_neighbors", True))

    def _apply_zap_paths(self, transitions: list) -> list:
        attach_episode_zap_paths(
            transitions,
            max_atoms=self._zap_max_atoms(),
            gamma=float(self.cfg.algorithm.gamma),
        )
        return transitions

    def _reattach_zap_paths(self, replay_trajectories: list) -> list:
        if not replay_trajectories:
            return replay_trajectories
        episode: list = []
        rebuilt: list = []
        for transition in replay_trajectories:
            episode.append(transition)
            done = (
                isinstance(transition.dones, torch.Tensor)
                and transition.dones.reshape(-1).to(torch.bool).any()
            )
            if done:
                self._apply_zap_paths(episode)
                rebuilt.extend(episode)
                episode = []
        if episode:
            self._apply_zap_paths(episode)
            rebuilt.extend(episode)
        return rebuilt

    def _policy_actions_for_z(
        self,
        z: torch.Tensor,
        template_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run the actor on a batch of path $$z$$ rows using a template obs."""
        batch = int(z.shape[0])
        obs: dict[str, torch.Tensor] = {}
        for key, value in template_obs.items():
            if not torch.is_tensor(value):
                continue
            row = value
            while row.dim() > 1 and row.shape[0] == 1:
                row = row[0]
            if key == "z_rl":
                obs[key] = z.to(device=value.device, dtype=value.dtype)
                continue
            obs[key] = (
                row.unsqueeze(0)
                .expand(batch, *row.shape)
                .reshape(batch, *row.shape)
                .contiguous()
            )
        actions, _, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=obs,
            deterministic=True,
        )
        return actions.reshape(batch, -1).detach()

    def _local_bootstrap_q(
        self,
        z: torch.Tensor,
        a_pi: torch.Tensor,
        template_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        obs: dict[str, torch.Tensor] = {}
        device = None
        dtype = None
        for key, value in template_obs.items():
            if not torch.is_tensor(value):
                continue
            row = value
            while row.dim() > 1 and row.shape[0] == 1:
                row = row[0]
            device = value.device
            dtype = value.dtype
            if key == "z_rl":
                obs[key] = z.reshape(1, -1).to(device=device, dtype=dtype)
                continue
            obs[key] = row.reshape(1, *row.shape).to(device=device, dtype=dtype)
        all_q = self.target_model(
            forward_type=ForwardType.SAC_Q,
            obs=obs,
            actions=a_pi.reshape(1, -1).to(device=device, dtype=dtype),
        )
        return self._min_twin_q(all_q).reshape(()).detach().float()

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        if self._use_tsac_critic():
            raise ValueError("ZAP cannot be combined with the TSAC critic head.")
        if self.cfg.algorithm.get("q_head_type", "default") == "crossq":
            raise ValueError("ZAP cannot be combined with crossq.")

        curr_obs = batch["curr_obs"]
        actions = batch["actions"]
        path_z = batch.get("zap_path_z")
        path_a = batch.get("zap_path_a")
        path_r = batch.get("zap_path_r")
        path_h = batch.get("zap_path_horizon")
        path_done = batch.get("zap_path_done")
        path_len = batch.get("zap_path_len")
        if (
            not isinstance(path_z, torch.Tensor)
            or not isinstance(path_a, torch.Tensor)
            or not isinstance(path_r, torch.Tensor)
            or not isinstance(path_done, torch.Tensor)
            or not isinstance(path_len, torch.Tensor)
        ):
            raise ValueError(
                "ZAP critic requires zap_path_* fields. Ensure "
                "attach_episode_zap_paths ran at transition ingest."
            )

        def _rows(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(-1, *tensor.shape[2:])

        path_z = _rows(path_z)
        path_a = _rows(path_a)
        path_r = _rows(path_r)
        path_done = _rows(path_done)
        path_len = _rows(path_len).reshape(-1)
        path_h = _rows(path_h) if isinstance(path_h, torch.Tensor) else torch.ones_like(path_r)

        batch_size = int(path_z.shape[0])
        gamma = float(self.cfg.algorithm.gamma)
        obs_flat = _flatten_batch_obs(curr_obs, batch_size)

        neighbor_z = neighbor_a = neighbor_q = None
        if self._zap_enable_neighbors() and batch_size > 1:
            flat_z = obs_flat["z_rl"].reshape(batch_size, -1)
            a_pi_batch = self._policy_actions_for_z(flat_z, obs_flat)
            neighbor_z = flat_z.detach().float()
            neighbor_a = actions.reshape(batch_size, -1).detach().float()
            with torch.no_grad():
                all_q = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=obs_flat,
                    actions=a_pi_batch.to(
                        device=flat_z.device, dtype=obs_flat["z_rl"].dtype
                    ),
                )
                neighbor_q = (
                    self._min_twin_q(all_q).reshape(batch_size).detach().float()
                )

        targets = []
        metric_list = []
        for i in range(batch_size):
            length = max(int(path_len[i].item()), 1)
            z_i = path_z[i, :length]
            a_buf_i = path_a[i, :length]
            r_i = path_r[i, :length]
            d_i = path_done[i, :length]
            h_i = path_h[i, :length]
            template = {key: value[i : i + 1] for key, value in obs_flat.items()}
            a_pi_i = self._policy_actions_for_z(z_i, template)
            include, _ = select_zap_include(
                path_z=z_i,
                path_a_buf=a_buf_i,
                path_a_pi=a_pi_i,
                path_dones=d_i,
                tau_z=self._zap_tau_z(),
                tau_a=self._zap_tau_a(),
                truncate_eps=self._zap_truncate_eps(),
            )
            boot_idx = include if include < length else include - 1
            local_q = self._local_bootstrap_q(
                z_i[boot_idx], a_pi_i[boot_idx], template
            )
            if neighbor_z is not None:
                mask = torch.ones(batch_size, dtype=torch.bool, device=neighbor_z.device)
                mask[i] = False
                n_z, n_a, n_q = neighbor_z[mask], neighbor_a[mask], neighbor_q[mask]
            else:
                n_z = n_a = n_q = None
            y_i, metrics = compute_zap_path_target(
                path_z=z_i,
                path_a_buf=a_buf_i,
                path_a_pi=a_pi_i,
                path_rewards=r_i,
                path_dones=d_i,
                path_horizons=h_i,
                gamma=gamma,
                tau_z=self._zap_tau_z(),
                tau_a=self._zap_tau_a(),
                truncate_eps=self._zap_truncate_eps(),
                neighbor_z=n_z,
                neighbor_a=n_a,
                neighbor_q=n_q,
                local_bootstrap_q=local_q,
                neighbor_top_k=self._zap_neighbor_k(),
                uniform_entropy_ratio=self._zap_uniform_entropy_ratio(),
            )
            targets.append(y_i)
            metric_list.append(metrics)

        target_q_values = torch.stack(targets, dim=0).to(self.torch_dtype)
        if actions.dim() >= 2:
            target_q_values = target_q_values.reshape(*actions.shape[:2], 1)
        else:
            target_q_values = target_q_values.reshape(-1, 1)

        all_data_q_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
            detach_encoder=False,
        )
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        critic_metrics = {"q_data": float(all_data_q_values.mean().item())}
        critic_metrics.update(stack_path_metrics(metric_list))
        self._last_zap_metrics = critic_metrics
        return critic_loss, critic_metrics


class ZAPFSDPPolicy(ZAPCriticMixin, RLTACFSDPPolicy):
    """Synchronous RLT Stage 2 actor with ZAP critic targets."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._last_zap_metrics = {}

    def _transition_replay_trajectories(self, trajectory):
        replay_trajectories, completed = super()._transition_replay_trajectories(
            trajectory
        )
        return self._reattach_zap_paths(replay_trajectories), completed


class AsyncZAPFSDPPolicy(ZAPCriticMixin, AsyncRLTACFSDPPolicy):
    """Async RLT Stage 2 actor with ZAP critic targets."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._last_zap_metrics = {}

    def _transition_replay_trajectories(self, trajectory):
        replay_trajectories, completed = super()._transition_replay_trajectories(
            trajectory
        )
        return self._reattach_zap_paths(replay_trajectories), completed
