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
gated by $$(z, a)$$ kernels. Neighbor mix matches a short history at
truncation and borrows that row's local bootstrap $$Q$$.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.algorithms.zap.backup import (
    attach_episode_zap_paths,
    compute_zap_path_targets_batch,
    flatten_path_actor_obs,
    gather_path_atom,
    gather_path_history,
    reduce_zap_metrics,
    select_zap_include_batch,
    zap_bootstrap_index,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    AsyncRLTACFSDPPolicy,
    RLTACFSDPPolicy,
)


def _flatten_batch_obs(
    obs: dict[str, torch.Tensor], batch_size: int
) -> dict[str, torch.Tensor]:
    """Align replay-batch obs to ``batch_size`` rows.

    Replay samples are already per-transition rows ``[B, ...]``, so the
    leading dim is preserved and only validated.
    """
    flat = {}
    for key, value in obs.items():
        if not torch.is_tensor(value):
            continue
        if value.dim() < 1 or int(value.shape[0]) != batch_size:
            raise ValueError(
                f"ZAP critic expected obs['{key}'] with leading dim "
                f"{batch_size}, got shape {tuple(value.shape)}."
            )
        flat[key] = value.reshape(batch_size, *value.shape[1:])
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

    def _zap_neighbor_tau_z(self) -> float:
        value = self.cfg.algorithm.get("za_neighbor_tau_z", None)
        if value is None:
            return self._zap_tau_z()
        return float(value)

    def _zap_neighbor_tau_a(self) -> float:
        value = self.cfg.algorithm.get("za_neighbor_tau_a", None)
        if value is None:
            return self._zap_tau_a()
        return float(value)

    def _zap_uniform_entropy_ratio(self) -> float:
        return float(self.cfg.algorithm.get("za_uniform_entropy_ratio", 0.95))

    def _zap_enable_neighbors(self) -> bool:
        return bool(self.cfg.algorithm.get("za_enable_neighbors", True))

    def _zap_neighbor_hist_len(self) -> int:
        return int(self.cfg.algorithm.get("za_neighbor_hist_len", 0))

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
        """Run the actor on a batch of $$z$$ rows using aligned template obs."""
        batch = int(z.shape[0])
        obs = self._obs_with_z(template_obs, z, batch)
        actions, _, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=obs,
            deterministic=True,
        )
        return actions.reshape(batch, -1).detach()

    def _obs_with_z(
        self,
        template_obs: dict[str, torch.Tensor],
        z: torch.Tensor,
        batch: int,
    ) -> dict[str, torch.Tensor]:
        obs: dict[str, torch.Tensor] = {}
        z_flat = z.reshape(batch, -1)
        for key, value in template_obs.items():
            if not torch.is_tensor(value):
                continue
            if key == "z_rl":
                obs[key] = z_flat.to(device=value.device, dtype=value.dtype)
                continue
            if value.shape[0] == batch:
                obs[key] = value
                continue
            row = value[0] if value.dim() >= 1 and value.shape[0] == 1 else value
            obs[key] = row.unsqueeze(0).expand(batch, *row.shape).contiguous()
        return obs

    def _policy_actions_for_paths(
        self,
        path_z: torch.Tensor,
        path_proprio: torch.Tensor,
        path_ref: torch.Tensor,
    ) -> torch.Tensor:
        """One actor forward over every path atom with that atom's (ref, z, proprio)."""
        batch, length = int(path_z.shape[0]), int(path_z.shape[1])
        obs = flatten_path_actor_obs(path_z, path_proprio, path_ref)
        device = path_z.device
        dtype = path_z.dtype
        obs = {
            key: value.to(device=device, dtype=dtype) for key, value in obs.items()
        }
        actions, _, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=obs,
            deterministic=True,
        )
        return actions.reshape(batch, length, -1).detach()

    def _q_values_for_z(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        template_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch = int(z.shape[0])
        obs = self._obs_with_z(template_obs, z, batch)
        return self._q_values_for_obs(obs, actions)

    def _q_values_for_state(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate target Q at ``(z, proprio, a)`` without borrowing query obs."""
        batch = int(z.shape[0])
        obs = {
            "z_rl": z.reshape(batch, -1),
            "proprio": proprio.reshape(batch, -1).to(
                device=z.device, dtype=z.dtype
            ),
        }
        return self._q_values_for_obs(obs, actions)

    def _q_values_for_obs(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(obs["z_rl"].shape[0])
        all_q = self.target_model(
            forward_type=ForwardType.SAC_Q,
            obs=obs,
            actions=actions.reshape(batch, -1).to(
                device=obs["z_rl"].device, dtype=obs["z_rl"].dtype
            ),
        )
        return self._min_twin_q(all_q).reshape(batch).detach().float()

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
        path_proprio = batch.get("zap_path_proprio")
        path_ref = batch.get("zap_path_ref")
        path_r = batch.get("zap_path_r")
        path_h = batch.get("zap_path_horizon")
        path_done = batch.get("zap_path_done")
        path_len = batch.get("zap_path_len")
        if (
            not isinstance(path_z, torch.Tensor)
            or not isinstance(path_a, torch.Tensor)
            or not isinstance(path_proprio, torch.Tensor)
            or not isinstance(path_ref, torch.Tensor)
            or not isinstance(path_r, torch.Tensor)
            or not isinstance(path_done, torch.Tensor)
            or not isinstance(path_len, torch.Tensor)
        ):
            raise ValueError(
                "ZAP critic requires zap_path_* fields including "
                "zap_path_proprio and zap_path_ref. Ensure "
                "attach_episode_zap_paths ran at transition ingest."
            )

        def _path_rows(tensor: torch.Tensor, *, feature: bool) -> torch.Tensor:
            if tensor.dim() >= 4:
                return tensor.reshape(-1, *tensor.shape[2:])
            if tensor.dim() == 3 and (not feature) and tensor.shape[1] == 1:
                return tensor.squeeze(1)
            return tensor

        path_z = _path_rows(path_z, feature=True)
        path_a = _path_rows(path_a, feature=True)
        path_proprio = _path_rows(path_proprio, feature=True)
        path_ref = _path_rows(path_ref, feature=True)
        path_r = _path_rows(path_r, feature=False)
        path_done = _path_rows(path_done, feature=False)
        path_len = _path_rows(path_len, feature=False).reshape(-1)
        path_h = (
            _path_rows(path_h, feature=False)
            if isinstance(path_h, torch.Tensor)
            else torch.ones_like(path_r)
        )

        batch_size = int(path_z.shape[0])
        gamma = float(self.cfg.algorithm.gamma)
        obs_flat = _flatten_batch_obs(curr_obs, batch_size)
        device = obs_flat["z_rl"].device
        path_z = path_z.to(device)
        path_a = path_a.to(device)
        path_proprio = path_proprio.to(device)
        path_ref = path_ref.to(device)
        path_r = path_r.to(device)
        path_done = path_done.to(device)
        path_len = path_len.to(device).clamp(min=1)
        path_h = path_h.to(device)
        used_len = int(path_len.max().item())
        path_a = path_a[:, :used_len]
        path_r = path_r[:, :used_len]
        path_done = path_done[:, :used_len]
        path_h = path_h[:, :used_len]
        # Keep one extra state atom after the last reward for s_{t+n} bootstrap.
        state_len = min(int(path_z.shape[1]), used_len + 1)
        path_z = path_z[:, :state_len]
        path_proprio = path_proprio[:, :state_len]
        path_ref = path_ref[:, :state_len]

        with torch.no_grad():
            a_pi = self._policy_actions_for_paths(path_z, path_proprio, path_ref)
            include, _ = select_zap_include_batch(
                path_z=path_z[:, :used_len],
                path_a_buf=path_a,
                path_a_pi=a_pi[:, :used_len],
                path_dones=path_done,
                path_len=path_len,
                tau_z=self._zap_tau_z(),
                tau_a=self._zap_tau_a(),
                truncate_eps=self._zap_truncate_eps(),
            )
            trunc_idx = zap_bootstrap_index(include, z_len=int(path_z.shape[1]))
            trunc_z = gather_path_atom(path_z, trunc_idx)
            trunc_a = gather_path_atom(a_pi, trunc_idx)
            trunc_proprio = gather_path_atom(path_proprio, trunc_idx)
            local_q = self._q_values_for_state(trunc_z, trunc_proprio, trunc_a)

            neighbor_z = neighbor_a = neighbor_q = neighbor_query = exclude_mask = None
            hist_len = self._zap_neighbor_hist_len()
            if self._zap_enable_neighbors() and batch_size > 1:
                if hist_len >= 1:
                    neighbor_query = gather_path_history(
                        path_z,
                        path_proprio,
                        trunc_idx,
                        hist_len=hist_len,
                    )
                else:
                    neighbor_query = trunc_z.detach().float()
                neighbor_z = neighbor_query
                neighbor_a = trunc_a.detach().float()
                neighbor_q = local_q
                exclude_mask = torch.eye(
                    batch_size, dtype=torch.bool, device=device
                )

            targets, zap_metrics = compute_zap_path_targets_batch(
                path_z=path_z,
                path_a_buf=path_a,
                path_a_pi=a_pi,
                path_rewards=path_r,
                path_dones=path_done,
                path_horizons=path_h,
                path_len=path_len,
                gamma=gamma,
                tau_z=self._zap_tau_z(),
                tau_a=self._zap_tau_a(),
                truncate_eps=self._zap_truncate_eps(),
                local_bootstrap_q=local_q,
                neighbor_z=neighbor_z,
                neighbor_a=neighbor_a,
                neighbor_q=neighbor_q,
                neighbor_query=neighbor_query,
                exclude_mask=exclude_mask,
                neighbor_top_k=self._zap_neighbor_k(),
                uniform_entropy_ratio=self._zap_uniform_entropy_ratio(),
                neighbor_tau_z=self._zap_neighbor_tau_z(),
                neighbor_tau_a=self._zap_neighbor_tau_a(),
            )

        # One target per replay row; twin-Q broadcast via expand_as below.
        target_q_values = targets.to(self.torch_dtype).reshape(batch_size, 1)

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
        critic_metrics.update(reduce_zap_metrics(zap_metrics))
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
