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

"""Long-horizon RLT Stage-2 actor workers.

These workers reuse the existing RLT actor-critic losses and replay plumbing,
but select transition replay for long-horizon environments such as CALVIN that
do not naturally trigger the original ManiSkill RLT transition path.
"""

from __future__ import annotations

from rlinf.algorithms.rlt_lh.transition import use_lh_simulator_transition_replay
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    collect_trajectory_replay_metrics,
    trajectory_has_bool_tensor,
)
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    AsyncRLTACFSDPPolicy,
    RLTACFSDPPolicy,
)


class RLTLHReplayMixin:
    """Route trajectory ingestion through long-horizon transition replay."""

    def forward_critic(self, batch):
        if use_lh_simulator_transition_replay(self.cfg):
            batch = dict(batch)
            batch["terminations"] = batch["dones"]
        return super().forward_critic(batch)

    def _ingest_rollout_trajectories(self, recv_list: list[Trajectory]):
        if not use_lh_simulator_transition_replay(self.cfg):
            return super()._ingest_rollout_trajectories(recv_list)

        replay_list = []
        completed = 0
        for traj in recv_list:
            assert isinstance(traj, Trajectory)
            transition_trajs, completed_count = self._transition_replay_trajectories(
                traj
            )
            replay_list.extend(transition_trajs)
            completed += completed_count

        self._last_replay_metrics = {
            **self._transition_replay_metrics(replay_list),
            **collect_trajectory_replay_metrics(recv_list, reducer=all_reduce_dict),
        }
        self.replay_buffer.add_trajectories(replay_list)

        if self.demo_buffer is not None:
            intervene_traj_list = [
                traj
                for traj in replay_list
                if trajectory_has_bool_tensor(traj.intervene_flags)
            ]
            if intervene_traj_list:
                self.demo_buffer.add_trajectories(intervene_traj_list)

        return len(replay_list), completed


class RLTLHACFSDPPolicy(RLTLHReplayMixin, RLTACFSDPPolicy):
    """Synchronous long-horizon RLT Stage-2 actor worker."""


class AsyncRLTLHACFSDPPolicy(RLTLHReplayMixin, AsyncRLTACFSDPPolicy):
    """Asynchronous long-horizon RLT Stage-2 actor worker."""
