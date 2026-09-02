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

"""Action routing for the long-horizon RLT Stage-2 rollout path."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.algorithms.rlt.route import SimulatorRLTRoute

__all__ = ["LongHorizonRLTRoute", "build_rlt_lh_route"]


class LongHorizonRLTRoute(SimulatorRLTRoute):
    """Simulator actor/reference/expert routing for long-horizon RLT.

    The phase and intervention flags are still produced by the environment.
    History is not consumed by the router itself; it is passed through the RLT
    observation and later encoded by the Stage-2 policy.

    Long-horizon environments such as CALVIN do not hand-craft an insertion
    ``critical_phase`` flag. For those environments the student actor is
    allowed to act at every chunk and history becomes the signal that
    distinguishes where the episode currently is.
    """

    def __init__(
        self,
        *,
        use_schedule: bool,
        warmup_updates: int,
        default_actor_switch: bool = False,
    ) -> None:
        super().__init__(use_schedule=use_schedule, warmup_updates=warmup_updates)
        self.default_actor_switch = bool(default_actor_switch)

    def route(self, ctx: Any) -> Any:
        if ctx.rlt_switch_flags is None and self.default_actor_switch:
            ctx.rlt_switch_flags = torch.ones(
                (ctx.student_actions.shape[0], 1),
                dtype=torch.bool,
                device=ctx.student_actions.device,
            )
        return super().route(ctx)


def build_rlt_lh_route(cfg: Any) -> LongHorizonRLTRoute:
    schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
    train_env_cfg = cfg.env.get("train", {}) or {}
    env_type = str(train_env_cfg.get("env_type", ""))
    return LongHorizonRLTRoute(
        use_schedule=bool(schedule_cfg.get("enable", False)),
        warmup_updates=int(schedule_cfg.get("warmup_post_collect_updates", 0)),
        default_actor_switch=env_type != "maniskill_rlt",
    )
