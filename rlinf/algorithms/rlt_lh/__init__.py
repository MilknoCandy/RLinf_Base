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

"""Long-horizon RLT Stage-2 helpers.

Kept separate from :mod:`rlinf.algorithms.rlt` so the original RLT code is
unchanged.
"""

from rlinf.algorithms.rlt_lh.history import (
    RLTHistoryBuffer,
    build_history_summary,
    extract_rlt_lh_phase,
    extract_rlt_lh_subtask_success,
    history_summary_dim,
)
from rlinf.algorithms.rlt_lh.rollout import predict_rlt_lh_actions
from rlinf.algorithms.rlt_lh.route import (
    LongHorizonRLTRoute,
    build_rlt_lh_route,
)
from rlinf.algorithms.rlt_lh.transition import (
    RLT_LH_OBS_KEYS,
    RLT_LH_TRANSITION_PREFIX,
    extract_rlt_lh_obs_from_forward_inputs,
    update_rlt_lh_transitions,
    use_lh_simulator_transition_replay,
)

__all__ = [
    "LongHorizonRLTRoute",
    "RLTHistoryBuffer",
    "RLT_LH_OBS_KEYS",
    "RLT_LH_TRANSITION_PREFIX",
    "build_history_summary",
    "build_rlt_lh_route",
    "extract_rlt_lh_phase",
    "extract_rlt_lh_subtask_success",
    "extract_rlt_lh_obs_from_forward_inputs",
    "history_summary_dim",
    "predict_rlt_lh_actions",
    "update_rlt_lh_transitions",
    "use_lh_simulator_transition_replay",
]
