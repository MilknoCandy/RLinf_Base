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

"""ZAP: (Z, A) Path metric-gated n-step for RLT Stage 2."""

from rlinf.algorithms.zap.backup import (
    attach_episode_zap_paths,
    compute_zap_path_target,
    neighbor_weights,
    path_nstep_return,
    select_zap_include,
    stack_path_metrics,
)
from rlinf.algorithms.zap.critic import ZAP_LOSS_TYPE, is_zap_loss
from rlinf.algorithms.zap.kernels import (
    continuation_factor,
    cosine_kernel_z,
    rbf_kernel_a,
)

__all__ = [
    "ZAP_LOSS_TYPE",
    "attach_episode_zap_paths",
    "compute_zap_path_target",
    "continuation_factor",
    "cosine_kernel_z",
    "is_zap_loss",
    "neighbor_weights",
    "path_nstep_return",
    "rbf_kernel_a",
    "select_zap_include",
    "stack_path_metrics",
]
