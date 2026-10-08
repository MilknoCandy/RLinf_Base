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

"""Bee: intervention-adaptive residual RL over a frozen VLA (arXiv 2609.27450)."""

from rlinf.algorithms.bee.losses import (
    bee_actor_loss,
    bee_multiplier_loss,
    correction_nll_loss,
    mahalanobis_rho,
)
from rlinf.algorithms.bee.modules import BeeCorrectionModel, BeeMultiplier
from rlinf.algorithms.bee.schedule import (
    correction_update_steps,
    should_update_correction,
)

__all__ = [
    "BeeCorrectionModel",
    "BeeMultiplier",
    "bee_actor_loss",
    "bee_multiplier_loss",
    "correction_nll_loss",
    "correction_update_steps",
    "mahalanobis_rho",
    "should_update_correction",
]
