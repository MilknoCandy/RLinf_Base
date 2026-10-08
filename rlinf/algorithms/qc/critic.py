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

"""Chunked-critic loss names for RLT Stage 2 comparison methods.

These are online-only adaptations of Q-chunking (QC), Decoupled Q-Chunking
(DQC), and Adaptive Q-Chunking (AQC). The frozen Stage-1 VLA is the behavior
prior; there is no separate offline critic pretrain.
"""

from __future__ import annotations

QC_LOSS_TYPE = "rlt_qc"
DQC_LOSS_TYPE = "rlt_dqc"
AQC_LOSS_TYPE = "rlt_aqc"
ZAP_LOSS_TYPE = "rlt_zap"
QC_REQUIRED_N_STEP = 1
RLT_STAGE2_LOSS_TYPES = frozenset(
    {
        "rlt_ac",
        "rlt_td3",
        QC_LOSS_TYPE,
        DQC_LOSS_TYPE,
        AQC_LOSS_TYPE,
        ZAP_LOSS_TYPE,
        "bee",
    }
)


def validate_qc_n_step(n_step: int) -> None:
    """Require a one-chunk (unbiased) Q-chunking backup."""
    if int(n_step) != QC_REQUIRED_N_STEP:
        raise ValueError(
            "Q-chunking critic requires algorithm.n_step="
            f"{QC_REQUIRED_N_STEP} so the backup length matches the critic "
            f"action chunk. Got n_step={n_step}."
        )


def is_rlt_stage2_loss(loss_type: str) -> bool:
    """Return True for RLT Stage 2 losses that reuse the RLT env/rollout path."""
    return str(loss_type) in RLT_STAGE2_LOSS_TYPES
