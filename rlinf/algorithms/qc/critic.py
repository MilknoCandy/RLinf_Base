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

"""Q-chunking critic constraints for RLT Stage 2.

The unbiased Q-chunking backup matches one action chunk: discounted rewards
inside the chunk plus ``gamma ** chunk_len * Q(next, a_next)``. Looking ahead
more chunks while the critic still consumes one chunk is the biased n-step
ablation from the paper, not Q-chunking.
"""

from __future__ import annotations

QC_LOSS_TYPE = "rlt_qc"
QC_REQUIRED_N_STEP = 1
RLT_STAGE2_LOSS_TYPES = frozenset({"rlt_ac", "rlt_td3", QC_LOSS_TYPE})


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
