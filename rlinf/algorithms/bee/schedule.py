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

"""Correction-model update schedule from Bee Appendix C / Algorithm 1."""

from __future__ import annotations


def should_update_correction(
    new_corrections_since_update: int,
    *,
    update_interval_m: int = 100,
) -> bool:
    """True once at least ``M`` new intervention samples have arrived."""
    m = int(update_interval_m)
    if m <= 0:
        return False
    return int(new_corrections_since_update) >= m


def correction_update_steps(
    *,
    update_interval_m: int = 100,
    update_to_data_n: float = 0.12,
) -> int:
    """Number of Correction Model gradient steps: ``NM`` (Table III)."""
    steps = int(round(float(update_interval_m) * float(update_to_data_n)))
    return max(steps, 0)
