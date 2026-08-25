# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Factory for the official-OpenPI-backed RLT Idea3 variant."""

from __future__ import annotations

from typing import Any

from rlinf.models.embodiment.rlt_idea1_openpi import _build_openpi_idea_model
from rlinf.models.embodiment.rlt_idea3_openpi.rlt_idea3_openpi_action_model import (
    OpenPiIdea3ActionModel,
)


def get_model(cfg: Any, torch_dtype=None):
    return _build_openpi_idea_model(cfg, torch_dtype, OpenPiIdea3ActionModel)
