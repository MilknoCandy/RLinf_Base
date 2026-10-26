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

import json
import os
from typing import Optional

import torch


def load_task_success_seeds(
    seeds_path: Optional[str], task_name: str
) -> Optional[list[int]]:
    """Return ``success_seeds`` for ``task_name``, or ``None`` if missing.

    Tasks outside ``eval_seeds.json`` must not crash eval: the env falls back
    to random reset seeds so a 50-task sweep can still run.
    """
    if not seeds_path or not os.path.exists(seeds_path):
        return None
    with open(seeds_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return None
    task = data.get(task_name)
    if not isinstance(task, dict):
        return None
    seeds = task.get("success_seeds")
    if not seeds:
        return None
    return [int(seed) for seed in seeds]


def partition_success_seeds(
    success_seeds: torch.Tensor,
    *,
    base_seed: int,
    seed_offset: int,
    total_num_processes: int,
    num_group: int,
) -> torch.Tensor:
    """Shuffle success seeds globally and return the non-overlapping worker slice."""
    global_generator = torch.Generator()
    global_generator.manual_seed(base_seed)
    shuffled_indices = torch.randperm(success_seeds.numel(), generator=global_generator)
    shuffled_seeds = success_seeds[shuffled_indices]

    seeds_per_worker = shuffled_seeds.numel() // total_num_processes
    start = seed_offset * seeds_per_worker
    end = start + seeds_per_worker
    worker_seeds = shuffled_seeds[start:end]

    keep_count = (worker_seeds.numel() // num_group) * num_group
    return worker_seeds[:keep_count]
