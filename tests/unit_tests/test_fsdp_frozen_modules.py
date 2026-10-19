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

"""Tests for FSDP helpers that exclude fully frozen submodules."""

import torch.nn as nn

from rlinf.hybrid_engines.fsdp.utils import collect_fully_frozen_modules


class _ToyPolicy(nn.Module):
    def __init__(self, freeze_mem: bool = True):
        super().__init__()
        self.actor = nn.Linear(8, 4)
        self.q_head = nn.Linear(8, 1)
        self.mem = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
        if freeze_mem:
            self.mem.requires_grad_(False)


def test_collect_fully_frozen_modules_finds_mem():
    model = _ToyPolicy(freeze_mem=True)
    ignored = collect_fully_frozen_modules(model)
    assert ignored == [model.mem]
    assert all(not p.requires_grad for p in ignored[0].parameters())
    managed = [
        p for name, p in model.named_parameters() if not name.startswith("mem.")
    ]
    assert managed and all(p.requires_grad for p in managed)


def test_collect_fully_frozen_modules_empty_when_trainable():
    model = _ToyPolicy(freeze_mem=False)
    assert collect_fully_frozen_modules(model) == []


def test_collect_skips_paramless_children():
    class EmptyChild(nn.Module):
        def forward(self, x):
            return x

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.empty = EmptyChild()
            self.head = nn.Linear(2, 2)

    assert collect_fully_frozen_modules(Root()) == []
