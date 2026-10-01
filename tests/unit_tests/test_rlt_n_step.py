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

import torch

from rlinf.algorithms.rlt.n_step import (
    compute_n_step_chunk_target,
    discounted_chunk_reward,
)


def test_discounted_chunk_reward_matches_manual_sum():
    rewards = torch.tensor([1.0, 0.0, 2.0])
    gamma = 0.5
    got = discounted_chunk_reward(rewards, gamma)
    expected = 1.0 + 0.5 * 0.0 + 0.25 * 2.0
    assert abs(float(got) - expected) < 1e-6


def test_one_step_matches_single_chunk_and_bootstraps():
    rewards = [torch.tensor([1.0, 0.0])]
    dones = [False]
    n_step_return, discount, offset, bootstrapped = compute_n_step_chunk_target(
        rewards, dones, gamma=0.9, n_step=1
    )
    assert bootstrapped
    assert offset == 0
    assert abs(float(discount) - 0.9**2) < 1e-6
    assert abs(float(n_step_return) - 1.0) < 1e-6


def test_two_step_folds_second_chunk_and_discounts_bootstrap():
    rewards = [
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([3.0]),
    ]
    dones = [False, False, False]
    n_step_return, discount, offset, bootstrapped = compute_n_step_chunk_target(
        rewards, dones, gamma=0.5, n_step=2
    )
    assert bootstrapped
    assert offset == 1
    assert abs(float(n_step_return) - (1.0 + 0.5 * 2.0)) < 1e-6
    assert abs(float(discount) - (0.5 * 0.5)) < 1e-6


def test_n_step_stops_at_terminal_without_bootstrap():
    rewards = [
        torch.tensor([1.0]),
        torch.tensor([4.0]),
        torch.tensor([8.0]),
    ]
    dones = [False, True, False]
    n_step_return, discount, offset, bootstrapped = compute_n_step_chunk_target(
        rewards, dones, gamma=0.5, n_step=4
    )
    assert not bootstrapped
    assert offset == 1
    assert abs(float(discount) - 0.0) < 1e-6
    assert abs(float(n_step_return) - (1.0 + 0.5 * 4.0)) < 1e-6
