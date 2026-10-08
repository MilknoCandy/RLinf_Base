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

from rlinf.algorithms.qc.critic import ZAP_LOSS_TYPE, is_rlt_stage2_loss
from rlinf.algorithms.zap.backup import (
    accumulate_product_and_truncate,
    attach_episode_zap_paths,
    compute_zap_path_target,
    compute_zap_path_targets_batch,
    neighbor_weights,
    neighbor_weights_batch,
    path_nstep_return,
    select_zap_include,
    select_zap_include_batch,
)
from rlinf.algorithms.zap.kernels import continuation_factor, cosine_kernel_z, rbf_kernel_a


def test_zap_is_stage2_loss():
    assert ZAP_LOSS_TYPE == "rlt_zap"
    assert is_rlt_stage2_loss("rlt_zap")


def test_kernels_peak_at_identity():
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k_same = cosine_kernel_z(z[:1], z[:1], temperature=0.1)
    k_diff = cosine_kernel_z(z[:1], z[1:], temperature=0.1)
    assert float(k_same) > float(k_diff)
    a = torch.zeros(1, 3)
    b = torch.ones(1, 3)
    assert float(rbf_kernel_a(a, a, temperature=0.05)) > float(
        rbf_kernel_a(a, b, temperature=0.05)
    )


def test_continuation_cuts_when_action_leaves_pi():
    z = torch.ones(1, 4)
    a_pi = torch.zeros(1, 3)
    a_near = torch.zeros(1, 3)
    a_far = torch.ones(1, 3) * 5
    c_near = continuation_factor(z, z, a_pi, a_near, tau_z=0.1, tau_a=0.05)
    c_far = continuation_factor(z, z, a_pi, a_far, tau_z=0.1, tau_a=0.05)
    assert float(c_near) > 0.9
    assert float(c_far) < 0.01


def test_product_truncate_stops_before_low_c():
    c = torch.tensor([0.9, 0.9, 1e-4, 0.9])
    tau_extra, products = accumulate_product_and_truncate(c, truncate_eps=0.01)
    assert tau_extra == 2
    assert float(products[1]) > 0.01
    assert float(products[2]) < 0.01


def test_path_nstep_return_with_horizons():
    rewards = torch.tensor([1.0, 1.0])
    horizons = torch.tensor([2.0, 2.0])
    total, discount = path_nstep_return(
        rewards, gamma=0.5, horizons=horizons, num_steps=2
    )
    # 1 + (0.5**2) * 1 = 1.25; bootstrap discount (0.5**2)**2 = 0.0625
    assert abs(float(total) - 1.25) < 1e-5
    assert abs(discount - 0.0625) < 1e-5


def test_select_include_stays_one_when_c_collapses():
    z = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])])
    a_buf = torch.zeros(2, 2)
    a_pi = torch.zeros(2, 2)
    dones = torch.tensor([False, False])
    include, _ = select_zap_include(
        path_z=z,
        path_a_buf=a_buf,
        path_a_pi=a_pi,
        path_dones=dones,
        tau_z=0.05,
        tau_a=0.05,
        truncate_eps=0.01,
    )
    assert include == 1


def test_select_include_extends_when_cell_stable():
    z = torch.ones(4, 3)
    a_buf = torch.zeros(4, 2)
    a_pi = torch.zeros(4, 2)
    dones = torch.zeros(4, dtype=torch.bool)
    include, mean_c = select_zap_include(
        path_z=z,
        path_a_buf=a_buf,
        path_a_pi=a_pi,
        path_dones=dones,
        tau_z=0.1,
        tau_a=0.05,
        truncate_eps=0.01,
    )
    assert include == 4
    assert mean_c > 0.9


def test_neighbor_weights_fallback_when_z_uniform():
    z_q = torch.tensor([1.0, 0.0])
    a_q = torch.zeros(2)
    # Equally dissimilar neighbors under a large tau_z -> near-uniform z kernel.
    neighbor_z = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [0.7071, 0.7071],
        ]
    )
    neighbor_a = torch.zeros(4, 2)
    weights, metrics = neighbor_weights(
        z_q,
        a_q,
        neighbor_z,
        neighbor_a,
        tau_z=10.0,
        tau_a=0.05,
        top_k=None,
        uniform_entropy_ratio=0.5,
    )
    assert metrics["zap/neighbor_fallback"] == 1.0
    assert float(weights.sum()) == 0.0


def test_batch_neighbor_weights_match_scalar():
    z = torch.tensor(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.2, 0.8],
        ]
    )
    a = torch.tensor(
        [
            [0.0, 0.0],
            [0.01, 0.0],
            [1.0, 1.0],
            [0.8, 0.9],
        ]
    )
    exclude = torch.eye(4, dtype=torch.bool)
    w_batch, _ = neighbor_weights_batch(
        z,
        a,
        z,
        a,
        tau_z=0.1,
        tau_a=0.05,
        top_k=3,
        uniform_entropy_ratio=0.95,
        exclude_mask=exclude,
    )
    for i in range(4):
        mask = torch.ones(4, dtype=torch.bool)
        mask[i] = False
        w_i, _ = neighbor_weights(
            z[i],
            a[i],
            z[mask],
            a[mask],
            tau_z=0.1,
            tau_a=0.05,
            top_k=3,
        )
        assert torch.allclose(w_batch[i][mask], w_i, atol=1e-5)
        assert float(w_batch[i][i]) == 0.0 or float(w_batch[i].sum()) == 0.0


def test_zap_target_uses_local_bootstrap_without_neighbors():
    z = torch.ones(3, 2)
    a = torch.zeros(3, 2)
    rewards = torch.tensor([1.0, 0.0, 0.0])
    dones = torch.tensor([False, False, False])
    horizons = torch.ones(3)
    y, metrics = compute_zap_path_target(
        path_z=z,
        path_a_buf=a,
        path_a_pi=a,
        path_rewards=rewards,
        path_dones=dones,
        path_horizons=horizons,
        gamma=0.9,
        tau_z=0.1,
        tau_a=0.05,
        truncate_eps=0.01,
        local_bootstrap_q=torch.tensor(2.0),
        neighbor_z=None,
    )
    assert metrics["zap/tau"] == 3.0
    assert metrics["zap/bootstrap"] == 1.0
    # 1 + 0 + 0 + (0.9**3) * 2
    assert abs(float(y) - (1.0 + (0.9**3) * 2.0)) < 1e-5


def test_batch_include_matches_scalar():
    z = torch.ones(2, 4, 3)
    z[1] = torch.tensor([1.0, 0.0, 0.0])
    z[1, 2:] = torch.tensor([0.0, 1.0, 0.0])
    a_buf = torch.zeros(2, 4, 2)
    a_pi = torch.zeros(2, 4, 2)
    dones = torch.zeros(2, 4, dtype=torch.bool)
    path_len = torch.tensor([4, 4])
    include, mean_c = select_zap_include_batch(
        path_z=z,
        path_a_buf=a_buf,
        path_a_pi=a_pi,
        path_dones=dones,
        path_len=path_len,
        tau_z=0.1,
        tau_a=0.05,
        truncate_eps=0.01,
    )
    for i in range(2):
        inc_i, mean_i = select_zap_include(
            path_z=z[i],
            path_a_buf=a_buf[i],
            path_a_pi=a_pi[i],
            path_dones=dones[i],
            tau_z=0.1,
            tau_a=0.05,
            truncate_eps=0.01,
        )
        assert int(include[i]) == inc_i
        assert abs(float(mean_c[i]) - mean_i) < 1e-5


def test_batch_targets_match_scalar_without_neighbors():
    z = torch.ones(2, 3, 2)
    a = torch.zeros(2, 3, 2)
    rewards = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    dones = torch.zeros(2, 3, dtype=torch.bool)
    horizons = torch.ones(2, 3)
    local_q = torch.tensor([2.0, 1.0])
    y_batch, _ = compute_zap_path_targets_batch(
        path_z=z,
        path_a_buf=a,
        path_a_pi=a,
        path_rewards=rewards,
        path_dones=dones,
        path_horizons=horizons,
        path_len=torch.tensor([3, 3]),
        gamma=0.9,
        tau_z=0.1,
        tau_a=0.05,
        truncate_eps=0.01,
        local_bootstrap_q=local_q,
        neighbor_z=None,
    )
    for i in range(2):
        y_i, _ = compute_zap_path_target(
            path_z=z[i],
            path_a_buf=a[i],
            path_a_pi=a[i],
            path_rewards=rewards[i],
            path_dones=dones[i],
            path_horizons=horizons[i],
            gamma=0.9,
            tau_z=0.1,
            tau_a=0.05,
            truncate_eps=0.01,
            local_bootstrap_q=local_q[i],
            neighbor_z=None,
        )
        assert abs(float(y_batch[i]) - float(y_i)) < 1e-5


def test_attach_episode_windows_are_causal():
    class _Tr:
        def __init__(self, z, a, r, done):
            self.curr_obs = {"z_rl": z}
            self.actions = a
            self.rewards = r
            self.dones = done

    transitions = [
        _Tr(torch.tensor([float(i), 0.0]), torch.zeros(2), torch.ones(3), torch.tensor(False))
        for i in range(5)
    ]
    transitions[-1].dones = torch.tensor(True)
    attach_episode_zap_paths(transitions, max_atoms=4, gamma=0.99)
    assert int(transitions[0].zap_path_len) == 4
    assert int(transitions[3].zap_path_len) == 2
    assert torch.allclose(
        transitions[0].zap_path_z[0, 0, :3, 0], torch.tensor([0.0, 1.0, 2.0])
    )
    assert bool(transitions[0].zap_path_done[0, 0, 3]) is False
    assert bool(transitions[4].zap_path_done[0, 0, 0]) is True
