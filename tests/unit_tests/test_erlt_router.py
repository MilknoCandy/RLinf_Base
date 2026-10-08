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

from rlinf.models.embodiment.modules.erlt import (
    ERLTLatentActor,
    ERLTRouter,
    expert_action_target,
    is_erlt_routing_update,
)


def test_routing_update_lands_on_every_200th_step():
    assert is_erlt_routing_update(0, 200) is False
    assert is_erlt_routing_update(198, 200) is False
    assert is_erlt_routing_update(199, 200) is True
    assert is_erlt_routing_update(399, 200) is True


def test_layer_logits_outside_the_active_set_get_no_gradient():
    router = ERLTRouter(
        hidden_dim=4,
        num_tokens=1,
        layer_indices=(0, 3),
        logit_slots=4,
        temperature=1.0,
    )
    summaries = torch.randn(2, 2, 4)
    router.combine(summaries).sum().backward()
    assert router.layer_logits.grad[:2].abs().sum() > 0
    assert router.layer_logits.grad[2:].abs().sum() == 0


def test_expert_action_target_keeps_ten_by_seven():
    actions = torch.arange(2 * 10 * 32, dtype=torch.float32).reshape(2, 10, 32)
    target = expert_action_target(actions, action_chunk=10, action_dim=7)
    assert target.shape == (2, 70)
    assert torch.equal(target[0], actions[0, :, :7].reshape(-1))


def test_latent_actor_repeats_one_noise_vector_and_clamps_std():
    actor = ERLTLatentActor(
        input_dim=8,
        output_dim=4,
        hidden_dims=(8,),
        action_horizon=10,
    )
    actor.log_std_layer.bias.data.fill_(100.0)
    features = torch.zeros(2, 8)
    noise, log_prob = actor.sample(features, deterministic=False)
    assert noise.shape == (2, 10, 4)
    assert torch.equal(noise[:, 0, :], noise[:, -1, :])
    assert log_prob.shape == (2,)
    scale = actor._distribution(features).base_dist.scale
    assert torch.all(scale <= torch.exp(torch.tensor(2.0)) + 1e-5)
    assert torch.all(scale >= torch.exp(torch.tensor(-20.0)) - 1e-5)
