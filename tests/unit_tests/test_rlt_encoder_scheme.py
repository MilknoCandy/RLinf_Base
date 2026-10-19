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

from types import SimpleNamespace

from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    bind_rlt_mem_scheme,
    infer_rlt_encoder_kind,
    resolve_rlt_mem_scheme,
)


def test_infer_old_maniskill_stage1_keys():
    state_dict = {
        "rlt_module.encoder.rl_token_embed": object(),
        "rlt_module.encoder.rl_token_pos_enc": object(),
        "rlt_module.encoder.layers.0.self_attn.in_proj_weight": object(),
    }
    assert infer_rlt_encoder_kind(state_dict) == "token"
    assert resolve_rlt_mem_scheme(mem_scheme=1, state_dict=state_dict) == 0


def test_infer_cross_encoder_keys_keep_yaml_scheme():
    state_dict = {
        "rlt_module.encoder.z_init": object(),
        "rlt_module.encoder.out_norm.weight": object(),
    }
    assert infer_rlt_encoder_kind(state_dict) == "cross"
    assert (
        resolve_rlt_mem_scheme(
            encoder_type="append_self_attention",
            mem_scheme=2,
            state_dict=state_dict,
        )
        == 2
    )
    assert (
        resolve_rlt_mem_scheme(
            encoder_type="append_self_attention",
            mem_scheme=1,
            state_dict=state_dict,
        )
        == 1
    )


def test_yaml_append_self_attention_selects_old_encoder_without_ckpt():
    assert (
        resolve_rlt_mem_scheme(
            encoder_type="append_self_attention",
            mem_scheme=1,
        )
        == 0
    )
    assert resolve_rlt_mem_scheme(mem_scheme=0) == 0


def test_return_prefix_keeps_cross_encoder_even_with_legacy_yaml_type():
    assert (
        resolve_rlt_mem_scheme(
            encoder_type="append_self_attention",
            mem_scheme=1,
            return_prefix=True,
        )
        == 1
    )
    assert (
        resolve_rlt_mem_scheme(
            encoder_type="append_self_attention",
            mem_scheme=1,
            mem_len_max=16,
        )
        == 1
    )


def test_bind_writes_scheme_zero_from_old_keys():
    cfg = SimpleNamespace(
        rlt_encoder_type="append_self_attention",
        rlt_mem_scheme=1,
        rlt_mem_len_max=1,
        rlt_return_prefix=False,
    )
    scheme = bind_rlt_mem_scheme(
        cfg,
        {"_fsdp_wrapped_module.rlt_module.encoder.rl_token_embed": object()},
    )
    assert scheme == 0
    assert cfg.rlt_mem_scheme == 0
    assert cfg.rlt_encoder_type == "append_self_attention"
