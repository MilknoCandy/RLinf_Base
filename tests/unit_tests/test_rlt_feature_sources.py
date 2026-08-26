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

import pytest
import torch

from rlinf.models.embodiment.openpi_rlt_probe.feature_sources import (
    RLTFeatureBundle,
    mean_pool_features,
    select_features,
)


def _bundle():
    batch, prefix_len, lang_len = 2, 7, 3
    return RLTFeatureBundle(
        prefix_output=torch.randn(batch, prefix_len, 2048),
        prefix_mask=torch.ones(batch, prefix_len, dtype=torch.bool),
        prompt_tokens=torch.ones(batch, lang_len, dtype=torch.long),
        action_output=torch.randn(batch, 4, 1024),
        action_mask=torch.ones(batch, 4, dtype=torch.bool),
    )


def test_all_returns_full_prefix():
    bundle = _bundle()
    features, mask = select_features("all", bundle)
    assert features.shape == bundle.prefix_output.shape
    assert mask.shape == bundle.prefix_mask.shape


def test_image_slices_image_tokens():
    bundle = _bundle()
    features, mask = select_features("image", bundle)
    assert features.shape == (2, 4, 2048)  # prefix_len - lang_len


def test_language_slices_prompt_tokens():
    bundle = _bundle()
    features, mask = select_features("language", bundle)
    assert features.shape == (2, 3, 2048)
    assert torch.equal(features, bundle.prefix_output[:, 4:7])


def test_action_uses_suffix_features():
    bundle = _bundle()
    features, mask = select_features("action", bundle)
    assert features.shape == bundle.action_output.shape


def test_unknown_source_raises():
    bundle = _bundle()
    with pytest.raises(KeyError):
        select_features("nope", bundle)


def test_mean_pool_features_without_mask():
    features = torch.randn(2, 4, 2048)
    pooled = mean_pool_features(features, None)
    assert pooled.shape == (2, 2048)
    assert torch.allclose(pooled, features.mean(dim=1))


def test_mean_pool_features_with_mask():
    features = torch.randn(2, 4, 2048)
    mask = torch.tensor(
        [[True, True, False, False], [True, False, False, False]],
        dtype=torch.bool,
    )
    pooled = mean_pool_features(features, mask)
    expected = torch.stack(
        [features[0, :2].mean(dim=0), features[1, :1].mean(dim=0)]
    )
    assert torch.allclose(pooled, expected)
