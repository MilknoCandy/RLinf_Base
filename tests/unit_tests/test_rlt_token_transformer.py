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

from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)


def _make_model(*, prefix_seq_len: int = 5) -> RLTTokenTransformer:
    torch.manual_seed(0)
    return RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=prefix_seq_len,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )


def test_stage1_encoder_depth_matches_rlt_layers():
    model = RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=5,
        num_layers=2,
        num_heads=2,
        dropout_rate=0.0,
    )
    attentions = [
        module
        for module in model.encoder.modules()
        if module.__class__.__name__ == "MultiheadAttention"
    ]
    assert len(attentions) == 2
    assert len(model.encoder.layers) == 2
    prefix = torch.randn(2, model.prefix_seq_len, model.input_dim)
    z = model.encode(prefix)
    assert z.shape == (2, 1, model.embed_dim)


def test_decoder_causal_mask_blocks_future_teacher_targets():
    model = _make_model()
    model.eval()
    rl_tokens = torch.randn(1, 1, model.embed_dim)
    targets = torch.randn(1, model.prefix_seq_len, model.input_dim)

    changed_targets = targets.clone()
    changed_targets[:, 2:] += 100.0

    original_output = model.decode(rl_tokens, targets)
    changed_output = model.decode(rl_tokens, changed_targets)

    # target[2:] enters decoder positions 3+, so positions 0..2 must not
    # change when causal attention prevents access to future positions.
    torch.testing.assert_close(
        original_output[:, :3],
        changed_output[:, :3],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(original_output[:, 3:], changed_output[:, 3:])


def test_window_write_keeps_earlier_chunks():
    model = _make_model(prefix_seq_len=4)
    model.eval()
    hist = torch.randn(1, 2, 4, model.input_dim)
    valid = torch.ones(1, 2, dtype=torch.bool)
    z_full, _ = model.encode_window(hist, None, valid)

    changed = hist.clone()
    changed[:, 0] = changed[:, 0] + 4
    z_changed, _ = model.encode_window(changed, None, valid)
    assert not torch.allclose(z_full, z_changed)

    last_only = torch.tensor([[False, True]])
    z_last, _ = model.encode_window(hist, None, last_only)
    torch.testing.assert_close(z_last, model.encode(hist[:, 1]))


def test_sequence_loss_reconstructs_only_the_last_chunk():
    model = _make_model(prefix_seq_len=4)
    model.train()
    hist = torch.randn(2, 3, 4, model.input_dim)
    valid = torch.ones(2, 3, dtype=torch.bool)
    calls = []
    original = model.decode

    def _count(rl_tokens, target_embeddings, mask=None):
        calls.append(tuple(target_embeddings.shape))
        return original(rl_tokens, target_embeddings, mask)

    model.decode = _count
    loss, _ = model.sequence_loss(hist, None, valid, mem_len_min=3, mem_len_max=3)
    loss.backward()
    assert calls == [(2, 4, model.input_dim)]
    assert model.encoder.z_init.grad is not None


def test_step_memory_resets_done_rows():
    model = _make_model(prefix_seq_len=4)
    model.eval()
    prefix = torch.randn(2, 4, model.input_dim)
    first = model.step_memory(prefix, None, None, None)
    second = model.step_memory(prefix, None, first, torch.tensor([False, True]))
    restarted = model.step_memory(prefix[1:], None, None, None)
    torch.testing.assert_close(second[1], restarted[0])
    assert not torch.allclose(second[0], first[0])


def test_sequence_loss_trains_across_the_window():
    model = _make_model(prefix_seq_len=4)
    model.train()
    hist = torch.randn(2, 3, 4, model.input_dim)
    valid = torch.ones(2, 3, dtype=torch.bool)
    loss, metrics = model.sequence_loss(
        hist, None, valid, mem_len_min=3, mem_len_max=3
    )
    loss.backward()
    assert model.encoder.z_init.grad is not None
    assert float(model.encoder.z_init.grad.abs().sum()) > 0
    assert float(metrics["mem_loop_len"]) == 3


def test_sequence_loss_samples_suffix_length():
    model = _make_model(prefix_seq_len=4)
    model.train()
    torch.manual_seed(0)
    valid = torch.ones(32, 10, dtype=torch.bool)
    counts = []
    for _ in range(30):
        cropped = model._crop_window_valid(valid, 1, 10)
        counts.append(cropped.sum(dim=-1))
    counts = torch.cat(counts)
    assert int(counts.min()) == 1
    assert int(counts.max()) == 10


def test_crop_draws_zero_length_and_skips_that_row():
    model = _make_model(prefix_seq_len=4)
    model.train()
    torch.manual_seed(0)
    valid = torch.ones(64, 10, dtype=torch.bool)
    counts = []
    for _ in range(40):
        cropped = model._crop_window_valid(valid, 0, 10)
        counts.append(cropped.sum(dim=-1))
    counts = torch.cat(counts)
    assert int(counts.min()) == 0
    assert int(counts.max()) == 10

    hist = torch.randn(2, 3, 4, model.input_dim)
    row_valid = torch.tensor(
        [[False, False, False], [True, True, True]], dtype=torch.bool
    )
    calls = []
    original = model.decode

    def _count(rl_tokens, target_embeddings, mask=None):
        calls.append(tuple(target_embeddings.shape))
        return original(rl_tokens, target_embeddings, mask)

    model.decode = _count
    loss, metrics = model.sequence_loss(hist, None, row_valid)
    loss.backward()
    assert calls == [(1, 4, model.input_dim)]
    assert float(metrics["mem_loop_len"]) == 1.5
    assert model.encoder.z_init.grad is not None


def test_loss_masks_trailing_padding():
    model = _make_model(prefix_seq_len=4)
    model.eval()
    prefix_embs = torch.randn(2, 4, model.input_dim)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    loss, _ = model.loss(prefix_embs, mask)
    reconstructed, _ = model.reconstruct(prefix_embs, mask)
    valid = mask.unsqueeze(-1).to(dtype=torch.float32)
    expected_loss = (
        torch.square(reconstructed.float() - prefix_embs.float()) * valid
    ).sum() / (valid.sum() * model.input_dim)
    torch.testing.assert_close(loss, expected_loss)

    changed_padding = prefix_embs.clone()
    changed_padding[~mask] += 1000.0
    changed_loss, _ = model.loss(changed_padding, mask)
    torch.testing.assert_close(loss, changed_loss, rtol=1e-5, atol=1e-5)


def test_reconstruct_output_shape_matches_prefix_embeddings():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(3, 4, model.input_dim)

    reconstructed, _ = model.reconstruct(prefix_embs)

    assert reconstructed.shape == prefix_embs.shape


def test_reconstruct_detaches_targets_but_trains_encoder_and_decoder():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(2, 4, model.input_dim, requires_grad=True)

    loss, _ = model.loss(prefix_embs)
    loss.backward()

    assert prefix_embs.grad is None
    encoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    decoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad_norm > 0
    assert decoder_grad_norm > 0
