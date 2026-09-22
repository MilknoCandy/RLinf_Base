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

from rlinf.algorithms.rlt.b2_feedback import (
    append_feedback_to_prompts,
    format_peg_insertion_feedback,
    peg_insertion_distance,
)
from rlinf.algorithms.rlt.b2_loop import B2LoopState
from rlinf.models.embodiment.modules.rlt_b2_select import (
    select_topk_image_tokens,
    text_to_image_attention_scores,
)
from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenTransformer


def test_format_feedback_includes_success_and_distance_change():
    success = torch.tensor([True, False, False])
    delta = torch.tensor([-0.02, 0.03, 0.0])
    sentences = format_peg_insertion_feedback(success=success, delta_distance=delta)
    assert "succeeded" in sentences[0]
    assert "closer" in sentences[0]
    assert "has not succeeded" in sentences[1]
    assert "farther" in sentences[1]
    assert "unchanged" in sentences[2]


def test_append_feedback_keeps_instruction_separate():
    prompts = ["insert the peg in the hole", "insert the peg in the hole"]
    feedback = ["Insertion succeeded. The peg moved closer to the hole.", ""]
    merged = append_feedback_to_prompts(prompts, feedback)
    assert merged[0].startswith("insert the peg in the hole ")
    assert "succeeded" in merged[0]
    assert merged[1] == "insert the peg in the hole"


def test_peg_insertion_distance_decreases_when_closer_to_hole():
    farther = peg_insertion_distance(
        torch.tensor([0.0]), torch.tensor([0.02]), torch.tensor([0.02])
    )
    closer = peg_insertion_distance(
        torch.tensor([0.04]), torch.tensor([0.005]), torch.tensor([0.005])
    )
    assert closer < farther


def test_topk_keeps_fixed_ratio_of_highest_text_attention():
    prefix = torch.arange(8, dtype=torch.float32).view(1, 8, 1).expand(1, 8, 4).clone()
    mask = torch.tensor([[True, True, True, True, False, True, True, True]])
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, 5.0, 0.05, 0.4, 0.3]])
    selected, selected_mask = select_topk_image_tokens(
        prefix,
        mask,
        attn_probs=scores,
        num_image_tokens=8,
        keep_ratio=0.5,
        lang_len=0,
    )
    assert selected.shape[1] == 4
    assert bool(selected_mask.all())
    kept = selected[0, :, 0].tolist()
    assert kept[0] == 1.0
    assert kept[1] == 3.0
    assert 4.0 not in kept


def test_text_to_image_scores_pool_language_queries():
    # [B, H, T, S] with 4 image keys and 2 language queries.
    attn = torch.zeros(1, 2, 6, 6)
    attn[:, :, 4, 1] = 1.0
    attn[:, :, 5, 3] = 1.0
    scores = text_to_image_attention_scores(
        attn, num_image_tokens=4, lang_len=2
    )
    assert scores.shape == (1, 4)
    assert torch.argmax(scores, dim=-1).item() in {1, 3}


def test_encoder_accepts_previous_z_as_rl_token():
    torch.manual_seed(0)
    model = RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=4,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )
    model.eval()
    prefix = torch.randn(2, 4, 8)
    default_z = model.encode_flat(prefix)
    injected = torch.randn(2, 1, 8)
    looped_z = model.encode_flat(prefix, rl_token=injected)
    assert default_z.shape == (2, 8)
    assert looped_z.shape == (2, 8)
    assert not torch.allclose(default_z, looped_z)


def test_b2_loop_resets_token_on_done_and_skips_first_step_feedback():
    loop = B2LoopState()
    init = torch.ones(1, 8)
    batch = 2
    device = torch.device("cpu")
    token0 = loop.build_rl_token(
        init_token=init,
        batch_size=batch,
        dones=None,
        device=device,
        dtype=torch.float32,
    )
    assert torch.allclose(token0, torch.ones(2, 1, 8))

    infos0 = {
        "success_current": torch.zeros(2, dtype=torch.bool),
        "peg_head_hole_x": torch.zeros(2),
        "peg_head_hole_abs_y": torch.ones(2) * 0.05,
        "peg_head_hole_abs_z": torch.ones(2) * 0.05,
    }
    feedback0 = loop.feedback_sentences(
        batch_size=batch, dones=None, env_infos=infos0, device=device
    )
    assert feedback0 == ["", ""]
    loop.commit_z(torch.arange(16, dtype=torch.float32).view(2, 8))

    infos1 = {
        "success_current": torch.tensor([False, True]),
        "peg_head_hole_x": torch.tensor([0.02, 0.05]),
        "peg_head_hole_abs_y": torch.tensor([0.01, 0.001]),
        "peg_head_hole_abs_z": torch.tensor([0.01, 0.001]),
    }
    feedback1 = loop.feedback_sentences(
        batch_size=batch, dones=None, env_infos=infos1, device=device
    )
    assert "closer" in feedback1[0]
    assert "succeeded" in feedback1[1]

    dones = torch.tensor([True, False])
    token2 = loop.build_rl_token(
        init_token=init,
        batch_size=batch,
        dones=dones,
        device=device,
        dtype=torch.float32,
    )
    assert torch.allclose(token2[0], torch.ones(1, 8))
    assert not torch.allclose(token2[1], torch.ones(1, 8))
    feedback2 = loop.feedback_sentences(
        batch_size=batch, dones=dones, env_infos=infos1, device=device
    )
    assert feedback2[0] == ""
    assert feedback2[1] != ""


def test_readout_heads_preserve_batch_time_shape():
    from rlinf.models.embodiment.modules.rlt_b2_heads import B2ReadoutHeads

    heads = B2ReadoutHeads(z_dim=8, hidden_dim=8)
    z = torch.randn(2, 5, 8)
    distance, logit = heads(z)
    assert distance.shape == (2, 5)
    assert logit.shape == (2, 5)


def test_dump_writer_flushes_previous_episode_on_auto_reset(tmp_path):
    from rlinf.algorithms.rlt.b2_dump import B2DumpWriter

    writer = B2DumpWriter(tmp_path, rank=0, auto_reset=True, min_steps=2)
    tokens = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    writer.append(
        image_tokens=tokens,
        image_mask=mask,
        distance=torch.tensor([0.2]),
        success=torch.tensor([False]),
        dones=torch.tensor([False]),
    )
    writer.append(
        image_tokens=tokens + 1,
        image_mask=mask,
        distance=torch.tensor([0.1]),
        success=torch.tensor([False]),
        dones=torch.tensor([False]),
    )
    writer.append(
        image_tokens=tokens + 2,
        image_mask=mask,
        distance=torch.tensor([0.4]),
        success=torch.tensor([False]),
        dones=torch.tensor([True]),
    )
    written = list(tmp_path.glob("*.pt"))
    assert len(written) == 1
    episode = torch.load(written[0], map_location="cpu", weights_only=False)
    assert episode["image_tokens"].shape[0] == 2
    assert episode["distance"][0] == 0.2
    writer.close()
    written = list(tmp_path.glob("*.pt"))
    assert len(written) == 1


def test_sft_loss_counts_only_t_ge_1_and_truncates_bptt():
    from rlinf.algorithms.rlt.b2_sft import B2SFTModel
    from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenEncoder

    torch.manual_seed(0)
    encoder = RLTTokenEncoder(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=4,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )
    model = B2SFTModel(encoder, hidden_dim=8)
    image_tokens = torch.randn(2, 5, 4, 8)
    image_mask = torch.ones(2, 5, 4, dtype=torch.bool)
    distance = torch.randn(2, 5)
    success = torch.zeros(2, 5)
    success[:, -1] = 1.0
    valid = torch.ones(2, 5, dtype=torch.bool)

    loss_full, metrics = model.compute_loss(
        image_tokens=image_tokens,
        image_mask=image_mask,
        distance=distance,
        success=success,
        valid=valid,
        bptt_k=0,
    )
    assert int(metrics["num_steps"].item()) == 8
    loss_full.backward()
    assert encoder.rl_token_embed.grad is not None
    assert encoder.rl_token_embed.grad.abs().sum() > 0

    encoder.zero_grad()
    model.zero_grad()
    loss_k1, _ = model.compute_loss(
        image_tokens=image_tokens,
        image_mask=image_mask,
        distance=distance,
        success=success,
        valid=valid,
        bptt_k=1,
    )
    loss_k1.backward()
    grad = encoder.rl_token_embed.grad
    assert grad is None or float(grad.abs().sum()) == 0.0


def test_collate_pads_variable_length_episodes():
    from rlinf.algorithms.rlt.b2_sft import collate_b2_episodes

    short = {
        "image_tokens": torch.randn(2, 3, 4),
        "image_mask": torch.ones(2, 3, dtype=torch.bool),
        "distance": torch.tensor([0.2, 0.1]),
        "success": torch.tensor([False, False]),
    }
    long = {
        "image_tokens": torch.randn(4, 5, 4),
        "image_mask": torch.ones(4, 5, dtype=torch.bool),
        "distance": torch.tensor([0.4, 0.3, 0.2, 0.1]),
        "success": torch.tensor([False, False, False, True]),
    }
    batch = collate_b2_episodes([short, long])
    assert batch["image_tokens"].shape == (2, 4, 5, 4)
    assert bool(batch["valid"][0, :2].all())
    assert not bool(batch["valid"][0, 2:].any())
    assert bool(batch["valid"][1].all())


def test_load_encoder_weights_from_stage1_key_layout(tmp_path):
    from rlinf.algorithms.rlt.b2_sft import load_encoder_weights
    from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenEncoder

    src = RLTTokenEncoder(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=4,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )
    wrapped = {
        f"rlt_module.encoder.{key}": value for key, value in src.state_dict().items()
    }
    ckpt = tmp_path / "full_weights.pt"
    torch.save(wrapped, ckpt)
    dst = RLTTokenEncoder(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=4,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )
    load_encoder_weights(dst, ckpt)
    for left, right in zip(src.parameters(), dst.parameters()):
        assert torch.allclose(left, right)
