# Copyright 2026 The RLinf Authors.

import torch

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def _policy():
    return RLTMLPPolicy(
        z_dim=32,
        proprio_dim=4,
        action_dim=8,
        num_action_chunks=2,
        use_mem=True,
        loop_prefix_len=8,
        loop_num_heads=4,
        loop_num_layers=1,
    )


def _obs(batch: int, prefix_len: int = 8, z_dim: int = 32):
    return {
        "z_rl": torch.randn(batch, z_dim),
        "proprio": torch.randn(batch, 4),
        "ref_chunk": torch.randn(batch, 2, 8),
        "prefix_embs": torch.randn(batch, prefix_len, z_dim),
        "prefix_mask": torch.ones(batch, prefix_len, dtype=torch.bool),
        "z_prev": torch.randn(batch, z_dim),
        "prev_action": torch.randn(batch, 16),
        "prev_reward": torch.randn(batch, 1),
    }


def test_clone_ready_buffer_starts_off():
    policy = _policy()
    assert hasattr(policy, "student_clone_ready")
    assert not policy.is_student_clone_ready()
    policy.set_student_clone_ready(True)
    assert policy.is_student_clone_ready()


def test_actor_state_includes_ref_chunk():
    policy = _policy()
    batch = 3
    obs = _obs(batch)
    policy.train()
    state = policy._actor_state(obs)
    assert state.shape == (batch, 16 + 32 + 4)
    action, _, _ = policy.sac_forward(obs)
    assert action.shape == (batch, 16)
    assert action.shape[-1] == policy._get_ref_chunk(obs).shape[-1]


def test_history_changes_policy_action():
    policy = _policy()
    policy.eval()
    obs = _obs(2)
    other = dict(obs)
    other["z_prev"] = torch.randn_like(obs["z_prev"])
    action_a, _, _ = policy.sac_forward(obs, deterministic=True)
    action_b, _, _ = policy.sac_forward(other, deterministic=True)
    assert not torch.allclose(action_a, action_b, atol=1e-5)


def _hist_obs(batch: int = 2, steps: int = 3, prefix_len: int = 8, z_dim: int = 32):
    obs = _obs(batch, prefix_len, z_dim)
    obs["hist_prefix_embs"] = torch.randn(batch, steps, prefix_len, z_dim)
    obs["hist_prefix_mask"] = torch.ones(batch, steps, prefix_len, dtype=torch.bool)
    obs["hist_action"] = torch.randn(batch, steps, 16)
    obs["hist_reward"] = torch.randn(batch, steps, 1)
    obs["hist_ref"] = torch.randn(batch, steps, 16)
    obs["hist_valid"] = torch.ones(batch, steps, dtype=torch.bool)
    return obs


def test_recon_target_is_topk_while_image_span_stays_full():
    from rlinf.models.embodiment.modules.rlt_mem_write import (
        full_image_tokens_and_recon_mask,
    )

    image = torch.zeros(1, 4, 8)
    image[0, 0] = 1
    text = image[:, :1]
    prefix = torch.cat([image, text], dim=1)
    tokens, mask, recon = full_image_tokens_and_recon_mask(prefix, None, 1, 0.5)
    assert tokens.shape == (1, 4, 8)
    torch.testing.assert_close(tokens, image)
    assert mask.shape == (1, 4)
    assert bool(mask.all())
    assert recon.shape == (1, 4)
    assert int(recon.sum()) == 2
    assert bool(recon[0, 0])


def test_scheme2_alternates_cross_and_self_and_returns_z():
    policy = RLTMLPPolicy(
        z_dim=32,
        proprio_dim=4,
        action_dim=8,
        num_action_chunks=2,
        use_mem=True,
        loop_prefix_len=8,
        loop_num_heads=4,
        loop_num_layers=2,
        mem_scheme=2,
    )
    encoder = policy.loop_encoder
    assert encoder.block_kinds == ("cross", "self")
    assert len(encoder.cross_layers) == 1
    assert len(encoder.self_layers) == 1
    obs = _obs(2)
    z = policy._write_mem(obs)
    assert z.shape == (2, 32)
    other = dict(obs)
    other["prefix_embs"] = torch.randn_like(obs["prefix_embs"])
    z_other = policy._write_mem(other)
    assert not torch.allclose(z, z_other, atol=1e-5)


def test_scheme2_write_accepts_bf16_prefix_with_fp32_weights():
    policy = RLTMLPPolicy(
        z_dim=32,
        proprio_dim=4,
        action_dim=8,
        num_action_chunks=2,
        use_mem=True,
        loop_prefix_len=8,
        loop_num_heads=4,
        loop_num_layers=2,
        mem_scheme=2,
    ).float()
    obs = _obs(2)
    obs["prefix_embs"] = obs["prefix_embs"].to(dtype=torch.bfloat16)
    obs["z_prev"] = obs["z_prev"].to(dtype=torch.bfloat16)
    z = policy._write_mem(obs)
    assert z.shape == (2, 32)
    assert z.dtype == torch.float32


def test_train_loop_samples_suffix_length():
    policy = RLTMLPPolicy(
        z_dim=32,
        proprio_dim=4,
        action_dim=8,
        num_action_chunks=2,
        use_mem=True,
        loop_prefix_len=8,
        loop_num_heads=4,
        loop_num_layers=1,
        mem_len_min=1,
        mem_unroll_len=10,
    )
    policy.train()
    obs = _hist_obs(steps=10)
    policy._crop_hist_to_random_len(obs)
    lengths = obs["_mem_loop_len"]
    assert lengths.shape == (2,)
    assert int(lengths.min()) >= 1
    assert int(lengths.max()) <= 10
    valid = obs["hist_valid"]
    for row, length in enumerate(lengths.tolist()):
        assert valid[row, -length:].all()
        if length < 10:
            assert not valid[row, : 10 - length].any()
    again = valid.clone()
    policy._crop_hist_to_random_len(obs)
    torch.testing.assert_close(obs["hist_valid"].to(torch.int), again.to(torch.int))


def test_unroll_uses_earlier_chunk_not_just_current_prefix():
    policy = _policy()
    policy.eval()
    obs = _hist_obs()
    other = dict(obs)
    other["hist_prefix_embs"] = obs["hist_prefix_embs"].clone()
    other["hist_prefix_embs"][:, 0] = torch.randn_like(obs["hist_prefix_embs"][:, 0])
    action_a, _, _ = policy.sac_forward(obs, deterministic=True)
    action_b, _, _ = policy.sac_forward(other, deterministic=True)
    assert not torch.allclose(action_a, action_b, atol=1e-5)
    z_a, _, _ = policy._unroll_from_obs(obs)
    z_b, _, _ = policy._unroll_from_obs(other)
    assert not torch.allclose(z_a, z_b, atol=1e-5)


def test_encoder_ignores_ref_chunk():
    policy = _policy()
    obs = _obs(2)
    z1 = policy._write_mem(obs)
    obs["ref_chunk"] = torch.randn_like(obs["ref_chunk"])
    z2 = policy._write_mem(obs)
    assert torch.allclose(z1, z2)


def test_reconstruction_ignores_unselected_tokens():
    policy = _policy()
    batch = 2
    obs = _obs(batch)
    recon_mask = torch.zeros(batch, 8, dtype=torch.bool)
    recon_mask[:, :4] = True
    obs["recon_mask"] = recon_mask
    loss, metrics = policy.reconstruction_loss(obs)
    assert loss.ndim == 0
    assert "recon_loss" in metrics
    changed = dict(obs)
    changed["prefix_embs"] = obs["prefix_embs"].clone()
    changed["prefix_embs"][:, 4:] += 10
    changed_loss, _ = policy.reconstruction_loss(changed)
    torch.testing.assert_close(loss, changed_loss)


def test_write_scale_does_not_grow_z():
    policy = _policy()
    obs = _obs(1)
    z_small = policy._write_mem(obs)
    obs["prefix_embs"] = obs["prefix_embs"] * 50
    obs["z_prev"] = z_small
    z_large = policy._write_mem(dict(obs))
    assert z_large.norm().item() < z_small.norm().item() * 5


def test_different_tokens_change_z_direction():
    policy = _policy()
    policy.eval()
    obs = _obs(1)
    z_a = policy._write_mem(obs)
    other = dict(obs)
    other["prefix_embs"] = torch.randn_like(obs["prefix_embs"])
    z_b = policy._write_mem(other)
    cosine = torch.nn.functional.cosine_similarity(z_a, z_b, dim=-1)
    assert cosine.item() < 0.99


def test_rollout_mem_resets_on_done():
    policy = _policy()
    policy.eval()
    batch = 2
    obs = {
        "z_rl": torch.randn(batch, 32),
        "proprio": torch.randn(batch, 4),
        "ref_chunk": torch.randn(batch, 2, 8),
        "prefix_embs": torch.randn(batch, 8, 32),
        "prefix_mask": torch.ones(batch, 8, dtype=torch.bool),
    }
    actions, result = policy.predict_action_batch(obs, mode="eval")
    first_z = result["forward_inputs"]["z_rl"].clone()
    policy.commit_rollout_action(actions)
    dones = torch.tensor([[True], [False]])
    _, result2 = policy.predict_action_batch(
        obs, mode="eval", dones=dones, rewards=torch.ones(batch, 2)
    )
    init = policy.loop_encoder.initial_mem(batch, first_z.device, first_z.dtype)
    assert torch.allclose(result2["forward_inputs"]["z_prev"][0], init[0])
    assert not torch.allclose(result2["forward_inputs"]["z_prev"][1], init[1])
    assert result2["forward_inputs"]["z_rl"].shape == first_z.shape


def test_stage1_ckpt_writes_z_and_resets_done_rows():
    import tempfile
    from pathlib import Path

    from rlinf.models.embodiment.modules.rlt_token_transformer import (
        RLTTokenTransformer,
    )

    module = RLTTokenTransformer(
        input_dim=32,
        embed_dim=32,
        prefix_seq_len=8,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        _run_stage1_ckpt_policy(module, Path(tmp) / "full_weights.pt")


def _run_stage1_ckpt_policy(module, ckpt):
    torch.save(
        {f"mem.{key}": value for key, value in module.state_dict().items()},
        ckpt,
    )
    policy = RLTMLPPolicy(
        z_dim=32,
        proprio_dim=4,
        action_dim=8,
        num_action_chunks=2,
        use_mem=True,
        mem_ckpt=str(ckpt),
        rlt_input_dim=32,
        rlt_prefix_seq_len=8,
        rlt_mlp_ratio=2.0,
        loop_num_heads=4,
        loop_num_layers=1,
    )
    policy.eval()
    assert policy.loop_encoder is None
    assert policy.mem is not None
    assert not any(param.requires_grad for param in policy.mem.parameters())
    batch = 2
    obs = {
        "z_rl": torch.zeros(batch, 32),
        "proprio": torch.randn(batch, 4),
        "ref_chunk": torch.randn(batch, 2, 8),
        "prefix_embs": torch.randn(batch, 8, 32),
        "prefix_mask": torch.ones(batch, 8, dtype=torch.bool),
    }
    _, result = policy.predict_action_batch(obs, mode="eval")
    expected = module.step_memory(obs["prefix_embs"], obs["prefix_mask"], None)
    assert torch.allclose(
        result["forward_inputs"]["z_rl"], expected.to(dtype=torch.float32)
    )
    dones = torch.tensor([[True], [False]])
    _, result2 = policy.predict_action_batch(
        obs, mode="eval", dones=dones, rewards=torch.ones(batch, 2)
    )
    init = policy.mem.encoder.z_init.detach().reshape(1, -1)
    assert torch.allclose(result2["forward_inputs"]["z_prev"][0], init[0])
    assert not torch.allclose(result2["forward_inputs"]["z_prev"][1], init[0])
