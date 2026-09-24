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


def test_actor_state_is_looped_z_and_proprio():
    policy = _policy()
    batch = 3
    obs = _obs(batch)
    policy.train()
    state = policy._actor_state(obs)
    assert state.shape == (batch, 32 + 4)
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


def test_unroll_uses_earlier_chunk_not_just_current_ref():
    policy = _policy()
    policy.eval()
    obs = _hist_obs()
    other = dict(obs)
    other["hist_action"] = torch.randn_like(obs["hist_action"])
    other["hist_action"][:, -1] = obs["hist_action"][:, -1]
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


def test_predictive_loss_fits_ref_not_image():
    policy = _policy()
    batch = 2
    obs = _obs(batch)
    ref = policy._get_ref_chunk(obs)
    pred_ref, pred_r, metrics = policy.predictive_loss(obs, ref, torch.zeros(batch, 1))
    assert pred_ref.ndim == 0
    assert pred_r.ndim == 0
    assert "pred_ref_loss" in metrics
    assert "pred_rec_loss" not in metrics
    action, _, _ = policy.sac_forward(obs)
    assert not torch.allclose(action, torch.tanh(ref), atol=1e-3)


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
