# Copyright 2026 The RLinf Authors.

import torch

from rlinf.algorithms.rlt.ac_memory import (
    ACMemoryWriter,
    ActorCorrectionBank,
    CriticReturnBank,
    MemoryStep,
    memory_confidence,
    mix_q_values,
    retrieve_returns,
)
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def _actor_bank(capacity: int = 4, z_dim: int = 4, action_dim: int = 2, topk: int = 2):
    bank = ActorCorrectionBank(
        z=torch.zeros(capacity, z_dim),
        delta=torch.zeros(capacity, action_dim),
        returns=torch.zeros(capacity),
        advantage=torch.zeros(capacity),
        episode=torch.full((capacity,), -1, dtype=torch.long),
        time=torch.zeros(capacity, dtype=torch.long),
        valid=torch.zeros(capacity),
        topk=topk,
        tau=0.1,
    )
    return bank


def _writer(capacity: int = 4, discount: float = 0.5):
    critic = CriticReturnBank(capacity, z_dim=4, action_dim=2)
    return ACMemoryWriter(critic, _actor_bank(capacity), discount)


def _step(traj_t: int, reward: float, done: bool, z_value: float = 1.0, delta=None):
    if delta is None:
        delta = torch.tensor([0.2, -0.1])
    return MemoryStep(
        traj_t=traj_t,
        z=torch.full((4,), z_value),
        delta=delta,
        reward=reward,
        done=done,
    )


def test_use_mem_and_actor_mem_are_mutually_exclusive():
    try:
        RLTMLPPolicy(
            z_dim=8,
            proprio_dim=2,
            action_dim=2,
            num_action_chunks=1,
            use_mem=True,
            use_actor_mem=True,
            loop_prefix_len=4,
            loop_num_heads=2,
            loop_num_layers=1,
        )
    except ValueError as exc:
        assert "use_actor_mem" in str(exc)
        return
    raise AssertionError("use_mem and use_actor_mem must not both construct.")


def test_actor_memory_changes_action_without_changing_residual():
    policy = RLTMLPPolicy(
        z_dim=4,
        proprio_dim=2,
        action_dim=2,
        num_action_chunks=1,
        use_actor_mem=True,
        ac_mem_capacity=4,
        ac_mem_topk=1,
        ac_mem_tau=0.1,
    )
    policy.eval()
    obs = {
        "z_rl": torch.ones(1, 4),
        "proprio": torch.zeros(1, 2),
        "ref_chunk": torch.zeros(1, 2),
    }
    empty_action, _, empty_residual = policy.sac_forward(obs, deterministic=True)
    policy.actor_correction_bank.upsert(
        torch.ones(4),
        torch.tensor([0.4, -0.2]),
        1.0,
        1.0,
        episode_id=0,
        time=0,
    )
    filled_action, _, filled_residual = policy.sac_forward(obs, deterministic=True)
    assert torch.allclose(empty_residual, filled_residual)
    assert torch.allclose(filled_action, empty_action + torch.tensor([[0.4, -0.2]]))
    assert not torch.allclose(empty_action, filled_action)


def test_actor_read_hides_same_episode_future():
    policy = RLTMLPPolicy(
        z_dim=4,
        proprio_dim=2,
        action_dim=2,
        num_action_chunks=1,
        use_actor_mem=True,
        ac_mem_capacity=4,
        ac_mem_topk=2,
        ac_mem_tau=0.1,
    )
    bank = policy.actor_correction_bank
    z = torch.ones(4)
    bank.upsert(z, torch.tensor([1.0, 0.0]), 1.0, 1.0, episode_id=3, time=0)
    bank.upsert(z, torch.tensor([0.0, 1.0]), 1.0, 1.0, episode_id=3, time=1)
    hidden = bank.retrieve_action(
        z.unsqueeze(0),
        torch.tensor([3]),
        torch.tensor([1]),
    )
    assert torch.allclose(hidden, torch.tensor([[1.0, 0.0]]), atol=1e-5)
    rollout = bank.retrieve_action(
        z.unsqueeze(0),
        torch.tensor([-1]),
        torch.tensor([0]),
    )
    assert rollout.shape == (1, 2)
    assert float(rollout.abs().sum()) > 0.5


def test_negative_advantage_stays_out_of_actor_bank():
    writer = _writer()
    writer.ingest_env(
        0, [_step(0, reward=-1.0, done=True, delta=torch.tensor([0.5, 0.5]))]
    )
    assert int((writer.actor_bank.valid > 0.5).sum()) == 0
    assert int((writer.critic_bank.valid > 0.5).sum()) == 1
    writer.ingest_env(
        1, [_step(0, reward=1.0, done=True, delta=torch.tensor([0.3, 0.0]))]
    )
    assert int((writer.actor_bank.valid > 0.5).sum()) == 1


def test_continuation_updates_earlier_return():
    writer = _writer(discount=0.5)
    writer.ingest_env(0, [_step(0, reward=0.0, done=False)])
    writer.ingest_env(1, [_step(0, reward=1.0, done=True)])
    writer.ingest_env(0, [_step(0, reward=1.0, done=True)])
    episode = int(writer.critic_bank.episode[0])
    same = writer.critic_bank.episode == episode
    returns = writer.critic_bank.returns[same & (writer.critic_bank.valid > 0.5)]
    assert torch.allclose(returns.sort().values, torch.tensor([0.5, 1.0]))


def test_critic_read_excludes_current_transition():
    bank = CriticReturnBank(4, z_dim=4, action_dim=2)
    z = torch.ones(4)
    bank.write(z, torch.zeros(2), 1.0, episode_id=1, time=0)
    bank.write(z, torch.zeros(2), 100.0, episode_id=1, time=1)
    mean, _, count = retrieve_returns(
        bank,
        z.unsqueeze(0),
        torch.tensor([1]),
        torch.tensor([1]),
        topk=4,
        tau=0.1,
    )
    assert int(count.item()) == 1
    assert torch.allclose(mean, torch.tensor([1.0]))
    q = torch.zeros(1, 2)
    confidence = memory_confidence(torch.zeros(1), count, lambda_max=0.5, sigma0=1.0)
    mixed = mix_q_values(q, mean, confidence)
    assert torch.allclose(mixed, torch.full((1, 2), 0.5))
    other = mean.clone()
    bank.returns[0] = 3.0
    mean2, _, _ = retrieve_returns(
        bank,
        z.unsqueeze(0),
        torch.tensor([1]),
        torch.tensor([1]),
        topk=4,
        tau=0.1,
    )
    assert not torch.allclose(mean2, other)


def test_empty_memory_confidence_is_zero():
    confidence = memory_confidence(
        torch.zeros(2), torch.zeros(2), lambda_max=0.5, sigma0=1.0
    )
    assert torch.allclose(confidence, torch.zeros(2))
