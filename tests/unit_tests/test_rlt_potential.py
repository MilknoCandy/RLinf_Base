# Copyright 2026 The RLinf Authors.

import torch

from rlinf.envs.maniskill.rlt_potential import (
    peg_insertion_potential,
    success_potential_step_reward,
)


def test_potential_rises_when_closer_to_hole():
    far = peg_insertion_potential(
        torch.tensor([-0.10]),
        torch.tensor([0.04]),
        torch.tensor([0.04]),
    )
    near = peg_insertion_potential(
        torch.tensor([-0.02]),
        torch.tensor([0.01]),
        torch.tensor([0.01]),
    )
    assert near.item() > far.item()


def test_first_step_has_no_shaping():
    phi = torch.tensor([0.05, -0.02])
    reward = success_potential_step_reward(
        torch.zeros(2, dtype=torch.bool),
        phi,
        torch.zeros(2),
        torch.zeros(2, dtype=torch.bool),
        coef=1.0,
        gamma=0.99,
    )
    assert torch.allclose(reward, torch.zeros(2))


def test_approach_gives_positive_reward_before_success():
    prev = torch.tensor([-0.08])
    cur = torch.tensor([-0.03])
    reward = success_potential_step_reward(
        torch.zeros(1, dtype=torch.bool),
        cur,
        prev,
        torch.ones(1, dtype=torch.bool),
        coef=1.0,
        gamma=0.99,
    )
    assert reward.item() > 0.0


def test_success_adds_sparse_bonus():
    phi = torch.tensor([0.0])
    reward = success_potential_step_reward(
        torch.ones(1, dtype=torch.bool),
        phi,
        phi,
        torch.ones(1, dtype=torch.bool),
        coef=1.0,
        gamma=1.0,
    )
    assert torch.allclose(reward, torch.ones(1))
