# Copyright 2026 The RLinf Authors.

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACReplayMixin


class _ReplayHarness(RLTACReplayMixin):
    def __init__(self, flat: dict):
        self.cfg = OmegaConf.create(
            {
                "env": {"train": {"auto_reset": False}},
                "algorithm": {"n_step": 1, "gamma": 0.99},
            }
        )
        self.replay_buffer = SimpleNamespace(_flatten_trajectory=lambda _traj: flat)


def _flat_row(*, t: int, b: int, record: bool) -> dict:
    z = torch.zeros(t, b, 4)
    proprio = torch.zeros(t, b, 3)
    ref = torch.zeros(t, b, 2)
    actions = torch.zeros(t + 1, b, 2)
    record_t = torch.ones(t + 1, b, 1, dtype=torch.bool)
    if not record:
        record_t[:] = False
    flat = {
        "actions": actions.reshape(-1, 2),
        "rewards": torch.zeros((t + 1) * b, 1),
        "dones": torch.zeros((t + 1) * b, 1, dtype=torch.bool),
        "terminations": torch.zeros((t + 1) * b, 1, dtype=torch.bool),
        "truncations": torch.zeros((t + 1) * b, 1, dtype=torch.bool),
        "forward_inputs": {"record_transition": record_t.reshape(-1, 1)},
        "curr_obs": {
            "z_rl": z.reshape(-1, 4),
            "proprio": proprio.reshape(-1, 3),
            "ref_chunk": ref.reshape(-1, 2),
        },
        "next_obs": {
            "z_rl": z.reshape(-1, 4),
            "proprio": proprio.reshape(-1, 3),
            "ref_chunk": ref.reshape(-1, 2),
        },
    }
    return flat


def test_ingest_skips_bootstrap_row_without_curr_obs():
    t, b = 4, 64
    flat = _flat_row(t=t, b=b, record=True)
    traj = Trajectory(
        actions=torch.zeros(t + 1, b, 2),
        rewards=torch.zeros(t + 1, b, 1),
        dones=torch.zeros(t + 1, b, 1, dtype=torch.bool),
    )
    worker = _ReplayHarness(flat)
    rows, _completed = worker._transition_replay_trajectories(traj)
    assert len(rows) == t * b
    assert all(row.curr_obs and "z_rl" in row.curr_obs for row in rows)
