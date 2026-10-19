# Copyright 2026 The RLinf Authors.

from types import SimpleNamespace

import torch

from rlinf.algorithms.rlt.transition import update_rlt_transitions


class _Builder:
    def __init__(self):
        self.rows = []

    def append_transitions(self, curr, nxt):
        self.rows.append((curr, nxt))


def test_human_intervene_keeps_vla_ref_as_actor_input():
    ref = torch.full((1, 2, 3), 0.25)
    pending = [
        {
            "z_rl": torch.zeros(1, 4),
            "proprio": torch.zeros(1, 2),
            "ref_chunk": ref.clone(),
        }
    ]
    builders = [_Builder()]
    fi = {
        "rlt_transition_z_rl": torch.zeros(1, 4),
        "rlt_transition_proprio": torch.zeros(1, 2),
        "rlt_transition_ref_chunk": torch.full((1, 2, 3), 0.3),
    }
    update_rlt_transitions(
        0,
        pending,
        builders,
        SimpleNamespace(forward_inputs=fi),
        cache_current=False,
    )
    curr, _ = builders[0].rows[0]
    assert torch.allclose(curr["ref_chunk"], torch.full_like(ref, 0.25))
