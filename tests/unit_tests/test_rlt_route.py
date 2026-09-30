# Copyright 2026 The RLinf Authors.

import torch

from rlinf.algorithms.rlt.route import RLTRouteContext, SimulatorRLTRoute


def _ctx(*, mode: str, version: int, critical: bool = True):
    batch, chunk_len, action_dim = 2, 2, 3
    student = torch.ones(batch, chunk_len, action_dim)
    ref = torch.full((batch, chunk_len, action_dim), 0.25)
    switch = torch.full((batch,), critical, dtype=torch.bool)
    return RLTRouteContext(
        env_obs={},
        rlt_obs={"ref_chunk": ref},
        student_actions=student,
        result={"forward_inputs": {"ref_chunk": ref.clone()}},
        mode=mode,
        rlt_switch_flags=switch,
        version=version,
    )


def test_full_task_without_phase_flag_uses_student_after_warmup():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=10,
        full_task=True,
    )
    ctx = _ctx(mode="train", version=10)
    ctx.rlt_switch_flags = None
    out = route.route(ctx)
    assert torch.allclose(out.actions, torch.ones_like(out.actions))
    assert out.result["forward_inputs"]["record_transition"].all()


def test_original_schedule_uses_student_after_warmup():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=30000,
    )
    out = route.route(_ctx(mode="train", version=50000))
    assert torch.allclose(out.actions, torch.ones_like(out.actions))
