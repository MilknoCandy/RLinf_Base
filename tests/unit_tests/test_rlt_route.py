# Copyright 2026 The RLinf Authors.

import torch

from rlinf.algorithms.rlt.route import RLTRouteContext, SimulatorRLTRoute


def _ctx(*, mode: str, version: int, clone_ready: bool, critical: bool = True):
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
        clone_ready=clone_ready,
    )


def test_train_stays_on_ref_until_clone_ready():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=30000,
        collect_student_when_ready=True,
    )
    out = route.route(_ctx(mode="train", version=50000, clone_ready=False))
    assert torch.allclose(out.actions, torch.full_like(out.actions, 0.25))
    assert not out.result["forward_inputs"]["actor_switch"].any()


def test_eval_uses_student_after_warmup_even_if_clone_not_ready():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=30000,
        collect_student_when_ready=True,
    )
    out = route.route(_ctx(mode="eval", version=50000, clone_ready=False))
    assert torch.allclose(out.actions, torch.ones_like(out.actions))
    assert out.result["forward_inputs"]["actor_switch"].all()


def test_train_uses_student_when_clone_ready():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=30000,
        collect_student_when_ready=True,
    )
    out = route.route(_ctx(mode="train", version=50000, clone_ready=True))
    assert torch.allclose(out.actions, torch.ones_like(out.actions))
    assert out.result["forward_inputs"]["actor_switch"].all()


def test_original_schedule_ignores_clone_ready():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=30000,
        collect_student_when_ready=False,
    )
    out = route.route(_ctx(mode="train", version=50000, clone_ready=False))
    assert torch.allclose(out.actions, torch.ones_like(out.actions))
