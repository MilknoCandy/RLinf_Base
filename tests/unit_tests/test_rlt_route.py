# Copyright 2026 The RLinf Authors.

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.route import (
    RLTRouteContext,
    SimulatorRLTRoute,
    build_rlt_route,
)
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay


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


def _stage2_cfg(env_type: str, *, mode: str = "none"):
    return OmegaConf.create(
        {
            "env": {"train": {"env_type": env_type}},
            "algorithm": {
                "rlt_schedule": {"enable": True, "warmup_post_collect_updates": 5},
                "rlt_phase_gate": {"mode": mode},
            },
        }
    )


def test_metaworld_uses_simulator_replay_and_full_task():
    cfg = _stage2_cfg("metaworld", mode="none")
    assert use_simulator_transition_replay(cfg)
    route = build_rlt_route(cfg)
    assert isinstance(route, SimulatorRLTRoute)
    assert route.full_task is True
    assert route.phase_gate is None


def test_metaworld_vlm_mode_enables_phase_gate():
    cfg = _stage2_cfg("metaworld", mode="vlm")
    route = build_rlt_route(cfg)
    assert isinstance(route, SimulatorRLTRoute)
    assert route.full_task is False
    assert route.phase_gate is not None


def test_maniskill_rlt_uses_simulator_replay_without_full_task():
    cfg = _stage2_cfg("maniskill_rlt", mode="none")
    assert use_simulator_transition_replay(cfg)
    route = build_rlt_route(cfg)
    assert isinstance(route, SimulatorRLTRoute)
    assert route.full_task is False
    assert route.phase_gate is None
