# Copyright 2026 The RLinf Authors.

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.route import (
    RLTRouteContext,
    SimulatorRLTRoute,
    build_rlt_route,
    expert_disagreement_mask,
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


def _stage2_cfg(env_type: str):
    return OmegaConf.create(
        {
            "env": {"train": {"env_type": env_type}},
            "algorithm": {
                "rlt_schedule": {"enable": True, "warmup_post_collect_updates": 5},
            },
        }
    )


def test_metaworld_uses_simulator_replay_and_full_task():
    cfg = _stage2_cfg("metaworld")
    assert use_simulator_transition_replay(cfg)
    route = build_rlt_route(cfg)
    assert isinstance(route, SimulatorRLTRoute)
    assert route.full_task is True


def test_maniskill_rlt_uses_simulator_replay_without_full_task():
    cfg = _stage2_cfg("maniskill_rlt")
    assert use_simulator_transition_replay(cfg)
    route = build_rlt_route(cfg)
    assert isinstance(route, SimulatorRLTRoute)
    assert route.full_task is False


def test_eval_only_libero_uses_full_task_actor_without_schedule():
    """Standalone eval YAMLs omit env.train / algorithm; residual must still run."""
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": True},
            "env": {"eval": {"env_type": "libero"}},
        }
    )
    assert use_simulator_transition_replay(cfg)
    route = build_rlt_route(cfg)
    assert isinstance(route, SimulatorRLTRoute)
    assert route.full_task is True
    assert route.use_schedule is False
    ctx = _ctx(mode="eval", version=0)
    ctx.rlt_switch_flags = None
    out = route.route(ctx)
    assert torch.allclose(out.actions, torch.ones_like(out.actions))
    assert out.result["forward_inputs"]["actor_switch"].all()


class _FakeExpert:
    def predict_action_batch(self, env_obs, mode="eval", compute_values=False):
        del env_obs, mode, compute_values
        return torch.full((2, 2, 3), 7.0), {}


def test_expert_takeover_does_not_wait_for_residual_warmup():
    route = SimulatorRLTRoute(use_schedule=True, warmup_updates=10)
    ctx = _ctx(mode="train", version=0)
    ctx.expert_model = _FakeExpert()
    ctx.intervene_requested = torch.ones(2, dtype=torch.bool)
    out = route.route(ctx)
    assert torch.allclose(out.actions, torch.full_like(out.actions, 7.0))
    assert out.result["intervene_flags"].all()
    assert not out.result["forward_inputs"]["actor_switch"].any()


def test_expert_takeover_keeps_vla_ref_by_default():
    route = SimulatorRLTRoute(use_schedule=True, warmup_updates=0)
    ctx = _ctx(mode="train", version=1)
    ctx.expert_model = _FakeExpert()
    ctx.intervene_requested = torch.ones(2, dtype=torch.bool)
    out = route.route(ctx)
    assert torch.allclose(out.actions, torch.full_like(out.actions, 7.0))
    ref = out.result["forward_inputs"]["ref_chunk"]
    assert torch.allclose(ref, torch.full_like(ref, 0.25))


def test_disagreement_mask_ignores_gripper_and_flags_far_pose():
    student = torch.zeros(2, 2, 3)
    expert = torch.zeros(2, 2, 3)
    expert[0, :, :2] = 0.5
    expert[1, :, 2] = 1.0
    mask = expert_disagreement_mask(
        student, expert, l1_threshold=0.12, ignore_dims=(2,)
    )
    assert mask.tolist() == [True, False]


def test_correct_when_disagree_takes_over_only_the_far_env():
    class _SplitExpert:
        def predict_action_batch(self, env_obs, mode="eval", compute_values=False):
            del env_obs, mode, compute_values
            actions = torch.zeros(2, 2, 3)
            actions[0] = 7.0
            return actions, {}

    route = SimulatorRLTRoute(
        use_schedule=False,
        warmup_updates=0,
        full_task=True,
        correct_when_disagree=True,
        correct_action_l1=0.12,
        correct_ignore_dims=(2,),
    )
    ctx = _ctx(mode="train", version=1)
    ctx.rlt_switch_flags = None
    ctx.student_actions = torch.zeros(2, 2, 3)
    ctx.student_actions[0] = 1.0
    ctx.expert_model = _SplitExpert()
    out = route.route(ctx)
    assert torch.allclose(out.actions[0], torch.full((2, 3), 7.0))
    assert torch.allclose(out.actions[1], torch.zeros(2, 3))
    assert out.result["intervene_flags"][0].all()
    assert not out.result["intervene_flags"][1].any()
    assert out.result["rlt_correct_metrics"]["correct_disagree_rate"] == 0.5


def test_correct_when_disagree_lets_matching_residual_act():
    class _MatchExpert:
        def predict_action_batch(self, env_obs, mode="eval", compute_values=False):
            del env_obs, mode, compute_values
            return torch.ones(2, 2, 3), {}

    route = SimulatorRLTRoute(
        use_schedule=False,
        warmup_updates=0,
        full_task=True,
        correct_when_disagree=True,
        correct_action_l1=0.12,
    )
    ctx = _ctx(mode="train", version=1)
    ctx.rlt_switch_flags = None
    ctx.expert_model = _MatchExpert()
    out = route.route(ctx)
    assert torch.allclose(out.actions, torch.ones_like(out.actions))
    assert not out.result["intervene_flags"].any()
    assert out.result["forward_inputs"]["actor_switch"].all()


def test_correct_when_disagree_skipped_in_eval():
    route = SimulatorRLTRoute(
        use_schedule=False,
        warmup_updates=0,
        full_task=True,
        correct_when_disagree=True,
        correct_action_l1=0.01,
    )
    ctx = _ctx(mode="eval", version=0)
    ctx.rlt_switch_flags = None
    ctx.expert_model = _FakeExpert()
    out = route.route(ctx)
    assert torch.allclose(out.actions, torch.ones_like(out.actions))
    assert not out.result["intervene_flags"].any()


def test_build_route_reads_correct_when_disagree():
    cfg = OmegaConf.create(
        {
            "env": {"train": {"env_type": "libero"}},
            "algorithm": {
                "rlt_schedule": {"enable": True, "warmup_post_collect_updates": 5},
                "bee": {
                    "correct_when_disagree": True,
                    "correct_action_l1": 0.2,
                    "correct_ignore_dims": [6],
                },
            },
        }
    )
    route = build_rlt_route(cfg)
    assert route.correct_when_disagree is True
    assert route.correct_action_l1 == 0.2
    assert route.correct_ignore_dims == (6,)
