# Copyright 2026 The RLinf Authors.

from __future__ import annotations

import numpy as np

from rlinf.envs.libero.pro_pert import choose_pro_candidate, pro_pert_tag_from_path


def test_pro_pert_tag_from_path():
    assert (
        pro_pert_tag_from_path("/data/bddl/libero_spatial_lan/pick_bowl.bddl")
        == "_lan"
    )
    assert (
        pro_pert_tag_from_path("/data/bddl/libero_spatial_object/pick_bowl.bddl")
        == "_object"
    )
    assert pro_pert_tag_from_path("/data/bddl/libero_spatial/pick_bowl.bddl") == ""


def test_choose_pro_candidate_respects_lan_weight():
    candidates = [
        "/bddl/libero_spatial_lan/a.bddl",
        "/bddl/libero_spatial_object/a.bddl",
        "/bddl/libero_spatial_swap/a.bddl",
        "/bddl/libero_spatial_task/a.bddl",
    ]
    rng = np.random.default_rng(0)
    tags = [
        pro_pert_tag_from_path(
            choose_pro_candidate(
                candidates, {"lan": 0.85, "object": 0.05, "swap": 0.05, "task": 0.05}, rng
            )
        )
        for _ in range(200)
    ]
    assert tags.count("_lan") > 120


def test_choose_pro_candidate_equal_weights_are_type_uniform():
    """Equal type weights must not follow file counts (lan has extra BDDLs)."""
    candidates = [
        "/bddl/libero_spatial_lan/a.bddl",
        "/bddl/libero_spatial_lan/b.bddl",
        "/bddl/libero_spatial_lan/c.bddl",
        "/bddl/libero_spatial_object/a.bddl",
        "/bddl/libero_spatial_swap/a.bddl",
        "/bddl/libero_spatial_task/a.bddl",
    ]
    rng = np.random.default_rng(0)
    weights = {"lan": 0.25, "object": 0.25, "swap": 0.25, "task": 0.25}
    tags = [
        pro_pert_tag_from_path(choose_pro_candidate(candidates, weights, rng))
        for _ in range(4000)
    ]
    counts = [tags.count(tag) for tag in ("_lan", "_object", "_swap", "_task")]
    assert min(counts) > 800
    assert max(counts) < 1200


def test_choose_pro_candidate_uniform_without_weights():
    candidates = [
        "/bddl/libero_spatial_lan/a.bddl",
        "/bddl/libero_spatial_object/a.bddl",
    ]
    rng = np.random.default_rng(1)
    picked = {
        choose_pro_candidate(candidates, None, rng) for _ in range(40)
    }
    assert picked == set(candidates)
