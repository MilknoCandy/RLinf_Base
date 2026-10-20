# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LIBERO-PRO perturbation sampling (language/stove vs object/swap/task)."""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

PRO_PERT_TAGS = ("_lan", "_object", "_swap", "_task")


def pro_pert_tag_from_path(path: str) -> str:
    """Return ``_lan`` / ``_object`` / ``_swap`` / ``_task`` from a BDDL path."""
    folder = os.path.basename(os.path.dirname(os.path.abspath(str(path))))
    for tag in PRO_PERT_TAGS:
        if folder.endswith(tag):
            return tag
    return ""


def choose_pro_candidate(
    candidates: list[str],
    weights: dict[str, float] | None,
    rng: np.random.Generator,
) -> str:
    """Sample a BDDL path. ``weights`` keys are ``lan``/``object``/``swap``/``task``.

    First pick a perturbation type by weight, then uniformly among files of
    that type. Missing types are skipped. ``weights=None`` is uniform over files.
    """
    if not candidates:
        raise ValueError("choose_pro_candidate requires at least one candidate.")
    if weights is None:
        return str(rng.choice(candidates))
    groups: dict[str, list[str]] = defaultdict(list)
    for path in candidates:
        tag = pro_pert_tag_from_path(path)
        if tag:
            groups[tag].append(path)
    labels: list[str] = []
    probs: list[float] = []
    for tag in PRO_PERT_TAGS:
        files = groups.get(tag) or []
        key = tag.lstrip("_")
        weight = float(weights.get(key, 0.0))
        if files and weight > 0.0:
            labels.append(tag)
            probs.append(weight)
    if not labels:
        return str(rng.choice(candidates))
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    chosen = str(rng.choice(np.asarray(labels), p=p))
    return str(rng.choice(groups[chosen]))
