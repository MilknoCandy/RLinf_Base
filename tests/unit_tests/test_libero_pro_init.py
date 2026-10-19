# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import torch

from rlinf.envs.libero.utils import resolve_libero_pro_init_states, seed_intervene_flags


def _write_init(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([{"i": i} for i in range(n)], path)


def test_resolve_uses_pert_init_when_nonempty(tmp_path: Path):
    init_root = tmp_path / "init_files"
    name = "pick_bowl.pruned_init"
    _write_init(init_root / "libero_spatial_task" / name, 3)
    _write_init(init_root / "libero_spatial" / name, 50)
    states, path, folder = resolve_libero_pro_init_states(
        init_root=str(init_root),
        pert_folder="libero_spatial_task",
        suite_folder="libero_spatial",
        init_states_file=name,
    )
    assert folder == "libero_spatial_task"
    assert len(states) == 3
    assert path.endswith(f"libero_spatial_task/{name}")


def test_resolve_falls_back_to_suite_when_task_init_empty(tmp_path: Path):
    init_root = tmp_path / "init_files"
    name = "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.pruned_init"
    _write_init(init_root / "libero_spatial_task" / name, 0)
    _write_init(init_root / "libero_spatial" / name, 50)
    states, path, folder = resolve_libero_pro_init_states(
        init_root=str(init_root),
        pert_folder="libero_spatial_task",
        suite_folder="libero_spatial",
        init_states_file=name,
    )
    assert folder == "libero_spatial"
    assert len(states) == 50
    assert "libero_spatial/" in path.replace("\\", "/")


def test_resolve_raises_when_both_missing(tmp_path: Path):
    init_root = tmp_path / "init_files"
    name = "missing.pruned_init"
    try:
        resolve_libero_pro_init_states(
            init_root=str(init_root),
            pert_folder="libero_spatial_task",
            suite_folder="libero_spatial",
            init_states_file=name,
        )
    except FileNotFoundError as exc:
        assert "missing.pruned_init" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_seed_intervene_flags_are_all_or_nothing():
    on = seed_intervene_flags(3, True)
    off = seed_intervene_flags(3, False)
    assert on.shape == (3, 1)
    assert on.all()
    assert not off.any()
