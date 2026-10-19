# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import torch

from rlinf.utils.logging import get_logger

if TYPE_CHECKING:
    from libero.libero.benchmark import Benchmark


def get_libero_type() -> str:
    """
    Returns the type of LIBERO, which can be "standard", "pro", or "plus".
    """
    return os.environ.get("LIBERO_TYPE", "standard").lower()


libero_type = get_libero_type()

if libero_type == "pro":
    try:
        import liberopro.liberopro.benchmark as benchmark
        from liberopro.liberopro.benchmark import Benchmark
    except ImportError:
        print(
            "[Utils] Warning: LIBERO_TYPE=pro but 'liberopro' not found. Falling back to 'libero'."
        )
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark

elif libero_type == "plus":
    try:
        import liberoplus.liberoplus.benchmark as benchmark
        from liberoplus.liberoplus.benchmark import Benchmark
    except ImportError:
        print(
            "[Utils] Warning: LIBERO_TYPE=plus but 'liberoplus' not found. Falling back to 'libero'."
        )
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark

else:
    try:
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark
    except ImportError:
        try:
            import liberopro.liberopro.benchmark as benchmark
            from liberopro.liberopro.benchmark import Benchmark
        except ImportError:
            try:
                import liberoplus.liberoplus.benchmark as benchmark
                from liberoplus.liberoplus.benchmark import Benchmark
            except ImportError:
                raise ImportError(
                    "No valid LIBERO package (libero, liberopro, or liberoplus) found."
                )


def get_libero_image(obs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Extracts image from observations and preprocesses it.

    Args:
        obs: Observation dictionary from LIBERO environment

    Returns:
        Preprocessed image as numpy array
    """
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(
    obs: dict[str, np.ndarray], resize_size: Union[int, tuple[int, int]] = 224
) -> np.ndarray:
    """
    Extracts wrist camera image from observations and preprocesses it.

    Args:
        obs: Observation dictionary from LIBERO environment
        resize_size: Target size for resizing

    Returns:
        Preprocessed wrist camera image as numpy array
    """
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_benchmark_overridden(benchmark_name) -> Benchmark:
    """
    Return the Benchmark class for a given name.
    For "libero_130": return a dynamically aggregated class from all suites.
    For others: delegate to the original LIBERO get_benchmark.

    Args:
        benchmark_name: Name of the benchmark to get

    Returns:
        Benchmark class
    """
    name = str(benchmark_name).lower()
    if name != "libero_130":
        return benchmark.get_benchmark(benchmark_name)

    libero_cls = benchmark.BENCHMARK_MAPPING.get("libero_130", None)
    if libero_cls is not None:
        return libero_cls

    # Build aggregated task map once, preserving order and de-duplicating by task name
    aggregated_task_map: dict[str, benchmark.Task] = {}
    suites = getattr(benchmark, "libero_suites", [])
    for suite_name in suites:
        suite_map = benchmark.task_maps.get(suite_name, {})
        for task_name, task in suite_map.items():
            if task_name not in aggregated_task_map:
                aggregated_task_map[task_name] = task

    class LIBERO_ALL(Benchmark):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = "libero_130"
            self._make_benchmark()

        def _make_benchmark(self):
            tasks = list(aggregated_task_map.values())
            self.tasks = tasks
            self.n_tasks = len(self.tasks)

    # Register for discoverability/help
    benchmark.BENCHMARK_MAPPING["libero_130"] = LIBERO_ALL
    return LIBERO_ALL


def build_interleaved_eval_reset_state_ids(
    trial_id_bins: list[int],
    cumsum_trial_id_bins: np.ndarray,
) -> np.ndarray:
    """Order (task0, trial0), (task1, trial0), ... for even parallel coverage."""
    interleaved = []
    num_tasks = len(trial_id_bins)
    max_trials = max(trial_id_bins) if trial_id_bins else 0
    for trial in range(max_trials):
        for task_id in range(num_tasks):
            if trial < trial_id_bins[task_id]:
                start = cumsum_trial_id_bins[task_id - 1] if task_id > 0 else 0
                interleaved.append(start + trial)
    return np.array(interleaved, dtype=np.int64)


def distribute_reset_state_ids_round_robin(
    reset_state_ids: np.ndarray,
    total_num_processes: int,
) -> np.ndarray:
    """Assign each reset state to exactly one rank (round-robin)."""
    n_procs = total_num_processes
    n_states = len(reset_state_ids)
    per_rank_counts = np.bincount(np.arange(n_states) % n_procs, minlength=n_procs)
    max_per_rank = int(per_rank_counts.max())
    distributed = np.full((n_procs, max_per_rank), -1, dtype=np.int64)
    counters = np.zeros(n_procs, dtype=int)
    for i, state_id in enumerate(reset_state_ids):
        rank = int(i % n_procs)
        distributed[rank, counters[rank]] = state_id
        counters[rank] += 1
    return distributed


_LIBERO_PRO_PERTS = ("lan", "object", "swap", "task")


def perturbation_label_from_folder(folder: Optional[str]) -> str:
    """Map a LIBERO-Pro BDDL folder name to a short perturbation label."""
    if not folder:
        return "orig"
    name = os.path.basename(str(folder).rstrip("/"))
    for pert in _LIBERO_PRO_PERTS:
        if name.endswith(f"_{pert}"):
            return pert
    return "orig"


def load_nonempty_pruned_init(path: str):
    """Load a ``.pruned_init`` file, or return None if missing or empty."""
    if not path or not os.path.isfile(path):
        return None
    try:
        loaded = torch.load(path, weights_only=False)
    except TypeError:
        loaded = torch.load(path)
    except (OSError, RuntimeError, ValueError):
        return None
    if loaded is None:
        return None
    if hasattr(loaded, "__len__") and len(loaded) == 0:
        return None
    return loaded


def resolve_libero_pro_init_states(
    *,
    init_root: str,
    pert_folder: str,
    suite_folder: str,
    init_states_file: str,
):
    """Load LIBERO-Pro init states, falling back to the original suite file.

    Official ``*_task`` assets leave some ``.pruned_init`` files empty. The
    corresponding original-suite file uses the same scene, so it is a valid
    last resort for eval instead of aborting the whole perturbation.
    """
    candidates: list[tuple[str, str]] = []
    pert_folder = str(pert_folder or "").strip()
    suite_folder = str(suite_folder or "").strip()
    if pert_folder:
        candidates.append(
            (pert_folder, os.path.join(init_root, pert_folder, init_states_file))
        )
    if suite_folder and suite_folder != pert_folder:
        candidates.append(
            (suite_folder, os.path.join(init_root, suite_folder, init_states_file))
        )
    for folder, path in candidates:
        states = load_nonempty_pruned_init(path)
        if states is not None:
            return states, path, folder
    tried = ", ".join(path for _, path in candidates) or "<none>"
    raise FileNotFoundError(
        "No non-empty LIBERO-Pro pruned_init found for "
        f"{init_states_file!r} (tried {tried})."
    )


def seed_intervene_flags(num_envs: int, seed_active: bool) -> np.ndarray:
    """All-or-nothing expert flag for Appendix D seed-correction episodes."""
    return np.full((int(num_envs), 1), bool(seed_active), dtype=bool)


def merge_expert_intervene_flags(
    num_envs: int,
    *,
    seed_active: bool,
    progress_flags: np.ndarray | None = None,
    always: bool = False,
) -> np.ndarray:
    """OR seed / progress / always-on expert takeover (train collect only)."""
    flags = seed_intervene_flags(num_envs, seed_active)[:, 0]
    if progress_flags is not None:
        flags = flags | np.asarray(progress_flags, dtype=bool).reshape(-1)
    if always:
        flags = np.ones(int(num_envs), dtype=bool)
    return flags


def count_new_episode_dones(prev_dones: np.ndarray, dones: np.ndarray) -> tuple[np.ndarray, int]:
    """Count episode ends once. Sticky done=True after success must not re-count."""
    prev = np.asarray(prev_dones, dtype=bool).reshape(-1)
    cur = np.asarray(dones, dtype=bool).reshape(-1)
    if prev.shape != cur.shape:
        prev = np.zeros_like(cur)
    return cur.copy(), int((cur & ~prev).sum())


_BOWL_NEEDLES = ("bowl", "mug", "cup", "pot", "pan", "basket")
_PLATE_NEEDLES = ("plate", "platter", "dish", "saucer", "tray")


def _named_pos(obs: dict, needles: tuple[str, ...]) -> np.ndarray | None:
    for key, value in obs.items():
        name = str(key).lower()
        if not name.endswith("_pos"):
            continue
        if name.startswith("robot0_"):
            continue
        if not any(needle in name for needle in needles):
            continue
        pos = np.asarray(value, dtype=np.float64).reshape(-1)
        if pos.size >= 3:
            return pos[:3]
    return None


def bowl_place_progress_from_obs(
    obs: dict,
    *,
    grasp_dist: float = 0.07,
    gripper_closed: float = 0.045,
) -> tuple[float, bool]:
    """Scalar bowl-place progress from a LIBERO raw observation.

    Approach is ``-||ee - bowl||``. A grasp jumps the score by ``+1`` and
    then tracks ``-||bowl - plate||``. Returns ``(progress, valid)``; ``valid``
    is false when the bowl or plate pose is missing. LIBERO-PRO object
    perturbations often rename the bowl; callers must still stall on
    ``valid=False`` so the expert can return.
    """
    try:
        ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(-1)[:3]
    except (KeyError, TypeError, ValueError):
        return 0.0, False
    bowl = _named_pos(obs, _BOWL_NEEDLES)
    plate = _named_pos(obs, _PLATE_NEEDLES)
    if bowl is None or plate is None or ee.size < 3:
        return 0.0, False
    grip = np.asarray(
        obs.get("robot0_gripper_qpos", (0.04, 0.04)), dtype=np.float64
    ).reshape(-1)
    grasped = bool(
        np.linalg.norm(ee - bowl) < float(grasp_dist)
        and float(np.mean(np.abs(grip))) < float(gripper_closed)
    )
    if grasped:
        return 1.0 - float(np.linalg.norm(bowl - plate)), True
    return -float(np.linalg.norm(ee - bowl)), True


def update_progress_intervene_state(
    *,
    best: np.ndarray,
    stall: np.ndarray,
    burst_left: np.ndarray,
    initialized: np.ndarray,
    progress: np.ndarray,
    valid: np.ndarray,
    success: np.ndarray,
    done: np.ndarray,
    min_delta: float,
    stall_steps: int,
    burst_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Event-triggered short expert burst when bowl-place progress stalls.

    Unlike a horizon clock, each env triggers only after ``stall_steps``
    without improvement, then holds the expert for ``burst_steps`` and
    releases so the residual can try again.

    ``valid=False`` (missing / renamed object poses) still counts as a stall
    so the expert returns after seed. Only a measured improvement resets it.
    """
    best = np.asarray(best, dtype=np.float64).copy()
    stall = np.asarray(stall, dtype=np.int32).copy()
    burst_left = np.asarray(burst_left, dtype=np.int32).copy()
    initialized = np.asarray(initialized, dtype=bool).copy()
    progress = np.asarray(progress, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    success = np.asarray(success, dtype=bool)
    done = np.asarray(done, dtype=bool)

    # Poses may be missing under LIBERO-PRO object perturbations. Keep
    # watching those envs so ZAP can still take over after the seed quota.
    active = (~success) & (~done)
    in_burst = active & (burst_left > 0)
    burst_left = np.where(in_burst, burst_left - 1, burst_left)

    watch = active & (~in_burst)
    improved = (
        watch & initialized & valid & (progress > best + float(min_delta))
    )
    first = watch & (~initialized)
    update_best = improved | (first & valid) | (in_burst & valid)
    best = np.where(update_best, np.maximum(best, progress), best)
    initialized = initialized | first | (in_burst & valid)

    stalled_now = watch & initialized & (~improved) & (~first)
    stall = np.where(stalled_now, stall + 1, stall)
    stall = np.where(first | improved | in_burst | (~active), 0, stall)

    trigger = watch & (stall >= int(stall_steps)) & (int(stall_steps) > 0)
    burst_left = np.where(trigger, int(burst_steps), burst_left)
    stall = np.where(trigger, 0, stall)

    burst_left = np.where(success | done, 0, burst_left)
    stall = np.where(success | done, 0, stall)
    initialized = initialized & (~done)
    best = np.where(done, np.full_like(best, -1.0e6), best)

    intervene = (burst_left > 0) | trigger
    intervene = intervene & (~success) & (~done)
    return best, stall, burst_left, initialized, intervene.astype(bool)


def record_completed_episode_task_stats(
    env_idx: np.ndarray,
    final_info: dict,
    task_ids: np.ndarray,
    trial_ids: np.ndarray,
    num_envs: int,
    eval_seen_trials: set[tuple[int, int]],
    task_success_stats: dict[int, dict[str, int]],
    logger=None,
    pert_folders: Optional[Sequence[str]] = None,
    task_names: Optional[dict[int, str]] = None,
) -> np.ndarray:
    """Record per-task eval stats and return which envs count toward metrics.

    In eval mode each (task_id, trial_id) is counted at most once. Duplicate
    completions from auto_reset cycling are excluded so aggregated eval metrics
    match the benchmark suite size instead of counting every episode termination.

    When ``pert_folders`` is set (LIBERO-Pro), each log line includes ``pert=``.
    """
    logger = logger or get_logger()
    count_mask = np.zeros(num_envs, dtype=bool)

    episode = final_info.get("episode")
    if not episode or "success_once" not in episode:
        return count_mask

    success = episode["success_once"]
    if isinstance(success, torch.Tensor):
        success = success.cpu().numpy()
    else:
        success = np.asarray(success)

    for eid in env_idx:
        tid = int(task_ids[eid])
        trial_id = int(trial_ids[eid])
        # Idle eval slots are padded with -1 when E does not divide S.
        if tid < 0 or trial_id < 0:
            continue
        ok = bool(success[eid])
        trial_key = (tid, trial_id)
        if trial_key in eval_seen_trials:
            logger.warning(
                f"[libero eval] duplicate episode skipped: "
                f"task_id={tid}, trial_id={trial_id}"
            )
            continue
        eval_seen_trials.add(trial_key)
        count_mask[eid] = True
        if tid not in task_success_stats:
            task_success_stats[tid] = {"success": 0, "total": 0}
        task_success_stats[tid]["total"] += 1
        task_success_stats[tid]["success"] += int(ok)
        pert = "orig"
        if pert_folders is not None and eid < len(pert_folders):
            pert = perturbation_label_from_folder(pert_folders[eid])
        raw_name = ""
        if task_names and tid in task_names:
            raw_name = str(task_names[tid])
        task_name = raw_name.replace("'", " ")
        logger.info(
            f"[libero eval] task_id={tid}, trial_id={trial_id}, "
            f"pert={pert}, success={ok}, task='{task_name}'"
        )
    return count_mask
