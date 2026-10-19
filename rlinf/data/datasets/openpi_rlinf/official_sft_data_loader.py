# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapters for the official OpenPI PyTorch SFT data loader."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from rlinf.config import SupportedModel
from rlinf.data.storage.lerobot import (
    fix_lerobot_episode_data_index,
    resolve_lerobot_dataset_root,
    resolve_lerobot_repo_id,
)

_LOG = logging.getLogger(__name__)
_TASK_WS_RE = re.compile(r"\s+")


def normalize_sft_task_name(text: str) -> str:
    """Normalize a LIBERO / LeRobot task string for equality checks.

    Spatial suite names use underscores (``pick_up_the_black_bowl_...``);
    ``physical-intelligence/libero`` stores the same prompt with spaces.
    """
    return _TASK_WS_RE.sub(" ", str(text).replace("_", " ").strip().lower())


def collect_sft_task_filter(cfg: Any, data_paths: Any = None) -> list[str] | None:
    """Return ``data.task_filter`` or a per-path override, if any."""
    path_filter = _task_filter_from_data_paths(data_paths)
    if path_filter is not None:
        return path_filter
    return _as_str_list(OmegaConf.select(cfg, "data.task_filter", default=None))


def resolve_lerobot_task_filter_episodes(
    dataset_root: str | Path,
    task_filter: Sequence[str],
) -> list[int]:
    """Episode indices whose LeRobot task prompt matches ``task_filter``.

    Args:
        dataset_root: On-disk LeRobot dataset (must have ``meta/episodes.jsonl``).
        task_filter: Language prompts to keep. Underscores and spaces are
            equivalent. Matching is case-insensitive.

    Returns:
        Sorted unique episode indices.

    Raises:
        FileNotFoundError: If episode metadata is missing.
        ValueError: If ``task_filter`` is empty or matches no episode.
    """
    wanted = {
        normalize_sft_task_name(name) for name in task_filter if str(name).strip()
    }
    if not wanted:
        raise ValueError("data.task_filter must contain at least one non-empty prompt.")

    root = Path(dataset_root)
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(
            "OpenPI SFT task_filter needs meta/episodes.jsonl under "
            f"{root}. Point data.train_data_paths at a local LeRobot copy "
            "(e.g. libero_data), not a Hub id that has not been downloaded."
        )

    matched: list[int] = []
    available: set[str] = set()
    with episodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tasks = _episode_task_strings(record)
            available.update(tasks)
            if any(normalize_sft_task_name(task) in wanted for task in tasks):
                matched.append(int(record["episode_index"]))

    if not matched:
        preview = ", ".join(sorted(available)[:8])
        raise ValueError(
            "data.task_filter matched 0 episodes in "
            f"{episodes_path}. filter={sorted(wanted)}; "
            f"example tasks in the dataset: {preview}."
        )
    return sorted(set(matched))


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the SFT loader provided by OpenPI for a LeRobot dataset."""
    del rank
    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError(
            "OpenPI SFT requires data.train_data_paths to be set to a local "
            "dataset path or LeRobot repo id."
        )

    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    model_type = SupportedModel(model_cfg.model_type)
    batch_size = cfg.actor.micro_batch_size
    if eval_dataset:
        batch_size = cfg.actor.get("eval_batch_size", batch_size)

    config = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=batch_size * world_size,
        repo_id=repo_id,
        data_kwargs=getattr(model_cfg, "openpi_data", None),
    )
    if model_type == SupportedModel.OPENPI_RLINF:
        config = dataclasses.replace(
            config,
            num_workers=int(
                OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
            ),
            seed=int(OmegaConf.select(cfg, "actor.seed", default=config.seed)),
        )
        _validate_openpi_rlinf_model_shape(model_cfg, config)

    task_filter = collect_sft_task_filter(cfg, data_paths)
    restore_create = None
    if task_filter:
        dataset_root = resolve_lerobot_dataset_root(repo_id)
        episodes = resolve_lerobot_task_filter_episodes(dataset_root, task_filter)
        _LOG.info(
            "OpenPI SFT task_filter kept %d episodes from %s (%s).",
            len(episodes),
            dataset_root,
            task_filter,
        )
        restore_create = openpi_data_loader.create_torch_dataset

        def _create_filtered_torch_dataset(
            data_config: Any,
            action_horizon: int,
            model_config: Any,
        ) -> Any:
            return _create_torch_dataset_for_episodes(
                data_config,
                action_horizon,
                model_config,
                episodes=episodes,
                dataset_root=dataset_root,
            )

        openpi_data_loader.create_torch_dataset = _create_filtered_torch_dataset
    try:
        data_loader = openpi_data_loader.create_data_loader(
            config, framework="pytorch", shuffle=not eval_dataset
        )
    finally:
        if restore_create is not None:
            openpi_data_loader.create_torch_dataset = restore_create
    data_loader = _maybe_attach_rlt_chunk_window(cfg, data_loader)
    return data_loader, data_loader.data_config()


def _as_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in value]
    return [str(value)]


def _task_filter_from_data_paths(data_paths: Any) -> list[str] | None:
    if data_paths is None:
        return None
    if isinstance(data_paths, (dict, DictConfig)):
        return _as_str_list(data_paths.get("task_filter"))
    if isinstance(data_paths, (list, tuple, ListConfig)) and data_paths:
        return _task_filter_from_data_paths(data_paths[0])
    return None


def _episode_task_strings(record: dict[str, Any]) -> list[str]:
    tasks = record.get("tasks")
    if isinstance(tasks, str):
        return [tasks]
    if isinstance(tasks, Iterable) and not isinstance(tasks, (bytes, bytearray)):
        return [str(task) for task in tasks]
    task = record.get("task")
    if task is None:
        return []
    return [str(task)]


def _create_torch_dataset_for_episodes(
    data_config: Any,
    action_horizon: int,
    model_config: Any,
    *,
    episodes: Sequence[int],
    dataset_root: Path,
) -> Any:
    """OpenPI ``create_torch_dataset`` with a LeRobot ``episodes=`` subset."""
    import openpi.training.data_loader as openpi_data_loader
    import openpi.transforms as openpi_transforms

    del model_config
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    lerobot_dataset = openpi_data_loader._import_lerobot_dataset()
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=dataset_root)
    episode_ids = list(episodes)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episode_ids,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    fix_lerobot_episode_data_index(dataset, episode_ids)
    if data_config.prompt_from_task:
        dataset = openpi_data_loader.TransformedDataset(
            dataset,
            [openpi_transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )
    return dataset


def _maybe_attach_rlt_chunk_window(cfg: Any, data_loader: Any) -> Any:
    """Load a same-episode chunk window when stage-1 memory training is on."""
    use_rlt = bool(OmegaConf.select(cfg, "actor.model.openpi.use_rlt", default=False))
    mem_len_max = int(
        OmegaConf.select(cfg, "actor.model.openpi.rlt_mem_len_max", default=1)
    )
    if not use_rlt or mem_len_max <= 1:
        return data_loader
    mem_len_min = int(
        OmegaConf.select(cfg, "actor.model.openpi.rlt_mem_len_min", default=1)
    )
    if mem_len_min < 0 or mem_len_min > mem_len_max:
        raise ValueError(
            "actor.model.openpi.rlt_mem_len_min must lie in "
            f"[0, rlt_mem_len_max]; got min={mem_len_min} max={mem_len_max}."
        )
    stride = OmegaConf.select(cfg, "actor.model.openpi.rlt_mem_stride", default=None)
    if stride is None:
        stride = int(cfg.actor.model.num_action_chunks)
    else:
        stride = int(stride)
    if stride < 1:
        raise ValueError(
            f"actor.model.openpi.rlt_mem_stride must be positive, got {stride}."
        )
    from rlinf.data.datasets.openpi_rlinf.rlt_chunk_window import (
        attach_rlt_chunk_window,
    )

    return attach_rlt_chunk_window(data_loader, mem_len_max=mem_len_max, stride=stride)


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the inner PyTorch ``DataLoader`` length used by OpenPI."""
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None) or getattr(
        openpi_loader, "torch_loader", None
    )
    if torch_loader is None:
        raise TypeError(
            "OpenPI dataloader does not expose an inner torch DataLoader; "
            "cannot infer steps per epoch from len()."
        )
    return len(torch_loader)


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    """Return whether ``data_loader`` has OpenPI's loader wrapper layout."""
    return getattr(data_loader, "_data_loader", None) is not None


def _validate_openpi_rlinf_model_shape(model_cfg: Any, openpi_config: Any) -> None:
    """Keep the local Pi0 architecture consistent with the OpenPI config."""
    local_horizon = int(model_cfg.num_action_chunks)
    official_horizon = int(openpi_config.model.action_horizon)
    if local_horizon != official_horizon:
        raise ValueError(
            "openpi_rlinf SFT action horizon must match the official OpenPI "
            f"config: actor.model.num_action_chunks={local_horizon}, "
            f"{model_cfg.openpi.config_name}.model.action_horizon="
            f"{official_horizon}."
        )

    local_action_dim = int(model_cfg.openpi.model_action_dim)
    official_action_dim = int(openpi_config.model.action_dim)
    if local_action_dim != official_action_dim:
        raise ValueError(
            "openpi_rlinf SFT model action dim must match the official OpenPI "
            f"config: actor.model.openpi.model_action_dim={local_action_dim}, "
            f"{model_cfg.openpi.config_name}.model.action_dim="
            f"{official_action_dim}."
        )
