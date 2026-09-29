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

"""Same-episode chunk windows for stage-1 RLT memory training."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from rlinf.data.storage.lerobot.compat import episode_boundaries

logger = logging.getLogger(__name__)

_HIST_KEYS = (
    "image",
    "image_mask",
    "state",
    "tokenized_prompt",
    "tokenized_prompt_mask",
    "token_ar_mask",
    "token_loss_mask",
    "pcd_xyz",
)


def chunk_window_indices(
    anchor: int,
    episode_start: int,
    *,
    stride: int,
    length: int,
) -> tuple[list[int], list[bool]]:
    """Return ``length`` chunk indices ending at ``anchor``, oldest first.

    Indices that fall before ``episode_start`` are marked invalid. The last
    index is always ``anchor``.
    """
    if length < 1:
        raise ValueError(f"window length must be positive, got {length}.")
    if stride < 1:
        raise ValueError(f"chunk stride must be positive, got {stride}.")
    indices: list[int] = []
    valid: list[bool] = []
    for offset in range(length - 1, -1, -1):
        src = int(anchor) - offset * int(stride)
        ok = src >= int(episode_start)
        indices.append(src if ok else int(episode_start))
        valid.append(ok)
    return indices, valid


def _as_sample_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if (
        isinstance(item, (tuple, list))
        and len(item) == 2
        and isinstance(item[0], dict)
    ):
        merged = dict(item[0])
        merged["actions"] = item[1]
        return merged
    raise TypeError(
        "RLT chunk window expects a dict sample or a (observation, actions) "
        f"pair; got {type(item)!r}."
    )


def _hist_fields(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: sample[key] for key in _HIST_KEYS if key in sample}


def _stack_tree(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {key: _stack_tree([item[key] for item in items]) for key in first}
    arrays = [np.asarray(item) for item in items]
    return np.stack(arrays, axis=0)


def _map_index(index: int, maps: list[Any]) -> int:
    mapped = int(index)
    for mapping in maps:
        mapped = int(mapping[mapped])
    return mapped


def _find_episode_dataset(dataset: Any) -> tuple[Any, list[int], list[int], list[Any]]:
    """Walk wrappers until a LeRobot-style episode index is found."""
    maps: list[Any] = []
    current = dataset
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        try:
            starts, ends = episode_boundaries(current)
            return current, starts, ends, maps
        except (RuntimeError, TypeError, AttributeError, KeyError):
            pass
        indices = getattr(current, "indices", None)
        if indices is None:
            indices = getattr(current, "_indices", None)
        child = getattr(current, "_dataset", None)
        if child is None:
            child = getattr(current, "dataset", None)
        if indices is not None and child is not None:
            maps.append(indices)
        if child is None or child is current:
            break
        current = child
    raise RuntimeError(
        "Stage-1 RLT chunk window could not find episode boundaries on the "
        f"SFT dataset ({type(dataset).__name__}). The loader needs a LeRobot "
        "episode index so history stays inside one episode."
    )


class RLTChunkWindowDataset(torch.utils.data.Dataset):
    """Attach the last ``mem_len_max`` same-episode chunks to each frame.

    ``stride`` is the number of dataset rows between successive memory chunks.
    On a frame-level LeRobot episode that is one action chunk.
    """

    def __init__(self, dataset: Any, *, mem_len_max: int, stride: int):
        super().__init__()
        self._dataset = dataset
        self._mem_len_max = int(mem_len_max)
        self._stride = int(stride)
        if self._mem_len_max < 1 or self._stride < 1:
            raise ValueError(
                "mem_len_max and stride must be positive, got "
                f"mem_len_max={self._mem_len_max} stride={self._stride}."
            )
        _base, starts, ends, maps = _find_episode_dataset(dataset)
        self._index_maps = maps
        span = max(ends) if ends else 0
        frame_start = np.full(span, -1, dtype=np.int64)
        for start, end in zip(starts, ends, strict=True):
            frame_start[start:end] = start
        self._frame_start = frame_start
        self._base_to_outer: dict[int, int] | None = None
        if maps:
            inverse: dict[int, int] = {}
            for outer in range(len(dataset)):
                inverse[_map_index(outer, maps)] = outer
            self._base_to_outer = inverse

    def __len__(self) -> int:
        return len(self._dataset)

    def _outer_index(self, base_index: int) -> int:
        if self._base_to_outer is None:
            return int(base_index)
        try:
            return self._base_to_outer[int(base_index)]
        except KeyError as exc:
            raise KeyError(
                f"Base frame {base_index} is not in the SFT dataset index map."
            ) from exc

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = _as_sample_dict(self._dataset[index])
        base_index = _map_index(index, self._index_maps)
        if base_index < 0 or base_index >= len(self._frame_start):
            raise IndexError(
                f"Frame {base_index} is outside the episode index "
                f"(size {len(self._frame_start)})."
            )
        episode_start = int(self._frame_start[base_index])
        if episode_start < 0:
            raise IndexError(
                f"Frame {base_index} is not covered by any episode boundary."
            )
        indices, valid = chunk_window_indices(
            base_index,
            episode_start,
            stride=self._stride,
            length=self._mem_len_max,
        )
        anchor_fields = _hist_fields(sample)
        frames: list[dict[str, Any]] = []
        for src, ok in zip(indices, valid, strict=True):
            if not ok or src == base_index:
                frames.append(anchor_fields)
                continue
            other = _as_sample_dict(self._dataset[self._outer_index(src)])
            frames.append(_hist_fields(other))
        sample["rlt_hist"] = _stack_tree(frames)
        sample["rlt_hist_valid"] = np.asarray(valid, dtype=np.bool_)
        return sample


class RLTChunkWindowLoader:
    """Yield ``(observation, actions, history)`` from an OpenPI loader.

    The official iterator drops every key except the observation and actions.
    This wrapper reads the torch batch before that split.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        # Keep OpenPI's TorchDataLoader so batch-count helpers still walk
        # ``loader._data_loader._data_loader``.
        self._data_loader = inner._data_loader

    def data_config(self) -> Any:
        return self._inner.data_config()

    def __len__(self) -> int:
        torch_loader = getattr(self._data_loader, "_data_loader", None)
        if torch_loader is None:
            raise TypeError("OpenPI loader does not expose a torch DataLoader.")
        return len(torch_loader)

    def __iter__(self):
        for batch in self._data_loader:
            yield _split_rlt_batch(batch)


def _split_rlt_batch(batch: Any) -> dict[str, Any]:
    if not isinstance(batch, dict):
        raise TypeError(
            "RLT chunk window expected a collated dict batch; "
            f"got {type(batch)!r}."
        )
    if "rlt_hist" not in batch or "rlt_hist_valid" not in batch:
        raise KeyError(
            "SFT batch is missing rlt_hist. The chunk-window dataset was not "
            f"applied; keys={sorted(batch)}."
        )
    observation = {
        key: value
        for key, value in batch.items()
        if key not in ("actions", "rlt_hist", "rlt_hist_valid")
    }
    return {
        "observation": observation,
        "actions": batch["actions"],
        "rlt_hist": batch["rlt_hist"],
        "rlt_hist_valid": batch["rlt_hist_valid"],
    }


def _torch_loader(openpi_loader: Any) -> Any:
    mid = getattr(openpi_loader, "_data_loader", None)
    inner = getattr(mid, "_data_loader", None) if mid is not None else None
    if inner is None or not hasattr(inner, "dataset"):
        raise TypeError(
            "OpenPI SFT loader has no torch dataset to attach a chunk window to."
        )
    return inner


def _install_dataset(loader: Any, dataset: Any) -> None:
    """Swap the dataset the loader will iterate.

    PyTorch 2.8+ makes ``DataLoader.dataset`` read-only after ``__init__`` and
    stores the real object on ``_dataset``. Older loaders keep a public
    attribute.
    """
    if hasattr(loader, "_dataset"):
        loader._dataset = dataset
        return
    loader.dataset = dataset


def attach_rlt_chunk_window(
    data_loader: Any, *, mem_len_max: int, stride: int
) -> RLTChunkWindowLoader:
    """Wrap the OpenPI dataset so each sample carries a chunk window."""
    torch_loader = _torch_loader(data_loader)
    _install_dataset(
        torch_loader,
        RLTChunkWindowDataset(
            torch_loader.dataset,
            mem_len_max=mem_len_max,
            stride=stride,
        ),
    )
    logger.info(
        "Stage-1 RLT SFT loads %d same-episode chunks at stride %d.",
        int(mem_len_max),
        int(stride),
    )
    return RLTChunkWindowLoader(data_loader)
