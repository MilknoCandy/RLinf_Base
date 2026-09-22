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

"""B2 execution-feedback sentences from privileged env state."""

from __future__ import annotations

from typing import Any

import torch

B2_FEEDBACK_INFO_KEYS = (
    "success_current",
    "success",
    "peg_head_hole_x",
    "peg_head_hole_abs_y",
    "peg_head_hole_abs_z",
)


def select_b2_env_infos(env_infos: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the privileged fields needed to write B2 feedback sentences."""
    if not env_infos:
        return None
    selected = {
        key: env_infos[key] for key in B2_FEEDBACK_INFO_KEYS if key in env_infos
    }
    return selected or None

_DISTANCE_EPS = 1.0e-4


def peg_insertion_distance(
    hole_x: torch.Tensor,
    abs_y: torch.Tensor,
    abs_z: torch.Tensor,
) -> torch.Tensor:
    """Scalar distance: smaller means closer to a successful insertion."""
    yz = torch.sqrt(abs_y.float() ** 2 + abs_z.float() ** 2)
    return yz - hole_x.float()


def format_peg_insertion_feedback(
    *,
    success: torch.Tensor,
    delta_distance: torch.Tensor,
    off_axis: torch.Tensor | None = None,
    distance_eps: float = _DISTANCE_EPS,
) -> list[str]:
    """Write a short result-change sentence for each batch item.

    The sentence always includes success/failure and closer/farther. It is
    intended for the next VLM prefix, not as encoder input tokens.
    """
    success_flat = success.reshape(-1).to(dtype=torch.bool)
    delta_flat = delta_distance.reshape(-1).to(dtype=torch.float32)
    if success_flat.numel() != delta_flat.numel():
        raise ValueError(
            "success and delta_distance must have the same batch size, got "
            f"{success_flat.numel()} and {delta_flat.numel()}."
        )
    off_axis_flat = None
    if off_axis is not None:
        off_axis_flat = off_axis.reshape(-1).to(dtype=torch.bool)
        if off_axis_flat.numel() != success_flat.numel():
            raise ValueError(
                "off_axis must match the success batch size, got "
                f"{off_axis_flat.numel()} and {success_flat.numel()}."
            )

    sentences: list[str] = []
    for index in range(success_flat.numel()):
        if bool(success_flat[index]):
            result = "Insertion succeeded."
        else:
            result = "Insertion has not succeeded."
        delta = float(delta_flat[index])
        if delta < -distance_eps:
            motion = "The peg moved closer to the hole"
        elif delta > distance_eps:
            motion = "The peg moved farther from the hole"
        else:
            motion = "The peg distance to the hole is unchanged"
        if off_axis_flat is not None and bool(off_axis_flat[index]):
            sentences.append(f"{result} {motion}, but is still slightly off-axis.")
        else:
            sentences.append(f"{result} {motion}.")
    return sentences


def _as_bool_1d(value: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.to(device=device).reshape(-1)
    if tensor.numel() == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(
            f"Expected {batch_size} feedback flags, got {tensor.numel()}."
        )
    return tensor.to(dtype=torch.bool)


def _as_float_1d(value: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, dtype=torch.float32, device=device)
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(
            f"Expected {batch_size} feedback scalars, got {tensor.numel()}."
        )
    return tensor


def distance_from_env_infos(
    env_infos: dict[str, Any] | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return peg-insertion distance if the required info keys are present."""
    if not env_infos:
        return None
    if "peg_head_hole_x" not in env_infos:
        return None
    hole_x = _as_float_1d(env_infos["peg_head_hole_x"], batch_size, device)
    abs_y = _as_float_1d(
        env_infos.get("peg_head_hole_abs_y"), batch_size, device
    )
    abs_z = _as_float_1d(
        env_infos.get("peg_head_hole_abs_z"), batch_size, device
    )
    return peg_insertion_distance(hole_x, abs_y, abs_z)


def success_from_env_infos(
    env_infos: dict[str, Any] | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Read per-env success, preferring ``success_current`` then ``success``."""
    if not env_infos:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    raw = env_infos.get("success_current", env_infos.get("success"))
    return _as_bool_1d(raw, batch_size, device)


def append_feedback_to_prompts(
    prompts: list[str],
    feedback_sentences: list[str] | None,
) -> list[str]:
    """Append a result-change sentence to each instruction without merging them."""
    if not feedback_sentences:
        return list(prompts)
    if len(feedback_sentences) != len(prompts):
        raise ValueError(
            "feedback_sentences must match the prompt batch size, got "
            f"{len(feedback_sentences)} and {len(prompts)}."
        )
    merged = []
    for prompt, feedback in zip(prompts, feedback_sentences):
        text = str(prompt).strip()
        extra = str(feedback).strip()
        merged.append(f"{text} {extra}".strip() if extra else text)
    return merged
