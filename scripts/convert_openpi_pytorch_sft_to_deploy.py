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

"""Convert an official OpenPI PyTorch SFT checkpoint to deploy weights.

This is intended for RLinf SFT runs using ``model/pi0_5`` or ``model/pi0``
(the ``OpenPi0ForRLActionPrediction`` model). The saved
``actor/model_state_dict/full_weights.pt`` contains ``paligemma_with_expert.*``
keys with optional FSDP wrapper prefixes. The standalone OpenPI eval script
(``toolkits/standalone_eval_scripts/openpi/calvin_eval.py``) loads a
``model.safetensors`` checkpoint instead, so this script strips wrapper
prefixes and writes the expected deploy layout.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import safetensors.torch
import torch

from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict, strip_wrapper_prefix


_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)


def resolve_full_weights(ckpt: str | pathlib.Path) -> pathlib.Path:
    """Find the consolidated ``full_weights.pt`` from a file or directory."""
    ckpt = pathlib.Path(ckpt)
    if ckpt.is_file():
        return ckpt
    for relative_path in _WEIGHTS_CANDIDATES:
        candidate = ckpt / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No full_weights.pt found under {ckpt}; looked at "
        f"{[str(ckpt / p) for p in _WEIGHTS_CANDIDATES]}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        required=True,
        help=(
            "SFT checkpoint: a global_step_* directory, an actor directory, "
            "a model_state_dict directory, or a full_weights.pt file"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output deploy directory that will contain model.safetensors",
    )
    parser.add_argument(
        "--norm-stats",
        required=True,
        help="input norm_stats.json to copy into the deploy directory",
    )
    parser.add_argument(
        "--asset-id",
        default="InternRobotics/InternData-Calvin_ABC",
        help=(
            "OpenPI asset id used by the eval loader to locate norm_stats.json; "
            "defaults to the pi05_calvin/pi0_calvin repo id"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="storage dtype for model.safetensors (default: bf16)",
    )
    args = parser.parse_args()

    weights_path = resolve_full_weights(args.ckpt)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = torch.load(
        str(weights_path), map_location="cpu", weights_only=False, mmap=True
    )
    state_dict = as_state_dict(loaded)
    print(f"Loaded {weights_path} ({len(state_dict)} tensors).")
    print("First keys:")
    for key in list(state_dict)[:20]:
        print(f"  {key}")

    cast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    bare = strip_wrapper_prefix(state_dict, cast_dtype=cast_dtype)
    model_path = output_dir / "model.safetensors"
    safetensors.torch.save_file(bare, str(model_path))
    print(f"Wrote {model_path} ({len(bare)} tensors, {args.dtype}).")

    norm_stats_path = pathlib.Path(args.norm_stats)
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"norm stats not found: {norm_stats_path}")
    norm_stats_out = output_dir / args.asset_id / "norm_stats.json"
    norm_stats_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(norm_stats_path, norm_stats_out)
    print(f"Copied norm stats to {norm_stats_out}.")


if __name__ == "__main__":
    main()
