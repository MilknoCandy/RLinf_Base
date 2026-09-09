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

"""Standalone CALVIN evaluation for an RLT Stage-2 actor.

Unlike :mod:`toolkits.standalone_eval_scripts.openpi.calvin_eval`, this script
loads *two* models:

* ``rollout.rlt_feature_model`` (``openpi`` or ``openpi_rlt_probe``), which
  produces ``z_rl`` / ``proprio`` / ``ref_chunk`` via ``extract_rlt_obs``.
* ``actor.model`` (``rlt_mlp_policy``), the small Stage-2 head that predicts the
  action chunk from those RLT features.

It reuses the exact Stage-2 training config through Hydra so the RLT feature
hyperparameters stay in sync with Stage 1. Run from the repo root::

    python toolkits/standalone_eval_scripts/openpi/calvin_rlt_eval.py \\
        --config-name calvin_rlt_stage2_ac_mlp \\
        runner.resume_dir=/path/to/.../checkpoints/global_step_<N> \\
        rollout.rlt_feature_model.model_path=/path/to/calvin_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_<step>/actor \\
        rollout.rlt_feature_model.openpi_data.repo_id=/path/to/InternData-Calvin_ABC \\
        rollout.rlt_feature_model.openpi_data.norm_stats_path=/path/to/InternData-Calvin_ABC/norm_stats.json \\
        +eval.num_trials=1000 +eval.max_steps=480
"""

from __future__ import annotations

import collections
import copy
import os
import pathlib
import sys

# Make ``rlinf`` and ``toolkits`` importable when this file is launched
# directly, without relying on the caller having set PYTHONPATH.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault(
    "EMBODIED_PATH",
    os.path.join(_REPO_ROOT, "examples", "embodiment"),
)

import hydra  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tqdm  # noqa: E402
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition  # noqa: E402
from calvin_env.envs.play_table_env import get_env  # noqa: E402

from rlinf.config import validate_cfg  # noqa: E402
from rlinf.envs.calvin import ENV_CFG_DIR, _get_calvin_tasks_and_reward  # noqa: E402
from rlinf.models import get_model  # noqa: E402
from toolkits.standalone_eval_scripts.openpi import setup_logger  # noqa: E402


def _print_performance(logger, episode_solved_subtasks, per_subtask_success):
    logger.info("#####################################################")
    logger.info(f"Avg solved subtasks: {np.mean(episode_solved_subtasks)}\n")

    logger.info("Per sequence_length avg success:")
    for i in range(1, 6):
        logger.info(
            f"{i}: {np.sum(np.array(episode_solved_subtasks) >= i) / len(episode_solved_subtasks) * 100}%"
        )

    logger.info("\n Per subtask avg success:")
    for key in per_subtask_success:
        logger.info(f"{key}: \t\t\t {np.mean(per_subtask_success[key]) * 100}%")
    logger.info("#####################################################")


def _build_actor_from_checkpoint(cfg, device):
    actor_cfg = copy.deepcopy(cfg.actor.model)
    actor = get_model(actor_cfg)
    actor = actor.to(device)

    resume_dir = cfg.runner.get("resume_dir", None)
    ckpt_path = cfg.runner.get("ckpt_path", None)
    if ckpt_path:
        weights_path = ckpt_path
    elif resume_dir:
        weights_path = os.path.join(
            resume_dir, "actor", "model_state_dict", "full_weights.pt"
        )
    else:
        raise ValueError(
            "Provide either runner.resume_dir (checkpoints/global_step_<N>) or "
            "runner.ckpt_path (path to the MLP full_weights.pt)."
        )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"RLT Stage-2 actor weights not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()
    return actor


def _build_feature_model(cfg, device):
    feature_cfg = copy.deepcopy(cfg.rollout.rlt_feature_model)
    feature_model = get_model(feature_cfg)
    feature_model = feature_model.to(device)
    feature_model.eval()
    feature_model.requires_grad_(False)
    return feature_model


def _env_obs_from_calvin(raw_obs, prompt):
    img = np.asarray(raw_obs["rgb_obs"]["rgb_static"])
    wrist_img = np.asarray(raw_obs["rgb_obs"]["rgb_gripper"])
    state = np.asarray(raw_obs["robot_obs"][:7], dtype=np.float32)
    return {
        "main_images": torch.from_numpy(img).unsqueeze(0),
        "wrist_images": torch.from_numpy(wrist_img).unsqueeze(0),
        "states": torch.from_numpy(state).unsqueeze(0),
        "task_descriptions": [prompt],
        "extra_view_images": None,
    }


def main(cfg):
    cfg.runner.task_type = "embodied_eval"
    cfg = validate_cfg(cfg)

    eval_cfg = cfg.get("eval", {}) or {}
    num_trials = int(eval_cfg.get("num_trials", 1000))
    max_steps = int(eval_cfg.get("max_steps", 480))
    action_chunk = int(eval_cfg.get("action_chunk", cfg.actor.model.num_action_chunks))
    num_save_videos = int(eval_cfg.get("num_save_videos", 10))
    video_temp_subsample = int(eval_cfg.get("video_temp_subsample", 10))
    exp_name = str(eval_cfg.get("exp_name", cfg.runner.logger.experiment_name))
    log_dir = str(eval_cfg.get("log_dir", "logs"))

    logger = setup_logger(exp_name, log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Building RLT feature model...")
    feature_model = _build_feature_model(cfg, device)
    logger.info("Building RLT Stage-2 actor...")
    actor = _build_actor_from_checkpoint(cfg, device)
    logger.info("Models ready.")

    env = get_env(ENV_CFG_DIR, show_gui=False)
    task_definitions, task_instructions, task_reward = _get_calvin_tasks_and_reward(
        num_trials
    )

    episode_solved_subtasks = []
    per_subtask_success = collections.defaultdict(list)
    for i, (initial_state, task_sequence) in enumerate(tqdm.tqdm(task_definitions)):
        logger.info(f"Starting episode {i + 1}...")
        logger.info(f"Task sequence: {task_sequence}")
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        rollout_images = []
        solved_subtasks = 0
        for subtask in task_sequence:
            start_info = env.get_info()
            action_plan = collections.deque()
            obs = env.get_obs()
            done = False
            for _ in range(max_steps):
                img = obs["rgb_obs"]["rgb_static"]
                rollout_images.append(img)

                if not action_plan:
                    prompt = str(task_instructions[subtask][0])
                    env_obs = _env_obs_from_calvin(obs, prompt)
                    rlt_obs = feature_model.extract_rlt_obs(env_obs)
                    chunk_actions, _ = actor.predict_action_batch(
                        env_obs=rlt_obs,
                        mode="eval",
                    )
                    chunk_actions = chunk_actions[0].detach().cpu().numpy()
                    assert len(chunk_actions) >= action_chunk, (
                        f"Replan every {action_chunk} steps but actor predicted "
                        f"{len(chunk_actions)}."
                    )
                    action_plan.extend(chunk_actions[:action_chunk])

                action = action_plan.popleft().copy()
                action[-1] = 1 if action[-1] > 0 else -1
                obs, _, _, current_info = env.step(action)

                current_task_info = task_reward.get_task_info_for_set(
                    start_info, current_info, {subtask}
                )
                if len(current_task_info) > 0:
                    done = True
                    solved_subtasks += 1
                    break

            per_subtask_success[subtask].append(int(done))
            if not done:
                break

        episode_solved_subtasks.append(solved_subtasks)
        if len(episode_solved_subtasks) <= num_save_videos:
            idx = len(episode_solved_subtasks)
            is_success = solved_subtasks == len(task_sequence)
            suffix = "success" if is_success else "failure"
            out_path = (
                pathlib.Path(f"{log_dir}/{exp_name}/")
                / f"rollout_{idx}_{suffix}.mp4"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                out_path,
                [np.asarray(x) for x in rollout_images[::video_temp_subsample]],
                fps=50 // video_temp_subsample,
            )

        logger.info(f"Solved subtasks: {solved_subtasks}")
        _print_performance(logger, episode_solved_subtasks, per_subtask_success)

    env.close()

    logger.info(f"results/avg_num_subtasks: {np.mean(episode_solved_subtasks)}")
    for i in range(1, 6):
        logger.info(
            f"results/avg_success_len_{i}: {np.sum(np.array(episode_solved_subtasks) >= i) / len(episode_solved_subtasks)}"
        )
    for key in per_subtask_success:
        logger.info(f"results/avg_success__{key}: {np.mean(per_subtask_success[key])}")


@hydra.main(
    version_base="1.1",
    config_path=os.path.abspath(
        os.path.join(_REPO_ROOT, "examples", "embodiment", "config")
    ),
    config_name="calvin_rlt_stage2_ac_mlp",
)
def _hydra_entry(cfg) -> None:
    main(cfg)


if __name__ == "__main__":
    _hydra_entry()
