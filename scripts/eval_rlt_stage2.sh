#!/usr/bin/env bash
# =============================================================================
# RLT Stage 2 Evaluation Script
#
# Usage:
#   bash scripts/eval_rlt_stage2.sh <CKPT_PATH> [STAGE1_PATH] [EXPERT_PATH]
#
#   CKPT_PATH   – Path to Stage2 checkpoint (required)
#                  e.g. /path/to/results/exp/checkpoints/global_step_500/actor
#   STAGE1_PATH – Path to Stage1 RLT feature model (optional, fallback to YAML)
#   EXPERT_PATH – Path to expert takeover model (optional, fallback to YAML)
#
# Examples:
#   # Minimal: only Stage2 ckpt, paths come from YAML
#   bash scripts/eval_rlt_stage2.sh /path/to/checkpoints/global_step_500/actor
#
#   # Full: override all model paths
#   bash scripts/eval_rlt_stage2.sh \
#       /path/to/stage2/checkpoints/global_step_500/actor \
#       /path/to/stage1/checkpoints/global_step_2000/actor \
#       /path/to/sft_model
#
# Env vars (optional):
#   TOTAL_NUM_ENVS      – Number of parallel eval envs (default: 256)
#   ROLLOUT_EPOCH       – Number of eval rollout epochs (default: 10)
#   MAX_EPISODE_STEPS   – Max steps per episode (default: 500)
#   SAVE_VIDEO          – Save mp4 videos (default: true)
#   CONFIG_NAME         – Hydra config name (default: maniskill_rlt_stage2_ac_mlp)
# =============================================================================
set -euo pipefail

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$SCRIPT_DIR")"
EVAL_ENTRY="${REPO_PATH}/evaluations/eval_embodied_agent.py"
CONFIG_PATH="${REPO_PATH}/examples/embodiment/config"

# --- Arguments ---
CKPT_PATH="${1:-}"
if [ -z "${CKPT_PATH}" ]; then
    echo "Usage: $0 <CKPT_PATH> [STAGE1_PATH] [EXPERT_PATH]"
    echo ""
    echo "  CKPT_PATH   – Path to Stage2 checkpoint (required)"
    echo "  STAGE1_PATH – Path to Stage1 RLT feature model (optional)"
    echo "  EXPERT_PATH – Path to expert takeover model (optional)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/checkpoints/global_step_500/actor"
    exit 1
fi

STAGE1_PATH="${2:-}"
EXPERT_PATH="${3:-}"

# --- Tunables ---
CONFIG_NAME="${CONFIG_NAME:-maniskill_rlt_stage2_ac_mlp}"
TOTAL_NUM_ENVS="${TOTAL_NUM_ENVS:-256}"
ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-10}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"

# --- Simulation env vars ---
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# --- Misc ---
TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
LOG_DIR="${REPO_PATH}/logs/eval/${TIMESTAMP}-rlt_stage2"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval.log"

# --- Build Hydra overrides ---
OVERRIDES=(
    "runner.ckpt_path=${CKPT_PATH}"
    "env.eval.total_num_envs=${TOTAL_NUM_ENVS}"
    "env.eval.rollout_epoch=${ROLLOUT_EPOCH}"
    "env.eval.max_episode_steps=${MAX_EPISODE_STEPS}"
    "env.eval.video_cfg.save_video=${SAVE_VIDEO}"
    "env.eval.video_cfg.video_base_dir=${LOG_DIR}/video/eval"
    "runner.logger.log_path=${LOG_DIR}"
    "runner.logger.experiment_name=rlt_stage2_eval"
)

# Override Stage1 feature model path if provided
if [ -n "${STAGE1_PATH}" ]; then
    OVERRIDES+=("rollout.rlt_feature_model.model_path=${STAGE1_PATH}")
fi

# Override expert model path if provided
if [ -n "${EXPERT_PATH}" ]; then
    OVERRIDES+=("rollout.expert_model.model_path=${EXPERT_PATH}")
fi

# --- Print config ---
echo "============================================"
echo "  RLT Stage 2 Evaluation"
echo "============================================"
echo "  Config:       ${CONFIG_NAME}"
echo "  Checkpoint:   ${CKPT_PATH}"
echo "  Stage1 model: ${STAGE1_PATH:-'(from YAML)'}"
echo "  Expert model: ${EXPERT_PATH:-'(from YAML)'}"
echo "  Envs:         ${TOTAL_NUM_ENVS}"
echo "  Epochs:       ${ROLLOUT_EPOCH}"
echo "  Max steps:    ${MAX_EPISODE_STEPS}"
echo "  Save video:   ${SAVE_VIDEO}"
echo "  Log dir:      ${LOG_DIR}"
echo "============================================"

# --- Run ---
CMD=(
    python "${EVAL_ENTRY}"
    --config-path "${CONFIG_PATH}"
    --config-name "${CONFIG_NAME}"
    "${OVERRIDES[@]}"
)

echo ""
echo "Command:"
printf '  %s\n' "${CMD[@]}"
echo ""

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

# --- Summary ---
echo ""
echo "============================================"
echo "  Evaluation complete!"
echo "  Log:    ${LOG_FILE}"
if [ "${SAVE_VIDEO}" = "true" ]; then
    echo "  Videos: ${LOG_DIR}/video/eval/"
fi
echo "============================================"

