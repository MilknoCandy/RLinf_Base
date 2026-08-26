#! /bin/bash
#
# Serial runner for RLT Ablation 1. Switch rlt_feature_source at runtime so a
# single script can sweep sources (and both modes) without editing YAML files.
#
# Usage:
#   bash examples/embodiment/run_rlt_ablation1_serial.sh <mode> [sources]
#
# <mode>   : token | none | all   (all = run token then none)
# [sources]: space-separated list: all image language text action
#
# Examples:
#   MODEL_PATH=/path/to/stage1/actor bash examples/embodiment/run_rlt_ablation1_serial.sh token
#   MODEL_PATH=/path/to/stage1/actor bash examples/embodiment/run_rlt_ablation1_serial.sh none "image action"
#   MODEL_PATH=/path/to/stage1/actor bash examples/embodiment/run_rlt_ablation1_serial.sh all "all image"

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-"LIBERO"}

MODE="${1:-all}"
SOURCES="${2:-all image language action}"
MODEL_PATH="${MODEL_PATH:-}"

config_for_mode() {
  case "$1" in
    token) echo "maniskill_rlt_stage2_ablation1_rlt_token" ;;
    none)  echo "maniskill_rlt_stage2_ablation1_no_rlt" ;;
    *)     return 1 ;;
  esac
}

run_mode() {
  local mode="$1"
  local config_name
  config_name=$(config_for_mode "$mode")

  local model_override=""
  if [ -n "$MODEL_PATH" ]; then
    model_override=" rollout.rlt_feature_model.model_path=${MODEL_PATH}"
  fi

  for src in $SOURCES; do
    local stamp
    stamp=$(date +'%Y%m%d-%H%M%S')
    local log_dir="${REPO_PATH}/logs/${stamp}-${config_name}-${src}"
    local log_file="${log_dir}/run.log"
    mkdir -p "$log_dir"

    local cmd="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config --config-name ${config_name} rollout.rlt_feature_model.openpi.rlt_feature_source=${src} runner.logger.experiment_name=${config_name}_${src} runner.logger.log_path=${log_dir}${model_override}"

    echo "[serial-run] mode=${mode} source=${src}" | tee -a "$log_file"
    echo "$cmd" | tee -a "$log_file"
    $cmd 2>&1 | tee -a "$log_file"
  done
}

case "$MODE" in
  token|none)
    run_mode "$MODE"
    ;;
  all)
    run_mode token
    run_mode none
    ;;
  *)
    echo "unknown mode: ${MODE} (expected token|none|all)" >&2
    exit 1
    ;;
esac
