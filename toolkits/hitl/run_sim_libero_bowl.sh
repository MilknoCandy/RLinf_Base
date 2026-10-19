#!/usr/bin/env bash
# Generic LIBERO-Pro Bowl HITL / smoke launcher. No Ray, no VLA, no training.
#
# Fill machine-specific paths via env vars or ``--env-file``, then pass the
# usual Python flags (``--mode hitl|smoke``, ``--episodes``, ``--root``, …).
#
# Examples:
#   # Already in the right conda env; only EGL path needed:
#   export NVGL_DIR=/path/to/NVIDIA-Linux-x86_64-...
#   export HITL_SKIP_CONDA=1
#   bash toolkits/hitl/run_sim_libero_bowl.sh --mode hitl --episodes 1
#
#   # Full profile:
#   bash toolkits/hitl/run_sim_libero_bowl.sh \
#     --env-file toolkits/hitl/env.local.sh \
#     --mode hitl --episodes 1 --root toolkits/hitl/live
#
# Exp node convenience wrapper: toolkits/hitl/run_sim_libero_bowl_exp.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_DEFAULT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"

_usage() {
    cat <<'EOF'
Generic LIBERO-Pro Bowl HITL / smoke launcher. No Ray, no VLA, no training.

Fill machine-specific paths via env vars or --env-file, then pass the usual
Python flags (--mode hitl|smoke, --episodes, --root, ...).

Examples:
  export NVGL_DIR=/path/to/NVIDIA-Linux-x86_64-...
  export HITL_SKIP_CONDA=1
  bash toolkits/hitl/run_sim_libero_bowl.sh --mode hitl --episodes 1

  bash toolkits/hitl/run_sim_libero_bowl.sh \
    --env-file toolkits/hitl/env.local.sh \
    --mode hitl --episodes 1

Exp wrapper: toolkits/hitl/run_sim_libero_bowl_exp.sh

Shell options (parsed here; everything else goes to test_libero_bowl_sim.py):
  --env-file PATH   Source a machine profile (also: HITL_ENV)
  --help            Show this help

Key environment variables:
  REPO_PATH              Repo root (default: auto from this script)
  MINICONDA_PATH         Miniconda root (skip with HITL_SKIP_CONDA=1)
  CONDA_ENV              Conda env name (default: sdw_rlinf-pi312)
  CUDA_HOME              Optional CUDA toolkit
  NVGL_DIR               Dir with libEGL_nvidia.so.* (needed unless HITL_SKIP_EGL=1)
  NVGL_EGL_LIB           Explicit EGL soname under NVGL_DIR (optional auto-detect)
  CUDA_VISIBLE_DEVICES   Default 0
  MUJOCO_EGL_DEVICE_ID   Default 0
  HITL_MAILBOX_ROOT      Default mailbox dir (passed as --root if you omit --root)
  HITL_SKIP_CONDA=1      Do not conda activate
  HITL_SKIP_EGL=1        Do not write EGL/Vulkan ICD JSON
  HITL_SKIP_ASSETS=1     Do not run fix_libero_assets.sh
EOF
}

_ENV_FILE="${HITL_ENV:-}"
_PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            _ENV_FILE="${2:?--env-file needs a path}"
            shift 2
            ;;
        --help|-h)
            _usage
            exit 0
            ;;
        *)
            _PY_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "${_ENV_FILE}" ]]; then
    if [[ ! -f "${_ENV_FILE}" ]]; then
        echo "[hitl] ERROR: env file not found: ${_ENV_FILE}" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${_ENV_FILE}"
    echo "[hitl] sourced env file: ${_ENV_FILE}"
fi

export REPO_PATH="${REPO_PATH:-${_REPO_DEFAULT}}"
export EMBODIED_PATH="${EMBODIED_PATH:-${REPO_PATH}/examples/embodiment}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LIBERO_TYPE="${LIBERO_TYPE:-pro}"
export LIBERO_PERTURBATION="${LIBERO_PERTURBATION:-all}"

cd "${REPO_PATH}"

# ------------------------------------------------------------
# Conda (optional)
# ------------------------------------------------------------
if [[ "${HITL_SKIP_CONDA:-0}" != "1" ]]; then
    if [[ -z "${MINICONDA_PATH:-}" ]]; then
        echo "[hitl] ERROR: set MINICONDA_PATH or HITL_SKIP_CONDA=1" >&2
        exit 1
    fi
    export PATH="${MINICONDA_PATH}/bin:${PATH}"
    # shellcheck source=/dev/null
    source "${MINICONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-sdw_rlinf-pi312}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# ------------------------------------------------------------
# CUDA toolkit libs (optional)
# ------------------------------------------------------------
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${CUDNN_HOME:-}" ]]; then
    export CPATH="${CUDNN_HOME}/include:${CPATH:-}"
    export LIBRARY_PATH="${CUDNN_HOME}/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${NCCL_HOME:-}" ]]; then
    export CPATH="${NCCL_HOME}/include:${CPATH:-}"
    export LIBRARY_PATH="${NCCL_HOME}/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

# ------------------------------------------------------------
# OpenPI / HF caches + libero assets
# ------------------------------------------------------------
if [[ -n "${CODE_DEPS:-}" ]]; then
    export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${CODE_DEPS}/openpi}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${CODE_DEPS}/hub}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
fi

if [[ "${HITL_SKIP_ASSETS:-0}" != "1" ]]; then
    if [[ -f "${REPO_PATH}/scripts/liberopro/fix_libero_assets.sh" ]]; then
        # shellcheck source=/dev/null
        source "${REPO_PATH}/scripts/liberopro/fix_libero_assets.sh"
        fix_libero_and_liberopro_assets || exit 1
    else
        echo "[hitl] WARN: fix_libero_assets.sh missing; continuing" >&2
    fi
fi

# ------------------------------------------------------------
# Headless EGL / Vulkan ICD
# ------------------------------------------------------------
export RENDER_BACKEND="${RENDER_BACKEND:-egl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

_resolve_egl_lib() {
    local dir="$1"
    if [[ -n "${NVGL_EGL_LIB:-}" ]]; then
        echo "${NVGL_EGL_LIB}"
        return 0
    fi
    local candidate
    # Prefer a versioned soname, then .so.0, then any match.
    for candidate in "${dir}"/libEGL_nvidia.so.[0-9]* "${dir}"/libEGL_nvidia.so.0; do
        if [[ -f "${candidate}" ]]; then
            basename "${candidate}"
            return 0
        fi
    done
    return 1
}

if [[ "${HITL_SKIP_EGL:-0}" != "1" ]]; then
    if [[ -z "${NVGL_DIR:-}" ]]; then
        echo "[hitl] ERROR: set NVGL_DIR (NVIDIA driver extract dir) or HITL_SKIP_EGL=1" >&2
        exit 1
    fi
    if [[ ! -d "${NVGL_DIR}" ]]; then
        echo "[hitl] ERROR: NVGL_DIR is not a directory: ${NVGL_DIR}" >&2
        exit 1
    fi
    _EGL_LIB="$(_resolve_egl_lib "${NVGL_DIR}")" || {
        echo "[hitl] ERROR: no libEGL_nvidia.so.* under ${NVGL_DIR}" >&2
        exit 1
    }
    if [[ ! -f "${NVGL_DIR}/${_EGL_LIB}" ]]; then
        echo "[hitl] ERROR: missing NVIDIA EGL lib: ${NVGL_DIR}/${_EGL_LIB}" >&2
        exit 1
    fi
    export NVGL_EGL_LIB="${_EGL_LIB}"

    _LD_PREPEND="${NVGL_DIR}"
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        _LD_PREPEND="${CONDA_PREFIX}/lib:${_LD_PREPEND}"
    fi
    if [[ -n "${OSMESA_DEP_DIR:-}" ]]; then
        _LD_PREPEND="${_LD_PREPEND}:${OSMESA_DEP_DIR}"
    fi
    export LD_LIBRARY_PATH="${_LD_PREPEND}:${LD_LIBRARY_PATH:-}"

    EGL_VENDOR_DIR="${EGL_VENDOR_DIR:-${HOME}/.rlinf_egl_vendor.d}"
    mkdir -p "${EGL_VENDOR_DIR}"
    cat > "${EGL_VENDOR_DIR}/10_nvidia.json" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${NVGL_DIR}/${NVGL_EGL_LIB}"
  }
}
EOF
    export __EGL_VENDOR_LIBRARY_DIRS="${EGL_VENDOR_DIR}"

    mkdir -p "${HOME}/vulkan/icd.d"
    cat > "${HOME}/vulkan/icd.d/nvidia_icd.json" <<EOF
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "${NVGL_DIR}/${NVGL_EGL_LIB}",
        "api_version": "1.3.0"
    }
}
EOF
    export VK_DRIVER_FILES="${HOME}/vulkan/icd.d/nvidia_icd.json"
    unset VK_ICD_FILENAMES || true
fi

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-${UID:-$(id -u)}}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}" || true
unset DISPLAY || true

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

# Default mailbox root when the caller did not pass --root.
_HAS_ROOT=0
for _arg in "${_PY_ARGS[@]+"${_PY_ARGS[@]}"}"; do
    if [[ "${_arg}" == "--root" || "${_arg}" == --root=* ]]; then
        _HAS_ROOT=1
        break
    fi
done
if [[ "${_HAS_ROOT}" -eq 0 ]]; then
    _MAILBOX="${HITL_MAILBOX_ROOT:-${REPO_PATH}/toolkits/hitl/live}"
    mkdir -p "${_MAILBOX}"
    _PY_ARGS=(--root "${_MAILBOX}" "${_PY_ARGS[@]+"${_PY_ARGS[@]}"}")
fi

echo "[hitl] python=$(command -v python)"
echo "[hitl] REPO_PATH=${REPO_PATH}"
echo "[hitl] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "[hitl] MUJOCO_GL=${MUJOCO_GL} __EGL_VENDOR_LIBRARY_DIRS=${__EGL_VENDOR_LIBRARY_DIRS:-}"
echo "[hitl] LIBERO_TYPE=${LIBERO_TYPE} LIBERO_PERTURBATION=${LIBERO_PERTURBATION}"
echo "[hitl] args: ${_PY_ARGS[*]}"

exec python "${REPO_PATH}/toolkits/hitl/test_libero_bowl_sim.py" "${_PY_ARGS[@]}"
