# Example machine profile for toolkits/hitl/run_sim_libero_bowl.sh
# Copy to env.local.sh (gitignored) or export these before launch:
#
#   cp toolkits/hitl/env.example.sh toolkits/hitl/env.local.sh
#   # edit env.local.sh
#   bash toolkits/hitl/run_sim_libero_bowl.sh --env-file toolkits/hitl/env.local.sh --mode hitl --episodes 1
#
# Only fill what your machine needs. Empty / unset values keep script defaults.

# --- required for a clean conda + CUDA node ---
# export MINICONDA_PATH=/path/to/miniconda3
# export CONDA_ENV=sdw_rlinf-pi312
# export CUDA_HOME=/path/to/cuda-12.x

# --- required for headless EGL (LIBERO OffScreenRenderEnv) ---
# Directory that contains libEGL_nvidia.so.*
# export NVGL_DIR=/path/to/NVIDIA-Linux-x86_64-535.183.01
# Optional explicit soname; default auto-picks libEGL_nvidia.so.<ver> or .so.0
# export NVGL_EGL_LIB=libEGL_nvidia.so.535.183.01

# Optional OSMesa / extra GL deps prepended to LD_LIBRARY_PATH
# export OSMESA_DEP_DIR=/path/to/osmesa/lib64

# --- GPU selection ---
# export CUDA_VISIBLE_DEVICES=0
# export MUJOCO_EGL_DEVICE_ID=0

# --- repo / assets (usually auto) ---
# export REPO_PATH=/path/to/RLinf_Base
# export CODE_DEPS=/path/to/code_deps          # openpi + HF hub caches
# export HF_HUB_CACHE=/path/to/code_deps/hub

# --- mailbox root shown in Jupyter ---
# export HITL_MAILBOX_ROOT=/path/to/RLinf_Base/toolkits/hitl/live

# --- skip steps when already set up in the shell ---
# export HITL_SKIP_CONDA=1
# export HITL_SKIP_EGL=1
# export HITL_SKIP_ASSETS=1
