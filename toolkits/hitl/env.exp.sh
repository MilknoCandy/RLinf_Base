# Exp-node profile for toolkits/hitl/run_sim_libero_bowl.sh
# Used by run_sim_libero_bowl_exp.sh. Override any var before sourcing.

export RLinf_HOME="${RLinf_HOME:-/home/ma-user/work/algorithm/sdw_calvin/RLinf_Base}"
export RLinf_DATAROOT="${RLinf_DATAROOT:-/home/ma-user/work/dataset/recaption60M/Datasets_for_Intern/VLA_datasets}"
export RLinf_OUTPUTS="${RLinf_OUTPUTS:-/home/ma-user/work/model/xiaoyi_tmpstorage/sdw_files}"
export RLinf_PRETRAINMODELS="${RLinf_PRETRAINMODELS:-${RLinf_OUTPUTS}/local_models}"

export REPO_PATH="${REPO_PATH:-${RLinf_HOME}}"
export MINICONDA_PATH="${MINICONDA_PATH:-/home/ma-user/work/dataset/Common_wl/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-sdw_rlinf-pi312}"

export CUDA_HOME="${CUDA_HOME:-/home/ma-user/work/dataset/Common_wl/pkgs_llb/cuda-12.8}"
export CUDNN_HOME="${CUDNN_HOME:-/home/ma-user/work/dataset/Common_wl/pkgs_llb/cudnn}"
export NCCL_HOME="${NCCL_HOME:-/home/ma-user/work/dataset/Common_wl/pkgs_llb/nccl}"

export CODE_DEPS="${CODE_DEPS:-${RLinf_OUTPUTS}/code_deps}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CODE_DEPS}/hub}"

export NVGL_DIR="${NVGL_DIR:-/home/ma-user/work/dataset/Common_wl/pkgs_llb/NVIDIA-Linux-x86_64-535.183.01}"
export NVGL_EGL_LIB="${NVGL_EGL_LIB:-libEGL_nvidia.so.535.183.01}"
export OSMESA_DEP_DIR="${OSMESA_DEP_DIR:-/home/ma-user/work/dataset/trellis_ckpt/blender_envs/usr/lib64}"

# Leave GPU 0 to a running actor; env workers on exp sit on GPU 1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

export HITL_MAILBOX_ROOT="${HITL_MAILBOX_ROOT:-${REPO_PATH}/toolkits/hitl/live}"
