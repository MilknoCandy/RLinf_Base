# make sure your conda env has libvulkan-loader, vulkan-tools
export NVGL_DIR=/path/to/NVIDIA-Linux-x86_64-xxx.xxx.xx
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${NVGL_DIR}/lib:$LD_LIBRARY_PATH"

mkdir -p ${HOME}/vulkan/icd.d
cat <<EOF > ${HOME}/vulkan/icd.d/nvidia_icd.json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "${NVGL_DIR}/libEGL_nvidia.so.xxx.xxx.xx",
        "api_version": "1.3.0"
    }
}
EOF

export VK_DRIVER_FILES_PATH="${HOME}/vulkan/icd.d/nvidia_icd.json"
unset VK_ICD_FILENAMES

export XDG_RUNTIME_DIR=/tmp/xdg-runtime-${UID}
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

unsert DISPLAY

python -m sapien.example.offscreen