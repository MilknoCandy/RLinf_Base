#!/usr/bin/env bash
# verify_openpi_imports.sh — Step-by-step import verification for OpenPI models
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

check_step() {
    local desc="$1"; shift
    echo -e "${CYAN}[$(date +%H:%M:%S)] $desc${NC}"
    if "$@" 2>&1; then
        echo -e "${GREEN}  => OK${NC}"
    else
        echo -e "${RED}  => FAILED${NC}"
        return 1
    fi
}

echo "========== RLinf OpenPI Import Verification =========="

# 1) Python & repo path
check_step "1. Python runtime" python --version
check_step "2. REPO_PATH in PYTHONPATH" python -c "import rlinf; print('rlinf from', rlinf.__file__)"

# 2) openpi package (physical-intelligence)
check_step "3. import openpi (physical-intelligence)" \
    python -c "import openpi; print('openpi OK')" || true

# 3) dataconfig import — this is the common failure point
echo ""
echo -e "${YELLOW}--- dataconfig chain (most likely failure) ---${NC}"
check_step "4. import maniskill_rlt_dataconfig" \
    python -c "from rlinf.models.embodiment.openpi.dataconfig.maniskill_rlt_dataconfig import LeRobotRLTManiSkillJointDataConfig; print('RLT dataconfig OK')" || true

check_step "5. import dataconfig __init__ (get_openpi_config)" \
    python -c "from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config; print('get_openpi_config OK')" || true

# 4) openpi (JAX) get_model
echo ""
echo -e "${YELLOW}--- model factory chain ---${NC}"
check_step "6. import openpi get_model (JAX)" \
    python -c "from rlinf.models.embodiment.openpi import get_model; print('openpi get_model OK')" || true

# 5) openpi_pytorch get_model
check_step "7. import openpi_pytorch get_model (PyTorch)" \
    python -c "from rlinf.models.embodiment.openpi_pytorch import get_model; print('openpi_pytorch get_model OK')" || true

# 6) pi0_model submodules
check_step "8. import pi0_model (gemma, pi0_config, model)" \
    python -c "from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma, pi0_config, model; print('pi0_model OK')" || true

# 7) model_builders
check_step "9. import model_builders" \
    python -c "from rlinf.models.embodiment.openpi_pytorch.utils.model_builders import _build_eval_model, _build_rl_model, _build_sft_model; print('model_builders OK')" || true

# 8) List available openpi config names
echo ""
echo -e "${YELLOW}--- Available openpi TrainConfig names ---${NC}"
check_step "10. list config names" \
    python -c "
from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
print('  Available configs:')
for name in sorted(_CONFIGS_DICT.keys()):
    print(f'    - {name}')
" || true

echo ""
echo -e "${GREEN}========== Verification complete ==========${NC}"
