#!/usr/bin/env bash
# =============================================================================
# cluster/setup.sh
# One-time environment setup for the Brain Optimisation pipeline on
# Stanford FarmShare (rice.stanford.edu).
#
# Run this ONCE from a login node:
#   ssh <sunetid>@rice.stanford.edu
#   cd ~/brain_optimisation
#   bash cluster/setup.sh
#
# What it does:
#   1. Loads system modules (Python/Conda, CUDA)
#   2. Creates a conda environment: brain_opt
#   3. Installs PyTorch with CUDA 12.x support
#   4. Installs all project requirements
#   5. Installs the tribev2 package
#   6. Pre-downloads the HCP MMP brain atlas (~50 MB, via MNE)
#   7. Creates the scratch directory layout
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="brain_opt"
PYTHON_VERSION="3.12"

# ── Scratch storage ───────────────────────────────────────────────────────────
# FarmShare scratch: /scratch/users/<sunetid>  (fast, node-accessible)
# If you have Oak storage, set OAK_SCRATCH to $OAK/brain_optimisation instead.
SCRATCH_ROOT="/scratch/users/${USER}"
HF_CACHE="${SCRATCH_ROOT}/hf_cache"
PROJECT_SCRATCH="${SCRATCH_ROOT}/brain_optimisation"
MNE_DATA="${SCRATCH_ROOT}/mne_data"

echo "============================================"
echo "Brain Optimisation — FarmShare Setup"
echo "Project dir : $PROJECT_DIR"
echo "Scratch     : $PROJECT_SCRATCH"
echo "HF cache    : $HF_CACHE"
echo "============================================"

# ── 1. Load system modules ────────────────────────────────────────────────────
# FarmShare module names. Run 'module avail' to see what is currently installed.
echo "[1/7] Loading modules..."
module purge

# Try common FarmShare module name patterns for conda/python
for MOD in "python/3.12.1" "python/3.11.0" "miniconda/4.12.0" "miniconda" "anaconda"; do
    module load "$MOD" 2>/dev/null && { echo "  Loaded: $MOD"; break; } || true
done

# Try common CUDA module name patterns
for MOD in "cuda/12.2.0" "cuda/12.1.1" "cuda/12.0.0" "cuda/11.8.0" "CUDA/12.2.0"; do
    module load "$MOD" 2>/dev/null && { echo "  Loaded: $MOD"; break; } || true
done

# Verify conda is accessible
command -v conda >/dev/null || {
    echo ""
    echo "ERROR: conda not found after module load."
    echo "Run 'module avail' on FarmShare and find the correct conda/python module,"
    echo "then update the module load lines in this script."
    echo ""
    echo "Common FarmShare commands to find the right module:"
    echo "  module avail 2>&1 | grep -i conda"
    echo "  module avail 2>&1 | grep -i python"
    exit 1
}
echo "  conda: $(conda --version)"

# ── 2. Create conda environment ───────────────────────────────────────────────
echo "[2/7] Creating conda environment: $ENV_NAME..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  '$ENV_NAME' already exists. Skipping. (Remove with: conda env remove -n $ENV_NAME)"
else
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "  Python: $(python --version)"

# ── 3. Install PyTorch ────────────────────────────────────────────────────────
# FarmShare GPU nodes typically have CUDA 12.x.
# If CUDA 11.8 is the latest available, replace cu121 → cu118 below.
echo "[3/7] Installing PyTorch (CUDA 12.1)..."
pip install --quiet \
    "torch>=2.5.1,<2.7" \
    "torchvision>=0.20,<0.22" \
    --index-url https://download.pytorch.org/whl/cu121

python -c "
import torch
print(f'  torch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# ── 4. Install project requirements ───────────────────────────────────────────
echo "[4/7] Installing requirements.txt..."
pip install --quiet -r "$PROJECT_DIR/requirements.txt"

# ── 5. Install tribev2 ────────────────────────────────────────────────────────
echo "[5/7] Installing tribev2 (with plotting extras)..."
pip install --quiet -e "$PROJECT_DIR/tribev2/.[plotting]"

# ── 6. Pre-download HCP brain atlas ───────────────────────────────────────────
echo "[6/7] Pre-downloading HCP MMP brain atlas (MNE)..."
mkdir -p "$MNE_DATA"
python - <<EOF
import os, sys, warnings
os.environ["MNE_DATA"] = "$MNE_DATA"
sys.path.insert(0, "$PROJECT_DIR/tribev2")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tribev2.utils import get_hcp_labels
    labels = get_hcp_labels(mesh="fsaverage5", hemi="both")
    print(f"  HCP atlas loaded: {len(labels)} regions")
EOF

# ── 7. Create scratch layout ──────────────────────────────────────────────────
echo "[7/7] Creating directories at $PROJECT_SCRATCH..."
mkdir -p \
    "$HF_CACHE" \
    "$PROJECT_SCRATCH/images" \
    "$PROJECT_SCRATCH/outputs" \
    "$PROJECT_SCRATCH/plots" \
    "$PROJECT_SCRATCH/checkpoints" \
    "$PROJECT_SCRATCH/generated" \
    "$PROJECT_SCRATCH/logs"

# Write config for all job scripts to source
cat > "$PROJECT_DIR/cluster/cluster_config.env" <<ENVFILE
# Auto-generated by cluster/setup.sh
# Source this at the top of every job script.
export BRAIN_OPT_PROJECT="$PROJECT_DIR"
export BRAIN_OPT_SCRATCH="$PROJECT_SCRATCH"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export MNE_DATA="$MNE_DATA"
export CONDA_ENV="$ENV_NAME"
ENVFILE

echo ""
echo "============================================"
echo "Setup complete."
echo ""
echo "Config: cluster/cluster_config.env"
echo ""
echo "Submit a job:"
echo "  bash cluster/submit_pipeline.sh /path/to/image.jpg"
echo ""
echo "Interactive session:"
echo "  salloc -p gpu --gres=gpu:1 --mem=48G --time=2:00:00"
echo "  source cluster/cluster_config.env && conda activate $ENV_NAME"
echo "============================================"
