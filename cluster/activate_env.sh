#!/usr/bin/env bash
# =============================================================================
# cluster/activate_env.sh
# Sourced by every SLURM job script to load modules and activate the conda env.
#
# Usage (in job scripts):
#   source "$(dirname "${BASH_SOURCE[0]}")/../activate_env.sh"
# =============================================================================

# Load config (sets CONDA_ENV, HF_HOME, etc.)
CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CLUSTER_DIR/cluster_config.env"

# Purge and reload modules
module purge

# CUDA — try common FarmShare versions newest-first
for MOD in "cuda/12.2.0" "cuda/12.1.1" "cuda/12.0.0" "cuda/11.8.0" "CUDA/12.2.0"; do
    module load "$MOD" 2>/dev/null && break || true
done

# Conda/Python
for MOD in "python/3.12.1" "python/3.11.0" "miniconda/4.12.0" "miniconda" "anaconda"; do
    module load "$MOD" 2>/dev/null && break || true
done

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
