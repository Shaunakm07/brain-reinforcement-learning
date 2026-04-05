#!/usr/bin/env bash
#SBATCH --job-name=brain-plots
#SBATCH --partition=normal
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/users/%u/brain_optimisation/logs/%x_%j.out
#SBATCH --error=/scratch/users/%u/brain_optimisation/logs/%x_%j.err
# =============================================================================
# Run plot_brain.py + brain_regions.py + plot_regions.py (CPU only).
# Submit: sbatch cluster/sbatch/plots.sh /path/to/brain_response.npy [name]
# =============================================================================
set -euo pipefail

source "$HOME/brain_optimisation/cluster/activate_env.sh"

NPY_PATH="${1:?Usage: sbatch plots.sh /path/to/brain_response.npy [name]}"
NAME="${2:-$(basename "$NPY_PATH" .npy)}"
PLOT_DIR="$BRAIN_OPT_SCRATCH/plots/$NAME"

echo "Job    : $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Input  : $NPY_PATH"
echo "Output : $PLOT_DIR"
mkdir -p "$PLOT_DIR"

cd "$BRAIN_OPT_PROJECT"

echo "--- plot_brain.py ---"
python plot_brain.py "$NPY_PATH" "$PLOT_DIR"

echo "--- brain_regions.py ---"
python brain_regions.py "$NPY_PATH" 30 | tee "$PLOT_DIR/region_rankings.txt"

echo "--- plot_regions.py ---"
python plot_regions.py "$NPY_PATH" "$PLOT_DIR"

echo "Done. Files in $PLOT_DIR:"
ls "$PLOT_DIR"
