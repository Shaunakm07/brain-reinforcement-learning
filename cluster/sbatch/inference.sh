#!/usr/bin/env bash
#SBATCH --job-name=brain-inference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/%u/brain_optimisation/logs/%x_%j.out
#SBATCH --error=/scratch/users/%u/brain_optimisation/logs/%x_%j.err
# =============================================================================
# Run run_image.py on a single image.
# Submit: sbatch cluster/sbatch/inference.sh /path/to/image.jpg [output_name]
# =============================================================================
set -euo pipefail

source "$HOME/brain_optimisation/cluster/activate_env.sh"

IMAGE_PATH="${1:?Usage: sbatch inference.sh /path/to/image.jpg [output_name]}"
OUTPUT_NAME="${2:-$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')}"
OUTPUT_NPY="$BRAIN_OPT_SCRATCH/outputs/${OUTPUT_NAME}.npy"

echo "Job    : $SLURM_JOB_ID on $SLURM_NODELIST"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Image  : $IMAGE_PATH"
echo "Output : $OUTPUT_NPY"

mkdir -p "$BRAIN_OPT_SCRATCH/outputs"
cd "$BRAIN_OPT_PROJECT"

python run_image.py "$IMAGE_PATH"
mv brain_response.npy "$OUTPUT_NPY"
echo "Saved: $OUTPUT_NPY"
