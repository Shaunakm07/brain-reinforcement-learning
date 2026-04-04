#!/usr/bin/env bash
#SBATCH --job-name=brain-batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/users/%u/brain_optimisation/logs/batch_%A_%a.out
#SBATCH --error=/scratch/users/%u/brain_optimisation/logs/batch_%A_%a.err
# --array is set by submit_pipeline.sh (e.g. --array=0-9%5)
# =============================================================================
# Array job: run run_image.py on many images in parallel.
# Each array task processes one image from image_list.txt.
#
# Submit:
#   sbatch --array=0-$(( $(wc -l < image_list.txt) - 1 ))%5 \
#          cluster/sbatch/batch_inference.sh image_list.txt
#
# Or use: bash cluster/submit_pipeline.sh --batch image_list.txt
# =============================================================================
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../activate_env.sh"

IMAGE_LIST="${1:?Usage: sbatch --array=0-N batch_inference.sh image_list.txt}"

# Pick this task's image (SLURM_ARRAY_TASK_ID is 0-indexed, sed is 1-indexed)
IMAGE_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$IMAGE_LIST")
[[ -z "$IMAGE_PATH" ]] && { echo "No image at index $SLURM_ARRAY_TASK_ID"; exit 1; }

OUTPUT_NAME=$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')
OUTPUT_NPY="$BRAIN_OPT_SCRATCH/outputs/${OUTPUT_NAME}.npy"

echo "Task   : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "Image  : $IMAGE_PATH"
echo "Output : $OUTPUT_NPY"

# Skip if already computed (safe to re-run interrupted arrays)
if [[ -f "$OUTPUT_NPY" ]]; then
    echo "Already exists, skipping."
    exit 0
fi

mkdir -p "$BRAIN_OPT_SCRATCH/outputs"
cd "$BRAIN_OPT_PROJECT"
python run_image.py "$IMAGE_PATH"
mv brain_response.npy "$OUTPUT_NPY"
echo "Saved: $OUTPUT_NPY"
