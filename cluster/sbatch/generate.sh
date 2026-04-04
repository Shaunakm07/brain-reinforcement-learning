#!/usr/bin/env bash
#SBATCH --job-name=brain-generate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/%u/brain_optimisation/logs/%x_%j.out
#SBATCH --error=/scratch/users/%u/brain_optimisation/logs/%x_%j.err
# =============================================================================
# Generate images from a trained checkpoint.
# Submit: sbatch cluster/sbatch/generate.sh /path/to/checkpoint.pt [args]
# =============================================================================
set -euo pipefail

source "$HOME/brain_optimisation/cluster/activate_env.sh"

CHECKPOINT="${1:?Usage: sbatch generate.sh /path/to/checkpoint.pt [--n N] [--analyse] [--run-name NAME]}"
shift
RUN_NAME="generated_$(date +%Y%m%d_%H%M%S)"
GEN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name) RUN_NAME="$2"; shift 2 ;;
        *) GEN_ARGS+=("$1"); shift ;;
    esac
done

OUT_DIR="$BRAIN_OPT_SCRATCH/generated/$RUN_NAME"
echo "Job        : $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Checkpoint : $CHECKPOINT"
echo "Output     : $OUT_DIR"

cd "$BRAIN_OPT_PROJECT"
python -m brain_steer.generate \
    --checkpoint "$CHECKPOINT" \
    --cache      "$HF_HOME" \
    --out        "$OUT_DIR" \
    "${GEN_ARGS[@]}"

echo "Done: $OUT_DIR"
ls "$OUT_DIR"
