#!/usr/bin/env bash
#SBATCH --job-name=brain-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/users/%u/brain_optimisation/logs/%x_%j.out
#SBATCH --error=/scratch/users/%u/brain_optimisation/logs/%x_%j.err
# =============================================================================
# Train the brain_steer generator.
# All arguments (except --run-name) are passed to brain_steer/train.py.
#
# Submit examples:
#   sbatch cluster/sbatch/train.sh \
#       --target V1 V2 V3 --steps 200 --run-name visual
#
#   sbatch cluster/sbatch/train.sh \
#       --target FFC STSda STSdp \
#       --suppress V1 V2 \
#       --temperature 0.05 --steps 500 --run-name faces
#
#   sbatch cluster/sbatch/train.sh \
#       --target A1 LBelt MBelt PBelt \
#       --num-frames 8 --steps 300 --run-name auditory
# =============================================================================
set -euo pipefail

source "$HOME/brain_optimisation/cluster/activate_env.sh"

# Strip --run-name from args; pass the rest to train.py
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name) RUN_NAME="$2"; shift 2 ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done

CHECKPOINT_DIR="$BRAIN_OPT_SCRATCH/checkpoints/$RUN_NAME"
echo "Job    : $SLURM_JOB_ID on $SLURM_NODELIST"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Run    : $RUN_NAME"
echo "Args   : ${TRAIN_ARGS[*]}"
echo "Output : $CHECKPOINT_DIR"

cd "$BRAIN_OPT_PROJECT"
python -m brain_steer.train \
    "${TRAIN_ARGS[@]}" \
    --cache "$HF_HOME" \
    --out   "$CHECKPOINT_DIR"

echo "Training complete: $CHECKPOINT_DIR"
