#!/usr/bin/env bash
# =============================================================================
# cluster/submit_pipeline.sh
# Convenience script: submit the full pipeline as chained SLURM jobs.
#
# Modes
# -----
#
# 1. Single image — full forward pipeline (inference + plots):
#      bash cluster/submit_pipeline.sh /path/to/image.jpg
#
# 2. Single image — inference + train generator + generate:
#      bash cluster/submit_pipeline.sh /path/to/image.jpg \
#          --train \
#          --target FFC STSda STSdp \
#          --suppress V1 V2 \
#          --steps 500 \
#          --run-name faces
#
# 3. Batch — process many images in parallel:
#      bash cluster/submit_pipeline.sh --batch image_list.txt
#
# 4. Train only (if you already have a brain_response.npy):
#      bash cluster/submit_pipeline.sh --train-only \
#          --target V1 V2 V3 \
#          --steps 200 \
#          --run-name visual
#
# Job chaining
# ------------
# Jobs are submitted with --dependency=afterok so each step only runs
# if the previous step succeeded. If any step fails, the chain stops.
# Check status with: squeue -u $USER
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster_config.env"
SBATCH_DIR="$SCRIPT_DIR/sbatch"

# ── Parse arguments ───────────────────────────────────────────────────────────
MODE="single"           # single | batch | train-only
IMAGE_PATH=""
IMAGE_LIST=""
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
DO_TRAIN=false
TRAIN_ARGS=()
GENERATE_N=16

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)        MODE="batch"; IMAGE_LIST="$2"; shift 2 ;;
        --train)        DO_TRAIN=true; shift ;;
        --train-only)   MODE="train-only"; shift ;;
        --run-name)     RUN_NAME="$2"; shift 2 ;;
        --generate-n)   GENERATE_N="$2"; shift 2 ;;
        --*)            TRAIN_ARGS+=("$1"); shift ;;
        *)
            if [[ -z "$IMAGE_PATH" ]]; then IMAGE_PATH="$1"; shift
            else TRAIN_ARGS+=("$1"); shift; fi ;;
    esac
done

mkdir -p "$BRAIN_OPT_SCRATCH/logs"

# ── Helper: submit and capture job ID ─────────────────────────────────────────
submit() {
    # Usage: submit [--dependency=afterok:N] script.sh [args...]
    local dep=""
    if [[ "$1" == --dependency* ]]; then dep="$1"; shift; fi
    local script="$1"; shift
    local job_id
    job_id=$(sbatch $dep \
        --output="$BRAIN_OPT_SCRATCH/logs/%x_%j.out" \
        --error="$BRAIN_OPT_SCRATCH/logs/%x_%j.err" \
        "$script" "$@" --parsable 2>/dev/null) \
        || job_id=$(sbatch $dep "$script" "$@" --parsable)
    echo "$job_id"
}

echo "============================================"
echo "Brain Optimisation Pipeline Submission"
echo "Mode      : $MODE"
echo "Run name  : $RUN_NAME"
echo "Scratch   : $BRAIN_OPT_SCRATCH"
echo "============================================"

# ── Mode: single image ────────────────────────────────────────────────────────
if [[ "$MODE" == "single" ]]; then
    [[ -z "$IMAGE_PATH" ]] && { echo "ERROR: provide an image path"; exit 1; }
    OUTPUT_NAME=$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')
    NPY_PATH="$BRAIN_OPT_SCRATCH/outputs/${OUTPUT_NAME}.npy"

    # Job 1: inference
    JOB1=$(sbatch \
        --output="$BRAIN_OPT_SCRATCH/logs/inference_%j.out" \
        --error="$BRAIN_OPT_SCRATCH/logs/inference_%j.err" \
        --parsable \
        "$SBATCH_DIR/inference.sh" "$IMAGE_PATH" "$OUTPUT_NAME")
    echo "Submitted inference      : job $JOB1"

    # Job 2: plots (depends on inference)
    JOB2=$(sbatch \
        --dependency=afterok:"$JOB1" \
        --output="$BRAIN_OPT_SCRATCH/logs/plots_%j.out" \
        --error="$BRAIN_OPT_SCRATCH/logs/plots_%j.err" \
        --parsable \
        "$SBATCH_DIR/plots.sh" "$NPY_PATH" "$OUTPUT_NAME")
    echo "Submitted plots          : job $JOB2 (after $JOB1)"

    # Optional: train + generate
    if $DO_TRAIN; then
        TRAIN_ARGS_WITH_NAME=(--run-name "$RUN_NAME" "${TRAIN_ARGS[@]}")
        JOB3=$(sbatch \
            --dependency=afterok:"$JOB1" \
            --output="$BRAIN_OPT_SCRATCH/logs/train_%j.out" \
            --error="$BRAIN_OPT_SCRATCH/logs/train_%j.err" \
            --parsable \
            "$SBATCH_DIR/train.sh" "${TRAIN_ARGS_WITH_NAME[@]}")
        echo "Submitted training       : job $JOB3 (after $JOB1)"

        CKPT_PATH="$BRAIN_OPT_SCRATCH/checkpoints/$RUN_NAME/checkpoint_final.pt"
        JOB4=$(sbatch \
            --dependency=afterok:"$JOB3" \
            --output="$BRAIN_OPT_SCRATCH/logs/generate_%j.out" \
            --error="$BRAIN_OPT_SCRATCH/logs/generate_%j.err" \
            --parsable \
            "$SBATCH_DIR/generate.sh" "$CKPT_PATH" \
                --n "$GENERATE_N" --analyse \
                --run-name "${RUN_NAME}_generated")
        echo "Submitted generation     : job $JOB4 (after $JOB3)"
    fi

# ── Mode: batch (array job) ───────────────────────────────────────────────────
elif [[ "$MODE" == "batch" ]]; then
    [[ -z "$IMAGE_LIST" ]] && { echo "ERROR: provide --batch image_list.txt"; exit 1; }
    N_IMAGES=$(wc -l < "$IMAGE_LIST")
    echo "Images in list: $N_IMAGES"

    ARRAY_JOB=$(sbatch \
        --array="0-$((N_IMAGES - 1))%10" \
        --output="$BRAIN_OPT_SCRATCH/logs/batch_%A_%a.out" \
        --error="$BRAIN_OPT_SCRATCH/logs/batch_%A_%a.err" \
        --parsable \
        "$SBATCH_DIR/batch_inference.sh" "$IMAGE_LIST")
    echo "Submitted batch array    : job $ARRAY_JOB (${N_IMAGES} tasks, max 10 concurrent)"
    echo ""
    echo "After all tasks complete, run plots with:"
    echo "  bash cluster/submit_pipeline.sh --batch-plots $IMAGE_LIST"

# ── Mode: train only ──────────────────────────────────────────────────────────
elif [[ "$MODE" == "train-only" ]]; then
    TRAIN_ARGS_WITH_NAME=(--run-name "$RUN_NAME" "${TRAIN_ARGS[@]}")
    JOB1=$(sbatch \
        --output="$BRAIN_OPT_SCRATCH/logs/train_%j.out" \
        --error="$BRAIN_OPT_SCRATCH/logs/train_%j.err" \
        --parsable \
        "$SBATCH_DIR/train.sh" "${TRAIN_ARGS_WITH_NAME[@]}")
    echo "Submitted training       : job $JOB1"

    CKPT_PATH="$BRAIN_OPT_SCRATCH/checkpoints/$RUN_NAME/checkpoint_final.pt"
    JOB2=$(sbatch \
        --dependency=afterok:"$JOB1" \
        --output="$BRAIN_OPT_SCRATCH/logs/generate_%j.out" \
        --error="$BRAIN_OPT_SCRATCH/logs/generate_%j.err" \
        --parsable \
        "$SBATCH_DIR/generate.sh" "$CKPT_PATH" \
            --n "$GENERATE_N" --analyse \
            --run-name "${RUN_NAME}_generated")
    echo "Submitted generation     : job $JOB2 (after $JOB1)"
fi

echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f $BRAIN_OPT_SCRATCH/logs/*.out"
echo ""
echo "Outputs will be in:"
echo "  $BRAIN_OPT_SCRATCH/"
