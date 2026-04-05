#!/usr/bin/env bash
# =============================================================================
# cluster/submit_train_farmshare.sh
# FarmShare-focused entrypoint for brain_steer training.
#
# This wrapper validates arguments, applies safe defaults for FarmShare GPU
# nodes, and submits train + optional generate jobs as a dependency chain.
#
# Usage examples:
#   bash cluster/submit_train_farmshare.sh \
#       --target V1 V2 V3 \
#       --run-name visual_core
#
#   bash cluster/submit_train_farmshare.sh \
#       --target FFC STSda STSdp \
#       --suppress V1 V2 \
#       --steps 500 \
#       --temperature 0.05 \
#       --num-frames 8 \
#       --run-name faces \
#       --generate 32 --analyse
#
# Notes:
# - Run `bash cluster/setup.sh` once before first submission.
# - All outputs are written to $BRAIN_OPT_SCRATCH (from cluster_config.env).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster_config.env"

SBATCH_TRAIN_SCRIPT="$SCRIPT_DIR/sbatch/train.sh"
SBATCH_GENERATE_SCRIPT="$SCRIPT_DIR/sbatch/generate.sh"

RUN_NAME="train_$(date +%Y%m%d_%H%M%S)"
TRAIN_ARGS=()
GEN_ARGS=()
TARGETS=()
SUPPRESS=()
GENERATE_N=0
ANALYSE=false

# Optional sbatch overrides
PARTITION=""
TIME=""
MEM=""
CPUS=""
GRES=""

usage() {
  cat <<USAGE
Usage:
  bash cluster/submit_train_farmshare.sh --target ROI [ROI ...] [options]

Required:
  --target ROI [ROI ...]         One or more HCP MMP regions to maximize.

Common options:
  --suppress ROI [ROI ...]       Regions to explicitly suppress.
  --steps N                      Training steps (default from train.py: 500).
  --temperature FLOAT            Softmax temperature (default 0.1).
  --num-frames N                 Frames for V-JEPA2 (64 full, 8 low-memory).
  --run-name NAME                Output subdirectory name under checkpoints/.
  --seed N                       Random seed.

Optional generation stage:
  --generate N                   After training, generate N images.
  --analyse                      Use with --generate to run brain analysis.

FarmShare sbatch override options (advanced):
  --partition NAME               Override partition (default from sbatch script).
  --time HH:MM:SS                Override wall-time.
  --mem SIZE                     Override memory (e.g., 64G).
  --cpus N                       Override cpus-per-task.
  --gres SPEC                    Override gres (e.g., gpu:1).

Examples:
  bash cluster/submit_train_farmshare.sh --target V1 V2 V3 --run-name visual

  bash cluster/submit_train_farmshare.sh \
      --target FFC STSda STSdp \
      --suppress V1 V2 \
      --steps 500 --temperature 0.05 --num-frames 8 \
      --run-name faces --generate 16 --analyse
USAGE
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do TARGETS+=("$1"); shift; done
      ;;
    --suppress)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SUPPRESS+=("$1"); shift; done
      ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --generate) GENERATE_N="$2"; shift 2 ;;
    --analyse) ANALYSE=true; shift ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --*)
      # Forward unknown long options to train.py
      TRAIN_ARGS+=("$1")
      if [[ $# -ge 2 && ! "$2" =~ ^-- ]]; then
        TRAIN_ARGS+=("$2")
        shift 2
      else
        shift
      fi
      ;;
    *)
      echo "ERROR: unexpected argument '$1'"
      usage
      exit 1
      ;;
  esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "ERROR: --target is required (one or more ROI names)."
  exit 1
fi
if ! [[ "$GENERATE_N" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --generate must be a non-negative integer."
  exit 1
fi

mkdir -p "$BRAIN_OPT_SCRATCH/logs"

TRAIN_ARGS=(--target "${TARGETS[@]}" "${TRAIN_ARGS[@]}")
if [[ ${#SUPPRESS[@]} -gt 0 ]]; then
  TRAIN_ARGS+=(--suppress "${SUPPRESS[@]}")
fi
TRAIN_ARGS+=(--run-name "$RUN_NAME")

SBATCH_OVERRIDES=()
[[ -n "$PARTITION" ]] && SBATCH_OVERRIDES+=(--partition "$PARTITION")
[[ -n "$TIME" ]] && SBATCH_OVERRIDES+=(--time "$TIME")
[[ -n "$MEM" ]] && SBATCH_OVERRIDES+=(--mem "$MEM")
[[ -n "$CPUS" ]] && SBATCH_OVERRIDES+=(--cpus-per-task "$CPUS")
[[ -n "$GRES" ]] && SBATCH_OVERRIDES+=(--gres "$GRES")

echo "============================================"
echo "FarmShare Training Submission"
echo "Run name      : $RUN_NAME"
echo "Targets       : ${TARGETS[*]}"
echo "Suppress      : ${SUPPRESS[*]:-none}"
echo "Scratch       : $BRAIN_OPT_SCRATCH"
echo "Sbatch extras : ${SBATCH_OVERRIDES[*]:-none}"
echo "Train args    : ${TRAIN_ARGS[*]}"
echo "============================================"

TRAIN_JOB=$(sbatch \
  "${SBATCH_OVERRIDES[@]}" \
  --output="$BRAIN_OPT_SCRATCH/logs/train_%j.out" \
  --error="$BRAIN_OPT_SCRATCH/logs/train_%j.err" \
  --parsable \
  "$SBATCH_TRAIN_SCRIPT" "${TRAIN_ARGS[@]}")

echo "Submitted training : job $TRAIN_JOB"

if [[ "$GENERATE_N" -gt 0 ]]; then
  CKPT_PATH="$BRAIN_OPT_SCRATCH/checkpoints/$RUN_NAME/checkpoint_final.pt"
  GEN_ARGS=(--n "$GENERATE_N" --run-name "${RUN_NAME}_generated")
  $ANALYSE && GEN_ARGS+=(--analyse)

  GEN_JOB=$(sbatch \
    "${SBATCH_OVERRIDES[@]}" \
    --dependency=afterok:"$TRAIN_JOB" \
    --output="$BRAIN_OPT_SCRATCH/logs/generate_%j.out" \
    --error="$BRAIN_OPT_SCRATCH/logs/generate_%j.err" \
    --parsable \
    "$SBATCH_GENERATE_SCRIPT" "$CKPT_PATH" "${GEN_ARGS[@]}")

  echo "Submitted generate : job $GEN_JOB (after $TRAIN_JOB)"
fi

echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f $BRAIN_OPT_SCRATCH/logs/train_${TRAIN_JOB}.out"
