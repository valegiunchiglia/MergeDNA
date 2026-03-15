#!/usr/bin/env bash

# Sample genomics evaluation launcher (SFT + LoRA).
# Usage:
#   bash scripts/run_eval_genomics_sample.sh
#
# Optional overrides:
#   CHECKPOINT=checkpoints/mergedna_best_val_mtr.pt TASK_GROUP=enhancer bash scripts/run_eval_genomics_sample.sh
#   SYNTHETIC=0 DATA_ROOT=/path/to/genomic-benchmark bash scripts/run_eval_genomics_sample.sh
#   CONDA_ENV=mergedna bash scripts/run_eval_genomics_sample.sh

set -euo pipefail

# Conda activation (enabled by default).
# Override env name with CONDA_ENV=your_env.
# Set SKIP_CONDA=1 to bypass activation.
SKIP_CONDA="${SKIP_CONDA:-0}"
CONDA_ENV="${CONDA_ENV:-mergedna}"
if [[ "${SKIP_CONDA}" != "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
  else
    echo "Conda activation requested, but 'conda' is not available in PATH."
    echo "Install conda/miniconda, or run with SKIP_CONDA=1."
    exit 1
  fi
fi

CHECKPOINT="${CHECKPOINT:-checkpoints/eval_smoke_current.pt}"
TASK_GROUP="${TASK_GROUP:-species}" # enhancer | species | regulatory
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-1}"
LR_GRID="${LR_GRID:-1e-4}"
WD_GRID="${WD_GRID:-0.0}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-1.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
SEED="${SEED:-42}"

# Data mode:
# - SYNTHETIC=1 (default): use built-in synthetic splits
# - SYNTHETIC=0: load train/val/test CSVs from DATA_ROOT
SYNTHETIC="${SYNTHETIC:-1}"
DATA_ROOT="${DATA_ROOT:-}"
SYNTH_TRAIN_SIZE="${SYNTH_TRAIN_SIZE:-8}"
SYNTH_VAL_SIZE="${SYNTH_VAL_SIZE:-4}"
SYNTH_TEST_SIZE="${SYNTH_TEST_SIZE:-4}"
SYNTH_POS_RATE="${SYNTH_POS_RATE:-0.5}"

CMD=(
  python3 scripts/eval_genomics.py
  --checkpoint "${CHECKPOINT}"
  --task-group "${TASK_GROUP}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr-grid "${LR_GRID}"
  --wd-grid "${WD_GRID}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --seed "${SEED}"
)

if [[ "${SYNTHETIC}" == "1" ]]; then
  CMD+=(
    --synthetic
    --synthetic-train-size "${SYNTH_TRAIN_SIZE}"
    --synthetic-val-size "${SYNTH_VAL_SIZE}"
    --synthetic-test-size "${SYNTH_TEST_SIZE}"
    --synthetic-positive-rate "${SYNTH_POS_RATE}"
  )
else
  if [[ -z "${DATA_ROOT}" ]]; then
    echo "SYNTHETIC=0 requires DATA_ROOT=/path/to/genomic-benchmark"
    exit 1
  fi
  CMD+=(--data-root "${DATA_ROOT}")
fi

echo "Running: ${CMD[*]}"
PYTHONPATH=src "${CMD[@]}"
