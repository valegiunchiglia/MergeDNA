#!/usr/bin/env bash

# Sample protein-fitness linear-probe evaluation launcher.
# Usage:
#   bash scripts/run_eval_protein_fitness_sample.sh
#
# Optional overrides:
#   CHECKPOINT=checkpoints/mergedna_best_val_mtr.pt TASK_NAME=protein_fitness bash scripts/run_eval_protein_fitness_sample.sh
#   SYNTHETIC=0 DATA_ROOT=/path/to/protein_fitness_data bash scripts/run_eval_protein_fitness_sample.sh
#   CONDA_ENV=mergedna bash scripts/run_eval_protein_fitness_sample.sh

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
TASK_NAME="${TASK_NAME:-protein_fitness}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_RUNS="${N_RUNS:-3}"
ALPHA_GRID="${ALPHA_GRID:-0.0,1e-6,1e-4,1e-2,1.0}"
ALPHABET="${ALPHABET:-ACGT}"
SEED="${SEED:-42}"
SEQ_LEN="${SEQ_LEN:-64}"

# Data mode:
# - SYNTHETIC=1 (default): use built-in synthetic regression splits
# - SYNTHETIC=0: load train/val/test CSVs from DATA_ROOT/TASK_NAME
SYNTHETIC="${SYNTHETIC:-1}"
DATA_ROOT="${DATA_ROOT:-}"
SYNTH_TRAIN_SIZE="${SYNTH_TRAIN_SIZE:-128}"
SYNTH_VAL_SIZE="${SYNTH_VAL_SIZE:-64}"
SYNTH_TEST_SIZE="${SYNTH_TEST_SIZE:-64}"

CMD=(
  python3 scripts/eval_protein_fitness.py
  --checkpoint "${CHECKPOINT}"
  --task-name "${TASK_NAME}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --seq-len "${SEQ_LEN}"
  --seed "${SEED}"
  --n-runs "${N_RUNS}"
  --alpha-grid "${ALPHA_GRID}"
  --alphabet "${ALPHABET}"
)

if [[ "${SYNTHETIC}" == "1" ]]; then
  CMD+=(
    --synthetic
    --synthetic-train-size "${SYNTH_TRAIN_SIZE}"
    --synthetic-val-size "${SYNTH_VAL_SIZE}"
    --synthetic-test-size "${SYNTH_TEST_SIZE}"
  )
else
  if [[ -z "${DATA_ROOT}" ]]; then
    echo "SYNTHETIC=0 requires DATA_ROOT=/path/to/protein_fitness_data"
    exit 1
  fi
  CMD+=(--data-root "${DATA_ROOT}")
fi

echo "Running: ${CMD[*]}"
PYTHONPATH=src "${CMD[@]}"
