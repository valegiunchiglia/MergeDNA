#!/usr/bin/env bash

# Sample training launcher for MergeDNA.
# Usage:
#   bash scripts/run_train_sample.sh
#
# Optional overrides:
#   DEVICE=cuda STEPS=1000 BATCH_SIZE=8 SEQ_LEN=512 bash scripts/run_train_sample.sh
#   CONDA_ENV=mergedna bash scripts/run_train_sample.sh

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

DEVICE="${DEVICE:-cpu}"
STEPS="${STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-256}"
OUT_DIR="${OUT_DIR:-checkpoints/sample_run}"

# Toggle Weights & Biases by setting WANDB=1.
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-mergedna}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mergedna-sample}"
WANDB_MODE="${WANDB_MODE:-online}" # online | offline | disabled

CMD=(
  python3 scripts/main.py
  --steps "${STEPS}"
  --batch-size "${BATCH_SIZE}"
  --seq-len "${SEQ_LEN}"
  --device "${DEVICE}"
  --out-dir "${OUT_DIR}"
  --log-every 10
  --eval-every 50
  --ckpt-every 100
)

if [[ "${WANDB}" == "1" ]]; then
  CMD+=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "${WANDB_RUN_NAME}"
    --wandb-mode "${WANDB_MODE}"
  )
fi

echo "Running: ${CMD[*]}"
PYTHONPATH=src "${CMD[@]}"

