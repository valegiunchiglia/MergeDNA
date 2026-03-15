# MergeDNA (From Scratch)

From-scratch PyTorch implementation of MergeDNA based on the paper:

- [MergeDNA (arXiv:2511.14806)](https://arxiv.org/pdf/2511.14806)

## What is implemented

- Hierarchical architecture: Local Encoder -> Latent Encoder/Decoder -> Local Decoder
- Dynamic token merging with source-matrix tracking (`S`, `S'`)
- Three pretraining losses (paper Eq. 8):
  - `L_MTR(theta)`
  - `lambda * L_MTR(theta \ {phi})`
  - `L_AMTM(theta)`
- W&B logging and periodic validation
- Best-checkpoint saving by `val_mtr`

## Project layout

```text
src/mergedna/
  config.py
  data.py
  model.py
  losses.py
  train.py
scripts/
  main.py              # training entrypoint
  run_train_sample.sh  # sample launcher
```

## Setup

### Option A: Conda (recommended)

```bash
conda create -n mergedna python=3.11 -y
conda activate mergedna
pip install -r requirements.txt
pip install -e .
```

### Option B: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If editable install is not needed:

```bash
PYTHONPATH=src python scripts/main.py --help
```

Using the sample launcher with conda (default env name is `mergedna`):

```bash
bash scripts/run_train_sample.sh
```

Override the env name:

```bash
CONDA_ENV=my_env bash scripts/run_train_sample.sh
```

Bypass conda activation:

```bash
SKIP_CONDA=1 bash scripts/run_train_sample.sh
```

## Quick start

```bash
PYTHONPATH=src python scripts/main.py \
  --steps 20 \
  --batch-size 4 \
  --seq-len 256 \
  --device cpu
```

## Local merge modes

- `adjacent` (default): merge adjacent pairs only (best biological contiguity)
- `bipartite`: ToMe-style bipartite matching within each window
- `full_pairwise`: score all pairs in each local window

Example:

```bash
PYTHONPATH=src python scripts/main.py \
  --local-merge-mode adjacent
```

## Paper-like preset

```bash
PYTHONPATH=src python scripts/main.py \
  --preset paper \
  --seq-len 4096 \
  --batch-size 1 \
  --device cuda
```

Note: this preset is heavy and may require substantial GPU memory.

## Optional FASTA training

```bash
PYTHONPATH=src python scripts/main.py \
  --train-fasta /path/to/train.fasta \
  --val-fasta /path/to/val.fasta \
  --steps 2000 \
  --seq-len 4096 \
  --device cuda
```

Supports plain FASTA and `.gz` FASTA files.

## W&B logging

```bash
PYTHONPATH=src python scripts/main.py \
  --steps 200 \
  --device cuda \
  --wandb \
  --wandb-project mergedna \
  --wandb-run-name mergedna-challenge
```

Modes:
- `--wandb-mode online`
- `--wandb-mode offline`
- `--wandb-mode disabled`

## Checkpoints

- Periodic checkpoints: `checkpoints/mergedna_step_*.pt`
- Best checkpoint by validation MTR: `checkpoints/mergedna_best_val_mtr.pt`
- Final checkpoint: `checkpoints/mergedna_final.pt`

