# MergeDNA

PyTorch implementation of MergeDNA based on the paper:

- [MergeDNA (arXiv:2511.14806)](https://arxiv.org/pdf/2511.14806)

## What is implemented

- Hierarchical architecture: Local Encoder -> Latent Encoder/Decoder -> Local Decoder
- Dynamic token merging with source-matrix tracking (`S`, `S'`)
- Three pretraining losses (paper Eq. 8):
  - `L_MTR(theta)`
  - `lambda * L_MTR(theta \ {phi})`
  - `L_AMTM(theta)`
- Paper-style optimization defaults:
  - AdamW (`lr=1e-4`, betas `(0.9, 0.95)`, `weight_decay=1e-8`)
  - linear warmup + cosine annealing scheduler (`warmup_steps=10000`)
- W&B logging and periodic validation
- Best-checkpoint saving by `val_mtr`

## Project layout

```text
src/mergedna/
  config.py
  blocks.py
  scoring.py
  merge_ops.py
  data.py
  model.py            # MergeDNA model assembly + forward paths
  losses.py
  train.py
  eval/
    __init__.py
    data.py           # task loading, synthetic data, dataloaders
    models.py         # LoRA adapters + sequence classifier
    train_eval.py     # train/eval loops + HP search helpers
  __init__.py
scripts/
  main.py              # training entrypoint
  eval_genomics.py     # downstream SFT+LoRA evaluator (thin CLI)
  eval_protein_fitness.py # frozen latent linear-probe evaluator
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

Disable scheduler (optional):

```bash
PYTHONPATH=src python scripts/main.py --lr-scheduler none
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

## Transformer block style

- `llama` (default): RMSNorm + SwiGLU with pre-norm residual blocks
- `standard`: LayerNorm + GELU feed-forward with post-norm residual blocks

Example:

```bash
PYTHONPATH=src python scripts/main.py \
  --block-style llama
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

## Downstream evaluation

Run Genomics Benchmark-style evaluation (frozen encoder + LoRA + MLP head):

```bash
PYTHONPATH=src python scripts/eval_genomics.py \
  --checkpoint checkpoints/mergedna_best_val_mtr.pt \
  --task-group enhancer \
  --data-root /path/to/genomic-benchmark
```

Synthetic smoke-test mode (no CSV files needed):

```bash
PYTHONPATH=src python scripts/eval_genomics.py \
  --checkpoint /path/to/current-compatible-checkpoint.pt \
  --task-group species \
  --synthetic \
  --synthetic-train-size 8 \
  --synthetic-val-size 4 \
  --synthetic-test-size 4 \
  --epochs 1 \
  --lr-grid 1e-4 \
  --wd-grid 0.0 \
  --batch-size 4 \
  --device cpu
```

Core split modules and import paths:
- `from mergedna.eval.data import GENOMICS_TASK_GROUPS, load_task_raw, load_task_synthetic, make_loaders, infer_num_classes_from_labels`
- `from mergedna.eval.models import LoRALinear, SequenceClassifier, attach_lora_adapters, build_frozen_lora_backbone`
- `from mergedna.eval.train_eval import set_seed, parse_float_grid, evaluate_loader, train_one_setting, select_best_setting`

### Protein fitness (linear probe, 3-run average)

Following the paper-style protocol, you can freeze the pretrained backbone,
extract latent embeddings, train a linear regressor, and average metrics across runs:

```bash
PYTHONPATH=src python scripts/eval_protein_fitness.py \
  --checkpoint /path/to/current-compatible-checkpoint.pt \
  --task-name protein_fitness \
  --data-root /path/to/protein_fitness_data \
  --alpha-grid 0.0,1e-6,1e-4,1e-2,1.0 \
  --n-runs 3 \
  --batch-size 32 \
  --device cpu
```

Synthetic smoke-test mode:

```bash
PYTHONPATH=src python scripts/eval_protein_fitness.py \
  --checkpoint /path/to/current-compatible-checkpoint.pt \
  --synthetic \
  --n-runs 3 \
  --batch-size 8 \
  --device cpu
```

Expected file layout for real data:
- `<data_root>/<task_name>/train.csv`
- `<data_root>/<task_name>/val.csv`
- `<data_root>/<task_name>/test.csv`

Expected columns:
- sequence column: one of `sequence, seq, dna, protein, text`
- fitness/target column: one of `fitness, target, y, label`

## Checkpoints

- Periodic checkpoints: `checkpoints/mergedna_step_*.pt`
- Best checkpoint by validation MTR: `checkpoints/mergedna_best_val_mtr.pt`
- Final checkpoint: `checkpoints/mergedna_final.pt`

