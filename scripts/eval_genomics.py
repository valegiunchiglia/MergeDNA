#!/usr/bin/env python3
"""
Minimal Genomics Benchmark evaluator aligned with paper A.2 setup:
- Supervised fine-tuning (SFT) with a task head,
- frozen pretrained encoder backbone,
- LoRA adapters for parameter-efficient adaptation,
- AdamW optimization and val-based hyperparameter selection.

Supported task groups:
- enhancer
- species
- regulatory

Expected data layout:
  <data_root>/<task_name>/train.csv
  <data_root>/<task_name>/val.csv
  <data_root>/<task_name>/test.csv

Each CSV/TSV file must contain:
- a sequence column (one of: sequence, seq, dna, text)
- a label column (one of: label, target, y)
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch

from mergedna.config import MergeDNAConfig
from mergedna.eval.data import (
    GENOMICS_TASK_GROUPS,
    load_task_raw,
    load_task_synthetic,
    make_loaders,
    remap_labels_to_contiguous,
)
from mergedna.eval.train_eval import parse_float_grid, select_best_setting, set_seed
from mergedna.model import MergeDNA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genomics Benchmark evaluator with SFT + LoRA.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint.")
    parser.add_argument(
        "--task-group",
        type=str,
        required=True,
        choices=sorted(GENOMICS_TASK_GROUPS.keys()),
        help="Task group to evaluate: enhancer/species/regulatory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory containing per-task train/val/test CSVs. Required unless --synthetic is set.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use built-in synthetic labeled datasets instead of loading CSV files.",
    )
    parser.add_argument("--synthetic-train-size", type=int, default=1024)
    parser.add_argument("--synthetic-val-size", type=int, default=256)
    parser.add_argument("--synthetic-test-size", type=int, default=256)
    parser.add_argument("--synthetic-positive-rate", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--lr-grid",
        type=str,
        default="1e-5,5e-5,1e-4",
        help="Comma-separated LR candidates.",
    )
    parser.add_argument(
        "--wd-grid",
        type=str,
        default="0.0,0.01",
        help="Comma-separated weight decay candidates.",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[MergeDNA, MergeDNAConfig]:
    # Load the checkpoint and extract the model configuration and state dictionary.
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_cfg" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint missing 'model_cfg' or 'model_state_dict'.")
    model_cfg = MergeDNAConfig(**ckpt["model_cfg"])
    model = MergeDNA(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, model_cfg


def _validate_args(args: argparse.Namespace) -> None:
    if not args.synthetic and not args.data_root:
        raise ValueError("--data-root is required unless --synthetic is enabled.")
    if args.synthetic:
        if not (0.0 <= args.synthetic_positive_rate <= 1.0):
            raise ValueError("--synthetic-positive-rate must be in [0, 1].")
        if (
            args.synthetic_train_size <= 0
            or args.synthetic_val_size <= 0
            or args.synthetic_test_size <= 0
        ):
            raise ValueError("Synthetic split sizes must all be > 0.")


def _apply_label_map(labels: List[int], label_to_id: dict[int, int], split_name: str) -> List[int]:
    remapped: List[int] = []
    for y in labels:
        y_int = int(y)
        if y_int not in label_to_id:
            known = sorted(label_to_id.keys())
            raise ValueError(
                f"Found unseen label {y_int} in {split_name} split. "
                f"Known train labels: {known}"
            )
        remapped.append(label_to_id[y_int])
    return remapped


def main() -> None:
    args = parse_args()
    _validate_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    # Load model from checkpoint.
    model, model_cfg = _build_model_from_checkpoint(args.checkpoint, device)
    seq_len = args.seq_len if args.seq_len is not None else model_cfg.max_seq_len
    # Coverts it to list for hyperparameter search.
    lr_grid = parse_float_grid(args.lr_grid)
    wd_grid = parse_float_grid(args.wd_grid)

    task_names = GENOMICS_TASK_GROUPS[args.task_group]
    all_test_acc: List[float] = []
    print(f"[eval] task_group={args.task_group} tasks={task_names}")
    print(
        "[eval] protocol=SFT+LoRA "
        f"batch_size={args.batch_size} epochs={args.epochs} "
        f"lr_grid={lr_grid} wd_grid={wd_grid} lora_rank={args.lora_rank}"
    )
    if args.synthetic:
        print(
            "[eval] data=synthetic "
            f"train={args.synthetic_train_size} val={args.synthetic_val_size} "
            f"test={args.synthetic_test_size} pos_rate={args.synthetic_positive_rate}"
        )

    for task_name in task_names:
        if args.synthetic:
            train_seq, train_y, val_seq, val_y, test_seq, test_y = load_task_synthetic(
                task_name=task_name,
                seq_len=seq_len,
                seed=args.seed,
                train_size=args.synthetic_train_size,
                val_size=args.synthetic_val_size,
                test_size=args.synthetic_test_size,
                positive_rate=args.synthetic_positive_rate,
            )
        else:
            train_seq, train_y, val_seq, val_y, test_seq, test_y = load_task_raw(
                data_root=args.data_root,
                task_name=task_name,
            )
        # Fit label mapping from train only to avoid leaking held-out label space.
        (remapped_train_only, label_to_id) = remap_labels_to_contiguous(train_y)
        train_y = remapped_train_only[0]
        val_y = _apply_label_map(val_y, label_to_id=label_to_id, split_name="val")
        test_y = _apply_label_map(test_y, label_to_id=label_to_id, split_name="test")
        train_loader, val_loader, test_loader = make_loaders(
            train_seq=train_seq,
            train_y=train_y,
            val_seq=val_seq,
            val_y=val_y,
            test_seq=test_seq,
            test_y=test_y,
            seq_len=seq_len,
            batch_size=args.batch_size,
        )
        n_classes = max(1, len(label_to_id))
        best = select_best_setting(
            base_model=model,
            d_model=model_cfg.d_model,
            n_classes=n_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr_grid=lr_grid,
            wd_grid=wd_grid,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            seed=args.seed,
        )
        all_test_acc.append(best["test_acc"])
        print(
            f"[task={task_name}] "
            f"best_lr={best['lr']:.1e} best_wd={best['weight_decay']:.3g} "
            f"train_acc={best['train_acc']:.4f} "
            f"val_acc={best['val_acc']:.4f} test_acc={best['test_acc']:.4f}"
        )

    avg = sum(all_test_acc) / max(1, len(all_test_acc))
    print(f"[group={args.task_group}] avg_test_acc={avg:.4f}")


if __name__ == "__main__":
    main()
