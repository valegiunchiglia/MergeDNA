#!/usr/bin/env python3
"""
Protein fitness linear-probe evaluator.

Protocol:
- load pretrained MergeDNA checkpoint,
- freeze backbone and extract latent embeddings,
- fit ridge linear regression on train embeddings,
- pick best alpha on validation (by Pearson),
- report test metrics,
- average over 3 runs (configurable).
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch

from mergedna.config import MergeDNAConfig
from mergedna.eval.data import (
    load_task_regression_raw,
    load_task_regression_synthetic,
    make_regression_loaders,
)
from mergedna.eval.models import validate_alphabet_compatibility
from mergedna.eval.train_eval import (
    average_metrics,
    parse_float_grid,
    run_linear_probe_once,
    set_seed,
)
from mergedna.model import MergeDNA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protein fitness linear-probe evaluator.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint.")
    parser.add_argument("--task-name", type=str, default="protein_fitness")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="0.0,1e-6,1e-4,1e-2,1.0",
        help="Comma-separated ridge alpha candidates.",
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default="ACGT",
        help="Tokenizer alphabet used to map sequence chars to token ids.",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-train-size", type=int, default=128)
    parser.add_argument("--synthetic-val-size", type=int, default=64)
    parser.add_argument("--synthetic-test-size", type=int, default=64)
    return parser.parse_args()


def _build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[MergeDNA, MergeDNAConfig]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_cfg" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint missing 'model_cfg' or 'model_state_dict'.")
    model_cfg = MergeDNAConfig(**ckpt["model_cfg"])
    model = MergeDNA(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, model_cfg


def _validate_args(args: argparse.Namespace) -> None:
    if not args.synthetic and not args.data_root:
        raise ValueError("--data-root is required unless --synthetic is enabled.")
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be > 0.")
    if args.synthetic:
        if (
            args.synthetic_train_size <= 0
            or args.synthetic_val_size <= 0
            or args.synthetic_test_size <= 0
        ):
            raise ValueError("Synthetic split sizes must all be > 0.")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    model, model_cfg = _build_model_from_checkpoint(args.checkpoint, device)
    # check if the vocabulary is compatible with the model
    validate_alphabet_compatibility(model, args.alphabet)
    seq_len = args.seq_len if args.seq_len is not None else model_cfg.max_seq_len
    # Hyperparameter search grid - convert to lists if strings
    alpha_grid = parse_float_grid(args.alpha_grid)

    if args.synthetic:
        train_seq, train_y, val_seq, val_y, test_seq, test_y = load_task_regression_synthetic(
            seq_len=seq_len,
            alphabet=list(args.alphabet.strip().upper()),
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            seed=args.seed,
        )
    else:
        train_seq, train_y, val_seq, val_y, test_seq, test_y = load_task_regression_raw(
            data_root=args.data_root,
            task_name=args.task_name,
        )

    train_loader, val_loader, test_loader = make_regression_loaders(
        train_seq=train_seq,
        train_y=train_y,
        val_seq=val_seq,
        val_y=val_y,
        test_seq=test_seq,
        test_y=test_y,
        seq_len=seq_len,
        batch_size=args.batch_size,
        alphabet=args.alphabet,
    )

    run_results: List[dict] = []
    print(
        "[protein-fitness] protocol=frozen-latent-linear-probe "
        f"task={args.task_name} n_runs={args.n_runs} alpha_grid={alpha_grid}"
    )
    for run_idx in range(args.n_runs):
        run_seed = args.seed + run_idx
        # Linear probe once for each hyperparameter in the grid.
        metrics = run_linear_probe_once(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            alpha_grid=alpha_grid,
            device=device,
            seed=run_seed,
        )
        run_results.append(metrics)
        print(
            f"[run={run_idx + 1}/{args.n_runs}] "
            f"best_alpha={metrics['best_alpha']:.3g} "
            f"val_pearson={metrics['val_pearson']:.4f} "
            f"test_pearson={metrics['test_pearson']:.4f} "
            f"test_spearman={metrics['test_spearman']:.4f} "
            f"test_mse={metrics['test_mse']:.6f}"
        )

    avg = average_metrics(run_results)
    print(
        "[avg] "
        f"best_alpha={avg['best_alpha']:.3g} "
        f"val_pearson={avg['val_pearson']:.4f} "
        f"test_pearson={avg['test_pearson']:.4f} "
        f"test_spearman={avg['test_spearman']:.4f} "
        f"test_mse={avg['test_mse']:.6f}"
    )


if __name__ == "__main__":
    main()
