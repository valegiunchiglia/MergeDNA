#!/usr/bin/env python3
"""
CLI entry point for MergeDNA training.

This script is intentionally explicit and verbose for technical challenge clarity.
"""

import argparse

from mergedna.config import MergeDNAConfig, MergeDNATrainConfig
from mergedna.train import train_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MergeDNA from scratch.")

    # Runtime
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset",
        type=str,
        default="challenge",
        choices=["challenge", "paper"],
        help="challenge: lighter defaults, paper: closer to architecture in paper.",
    )

    # Data
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--train-fasta", type=str, default=None)
    parser.add_argument("--val-fasta", type=str, default=None)

    # Optimization
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA.")

    # Model size
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--local-layers", type=int, default=4)
    parser.add_argument("--latent-enc-layers", type=int, default=6)
    parser.add_argument("--latent-dec-layers", type=int, default=2)
    parser.add_argument("--local-dec-layers", type=int, default=2)
    parser.add_argument("--local-window-size", type=int, default=16)
    parser.add_argument("--latent-keep-ratio", type=float, default=0.5)
    parser.add_argument(
        "--local-merge-mode",
        type=str,
        default="adjacent",
        choices=["adjacent", "bipartite", "full_pairwise"],
        help="Local merge operator: adjacent, bipartite, or full_pairwise.",
    )

    # Logging/checkpoints
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="checkpoints")

    # Optional W&B
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="mergedna")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode. Use offline when internet is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.preset == "paper":
        # Paper-like architectural defaults from implementation section:
        # D=1024, local/latent/latent-dec/local-dec = 4/20/4/2, window=16.
        # Note: this is very heavy for local machines.
        model_cfg = MergeDNAConfig(
            max_seq_len=args.seq_len,
            d_model=1024,
            n_heads=16,
            local_encoder_layers=4,
            local_decoder_layers=2,
            latent_encoder_layers=20,
            latent_decoder_layers=4,
            local_window_size=16,
            latent_keep_ratio=0.5,
            local_merge_mode=args.local_merge_mode,
        )
    else:
        model_cfg = MergeDNAConfig(
            max_seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            local_encoder_layers=args.local_layers,
            local_decoder_layers=args.local_dec_layers,
            latent_encoder_layers=args.latent_enc_layers,
            latent_decoder_layers=args.latent_dec_layers,
            local_window_size=args.local_window_size,
            latent_keep_ratio=args.latent_keep_ratio,
            local_merge_mode=args.local_merge_mode,
        )

    train_cfg = MergeDNATrainConfig(
        seed=args.seed,
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        train_fasta=args.train_fasta,
        val_fasta=args.val_fasta,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=args.amp,
        log_every=args.log_every,
        eval_every=args.eval_every,
        ckpt_every=args.ckpt_every,
        out_dir=args.out_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )

    train_loop(model_cfg=model_cfg, train_cfg=train_cfg)


if __name__ == "__main__":
    main()
