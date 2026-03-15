from __future__ import annotations

import os
import random
import math
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import MergeDNAConfig, MergeDNATrainConfig
from .data import FASTADataset, RandomDNADataset
from .losses import amtm_loss, mtr_loss, sample_amtm_masks
from .model import MergeDNA


def _seed_worker(worker_id: int) -> None:
    """
    Ensure each DataLoader worker has a distinct RNG state.

    This avoids repeated random windows across workers when dataset sampling
    uses Python's random module.
    """
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "rng"):
        worker_info.dataset.rng.seed(seed + worker_id)


def _init_wandb(
    model_cfg: MergeDNAConfig, train_cfg: MergeDNATrainConfig
) -> Optional[Any]:
    """
    Initialize W&B lazily and safely.

    This keeps training usable even when wandb is not installed or disabled.
    """
    if not train_cfg.use_wandb or train_cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging requested but wandb is not installed. "
            "Install with `pip install wandb` or disable with --wandb-mode disabled."
        ) from exc

    # Initialise the wandb run for better tracking and logging.
    run = wandb.init(
        project=train_cfg.wandb_project,
        entity=train_cfg.wandb_entity,
        name=train_cfg.wandb_run_name,
        mode=train_cfg.wandb_mode,
        config={
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
        },
    )
    return run


def set_seed(seed: int) -> None:
    """
    Reproducibility helper.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_dataloaders(train_cfg: MergeDNATrainConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val dataloaders from FASTA (if provided) or synthetic DNA.
    """
    if train_cfg.train_fasta:
        train_ds = FASTADataset(
            fasta_path=train_cfg.train_fasta,
            seq_len=train_cfg.seq_len,
            n_samples=train_cfg.train_samples,
            seed=train_cfg.seed,
        )
    else:
        train_ds = RandomDNADataset(
            n_samples=train_cfg.train_samples, seq_len=train_cfg.seq_len, seed=train_cfg.seed
        )

    if train_cfg.val_fasta:
        val_ds = FASTADataset(
            fasta_path=train_cfg.val_fasta,
            seq_len=train_cfg.seq_len,
            n_samples=train_cfg.val_samples,
            seed=train_cfg.seed + 1,
        )
    else:
        val_ds = RandomDNADataset(
            n_samples=train_cfg.val_samples, seq_len=train_cfg.seq_len, seed=train_cfg.seed + 1
        )

    # Generator to improve reproducibility
    train_gen = torch.Generator()
    train_gen.manual_seed(train_cfg.seed)
    val_gen = torch.Generator()
    val_gen.manual_seed(train_cfg.seed + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_worker if train_cfg.num_workers > 0 else None,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_seed_worker if train_cfg.num_workers > 0 else None,
        generator=val_gen,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: MergeDNA, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Lightweight evaluation using only standard MTR path.
    """
    model.eval()
    losses = []
    for batch in val_loader:
        batch = batch.to(device)
        outputs = model.forward_batch(batch, sampled_local_keep_ratio=0.5, latent_selective=False)
        losses.append(mtr_loss(outputs, batch).item())
    return {"val_mtr": float(np.mean(losses)) if losses else float("nan")}


def _sample_local_keep_ratio(cfg: MergeDNAConfig) -> float:
    """
    Compression-ratio sampling strategy for local tokenizer (paper Sec. 3.3).
    """
    if cfg.local_keep_ratio_sampling == "uniform":
        return random.uniform(cfg.local_keep_ratio_min, cfg.local_keep_ratio_max)

    # Default: truncated Gaussian around mean (paper describes Gaussian sampling).
    for _ in range(32):
        value = random.gauss(cfg.local_keep_ratio_mean, cfg.local_keep_ratio_std)
        if cfg.local_keep_ratio_min <= value <= cfg.local_keep_ratio_max:
            return value
    # Fallback if rare repeated misses.
    return max(cfg.local_keep_ratio_min, min(cfg.local_keep_ratio_max, cfg.local_keep_ratio_mean))


def _build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: MergeDNATrainConfig
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """
    Build paper-aligned LR scheduler:
    - linear warmup
    - cosine decay
    """
    if train_cfg.lr_scheduler == "none":
        return None

    total_steps = max(1, train_cfg.steps)
    # Cannot be negative nor equal to total steps (or greater)
    warmup_steps = max(0, min(train_cfg.warmup_steps, total_steps - 1))

    def lr_lambda(step: int) -> float:
        # step is 0-based in LambdaLR internals.
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps + 1:
            return 1.0
        progress = (step - warmup_steps) / float(total_steps - warmup_steps - 1)
        progress = max(0.0, min(1.0, progress))
        # Cosine annealing from 1.0 -> 0.0
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_loop(model_cfg: MergeDNAConfig, train_cfg: MergeDNATrainConfig) -> None:
    """
    Main pretraining loop implementing paper-style total objective:

    Eq. (8):
      L_total = L_MTR(theta)
              + lambda * L_MTR(theta \\ {phi})    [latent selective, frozen local encoder]
              + L_AMTM(theta)
    """
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    os.makedirs(train_cfg.out_dir, exist_ok=True)
    train_loader, val_loader = make_dataloaders(train_cfg)
    wandb_run = _init_wandb(model_cfg=model_cfg, train_cfg=train_cfg)

    model = MergeDNA(model_cfg)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = _build_scheduler(optimizer=optimizer, train_cfg=train_cfg)
    scaler = torch.amp.GradScaler(enabled=(train_cfg.amp and device.type == "cuda"))
    best_val_mtr = float("inf")
    best_ckpt_path = os.path.join(train_cfg.out_dir, "mergedna_best_val_mtr.pt")

    train_iter = iter(train_loader)
    for step in tqdm(range(1, train_cfg.steps + 1), desc="Training"):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch.to(device)

        sampled_keep_ratio = _sample_local_keep_ratio(model_cfg)

        optimizer.zero_grad(set_to_none=True)
        # Sets automatic mixed precision for training.
        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(train_cfg.amp and device.type == "cuda"))
        with autocast_ctx:
            # (1) L_MTR(theta): full model reconstruction.
            out_main = model.forward_batch(
                batch_tokens=batch,
                sampled_local_keep_ratio=sampled_keep_ratio,
                latent_selective=False,
                freeze_local_encoder=False,
            )
            loss_main = mtr_loss(out_main, batch)

            # (2) lambda * L_MTR(theta \\ {phi}):
            #     latent selective reconstruction with frozen local encoder.
            out_latent = model.forward_batch(
                batch_tokens=batch,
                sampled_local_keep_ratio=sampled_keep_ratio,
                latent_selective=True,
                freeze_local_encoder=True,
            )
            loss_latent = mtr_loss(out_latent, batch)

            # AMTM mask sampling from latent grouping S' (Sec. 3.4).
            _, masks_base = sample_amtm_masks(out_latent, batch)

            # (3) L_AMTM(theta): masked prediction on informative positions.
            loss_amtm = amtm_loss(
                model=model,
                batch_tokens=batch,
                masks_base=masks_base,
                sampled_local_keep_ratio=sampled_keep_ratio,
            )

            # Explicit paper-loss terms for clear reporting:
            # - L_MTR(theta)
            # - lambda * L_MTR(theta \ {phi})
            # - L_AMTM(theta)
            loss_latent_weighted = train_cfg.lambda_latent_mtr * loss_latent
            total_loss = loss_main + loss_latent_weighted + loss_amtm

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        if step % train_cfg.log_every == 0:
            train_metrics = {
                "train/loss_total": total_loss.item(),
                "train/loss_mtr_theta": loss_main.item(),  # L_MTR(theta)
                "train/loss_mtr_theta_minus_phi": loss_latent.item(),  # L_MTR(theta \ {phi})
                "train/loss_mtr_theta_minus_phi_weighted": loss_latent_weighted.item(),  # lambda * ...
                "train/loss_amtm_theta": loss_amtm.item(),  # L_AMTM(theta)
                "train/lambda_latent_mtr": train_cfg.lambda_latent_mtr,
                "train/local_keep_ratio": sampled_keep_ratio,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            print(
                "[train step={step}] total={loss:.4f} "
                "L_MTR(theta)={mtr:.4f} "
                "L_MTR(theta\\{{phi}})={lat_mtr:.4f} "
                "lambda*L_MTR(theta\\{{phi}})={lat_mtr_w:.4f} "
                "L_AMTM(theta)={amtm:.4f}".format(
                    step=step,
                    loss=train_metrics["train/loss_total"],
                    mtr=train_metrics["train/loss_mtr_theta"],
                    lat_mtr=train_metrics["train/loss_mtr_theta_minus_phi"],
                    lat_mtr_w=train_metrics["train/loss_mtr_theta_minus_phi_weighted"],
                    amtm=train_metrics["train/loss_amtm_theta"],
                )
            )
            if wandb_run is not None:
                wandb_run.log(train_metrics, step=step)

        if step % train_cfg.eval_every == 0:
            metrics = evaluate(model, val_loader, device)
            print(f"[eval step={step}] {metrics}")
            if wandb_run is not None:
                wandb_run.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            val_mtr = metrics.get("val_mtr", float("inf"))
            if val_mtr < best_val_mtr:
                best_val_mtr = val_mtr
                torch.save(
                    {
                        "step": step,
                        "best_val_mtr": best_val_mtr,
                        "model_cfg": asdict(model_cfg),
                        "train_cfg": asdict(train_cfg),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    },
                    best_ckpt_path,
                )
                print(f"[best-checkpoint] saved {best_ckpt_path} (val_mtr={best_val_mtr:.6f})")

        if step % train_cfg.ckpt_every == 0:
            ckpt_path = os.path.join(train_cfg.out_dir, f"mergedna_step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model_cfg": asdict(model_cfg),
                    "train_cfg": asdict(train_cfg),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                },
                ckpt_path,
            )
            print(f"[checkpoint] saved {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(train_cfg.out_dir, "mergedna_final.pt")
    torch.save(
        {
            "step": train_cfg.steps,
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        final_path,
    )
    print(f"[done] saved final checkpoint at {final_path}")
    if wandb_run is not None:
        # Attach final checkpoint path for run traceability.
        wandb_run.summary["final_checkpoint"] = final_path
        wandb_run.summary["best_checkpoint"] = best_ckpt_path
        wandb_run.summary["best_val_mtr"] = best_val_mtr
        wandb_run.finish()
