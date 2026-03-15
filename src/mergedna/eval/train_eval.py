"""
Training and evaluation helpers for downstream genomics tasks.
"""

from __future__ import annotations

import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mergedna.model import MergeDNA

from .models import SequenceClassifier, build_frozen_lora_backbone


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_loader(model: SequenceClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    n_total = 0
    n_correct = 0
    with torch.no_grad():
        for batch_tokens, batch_labels in loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_tokens)
            pred = torch.argmax(logits, dim=-1)
            n_total += batch_labels.numel()
            n_correct += int((pred == batch_labels).sum().item())
    return float(n_correct) / float(max(1, n_total))


def train_one_setting(
    backbone: MergeDNA,
    d_model: int,
    n_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> Dict[str, float]:
    set_seed(seed)
    model = SequenceClassifier(backbone=backbone, d_model=d_model, n_classes=n_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        for batch_tokens, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_tokens)
            loss = F.cross_entropy(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_loader(model, val_loader, device=device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_acc": evaluate_loader(model, train_loader, device=device),
        "val_acc": evaluate_loader(model, val_loader, device=device),
        "test_acc": evaluate_loader(model, test_loader, device=device),
    }


def select_best_setting(
    base_model: MergeDNA,
    d_model: int,
    n_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr_grid: List[float],
    wd_grid: List[float],
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    seed: int,
) -> Dict[str, float]:
    best = {"val_acc": -1.0}
    for lr in lr_grid:
        for wd in wd_grid:
            # Add adapters and update for LORA
            backbone, n_lora = build_frozen_lora_backbone(
                base_model=base_model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                device=device,
            )
            metrics = train_one_setting(
                backbone=backbone,
                d_model=d_model,
                n_classes=n_classes,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=epochs,
                lr=lr,
                weight_decay=wd,
                seed=seed,
            )
            candidate = {
                "lr": lr,
                "weight_decay": wd,
                "n_lora_modules": float(n_lora),
                **metrics,
            }
            if candidate["val_acc"] > best["val_acc"]:
                best = candidate
    return best


def parse_float_grid(raw: str) -> List[float]:
    values = [s.strip() for s in raw.split(",") if s.strip()]
    if not values:
        raise ValueError("Hyperparameter grid cannot be empty.")
    return [float(v) for v in values]
