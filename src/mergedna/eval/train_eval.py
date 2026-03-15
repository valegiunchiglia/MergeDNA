"""
Training and evaluation helpers for downstream genomics tasks.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from mergedna.model import MergeDNA

from .models import SequenceClassifier, build_frozen_lora_backbone, collect_latent_features


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


def fit_ridge_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> Ridge:
    """
    Ridge regression with bias term via scikit-learn:
    min ||Xw + b - y||^2 + alpha * ||w||^2
    """
    if x.ndim != 2:
        raise ValueError("x must be [N, D]")
    if y.ndim != 1:
        raise ValueError("y must be [N]")
    # Keep alpha strictly positive for numerical stability when users include 0.0.
    effective_alpha = max(float(alpha), 1e-12)
    reg = Ridge(alpha=effective_alpha, fit_intercept=True, solver="svd")
    reg.fit(
        x.detach().to(dtype=torch.float64).cpu().numpy(),
        y.detach().to(dtype=torch.float64).cpu().numpy(),
    )
    return reg


def predict_linear(x: torch.Tensor, reg: Ridge) -> torch.Tensor:
    pred_np = reg.predict(x.detach().cpu().numpy())
    return torch.tensor(pred_np, dtype=x.dtype)


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.mean((y_true - y_pred) ** 2).item())


def pearson_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y1 = y_true - torch.mean(y_true)
    y2 = y_pred - torch.mean(y_pred)
    denom = torch.sqrt(torch.sum(y1 * y1) * torch.sum(y2 * y2)).clamp(min=1e-12)
    return float((torch.sum(y1 * y2) / denom).item())


def _rank(x: torch.Tensor) -> torch.Tensor:
    idx = torch.argsort(x, dim=0)
    ranks = torch.empty_like(x, dtype=torch.float32)
    ranks[idx] = torch.arange(x.shape[0], dtype=torch.float32)
    return ranks


def spearman_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return pearson_corr(_rank(y_true), _rank(y_pred))


def evaluate_regression(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
    }


def _select_best_ridge_alpha(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    alpha_grid: Iterable[float],
) -> tuple[float, Dict[str, float], Ridge]:
    best_alpha = None
    best_metrics = {"pearson": -1e9}
    best_reg = None
    for alpha in alpha_grid:
        reg = fit_ridge_regression(x_train, y_train, alpha=float(alpha))
        val_pred = predict_linear(x_val, reg=reg)
        metrics = evaluate_regression(y_val, val_pred)
        if metrics["pearson"] > best_metrics["pearson"]:
            best_alpha = float(alpha)
            best_metrics = metrics
            best_reg = reg
    if best_alpha is None or best_reg is None:
        raise RuntimeError("Alpha grid selection failed.")
    return best_alpha, best_metrics, best_reg


def run_linear_probe_once(
    model: MergeDNA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    alpha_grid: List[float],
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    set_seed(seed)
    # Extract latent embeddings
    x_train, y_train = collect_latent_features(model, train_loader, device=device)
    x_val, y_val = collect_latent_features(model, val_loader, device=device)
    x_test, y_test = collect_latent_features(model, test_loader, device=device)
    # Fit ridge regression for each alpha in the grid.
    best_alpha, val_metrics, reg = _select_best_ridge_alpha(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        alpha_grid=alpha_grid,
    )
    train_metrics = evaluate_regression(y_train, predict_linear(x_train, reg=reg))
    test_metrics = evaluate_regression(y_test, predict_linear(x_test, reg=reg))
    return {
        "best_alpha": best_alpha,
        "train_mse": train_metrics["mse"],
        "train_pearson": train_metrics["pearson"],
        "train_spearman": train_metrics["spearman"],
        "val_mse": val_metrics["mse"],
        "val_pearson": val_metrics["pearson"],
        "val_spearman": val_metrics["spearman"],
        "test_mse": test_metrics["mse"],
        "test_pearson": test_metrics["pearson"],
        "test_spearman": test_metrics["spearman"],
    }


def average_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        raise ValueError("No results to average.")
    keys = results[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = float(sum(float(r[k]) for r in results) / len(results))
    return out
