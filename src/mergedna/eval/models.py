"""
Model-side utilities for downstream genomics evaluation.

This module provides:
- LoRA wrappers and adapter attachment,
- sequence classification head over MergeDNA encoder features,
- backbone builder with frozen base parameters + trainable LoRA adapters.
"""

from __future__ import annotations

import copy
from typing import List, Tuple

import torch
import torch.nn as nn

from mergedna.model import MergeDNA


class LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper for nn.Linear:
    y = base(x) + (alpha / r) * B(A(x))
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.rank = rank
        self.scaling = alpha / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.b(self.a(self.dropout(x))) * self.scaling

    @property
    def weight(self) -> torch.nn.Parameter:
        """
        Expose base linear weight for modules (e.g., MultiheadAttention) that
        access `out_proj.weight` directly.
        """
        return self.base.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        """
        Expose base linear bias for modules that access `out_proj.bias` directly.
        """
        return self.base.bias


def _replace_linear_with_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += _replace_linear_with_lora(child, rank=rank, alpha=alpha, dropout=dropout)
    return replaced


def attach_lora_adapters(model: MergeDNA, rank: int, alpha: float, dropout: float) -> int:
    """
    Attach LoRA adapters to encoder blocks only (local + latent encoders).
    """
    replaced = 0
    replaced += _replace_linear_with_lora(
        model.local_blocks, rank=rank, alpha=alpha, dropout=dropout
    )
    replaced += _replace_linear_with_lora(
        model.latent_encoder, rank=rank, alpha=alpha, dropout=dropout
    )
    return replaced


class SequenceClassifier(nn.Module):
    """
    Sequence-level classifier over MergeDNA latent encoder features.

    Protocol:
    - encode sequence with local tokenizer + latent encoder,
    - mean-pool latent tokens,
    - classify with MLP head.
    """

    def __init__(self, backbone: MergeDNA, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        for i in range(batch_tokens.shape[0]):
            z_l, _ = self.backbone.local_encode(
                tokens=batch_tokens[i],
                sampled_local_keep_ratio=None,
                freeze=False,  # Base params are frozen via requires_grad=False.
            )
            z = z_l
            for block in self.backbone.latent_encoder:
                z = block(z)
            pooled = z.mean(dim=0)
            feats.append(pooled)
        x = torch.stack(feats, dim=0)
        return self.head(x)


def build_frozen_lora_backbone(
    base_model: MergeDNA,
    rank: int,
    alpha: float,
    dropout: float,
    device: torch.device,
) -> Tuple[MergeDNA, int]:
    """
    Clone base model, freeze original weights, then inject trainable LoRA layers.
    """
    model = copy.deepcopy(base_model).to(device)
    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False
    # Attach LoRA to encoder blocks.
    n_replaced = attach_lora_adapters(model, rank=rank, alpha=alpha, dropout=dropout)
    return model, n_replaced
