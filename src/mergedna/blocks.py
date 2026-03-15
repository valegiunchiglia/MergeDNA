from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RMSNorm(nn.Module):
    """
    Root-mean-square normalization used in LLaMA-style blocks.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return x_norm * self.weight


class SwiGLUFeedForward(nn.Module):
    """
    LLaMA-style SwiGLU feed-forward block.
    """

    def __init__(self, d_model: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        hidden = d_model * ff_mult
        self.w1 = nn.Linear(d_model, hidden)
        self.w3 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        return self.dropout(x)


def _build_norm(block_style: str, d_model: int) -> nn.Module:
    if block_style == "llama":
        return RMSNorm(d_model)
    return nn.LayerNorm(d_model)


def _build_ff(block_style: str, d_model: int, ff_mult: int, dropout: float) -> nn.Module:
    if block_style == "llama":
        return SwiGLUFeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout)
    return FeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout)


class LocalAttentionBlock(nn.Module):
    """
    Local-window attention block used in Local Encoder/Decoder.

    Paper mapping:
    - Local Encoder/Decoder in Sec. 3.2 / 3.3
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        local_window_size: int,
        block_style: str = "llama",
    ) -> None:
        super().__init__()
        self.block_style = block_style
        self.local_window_size = local_window_size
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # NOTE: we implemented two types of transformer blocks:
        # - llama: RMSNorm + SwiGLU + pre-norm residual
        # - standard: LayerNorm + GELU FFN + post-norm residual
        self.ff = _build_ff(
            block_style=block_style,
            d_model=d_model,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        self.norm1 = _build_norm(block_style=block_style, d_model=d_model)
        self.norm2 = _build_norm(block_style=block_style, d_model=d_model)

    def _build_local_attn_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """
        Attention mask where True means "disallowed" for nn.MultiheadAttention.
        """
        # Only tokens within the local window can attend to each other.
        idx = torch.arange(length, device=device)
        dist = (idx[:, None] - idx[None, :]).abs()
        # Keep neighbors inside local window.
        return dist > self.local_window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [L, D] -> batchify to [1, L, D]
        x_in = x.unsqueeze(0)
        attn_mask = self._build_local_attn_mask(x_in.shape[1], x_in.device)
        if self.block_style == "llama":
            x_norm = self.norm1(x_in)
            attn_out, _ = self.attn(
                x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False
            )
            x = x_in + attn_out
            x = x + self.ff(self.norm2(x))
        else:
            attn_out, _ = self.attn(x_in, x_in, x_in, attn_mask=attn_mask, need_weights=False)
            x = self.norm1(x_in + attn_out)
            x = self.norm2(x + self.ff(x))
        return x.squeeze(0)


class GlobalAttentionBlock(nn.Module):
    """
    Full attention block used in Latent Encoder/Decoder.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        block_style: str = "llama",
    ) -> None:
        super().__init__()
        self.block_style = block_style
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = _build_ff(
            block_style=block_style,
            d_model=d_model,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        self.norm1 = _build_norm(block_style=block_style, d_model=d_model)
        self.norm2 = _build_norm(block_style=block_style, d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.unsqueeze(0)
        if self.block_style == "llama":
            x_norm = self.norm1(x_in)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
            x = x_in + attn_out
            x = x + self.ff(self.norm2(x))
        else:
            attn_out, _ = self.attn(x_in, x_in, x_in, need_weights=False)
            x = self.norm1(x_in + attn_out)
            x = self.norm2(x + self.ff(x))
        return x.squeeze(0)
