from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import GlobalAttentionBlock, LocalAttentionBlock
from .config import MergeDNAConfig
from .merge_ops import merge_in_windows, merge_in_windows_with_budget, merge_to_target_length
from .scoring import MergeScorer


@dataclass
class ForwardOutput:
    """
    Structured output for one sequence forward pass.
    """

    logits: torch.Tensor  # [N, 4], nucleotide logits
    local_tokens: torch.Tensor  # [L, D]
    source_matrix: torch.Tensor  # [L, N]  (Eq. 2 style source matrix S)
    latent_source_matrix: Optional[torch.Tensor] = None  # [K, L] (for AMTM)


class MergeDNA(nn.Module):
    """
    MergeDNA main model.

    Key paper mapping:
    - Eq. (2): local tokenization with source matrix S
    - Eq. (3): latent context modeling
    - Eq. (4): unmerge + local decoder reconstruction
    - Sec. 3.3 / 3.4: token merging and adaptive latent compression
    """

    def __init__(self, cfg: MergeDNAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_token_id = 4  # extra token id used only for AMTM masking
        self.input_embed = nn.Embedding(5, cfg.d_model)  # A,C,G,T + [MASK]
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.in_dropout = nn.Dropout(cfg.dropout)

        self.local_blocks = nn.ModuleList(
            [
                LocalAttentionBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ff_mult=cfg.ff_mult,
                    dropout=cfg.dropout,
                    local_window_size=cfg.local_window_size,
                    block_style=cfg.block_style,
                )
                for _ in range(cfg.local_encoder_layers)
            ]
        )
        self.local_merge_scorers = nn.ModuleList(
            [MergeScorer(cfg.d_model) for _ in range(cfg.local_encoder_layers)]
        )

        self.latent_encoder = nn.ModuleList(
            [
                GlobalAttentionBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ff_mult=cfg.ff_mult,
                    dropout=cfg.dropout,
                    block_style=cfg.block_style,
                )
                for _ in range(cfg.latent_encoder_layers)
            ]
        )
        self.latent_decoder = nn.ModuleList(
            [
                GlobalAttentionBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ff_mult=cfg.ff_mult,
                    dropout=cfg.dropout,
                    block_style=cfg.block_style,
                )
                for _ in range(cfg.latent_decoder_layers)
            ]
        )
        self.local_decoder = nn.ModuleList(
            [
                LocalAttentionBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ff_mult=cfg.ff_mult,
                    dropout=cfg.dropout,
                    local_window_size=cfg.local_window_size,
                    block_style=cfg.block_style,
                )
                for _ in range(cfg.local_decoder_layers)
            ]
        )

        self.latent_merge_scorer = MergeScorer(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, 4)  # logits over A,C,G,T

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed one sequence of token ids [N] to [N, D].
        """
        n = tokens.shape[0]
        pos = torch.arange(n, device=tokens.device, dtype=torch.long)
        x = self.input_embed(tokens) + self.pos_embed(pos)
        return self.in_dropout(x)

    def local_encode(
        self,
        tokens: torch.Tensor,
        sampled_local_keep_ratio: Optional[float] = None,
        freeze: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the Local Encoder tokenizer and return merged tokens plus source map.

        Paper mapping:
        - Eq. (2): `(Z_L, S) = E_phi(X)`.
        - Sec. 3.3: each local block performs local attention + token merge.

        Args:
            tokens: Base token ids `[N]` in `{A,C,G,T}` (and optionally `[MASK]`).
            sampled_local_keep_ratio: Optional sampled keep ratio `L/N` for this
                step. If provided and target enforcement is enabled, the function
                plans layer-wise removals to approach target `L`.
            freeze: If `True`, run under `no_grad()` and detach outputs. This is
                used for the latent selective reconstruction loss
                `L_MTR(theta \\ {phi})` so local tokenizer params are not updated.

        Returns:
            - `z_l`: local merged tokens `[L, D]`,
            - `s`: source matrix `[L, N]`.

        Modes:
            - target-driven: exact-budget helper (`merge_in_windows_with_budget`)
              to better match sampled `L`,
            - ratio-driven: fixed per-layer merge ratios (`merge_in_windows`).
        """
        n = tokens.shape[0]
        x = self._embed_tokens(tokens)
        # Initialise the source matrix S as an identity matrix. Source matrix is a matrix that maps the local tokens to the base tokens 
        # Gives information about which tokens were merged together.
        s = torch.eye(n, device=tokens.device, dtype=x.dtype)

        target_len = None # No sampling of the local tokens is done.
        # Tells the model how aggressive the merging should be. Useful to prevent overfitting during pretraining.
        if sampled_local_keep_ratio is not None:
            target_len = int(round(n * sampled_local_keep_ratio))
            # Fallback to ensure the target length is not too small or too large (always at least 1 and at most the number of tokens in the sequence)
            target_len = max(1, min(n, target_len))

        # In the latent selection / compression step the local encoder is frozen (methods 3.4) - for second loss term
        if freeze:
            with torch.no_grad():
                for i, block in enumerate(self.local_blocks): # Attention blocks in the local encoder.
                    x = block(x)
                    if self.cfg.enforce_sampled_local_target_len and target_len is not None:
                        remaining_remove = max(0, x.shape[0] - target_len)
                        if remaining_remove > 0:
                            rem_weights = [
                                self.cfg.local_merge_ratios[j % len(self.cfg.local_merge_ratios)]
                                for j in range(i, len(self.local_blocks))
                            ]
                            # Guard against degenerate user configs.
                            rem_weights = [max(1e-6, float(w)) for w in rem_weights]
                            w_i = rem_weights[0]
                            sum_w = sum(rem_weights)
                            planned_remove = max(
                                1, int(round(remaining_remove * (w_i / sum_w)))
                            )
                            x, s = merge_in_windows_with_budget(
                                x=x,
                                s=s,
                                scorer=self.local_merge_scorers[i],
                                window_size=self.cfg.local_window_size,
                                n_remove=planned_remove,
                                merge_mode=self.cfg.local_merge_mode,
                            )
                    else:
                        ratio = float(
                            self.cfg.local_merge_ratios[i % len(self.cfg.local_merge_ratios)]
                        )
                        x, s = merge_in_windows(
                            x=x,
                            s=s,
                            scorer=self.local_merge_scorers[i],
                            window_size=self.cfg.local_window_size,
                            merge_ratio=ratio,
                            merge_mode=self.cfg.local_merge_mode,
                        )
            return x.detach(), s.detach()

        for i, block in enumerate(self.local_blocks):
            # Self attention block in the local encoder. Merging using context aware features
            x = block(x)
            # Token merging module
            if self.cfg.enforce_sampled_local_target_len and target_len is not None:
                # Match expected target length
                # Target-driven mode: start from one target_len, then split remaining removals across layers.
                remaining_remove = max(0, x.shape[0] - target_len)
                if remaining_remove > 0:
                    # the ration from the next layers
                    rem_weights = [
                        self.cfg.local_merge_ratios[j % len(self.cfg.local_merge_ratios)]
                        for j in range(i, len(self.local_blocks))
                    ]
                    rem_weights = [max(1e-6, float(w)) for w in rem_weights] # fix error if 0
                    # weight of the first layer
                    w_i = rem_weights[0]
                    # sum of the weights of remaining layers
                    sum_w = sum(rem_weights)
                    # keep at least one token in each layer
                    planned_remove = max(
                        1, int(round(remaining_remove * (w_i / sum_w)))
                    )
                    x, s = merge_in_windows_with_budget(
                        x=x,
                        s=s,
                        scorer=self.local_merge_scorers[i],
                        window_size=self.cfg.local_window_size,
                        n_remove=planned_remove,
                        merge_mode=self.cfg.local_merge_mode,
                    )
            else:
                # Ratio-driven mode: apply per-layer ratios regardless of exact final target.
                ratio = float(self.cfg.local_merge_ratios[i % len(self.cfg.local_merge_ratios)])
                x, s = merge_in_windows(
                    x=x,
                    s=s,
                    scorer=self.local_merge_scorers[i],
                    window_size=self.cfg.local_window_size,
                    merge_ratio=ratio,
                    merge_mode=self.cfg.local_merge_mode,
                )
        # Output of local encoder: local tokens Z_L and source matrix S.
        return x, s

    @staticmethod
    def _unmerge_tokens(tokens: torch.Tensor, source_matrix: torch.Tensor) -> torch.Tensor:
        """
        Token unmerging operation U(., .) (paper Sec. 3.2 and Sec. 3.3).

        If source_matrix is shape [L, N], returns [N, D] via S^T * Z_L.
        """
        # Uses the source matrix to unmerge the tokens back to the original sequence.
        return source_matrix.transpose(0, 1) @ tokens

    def _latent_encode_decode(self, z_l: torch.Tensor) -> torch.Tensor:
        """
        Standard latent path without latent compression.
        """
        # used for reconstraction from local tokens to base tokens
        x = z_l
        for block in self.latent_encoder:
            x = block(x)
        for block in self.latent_decoder:
            x = block(x)
        return x

    def _latent_selective_encode_decode(
        self, z_l: torch.Tensor, keep_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Latent selective path used for Sec. 3.4 objectives.

        Pipeline:
        1) latent encoder processes local tokens `Z_L`,
        2) global merge selects `K` salient tokens (`L -> K`) and yields `S'`,
        3) latent decoder operates at length `K`,
        4) unmerge with `S'` to recover length `L`.

        Args:
            z_l: Local tokens `[L, D]`.
            keep_ratio: Fraction of latent tokens to keep (`K ~= L * keep_ratio`).

        Returns:
            - `z_l_hat`: decoded latent sequence at local length `[L, D]`,
            - `s_prime`: latent source matrix `[K, L]`.
        """
        # Start from output of local encoder.
        x = z_l
        for block in self.latent_encoder:
            x = block(x)

        # get global context
        l = x.shape[0]
        k = max(1, int(l * keep_ratio)) # number of tokens to keep
        # Initiliase the source matrix S' as an identity matrix. Source matrix is a matrix that maps the latent tokens to the local tokens.
        s_prime = torch.eye(l, device=x.device, dtype=x.dtype)
        z_k, s_prime = merge_to_target_length(
            x=x,
            s=s_prime,
            scorer=self.latent_merge_scorer,
            target_len=k,
        )

        for block in self.latent_decoder:
            z_k = block(z_k)

        # Upsample K back to L using S' (Sec. 3.4 Selection and Reconstruction).
        z_l_hat = self._unmerge_tokens(z_k, s_prime)
        return z_l_hat, s_prime

    def _local_decode(self, z_l_hat: torch.Tensor, source_matrix: torch.Tensor) -> torch.Tensor:
        """
        Local Decoder:
        - unmerge L -> N with source matrix S
        - local refinement
        - output base logits (A,C,G,T)
        """
        # Upsample to local base length N, output A, C, G, T logits for reconstruction
        z_n = self._unmerge_tokens(z_l_hat, source_matrix)
        for block in self.local_decoder:
            z_n = block(z_n)
        return self.out_proj(z_n)

    def forward_one(
        self,
        tokens: torch.Tensor,
        sampled_local_keep_ratio: Optional[float] = None,
        latent_selective: bool = False,
        freeze_local_encoder: bool = False,
    ) -> ForwardOutput:
        """
        Forward one sequence [N].
        """
        z_l, s = self.local_encode(
            tokens=tokens,
            sampled_local_keep_ratio=sampled_local_keep_ratio,
            freeze=freeze_local_encoder,
        )
        if latent_selective:
            z_l_hat, s_prime = self._latent_selective_encode_decode(
                z_l=z_l, keep_ratio=self.cfg.latent_keep_ratio
            )
            logits = self._local_decode(z_l_hat=z_l_hat, source_matrix=s)
            return ForwardOutput(
                logits=logits,
                local_tokens=z_l,
                source_matrix=s,
                latent_source_matrix=s_prime,
            )
        z_l_hat = self._latent_encode_decode(z_l)
        logits = self._local_decode(z_l_hat=z_l_hat, source_matrix=s)
        return ForwardOutput(logits=logits, local_tokens=z_l, source_matrix=s)

    def forward_batch(
        self,
        batch_tokens: torch.Tensor,
        sampled_local_keep_ratio: Optional[float] = None,
        latent_selective: bool = False,
        freeze_local_encoder: bool = False,
    ) -> List[ForwardOutput]:
        """
        Forward a batch [B, N] as a list of per-sequence outputs.

        Note:
        Due to dynamic token lengths after merging, this implementation keeps
        per-sample processing explicit for readability.
        """
        outputs: List[ForwardOutput] = []
        for i in range(batch_tokens.shape[0]):
            outputs.append(
                self.forward_one(
                    tokens=batch_tokens[i],
                    sampled_local_keep_ratio=sampled_local_keep_ratio,
                    latent_selective=latent_selective,
                    freeze_local_encoder=freeze_local_encoder,
                )
            )
        return outputs
