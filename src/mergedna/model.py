from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MergeDNAConfig


@dataclass
class ForwardOutput:
    """
    Structured output for one sequence forward pass.
    """

    logits: torch.Tensor  # [N, 4], nucleotide logits
    local_tokens: torch.Tensor  # [L, D]
    source_matrix: torch.Tensor  # [L, N]  (Eq. 2 style source matrix S)
    latent_source_matrix: Optional[torch.Tensor] = None  # [K, L] (for AMTM)


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
    ) -> None:
        super().__init__()
        self.local_window_size = local_window_size
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

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
        attn_out, _ = self.attn(x_in, x_in, x_in, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x_in + attn_out)
        x = self.norm2(x + self.ff(x))
        return x.squeeze(0)


class GlobalAttentionBlock(nn.Module):
    """
    Full attention block used in Latent Encoder/Decoder.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.unsqueeze(0)
        attn_out, _ = self.attn(x_in, x_in, x_in, need_weights=False)
        x = self.norm1(x_in + attn_out)
        x = self.norm2(x + self.ff(x))
        return x.squeeze(0)


class MergeScorer(nn.Module):
    """
    Lightweight token grouping embedding (similar spirit to paper Sec. 3.3).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def adjacent_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adjacent-pair similarity inside one sequence/window.

        returns:
        - sim[i] corresponds to pair (i, i+1)
        """
        if x.shape[0] < 2:
            return x.new_zeros(0)
        # The projection is a linear transformation that maps the tokens to a lower-dimensional space.
        # from paper- using a lightweight grouping embedding as in DTEM
        g = F.normalize(self.proj(x), dim=-1)
        return (g[:-1] * g[1:]).sum(dim=-1)

    def bipartite_similarity(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ToMe-style bipartite similarity over alternating token sets.

        We split tokens as:
        - A = x[::2]
        - B = x[1::2]
        and compute cosine-like dot product on normalized projected features.

        Returns:
        - best_scores: [|A|], max similarity score for each token in A
        - best_dst_idx: [|A|], index in B for each token in A
        """
        if x.shape[0] < 2 or x[1::2].shape[0] == 0:
            empty_scores = x.new_zeros((x[::2].shape[0],))
            empty_dst = torch.zeros((x[::2].shape[0],), dtype=torch.long, device=x.device)
            return empty_scores, empty_dst

        # Lightweight projection as mentioned in the paper
        g = F.normalize(self.proj(x), dim=-1)
        a = g[::2]  # source candidates
        b = g[1::2]  # destination candidates
        # Cosine similairty 
        scores = a @ b.transpose(-1, -2)  # [|A|, |B|]
        # Find index and value of the best match for each token in A.
        best_scores, best_dst_idx = scores.max(dim=-1)
        return best_scores, best_dst_idx

    def full_pairwise_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full pairwise similarity matrix inside one sequence/window.
        """
        if x.shape[0] < 2:
            return x.new_zeros((x.shape[0], x.shape[0]))
        g = F.normalize(self.proj(x), dim=-1)
        return g @ g.transpose(-1, -2)


def _apply_bipartite_merge(
    x: torch.Tensor, s: torch.Tensor, scorer: MergeScorer, n_select: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply one ToMe-like bipartite soft matching merge step.

    Design mapping to ToMe (Bolya et al., 2023):
    - Partition alternating tokens into A and B.
    - For each token in A, find best match in B.
    - Keep top-r A->B matches.
    - Merge source token (A) into destination token (B).

    We additionally use source-matrix-derived token sizes to perform
    size-weighted averages (similar to "tracking token size" in ToMe).
    """
    if n_select <= 0 or x.shape[0] < 2:
        return x, s
    # Build two partitins, tokens at even vs odd positions, this are used for the similarity scores.
    ax = x[::2]
    bx = x[1::2]
    a_s = s[::2]
    b_s = s[1::2]
    if bx.shape[0] == 0:
        return x, s

    # Score best A -> B edge per A, then keep top-r sources.
    # Compute the similarit scores for each token in A to all tokens in B.
    best_scores, best_dst_idx = scorer.bipartite_similarity(x)
    n_select = min(n_select, ax.shape[0])
    # Sort the best scores in descending order to extract the top n_select tokens in A.
    src_idx = torch.argsort(best_scores, descending=True)[:n_select]

    # Unmerged tokens from A.
    unm_mask = torch.ones(ax.shape[0], dtype=torch.bool, device=x.device)
    unm_mask[src_idx] = False
    unm_ax = ax[unm_mask]
    unm_as = a_s[unm_mask]

    # Size-aware weighted merge:
    # token_size is number of original positions represented by each token row.
    a_size = a_s.sum(dim=-1, keepdim=True)  # [|A|, 1]
    b_size = b_s.sum(dim=-1, keepdim=True)  # [|B|, 1]

    src_x = ax[src_idx]
    src_s = a_s[src_idx]
    src_size = a_size[src_idx]
    dst_idx = best_dst_idx[src_idx]  # [r], into B

    # Weighted sum in feature space, then divide by updated token size.
    b_weighted = bx * b_size
    b_weighted = b_weighted.index_add(0, dst_idx, src_x * src_size)
    b_s_new = b_s.index_add(0, dst_idx, src_s)
    b_size_new = b_size.index_add(0, dst_idx, src_size)
    b_x_new = b_weighted / b_size_new.clamp(min=1e-6)

    # ToMe-style output ordering: unmerged A first, then updated B.
    new_x = torch.cat([unm_ax, b_x_new], dim=0)
    new_s = torch.cat([unm_as, b_s_new], dim=0)
    # NOTE: this is different compared to adjacent or full pairwise merge
    # In adjacent you preserve the sequence order
    # This could be an issue in biology
    return new_x, new_s


def _select_non_overlapping_adjacent_pairs(sim: torch.Tensor, n_select: int) -> List[int]:
    """
    Select top non-overlapping adjacent pair indices.

    If i is selected, pair is (i, i+1) and i-1 / i+1 are blocked.
    """
    # Sort the similarity scores in descending order to extract the top n_select adjacent pairs.
    if n_select <= 0 or sim.numel() == 0:
        return []
    sorted_idx = torch.argsort(sim, descending=True).tolist()
    selected: List[int] = []
    blocked = set()
    for i in sorted_idx:
        if len(selected) >= n_select:
            break
        if i in blocked or (i - 1) in blocked or (i + 1) in blocked:
            continue
        selected.append(i)
        blocked.add(i)
    return sorted(selected)


def _apply_adjacent_merge(
    x: torch.Tensor, s: torch.Tensor, scorer: MergeScorer, n_select: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Paper-faithful local merge: only adjacent token pairs can merge.
    """
    if n_select <= 0 or x.shape[0] < 2:
        return x, s

    # Calculate the similarity scores for each adjacent pair of tokens.
    sim = scorer.adjacent_similarity(x)
    # Maximum number of adjacent pairs that can be merged.
    max_pairs = x.shape[0] // 2
    # Select the top n_select adjacent pairs.
    n_select = min(n_select, max_pairs)
    pair_indices = _select_non_overlapping_adjacent_pairs(sim=sim, n_select=n_select)
    if not pair_indices: # this happens if there are no adjacent pairs to merge.
        return x, s

    pair_set = set(pair_indices)
    # tokens with higher sizes (represent more bases) weight more, first layer all 1
    sizes = s.sum(dim=-1, keepdim=True)  # [T, 1]
    new_x: List[torch.Tensor] = []
    new_s: List[torch.Tensor] = []
    i = 0
    while i < x.shape[0]:
        if i in pair_set:
            size_i = sizes[i]
            size_j = sizes[i + 1]
            # Weighted average of the two tokens.
            merged_x = (x[i] * size_i + x[i + 1] * size_j) / (size_i + size_j).clamp(min=1e-6)
            # Update the source matrix by doing the sum
            merged_s = s[i] + s[i + 1]
            new_x.append(merged_x)
            new_s.append(merged_s)
            i += 2
        else:
            # If the token is not in a pair, it is not merged and remains unchanged.
            new_x.append(x[i])
            new_s.append(s[i])
            i += 1
    # Stack the new tokens and source matrices into tensors.
    return torch.stack(new_x, dim=0), torch.stack(new_s, dim=0)


def _apply_full_pairwise_merge(
    x: torch.Tensor, s: torch.Tensor, scorer: MergeScorer, n_select: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full-pairwise local merge: score all token pairs in a window and
    select top non-overlapping pairs.
    """
    t = x.shape[0]
    if n_select <= 0 or t < 2:
        return x, s

    sim = scorer.full_pairwise_similarity(x)
    # Set diagonal to 0 and offset by 1 to avoid self-matches.
    iu, ju = torch.triu_indices(t, t, offset=1, device=x.device)
    if iu.numel() == 0:
        return x, s

    # Sort to identify the top n_select pairs.

    pair_scores = sim[iu, ju]
    order = torch.argsort(pair_scores, descending=True)



    max_pairs = t // 2
    n_select = min(n_select, max_pairs)

    selected_pairs: List[Tuple[int, int]] = []
    # Prevent to re-use the same token twice, greedy non overlapping pairs.
    used = set()
    for idx in order.tolist():
        i = int(iu[idx].item())
        j = int(ju[idx].item())
        if i in used or j in used:
            continue
        selected_pairs.append((i, j))
        used.add(i)
        used.add(j)
        if len(selected_pairs) >= n_select:
            break

    if not selected_pairs:
        return x, s
    # Create a dictionary to map each left token to its right token.
    pair_by_left = {i: j for i, j in selected_pairs}
    right_nodes = {j for _, j in selected_pairs}
    sizes = s.sum(dim=-1, keepdim=True)
    # Weighted average of the two tokens.
    new_x: List[torch.Tensor] = []
    new_s: List[torch.Tensor] = []
    for i in range(t):
        if i in right_nodes:
            continue
        if i in pair_by_left:
            j = pair_by_left[i]
            size_i = sizes[i]
            size_j = sizes[j]
            merged_x = (x[i] * size_i + x[j] * size_j) / (size_i + size_j).clamp(min=1e-6)
            merged_s = s[i] + s[j]
            new_x.append(merged_x)
            new_s.append(merged_s)
        else:
            new_x.append(x[i])
            new_s.append(s[i])

    return torch.stack(new_x, dim=0), torch.stack(new_s, dim=0)


def _apply_local_merge(
    x: torch.Tensor,
    s: torch.Tensor,
    scorer: MergeScorer,
    n_select: int,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch local merge mode.
    """
    # NOTE: I think the paper was not super clear about this. It says: 
    # 'In implementation, we compute a similarity score for each pair of tokens in a local window' 
    # this suggest full pairwise merge, but it also suggests they should be adjacent 
    # and implementation is from ToMe.
    if mode == "adjacent":
        return _apply_adjacent_merge(x=x, s=s, scorer=scorer, n_select=n_select)
    if mode == "bipartite":
        return _apply_bipartite_merge(x=x, s=s, scorer=scorer, n_select=n_select)
    if mode == "full_pairwise":
        return _apply_full_pairwise_merge(x=x, s=s, scorer=scorer, n_select=n_select)
    raise ValueError(f"Unsupported local_merge_mode: {mode}")


def merge_in_windows(
    x: torch.Tensor,
    s: torch.Tensor,
    scorer: MergeScorer,
    window_size: int,
    merge_ratio: float,
    merge_mode: str = "adjacent",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Local-window token merging used inside Local Encoder.

    Paper mapping:
    - Eq. (5) LocalToMeAttn style reduction with source matrix updates.
    """
    # Same but using only ratio and ignore target length.
    if x.shape[0] <= 1 or merge_ratio <= 0.0:
        return x, s

    merged_x_chunks: List[torch.Tensor] = []
    merged_s_chunks: List[torch.Tensor] = []

    for start in range(0, x.shape[0], window_size):
        end = min(start + window_size, x.shape[0])
        xw = x[start:end]
        sw = s[start:end]
        if xw.shape[0] < 2:
            merged_x_chunks.append(xw)
            merged_s_chunks.append(sw)
            continue

        # In one bipartite merge step, at most floor(T/2) tokens are removed.
        max_removable = xw.shape[0] // 2
        n_select = min(int(xw.shape[0] * merge_ratio), max_removable)
        xw_new, sw_new = _apply_local_merge(
            x=xw,
            s=sw,
            scorer=scorer,
            n_select=n_select,
            mode=merge_mode,
        )
        merged_x_chunks.append(xw_new)
        merged_s_chunks.append(sw_new)

    return torch.cat(merged_x_chunks, dim=0), torch.cat(merged_s_chunks, dim=0)


def merge_in_windows_with_budget(
    x: torch.Tensor,
    s: torch.Tensor,
    scorer: MergeScorer,
    window_size: int,
    n_remove: int,
    merge_mode: str = "adjacent",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Local-window merge with an explicit removal budget.

    This is a paper-fidelity helper: when local compression ratio L/N is sampled,
    we can target a specific number of removed tokens across layers instead of
    relying only on ratio scaling.
    """
    if x.shape[0] <= 1 or n_remove <= 0:
        return x, s

    x_chunks: List[torch.Tensor] = []
    s_chunks: List[torch.Tensor] = []
    capacities: List[int] = []

    for start in range(0, x.shape[0], window_size):
        end = min(start + window_size, x.shape[0])
        # to make sure it is local merging, we need to process the tokens in windows.
        xw = x[start:end]
        sw = s[start:end]
        x_chunks.append(xw)
        s_chunks.append(sw)
        # Maximum number of tokens that can be removed /merged in this window.
        capacities.append(xw.shape[0] // 2)

    total_cap = sum(capacities)
    if total_cap <= 0:
        return x, s

    n_remove = int(min(n_remove, total_cap))
    if n_remove <= 0:
        return x, s

    # Allocate removals proportionally to per-window capacity.
    alloc = [int(n_remove * cap / total_cap) for cap in capacities]
    used = sum(alloc)
    remain = n_remove - used
    if remain > 0:
        # Fill remaining budget in windows with highest residual capacity.
        residual = [cap - a for cap, a in zip(capacities, alloc)]
        order = sorted(range(len(residual)), key=lambda i: residual[i], reverse=True)
        for idx in order:
            if remain <= 0:
                break
            add = min(residual[idx], remain)
            alloc[idx] += add
            remain -= add

    merged_x_chunks: List[torch.Tensor] = []
    merged_s_chunks: List[torch.Tensor] = []
    # Now that we know how mant tokens to remove from each window, we can apply the bipartite merge.
    for xw, sw, n_select in zip(x_chunks, s_chunks, alloc):
        xw_new, sw_new = _apply_local_merge(
            x=xw,
            s=sw,
            scorer=scorer,
            n_select=n_select,
            mode=merge_mode,
        )
        merged_x_chunks.append(xw_new)
        merged_s_chunks.append(sw_new)

    return torch.cat(merged_x_chunks, dim=0), torch.cat(merged_s_chunks, dim=0)


def merge_to_target_length(
    x: torch.Tensor, s: torch.Tensor, scorer: MergeScorer, target_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Global (latent) merge until target length K is reached.

    Paper mapping:
    - Sec. 3.4 "Selection and Reconstruction": produce Z'_K and S'
    """
    target_len = max(1, min(target_len, x.shape[0]))
    while x.shape[0] > target_len:
        # How many merges we need in this round.
        need = x.shape[0] - target_len
        # Upper bound for one bipartite merge step.
        cap = x.shape[0] // 2
        n_select = min(need, cap)
        # NOTE: the mention of adajcent was mainly in the local encoder, since this is latent we leave bipartite
        x, s = _apply_bipartite_merge(
            x=x,
            s=s,
            scorer=scorer,
            n_select=n_select,
        )
    return x, s


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
        Local Encoder + source matrix update.
        # NOTE: this part of the code applies the local-window token merging.

        Returns:
        - local tokens Z_L (Eq. 2)
        - source matrix S in {0,1}^{L x N} (Eq. 2)
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
                            planned_remove = max(1, int(round(remaining_remove * (w_i / sum_w))))
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
                    planned_remove = max(1, int(round(remaining_remove * (w_i / sum_w))))
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
        Latent selective path for Sec. 3.4:
        - encode context
        - select K salient tokens via merge
        - decode at K
        - unmerge back to L
        """
        # Start from output of local encoder.
        x = z_l
        for block in self.latent_encoder:
            x = block(x)

        # get global context
        l = x.shape[0]
        k = max(1, int(l * keep_ratio)) # number of tokens to keep
        # Initiliase the source matrix S' as an identity matrix. Source matrix is a matrix that maps the latent tokens to the local tokens.
        s_prime = torch.eye(l, device=x.device, dtype=x.dtype)  # maps K <- L after merges
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
