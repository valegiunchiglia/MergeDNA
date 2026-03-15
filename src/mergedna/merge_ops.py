from __future__ import annotations

from typing import List, Tuple

import torch

from .scoring import MergeScorer


def _apply_bipartite_merge(
    x: torch.Tensor, s: torch.Tensor, scorer: MergeScorer, n_select: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply one ToMe-style bipartite merge step on a token sequence.

    This is the local merge primitive used by `merge_mode="bipartite"` and
    by latent global selection (`merge_to_target_length`). It follows the
    A/B partition strategy from ToMe:
    - split tokens into A (even positions) and B (odd positions),
    - compute best B destination for each A token,
    - keep the top `n_select` A->B matches by score,
    - merge A source tokens into chosen B destination tokens.

    Merge update rule:
    - features use size-weighted averaging,
    - source rows are summed (`S_new = S_dst + S_src`) to preserve coverage.

    Args:
        x: Token embeddings `[T, D]`.
        s: Source matrix `[T, N]` mapping token rows to original positions.
        scorer: Similarity module producing A->B best matches.
        n_select: Number of A tokens to merge into B in this step.

    Returns:
        `(x_new, s_new)` with shapes `[(T - n_select), D]` and
        `[(T - n_select), N]`.

    Note:
        Output order is ToMe-style (`unmerged A` then `updated B`), so unlike
        adjacent mode this does not strictly preserve original token order.
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
    Merge only adjacent pairs `(i, i+1)` within one window.

    This mode is the most biologically conservative interpretation of the
    MergeDNA local tokenizer text ("chunk adjacent bases into words") because
    it enforces contiguity.

    Args:
        x: Token embeddings `[T, D]` for one local window.
        s: Source matrix `[T, N]` aligned with `x`.
        scorer: Similarity module used to score adjacent pairs.
        n_select: Number of non-overlapping adjacent pairs to merge.

    Returns:
        `(x_new, s_new)` after merging up to `n_select` pairs.
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
    Merge mode that scores all pairs inside one window.

    Procedure:
    - build full pairwise similarity on `[T, D]`,
    - rank upper-triangular pairs `(i, j), i < j`,
    - greedily select top non-overlapping pairs,
    - merge each selected pair with size-weighted averaging.

    Args:
        x: Token embeddings `[T, D]`.
        s: Source matrix `[T, N]`.
        scorer: Similarity module.
        n_select: Maximum number of non-overlapping pairs to merge.

    Returns:
        `(x_new, s_new)` with reduced token count.

    Note:
        This matches the statement in the paper where it says that it "score each pair in a local window" literally,
        but is more expensive than adjacent/bipartite in practice, and shouldn't be used in the local encoder.
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
    Apply one local merge layer independently per window using a ratio budget.

    Paper mapping:
    - Sec. 3.3 / Eq. (5): local-window token merging with source updates.

    Args:
        x: Input tokens `[T, D]`.
        s: Source matrix `[T, N]`.
        scorer: Similarity module for pair selection.
        window_size: Local chunk size used for independent merging.
        merge_ratio: Fraction of tokens to remove per window (approximately).
        merge_mode: One of `adjacent`, `bipartite`, `full_pairwise`.

    Returns:
        `(x_new, s_new)` concatenated from all window-level merges.

    Note:
        This is the simpler per-layer ratio schedule. It does not guarantee a
        precise final target length across all local encoder layers.
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
    Apply one local merge layer with an explicit token-removal budget.

    Compared with `merge_in_windows`, this function tries to remove exactly
    `n_remove` tokens (subject to window capacity) by:
    1) computing each window capacity `floor(T_w/2)`,
    2) allocating budget proportionally to capacities,
    3) distributing leftover removals by residual capacity,
    4) running selected merge mode per window with the allocated counts.

    Args:
        x: Input token embeddings `[T, D]`.
        s: Source matrix `[T, N]`.
        scorer: Similarity module.
        window_size: Local chunk length.
        n_remove: Total tokens to remove in this layer.
        merge_mode: One of `adjacent`, `bipartite`, `full_pairwise`.

    Returns:
        `(x_new, s_new)` after budgeted per-window merges.

    Why this exists:
        It supports paper-style sampled compression by helping the local
        encoder hit a global target length more closely over multiple layers.
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
    Iteratively merge tokens until a latent target length `K` is reached.

    Paper mapping:
    - Sec. 3.4 ("Selection and Reconstruction"): produce `Z'_K` and `S'`.

    Args:
        x: Latent-token sequence `[L, D]`.
        s: Latent source matrix `[L, L]` (initialized as identity).
        scorer: Similarity module used for merge matching.
        target_len: Desired length `K` with `1 <= K <= L`.

    Returns:
        `(z_k, s_prime)` where:
        - `z_k` is `[K, D]`,
        - `s_prime` is `[K, L]`, mapping selected latent tokens back to local
          tokens for unmerging and AMTM importance estimation.
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
