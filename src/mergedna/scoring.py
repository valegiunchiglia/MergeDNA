from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
