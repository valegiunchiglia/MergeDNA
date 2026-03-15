from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from .model import ForwardOutput, MergeDNA


def mtr_loss(outputs: List[ForwardOutput], targets: torch.Tensor) -> torch.Tensor:
    """
    Compute `L_MTR` as base-level cross-entropy reconstruction.

    Paper mapping:
    - Eq. (6): Merged Token Reconstruction objective.

    Args:
        outputs: Per-sample model outputs; each `out.logits` is `[N, 4]`.
        targets: Ground-truth base ids `[B, N]`.

    Returns:
        Scalar mean cross-entropy across batch samples.
    """
    # Reconstruction loss - main - without latent selective merge
    losses = []
    for i, out in enumerate(outputs):
        # out.logits: [N, 4], targets[i]: [N]
        losses.append(F.cross_entropy(out.logits, targets[i]))
    return torch.stack(losses).mean()


def _compute_local_importance_probs(s_prime: torch.Tensor) -> torch.Tensor:
    """
    Build token importance distribution P_L from latent grouping matrix S'.

    Paper mapping (Sec. 3.4):
    - g_i = group size of latent token i
    - w_i = 1 / g_i
    - token prob in group i proportional to w_i / g_i = 1 / g_i^2

    Args:
        s_prime: [K, L] binary-ish source matrix mapping latent->local tokens
    Returns:
        probs: [L], sum to 1
    """
    # Accoridng to the paper, higher weight is assigned to smaller groups (more important)
    # g_i: number of local tokens represented by latent token i
    g = s_prime.sum(dim=1).clamp(min=1.0)  # [K]
    group_weight = 1.0 / (g * g)  # [K]

    # For each local token j, find its owning latent group i (one-hot partition).
    # If soft values appear, this still works by weighted accumulation.
    # token_weight[j] = sum_i S'[i,j] * group_weight[i]
    token_weight = s_prime.transpose(0, 1) @ group_weight  # [L]
    probs = token_weight / token_weight.sum().clamp(min=1e-8)
    return probs


def sample_amtm_masks(
    outputs_latent: List[ForwardOutput], targets: torch.Tensor
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Sample AMTM masks from latent grouping structure.

    Paper mapping:
    - Sec. 3.4 Adaptive Masked Token Modeling.
    - Build `P_L` from latent groups (`~1/g_i^2`), sample `K` local tokens
      without replacement, then map mask to base space via source matrix `S`.

    Args:
        outputs_latent: Outputs from latent-selective pass. Each item must
            provide `latent_source_matrix` (`S'`) and local `source_matrix` (`S`).
        targets: Target batch `[B, N]`, used for device placement.

    Returns:
        - `masks_local`: list of boolean local-token masks `[L]`,
        - `masks_base`: list of boolean base-token masks `[N]`.
    """
    masks_local: List[torch.Tensor] = []
    masks_base: List[torch.Tensor] = []

    for i, out in enumerate(outputs_latent):
        if out.latent_source_matrix is None:
            raise ValueError("AMTM requires latent_source_matrix from latent_selective pass.")
        s_prime = out.latent_source_matrix  # [K, L]
        s_local_to_base = out.source_matrix  # [L, N]
        # l is the number of local tokens and k is the number of latent tokens
        l = s_prime.shape[1]
        k = s_prime.shape[0]

        probs = _compute_local_importance_probs(s_prime=s_prime)
        # Sample exactly K local tokens without replacement (paper style).
       
        k_sample = min(k, l)
        # Randomly smaple k samples according to the importance probabilities
        idx = torch.multinomial(probs, num_samples=k_sample, replacement=False)
        mask_l = torch.zeros(l, dtype=torch.bool, device=targets.device)
        mask_l[idx] = True

        # Map local mask back to base space via source matrix S:
        # M_N = U(M_L, S) (Sec. 3.4), where S is [L, N].
        mask_l_float = mask_l.to(dtype=s_local_to_base.dtype)
        mask_n = (mask_l_float.unsqueeze(0) @ s_local_to_base).squeeze(0) > 0

        masks_local.append(mask_l)
        masks_base.append(mask_n)

    return masks_local, masks_base


def amtm_loss(
    model: MergeDNA,
    batch_tokens: torch.Tensor,
    masks_base: List[torch.Tensor],
    sampled_local_keep_ratio: float,
) -> torch.Tensor:
    """
    Compute AMTM loss (`L_AMTM`) on informative masked base positions.

    Paper mapping:
    - Eq. (7): masked modeling loss over selected informative tokens.

    Implementation details:
    - Replace masked base positions with model `mask_token_id`.
    - Run a standard (non-latent-selective) forward pass for prediction.
    - Compute CE only on masked base positions for each sample.

    Args:
        model: MergeDNA model instance.
        batch_tokens: Input base tokens `[B, N]`.
        masks_base: Boolean masks per sample, each `[N]`.
        sampled_local_keep_ratio: Local tokenizer keep ratio for this step.

    Returns:
        Scalar AMTM loss averaged across valid samples.
    """
    masked_batch = batch_tokens.clone()
    for i, mask_n in enumerate(masks_base):
        masked_batch[i][mask_n] = model.mask_token_id

    outputs = model.forward_batch(
        masked_batch,
        sampled_local_keep_ratio=sampled_local_keep_ratio,
        latent_selective=False,
        freeze_local_encoder=False,
    )

    losses = []
    for i, out in enumerate(outputs):
        mask_n = masks_base[i]
        if mask_n.sum() == 0:
            continue
        logits_masked = out.logits[mask_n]  # [K_eff, 4]
        targets_masked = batch_tokens[i][mask_n]  # [K_eff]
        losses.append(F.cross_entropy(logits_masked, targets_masked))
    if not losses:
        return torch.tensor(0.0, device=batch_tokens.device)
    return torch.stack(losses).mean()
