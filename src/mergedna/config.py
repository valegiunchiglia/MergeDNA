from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MergeDNAConfig:
    """
    Model architecture config.

    These defaults are intentionally smaller than the paper's 380M setup so the
    project can run locally for a technical challenge. You can scale these up.
    """

    vocab_size: int = 4  # DNA alphabet: A, C, G, T
    max_seq_len: int = 4096
    d_model: int = 256
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    # Transformer block internals:
    # - "llama": RMSNorm + SwiGLU + pre-norm residual
    # - "standard": LayerNorm + GELU FFN + post-norm residual
    block_style: str = "llama"

    # Local encoder/decoder depth (paper uses 4 and 2 respectively).
    local_encoder_layers: int = 4
    local_decoder_layers: int = 2

    # Latent encoder/decoder depth (paper uses 20 and 4 respectively).
    latent_encoder_layers: int = 6
    latent_decoder_layers: int = 2

    # Local attention window (paper reports 16).
    local_window_size: int = 16

    # Per-layer merge ratio in Local Encoder; each value is the fraction of
    # possible adjacent pairs merged per local layer window.
    local_merge_ratios: List[float] = field(
        default_factory=lambda: [0.25, 0.25, 0.20, 0.15]
    )
    # Local merge operator:
    # - adjacent: merge only adjacent token pairs (paper-faithful default)
    # - bipartite: ToMe-style bipartite matching within each local window
    # - full_pairwise: score all token pairs in each window
    local_merge_mode: str = "adjacent"

    # Latent compression ratio K/L (paper conceptually uses K < L, often L/2).
    latent_keep_ratio: float = 0.5
    # Latent selective merge behavior:
    # - posthoc: run latent encoder at L, then one global merge L->K
    # - interleaved: merge progressively between latent encoder blocks (paper-closer)
    latent_selective_mode: str = "interleaved"  # posthoc | interleaved

    # Compression ratio sampling range around expected L for local tokenizer.
    # Used during training for robustness (paper Sec. 3.3).
    local_keep_ratio_min: float = 0.4
    local_keep_ratio_max: float = 0.6
    local_keep_ratio_mean: float = 0.5
    local_keep_ratio_std: float = 0.05
    local_keep_ratio_sampling: str = "gaussian"  # gaussian | uniform

    # If True, Local Encoder tries to hit sampled target length more explicitly
    # by assigning per-layer merge budgets instead of only ratio scaling.
    enforce_sampled_local_target_len: bool = True


@dataclass
class MergeDNATrainConfig:
    """
    Training config with practical ML defaults.
    """

    seed: int = 42
    device: str = "cpu"

    # Data
    seq_len: int = 512
    batch_size: int = 8
    num_workers: int = 0
    train_samples: int = 2000
    val_samples: int = 200
    train_fasta: Optional[str] = None
    val_fasta: Optional[str] = None

    # Optimization
    steps: int = 1000
    lr: float = 1e-4
    weight_decay: float = 1e-8
    betas: tuple = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    amp: bool = True
    # Paper-aligned scheduler defaults:
    # Cosine annealing with linear warmup.
    lr_scheduler: str = "cosine"  # cosine | none
    warmup_steps: int = 10_000

    # Loss weights
    lambda_latent_mtr: float = 0.25  # as paper Eq. (8)

    # Logging/checkpoints
    log_every: int = 10
    eval_every: int = 100
    ckpt_every: int = 200
    out_dir: str = "checkpoints"

    # Optional Weights & Biases logging
    use_wandb: bool = False
    wandb_project: str = "mergedna"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # online | offline | disabled
