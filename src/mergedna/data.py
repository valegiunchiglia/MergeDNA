import random
import gzip
from pathlib import Path
from typing import Iterator, List

import torch
from torch.utils.data import Dataset


# DNA token mapping used across the project.
DNA_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}
ID_TO_DNA = {v: k for k, v in DNA_TO_ID.items()}


def encode_dna(seq: str) -> List[int]:
    """
    Convert DNA string to integer ids.

    Non-ACGT characters are skipped to keep this implementation simple and safe.
    """
    ids = []
    for ch in seq.upper():
        if ch in DNA_TO_ID:
            ids.append(DNA_TO_ID[ch])
    return ids


def _iter_fasta_sequences(path: str) -> Iterator[str]:
    """
    Stream FASTA sequences one record at a time.

    Supports plain text FASTA and gzip-compressed FASTA (.gz).
    """
    p = Path(path)
    # Open the file in text mode and gz mode
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        current: List[str] = []
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    yield "".join(current)
                    current = []
                continue
            current.append(line)
        if current:
            yield "".join(current)


def load_fasta_sequences(path: str) -> List[str]:
    """
    Minimal FASTA parser.

    We avoid external bio packages in this implementation.
    """
    return list(_iter_fasta_sequences(path))


class RandomDNADataset(Dataset):
    """
    Synthetic DNA dataset used for smoke tests and debugging.
    """

    def __init__(self, n_samples: int, seq_len: int, seed: int = 0) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Random integer DNA tokens in [0, 3].
        _ = idx
        return torch.tensor(
            [self.rng.randint(0, 3) for _ in range(self.seq_len)], dtype=torch.long
        )


class FASTADataset(Dataset):
    """
    FASTA window dataset.

    For each item, sample a random window from a random sequence.
    This approximates random chunking used in many sequence pretraining pipelines.
    """

    def __init__(
        self, fasta_path: str, seq_len: int, n_samples: int, seed: int = 0
    ) -> None:
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.rng = random.Random(seed)
        self.sequences = [encode_dna(s) for s in load_fasta_sequences(fasta_path)]
        self.sequences = [s for s in self.sequences if len(s) >= seq_len]
        if not self.sequences:
            raise ValueError(
                f"No sequences with length >= {seq_len} found in {fasta_path}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        _ = idx
        seq = self.sequences[self.rng.randint(0, len(self.sequences) - 1)]
        start = self.rng.randint(0, len(seq) - self.seq_len)
        window = seq[start : start + self.seq_len]
        return torch.tensor(window, dtype=torch.long)
