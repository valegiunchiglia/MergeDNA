"""
Data utilities for downstream genomics evaluation.

This module contains:
- task-group definitions,
- raw CSV/TSV loading helpers,
- synthetic task generation,
- tokenization dataset/dataloader builders.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from mergedna.data import encode_dna


GENOMICS_TASK_GROUPS: Dict[str, List[str]] = {
    "enhancer": [
        "mouse_enhancers",
        "human_enhancers_cohn",
        "human_enhancers_ensembl",
    ],
    "species": [
        "coding_vs_intergenomic",
        "human_or_worm",
    ],
    "regulatory": [
        "human_regulatory",
        "human_ocr_ensembl",
        "human_nontata_promoters",
    ],
}

# Potential names
SEQ_COL_CANDIDATES = ("sequence", "seq", "dna", "text")
LABEL_COL_CANDIDATES = ("label", "target", "y")
DNA_ALPHABET = ("A", "C", "G", "T")
PROTEIN_SEQ_COL_CANDIDATES = ("sequence", "seq", "dna", "protein", "text")
FITNESS_COL_CANDIDATES = ("fitness", "target", "y", "label")


def task_seed(base_seed: int, task_name: str) -> int:
    """
    Deterministically derive a per-task seed from global seed and task name.
    """
    return base_seed + sum(ord(c) for c in task_name)


def _rand_dna(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(DNA_ALPHABET) for _ in range(length))


def _inject_motif(seq: str, motif: str, rng: random.Random) -> str:
    """
    Insert motif at a random valid position in sequence.
    """
    if len(motif) > len(seq):
        return seq
    start = rng.randint(0, len(seq) - len(motif))
    return seq[:start] + motif + seq[start + len(motif) :]


def make_synthetic_binary_task(
    n_samples: int,
    seq_len: int,
    motif: str,
    seed: int,
    positive_rate: float = 0.5,
) -> Tuple[List[str], List[int]]:
    """
    Build synthetic binary classification data for quick evaluation sanity checks.

    Label rule:
    - y=1: motif is injected into random DNA sequence.
    - y=0: random DNA sequence without injection.

    Returns:
    - sequences: List[str], length n_samples
    - labels: List[int], length n_samples
    """
    rng = random.Random(seed)
    sequences: List[str] = []
    labels: List[int] = []
    for _ in range(n_samples):
        y = 1 if rng.random() < positive_rate else 0
        seq = _rand_dna(rng, seq_len)
        if y == 1:
            seq = _inject_motif(seq, motif=motif, rng=rng)
        sequences.append(seq)
        labels.append(y)
    return sequences, labels


def load_task_synthetic(
    task_name: str,
    seq_len: int,
    seed: int,
    train_size: int,
    val_size: int,
    test_size: int,
    positive_rate: float,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Build synthetic train/val/test splits for one task name.

    Motif is chosen deterministically from task name to give each task a distinct
    synthetic pattern while keeping reproducibility.
    """
    motif_bank = ["TATAAA", "CGCG", "CAGGTG", "AATAAA", "GATA", "CCGCGG", "GTAG", "CAAT"]
    motif = motif_bank[task_seed(seed, task_name) % len(motif_bank)]
    train_seq, train_y = make_synthetic_binary_task(
        n_samples=train_size,
        seq_len=seq_len,
        motif=motif,
        seed=task_seed(seed, f"{task_name}:train"),
        positive_rate=positive_rate,
    )
    val_seq, val_y = make_synthetic_binary_task(
        n_samples=val_size,
        seq_len=seq_len,
        motif=motif,
        seed=task_seed(seed, f"{task_name}:val"),
        positive_rate=positive_rate,
    )
    test_seq, test_y = make_synthetic_binary_task(
        n_samples=test_size,
        seq_len=seq_len,
        motif=motif,
        seed=task_seed(seed, f"{task_name}:test"),
        positive_rate=positive_rate,
    )
    return train_seq, train_y, val_seq, val_y, test_seq, test_y


def detect_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return ","


def resolve_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str:
    lower_to_original = {k.lower(): k for k in fieldnames}
    for cand in candidates:
        if cand in lower_to_original:
            return lower_to_original[cand]
    raise ValueError(f"Missing required column. Candidates: {candidates}. Found: {fieldnames}")


def load_labeled_sequences(path: str) -> Tuple[List[str], List[int]]:
    """
    Load one split file (CSV/TSV) into raw sequence strings and integer labels.

    Expected file format:
    - Header row required.
    - Sequence column name can be one of:
      `sequence`, `seq`, `dna`, `text`
    - Label column name can be one of:
      `label`, `target`, `y`
    - Delimiter can be comma, tab, or semicolon (auto-detected).

    Example rows:
    - sequence,label
      ACGTACGT...,1
      TTGCA...,0

    Returns:
    - `sequences`: List[str], length `N`
    - `labels`: List[int], length `N`
    """
    delimiter = detect_delimiter(path)
    sequences: List[str] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        seq_col = resolve_column(reader.fieldnames, SEQ_COL_CANDIDATES)
        label_col = resolve_column(reader.fieldnames, LABEL_COL_CANDIDATES)
        for row in reader:
            seq = row[seq_col].strip()
            label = int(row[label_col])
            sequences.append(seq)
            labels.append(label)
    return sequences, labels


def load_task_raw(
    data_root: str, task_name: str
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Load one task's train/val/test raw splits from disk.

    Expected directory layout:
    - `<data_root>/<task_name>/train.csv`
    - `<data_root>/<task_name>/val.csv`
    - `<data_root>/<task_name>/test.csv`

    Each file must follow `load_labeled_sequences` format
    (header + sequence/label columns).

    Returns:
    - `train_seq`: List[str] of length `N_train`
    - `train_y`: List[int] of length `N_train`
    - `val_seq`: List[str] of length `N_val`
    - `val_y`: List[int] of length `N_val`
    - `test_seq`: List[str] of length `N_test`
    - `test_y`: List[int] of length `N_test`
    """
    train_path = os.path.join(data_root, task_name, "train.csv")
    val_path = os.path.join(data_root, task_name, "val.csv")
    test_path = os.path.join(data_root, task_name, "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Expected train/val/test files for task '{task_name}' under {os.path.join(data_root, task_name)}"
        )
    train_seq, train_y = load_labeled_sequences(train_path)
    val_seq, val_y = load_labeled_sequences(val_path)
    test_seq, test_y = load_labeled_sequences(test_path)
    return train_seq, train_y, val_seq, val_y, test_seq, test_y


def encode_and_fit_length(seq: str, seq_len: int) -> List[int]:
    ids = encode_dna(seq)
    if len(ids) >= seq_len:
        return ids[:seq_len]
    # Right-pad with "A" token id (0) for short sequences.
    return ids + [0] * (seq_len - len(ids))


@dataclass
class SequenceLabelItem:
    tokens: torch.Tensor
    label: int


class SequenceLabelDataset(Dataset):
    """
    Dataset of fixed-length tokenized DNA sequences with class labels.

    Input:
    - `sequences`: List[str]
    - `labels`: List[int]
    - `seq_len`: int (target fixed length)

    Processing:
    - DNA string -> token ids using `encode_dna` (`A/C/G/T -> 0/1/2/3`)
    - truncate if length > `seq_len`
    - right-pad with `0` if length < `seq_len`

    __getitem__ output shapes:
    - `tokens`: torch.LongTensor of shape `[seq_len]`
    - `label`: torch.LongTensor scalar (`[]`)
    """

    def __init__(self, sequences: List[str], labels: List[int], seq_len: int) -> None:
        if len(sequences) != len(labels):
            raise ValueError("Sequences and labels must have equal length.")
        self.items: List[SequenceLabelItem] = []
        for seq, label in zip(sequences, labels):
            ids = encode_and_fit_length(seq, seq_len=seq_len)
            self.items.append(
                SequenceLabelItem(tokens=torch.tensor(ids, dtype=torch.long), label=int(label))
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        return item.tokens, torch.tensor(item.label, dtype=torch.long)


def make_loaders(
    train_seq: List[str],
    train_y: List[int],
    val_seq: List[str],
    val_y: List[int],
    test_seq: List[str],
    test_y: List[int],
    seq_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for one task after tokenization/padding.

    Batch output shapes:
    - `batch_tokens`: `[B, seq_len]` (LongTensor)
    - `batch_labels`: `[B]` (LongTensor)
    """
    train_ds = SequenceLabelDataset(train_seq, train_y, seq_len=seq_len)
    val_ds = SequenceLabelDataset(val_seq, val_y, seq_len=seq_len)
    test_ds = SequenceLabelDataset(test_seq, test_y, seq_len=seq_len)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def infer_num_classes_from_labels(*label_lists: List[int]) -> int:
    """
    Infer number of classes from integer-encoded label lists.

    This function returns the number of unique labels across provided splits.
    While collecting labels, values are cast to `int` for robustness.

    Args:
    - `*label_lists`: one or more label lists (e.g., train/val/test)

    Returns:
    - `num_classes` as number of unique labels, or `1` if all inputs are empty.
    """
    all_labels = set()
    for labels in label_lists:
        all_labels.update(int(x) for x in labels)
    return len(all_labels) if all_labels else 1


def remap_labels_to_contiguous(
    *label_lists: List[int],
) -> Tuple[List[List[int]], Dict[int, int]]:
    """
    Remap arbitrary integer labels to contiguous class ids `[0, C-1]`.

    This avoids mis-sized classifier heads and out-of-range class targets when
    datasets use sparse ids (e.g., labels `{1, 2}` or `{0, 2, 5}`).

    Args:
    - `*label_lists`: one or more label lists (e.g., train/val/test)

    Returns:
    - `remapped_lists`: list of remapped label lists in the same order,
    - `label_to_id`: mapping from original label -> contiguous id.
    """
    unique_labels = sorted({int(x) for labels in label_lists for x in labels})
    if not unique_labels:
        return [list(labels) for labels in label_lists], {}
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    remapped_lists: List[List[int]] = []
    for labels in label_lists:
        remapped_lists.append([label_to_id[int(x)] for x in labels])
    return remapped_lists, label_to_id


def load_labeled_regression(path: str) -> Tuple[List[str], List[float]]:
    """
    Load one split file (CSV/TSV) into raw sequence strings and scalar targets.
    """
    delimiter = detect_delimiter(path)
    sequences: List[str] = []
    fitness: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        seq_col = resolve_column(reader.fieldnames, PROTEIN_SEQ_COL_CANDIDATES)
        fit_col = resolve_column(reader.fieldnames, FITNESS_COL_CANDIDATES)
        for row in reader:
            sequences.append(row[seq_col].strip())
            fitness.append(float(row[fit_col]))
    return sequences, fitness


def load_task_regression_raw(
    data_root: str, task_name: str
) -> Tuple[List[str], List[float], List[str], List[float], List[str], List[float]]:
    """
    Load train/val/test regression splits from:
    - <data_root>/<task_name>/train.csv
    - <data_root>/<task_name>/val.csv
    - <data_root>/<task_name>/test.csv
    """
    train_path = os.path.join(data_root, task_name, "train.csv")
    val_path = os.path.join(data_root, task_name, "val.csv")
    test_path = os.path.join(data_root, task_name, "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Expected train/val/test files for task '{task_name}' under {os.path.join(data_root, task_name)}"
        )
    train_seq, train_y = load_labeled_regression(train_path)
    val_seq, val_y = load_labeled_regression(val_path)
    test_seq, test_y = load_labeled_regression(test_path)
    return train_seq, train_y, val_seq, val_y, test_seq, test_y


def _rand_sequence(rng: random.Random, alphabet: Sequence[str], length: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def _synthetic_fitness_score(seq: str, motif: str, rng: random.Random) -> float:
    # Count motif occurrences + small Gaussian noise.
    count = 0
    i = 0
    while i <= len(seq) - len(motif):
        if seq[i : i + len(motif)] == motif:
            count += 1
            i += len(motif)
        else:
            i += 1
    return float(count) + rng.gauss(0.0, 0.1)


def make_synthetic_regression_split(
    n_samples: int,
    seq_len: int,
    alphabet: Sequence[str],
    motif: str,
    seed: int,
) -> Tuple[List[str], List[float]]:
    """
    Synthetic regression data:
    - sequence is random over alphabet,
    - target is motif-count + small noise.
    """
    rng = random.Random(seed)
    seqs: List[str] = []
    y: List[float] = []
    for _ in range(n_samples):
        seq = _rand_sequence(rng, alphabet, seq_len)
        seqs.append(seq)
        y.append(_synthetic_fitness_score(seq, motif=motif, rng=rng))
    return seqs, y


def load_task_regression_synthetic(
    seq_len: int,
    alphabet: Sequence[str],
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> Tuple[List[str], List[float], List[str], List[float], List[str], List[float]]:
    motif = "".join(alphabet[: min(3, len(alphabet))]) if alphabet else "ACG"
    train_seq, train_y = make_synthetic_regression_split(
        n_samples=train_size,
        seq_len=seq_len,
        alphabet=alphabet,
        motif=motif,
        seed=seed + 1,
    )
    val_seq, val_y = make_synthetic_regression_split(
        n_samples=val_size,
        seq_len=seq_len,
        alphabet=alphabet,
        motif=motif,
        seed=seed + 2,
    )
    test_seq, test_y = make_synthetic_regression_split(
        n_samples=test_size,
        seq_len=seq_len,
        alphabet=alphabet,
        motif=motif,
        seed=seed + 3,
    )
    return train_seq, train_y, val_seq, val_y, test_seq, test_y


def build_alphabet_map(alphabet: str) -> Dict[str, int]:
    letters = [c for c in alphabet.strip().upper() if c]
    if len(letters) == 0:
        raise ValueError("Alphabet must contain at least one character.")
    unique = list(dict.fromkeys(letters))
    return {ch: i for i, ch in enumerate(unique)}


def encode_with_alphabet(seq: str, seq_len: int, alpha_map: Dict[str, int]) -> List[int]:
    ids: List[int] = []
    for ch in seq.upper():
        if ch not in alpha_map:
            raise ValueError(f"Unknown token '{ch}' for alphabet {''.join(alpha_map.keys())}")
        ids.append(alpha_map[ch])
    if len(ids) >= seq_len:
        return ids[:seq_len]
    return ids + [0] * (seq_len - len(ids))


@dataclass
class RegressionItem:
    tokens: torch.Tensor
    target: float


class RegressionSequenceDataset(Dataset):
    def __init__(self, sequences: List[str], targets: List[float], seq_len: int, alphabet: str) -> None:
        if len(sequences) != len(targets):
            raise ValueError("Sequences and targets must have equal length.")
        self.alpha_map = build_alphabet_map(alphabet)
        self.items: List[RegressionItem] = []
        for seq, target in zip(sequences, targets):
            ids = encode_with_alphabet(seq, seq_len=seq_len, alpha_map=self.alpha_map)
            self.items.append(
                RegressionItem(tokens=torch.tensor(ids, dtype=torch.long), target=float(target))
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        return item.tokens, torch.tensor(item.target, dtype=torch.float32)


def make_regression_loaders(
    train_seq: List[str],
    train_y: List[float],
    val_seq: List[str],
    val_y: List[float],
    test_seq: List[str],
    test_y: List[float],
    seq_len: int,
    batch_size: int,
    alphabet: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = RegressionSequenceDataset(train_seq, train_y, seq_len=seq_len, alphabet=alphabet)
    val_ds = RegressionSequenceDataset(val_seq, val_y, seq_len=seq_len, alphabet=alphabet)
    test_ds = RegressionSequenceDataset(test_seq, test_y, seq_len=seq_len, alphabet=alphabet)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
