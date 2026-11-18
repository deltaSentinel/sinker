"""
Stage 1 data handling for the URL signal.

Raw format: TSV with columns  url <TAB> label   (0 = benign, 1 = phishing),
one row per URL. Kept deliberately dumb so the feeds (stage 2) can drop
files into data/raw/ in any cadence without touching the training path.

Feature representation: URLs are lowercased, truncated to MAX_LEN
characters, and encoded as indices over a fixed ASCII-ish vocabulary
(no learned tokenizer needed at this stage).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MAX_LEN = 200

# Vocabulary: index 0 is padding. Printable ASCII (32..126) maps to itself
# minus offset, so any "normal" URL char needs no giant lookup table. Bytes
# outside the range (percent-encodings stay ASCII, so this is rare) map to
# UNK=1. Two reserved slots up front, rest is 1:1 with ord(c) - 30.
PAD_IDX = 0
UNK_IDX = 1
VOCAB_SIZE = 128


def encode_url(url: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Char-level encoding of one URL to a fixed-length index vector."""
    url = url.strip().lower()[:max_len]
    out = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(url):
        code = ord(ch)
        out[i] = code if code < VOCAB_SIZE else UNK_IDX
        # NOTE: printable ASCII maps to itself; control chars <32 are rare in
        # URLs and effectively become near-zero-frequency tokens, which is fine.
    return out


@dataclass
class Dataset:
    X: np.ndarray  # (N, MAX_LEN) int64
    y: np.ndarray  # (N,) float32
    urls: list[str]


def load_tsv(path: str | Path) -> Dataset:
    """Load url<TAB>label rows. Malformed rows are skipped and counted."""
    rows, labels, raw, skipped = [], [], [], 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2 or parts[1].strip() not in ("0", "1"):
                skipped += 1
                continue
            raw.append(parts[0])
            rows.append(encode_url(parts[0]))
            labels.append(float(parts[1].strip()))
    if not rows:
        raise ValueError(f"no usable rows found in {path}")
    print(f"[data] loaded {len(rows)} rows ({skipped} skipped) from {path}")
    return Dataset(
        X=np.stack(rows),
        y=np.asarray(labels, dtype=np.float32),
        urls=raw,
    )


def stratified_split(ds: Dataset, val_frac: float, seed: int):
    """Index-level stratified split; returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in (0.0, 1.0):
        idx = np.where(ds.y == cls)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * val_frac)
        val_idx.extend(idx[:cut])
        train_idx.extend(idx[cut:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def make_tensor_ds(ds: Dataset, idx: np.ndarray):
    return torch.utils.data.TensorDataset(
        torch.from_numpy(ds.X[idx]),
        torch.from_numpy(ds.y[idx]),
    )
