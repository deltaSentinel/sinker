"""Signal `hook`: character-level CNN URL classifier."""
from __future__ import annotations

import torch
import torch.nn as nn

from .data import MAX_LEN, VOCAB_SIZE


class HookCNN(nn.Module):
    """
    char-embedding -> parallel 1D convs (n-gram-like receptive fields)
    -> global max pool -> dense -> single logit (phishing score).

    Small enough to train on CPU in minutes at stage-1 data scale.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=k)
                for k in kernel_sizes
            ]
        )
        feat_dim = num_filters * len(kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, MAX_LEN) int64 -> (B,) logits."""
        emb = self.embedding(x).transpose(1, 2)      # (B, embed_dim, L)
        pooled = [
            torch.relu(conv(emb)).max(dim=2).values for conv in self.convs
        ]
        feats = torch.cat(pooled, dim=1)             # (B, feat_dim)
        return self.head(self.dropout(feats)).squeeze(1)
