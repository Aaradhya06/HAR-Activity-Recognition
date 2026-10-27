"""
Activity Classification Head
==============================
Final MLP classification head that maps the fused representation
to activity class logits.

The head is intentionally shallow to avoid overfitting when training
on small subsets of UCF101/HMDB51.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Classifier(nn.Module):
    """
    Multi-layer classification head with dropout regularization.

    Args:
        input_dim: Fused embedding dimension from fusion layer.
        num_classes: Number of activity classes.
        hidden_dim: Optional hidden layer (0 = linear projection only).
        dropout: Dropout probability before final linear.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_classes),
            )

        logger.info(
            "Classifier: input=%d, hidden=%d, classes=%d",
            input_dim, hidden_dim, num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) — fused feature vector.

        Returns:
            (B, num_classes) — raw class logits (no softmax).
        """
        return self.head(x)

    def predict(self, x: torch.Tensor, top_k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run prediction and return top-k probabilities and indices.

        Args:
            x: (B, input_dim)
            top_k: Number of top predictions to return.

        Returns:
            (probs, indices) — each (B, top_k)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            k = min(top_k, self.num_classes)
            top_probs, top_idx = torch.topk(probs, k, dim=-1)
        return top_probs, top_idx
