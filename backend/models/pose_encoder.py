"""
Pose Feature Encoder
=====================
Encodes human body keypoints (from MediaPipe Pose) into a compact
pose representation that is fused with the CNN/temporal visual stream.

Input:  (B, T, num_keypoints * 3) — flattened (x, y, visibility) per frame
Output: (B, pose_output_dim)      — aggregated pose embedding

The encoder:
  1. Per-frame MLP encoding
  2. Temporal mean-pooling
  3. Projection to output_dim

This deliberately lightweight design avoids adding heavy graph networks;
a GCN-based encoder could be substituted here as a future extension.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PoseEncoder(nn.Module):
    """
    MLP-based pose feature encoder for MediaPipe keypoints.

    Args:
        input_dim: Flattened keypoint dimension (num_kp * 3).
                   MediaPipe Pose: 33 keypoints → 99.
        hidden_dim: Intermediate MLP dimension.
        output_dim: Final pose embedding dimension.
        dropout: Dropout probability.
        num_frames: Sequence length (used for temporal pooling hint only).
    """

    def __init__(
        self,
        input_dim: int = 99,     # 33 × 3
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.2,
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Per-frame keypoint encoder
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal aggregation: lightweight 1D temporal attention
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, 1),  # score per frame
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        logger.info(
            "PoseEncoder: input=%d, hidden=%d, output=%d",
            input_dim, hidden_dim, output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pose keypoint sequence (B, T, input_dim).
               Missing keypoints should be filled with zeros before passing.

        Returns:
            (B, output_dim) — aggregated pose embedding.
        """
        B, T, _ = x.shape
        frame_feats = self.frame_encoder(x)          # (B, T, hidden_dim)

        # Soft temporal attention pooling
        scores = self.temporal_attn(frame_feats)     # (B, T, 1)
        weights = torch.softmax(scores, dim=1)       # (B, T, 1)
        pooled = (frame_feats * weights).sum(dim=1)  # (B, hidden_dim)

        return self.output_proj(pooled)              # (B, output_dim)


class NullPoseEncoder(nn.Module):
    """
    Drop-in replacement when pose estimation is disabled.
    Always returns a zero tensor with the correct output shape.
    """

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim
        logger.info("PoseEncoder: using NullPoseEncoder (pose disabled)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return torch.zeros(B, self.output_dim, device=x.device, dtype=x.dtype)
