"""
Fusion Model — Combines Visual + Pose Representations
=======================================================
Implements configurable fusion strategies for combining the temporal
visual embedding with the pose embedding before classification.

Supported strategies:
  - concat:    simple concatenation, then linear projection
  - add:       element-wise addition (requires matching dims)
  - attention: learned gating/attention between modalities

This is the integration point described in the report's proposed architecture:
"visual representation + pose representation → fusion → classifier"
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FusionType = Literal["concat", "add", "attention"]


class FusionLayer(nn.Module):
    """
    Multi-modal fusion: temporal visual embedding ⊕ pose embedding → fused.

    Args:
        visual_dim: Dimension of the temporal module output.
        pose_dim:   Dimension of the pose encoder output.
        output_dim: Fused representation dimension.
        fusion_type: Strategy — 'concat', 'add', or 'attention'.
        dropout: Dropout on fused representation.
    """

    def __init__(
        self,
        visual_dim: int = 256,
        pose_dim: int = 64,
        output_dim: int = 256,
        fusion_type: FusionType = "concat",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.output_dim = output_dim

        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(visual_dim + pose_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion_type == "add":
            assert visual_dim == pose_dim, (
                f"add fusion requires equal dims: {visual_dim} vs {pose_dim}"
            )
            self.visual_proj = nn.Linear(visual_dim, output_dim)
            self.pose_proj = nn.Linear(pose_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)
        elif fusion_type == "attention":
            # Gated attention: learns a weight for each modality
            self.gate = nn.Sequential(
                nn.Linear(visual_dim + pose_dim, 2),
                nn.Softmax(dim=-1),
            )
            self.visual_proj = nn.Linear(visual_dim, output_dim)
            self.pose_proj = nn.Linear(pose_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type!r}")

        logger.info(
            "FusionLayer: type=%s, visual_dim=%d, pose_dim=%d, output_dim=%d",
            fusion_type, visual_dim, pose_dim, output_dim,
        )

    def forward(self, visual: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual: (B, visual_dim) — temporal module output.
            pose:   (B, pose_dim)   — pose encoder output.

        Returns:
            (B, output_dim) — fused representation.
        """
        if self.fusion_type == "concat":
            combined = torch.cat([visual, pose], dim=-1)
            return self.fusion(combined)

        elif self.fusion_type == "add":
            fused = self.visual_proj(visual) + self.pose_proj(pose)
            return self.dropout(self.norm(fused))

        else:  # attention
            combined = torch.cat([visual, pose], dim=-1)
            gates = self.gate(combined)              # (B, 2)
            v_w, p_w = gates[:, 0:1], gates[:, 1:2] # (B, 1) each
            fused = v_w * self.visual_proj(visual) + p_w * self.pose_proj(pose)
            return self.dropout(self.norm(fused))
