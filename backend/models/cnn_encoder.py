"""
CNN Spatial Feature Extractor
==============================
Wraps a pretrained lightweight CNN backbone (MobileNetV3, EfficientNet-B0, or ResNet-18)
and removes its classification head, outputting per-frame feature embeddings.

The module accepts a batch of frames shaped (B, T, C, H, W) and returns
(B, T, feature_dim) for downstream temporal modelling.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)

BackboneType = Literal["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"]


class CNNEncoder(nn.Module):
    """
    Lightweight CNN encoder for per-frame spatial feature extraction.

    Args:
        backbone: CNN architecture name.
        pretrained: Whether to load ImageNet pretrained weights.
        output_dim: Projected feature dimension. If None, uses native backbone dim.
        freeze_backbone: If True, freeze backbone weights (feature extraction only).
        dropout: Dropout probability on projected features.
    """

    # Native output dimensions after adaptive pooling
    BACKBONE_DIMS: dict[str, int] = {
        "mobilenet_v3_small": 576,
        "mobilenet_v3_large": 960,
        "efficientnet_b0": 1280,
        "resnet18": 512,
    }

    def __init__(
        self,
        backbone: BackboneType = "mobilenet_v3_small",
        pretrained: bool = True,
        output_dim: int | None = 256,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.native_dim = self.BACKBONE_DIMS[backbone]
        self.output_dim = output_dim or self.native_dim

        # ── Build backbone ─────────────────────────────────────────────────
        weights_arg = "DEFAULT" if pretrained else None

        if backbone == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(weights=weights_arg)
            # Remove classifier head; keep features + avgpool
            self.feature_extractor = nn.Sequential(
                base.features,
                base.avgpool,
                nn.Flatten(1),
            )
        elif backbone == "mobilenet_v3_large":
            base = models.mobilenet_v3_large(weights=weights_arg)
            self.feature_extractor = nn.Sequential(
                base.features,
                base.avgpool,
                nn.Flatten(1),
            )
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights_arg)
            self.feature_extractor = nn.Sequential(
                base.features,
                base.avgpool,
                nn.Flatten(1),
            )
        elif backbone == "resnet18":
            base = models.resnet18(weights=weights_arg)
            # Remove fc layer
            self.feature_extractor = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
                base.avgpool,
                nn.Flatten(1),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}")

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            logger.info("Backbone weights frozen.")

        # ── Optional projection layer ──────────────────────────────────────
        if output_dim is not None and output_dim != self.native_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.native_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.projection = nn.Identity()

        logger.info(
            "CNNEncoder: backbone=%s, native_dim=%d, output_dim=%d, pretrained=%s",
            backbone, self.native_dim, self.output_dim, pretrained,
        )

    def encode_single_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of single frames (B, C, H, W) → (B, output_dim)."""
        feats = self.feature_extractor(x)
        return self.projection(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Frame sequence tensor (B, T, C, H, W).

        Returns:
            Feature sequence (B, T, output_dim).
        """
        B, T, C, H, W = x.shape
        # Merge batch and time dimensions for parallel CNN forward
        x_flat = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x_flat)         # (B*T, native_dim)
        feats = self.projection(feats)                  # (B*T, output_dim)
        return feats.view(B, T, self.output_dim)        # (B, T, output_dim)

    def count_parameters(self) -> int:
        """Returns number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
