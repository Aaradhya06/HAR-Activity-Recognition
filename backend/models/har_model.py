"""
HAR Model — Full End-to-End Architecture
=========================================
Assembles the complete Human Activity Recognition pipeline:

    frames (B, T, C, H, W)
        ↓
    CNNEncoder           → per-frame spatial embeddings (B, T, cnn_dim)
        ↓
    TemporalModule       → temporal representation (B, temporal_dim)
        ↓
    [in parallel]
    frames → PoseEncoder → pose representation (B, pose_dim)
        ↓
    FusionLayer          → fused embedding (B, fused_dim)
        ↓
    Classifier           → logits (B, num_classes)

This architecture is the project's primary research-inspired model,
described as: "Lightweight hybrid CNN + Temporal Attention + Pose Fusion".

A CNN+LSTM baseline variant is also supported for ablation studies.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .temporal_transformer import build_temporal_module
from .pose_encoder import PoseEncoder, NullPoseEncoder
from .fusion_model import FusionLayer
from .classifier import Classifier

logger = logging.getLogger(__name__)


class HARModel(nn.Module):
    """
    Complete HAR model: CNN → Temporal → Pose → Fusion → Classifier.

    Args:
        num_classes: Number of activity classes.
        backbone: CNN backbone name.
        backbone_pretrained: Use ImageNet weights.
        cnn_output_dim: CNN projection dimension.
        temporal_type: 'transformer', 'lstm', or 'gru'.
        temporal_cfg: Dict of temporal module hyperparameters.
        use_pose: Whether to incorporate pose branch.
        pose_input_dim: Flattened keypoint dimension.
        pose_hidden_dim: Pose MLP hidden size.
        pose_output_dim: Pose embedding dimension.
        fusion_type: Fusion strategy.
        fusion_dropout: Dropout in fusion.
        classifier_hidden: Hidden dim in classifier.
        classifier_dropout: Dropout in classifier.
        sequence_length: Number of frames per clip.
    """

    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = "mobilenet_v3_small",
        backbone_pretrained: bool = True,
        cnn_output_dim: int = 256,
        temporal_type: str = "transformer",
        temporal_cfg: Optional[dict] = None,
        use_pose: bool = True,
        pose_input_dim: int = 99,
        pose_hidden_dim: int = 128,
        pose_output_dim: int = 64,
        fusion_type: str = "concat",
        fusion_dropout: float = 0.3,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.4,
        sequence_length: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_pose = use_pose
        self.sequence_length = sequence_length

        # ── 1. CNN Spatial Encoder ─────────────────────────────────────────
        self.cnn = CNNEncoder(
            backbone=backbone,
            pretrained=backbone_pretrained,
            output_dim=cnn_output_dim,
            dropout=0.1,
        )

        # ── 2. Temporal Module ─────────────────────────────────────────────
        tcfg = temporal_cfg or {}
        tcfg["sequence_length"] = sequence_length
        self.temporal, temporal_out_dim = build_temporal_module(
            temporal_type=temporal_type,
            input_dim=cnn_output_dim,
            cfg=tcfg,
        )

        # ── 3. Pose Branch ─────────────────────────────────────────────────
        if use_pose:
            self.pose_encoder = PoseEncoder(
                input_dim=pose_input_dim,
                hidden_dim=pose_hidden_dim,
                output_dim=pose_output_dim,
                num_frames=sequence_length,
            )
            effective_pose_dim = pose_output_dim
        else:
            self.pose_encoder = NullPoseEncoder(output_dim=pose_output_dim)
            effective_pose_dim = pose_output_dim

        # ── 4. Fusion ──────────────────────────────────────────────────────
        self.fusion = FusionLayer(
            visual_dim=temporal_out_dim,
            pose_dim=effective_pose_dim,
            output_dim=temporal_out_dim,
            fusion_type=fusion_type,
            dropout=fusion_dropout,
        )

        # ── 5. Classifier ──────────────────────────────────────────────────
        self.classifier = Classifier(
            input_dim=temporal_out_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden,
            dropout=classifier_dropout,
        )

        self._log_architecture()

    def _log_architecture(self) -> None:
        total = self.count_parameters()
        trainable = self.count_parameters(trainable_only=True)
        logger.info(
            "HARModel ready — total params: %s, trainable: %s",
            f"{total:,}", f"{trainable:,}"
        )

    def forward(
        self,
        frames: torch.Tensor,
        pose_keypoints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            frames: Video frame sequence (B, T, C, H, W).
            pose_keypoints: Optional keypoint sequence (B, T, pose_input_dim).
                            If None and use_pose=True, zeros are used.

        Returns:
            logits: (B, num_classes) — unnormalized class scores.
        """
        B, T = frames.shape[:2]

        # 1. Spatial features
        spatial_feats = self.cnn(frames)          # (B, T, cnn_dim)

        # 2. Temporal modelling
        temporal_rep = self.temporal(spatial_feats)  # (B, temporal_dim)

        # 3. Pose encoding
        if pose_keypoints is None:
            pose_in = torch.zeros(B, T, self.pose_encoder.input_dim
                                  if hasattr(self.pose_encoder, 'input_dim') else 99,
                                  device=frames.device, dtype=frames.dtype)
        else:
            pose_in = pose_keypoints

        pose_rep = self.pose_encoder(pose_in)     # (B, pose_dim)

        # 4. Fusion
        fused = self.fusion(temporal_rep, pose_rep)  # (B, fused_dim)

        # 5. Classification
        logits = self.classifier(fused)           # (B, num_classes)
        return logits

    def predict(
        self,
        frames: torch.Tensor,
        pose_keypoints: Optional[torch.Tensor] = None,
        top_k: int = 5,
    ) -> dict:
        """
        Convenience method for inference.

        Returns:
            dict with 'logits', 'probabilities', 'top_k_probs', 'top_k_indices'
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(frames, pose_keypoints)
            probs = torch.softmax(logits, dim=-1)
            k = min(top_k, self.num_classes)
            top_probs, top_indices = torch.topk(probs, k, dim=-1)
        return {
            "logits": logits,
            "probabilities": probs,
            "top_k_probs": top_probs,
            "top_k_indices": top_indices,
        }

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg: dict) -> "HARModel":
        """Build model from a config dictionary (from config.yaml)."""
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})

        return cls(
            num_classes=model_cfg.get("num_classes", 10),
            backbone=model_cfg.get("backbone", "mobilenet_v3_small"),
            backbone_pretrained=model_cfg.get("backbone_pretrained", True),
            cnn_output_dim=model_cfg.get("transformer_d_model", 256),
            temporal_type=model_cfg.get("temporal_type", "transformer"),
            temporal_cfg=model_cfg,
            use_pose=model_cfg.get("use_pose", True),
            pose_input_dim=model_cfg.get("pose_input_dim", 99),
            pose_hidden_dim=model_cfg.get("pose_hidden_dim", 128),
            pose_output_dim=model_cfg.get("pose_output_dim", 64),
            fusion_type=model_cfg.get("fusion_type", "concat"),
            fusion_dropout=model_cfg.get("fusion_dropout", 0.3),
            classifier_hidden=128,
            classifier_dropout=model_cfg.get("classifier_dropout", 0.4),
            sequence_length=data_cfg.get("sequence_length", 16),
        )
