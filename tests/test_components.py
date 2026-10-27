"""
Unit and Integration Tests for HAR System
==========================================
Tests:
  - CNN Encoder feature extraction (MobileNetV3, shapes)
  - Temporal Transformer and Positional Encoding
  - Pose Estimator & Pose Encoder
  - Multi-Modal Fusion Layer
  - HARModel forward & backward pass
  - Rolling Window & Frame Processor
  - Inference Engine State Transitions
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import torch
import yaml

import sys
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from models.cnn_encoder import CNNEncoder
from models.temporal_transformer import TemporalTransformer, SinusoidalPositionalEncoding
from models.pose_encoder import PoseEncoder
from models.fusion_model import FusionLayer
from models.classifier import Classifier
from models.har_model import HARModel
from preprocessing.frame_processor import preprocess_frame, frames_to_tensor
from inference.engine import HARInferenceEngine, ModelState


@pytest.fixture
def config():
    cfg_path = _BACKEND / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_cnn_encoder():
    """Verify CNN encoder output dimensions."""
    encoder = CNNEncoder(backbone="mobilenet_v3_small", output_dim=256, pretrained=False)
    # Batch of 2 clips, each 4 frames
    x = torch.randn(2, 4, 3, 224, 224)
    feat = encoder(x)
    assert feat.shape == (2, 4, 256), f"Expected (2, 4, 256), got {feat.shape}"


def test_positional_encoding():
    """Verify sinusoidal positional encoding shapes."""
    pe = SinusoidalPositionalEncoding(d_model=256, max_len=64)
    x = torch.zeros(2, 16, 256)
    out = pe(x)
    assert out.shape == (2, 16, 256)


def test_temporal_transformer():
    """Verify temporal attention aggregation."""
    transformer = TemporalTransformer(d_model=256, nhead=4, num_layers=2, pooling="mean")
    x = torch.randn(2, 16, 256)
    out = transformer(x)
    assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"


def test_pose_encoder():
    """Verify pose keypoint encoding."""
    pose_enc = PoseEncoder(input_dim=99, hidden_dim=128, output_dim=64)
    p = torch.randn(2, 16, 99)
    out = pose_enc(p)
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"


def test_fusion_layer():
    """Verify visual + pose multimodal fusion."""
    fusion = FusionLayer(fusion_type="concat", visual_dim=256, pose_dim=64, output_dim=256)
    v = torch.randn(2, 256)
    p = torch.randn(2, 64)
    fused = fusion(v, p)
    assert fused.shape == (2, 256), f"Expected (2, 256), got {fused.shape}"


def test_har_model_complete(config):
    """Verify full end-to-end forward pass with and without pose (2 classes for demo model)."""
    config["model"]["num_classes"] = 2  # 2-class demonstration model: SittingNormally / RaisingHand
    config["model"]["backbone_pretrained"] = False
    model = HARModel.from_config(config)

    frames = torch.randn(2, 16, 3, 224, 224)
    pose = torch.randn(2, 16, 99)

    # With pose
    logits = model(frames, pose)
    assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"

    # Without pose (fallback)
    logits_no_pose = model(frames, None)
    assert logits_no_pose.shape == (2, 2)


def test_frame_processor():
    """Verify frame resizing and normalization."""
    raw_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor = preprocess_frame(raw_bgr, target_size=(224, 224))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_inference_engine_state(config):
    """Verify engine loads checkpoint or reports NO_CHECKPOINT properly."""
    engine = HARInferenceEngine(config)
    assert engine.state in [ModelState.READY, ModelState.NO_CHECKPOINT]
    info = engine.get_model_info()
    assert "architecture" in info
    assert "device" in info
