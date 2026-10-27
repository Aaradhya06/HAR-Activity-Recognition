"""
Inference Engine
=================
Handles model loading, state management, and prediction for both:
  - Batch video inference (uploaded files)
  - Real-time rolling-window inference (webcam / live stream)

Implements three model states:
  STATE 1: Project checkpoint available → use it
  STATE 2: Architecture ready but no checkpoint → "model not trained"
  STATE 3: External pretrained checkpoint with valid labels → optional use

Never silently substitutes an unrelated model.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from models.har_model import HARModel
from preprocessing.frame_processor import (
    frames_to_tensor,
    video_to_tensor,
    InferenceTimer,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from pose.pose_estimator import get_pose_estimator

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    READY = "ready"              # Checkpoint loaded, ready for inference
    NO_CHECKPOINT = "no_checkpoint"  # Architecture only, not trained
    EXTERNAL = "external"        # External pretrained checkpoint
    ERROR = "error"              # Failed to load


class PredictionResult:
    """Structured prediction result."""

    def __init__(
        self,
        activity: str,
        confidence: float,
        top_predictions: list[dict],
        inference_time_ms: float,
        fps: float,
        model_state: ModelState,
    ) -> None:
        self.activity = activity
        self.confidence = confidence
        self.top_predictions = top_predictions
        self.inference_time_ms = inference_time_ms
        self.fps = fps
        self.model_state = model_state

    def to_dict(self) -> dict:
        return {
            "activity": self.activity,
            "confidence": round(self.confidence, 4),
            "top_predictions": self.top_predictions,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "fps": round(self.fps, 2),
            "model_state": self.model_state.value,
        }


class HARInferenceEngine:
    """
    Central inference service for the HAR system.

    Thread-safe for use in FastAPI (single-worker mode).
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.inf_cfg = cfg["inference"]
        self.data_cfg = cfg["data"]
        self.model_cfg = cfg["model"]

        # Device
        device_str = self.inf_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        logger.info("Inference device: %s", self.device)

        # Model state
        self.model: Optional[HARModel] = None
        self.state = ModelState.NO_CHECKPOINT
        self.class_names: list[str] = []
        self.num_classes: int = 0

        # Timer
        self.timer = InferenceTimer()

        # Rolling window buffer
        self.frame_buffer: deque = deque(maxlen=self.inf_cfg.get("sequence_length", 16))
        self.pose_buffer: deque = deque(maxlen=self.inf_cfg.get("sequence_length", 16))

        # EMA smoothing — reduced alpha for faster response
        # alpha=0.5 means 50% old / 50% new (was 0.7 = very sticky)
        self.smoothing_alpha = self.inf_cfg.get("smoothing_alpha", 0.5)
        self._smoothed_probs: Optional[np.ndarray] = None

        # Confidence threshold for "Uncertain" display
        self.confidence_threshold = self.inf_cfg.get("confidence_threshold", 0.45)

        # Pose estimator
        pose_cfg = cfg.get("pose", {})
        self.pose_estimator = get_pose_estimator(
            min_detection_confidence=pose_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
            draw_landmarks=pose_cfg.get("draw_landmarks", True),
        )

        # Try to load checkpoint
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_path(p_str: str) -> Path:
        """Resolve a path whether running from backend, project root, or anywhere."""
        p = Path(p_str)
        if p.exists():
            return p
        clean = p_str.replace("../", "")
        p_clean = Path(clean)
        if p_clean.exists():
            return p_clean
        # Try relative to project root
        p_proj = Path(__file__).resolve().parent.parent.parent / clean
        if p_proj.exists():
            return p_proj
        # Try relative to backend dir
        p_backend = Path(__file__).resolve().parent.parent / clean
        if p_backend.exists():
            return p_backend
        return p

    def _load_class_names(self) -> list[str]:
        """Load class names from JSON file."""
        cn_str = self.data_cfg.get("class_names_file", "../data/class_names.json")
        cn_path = self._resolve_path(cn_str)
        if cn_path.exists():
            with open(cn_path, "r", encoding="utf-8") as f:
                names = json.load(f)
            logger.info("Loaded %d class names from %s", len(names), cn_path)
            return names
        logger.warning("class_names.json not found at %s", cn_path)
        return []

    def _load_model(self) -> None:
        """Attempt to load model checkpoint."""
        self.class_names = self._load_class_names()

        ckpt_str = self.inf_cfg.get("checkpoint_path", "../checkpoints/best_model.pth")
        ckpt_path = self._resolve_path(ckpt_str)

        # Build architecture (always succeeds)
        try:
            n_classes = len(self.class_names) if self.class_names else self.model_cfg.get("num_classes", 10)
            self.model_cfg["num_classes"] = n_classes
            self.model = HARModel.from_config(self.cfg)
            self.model.eval()
            self.num_classes = n_classes
        except Exception as e:
            logger.error("Failed to build model architecture: %s", e)
            self.state = ModelState.ERROR
            return

        if not ckpt_path.exists():
            logger.warning(
                "Checkpoint not found at %s. Model architecture ready but not trained.",
                ckpt_path,
            )
            self.state = ModelState.NO_CHECKPOINT
            return

        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            # Validate class names match
            ckpt_classes = checkpoint.get("class_names", [])
            if ckpt_classes and ckpt_classes != self.class_names:
                logger.warning(
                    "Checkpoint class names differ from class_names.json. "
                    "Using checkpoint names."
                )
                self.class_names = ckpt_classes

            # Try loading with current model config
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as load_err:
                err_str = str(load_err)
                # If pose keys are missing, the checkpoint was trained without pose
                if "pose_encoder" in err_str and "Missing key" in err_str:
                    logger.warning(
                        "Checkpoint was trained without pose branch. "
                        "Rebuilding model with use_pose=False."
                    )
                    self.model_cfg["use_pose"] = False
                    self.model = HARModel.from_config(self.cfg)
                    self.model.eval()
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    raise

            self.model.to(self.device)
            self.model.eval()
            self.state = ModelState.READY
            logger.info(
                "Model checkpoint loaded from %s (epoch %s, val_acc=%.4f)",
                ckpt_path,
                checkpoint.get("epoch", "?"),
                checkpoint.get("best_val_accuracy", float("nan")),
            )
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            self.state = ModelState.NO_CHECKPOINT

    def reload_model(self) -> dict:
        """Reload model (e.g., after training completes)."""
        self._load_model()
        return {"state": self.state.value, "classes": self.class_names}

    # ── Core prediction ───────────────────────────────────────────────────

    def _run_inference(
        self,
        frames_tensor: torch.Tensor,
        pose_tensor: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Run model forward pass and return prediction dict.

        Args:
            frames_tensor: (1, T, C, H, W)
            pose_tensor:   (1, T, 99) or None

        Returns:
            Raw prediction dict.
        """
        if self.model is None or self.state not in (ModelState.READY, ModelState.EXTERNAL):
            return {
                "state": self.state.value,
                "message": "Model not trained. Please run training first.",
            }

        frames_tensor = frames_tensor.to(self.device)
        if pose_tensor is not None:
            pose_tensor = pose_tensor.to(self.device)

        self.timer.start()
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = self.model(frames_tensor, pose_tensor)
            else:
                logits = self.model(frames_tensor, pose_tensor)

        elapsed_ms = self.timer.stop()

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (num_classes,)
        top_k = min(self.inf_cfg.get("top_k", 5), len(self.class_names))
        top_indices = np.argsort(probs)[::-1][:top_k]

        top_preds = [
            {
                "class": self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                "confidence": float(probs[i]),
            }
            for i in top_indices
        ]

        top_conf = float(probs[top_indices[0]])
        # Low confidence → label as "Detecting..." rather than showing wrong class
        display_activity = top_preds[0]["class"]
        if top_conf < self.confidence_threshold:
            display_activity = "Detecting..."

        return {
            "activity": display_activity,
            "confidence": top_conf,
            "top_predictions": top_preds,
            "inference_time_ms": elapsed_ms,
            "fps": 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
            "model_state": self.state.value,

        }

    # ── Public inference API ──────────────────────────────────────────────

    def predict_video(
        self,
        video_path: str,
        use_pose: bool = True,
    ) -> dict:
        """
        Full video prediction pipeline.

        Args:
            video_path: Path to uploaded video file.
            use_pose: Whether to extract pose during inference.

        Returns:
            Structured prediction dict.
        """
        from preprocessing.frame_processor import read_video_frames, sample_frames_uniform

        seq_len = self.inf_cfg.get("sequence_length", 16)
        frame_size = tuple(self.data_cfg.get("frame_size", [224, 224]))

        try:
            raw_frames, fps = read_video_frames(video_path)
            sampled = sample_frames_uniform(raw_frames, seq_len)
        except Exception as e:
            return {"error": str(e), "model_state": self.state.value}

        # POSE FIX: Training used zeros for pose — use zeros here too for consistency.
        # The model was trained with pose=zeros so real MediaPipe values would be
        # out-of-distribution and degrade predictions. Pass zeros to match training.
        pose_tensor = torch.zeros(1, seq_len, 99)

        frames_tensor = frames_to_tensor(
            sampled, num_frames=seq_len, frame_size=frame_size
        )

        result = self._run_inference(frames_tensor, pose_tensor)
        result["source_fps"] = fps
        result["num_frames_processed"] = len(sampled)
        return result

    def predict_frame_sequence(
        self,
        frames_bgr: list[np.ndarray],
        use_pose: bool = True,
    ) -> dict:
        """
        Predict from an already-loaded list of BGR frames.
        Used by webcam/rolling-window mode.
        """
        seq_len = self.inf_cfg.get("sequence_length", 16)
        frame_size = tuple(self.data_cfg.get("frame_size", [224, 224]))

        if len(frames_bgr) == 0:
            return {"error": "Empty frame list", "model_state": self.state.value}

        # POSE FIX: Use zeros pose to match training distribution.
        # Model was trained with zeros for pose (MediaPipe not run during training).
        pose_tensor = torch.zeros(1, seq_len, 99)

        frames_tensor = frames_to_tensor(
            frames_bgr, num_frames=seq_len, frame_size=frame_size
        )

        result = self._run_inference(frames_tensor, pose_tensor)

        # Apply EMA smoothing to probabilities for live display
        if "top_predictions" in result and self.class_names:
            result = self._apply_smoothing(result)

        return result

    def add_frame_to_buffer(self, frame_bgr: np.ndarray) -> bool:
        """
        Add frame to rolling window buffer.

        Returns:
            True if buffer is full and ready for inference.
        """
        self.frame_buffer.append(frame_bgr)
        return len(self.frame_buffer) >= self.inf_cfg.get("sequence_length", 16)

    def predict_from_buffer(self, use_pose: bool = True) -> Optional[dict]:
        """
        Run prediction on the current rolling window buffer.

        Returns None if buffer is not yet full.
        """
        seq_len = self.inf_cfg.get("sequence_length", 16)
        if len(self.frame_buffer) < seq_len:
            return None
        return self.predict_frame_sequence(list(self.frame_buffer), use_pose)

    def _apply_smoothing(self, result: dict) -> dict:
        """Apply EMA smoothing to prediction probabilities."""
        if not self.class_names:
            return result

        # Build probability vector
        probs = np.zeros(len(self.class_names))
        class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        for pred in result.get("top_predictions", []):
            idx = class_to_idx.get(pred["class"], -1)
            if idx >= 0:
                probs[idx] = pred["confidence"]

        alpha = self.smoothing_alpha
        if self._smoothed_probs is None or len(self._smoothed_probs) != len(self.class_names):
            # Reset if class count changed
            self._smoothed_probs = probs.copy()
        else:
            self._smoothed_probs = alpha * self._smoothed_probs + (1 - alpha) * probs

        # Renormalize
        s = self._smoothed_probs.sum()
        if s > 0:
            smoothed = self._smoothed_probs / s
        else:
            smoothed = self._smoothed_probs

        top_k = min(self.inf_cfg.get("top_k", 5), len(self.class_names))
        top_indices = np.argsort(smoothed)[::-1][:top_k]

        result["top_predictions"] = [
            {"class": self.class_names[i], "confidence": float(smoothed[i])}
            for i in top_indices
        ]
        result["activity"] = self.class_names[top_indices[0]]
        result["confidence"] = float(smoothed[top_indices[0]])
        return result

    def reset_rolling_buffer(self) -> None:
        """Clear the rolling window buffer and smoothing state."""
        self.frame_buffer.clear()
        self.pose_buffer.clear()
        self._smoothed_probs = None

    def get_model_info(self) -> dict:
        """Return structured model information."""
        info = {
            "state": self.state.value,
            "architecture": self.model_cfg.get("name", "HARModel"),
            "backbone": self.model_cfg.get("backbone", "unknown"),
            "temporal_type": self.model_cfg.get("temporal_type", "unknown"),
            "use_pose": self.model_cfg.get("use_pose", False),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "sequence_length": self.data_cfg.get("sequence_length", 16),
            "frame_size": self.data_cfg.get("frame_size", [224, 224]),
            "device": str(self.device),
            "pose_available": self.pose_estimator.available,
        }
        if self.model is not None:
            info["total_parameters"] = self.model.count_parameters()
            info["trainable_parameters"] = self.model.count_parameters(trainable_only=True)
        return info

    def get_inference_stats(self) -> dict:
        """Return current inference timing statistics."""
        return {
            "last_inference_ms": round(self.timer.last_ms, 2),
            "average_inference_ms": round(self.timer.average_ms, 2),
            "fps": round(self.timer.fps, 2),
            "total_inferences": len(self.timer.history),
        }
