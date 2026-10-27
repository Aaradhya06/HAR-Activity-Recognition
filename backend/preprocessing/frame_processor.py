"""
Video Frame Sampling and Preprocessing Pipeline
================================================
Handles all frame-level operations:
  - Video reading with OpenCV
  - Temporal frame sampling (uniform, random, dense)
  - Resizing, normalization, augmentation
  - Sequence construction for model input
  - Pose keypoint extraction via MediaPipe (optional)

This module is shared between training data loading, batch inference,
and the real-time rolling-window inference service.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Frame-level transforms ────────────────────────────────────────────────────

def preprocess_frame(
    frame_bgr: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Convert a BGR OpenCV frame to a normalized RGB tensor.

    Args:
        frame_bgr: (H, W, 3) BGR numpy array.
        target_size: (H, W) output size.
        mean: Per-channel mean for normalization.
        std:  Per-channel std for normalization.

    Returns:
        (3, H, W) float32 tensor, normalized.
    """
    # BGR → RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize
    h, w = target_size
    frame_resized = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    # HWC numpy → CHW torch float [0,1]
    tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    # Normalize
    tensor = TF.normalize(tensor, mean=mean, std=std)
    return tensor


def augment_frame(
    frame_tensor: torch.Tensor,
    horizontal_flip: bool = False,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
) -> torch.Tensor:
    """Apply data augmentation to a single frame tensor (C, H, W)."""
    if horizontal_flip:
        frame_tensor = TF.hflip(frame_tensor)
    if brightness_jitter > 0.0:
        factor = 1.0 + np.random.uniform(-brightness_jitter, brightness_jitter)
        frame_tensor = TF.adjust_brightness(frame_tensor, factor)
    if contrast_jitter > 0.0:
        factor = 1.0 + np.random.uniform(-contrast_jitter, contrast_jitter)
        frame_tensor = TF.adjust_contrast(frame_tensor, factor)
    return frame_tensor


# ── Video reading ─────────────────────────────────────────────────────────────

def read_video_frames(
    video_path: str | Path,
    max_frames: int = 300,
) -> tuple[list[np.ndarray], float]:
    """
    Read all frames from a video file.

    Args:
        video_path: Path to video file.
        max_frames:  Safety cap on number of frames to load.

    Returns:
        (frames_bgr_list, fps)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Video contains no readable frames: {video_path}")

    return frames, fps


# ── Temporal sampling ─────────────────────────────────────────────────────────

def sample_frames_uniform(
    frames: list[np.ndarray],
    num_frames: int,
    temporal_jitter: bool = False,
) -> list[np.ndarray]:
    """
    Sample exactly num_frames from the frame list using uniform spacing.

    Args:
        frames: Full list of BGR frames.
        num_frames: Desired sequence length.
        temporal_jitter: If True and there is slack, add random offset.

    Returns:
        List of num_frames sampled frames.
    """
    total = len(frames)
    if total == 0:
        raise ValueError("Cannot sample from empty frame list.")

    if total <= num_frames:
        # Repeat last frame to pad
        indices = list(range(total))
        while len(indices) < num_frames:
            indices.append(total - 1)
        return [frames[i] for i in indices]

    # Uniform spacing
    if temporal_jitter:
        # Random start within valid range
        max_start = total - num_frames
        start = np.random.randint(0, max_start + 1)
        step = 1.0
        indices = [start + int(i) for i in np.arange(0, num_frames)]
        indices = [min(i, total - 1) for i in indices]
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    return [frames[i] for i in indices]


def sample_frames_sliding(
    frames: list[np.ndarray],
    num_frames: int,
    stride: int = 1,
) -> list[list[np.ndarray]]:
    """
    Generate overlapping clips using sliding window.

    Args:
        frames: Full frame list.
        num_frames: Clip length.
        stride: Step between clip starts.

    Returns:
        List of clips (each clip is a list of num_frames frames).
    """
    clips = []
    total = len(frames)
    if total < num_frames:
        # Pad and return single clip
        padded = list(frames) + [frames[-1]] * (num_frames - total)
        return [padded]

    for start in range(0, total - num_frames + 1, stride):
        clips.append(frames[start: start + num_frames])
    return clips


# ── Full pipeline ─────────────────────────────────────────────────────────────

def video_to_tensor(
    video_path: str | Path,
    num_frames: int = 16,
    frame_size: tuple[int, int] = (224, 224),
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
    temporal_jitter: bool = False,
    augment: bool = False,
) -> tuple[torch.Tensor, float]:
    """
    Full pipeline: video file → (1, T, C, H, W) tensor ready for the model.

    Returns:
        (tensor, fps)
        tensor shape: (1, num_frames, 3, H, W)
    """
    raw_frames, fps = read_video_frames(video_path)
    sampled = sample_frames_uniform(raw_frames, num_frames, temporal_jitter)

    do_flip = augment and (np.random.random() < 0.5)
    processed = []
    for f in sampled:
        t = preprocess_frame(f, target_size=frame_size, mean=mean, std=std)
        if augment:
            t = augment_frame(t, horizontal_flip=do_flip,
                              brightness_jitter=0.2, contrast_jitter=0.2)
        processed.append(t)

    # Stack → (T, C, H, W) → (1, T, C, H, W)
    tensor = torch.stack(processed).unsqueeze(0)
    return tensor, fps


def frames_to_tensor(
    frames_bgr: list[np.ndarray],
    num_frames: int = 16,
    frame_size: tuple[int, int] = (224, 224),
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
    temporal_jitter: bool = False,
) -> torch.Tensor:
    """
    Convert a raw frame list to model-ready tensor (1, T, C, H, W).
    Used in real-time rolling window inference.
    """
    sampled = sample_frames_uniform(frames_bgr, num_frames, temporal_jitter)
    processed = [preprocess_frame(f, target_size=frame_size, mean=mean, std=std)
                 for f in sampled]
    return torch.stack(processed).unsqueeze(0)


# ── Inference timing ──────────────────────────────────────────────────────────

class InferenceTimer:
    """Simple wall-clock timer for measuring per-clip inference latency."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.last_ms: float = 0.0
        self.history: list[float] = []

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self.last_ms = elapsed_ms
        self.history.append(elapsed_ms)
        self._start = None
        return elapsed_ms

    @property
    def average_ms(self) -> float:
        return float(np.mean(self.history)) if self.history else 0.0

    @property
    def fps(self) -> float:
        if self.last_ms <= 0:
            return 0.0
        return 1000.0 / self.last_ms
