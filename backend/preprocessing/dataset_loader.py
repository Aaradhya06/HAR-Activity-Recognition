"""
Dataset Loader — UCF101 / HMDB51 / Custom Video Dataset
=========================================================
Implements a PyTorch Dataset for video action recognition.

Key features:
  - Reads video files from a structured directory
  - Supports UCF101 standard file/directory layout
  - Configurable class subset selection
  - Temporal jitter for training
  - Pose keypoint loading (if pre-extracted)
  - Train/val/test splitting with reproducible seeds
  - Data integrity validation (skips corrupt/empty videos)

Expected directory layout for UCF101:
    data/raw/UCF101/
        ApplyEyeMakeup/
            v_ApplyEyeMakeup_g01_c01.avi
            ...
        Archery/
            ...
        ...

Or any flat layout with subdirectories as class names.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.frame_processor import (
    video_to_tensor,
    frames_to_tensor,
    read_video_frames,
    sample_frames_uniform,
    preprocess_frame,
    augment_frame,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


class VideoActivityDataset(Dataset):
    """
    Dataset for video-based activity recognition.

    Args:
        video_paths:   List of paths to video files.
        labels:        Integer label for each video.
        class_names:   Ordered list of class name strings.
        num_frames:    Frames to sample per clip.
        frame_size:    (H, W) resize target.
        mean / std:    Normalization statistics.
        augment:       Whether to apply training augmentation.
        temporal_jitter: Random clip start (training only).
        pose_dir:      Optional directory with pre-extracted pose .npy files.
    """

    def __init__(
        self,
        video_paths: list[str],
        labels: list[int],
        class_names: list[str],
        num_frames: int = 16,
        frame_size: tuple[int, int] = (224, 224),
        mean: list[float] = IMAGENET_MEAN,
        std: list[float] = IMAGENET_STD,
        augment: bool = False,
        temporal_jitter: bool = False,
        pose_dir: Optional[str] = None,
    ) -> None:
        assert len(video_paths) == len(labels), "Mismatched paths and labels"
        self.video_paths = video_paths
        self.labels = labels
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.temporal_jitter = temporal_jitter
        self.pose_dir = Path(pose_dir) if pose_dir else None

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict:
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            frames_tensor, _ = video_to_tensor(
                video_path,
                num_frames=self.num_frames,
                frame_size=self.frame_size,
                mean=self.mean,
                std=self.std,
                temporal_jitter=self.temporal_jitter,
                augment=self.augment,
            )
            frames_tensor = frames_tensor.squeeze(0)  # (T, C, H, W)
        except Exception as e:
            logger.warning("Skipping corrupt video %s: %s", video_path, e)
            # Return zeros for corrupt video
            frames_tensor = torch.zeros(
                self.num_frames, 3, *self.frame_size
            )

        # Load pose keypoints if available
        pose_keypoints = self._load_pose(video_path)

        return {
            "frames": frames_tensor,            # (T, C, H, W)
            "pose": pose_keypoints,              # (T, 99) or zeros
            "label": torch.tensor(label, dtype=torch.long),
            "path": video_path,
        }

    def _load_pose(self, video_path: str) -> torch.Tensor:
        """Load pre-extracted pose keypoints or return zeros."""
        if self.pose_dir is not None:
            stem = Path(video_path).stem
            pose_file = self.pose_dir / f"{stem}.npy"
            if pose_file.exists():
                try:
                    data = np.load(str(pose_file))  # (T_orig, 99)
                    # Resample to num_frames
                    if len(data) != self.num_frames:
                        indices = np.linspace(0, len(data) - 1, self.num_frames, dtype=int)
                        data = data[indices]
                    return torch.from_numpy(data).float()
                except Exception as e:
                    logger.debug("Could not load pose for %s: %s", video_path, e)
        # Return zeros (NullPoseEncoder will handle this)
        return torch.zeros(self.num_frames, 99)


# ── Dataset factory ───────────────────────────────────────────────────────────

def discover_videos(
    root_dir: str | Path,
    selected_classes: Optional[list[str]] = None,
) -> tuple[list[str], list[int], list[str]]:
    """
    Walk a directory expecting:
        root_dir/<class_name>/<video_file>

    Args:
        root_dir: Top-level dataset directory.
        selected_classes: If provided, only use these classes. Names must
                          exactly match subdirectory names.

    Returns:
        (video_paths, int_labels, class_names)
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    # Discover class dirs
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No subdirectories (classes) found in {root}")

    all_class_names = [d.name for d in class_dirs]

    if selected_classes:
        missing = [c for c in selected_classes if c not in all_class_names]
        if missing:
            logger.warning(
                "Classes not found in dataset: %s. Available: %s",
                missing, all_class_names[:20]
            )
        class_names = [c for c in selected_classes if c in all_class_names]
        if not class_names:
            raise ValueError(
                f"None of the selected classes {selected_classes} exist in {root}. "
                f"Available: {all_class_names[:30]}"
            )
    else:
        class_names = all_class_names

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    video_paths: list[str] = []
    int_labels: list[int] = []

    for cls_name in class_names:
        cls_dir = root / cls_name
        videos = [
            str(f) for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not videos:
            logger.warning("No videos found for class: %s", cls_name)
            continue
        for v in sorted(videos):
            video_paths.append(v)
            int_labels.append(class_to_idx[cls_name])

    logger.info(
        "Discovered %d videos across %d classes from %s",
        len(video_paths), len(class_names), root
    )
    return video_paths, int_labels, class_names


def split_dataset(
    video_paths: list[str],
    labels: list[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
]:
    """
    Stratified train/val/test split.
    Returns ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)).
    """
    rng = random.Random(seed)
    # Group by class for stratification
    class_groups: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        class_groups[lbl].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for cls, indices in class_groups.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n >= 3:
            n_val = max(1, int(n * val_ratio))
            n_test = max(1, int(n * (1.0 - train_ratio - val_ratio)))
            n_train = max(1, n - n_val - n_test)
        else:
            n_train = max(1, int(n * train_ratio))
            n_val = 0
            n_test = n - n_train
        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train: n_train + n_val])
        test_idx.extend(shuffled[n_train + n_val:])

    def subset(indices: list[int]) -> tuple[list[str], list[int]]:
        return [video_paths[i] for i in indices], [labels[i] for i in indices]

    return subset(train_idx), subset(val_idx), subset(test_idx)


def build_dataloaders(
    cfg: dict,
    class_names_output_path: Optional[str] = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build train/val/test DataLoaders from config.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_cfg = cfg["data"]
    root = data_cfg["dataset_path"]
    selected = data_cfg.get("selected_classes") or None
    num_frames = data_cfg["sequence_length"]
    frame_size = tuple(data_cfg["frame_size"])
    mean = data_cfg["mean"]
    std = data_cfg["std"]
    train_ratio = data_cfg["train_ratio"]
    val_ratio = data_cfg["val_ratio"]
    seed = cfg["project"]["seed"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)

    # Discover
    video_paths, labels, class_names = discover_videos(root, selected)

    # Update num_classes in config
    cfg["model"]["num_classes"] = len(class_names)

    # Save class names
    if class_names_output_path:
        Path(class_names_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(class_names_output_path, "w") as f:
            json.dump(class_names, f, indent=2)
        logger.info("Saved class names -> %s", class_names_output_path)

    # Split
    (tr_paths, tr_lbl), (va_paths, va_lbl), (te_paths, te_lbl) = split_dataset(
        video_paths, labels, train_ratio, val_ratio, seed
    )

    logger.info("Split: train=%d, val=%d, test=%d", len(tr_paths), len(va_paths), len(te_paths))

    # Datasets
    train_ds = VideoActivityDataset(
        tr_paths, tr_lbl, class_names,
        num_frames=num_frames, frame_size=frame_size,
        mean=mean, std=std,
        augment=True, temporal_jitter=data_cfg.get("temporal_jitter", True),
    )
    val_ds = VideoActivityDataset(
        va_paths, va_lbl, class_names,
        num_frames=num_frames, frame_size=frame_size,
        mean=mean, std=std, augment=False,
    )
    test_ds = VideoActivityDataset(
        te_paths, te_lbl, class_names,
        num_frames=num_frames, frame_size=frame_size,
        mean=mean, std=std, augment=False,
    )

    # DataLoaders
    # Only drop last if there are multiple full batches to avoid losing all data on small sets
    drop_last = len(train_ds) > batch_size and (len(train_ds) % batch_size != 0)
    train_loader = DataLoader(
        train_ds, batch_size=min(batch_size, max(1, len(train_ds))), shuffle=True,
        num_workers=num_workers, pin_memory=data_cfg.get("pin_memory", True),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=data_cfg.get("pin_memory", True),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, class_names
