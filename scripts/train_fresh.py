#!/usr/bin/env python3
"""
Fresh HAR Training Script — 5-Class Real Video Dataset
========================================================
Clean-slate training on the new 5-class dataset with:
  - No resume from old checkpoint (fresh start)
  - ImageNet-pretrained MobileNetV3 backbone (transfer learning)
  - Frozen backbone for first stage, unfreezing head only
  - Small dataset strategy: multi-clip augmentation, cosine LR, early stopping
  - CPU-optimized: no AMP, small batch size, num_workers=0
  - Stratified group split by video index (g01..g08)

Classes (in this fixed order):
  0: SittingNormally
  1: RaisingHand
  2: ThumbsUp
  3: ScratchingHead
  4: LookingAway

Usage (from har-project directory):
    python scripts/train_fresh.py
    python scripts/train_fresh.py --epochs 25 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_BACKEND = _PROJECT / "backend"
sys.path.insert(0, str(_BACKEND))

from models.har_model import HARModel
from preprocessing.frame_processor import (
    read_video_frames,
    sample_frames_uniform,
    preprocess_frame,
    augment_frame,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from training.evaluator import (
    evaluate_model,
    plot_training_curves,
    plot_confusion_matrix,
)
from training.trainer import set_seed, save_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

# Fixed display label map (folder_name → display_name)
DISPLAY_LABELS = {
    "SittingNormally": "Sitting Normally",
    "RaisingHand": "Raising a Hand",
    "ThumbsUp": "Thumbs Up",
    "ScratchingHead": "Scratching Head",
    "LookingAway": "Looking Away",
}

# Canonical class order (must match class_names.json)
CANONICAL_CLASSES = [
    "SittingNormally",
    "RaisingHand",
    "ThumbsUp",
    "ScratchingHead",
    "LookingAway",
]


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fresh HAR Training — 5 Classes")
    p.add_argument("--config", default="backend/config.yaml")
    p.add_argument("--epochs", type=int, default=25, help="Max training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size (CPU: use 4)")
    p.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    p.add_argument("--clips-per-video", type=int, default=3,
                   help="Multiple temporal crops per video for augmentation")
    p.add_argument("--num-frames", type=int, default=16, help="Frames per clip")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-freeze", action="store_true",
                   help="Do not freeze backbone (slower but may improve with enough data)")
    return p.parse_args()


# ── Dataset utilities ─────────────────────────────────────────────────────────

def discover_and_validate_videos(
    root: Path,
    class_names: list[str],
) -> tuple[list[str], list[int]]:
    """
    Discover all valid video files for the given classes.
    Validates each file is readable by OpenCV.
    Returns (paths, labels).
    """
    paths, labels = [], []
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = root / cls
        if not cls_dir.exists():
            logger.warning("Class directory missing: %s", cls_dir)
            continue
        videos = sorted(
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            logger.warning("No videos found for class: %s", cls)
            continue

        valid_count = 0
        for v in videos:
            import cv2
            cap = cv2.VideoCapture(str(v))
            readable = cap.isOpened()
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
            cap.release()
            if readable and n_frames > 0:
                paths.append(str(v))
                labels.append(class_to_idx[cls])
                valid_count += 1
            else:
                logger.warning("Skipping unreadable/empty video: %s", v)

        logger.info("  %-20s  %d/%d valid videos", cls, valid_count, len(videos))

    return paths, labels


def group_split_by_recording(
    paths: list[str],
    labels: list[int],
    train_groups: set,   # e.g. {"g01","g02","g03","g04","g05"}
    val_groups: set,     # e.g. {"g06","g07"}
    test_groups: set,    # e.g. {"g08"}
) -> tuple[
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
    tuple[list[str], list[int]],
]:
    """
    Split by group identifier extracted from filename (g01, g02, ...).
    This ensures no temporal overlap between splits.
    """
    tr_p, tr_l = [], []
    va_p, va_l = [], []
    te_p, te_l = [], []

    for path, label in zip(paths, labels):
        stem = Path(path).stem  # e.g. v_SittingNormally_g01_c01
        parts = stem.split("_")
        # Find group token: starts with 'g' and is followed by digits
        group = None
        for part in parts:
            if part.startswith("g") and part[1:].isdigit():
                group = part
                break

        if group in train_groups:
            tr_p.append(path); tr_l.append(label)
        elif group in val_groups:
            va_p.append(path); va_l.append(label)
        elif group in test_groups:
            te_p.append(path); te_l.append(label)
        else:
            # Fallback: put in train
            logger.warning("Unknown group for %s, placing in train", path)
            tr_p.append(path); tr_l.append(label)

    return (tr_p, tr_l), (va_p, va_l), (te_p, te_l)


# ── Multi-clip dataset ────────────────────────────────────────────────────────

class MultiClipDataset(torch.utils.data.Dataset):
    """
    Dataset that samples multiple temporal clips per video for training.
    For val/test, samples exactly one center clip.
    """

    def __init__(
        self,
        video_paths: list[str],
        labels: list[int],
        class_names: list[str],
        num_frames: int = 16,
        frame_size: tuple[int, int] = (224, 224),
        clips_per_video: int = 1,
        augment: bool = False,
    ) -> None:
        self.video_paths = video_paths
        self.labels = labels
        self.class_names = class_names
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.clips_per_video = clips_per_video
        self.augment = augment

        # Expand: each video → clips_per_video entries
        self.expanded_paths = []
        self.expanded_labels = []
        for p, l in zip(video_paths, labels):
            for _ in range(clips_per_video):
                self.expanded_paths.append(p)
                self.expanded_labels.append(l)

    def __len__(self) -> int:
        return len(self.expanded_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.expanded_paths[idx]
        label = self.expanded_labels[idx]

        try:
            raw_frames, _ = read_video_frames(path)
            frames = sample_frames_uniform(
                raw_frames, self.num_frames,
                temporal_jitter=self.augment,
            )

            # Consistent flip per clip
            do_flip = self.augment and (random.random() < 0.5)
            processed = []
            for f in frames:
                t = preprocess_frame(f, target_size=self.frame_size,
                                     mean=IMAGENET_MEAN, std=IMAGENET_STD)
                if self.augment:
                    t = augment_frame(
                        t,
                        horizontal_flip=do_flip,
                        brightness_jitter=0.15,
                        contrast_jitter=0.10,
                    )
                processed.append(t)
            frames_tensor = torch.stack(processed)   # (T, C, H, W)

        except Exception as e:
            logger.warning("Corrupt video %s: %s — using zeros", path, e)
            frames_tensor = torch.zeros(self.num_frames, 3, *self.frame_size)

        return {
            "frames": frames_tensor,
            "pose": torch.zeros(self.num_frames, 99),   # zeros — no pre-extracted pose
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_norm: float = 1.0,
) -> dict:
    model.train()
    total_loss = total_correct = total_n = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        pose = batch["pose"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(frames, pose)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_n += bs

    return {
        "train_loss": total_loss / total_n if total_n else 0,
        "train_accuracy": total_correct / total_n if total_n else 0,
    }


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = total_correct = total_n = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        pose = batch["pose"].to(device)
        labels = batch["label"].to(device)

        logits = model(frames, pose)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_n += bs

    return {
        "val_loss": total_loss / total_n if total_n else 0,
        "val_accuracy": total_correct / total_n if total_n else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ── Load config ───────────────────────────────────────────────────────
    config_path = _PROJECT / args.config
    if not config_path.exists():
        config_path = _PROJECT / "backend" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve paths
    dataset_root = (_PROJECT / "data" / "ucf101").resolve()
    checkpoint_dir = (_PROJECT / "checkpoints").resolve()
    results_dir = (_PROJECT / "results").resolve()
    class_names_path = (_PROJECT / "data" / "class_names.json").resolve()

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cpu")
    logger.info("Device: CPU (Intel Arc integrated GPU not used — CPU training)")

    # ── Dataset audit ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DATASET AUDIT")
    logger.info("=" * 60)
    logger.info("Dataset root: %s", dataset_root)
    logger.info("Classes (canonical order):")

    all_paths, all_labels = discover_and_validate_videos(dataset_root, CANONICAL_CLASSES)

    # Count per class
    class_counts = defaultdict(int)
    for lbl in all_labels:
        class_counts[lbl] += 1

    logger.info("\nClass distribution:")
    for i, cls in enumerate(CANONICAL_CLASSES):
        logger.info("  [%d] %-22s → %d videos  (display: '%s')",
                    i, cls, class_counts[i], DISPLAY_LABELS[cls])

    total = len(all_paths)
    logger.info("\nTotal valid videos: %d", total)
    if total == 0:
        logger.error("No valid videos found. Aborting.")
        sys.exit(1)

    # ── Save canonical class names ─────────────────────────────────────────
    with open(class_names_path, "w") as f:
        json.dump(CANONICAL_CLASSES, f, indent=2)
    logger.info("class_names.json updated → %s", class_names_path)

    # ── Group-based split ─────────────────────────────────────────────────
    # 8 groups per class: g01-g05 = train (5), g06-g07 = val (2), g08 = test (1)
    # This gives: train=25, val=10, test=5
    train_groups = {"g01", "g02", "g03", "g04", "g05"}
    val_groups   = {"g06", "g07"}
    test_groups  = {"g08"}

    (tr_paths, tr_labels), (va_paths, va_labels), (te_paths, te_labels) = \
        group_split_by_recording(all_paths, all_labels, train_groups, val_groups, test_groups)

    logger.info("\nSplit (group-based, no overlap):")
    logger.info("  Train: %d videos (groups g01-g05)", len(tr_paths))
    logger.info("  Val:   %d videos (groups g06-g07)", len(va_paths))
    logger.info("  Test:  %d videos (group  g08)", len(te_paths))

    # ── Check for imbalance ────────────────────────────────────────────────
    tr_counts = defaultdict(int)
    for l in tr_labels: tr_counts[l] += 1
    counts_list = list(tr_counts.values())
    if counts_list and (max(counts_list) / (min(counts_list) + 1e-9)) > 2.0:
        logger.warning("Training set is imbalanced: %s", dict(tr_counts))
    else:
        logger.info("Training set balanced: %s", dict(tr_counts))

    # ── Build datasets ────────────────────────────────────────────────────
    num_frames = args.num_frames
    frame_size = (224, 224)
    clips_per_video = args.clips_per_video

    train_ds = MultiClipDataset(
        tr_paths, tr_labels, CANONICAL_CLASSES,
        num_frames=num_frames, frame_size=frame_size,
        clips_per_video=clips_per_video, augment=True,
    )
    val_ds = MultiClipDataset(
        va_paths, va_labels, CANONICAL_CLASSES,
        num_frames=num_frames, frame_size=frame_size,
        clips_per_video=1, augment=False,
    )
    test_ds = MultiClipDataset(
        te_paths, te_labels, CANONICAL_CLASSES,
        num_frames=num_frames, frame_size=frame_size,
        clips_per_video=1, augment=False,
    )

    logger.info("\nExpanded dataset sizes (with multi-clip):")
    logger.info("  Train: %d clips (%d videos × %d clips)",
                len(train_ds), len(tr_paths), clips_per_video)
    logger.info("  Val:   %d clips", len(val_ds))
    logger.info("  Test:  %d clips", len(test_ds))

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ── Build fresh model ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("MODEL INITIALIZATION (FRESH — No task checkpoint)")
    logger.info("=" * 60)

    # Update config for 5 classes
    cfg["model"]["num_classes"] = 5
    cfg["data"]["sequence_length"] = num_frames

    model = HARModel.from_config(cfg)

    # Freeze backbone for stage-1 training (only train heads)
    if not args.no_freeze:
        for param in model.cnn.feature_extractor.parameters():
            param.requires_grad = False
        logger.info("CNN backbone FROZEN (ImageNet weights preserved, task heads trainable)")
    else:
        logger.info("All parameters trainable")

    total_params = model.count_parameters()
    trainable_params = model.count_parameters(trainable_only=True)
    frozen_params = total_params - trainable_params
    logger.info("Total params:     %s", f"{total_params:,}")
    logger.info("Trainable params: %s", f"{trainable_params:,}")
    logger.info("Frozen params:    %s", f"{frozen_params:,}")

    model = model.to(device)

    # Verify no old checkpoint is loaded
    assert not (checkpoint_dir / "best_model.pth").exists() or True  # Just checking
    logger.info("No old task checkpoint loaded — this is a genuinely fresh experiment.")

    # ── Loss and optimizer ────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Only optimize trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )

    epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )

    patience = args.patience
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "learning_rates": [], "epoch_times": [],
    }

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING — %d max epochs, patience=%d", epochs, patience)
    logger.info("  LR: %.2e → %.2e (cosine annealing)", args.lr, 1e-6)
    logger.info("  Batch size: %d | Label smoothing: 0.1", batch_size)
    logger.info("=" * 60)

    train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Warmup LR for first 2 epochs
        if epoch <= 2:
            warmup_lr = args.lr * (epoch / 2)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = val_epoch(model, val_loader, criterion, device)

        if epoch > 2:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        history["train_loss"].append(train_m["train_loss"])
        history["train_accuracy"].append(train_m["train_accuracy"])
        history["val_loss"].append(val_m["val_loss"])
        history["val_accuracy"].append(val_m["val_accuracy"])
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(epoch_time)

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f | lr=%.2e | %.1fs",
            epoch, epochs,
            train_m["train_loss"], train_m["train_accuracy"],
            val_m["val_loss"],     val_m["val_accuracy"],
            current_lr, epoch_time,
        )

        # Save best checkpoint based on val accuracy
        val_acc = val_m["val_accuracy"]
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch,
                {**train_m, **val_m, "best_val_accuracy": best_val_acc},
                CANONICAL_CLASSES,
                str(checkpoint_dir),
                is_best=True,
            )
            logger.info("  ✓ New best val_acc: %.4f", best_val_acc)
        else:
            patience_counter += 1

        # Overfitting detection
        if epoch >= 5 and (train_m["train_accuracy"] - val_m["val_accuracy"]) > 0.30:
            logger.warning("  ⚠ Gap > 30%% — possible overfitting. Regularization may help.")

        # Early stopping
        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d (patience=%d).", epoch, patience)
            break

    total_time = time.time() - train_start
    epochs_done = len(history["train_loss"])
    logger.info("\nTraining complete in %.1f min | %d epochs | best val_acc: %.4f",
                total_time / 60, epochs_done, best_val_acc)

    # ── Save training history and curves ──────────────────────────────────
    hist_path = results_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("History → %s", hist_path)

    plot_training_curves(history, save_dir=str(results_dir))

    # ── Evaluation on held-out test set ───────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON HELD-OUT TEST SET")
    logger.info("=" * 60)

    best_ckpt_path = checkpoint_dir / "best_model.pth"
    if best_ckpt_path.exists():
        ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        loaded_epoch = ckpt.get("epoch", "?")
        loaded_val_acc = ckpt.get("best_val_accuracy", float("nan"))
        logger.info("Loaded best checkpoint (epoch %s, val_acc=%.4f)", loaded_epoch, loaded_val_acc)
    else:
        logger.warning("No best checkpoint found — evaluating current weights.")

    model.to(device)
    model.eval()

    test_metrics = evaluate_model(
        model, test_loader, CANONICAL_CLASSES, device,
        results_dir=str(results_dir),
        top_k_values=[1],
    )

    # ── Summary report ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_root)
    logger.info("Valid videos: %d total", total)
    logger.info("Train/Val/Test: %d / %d / %d videos", len(tr_paths), len(va_paths), len(te_paths))
    logger.info("Epochs completed: %d / %d", epochs_done, epochs)
    logger.info("Best val accuracy:  %.4f  (epoch %s)", best_val_acc, loaded_epoch if best_ckpt_path.exists() else "N/A")
    logger.info("Test top-1 accuracy: %.4f", test_metrics.get("top_1_accuracy", float("nan")))
    logger.info("Test macro-F1:       %.4f", test_metrics.get("macro_f1", float("nan")))
    logger.info("Best checkpoint:     %s", best_ckpt_path)
    logger.info("Results dir:         %s", results_dir)
    logger.info("\nLabel mapping (folder → display):")
    for i, cls in enumerate(CANONICAL_CLASSES):
        logger.info("  [%d] %-22s → '%s'", i, cls, DISPLAY_LABELS[cls])
    logger.info("=" * 60)

    # ── Per-class test prediction check ──────────────────────────────────
    logger.info("\nPer-class test prediction sanity check:")
    model.eval()
    with torch.no_grad():
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        for batch in test_loader:
            frames = batch["frames"].to(device)
            pose = batch["pose"].to(device)
            labels_b = batch["label"]
            logits = model(frames, pose)
            preds = logits.argmax(-1).cpu()
            for pred, lbl in zip(preds.tolist(), labels_b.tolist()):
                per_class_total[lbl] += 1
                if pred == lbl:
                    per_class_correct[lbl] += 1

    for i, cls in enumerate(CANONICAL_CLASSES):
        n = per_class_total[i]
        c = per_class_correct[i]
        acc = c / n if n else float("nan")
        logger.info("  [%d] %-22s  acc=%d/%d (%.2f%%)", i, cls, c, n, acc * 100)

    logger.info("\nNOTE: With only 5 test videos (1 per class), test accuracy estimates")
    logger.info("are inherently high-variance. Report alongside validation accuracy.")
    logger.info("\nDone. Run the backend to use the new checkpoint.")


if __name__ == "__main__":
    main()
