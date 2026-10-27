#!/usr/bin/env python3
"""
Local Fine-Tuning Script — HAR Deep Vision (CPU-Optimized)
============================================================
Trains the HAR model on the existing small UCF101 subset with:
  - Frozen MobileNetV3 backbone (uses pretrained ImageNet features)
  - Heavy data augmentation to compensate for small dataset
  - Multiple clips sampled per video (multi-crop)
  - Lower LR + cosine schedule
  - No AMP (CPU training)

Usage (from har-project directory):
    python scripts/finetune_local.py [--epochs 50] [--lr 0.001]

Expected outcome: ~50-80% accuracy on 6 classes with 24 videos.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from models.har_model import HARModel
from preprocessing.frame_processor import (
    read_video_frames, sample_frames_uniform, preprocess_frame, augment_frame,
    IMAGENET_MEAN, IMAGENET_STD,
)
from training.evaluator import evaluate_model, plot_confusion_matrix, plot_training_curves
from training.trainer import set_seed, save_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def discover_videos(root: Path, selected_classes: list[str] | None = None):
    """Return (paths, labels, class_names)."""
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    all_names = [d.name for d in class_dirs]
    if selected_classes:
        names = [c for c in selected_classes if c in all_names]
    else:
        names = all_names
    cls_to_idx = {n: i for i, n in enumerate(names)}
    paths, labels = [], []
    for name in names:
        for f in sorted((root / name).iterdir()):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(str(f))
                labels.append(cls_to_idx[name])
    return paths, labels, names


class MultiClipDataset(torch.utils.data.Dataset):
    """
    Each video is expanded into `clips_per_video` randomly-sampled clips.
    Heavy augmentation is applied to combat the tiny dataset.
    """

    def __init__(
        self,
        video_paths: list[str],
        labels: list[int],
        class_names: list[str],
        num_frames: int = 16,
        frame_size: tuple[int, int] = (224, 224),
        clips_per_video: int = 8,
        augment: bool = True,
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.class_names = class_names
        self.augment = augment

        # Expand: each video yields `clips_per_video` entries
        self.samples = []
        for path, label in zip(video_paths, labels):
            for _ in range(clips_per_video):
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            raw, _ = read_video_frames(path, max_frames=200)
        except Exception:
            raw = [np.zeros((224, 224, 3), dtype=np.uint8)]

        # Random temporal jitter clip
        frames = sample_frames_uniform(raw, self.num_frames, temporal_jitter=self.augment)

        do_flip = self.augment and random.random() < 0.5
        processed = []
        for f in frames:
            t = preprocess_frame(f, target_size=self.frame_size,
                                 mean=IMAGENET_MEAN, std=IMAGENET_STD)
            if self.augment:
                t = augment_frame(t, horizontal_flip=do_flip,
                                  brightness_jitter=0.3, contrast_jitter=0.3)
            processed.append(t)

        frames_tensor = torch.stack(processed)  # (T, C, H, W)
        pose_zeros = torch.zeros(self.num_frames, 99)

        return {
            "frames": frames_tensor,
            "pose": pose_zeros,
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


def build_split(
    video_paths: list[str],
    labels: list[int],
    class_names: list[str],
    seed: int = 42,
):
    """Return (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)."""
    rng = random.Random(seed)
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)

    tr, va, te = [], [], []
    for cls_indices in groups.values():
        shuffled = list(cls_indices)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n >= 3:
            te.append(shuffled[0])
            va.append(shuffled[1])
            tr.extend(shuffled[2:])
        elif n == 2:
            te.append(shuffled[0])
            tr.append(shuffled[1])
        else:
            tr.extend(shuffled)

    def sub(idx_list):
        return [video_paths[i] for i in idx_list], [labels[i] for i in idx_list]

    return sub(tr) + sub(va) + sub(te)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--clips-per-video", type=int, default=8,
                   help="Clips sampled per video per epoch (data augmentation via multi-crop)")
    p.add_argument("--config", default="backend/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    root = Path(__file__).resolve().parent.parent
    cfg_path = root / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dataset_path = root / "data" / "ucf101"
    checkpoint_dir = root / "checkpoints"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    selected = cfg["data"].get("selected_classes") or None
    video_paths, labels, class_names = discover_videos(dataset_path, selected)
    logger.info("Found %d videos, %d classes: %s", len(video_paths), len(class_names), class_names)

    tr_paths, tr_lbl, va_paths, va_lbl, te_paths, te_lbl = build_split(
        video_paths, labels, class_names, seed=42
    )
    logger.info("Split — Train: %d, Val: %d, Test: %d", len(tr_paths), len(va_paths), len(te_paths))

    num_frames = cfg["data"]["sequence_length"]
    frame_size = tuple(cfg["data"]["frame_size"])
    clips = args.clips_per_video

    train_ds = MultiClipDataset(tr_paths, tr_lbl, class_names,
                                num_frames=num_frames, frame_size=frame_size,
                                clips_per_video=clips, augment=True)
    val_ds   = MultiClipDataset(va_paths, va_lbl, class_names,
                                num_frames=num_frames, frame_size=frame_size,
                                clips_per_video=1, augment=False)
    test_ds  = MultiClipDataset(te_paths, te_lbl, class_names,
                                num_frames=num_frames, frame_size=frame_size,
                                clips_per_video=3, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
    val_loader   = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=4, shuffle=False, num_workers=0)

    logger.info("Train clips: %d, Val clips: %d, Test clips: %d",
                len(train_ds), len(val_ds), len(test_ds))

    # Build model
    cfg["model"]["num_classes"] = len(class_names)
    cfg["model"]["use_pose"] = False  # no pose during fast local training
    model = HARModel.from_config(cfg)

    # Freeze CNN backbone — use pretrained ImageNet features
    for param in model.cnn.feature_extractor.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters (backbone frozen): %s", f"{trainable:,}")

    device = torch.device("cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_val_acc = 0.0
    patience_count = 0
    patience_limit = 15
    history = {"train_loss": [], "train_accuracy": [],
               "val_loss": [], "val_accuracy": [], "learning_rates": [], "epoch_times": []}

    logger.info("=" * 60)
    logger.info("Starting fine-tuning: %d epochs, clips_per_video=%d", args.epochs, clips)
    logger.info("=" * 60)

    import time

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch in train_loader:
            frames = batch["frames"].to(device)
            pose = batch["pose"].to(device)
            labels_b = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(frames, None)  # no pose
            loss = criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = labels_b.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(-1) == labels_b).sum().item()
            total_samples += bs

        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                labels_b = batch["label"].to(device)
                logits = model(frames, None)
                loss = criterion(logits, labels_b)
                val_loss += loss.item() * labels_b.size(0)
                val_correct += (logits.argmax(-1) == labels_b).sum().item()
                val_total += labels_b.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        scheduler.step()
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        history["epoch_times"].append(epoch_time)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f | %.1fs",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, epoch_time
        )

        # Save best
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            patience_count = 0
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": class_names,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "best_val_accuracy": best_val_acc,
            }
            best_path = str(checkpoint_dir / "best_model.pth")
            torch.save(payload, best_path)
            logger.info("  [+] New best val_acc=%.4f → saved to %s", best_val_acc, best_path)
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Save history
    history_path = str(results_dir / "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, save_dir=str(results_dir))

    # Evaluation
    logger.info("=" * 60)
    logger.info("EVALUATION on test set")
    logger.info("=" * 60)

    # Load best checkpoint
    best_ckpt = checkpoint_dir / "best_model.pth"
    if best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded best checkpoint (val_acc=%.4f)", ckpt.get("best_val_accuracy", 0))

    from sklearn.metrics import classification_report, confusion_matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            frames = batch["frames"].to(device)
            logits = model(frames, None)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(all_labels, all_preds)
    logger.info("Test accuracy: %.4f (%d samples)", acc, len(all_labels))

    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    logger.info("\n%s", report)

    report_path = str(results_dir / "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Human Activity Recognition -- Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    import numpy as np
    plot_confusion_matrix(
        np.array(all_labels), np.array(all_preds), class_names,
        save_path=str(results_dir / "confusion_matrix.png"),
    )

    # Save metrics.json
    from sklearn.metrics import f1_score, precision_score, recall_score
    metrics = {
        "top_1_accuracy": round(acc, 4),
        "macro_precision": round(float(precision_score(all_labels, all_preds, average="macro", zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(all_labels, all_preds, average="macro", zero_division=0)), 4),
        "macro_f1":        round(float(f1_score(all_labels, all_preds, average="macro", zero_division=0)), 4),
        "weighted_f1":     round(float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)), 4),
        "total_parameters": model.count_parameters(),
        "trainable_parameters": model.count_parameters(trainable_only=True),
        "num_test_samples": len(all_labels),
        "num_classes_evaluated": len(class_names),
        "note": "Trained with frozen MobileNetV3 backbone on 24-video local subset",
    }
    with open(str(results_dir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Results saved to: %s", results_dir)
    logger.info("Best val_accuracy: %.4f", best_val_acc)
    logger.info("Test accuracy:     %.4f", acc)


if __name__ == "__main__":
    main()
