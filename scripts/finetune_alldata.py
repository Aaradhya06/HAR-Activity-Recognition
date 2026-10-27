#!/usr/bin/env python3
"""
All-Data Fine-Tuning Script — HAR Deep Vision (CPU-Optimized)
==============================================================
Trains on ALL 24 available videos (no train/val split) with:
  - Frozen MobileNetV3 backbone (ImageNet features)
  - Heavy multi-clip augmentation (15 clips per video)
  - Lower learning rate with warmup
  - Evaluates on full dataset after training

Usage (from har-project directory):
    python scripts/finetune_alldata.py [--epochs 80]

With only 24 videos this trains in ~15 min on CPU.
Expected test accuracy: 60-90% (evaluated on training data).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import random
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from models.har_model import HARModel
from preprocessing.frame_processor import (
    read_video_frames, sample_frames_uniform, preprocess_frame, augment_frame,
    IMAGENET_MEAN, IMAGENET_STD,
)
from training.evaluator import plot_confusion_matrix, plot_training_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_videos(root: Path, selected_classes=None):
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    all_names = [d.name for d in class_dirs]
    names = [c for c in selected_classes if c in all_names] if selected_classes else all_names
    cls_to_idx = {n: i for i, n in enumerate(names)}
    paths, labels = [], []
    for name in names:
        for f in sorted((root / name).iterdir()):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(str(f))
                labels.append(cls_to_idx[name])
    return paths, labels, names


class AllDataMultiClipDataset(torch.utils.data.Dataset):
    """All videos, each sampled clips_per_video times with augmentation."""

    def __init__(self, video_paths, labels, class_names, num_frames=16,
                 frame_size=(224, 224), clips_per_video=15, augment=True):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.class_names = class_names
        self.augment = augment
        self.samples = [(p, l) for p, l in zip(video_paths, labels)
                        for _ in range(clips_per_video)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            raw, _ = read_video_frames(path, max_frames=200)
        except Exception:
            raw = [np.zeros((224, 224, 3), dtype=np.uint8)]

        frames = sample_frames_uniform(raw, self.num_frames, temporal_jitter=self.augment)
        do_flip = self.augment and random.random() < 0.5
        processed = []
        for f in frames:
            t = preprocess_frame(f, target_size=self.frame_size,
                                 mean=IMAGENET_MEAN, std=IMAGENET_STD)
            if self.augment:
                t = augment_frame(t, horizontal_flip=do_flip,
                                  brightness_jitter=0.25, contrast_jitter=0.25)
            processed.append(t)

        return {
            "frames": torch.stack(processed),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


class EvalDataset(torch.utils.data.Dataset):
    """Non-augmented evaluation — 4 deterministic clips per video."""

    def __init__(self, video_paths, labels, num_frames=16,
                 frame_size=(224, 224), clips_per_video=4):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.samples = [(p, l) for p, l in zip(video_paths, labels)
                        for _ in range(clips_per_video)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            raw, _ = read_video_frames(path, max_frames=200)
        except Exception:
            raw = [np.zeros((224, 224, 3), dtype=np.uint8)]

        frames = sample_frames_uniform(raw, self.num_frames, temporal_jitter=False)
        processed = [preprocess_frame(f, target_size=self.frame_size,
                                      mean=IMAGENET_MEAN, std=IMAGENET_STD)
                     for f in frames]
        return {
            "frames": torch.stack(processed),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--clips-per-video", type=int, default=15)
    p.add_argument("--config", default="backend/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    root = Path(__file__).resolve().parent.parent
    with open(root / args.config) as f:
        cfg = yaml.safe_load(f)

    checkpoint_dir = root / "checkpoints"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    selected = cfg["data"].get("selected_classes") or None
    dataset_path = root / "data" / "ucf101"
    video_paths, labels, class_names = discover_videos(dataset_path, selected)
    logger.info("All %d videos, %d classes: %s", len(video_paths), len(class_names), class_names)

    # Save class names
    class_names_path = root / "data" / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)

    num_frames = cfg["data"]["sequence_length"]  # 16
    frame_size = tuple(cfg["data"]["frame_size"])  # (224, 224)
    clips = args.clips_per_video

    train_ds = AllDataMultiClipDataset(video_paths, labels, class_names,
                                       num_frames=num_frames, frame_size=frame_size,
                                       clips_per_video=clips, augment=True)
    eval_ds = EvalDataset(video_paths, labels, num_frames=num_frames,
                          frame_size=frame_size, clips_per_video=4)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=6, shuffle=True, num_workers=0, drop_last=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=6, shuffle=False, num_workers=0)

    logger.info("Train clips: %d (%d batches), Eval clips: %d",
                len(train_ds), len(train_loader), len(eval_ds))

    # Build model (without pose for speed)
    cfg["model"]["num_classes"] = len(class_names)
    cfg["model"]["use_pose"] = False
    model = HARModel.from_config(cfg)

    # Freeze CNN backbone
    for param in model.cnn.feature_extractor.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters (backbone frozen): %s", f"{trainable:,}")

    device = torch.device("cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=5e-4
    )

    # Cosine schedule with warmup
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-5
    )

    best_train_acc = 0.0
    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "learning_rates": [], "epoch_times": [],
    }

    logger.info("=" * 60)
    logger.info("All-data training: %d epochs, clips/video=%d", args.epochs, clips)
    logger.info("NOTE: Training on full dataset — eval is in-sample (demo metric)")
    logger.info("For true generalization, use Google Colab with more UCF101 data")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # LR warmup
        if epoch <= warmup_epochs:
            warmup_lr = args.lr * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch in train_loader:
            frames = batch["frames"].to(device)
            labels_b = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames, None)
            loss = criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = labels_b.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(-1) == labels_b).sum().item()
            total_samples += bs

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        if epoch > warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - t0

        # Use train metrics as val (honest in logs about this)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(train_loss)     # same — no separate val set
        history["val_accuracy"].append(train_acc)  # same
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        history["epoch_times"].append(epoch_time)

        logger.info(
            "Epoch %3d/%d | loss=%.4f acc=%.4f | lr=%.2e | %.1fs",
            epoch, args.epochs, train_loss, train_acc,
            optimizer.param_groups[0]["lr"], epoch_time
        )

        # Save checkpoint when train acc improves > 0.5%
        if train_acc > best_train_acc + 0.005:
            best_train_acc = train_acc
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": class_names,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": train_loss,
                "val_accuracy": train_acc,
                "best_val_accuracy": train_acc,
                "use_pose": False,
                "note": "Trained on full 24-video subset, backbone frozen",
            }
            torch.save(payload, str(checkpoint_dir / "best_model.pth"))
            logger.info("  [+] Saved best checkpoint (train_acc=%.4f)", train_acc)

        # Early stop at high accuracy
        if train_acc >= 0.97:
            logger.info("Reached %.1f%% training accuracy — stopping early.", train_acc * 100)
            break

    # Save history and plots
    with open(str(results_dir / "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, save_dir=str(results_dir))

    # Evaluate on full dataset (all 4 clips per video, majority vote per video)
    logger.info("=" * 60)
    logger.info("EVALUATION (full-dataset, 4 clips per video, majority vote)")
    logger.info("=" * 60)

    ckpt = torch.load(str(checkpoint_dir / "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            frames = batch["frames"].to(device)
            logits = model(frames, None)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())
            all_paths.extend(batch["path"])

    # Majority vote across clips for each video
    from collections import defaultdict, Counter
    video_preds: dict[str, list] = defaultdict(list)
    video_label: dict[str, int] = {}
    for pred, label, path in zip(all_preds, all_labels, all_paths):
        video_preds[path].append(pred)
        video_label[path] = label

    final_preds = []
    final_labels = []
    for path in video_preds:
        majority = Counter(video_preds[path]).most_common(1)[0][0]
        final_preds.append(majority)
        final_labels.append(video_label[path])

    acc = accuracy_score(final_labels, final_preds)
    logger.info("Per-video accuracy (majority vote): %.4f (%d videos)", acc, len(final_labels))
    logger.info("\n%s", classification_report(final_labels, final_preds,
                                              target_names=class_names, zero_division=0))

    # Measure actual inference latency on a single batch
    model.eval()
    _sample_batch = next(iter(eval_loader))
    _sample_frames = _sample_batch["frames"].to(device)
    # Warmup
    with torch.no_grad():
        _ = model(_sample_frames, None)
    _latency_runs = []
    with torch.no_grad():
        for _ in range(10):
            _t0 = time.time()
            _ = model(_sample_frames, None)
            _latency_runs.append((time.time() - _t0) * 1000 / _sample_frames.size(0))
    avg_latency_ms = round(float(np.mean(_latency_runs[2:])), 2)  # discard first 2 warmup runs

    # Save metrics
    metrics = {
        "top_1_accuracy": round(acc, 4),
        "macro_precision": round(float(precision_score(final_labels, final_preds, average="macro", zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(final_labels, final_preds, average="macro", zero_division=0)), 4),
        "macro_f1":        round(float(f1_score(final_labels, final_preds, average="macro", zero_division=0)), 4),
        "weighted_f1":     round(float(f1_score(final_labels, final_preds, average="weighted", zero_division=0)), 4),
        "total_parameters": model.count_parameters(),
        "trainable_parameters": model.count_parameters(trainable_only=True),
        "num_test_samples": len(final_labels),
        "num_classes_evaluated": len(class_names),
        "avg_inference_ms_per_sample": avg_latency_ms,
        "fps_estimate": round(1000.0 / max(avg_latency_ms, 0.01), 2),
        "evaluation_note": (
            "IN-SAMPLE evaluation on 24-video training set. "
            "For true generalization use Colab with full UCF101 data. "
            "Backbone: MobileNetV3-Small (frozen). Temporal: Transformer."
        ),
    }

    with open(str(results_dir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        np.array(final_labels), np.array(final_preds), class_names,
        save_path=str(results_dir / "confusion_matrix.png"),
    )

    report = classification_report(final_labels, final_preds,
                                   target_names=class_names, zero_division=0)
    with open(str(results_dir / "classification_report.txt"), "w") as f:
        f.write("Human Activity Recognition -- Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write("NOTE: In-sample evaluation on 24-video training set\n\n")
        f.write(report)

    logger.info("Best per-video accuracy: %.4f", acc)
    logger.info("Results saved to: %s", results_dir)


if __name__ == "__main__":
    main()
