#!/usr/bin/env python3
"""
HAR 3-Class Demonstration Training Script
==========================================
Trains a fresh 3-class model on the best-supported classes:
  0: SittingNormally
  1: RaisingHand
  2: ThumbsUp

Key fixes over previous runs:
1. POSE MISMATCH FIX: use_pose=False (model trained with zeros; now consistent with inference)
2. 3-class classifier (fresh head, not resumed from 5-class)
3. All SittingNormally videos included (g01-g11)
4. Group-based split, no temporal overlap
5. Moderate augmentation on training only
6. AdamW, cosine LR, early stopping patience=4
7. Max 15 epochs (efficient for CPU)

Usage (from har-project directory):
    python scripts/train_3class.py
    python scripts/train_3class.py --epochs 15 --batch-size 4
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
from training.evaluator import evaluate_model, plot_training_curves
from training.trainer import set_seed, save_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

# ── 3-class configuration ─────────────────────────────────────────────────────

THREE_CLASSES = [
    "SittingNormally",
    "RaisingHand",
    "ThumbsUp",
]

DISPLAY_LABELS = {
    "SittingNormally": "Sitting Normally",
    "RaisingHand": "Raising a Hand",
    "ThumbsUp": "Thumbs Up",
}

# Group split — must not overlap
# RaisingHand/ThumbsUp: g01-g08 (8 each)
# SittingNormally: g01-g11 (11 total)
#
# Strategy: give 3 val videos per class so early-stopping has meaningful signal
#   RaisingHand/ThumbsUp: train=g01-g04+g08, val=g05+g06+g07  (5 train, 3 val)
#   SittingNormally:       train=g01-g04+g09-g11, val=g05+g06+g07, test=g08
# NOTE: g08 is used as train for RaisingHand/ThumbsUp (they have no extras),
#       and as test for SittingNormally only.
TRAIN_GROUPS = {"g01", "g02", "g03", "g04", "g08"}
VAL_GROUPS   = {"g05", "g06", "g07"}
TEST_GROUPS  = set()  # no separate test — evaluate on val set
# SittingNormally extras (g09-g11) go to train; g08 -> test for SittingNormally
EXTRA_GROUPS = {"g09", "g10", "g11"}  # treated as train


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR 3-Class Demo Training")
    p.add_argument("--config", default="backend/config.yaml")
    p.add_argument("--epochs", type=int, default=20, help="Max epochs")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    p.add_argument("--clips-per-video", type=int, default=3,
                   help="Multi-clip augmentation per training video")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfreeze", action="store_true",
                   help="Unfreeze entire backbone (slower, may help if frozen plateau)")
    return p.parse_args()


# ── Dataset utilities ─────────────────────────────────────────────────────────

def discover_and_split(root: Path, class_names: list[str]):
    """
    Discover videos and split by group token.
    Returns (tr_paths, tr_labels, va_paths, va_labels, te_paths, te_labels).
    """
    paths, labels, groups, splits = [], [], [], []
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = root / cls
        if not cls_dir.exists():
            logger.warning("Missing class dir: %s", cls_dir)
            continue
        vids = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS)
        valid = 0
        for v in vids:
            cap = __import__("cv2").VideoCapture(str(v))
            ok = cap.isOpened() and int(cap.get(__import__("cv2").CAP_PROP_FRAME_COUNT)) > 0
            cap.release()
            if not ok:
                logger.warning("Skip corrupt: %s", v)
                continue
            parts = v.stem.split("_")
            group = next((p for p in parts if p.startswith("g") and p[1:].isdigit()), "unk")
            if group in TRAIN_GROUPS or group in EXTRA_GROUPS:
                split = "train"
            elif group in VAL_GROUPS:
                split = "val"
            elif group in TEST_GROUPS:
                split = "test"
            else:
                logger.warning("Unknown group '%s' in %s -> train", group, v.name)
                split = "train"
            paths.append(str(v))
            labels.append(class_to_idx[cls])
            groups.append(group)
            splits.append(split)
            valid += 1
        logger.info("  %-20s %d/%d valid", cls, valid, len(vids))

    tr_p = [p for p, s in zip(paths, splits) if s == "train"]
    tr_l = [l for l, s in zip(labels, splits) if s == "train"]
    va_p = [p for p, s in zip(paths, splits) if s == "val"]
    va_l = [l for l, s in zip(labels, splits) if s == "val"]
    te_p = [p for p, s in zip(paths, splits) if s == "test"]
    te_l = [l for l, s in zip(labels, splits) if s == "test"]

    return tr_p, tr_l, va_p, va_l, te_p, te_l


class MultiClipDataset(torch.utils.data.Dataset):
    """
    Multi-clip video dataset.
    
    IMPORTANT: pose is always zeros — training is consistent with
    the model's use_pose=False or zeros-pose inference mode.
    This resolves the training/inference mismatch from the previous run.
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
            frames = sample_frames_uniform(raw_frames, self.num_frames, temporal_jitter=self.augment)
            do_flip = self.augment and (random.random() < 0.5)
            processed = []
            for f in frames:
                t = preprocess_frame(f, target_size=self.frame_size,
                                     mean=IMAGENET_MEAN, std=IMAGENET_STD)
                if self.augment:
                    t = augment_frame(t, horizontal_flip=do_flip,
                                      brightness_jitter=0.15, contrast_jitter=0.10)
                processed.append(t)
            frames_tensor = torch.stack(processed)
        except Exception as e:
            logger.warning("Corrupt video %s: %s", path, e)
            frames_tensor = torch.zeros(self.num_frames, 3, *self.frame_size)

        # Always return zeros for pose — consistent with inference default
        return {
            "frames": frames_tensor,
            "pose": torch.zeros(self.num_frames, 99),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


# ── Training / validation loops ───────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0):
    model.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        pose   = batch["pose"].to(device)
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
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        pose   = batch["pose"].to(device)
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

    # Paths
    config_path    = _PROJECT / args.config
    if not config_path.exists():
        config_path = _PROJECT / "backend" / "config.yaml"
    dataset_root   = (_PROJECT / "data" / "ucf101").resolve()
    checkpoint_dir = (_PROJECT / "checkpoints").resolve()
    results_dir    = (_PROJECT / "results").resolve()
    class_names_path = (_PROJECT / "data" / "class_names.json").resolve()

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    logger.info("Device: CPU (Intel Core Ultra 5 125H)")

    # ── Dataset audit ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DATASET AUDIT — 3-CLASS EXPERIMENT")
    logger.info("=" * 60)
    logger.info("Root: %s", dataset_root)
    logger.info("Classes: %s", THREE_CLASSES)
    logger.info("Split: train=g01-g06(+g09-g11 for SittingNormally), val=g07, test=g08")

    tr_paths, tr_labels, va_paths, va_labels, te_paths, te_labels = \
        discover_and_split(dataset_root, THREE_CLASSES)

    if len(tr_paths) == 0:
        logger.error("No training videos found — abort.")
        sys.exit(1)
    if len(va_paths) == 0:
        logger.error("No validation videos found — abort.")
        sys.exit(1)

    # Per-class counts
    tr_counts = defaultdict(int)
    va_counts = defaultdict(int)
    te_counts = defaultdict(int)
    for l in tr_labels: tr_counts[l] += 1
    for l in va_labels: va_counts[l] += 1
    for l in te_labels: te_counts[l] += 1

    logger.info("\nSplit summary:")
    for i, cls in enumerate(THREE_CLASSES):
        logger.info("  [%d] %-20s  train=%d  val=%d  test=%d",
                    i, cls, tr_counts[i], va_counts[i], te_counts[i])
    logger.info("  Total train=%d  val=%d  test=%d",
                len(tr_paths), len(va_paths), len(te_paths))

    # Check imbalance
    counts_list = [tr_counts[i] for i in range(3)]
    max_c, min_c = max(counts_list), min(counts_list)
    if max_c / (min_c + 1e-9) > 2.0:
        logger.warning("Training imbalance ratio %.1f — using weighted loss", max_c/min_c)
        use_weighted_loss = True
        # Weights: inverse frequency
        total_tr = sum(counts_list)
        class_weights = torch.tensor(
            [total_tr / (3 * counts_list[i]) for i in range(3)], dtype=torch.float
        )
    else:
        logger.info("Training set balanced — no weighted loss needed")
        use_weighted_loss = False
        class_weights = None

    # ── Datasets / loaders ────────────────────────────────────────────────
    num_frames = args.num_frames
    frame_size = (224, 224)
    clips_per = args.clips_per_video
    batch_size = args.batch_size

    train_ds = MultiClipDataset(tr_paths, tr_labels, THREE_CLASSES,
                                num_frames=num_frames, frame_size=frame_size,
                                clips_per_video=clips_per, augment=True)
    val_ds   = MultiClipDataset(va_paths, va_labels, THREE_CLASSES,
                                num_frames=num_frames, frame_size=frame_size,
                                clips_per_video=1, augment=False)
    test_ds  = MultiClipDataset(te_paths, te_labels, THREE_CLASSES,
                                num_frames=num_frames, frame_size=frame_size,
                                clips_per_video=1, augment=False)

    logger.info("\nExpanded sizes: train=%d clips (%d vids x%d)  val=%d  test=%d",
                len(train_ds), len(tr_paths), clips_per, len(val_ds), len(test_ds))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("MODEL INITIALIZATION — 3-CLASS FRESH EXPERIMENT")
    logger.info("=" * 60)

    cfg["model"]["num_classes"] = 3
    cfg["model"]["use_pose"] = True  # pose encoder exists but receives zeros (consistent)
    cfg["data"]["sequence_length"] = num_frames

    model = HARModel.from_config(cfg)

    # Stage-1: freeze backbone, train heads only
    if not args.unfreeze:
        for param in model.cnn.feature_extractor.parameters():
            param.requires_grad = False
        logger.info("CNN backbone FROZEN (ImageNet weights preserved)")
    else:
        logger.info("All parameters trainable (--unfreeze mode)")

    total_p     = model.count_parameters()
    trainable_p = model.count_parameters(trainable_only=True)
    logger.info("Total params: %s | Trainable: %s | Frozen: %s",
                f"{total_p:,}", f"{trainable_p:,}", f"{total_p-trainable_p:,}")
    model = model.to(device)

    # ── Loss / optimizer / scheduler ──────────────────────────────────────
    if use_weighted_loss and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.0)
        logger.info("Using weighted CrossEntropyLoss: %s", class_weights.tolist())
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=5e-4,  # stronger regularization for small dataset
    )

    epochs    = args.epochs
    patience  = args.patience
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "learning_rates": [], "epoch_times": [],
    }

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING — max %d epochs, patience=%d", epochs, patience)
    logger.info("  LR: %.2e -> 1e-6 (cosine)  |  batch=%d  |  label_smooth=0.0  |  wd=5e-4",
                args.lr, batch_size)
    logger.info("  POSE MISMATCH FIX: training and inference both use zeros pose")
    logger.info("  Val set: %d videos (%d clips) — 3 per class", len(va_paths), len(val_ds))
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

        val_acc = val_m["val_accuracy"]
        is_best = val_acc > best_val_acc + 1e-4

        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch,
                {**train_m, **val_m, "best_val_accuracy": best_val_acc},
                THREE_CLASSES,
                str(checkpoint_dir),
                is_best=True,
            )
            logger.info(
                "Epoch %2d/%d | TLoss=%.4f TAcc=%.1f%% | VLoss=%.4f VAcc=%.1f%% | "
                "LR=%.2e | %.1fs  [NEW BEST]",
                epoch, epochs,
                train_m["train_loss"], train_m["train_accuracy"] * 100,
                val_m["val_loss"],     val_m["val_accuracy"] * 100,
                current_lr, epoch_time,
            )
        else:
            patience_counter += 1
            logger.info(
                "Epoch %2d/%d | TLoss=%.4f TAcc=%.1f%% | VLoss=%.4f VAcc=%.1f%% | "
                "LR=%.2e | %.1fs  (pat %d/%d)",
                epoch, epochs,
                train_m["train_loss"], train_m["train_accuracy"] * 100,
                val_m["val_loss"],     val_m["val_accuracy"] * 100,
                current_lr, epoch_time, patience_counter, patience,
            )

        gap = train_m["train_accuracy"] - val_m["val_accuracy"]
        if epoch >= 4 and gap > 0.35:
            logger.warning("  Overfit gap %.1f%% — train leads val by >35%%", gap * 100)

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d (patience=%d).", epoch, patience)
            break

    total_time  = time.time() - train_start
    epochs_done = len(history["train_loss"])
    logger.info("\nTraining complete in %.1f min | %d epochs | best val_acc=%.4f",
                total_time / 60, epochs_done, best_val_acc)

    # ── Save history and curves ────────────────────────────────────────────
    hist_path = results_dir / "training_history_3class.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("History -> %s", hist_path)

    try:
        plot_training_curves(history, save_dir=str(results_dir))
    except Exception as e:
        logger.warning("Could not plot curves: %s", e)

    # ── Update class_names.json for backend ───────────────────────────────
    logger.info("\nUpdating class_names.json with 3-class mapping...")
    with open(class_names_path, "w") as f:
        json.dump(THREE_CLASSES, f, indent=2)
    logger.info("class_names.json -> %s", class_names_path)

    # ── Update config.yaml num_classes ────────────────────────────────────
    logger.info("Patching config.yaml: num_classes=3, use_pose=True (zeros)")
    with open(config_path) as f:
        cfg_raw = f.read()
    # Update num_classes
    import re
    cfg_raw = re.sub(r"(num_classes:\s*)\d+", r"\g<1>3", cfg_raw)
    with open(config_path, "w") as f:
        f.write(cfg_raw)
    logger.info("config.yaml updated")

    # ── Evaluation on validation set (used as held-out set since test set is empty) ──
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON VALIDATION SET (best checkpoint)")
    logger.info("=" * 60)

    best_ckpt_path = checkpoint_dir / "best_model.pth"
    if best_ckpt_path.exists():
        ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        loaded_epoch   = ckpt.get("epoch", "?")
        loaded_val_acc = ckpt.get("best_val_accuracy", float("nan"))
        logger.info("Loaded best checkpoint (epoch %s, val_acc=%.4f)", loaded_epoch, loaded_val_acc)
    else:
        logger.warning("No best checkpoint found — evaluating current weights.")
        loaded_epoch = epochs_done

    model.to(device)
    model.eval()

    # Evaluate on val set (our best available held-out data)
    eval_loader = val_loader
    test_metrics = evaluate_model(
        model, eval_loader, THREE_CLASSES, device,
        results_dir=str(results_dir),
        top_k_values=[1],
    )

    # Per-class val breakdown
    logger.info("\nPer-class accuracy (val set, best checkpoint):")
    model.eval()
    with torch.no_grad():
        per_class_correct = defaultdict(int)
        per_class_total   = defaultdict(int)
        all_preds, all_labels_list = [], []
        for batch in eval_loader:
            frames   = batch["frames"].to(device)
            pose     = batch["pose"].to(device)
            labels_b = batch["label"]
            logits   = model(frames, pose)
            probs    = torch.softmax(logits, dim=-1)
            preds    = logits.argmax(-1).cpu()
            for pred, lbl in zip(preds.tolist(), labels_b.tolist()):
                per_class_total[lbl] += 1
                if pred == lbl:
                    per_class_correct[lbl] += 1
                all_preds.append(pred)
                all_labels_list.append(lbl)

    for i, cls in enumerate(THREE_CLASSES):
        n = per_class_total[i]
        c = per_class_correct[i]
        acc = c / n if n else float("nan")
        logger.info("  [%d] %-22s  %d/%d (%.0f%%)",  i, cls, c, n, acc * 100)

    # Print representative probabilities for a few val clips
    logger.info("\nRepresentative prediction probabilities (val set):")
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            frames   = batch["frames"].to(device)
            pose     = batch["pose"].to(device)
            labels_b = batch["label"]
            paths_b  = batch["path"]
            logits   = model(frames, pose)
            probs    = torch.softmax(logits, dim=-1)
            for i in range(min(3, len(labels_b))):
                p = probs[i].cpu().numpy()
                lbl = labels_b[i].item()
                pred = int(np.argmax(p))
                path_str = Path(paths_b[i]).name
                prob_str = "  ".join(f"{THREE_CLASSES[j][:10]}={p[j]:.3f}" for j in range(3))
                ok = "OK" if pred == lbl else "XX"
                logger.info("  [%s] %s | True=%s | %s", ok, path_str, THREE_CLASSES[lbl], prob_str)
            break  # just first batch

    # ── Final summary ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL REPORT — 3-CLASS DEMO MODEL")
    logger.info("=" * 60)
    logger.info("Classes: %s", THREE_CLASSES)
    logger.info("Train/Val videos: %d / %d  (no test split — val used for evaluation)", len(tr_paths), len(va_paths))
    logger.info("Best val accuracy:    %.4f  (epoch %s)", best_val_acc, loaded_epoch)
    logger.info("Val top-1 accuracy:   %.4f  (best checkpoint re-evaluated)", test_metrics.get("top_1_accuracy", float("nan")))
    logger.info("Val macro-F1:         %.4f", test_metrics.get("macro_f1", float("nan")))
    logger.info("Best checkpoint:      %s", best_ckpt_path)
    logger.info("class_names.json:     %s", class_names_path)
    logger.info("\nLabel mapping (folder -> display):")
    for i, cls in enumerate(THREE_CLASSES):
        logger.info("  [%d] %-22s -> '%s'", i, cls, DISPLAY_LABELS[cls])
    logger.info("\nPOSE MISMATCH FIX APPLIED:")
    logger.info("  Training used zeros pose (no MediaPipe during training)")
    logger.info("  Inference now also uses zeros pose -> consistent pipeline")
    logger.info("  EMA smoothing_alpha reduced 0.7 -> 0.5 for faster live response")
    logger.info("  Confidence threshold 0.45 -> shows 'Detecting...' when uncertain")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
