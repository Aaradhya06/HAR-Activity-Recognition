#!/usr/bin/env python3
"""
HAR Continuation Fine-Tuning -- 20 More Epochs (Credit-Efficient)
=================================================================
Resumes from the best checkpoint (epoch 5, val_acc=30%) and continues with:
  - UNFROZEN backbone (all 2.28M params trainable) -- richer features
  - Very low LR (5e-5 -> 1e-6 cosine) to avoid catastrophic forgetting
  - Aggressive augmentation (more temporal jitter, brightness/contrast)
  - Higher patience (10) so it doesn't quit too early
  - Stops at epoch 10 to print mid-report, then continues to epoch 20
  - Saves best checkpoint to checkpoints/best_model.pth (same path backend uses)

Strategy vs the previous 10-epoch frozen run:
  - Frozen backbone couldn't learn task-specific low-level features
  - Very low LR on unfreeze prevents destroying ImageNet weights
  - More augmentation clips (5 vs 3) expands the effective training set

Usage (from har-project directory):
    python scripts/continue_training.py
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# -- Path setup ----------------------------------------------------------------
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
from training.evaluator import evaluate_model
from training.trainer import set_seed, save_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

DISPLAY_LABELS = {
    "SittingNormally": "Sitting Normally",
    "RaisingHand": "Raising a Hand",
    "ThumbsUp": "Thumbs Up",
    "ScratchingHead": "Scratching Head",
    "LookingAway": "Looking Away",
}

CANONICAL_CLASSES = [
    "SittingNormally",
    "RaisingHand",
    "ThumbsUp",
    "ScratchingHead",
    "LookingAway",
]

# -- Hyperparameters -----------------------------------------------------------
TOTAL_NEW_EPOCHS   = 20
MID_REPORT_EPOCH   = 10
FINETUNE_LR        = 5e-5
FINETUNE_LR_MIN    = 1e-6
PATIENCE           = 10
CLIPS_PER_VIDEO    = 5
NUM_FRAMES         = 16
BATCH_SIZE         = 4
LABEL_SMOOTHING    = 0.1
WEIGHT_DECAY       = 1e-4
SEED               = 42


# -- Dataset utilities ---------------------------------------------------------

def discover_and_validate_videos(root, class_names):
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


def group_split_by_recording(paths, labels, train_groups, val_groups, test_groups):
    tr_p, tr_l = [], []
    va_p, va_l = [], []
    te_p, te_l = [], []
    for path, label in zip(paths, labels):
        stem = Path(path).stem
        parts = stem.split("_")
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
            logger.warning("Unknown group for %s, placing in train", path)
            tr_p.append(path); tr_l.append(label)
    return (tr_p, tr_l), (va_p, va_l), (te_p, te_l)


class MultiClipDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, labels, class_names,
                 num_frames=16, frame_size=(224, 224),
                 clips_per_video=1, augment=False):
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

    def __len__(self):
        return len(self.expanded_paths)

    def __getitem__(self, idx):
        path = self.expanded_paths[idx]
        label = self.expanded_labels[idx]
        try:
            raw_frames, _ = read_video_frames(path)
            frames = sample_frames_uniform(
                raw_frames, self.num_frames,
                temporal_jitter=self.augment,
            )
            do_flip = self.augment and (random.random() < 0.5)
            processed = []
            for f in frames:
                t = preprocess_frame(f, target_size=self.frame_size,
                                     mean=IMAGENET_MEAN, std=IMAGENET_STD)
                if self.augment:
                    t = augment_frame(
                        t,
                        horizontal_flip=do_flip,
                        brightness_jitter=0.20,
                        contrast_jitter=0.15,
                    )
                processed.append(t)
            frames_tensor = torch.stack(processed)
        except Exception as e:
            logger.warning("Corrupt video %s: %s -- using zeros", path, e)
            frames_tensor = torch.zeros(self.num_frames, 3, *self.frame_size)
        return {
            "frames": frames_tensor,
            "pose": torch.zeros(self.num_frames, 99),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


# -- Training helpers ----------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0):
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
def val_epoch(model, loader, criterion, device):
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


def print_mid_report(history, start_epoch, mid_epoch):
    """Print clean mid-training summary to stdout."""
    print("\n" + "=" * 65)
    print(f"  MID-TRAINING REPORT -- Epochs {start_epoch} to {mid_epoch}")
    print("=" * 65)
    print(f"  {'Epoch':>5}  {'TrainLoss':>9}  {'TrainAcc':>8}  "
          f"{'ValLoss':>8}  {'ValAcc':>7}  {'LR':>9}")
    print("  " + "-" * 57)
    best_va = max(history["val_accuracy"])
    for i, (tl, ta, vl, va, lr) in enumerate(zip(
        history["train_loss"], history["train_accuracy"],
        history["val_loss"],   history["val_accuracy"],
        history["learning_rates"],
    )):
        ep = start_epoch + i
        marker = "  <-- BEST" if va == best_va else ""
        print(f"  {ep:>5}  {tl:>9.4f}  {ta:>7.1%}  {vl:>8.4f}  "
              f"{va:>6.1%}  {lr:>9.2e}{marker}")
    print("=" * 65)
    best_ep = history["val_accuracy"].index(best_va) + start_epoch
    print(f"  Best val_acc so far: {best_va:.1%}  at epoch {best_ep}")
    print("=" * 65 + "\n")
    sys.stdout.flush()


# -- Main ----------------------------------------------------------------------

def main():
    set_seed(SEED)

    config_path = _PROJECT / "backend" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset_root   = (_PROJECT / "data" / "ucf101").resolve()
    checkpoint_dir = (_PROJECT / "checkpoints").resolve()
    results_dir    = (_PROJECT / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    logger.info("Device: CPU (Intel Core Ultra 5 125H)")

    # -- Dataset ---------------------------------------------------------------
    logger.info("Discovering and validating videos...")
    all_paths, all_labels = discover_and_validate_videos(dataset_root, CANONICAL_CLASSES)
    logger.info("Total valid videos: %d", len(all_paths))

    train_groups = {"g01", "g02", "g03", "g04", "g05"}
    val_groups   = {"g06", "g07"}
    test_groups  = {"g08"}

    (tr_paths, tr_labels), (va_paths, va_labels), (te_paths, te_labels) = \
        group_split_by_recording(all_paths, all_labels, train_groups, val_groups, test_groups)

    logger.info("Split -> Train: %d | Val: %d | Test: %d",
                len(tr_paths), len(va_paths), len(te_paths))

    train_ds = MultiClipDataset(
        tr_paths, tr_labels, CANONICAL_CLASSES,
        num_frames=NUM_FRAMES, clips_per_video=CLIPS_PER_VIDEO, augment=True,
    )
    val_ds = MultiClipDataset(
        va_paths, va_labels, CANONICAL_CLASSES,
        num_frames=NUM_FRAMES, clips_per_video=1, augment=False,
    )
    test_ds = MultiClipDataset(
        te_paths, te_labels, CANONICAL_CLASSES,
        num_frames=NUM_FRAMES, clips_per_video=1, augment=False,
    )

    logger.info("Train clips: %d (x%d augmentation) | Val: %d | Test: %d",
                len(train_ds), CLIPS_PER_VIDEO, len(val_ds), len(test_ds))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # -- Load model from best checkpoint ---------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("LOADING BEST CHECKPOINT (epoch 5, val_acc=30.0%%)")
    logger.info("=" * 60)

    cfg["model"]["num_classes"] = 5
    cfg["data"]["sequence_length"] = NUM_FRAMES
    model = HARModel.from_config(cfg)

    best_ckpt_path = checkpoint_dir / "best_model.pth"
    if not best_ckpt_path.exists():
        logger.error("No checkpoint found at %s -- aborting.", best_ckpt_path)
        sys.exit(1)

    ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    prev_epoch   = ckpt.get("epoch", 5)
    prev_val_acc = ckpt.get("best_val_accuracy", 0.30)
    logger.info("Checkpoint loaded: epoch=%d, prev_val_acc=%.4f", prev_epoch, prev_val_acc)

    # -- Unfreeze all parameters -----------------------------------------------
    for param in model.parameters():
        param.requires_grad = True
    trainable = model.count_parameters(trainable_only=True)
    logger.info("Backbone UNFROZEN -- %s params now trainable (was 1.36M frozen)", f"{trainable:,}")

    model = model.to(device)

    # -- Differential LR: backbone 10x lower than heads -----------------------
    backbone_params = list(model.cnn.feature_extractor.parameters())
    backbone_ids    = set(id(p) for p in backbone_params)
    head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": FINETUNE_LR / 10},  # 5e-6
        {"params": head_params,     "lr": FINETUNE_LR},       # 5e-5
    ], weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_NEW_EPOCHS, eta_min=FINETUNE_LR_MIN,
    )

    logger.info("\n" + "=" * 60)
    logger.info("FINE-TUNING -- %d epochs (global epochs %d to %d)",
                TOTAL_NEW_EPOCHS, prev_epoch + 1, prev_epoch + TOTAL_NEW_EPOCHS)
    logger.info("  Backbone LR: %.2e  |  Head LR: %.2e  -> %.2e cosine",
                FINETUNE_LR / 10, FINETUNE_LR, FINETUNE_LR_MIN)
    logger.info("  Clips/video: %d (was 3)  |  Patience: %d (was 5)",
                CLIPS_PER_VIDEO, PATIENCE)
    logger.info("  Mid-report: after epoch %d | Final: epoch %d",
                prev_epoch + MID_REPORT_EPOCH, prev_epoch + TOTAL_NEW_EPOCHS)
    logger.info("=" * 60)

    best_val_acc    = prev_val_acc
    patience_counter = 0
    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss":   [], "val_accuracy":   [],
        "learning_rates": [], "epoch_times": [],
    }

    train_start = time.time()

    for local_ep in range(1, TOTAL_NEW_EPOCHS + 1):
        global_ep = prev_epoch + local_ep
        t0 = time.time()

        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        head_lr    = optimizer.param_groups[1]["lr"]
        epoch_time = time.time() - t0
        val_acc    = val_m["val_accuracy"]

        history["train_loss"].append(train_m["train_loss"])
        history["train_accuracy"].append(train_m["train_accuracy"])
        history["val_loss"].append(val_m["val_loss"])
        history["val_accuracy"].append(val_m["val_accuracy"])
        history["learning_rates"].append(head_lr)
        history["epoch_times"].append(epoch_time)

        is_best = val_acc > best_val_acc + 1e-4
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, global_ep,
                {**train_m, **val_m, "best_val_accuracy": best_val_acc},
                CANONICAL_CLASSES,
                str(checkpoint_dir),
                is_best=True,
            )
            logger.info(
                "Epoch %2d (global %2d) | TLoss=%.4f TAcc=%.1f%% | "
                "VLoss=%.4f VAcc=%.1f%% | LR=%.2e | %.1fs  [NEW BEST]",
                local_ep, global_ep,
                train_m["train_loss"], train_m["train_accuracy"] * 100,
                val_m["val_loss"],     val_m["val_accuracy"] * 100,
                head_lr, epoch_time,
            )
        else:
            patience_counter += 1
            logger.info(
                "Epoch %2d (global %2d) | TLoss=%.4f TAcc=%.1f%% | "
                "VLoss=%.4f VAcc=%.1f%% | LR=%.2e | %.1fs  (pat %d/%d)",
                local_ep, global_ep,
                train_m["train_loss"], train_m["train_accuracy"] * 100,
                val_m["val_loss"],     val_m["val_accuracy"] * 100,
                head_lr, epoch_time, patience_counter, PATIENCE,
            )

        gap = train_m["train_accuracy"] - val_m["val_accuracy"]
        if gap > 0.30:
            logger.warning("  Overfit warning: train/val gap = %.1f%%", gap * 100)

        # Mid-report at epoch 10
        if local_ep == MID_REPORT_EPOCH:
            print_mid_report(history, prev_epoch + 1, global_ep)
            mid_path = results_dir / "continuation_history_mid.json"
            with open(mid_path, "w") as f:
                json.dump(history, f, indent=2)
            logger.info("Mid-history saved -> %s", mid_path)

        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info("Early stopping at local ep %d / global ep %d.",
                        local_ep, global_ep)
            break

    total_time  = time.time() - train_start
    epochs_done = len(history["train_loss"])
    logger.info("\nFine-tuning done in %.1f min | %d new epochs | best val_acc=%.4f",
                total_time / 60, epochs_done, best_val_acc)

    # -- Save full history -----------------------------------------------------
    hist_path = results_dir / "continuation_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("History -> %s", hist_path)

    # -- Final test evaluation -------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("HELD-OUT TEST EVALUATION")
    logger.info("=" * 60)

    ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    loaded_epoch   = ckpt.get("epoch", "?")
    loaded_val_acc = ckpt.get("best_val_accuracy", float("nan"))
    logger.info("Best checkpoint: epoch %s, val_acc=%.4f", loaded_epoch, loaded_val_acc)

    model.to(device)
    model.eval()

    test_metrics = evaluate_model(
        model, test_loader, CANONICAL_CLASSES, device,
        results_dir=str(results_dir),
        top_k_values=[1],
    )

    # Per-class breakdown
    logger.info("\nPer-class test accuracy:")
    model.eval()
    with torch.no_grad():
        per_class_correct = defaultdict(int)
        per_class_total   = defaultdict(int)
        for batch in test_loader:
            frames   = batch["frames"].to(device)
            pose     = batch["pose"].to(device)
            labels_b = batch["label"]
            logits   = model(frames, pose)
            preds    = logits.argmax(-1).cpu()
            for pred, lbl in zip(preds.tolist(), labels_b.tolist()):
                per_class_total[lbl] += 1
                if pred == lbl:
                    per_class_correct[lbl] += 1

    for i, cls in enumerate(CANONICAL_CLASSES):
        n = per_class_total[i]
        c = per_class_correct[i]
        acc = c / n if n else float("nan")
        logger.info("  [%d] %-22s  %d/%d  (%.1f%%)", i, cls, c, n, acc * 100)

    # Final summary
    print("\n" + "=" * 65)
    print("  CONTINUATION FINAL SUMMARY")
    print("=" * 65)
    print(f"  Previous best val_acc:   {prev_val_acc:.1%}  (epoch {prev_epoch}, frozen backbone)")
    print(f"  New best val_acc:        {best_val_acc:.1%}  (epoch {loaded_epoch}, unfrozen)")
    delta = best_val_acc - prev_val_acc
    sign  = "+" if delta >= 0 else "-"
    print(f"  Improvement:             {sign}{abs(delta):.1%}")
    print(f"  Test Top-1 Accuracy:     {test_metrics.get('top_1_accuracy', float('nan')):.1%}")
    print(f"  Test Macro-F1:           {test_metrics.get('macro_f1', float('nan')):.4f}")
    print(f"  Best checkpoint:         {best_ckpt_path}")
    print("=" * 65)
    print("\nBackend will load best_model.pth automatically on startup.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
