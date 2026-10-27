#!/usr/bin/env python3
"""
HAR Backbone-Unfreeze Fine-Tune Script
=======================================
Resumes from the best 3-class checkpoint (epoch 2, val=44.4%) and continues
training with the CNN backbone unfrozen using differential learning rates.

Why: frozen MobileNetV3 features (ImageNet) don't discriminate hand gestures
(RaisingHand, ThumbsUp) from sitting postures well. Unfreezing allows adaptation.

Strategy:
- CNN backbone LR: 5e-6 (very low, preserve pretrained features)
- Transformer / Pose / Fusion / Classifier LR: 5e-5 (10x faster)
- Weighted CrossEntropy: RaisingHand and ThumbsUp get 1.8x weight (they failed)
- Max 10 epochs, early stopping patience=5
- Same group split as train_3class.py (train=g01-g04+g08, val=g05-g07)

Usage:
    python scripts/finetune_unfreeze.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

_HERE  = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_BACKEND = _PROJECT / "backend"
sys.path.insert(0, str(_BACKEND))

from models.har_model import HARModel
from preprocessing.frame_processor import (
    read_video_frames, sample_frames_uniform,
    preprocess_frame, augment_frame, IMAGENET_MEAN, IMAGENET_STD,
)
from training.trainer import set_seed, save_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

THREE_CLASSES = ["SittingNormally", "RaisingHand", "ThumbsUp"]
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

TRAIN_GROUPS = {"g01", "g02", "g03", "g04", "g08"}
VAL_GROUPS   = {"g05", "g06", "g07"}
EXTRA_GROUPS = {"g09", "g10", "g11"}

# Hyperparams
CNN_LR     = 5e-6   # very low for frozen-then-unfrozen backbone
HEAD_LR    = 5e-5   # 10x faster for classification heads
MAX_EPOCHS = 10
PATIENCE   = 5
BATCH_SIZE = 4
CLIPS_PER  = 3
NUM_FRAMES = 16
FRAME_SIZE = (224, 224)
# Class weights: up-weight the classes the model failed on
# SittingNormally=1.0 (it already works), RaisingHand=2.0, ThumbsUp=1.5
CLASS_WEIGHTS = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float)


import random

class MultiClipDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, labels, num_frames=16, frame_size=(224,224),
                 clips_per_video=1, augment=False):
        self.augment = augment
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.expanded = [(p, l) for p, l in zip(video_paths, labels)
                         for _ in range(clips_per_video)]

    def __len__(self): return len(self.expanded)

    def __getitem__(self, idx):
        path, label = self.expanded[idx]
        try:
            raw_frames, _ = read_video_frames(path)
            frames = sample_frames_uniform(raw_frames, self.num_frames,
                                          temporal_jitter=self.augment)
            do_flip = self.augment and random.random() < 0.5
            processed = []
            for f in frames:
                t = preprocess_frame(f, target_size=self.frame_size,
                                     mean=IMAGENET_MEAN, std=IMAGENET_STD)
                if self.augment:
                    t = augment_frame(t, horizontal_flip=do_flip,
                                      brightness_jitter=0.2, contrast_jitter=0.15)
                processed.append(t)
            frames_tensor = torch.stack(processed)
        except Exception as e:
            logger.warning("Bad video %s: %s", path, e)
            frames_tensor = torch.zeros(self.num_frames, 3, *self.frame_size)
        return {
            "frames": frames_tensor,
            "pose":   torch.zeros(self.num_frames, 99),
            "label":  torch.tensor(label, dtype=torch.long),
            "path":   path,
        }


def discover_split(root):
    paths, labels, splits = [], [], []
    class_to_idx = {c: i for i, c in enumerate(THREE_CLASSES)}
    for cls in THREE_CLASSES:
        cls_dir = root / cls
        vids = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS)
        for v in vids:
            cap = cv2.VideoCapture(str(v))
            ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
            cap.release()
            if not ok: continue
            parts = v.stem.split("_")
            group = next((p for p in parts if p.startswith("g") and p[1:].isdigit()), "unk")
            if group in TRAIN_GROUPS or group in EXTRA_GROUPS:
                split = "train"
            elif group in VAL_GROUPS:
                split = "val"
            else:
                split = "train"
            paths.append(str(v))
            labels.append(class_to_idx[cls])
            splits.append(split)
    tr_p = [p for p,s in zip(paths,splits) if s=="train"]
    tr_l = [l for l,s in zip(labels,splits) if s=="train"]
    va_p = [p for p,s in zip(paths,splits) if s=="val"]
    va_l = [l for l,s in zip(labels,splits) if s=="val"]
    return tr_p, tr_l, va_p, va_l


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tot_loss = tot_correct = tot_n = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        pose   = batch["pose"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(frames, pose)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = labels.size(0)
        tot_loss += loss.item() * bs
        tot_correct += (logits.argmax(-1) == labels).sum().item()
        tot_n += bs
    return tot_loss/tot_n, tot_correct/tot_n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_correct = tot_n = 0
    per_correct = defaultdict(int)
    per_total   = defaultdict(int)
    for batch in loader:
        frames = batch["frames"].to(device)
        pose   = batch["pose"].to(device)
        labels = batch["label"]
        logits = model(frames, pose)
        loss = criterion(logits, labels.to(device))
        preds = logits.argmax(-1).cpu()
        bs = labels.size(0)
        tot_loss += loss.item() * bs
        tot_correct += (preds == labels).sum().item()
        tot_n += bs
        for p, l in zip(preds.tolist(), labels.tolist()):
            per_total[l] += 1
            if p == l: per_correct[l] += 1
    return tot_loss/tot_n, tot_correct/tot_n, per_correct, per_total


def main():
    set_seed(42)

    cfg_path = _PROJECT / "backend" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint_dir = _PROJECT / "checkpoints"
    results_dir    = _PROJECT / "results"
    dataset_root   = _PROJECT / "data" / "ucf101"
    class_names_path = _PROJECT / "data" / "class_names.json"
    device = torch.device("cpu")

    # ── Load best checkpoint from train_3class.py ─────────────────────────
    source_ckpt = checkpoint_dir / "best_model.pth"
    if not source_ckpt.exists():
        logger.error("No best_model.pth found. Run train_3class.py first.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("HAR BACKBONE-UNFREEZE FINE-TUNE")
    logger.info("=" * 60)
    logger.info("Resuming from: %s", source_ckpt)

    ckpt = torch.load(str(source_ckpt), map_location="cpu", weights_only=False)
    logger.info("  Checkpoint: epoch=%s  val_acc=%.4f  classes=%s",
                ckpt.get("epoch","?"), ckpt.get("best_val_accuracy",0),
                ckpt.get("class_names","?"))

    # Check it's a 3-class checkpoint
    sd = ckpt["model_state_dict"]
    cls_keys = [k for k in sd if "classifier" in k and "weight" in k]
    cls_out_dim = sd[cls_keys[-1]].shape[0] if cls_keys else -1
    if cls_out_dim != 3:
        logger.error("Checkpoint has %d classes, expected 3. Aborting.", cls_out_dim)
        sys.exit(1)
    logger.info("  Classifier output dim: %d (correct)", cls_out_dim)

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info("\nDataset split...")
    tr_paths, tr_labels, va_paths, va_labels = discover_split(dataset_root)

    tr_counts = defaultdict(int)
    va_counts = defaultdict(int)
    for l in tr_labels: tr_counts[l] += 1
    for l in va_labels: va_counts[l] += 1
    logger.info("  Train: %d vids  %s", len(tr_paths),
                {THREE_CLASSES[i]: tr_counts[i] for i in range(3)})
    logger.info("  Val:   %d vids  %s", len(va_paths),
                {THREE_CLASSES[i]: va_counts[i] for i in range(3)})

    train_ds = MultiClipDataset(tr_paths, tr_labels, NUM_FRAMES, FRAME_SIZE,
                                CLIPS_PER, augment=True)
    val_ds   = MultiClipDataset(va_paths, va_labels, NUM_FRAMES, FRAME_SIZE,
                                1, augment=False)
    logger.info("  Expanded train: %d clips (x%d)", len(train_ds), CLIPS_PER)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model: load weights, then UNFREEZE backbone ───────────────────────
    cfg["model"]["num_classes"] = 3
    cfg["model"]["use_pose"] = True
    cfg["data"]["sequence_length"] = NUM_FRAMES
    model = HARModel.from_config(cfg)
    model.load_state_dict(sd)
    model = model.to(device)

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True
    logger.info("\nAll parameters UNFROZEN for fine-tuning")

    # Differential parameter groups
    cnn_params  = list(model.cnn.feature_extractor.parameters())
    head_params = [p for p in model.parameters()
                   if not any(p is cp for cp in cnn_params)]
    optimizer = torch.optim.AdamW([
        {"params": cnn_params,  "lr": CNN_LR,  "weight_decay": 1e-4},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": 5e-4},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-7)

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device), label_smoothing=0.0)
    logger.info("Loss: weighted CE (SittingNormally=1.0, RaisingHand=2.0, ThumbsUp=1.5)")
    logger.info("LR:   CNN=%.1e  Heads=%.1e  (differential)", CNN_LR, HEAD_LR)
    logger.info("Max epochs=%d  Patience=%d", MAX_EPOCHS, PATIENCE)

    best_val_acc  = ckpt.get("best_val_accuracy", 0.0)
    best_prev_acc = best_val_acc
    patience_cnt  = 0
    history = {"train_loss":[], "train_accuracy":[],
               "val_loss":[], "val_accuracy":[], "epoch_times":[]}

    logger.info("\n%-6s %-10s %-8s %-10s %-8s %-6s" %
                ("Epoch","TLoss","TAcc","VLoss","VAcc","Note"))
    logger.info("-" * 55)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, per_cor, per_tot = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(va_acc)
        history["epoch_times"].append(elapsed)

        is_best = va_acc > best_val_acc + 1e-4
        note = ""
        if is_best:
            best_val_acc = va_acc
            patience_cnt = 0
            note = "[BEST]"
            save_checkpoint(
                model, optimizer, epoch,
                {"train_loss": tr_loss, "train_accuracy": tr_acc,
                 "val_loss": va_loss, "val_accuracy": va_acc,
                 "best_val_accuracy": best_val_acc},
                THREE_CLASSES,
                str(checkpoint_dir),
                is_best=True,
            )
        else:
            patience_cnt += 1
            note = f"pat {patience_cnt}/{PATIENCE}"

        logger.info("%2d/%-3d  %8.4f  %6.1f%%  %8.4f  %6.1f%%  %s",
                    epoch, MAX_EPOCHS, tr_loss, tr_acc*100,
                    va_loss, va_acc*100, note)

        # Per-class breakdown each epoch
        for i, cls in enumerate(THREE_CLASSES):
            n = per_tot[i]; c = per_cor[i]
            logger.info("        %-20s  %d/%d (%.0f%%)", cls, c, n, 100*c/n if n else 0)

        if patience_cnt >= PATIENCE:
            logger.info("Early stopping at epoch %d.", epoch)
            break

    logger.info("\nFine-tune complete. Best val_acc=%.4f (started=%.4f)",
                best_val_acc, best_prev_acc)

    # ── Save history ──────────────────────────────────────────────────────
    hist_path = results_dir / "training_history_finetune.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Final per-class eval on val set ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL PER-CLASS EVALUATION (best checkpoint)")
    logger.info("=" * 60)

    best_ckpt_path = checkpoint_dir / "best_model.pth"
    final_ckpt = torch.load(str(best_ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(final_ckpt["model_state_dict"])
    model.eval()

    _, final_va, final_per_cor, final_per_tot = val_epoch(
        model, val_loader, criterion, device)

    logger.info("Overall val accuracy: %.1f%%  (%.4f)", final_va*100, final_va)
    for i, cls in enumerate(THREE_CLASSES):
        n = final_per_tot[i]; c = final_per_cor[i]
        logger.info("  [%d] %-22s  %d/%d (%.0f%%)", i, cls, c, n, 100*c/n if n else 0)

    # Show probabilities for all val videos
    logger.info("\nProbabilities on all val videos:")
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            frames = batch["frames"]
            pose   = batch["pose"]
            labs   = batch["label"]
            paths  = batch["path"]
            logits = model(frames, pose)
            probs  = torch.softmax(logits, dim=-1)
            for i in range(len(labs)):
                p   = probs[i].numpy()
                lbl = labs[i].item()
                pred = int(np.argmax(p))
                ok  = "OK" if pred == lbl else "XX"
                ps  = "  ".join(f"{THREE_CLASSES[j][:10]}={p[j]:.3f}" for j in range(3))
                logger.info("  [%s] %-35s | True=%-18s | %s",
                            ok, Path(paths[i]).name, THREE_CLASSES[lbl], ps)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Best checkpoint: %s", best_ckpt_path)
    logger.info("  epoch=%s  val_acc=%.4f",
                final_ckpt.get("epoch","?"), final_ckpt.get("best_val_accuracy",0))
    logger.info("Classes: %s", THREE_CLASSES)
    logger.info("Inference fixes already applied:")
    logger.info("  - pose zeros (matching training)")
    logger.info("  - EMA alpha=0.5 (faster response)")
    logger.info("  - confidence_threshold=0.33")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
