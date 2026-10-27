#!/usr/bin/env python3
"""
HAR 2-Class Demonstration Training Script
==========================================
Classes: SittingNormally [0]  vs  RaisingHand [1]

These two are maximally visually distinct (seated vs arm raised overhead),
giving the best chance of reliable demo performance with a small dataset.

ThumbsUp is dropped because the model consistently confuses it with RaisingHand
(both involve raised hand; with only 5 training videos each, they can't be separated).

Usage:
    python scripts/train_2class.py
"""
from __future__ import annotations

import json, logging, random, sys, time, re
from collections import defaultdict
from pathlib import Path

import cv2, numpy as np, torch, torch.nn as nn, yaml

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_BACKEND = _PROJECT / "backend"
sys.path.insert(0, str(_BACKEND))

from models.har_model import HARModel
from preprocessing.frame_processor import (
    read_video_frames, sample_frames_uniform,
    preprocess_frame, augment_frame, IMAGENET_MEAN, IMAGENET_STD,
)
from training.trainer import set_seed, save_checkpoint

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

TWO_CLASSES     = ["SittingNormally", "RaisingHand"]
DISPLAY_LABELS  = {"SittingNormally": "Sitting Normally", "RaisingHand": "Raising a Hand"}
VIDEO_EXTS      = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

# Split — same groups as before
TRAIN_GROUPS = {"g01","g02","g03","g04","g08"}
VAL_GROUPS   = {"g05","g06","g07"}
EXTRA_GROUPS = {"g09","g10","g11"}   # SittingNormally extras → train

# Hyperparams
MAX_EPOCHS  = 15
PATIENCE    = 5
BATCH_SIZE  = 4
CLIPS_PER   = 4     # more clips per video since dataset is tiny
NUM_FRAMES  = 16
FRAME_SIZE  = (224, 224)
LR          = 2e-4
WEIGHT_DECAY= 5e-4


class MultiClipDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, clips_per=1, augment=False):
        self.data = [(p,l) for p,l in zip(paths,labels) for _ in range(clips_per)]
        self.augment = augment

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            raw, _ = read_video_frames(path)
            frames = sample_frames_uniform(raw, NUM_FRAMES, temporal_jitter=self.augment)
            flip   = self.augment and random.random() < 0.5
            proc   = [augment_frame(
                          preprocess_frame(f, target_size=FRAME_SIZE,
                                           mean=IMAGENET_MEAN, std=IMAGENET_STD),
                          horizontal_flip=flip,
                          brightness_jitter=0.2 if self.augment else 0,
                          contrast_jitter=0.15 if self.augment else 0)
                      for f in frames]
            frames_t = torch.stack(proc)
        except Exception as e:
            logger.warning("Bad video %s: %s", path, e)
            frames_t = torch.zeros(NUM_FRAMES, 3, *FRAME_SIZE)
        return {"frames": frames_t,
                "pose":   torch.zeros(NUM_FRAMES, 99),
                "label":  torch.tensor(label, dtype=torch.long),
                "path":   path}


def discover_split(root):
    cls2idx = {c:i for i,c in enumerate(TWO_CLASSES)}
    tr_p, tr_l, va_p, va_l = [], [], [], []
    for cls in TWO_CLASSES:
        cls_dir = root / cls
        vids = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        for v in vids:
            cap = cv2.VideoCapture(str(v))
            ok  = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
            cap.release()
            if not ok: continue
            parts = v.stem.split("_")
            grp = next((p for p in parts if p.startswith("g") and p[1:].isdigit()), "unk")
            if grp in VAL_GROUPS:
                va_p.append(str(v)); va_l.append(cls2idx[cls])
            else:   # TRAIN_GROUPS + EXTRA_GROUPS + unknown
                tr_p.append(str(v)); tr_l.append(cls2idx[cls])
    return tr_p, tr_l, va_p, va_l


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot_n = 0
    per_cor, per_tot = defaultdict(int), defaultdict(int)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            frames = batch["frames"].to(device)
            pose   = batch["pose"].to(device)
            labels = batch["label"].to(device)
            if train: optimizer.zero_grad(set_to_none=True)
            logits = model(frames, pose)
            loss   = criterion(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            preds = logits.argmax(-1)
            bs = labels.size(0)
            tot_loss    += loss.item() * bs
            tot_correct += (preds == labels).sum().item()
            tot_n += bs
            for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                per_tot[l] += 1
                if p == l: per_cor[l] += 1
    return tot_loss/tot_n, tot_correct/tot_n, per_cor, per_tot


def main():
    set_seed(42)
    cfg_path       = _PROJECT / "backend" / "config.yaml"
    checkpoint_dir = _PROJECT / "checkpoints"
    results_dir    = _PROJECT / "results"
    dataset_root   = _PROJECT / "data" / "ucf101"
    cn_path        = _PROJECT / "data" / "class_names.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cpu")

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("HAR 2-CLASS DEMO TRAINING: %s", TWO_CLASSES)
    logger.info("=" * 60)
    tr_p, tr_l, va_p, va_l = discover_split(dataset_root)

    tr_cnt = defaultdict(int); va_cnt = defaultdict(int)
    for l in tr_l: tr_cnt[l] += 1
    for l in va_l: va_cnt[l] += 1

    logger.info("Split summary:")
    for i, cls in enumerate(TWO_CLASSES):
        logger.info("  [%d] %-22s  train=%d  val=%d", i, cls, tr_cnt[i], va_cnt[i])
    logger.info("Expanded train: %d clips (%d vids x%d)", len(tr_p)*CLIPS_PER, len(tr_p), CLIPS_PER)

    train_ds = MultiClipDataset(tr_p, tr_l, CLIPS_PER, augment=True)
    val_ds   = MultiClipDataset(va_p, va_l, 1,         augment=False)
    train_ldr = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    val_ldr   = torch.utils.data.DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model (fresh 2-class init) ────────────────────────────────────────
    cfg["model"]["num_classes"] = 2
    cfg["model"]["use_pose"]    = True
    cfg["data"]["sequence_length"] = NUM_FRAMES
    model = HARModel.from_config(cfg).to(device)

    # Freeze backbone initially
    for p in model.cnn.feature_extractor.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Backbone FROZEN | Trainable params: %s", f"{trainable:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    patience_cnt = 0
    history = {"train_loss":[], "train_accuracy":[], "val_loss":[], "val_accuracy":[]}

    logger.info("\nTraining: max=%d epochs | patience=%d | LR=%.1e | batch=%d",
                MAX_EPOCHS, PATIENCE, LR, BATCH_SIZE)
    logger.info("%-6s %-10s %-8s %-10s %-8s %-12s", "Epoch","TLoss","TAcc","VLoss","VAcc","Note")
    logger.info("-" * 60)

    unfreeze_done = False

    for epoch in range(1, MAX_EPOCHS+1):
        # Warmup for first 2 epochs
        if epoch <= 2:
            for pg in optimizer.param_groups:
                pg["lr"] = LR * epoch / 2

        # Unfreeze backbone after epoch 4 if val is plateauing
        if epoch == 5 and not unfreeze_done:
            for p in model.cnn.feature_extractor.parameters():
                p.requires_grad = True
            # Add CNN params to optimizer with low LR
            optimizer.add_param_group({
                "params": list(model.cnn.feature_extractor.parameters()),
                "lr": 5e-6, "weight_decay": 1e-4
            })
            unfreeze_done = True
            logger.info("  [Epoch 5] Backbone UNFROZEN — CNN LR=5e-6")

        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_ldr, optimizer, criterion, device, train=True)
        va_loss, va_acc, per_cor, per_tot = run_epoch(model, val_ldr, optimizer, criterion, device, train=False)
        if epoch > 2: scheduler.step()
        elapsed = time.time()-t0

        history["train_loss"].append(tr_loss)
        history["train_accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(va_acc)

        is_best = va_acc > best_val_acc + 1e-4
        if is_best:
            best_val_acc = va_acc
            patience_cnt = 0
            note = "[BEST]"
            save_checkpoint(model, optimizer, epoch,
                {"train_loss":tr_loss,"train_accuracy":tr_acc,
                 "val_loss":va_loss,"val_accuracy":va_acc,
                 "best_val_accuracy":best_val_acc},
                TWO_CLASSES, str(checkpoint_dir), is_best=True)
        else:
            patience_cnt += 1
            note = f"pat {patience_cnt}/{PATIENCE}"

        logger.info("%2d/%-3d  %8.4f  %6.1f%%  %8.4f  %6.1f%%  %s  (%.0fs)",
                    epoch, MAX_EPOCHS, tr_loss, tr_acc*100, va_loss, va_acc*100, note, elapsed)
        for i, cls in enumerate(TWO_CLASSES):
            n=per_tot[i]; c=per_cor[i]
            logger.info("        %-22s  %d/%d (%.0f%%)", cls, c, n, 100*c/n if n else 0)

        if patience_cnt >= PATIENCE:
            logger.info("Early stopping at epoch %d.", epoch)
            break

    logger.info("\nTraining complete. Best val_acc=%.4f", best_val_acc)

    # ── Update config and class_names ─────────────────────────────────────
    with open(cn_path, "w") as f:
        json.dump(TWO_CLASSES, f, indent=2)
    logger.info("class_names.json -> %s", cn_path)

    with open(cfg_path) as f:
        cfg_raw = f.read()
    cfg_raw = re.sub(r"(num_classes:\s*)\d+", r"\g<1>2", cfg_raw)
    with open(cfg_path, "w") as f:
        f.write(cfg_raw)
    logger.info("config.yaml: num_classes=2")

    # ── Final eval ────────────────────────────────────────────────────────
    best_ckpt = checkpoint_dir / "best_model.pth"
    ckpt = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION (best checkpoint: epoch %s)", ckpt.get("epoch","?"))
    logger.info("=" * 60)

    _, final_acc, final_cor, final_tot = run_epoch(
        model, val_ldr, optimizer, criterion, device, train=False)
    logger.info("Overall val accuracy: %.1f%%", final_acc*100)
    for i, cls in enumerate(TWO_CLASSES):
        n=final_tot[i]; c=final_cor[i]
        logger.info("  [%d] %-22s  %d/%d (%.0f%%)", i, cls, c, n, 100*c/n if n else 0)

    # Probabilities
    logger.info("\nProbabilities on all val videos:")
    model.eval()
    with torch.no_grad():
        for batch in val_ldr:
            frames  = batch["frames"]
            pose    = batch["pose"]
            labs    = batch["label"]
            paths   = batch["path"]
            logits  = model(frames, pose)
            probs   = torch.softmax(logits, dim=-1)
            for i in range(len(labs)):
                p   = probs[i].numpy()
                lbl = labs[i].item()
                pred= int(np.argmax(p))
                ok  = "OK" if pred==lbl else "XX"
                ps  = "  ".join(f"{TWO_CLASSES[j][:10]}={p[j]:.3f}" for j in range(2))
                logger.info("  [%s] %-35s | True=%-18s | %s",
                            ok, Path(paths[i]).name, TWO_CLASSES[lbl], ps)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL REPORT — 2-CLASS DEMO MODEL")
    logger.info("=" * 60)
    logger.info("Classes: %s", TWO_CLASSES)
    logger.info("Best val accuracy: %.4f  (epoch %s)", best_val_acc, ckpt.get("epoch","?"))
    logger.info("Label mapping:")
    for i, cls in enumerate(TWO_CLASSES):
        logger.info("  [%d] %s  ->  '%s'", i, cls, DISPLAY_LABELS[cls])
    logger.info("Inference config:")
    logger.info("  pose_zeros=True (matches training)")
    logger.info("  EMA alpha=0.5  confidence_threshold=0.40")
    logger.info("  Backend class_names.json: %s", TWO_CLASSES)
    logger.info("=" * 60)

    # Save history
    with open(results_dir / "training_history_2class.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
