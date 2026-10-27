#!/usr/bin/env python3
"""
Held-Out Subject-Independent Evaluation Protocol — HAR Deep Vision
==================================================================
Provides a strict, leak-free evaluation by partitioning UCF-101 videos
by group ID (subject/recording environment):

Split Strategy:
  - Training Set:   Groups 1, 2, 3 (18 videos, 3 per class)
  - Held-Out Test:  Group 4 (6 videos, 1 per class — completely unseen)

This ensures zero overlap between training and test sets, strictly
preventing data leakage and measuring true out-of-sample generalization.

Also supports Leave-One-Group-Out (LOGO) 4-Fold Cross Validation for
statistical robustness across the entire 24-video sample.

Usage:
  python scripts/evaluate_heldout.py [--mode split|kfold] [--epochs 35]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from models.har_model import HARModel
from preprocessing.frame_processor import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    augment_frame,
    preprocess_frame,
    read_video_frames,
    sample_frames_uniform,
)
from training.evaluator import plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def extract_group_id(video_path: str) -> str:
    """Extract group identifier (e.g. 'g01', 'g02', 'g03', 'g04') from UCF101 filename."""
    match = re.search(r"_(g\d{2})_", Path(video_path).name)
    if match:
        return match.group(1)
    return "g00"


class GroupVideoDataset(torch.utils.data.Dataset):
    """Augmented dataset from a list of video paths."""

    def __init__(
        self,
        video_paths: list[str],
        labels: list[int],
        class_names: list[str],
        num_frames: int = 16,
        frame_size: tuple[int, int] = (224, 224),
        clips_per_video: int = 15,
        augment: bool = True,
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment
        self.samples = [
            (p, l)
            for p, l in zip(video_paths, labels)
            for _ in range(clips_per_video)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            raw, _ = read_video_frames(path, max_frames=200)
        except Exception:
            raw = [np.zeros((224, 224, 3), dtype=np.uint8)]

        frames = sample_frames_uniform(raw, self.num_frames, temporal_jitter=self.augment)
        do_flip = self.augment and (np.random.random() < 0.5)
        processed = []
        for f in frames:
            t = preprocess_frame(
                f,
                target_size=self.frame_size,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            )
            if self.augment:
                t = augment_frame(
                    t,
                    horizontal_flip=do_flip,
                    brightness_jitter=0.25,
                    contrast_jitter=0.25,
                )
            processed.append(t)

        return {
            "frames": torch.stack(processed),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


class DeterministicEvalDataset(torch.utils.data.Dataset):
    """Deterministic multi-clip dataset for clean inference evaluation."""

    def __init__(
        self,
        video_paths: list[str],
        labels: list[int],
        num_frames: int = 16,
        frame_size: tuple[int, int] = (224, 224),
        clips_per_video: int = 5,
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.samples = [
            (p, l)
            for p, l in zip(video_paths, labels)
            for _ in range(clips_per_video)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            raw, _ = read_video_frames(path, max_frames=200)
        except Exception:
            raw = [np.zeros((224, 224, 3), dtype=np.uint8)]

        frames = sample_frames_uniform(raw, self.num_frames, temporal_jitter=False)
        processed = [
            preprocess_frame(
                f,
                target_size=self.frame_size,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            )
            for f in frames
        ]
        return {
            "frames": torch.stack(processed),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


def discover_all_videos(dataset_dir: Path, selected_classes: list[str] | None = None):
    class_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    if selected_classes:
        class_dirs = [d for d in class_dirs if d.name in selected_classes]

    class_names = [d.name for d in class_dirs]
    video_paths = []
    labels = []
    for idx, cd in enumerate(class_dirs):
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            for vf in sorted(cd.glob(f"*{ext}")):
                video_paths.append(str(vf))
                labels.append(idx)
    return video_paths, labels, class_names


def train_and_eval_split(
    train_paths: list[str],
    train_labels: list[int],
    test_paths: list[str],
    test_labels: list[int],
    class_names: list[str],
    cfg: dict,
    epochs: int = 35,
    lr: float = 0.001,
    clips_per_video: int = 15,
) -> dict:
    """Train on train_paths, evaluate strictly on test_paths."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_frames = cfg["data"].get("sequence_length", 16)
    frame_size = tuple(cfg["data"].get("frame_size", [224, 224]))

    # Datasets
    train_ds = GroupVideoDataset(
        train_paths,
        train_labels,
        class_names,
        num_frames=num_frames,
        frame_size=frame_size,
        clips_per_video=clips_per_video,
        augment=True,
    )
    test_ds = DeterministicEvalDataset(
        test_paths,
        test_labels,
        num_frames=num_frames,
        frame_size=frame_size,
        clips_per_video=5,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=6, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=6, shuffle=False, num_workers=0
    )

    # Model
    model_cfg = dict(cfg)
    model_cfg["model"]["num_classes"] = len(class_names)
    model_cfg["model"]["use_pose"] = False
    model = HARModel.from_config(model_cfg)

    # Freeze CNN backbone
    for param in model.cnn.feature_extractor.parameters():
        param.requires_grad = False

    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - 3), eta_min=1e-5
    )

    logger.info("Training clean split: %d train clips, %d test clips over %d epochs",
                len(train_ds), len(test_ds), epochs)

    # Training loop
    model.train()
    for ep in range(1, epochs + 1):
        tot_loss, tot_corr, tot_n = 0.0, 0, 0
        for batch in train_loader:
            frames = batch["frames"].to(device)
            target = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(frames, None)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            bs = target.size(0)
            tot_loss += loss.item() * bs
            tot_corr += (logits.argmax(-1) == target).sum().item()
            tot_n += bs

        if ep > 3:
            scheduler.step()

        if ep % 10 == 0 or ep == epochs:
            logger.info("  Split Epoch %2d/%2d | loss=%.4f acc=%.4f",
                        ep, epochs, tot_loss / tot_n, tot_corr / tot_n)

    # Evaluation on held-out test data
    model.eval()
    all_preds, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            frames = batch["frames"].to(device)
            logits = model(frames, None)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())
            all_paths.extend(batch["path"])

    # Video-level majority vote
    video_preds = defaultdict(list)
    video_labels = {}
    for p, pred, lbl in zip(all_paths, all_preds, all_labels):
        video_preds[p].append(pred)
        video_labels[p] = lbl

    final_preds, final_targets = [], []
    for p in video_preds:
        maj = Counter(video_preds[p]).most_common(1)[0][0]
        final_preds.append(maj)
        final_targets.append(video_labels[p])

    acc = accuracy_score(final_targets, final_preds)
    p_macro = precision_score(final_targets, final_preds, average="macro", zero_division=0)
    r_macro = recall_score(final_targets, final_preds, average="macro", zero_division=0)
    f1_macro = f1_score(final_targets, final_preds, average="macro", zero_division=0)
    f1_wt = f1_score(final_targets, final_preds, average="weighted", zero_division=0)

    return {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_wt),
        "num_test_videos": len(final_targets),
        "true_labels": [int(x) for x in final_targets],
        "pred_labels": [int(x) for x in final_preds],
        "test_paths": list(video_preds.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description="Held-out evaluation protocol")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs for clean split")
    parser.add_argument("--mode", choices=["split", "kfold"], default="split",
                        help="'split' (hold out group 4) or 'kfold' (4-fold Leave-One-Group-Out CV)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    with open(root / "backend" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    dataset_path = root / "data" / "ucf101"
    selected = cfg["data"].get("selected_classes") or None
    video_paths, labels, class_names = discover_all_videos(dataset_path, selected)

    # Tag each video with its UCF101 group ID
    video_groups = [extract_group_id(p) for p in video_paths]
    unique_groups = sorted(list(set(video_groups)))
    logger.info("Found %d videos across %d classes and %d groups: %s",
                len(video_paths), len(class_names), len(unique_groups), unique_groups)

    if args.mode == "split":
        # Standard benchmark: Train on g01, g02, g03 (18 videos) -> Test strictly on g04 (6 videos)
        held_out_group = "g04"
        train_p, train_l, test_p, test_l = [], [], [], []
        for p, l, g in zip(video_paths, labels, video_groups):
            if g == held_out_group:
                test_p.append(p)
                test_l.append(l)
            else:
                train_p.append(p)
                train_l.append(l)

        logger.info("=" * 65)
        logger.info("HELD-OUT EVALUATION (Subject-Independent Group Split)")
        logger.info("  Training:  Groups g01, g02, g03 (%d videos)", len(train_p))
        logger.info("  Held-out:  Group %s (%d videos — 1 per class)", held_out_group, len(test_p))
        logger.info("=" * 65)

        res = train_and_eval_split(
            train_p, train_l, test_p, test_l, class_names, cfg,
            epochs=args.epochs, clips_per_video=15,
        )

        logger.info("=" * 65)
        logger.info("HELD-OUT TEST RESULTS (Genuinely Unseen Source Videos):")
        logger.info("  Top-1 Accuracy:     %.4f (%d/%d correct)",
                    res["accuracy"], int(res["accuracy"] * res["num_test_videos"]), res["num_test_videos"])
        logger.info("  Macro Precision:    %.4f", res["precision_macro"])
        logger.info("  Macro Recall:       %.4f", res["recall_macro"])
        logger.info("  Macro F1-Score:     %.4f", res["f1_macro"])
        logger.info("=" * 65)

        rep = classification_report(
            res["true_labels"], res["pred_labels"],
            target_names=class_names, zero_division=0,
        )
        logger.info("\n%s", rep)

        # Save held-out metrics
        out_metrics = {
            "evaluation_type": "HELD_OUT_SUBJECT_INDEPENDENT",
            "protocol": "Train on groups g01-g03 (18 videos), Test on group g04 (6 videos)",
            "top_1_accuracy": round(res["accuracy"], 4),
            "macro_precision": round(res["precision_macro"], 4),
            "macro_recall": round(res["recall_macro"], 4),
            "macro_f1": round(res["f1_macro"], 4),
            "weighted_f1": round(res["f1_weighted"], 4),
            "num_test_samples": res["num_test_videos"],
            "classes_evaluated": class_names,
            "data_leakage_status": "NONE - Zero overlap in source videos or actor groups",
            "limitation_note": (
                "Evaluated on local 24-video UCF-101 subset. "
                "With only 3 training videos per class, small sample size causes high variance. "
                "For comprehensive evaluation, run scripts/colab_training.ipynb on full UCF-101."
            ),
        }
        with open(results_dir / "held_out_metrics.json", "w") as f:
            json.dump(out_metrics, f, indent=2)

        # Plot held out confusion matrix
        plot_confusion_matrix(
            np.array(res["true_labels"]),
            np.array(res["pred_labels"]),
            class_names,
            save_path=str(results_dir / "held_out_confusion_matrix.png"),
        )
        logger.info("Saved held-out metrics and confusion matrix to %s", results_dir)

    elif args.mode == "kfold":
        # 4-Fold Leave-One-Group-Out Cross Validation
        logger.info("Running 4-Fold Leave-One-Group-Out Cross Validation...")
        all_true, all_pred = [], []
        fold_accs = []
        for fold, test_g in enumerate(unique_groups, 1):
            train_p, train_l, test_p, test_l = [], [], [], []
            for p, l, g in zip(video_paths, labels, video_groups):
                if g == test_g:
                    test_p.append(p)
                    test_l.append(l)
                else:
                    train_p.append(p)
                    train_l.append(l)

            logger.info("--- Fold %d/4: Testing on Group %s (%d videos) ---", fold, test_g, len(test_p))
            f_res = train_and_eval_split(
                train_p, train_l, test_p, test_l, class_names, cfg,
                epochs=args.epochs, clips_per_video=12,
            )
            fold_accs.append(f_res["accuracy"])
            all_true.extend(f_res["true_labels"])
            all_pred.extend(f_res["pred_labels"])
            logger.info("Fold %d Accuracy: %.4f", fold, f_res["accuracy"])

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        logger.info("=" * 65)
        logger.info("4-Fold LOGO Cross-Validation Mean Accuracy: %.4f (+/- %.4f)", mean_acc, std_acc)
        logger.info("=" * 65)

        kfold_metrics = {
            "evaluation_type": "LEAVE_ONE_GROUP_OUT_4_FOLD_CV",
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "fold_accuracies": [round(x, 4) for x in fold_accs],
            "total_evaluated_videos": len(all_true),
        }
        with open(results_dir / "kfold_metrics.json", "w") as f:
            json.dump(kfold_metrics, f, indent=2)


if __name__ == "__main__":
    main()
