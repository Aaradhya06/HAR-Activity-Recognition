#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass
"""
HAR Diagnosis Script
====================
Checks:
1. Checkpoint metadata (epoch, val_acc, class_names, classifier shape)
2. Training training script uses pose=zeros (mismatch with live inference)
3. Group split integrity -- what videos go where
4. Runs inference on held-out test videos and prints probabilities per class
5. Checks EMA smoothing direction and pose impact
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_BACKEND = _PROJECT / "backend"
sys.path.insert(0, str(_BACKEND))

from models.har_model import HARModel
from preprocessing.frame_processor import (
    read_video_frames, sample_frames_uniform, preprocess_frame,
    IMAGENET_MEAN, IMAGENET_STD,
)

CANONICAL_CLASSES = [
    "SittingNormally", "RaisingHand", "ThumbsUp", "ScratchingHead", "LookingAway",
]
DISPLAY = {
    "SittingNormally": "Sitting Normally",
    "RaisingHand": "Raising a Hand",
    "ThumbsUp": "Thumbs Up",
    "ScratchingHead": "Scratching Head",
    "LookingAway": "Looking Away",
}

SEP = "=" * 65

def frames_to_tensor(frames, num_frames=16, frame_size=(224, 224)):
    """BGR frames -> (1, T, C, H, W) normalized tensor."""
    total = len(frames)
    if total <= num_frames:
        indices = list(range(total)) + [total-1]*(num_frames-total)
    else:
        indices = np.linspace(0, total-1, num_frames, dtype=int).tolist()
    sampled = [frames[i] for i in indices]
    processed = []
    for f in sampled:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (frame_size[1], frame_size[0]), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).permute(2,0,1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
        std  = torch.tensor(IMAGENET_STD).view(3,1,1)
        t = (t - mean) / std
        processed.append(t)
    return torch.stack(processed).unsqueeze(0)  # (1, T, C, H, W)


def load_model_from_ckpt(ckpt_path, cfg):
    cfg["model"]["num_classes"] = 5
    cfg["data"]["sequence_length"] = 16
    model = HARModel.from_config(cfg)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def run_inference_on_video(model, video_path, use_pose_zeros=True, num_frames=16):
    """Returns (class_idx, probs) for a single video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < 300:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None, None
    frames_tensor = frames_to_tensor(frames, num_frames)
    pose_tensor = torch.zeros(1, num_frames, 99) if use_pose_zeros else None
    with torch.no_grad():
        logits = model(frames_tensor, pose_tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def main():
    print(SEP)
    print("  HAR DIAGNOSIS REPORT")
    print(SEP)

    # Load config
    cfg_path = _PROJECT / "backend" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(f"\n[CONFIG] Loaded from: {cfg_path}")
    print(f"  checkpoint_path: {cfg['inference']['checkpoint_path']}")
    print(f"  num_classes (config): {cfg['model']['num_classes']}")
    print(f"  selected_classes: {cfg['data']['selected_classes']}")
    print(f"  sequence_length: {cfg['data']['sequence_length']}")
    print(f"  frame_size: {cfg['data']['frame_size']}")

    # Load class_names.json
    cn_path = _PROJECT / "data" / "class_names.json"
    with open(cn_path) as f:
        class_names_json = json.load(f)
    print(f"\n[class_names.json] {class_names_json}")

    # Checkpoint inspection
    print(f"\n{SEP}")
    print("  CHECKPOINT INSPECTION")
    print(SEP)
    ckpt_files = {
        "best_model.pth": _PROJECT / "checkpoints" / "best_model.pth",
        "epoch_016.pth":  _PROJECT / "checkpoints" / "epoch_016.pth",
        "epoch_009.pth":  _PROJECT / "checkpoints" / "epoch_009.pth",
        "epoch_005.pth":  _PROJECT / "checkpoints" / "epoch_005.pth",
    }
    for name, path in ckpt_files.items():
        if not path.exists():
            print(f"  {name}: MISSING")
            continue
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", {})
        # Find classifier output shape
        cls_key = [k for k in sd if "classifier" in k and "weight" in k]
        cls_shape = sd[cls_key[-1]].shape if cls_key else "?"
        print(f"\n  {name}:")
        print(f"    Size:              {path.stat().st_size / 1e6:.2f} MB")
        print(f"    Epoch:             {ckpt.get('epoch', '?')}")
        print(f"    best_val_accuracy: {ckpt.get('best_val_accuracy', '?')}")
        print(f"    class_names:       {ckpt.get('class_names', '?')}")
        print(f"    classifier out:    {cls_shape}")
        print(f"    train_accuracy:    {ckpt.get('train_accuracy', '?')}")
        print(f"    val_accuracy:      {ckpt.get('val_accuracy', '?')}")

    # Dataset audit
    print(f"\n{SEP}")
    print("  DATASET AUDIT")
    print(SEP)
    dataset_root = _PROJECT / "data" / "ucf101"
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    train_groups = {"g01","g02","g03","g04","g05"}
    val_groups   = {"g06","g07"}
    test_groups  = {"g08"}

    all_videos = {}  # class -> list of (path, group)
    for cls in CANONICAL_CLASSES:
        cls_dir = dataset_root / cls
        vids = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        all_videos[cls] = []
        for v in vids:
            parts = v.stem.split("_")
            group = next((p for p in parts if p.startswith("g") and p[1:].isdigit()), "?")
            split = "train" if group in train_groups else ("val" if group in val_groups else ("test" if group in test_groups else "unk"))
            all_videos[cls].append((v, group, split))

    print(f"\n{'Class':<22} {'Total':>5} {'Train':>5} {'Val':>5} {'Test':>5}")
    print("-" * 50)
    for cls in CANONICAL_CLASSES:
        vids = all_videos[cls]
        tr = sum(1 for _, _, s in vids if s == "train")
        va = sum(1 for _, _, s in vids if s == "val")
        te = sum(1 for _, _, s in vids if s == "test")
        uk = sum(1 for _, _, s in vids if s == "unk")
        extra = f"  (+{uk} unassigned)" if uk else ""
        print(f"  {cls:<20} {len(vids):>5} {tr:>5} {va:>5} {te:>5}{extra}")

    # Model inference on labeled videos
    print(f"\n{SEP}")
    print("  PER-VIDEO INFERENCE TEST (best_model.pth)")
    print(SEP)
    best_ckpt = _PROJECT / "checkpoints" / "best_model.pth"
    if not best_ckpt.exists():
        print("  best_model.pth MISSING — cannot test!")
        return

    model, ckpt = load_model_from_ckpt(best_ckpt, cfg)
    ckpt_classes = ckpt.get("class_names", CANONICAL_CLASSES)
    print(f"\n  Checkpoint class_names: {ckpt_classes}")
    print(f"  num_classes in model output: {model.num_classes}")
    print()

    # Check pose branch behavior
    print(f"  use_pose in config: {cfg['model']['use_pose']}")
    print(f"  pose_encoder type: {type(model.pose_encoder).__name__}")
    print()

    print(f"  {'Video':<40} {'TrueClass':<22} {'PredClass':<22} {'TopConf':>8} {'Correct':>8}")
    print("  " + "-" * 108)

    correct = 0
    total = 0
    prob_matrix = defaultdict(list)  # true_class -> list of prob vectors

    for cls_idx, cls in enumerate(CANONICAL_CLASSES):
        for v, group, split in all_videos[cls]:
            pred_idx, probs = run_inference_on_video(model, v, use_pose_zeros=True)
            if pred_idx is None:
                print(f"  {v.name:<40} FAILED TO READ")
                continue
            pred_cls = ckpt_classes[pred_idx] if pred_idx < len(ckpt_classes) else f"idx{pred_idx}"
            ok = "OK" if pred_cls == cls else "XX"
            if pred_cls == cls:
                correct += 1
            total += 1
            prob_matrix[cls].append(probs)
            conf = float(probs[pred_idx])
            print(f"  {v.name:<40} {cls:<22} {pred_cls:<22} {conf:>8.3f} {ok:>8}")

    print(f"\n  Overall accuracy on all videos: {correct}/{total} = {100*correct/total:.1f}%")

    # Per-class average probabilities
    print(f"\n{SEP}")
    print("  AVERAGE PROBABILITIES PER TRUE CLASS")
    print(SEP)
    header = "  " + f"{'TrueClass':<22}" + "".join(f"  {c[:10]:>12}" for c in ckpt_classes)
    print(header)
    print("  " + "-" * (22 + len(ckpt_classes)*14))
    for cls in CANONICAL_CLASSES:
        if not prob_matrix[cls]:
            print(f"  {cls:<22}  (no data)")
            continue
        avg = np.mean(prob_matrix[cls], axis=0)
        row = f"  {cls:<22}" + "".join(f"  {avg[i]:>12.3f}" for i in range(len(ckpt_classes)))
        print(row)

    # EMA smoothing analysis
    print(f"\n{SEP}")
    print("  EMA SMOOTHING ANALYSIS (alpha=0.7)")
    print(SEP)
    alpha = cfg["inference"].get("smoothing_alpha", 0.7)
    print(f"  smoothing_alpha: {alpha}")
    print(f"  EMA formula: smoothed = {alpha} * prev + {1-alpha} * current")
    print(f"  Effect: very slow to adapt — only {(1-alpha)*100:.0f}% weight on new frame")
    print(f"  Time to reach 90% of new signal: ~{int(np.ceil(np.log(0.1)/np.log(alpha)))} inference windows")
    print(f"  At 16-frame windows: each window ≈ 0.5s at 30fps → ~{int(np.ceil(np.log(0.1)/np.log(alpha)))*0.5:.1f}s to respond")

    # Verdict
    print(f"\n{SEP}")
    print("  DIAGNOSIS SUMMARY")
    print(SEP)
    if correct/total < 0.5:
        print(f"  ⚠ Model accuracy on own dataset is LOW ({100*correct/total:.1f}%)")
        print(f"    → Model may be predicting random or biased classes")
    else:
        print(f"  ✓ Model accuracy on dataset videos: {100*correct/total:.1f}%")
    print()
    sys.stdout.flush()

if __name__ == "__main__":
    main()
