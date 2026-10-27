"""
Evaluation Pipeline
====================
Computes all metrics discussed in the project report:
  - Accuracy (top-1, top-5)
  - Precision, Recall, F1-score (per-class + macro/weighted)
  - Confusion matrix
  - Inference latency (ms per clip)
  - FPS
  - Model parameter count
  - Per-class breakdown

Saves:
  results/
    metrics.json
    classification_report.txt
    confusion_matrix.png
    training_curves.png
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    results_dir: str = "../results",
    top_k_values: list[int] = [1, 5],
) -> dict:
    """
    Full evaluation: runs model on test set and computes all metrics.

    Returns:
        metrics dict (also saved as JSON).
    """
    model.eval()
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    latencies_ms: list[float] = []

    logger.info("Running evaluation on %d batches...", len(test_loader))

    for batch in test_loader:
        frames = batch["frames"].to(device)
        pose = batch["pose"].to(device)
        labels = batch["label"]

        t0 = time.perf_counter()
        logits = model(frames, pose)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs)
        bs = labels.size(0)
        latencies_ms.append(elapsed_ms / bs)  # per-sample latency

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Top-k accuracy ────────────────────────────────────────────────────
    top_k_acc = {}
    for k in top_k_values:
        if k == 1:
            top_k_acc[f"top_{k}_accuracy"] = float(accuracy_score(all_labels, all_preds))
        else:
            # Compute top-k from stored probabilities
            all_probs_arr = np.array(all_probs)  # (N, C)
            correct_k = 0
            for i, label in enumerate(all_labels):
                top_k_indices = np.argsort(all_probs_arr[i])[::-1][:k]
                if label in top_k_indices:
                    correct_k += 1
            top_k_acc[f"top_{k}_accuracy"] = correct_k / len(all_labels)

    # ── Per-class metrics ─────────────────────────────────────────────────
    present_labels = sorted(list(set(all_labels.tolist())))
    present_names = [class_names[i] if i < len(class_names) else f"Class_{i}"
                     for i in present_labels]

    macro_prec = float(precision_score(all_labels, all_preds, average="macro", zero_division=0))
    macro_rec  = float(recall_score(all_labels, all_preds, average="macro", zero_division=0))
    macro_f1   = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))

    # ── Latency / FPS ─────────────────────────────────────────────────────
    avg_latency_ms = float(np.mean(latencies_ms))
    fps_estimate = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    # ── Parameters ────────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = {
        **top_k_acc,
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "avg_inference_ms_per_sample": round(avg_latency_ms, 2),
        "fps_estimate": round(fps_estimate, 2),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "num_test_samples": len(all_labels),
        "num_classes_evaluated": len(present_labels),
    }

    logger.info("Evaluation results:")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    # ── Save metrics.json ─────────────────────────────────────────────────
    metrics_path = str(Path(results_dir) / "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved -> %s", metrics_path)

    # ── Classification report ─────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )
    report_path = str(Path(results_dir) / "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Human Activity Recognition -- Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    logger.info("Classification report saved -> %s", report_path)

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm_path = plot_confusion_matrix(
        all_labels, all_preds, present_names,
        save_path=str(Path(results_dir) / "confusion_matrix.png"),
    )
    logger.info("Confusion matrix saved -> %s", cm_path)

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str,
    normalize: bool = True,
) -> str:
    """Plot and save a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm_norm = cm

    n = len(class_names)
    fig_size = max(8, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f" if normalize else "d",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax, vmin=0, vmax=1 if normalize else None,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 10 - n // 5))
    plt.yticks(rotation=0, fontsize=max(6, 10 - n // 5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_training_curves(
    history: dict,
    save_dir: str = "../results",
) -> list[str]:
    """Plot and save training/validation loss and accuracy curves."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = []

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        logger.warning("No training history to plot.")
        return []

    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#4C72B0", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="#DD8452",
            linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = str(Path(save_dir) / "training_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    # Accuracy curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, [x * 100 for x in history["train_accuracy"]],
            label="Train Accuracy", color="#4C72B0", linewidth=2)
    ax.plot(epochs, [x * 100 for x in history["val_accuracy"]],
            label="Val Accuracy", color="#DD8452", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = str(Path(save_dir) / "validation_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    logger.info("Training curves saved -> %s", save_dir)
    return saved_paths


def generate_model_comparison_table(
    results: dict[str, dict],
    save_path: str,
) -> str:
    """
    Generate a comparison table across model variants (ablation study).

    Args:
        results: {model_name: metrics_dict}
        save_path: Output text file path.
    """
    header = f"{'Model':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'FPS':>8} {'Params':>12}"
    sep = "-" * len(header)
    lines = [
        "HAR Model Comparison -- Ablation Study",
        "=" * len(header),
        header,
        sep,
    ]
    for name, m in results.items():
        acc = f"{m.get('top_1_accuracy', float('nan')):.4f}"
        prec = f"{m.get('macro_precision', float('nan')):.4f}"
        rec = f"{m.get('macro_recall', float('nan')):.4f}"
        f1 = f"{m.get('macro_f1', float('nan')):.4f}"
        fps = f"{m.get('fps_estimate', float('nan')):.1f}"
        params = str(m.get('total_parameters', 'N/A'))
        lines.append(f"{name:<35} {acc:>10} {prec:>10} {rec:>10} {f1:>8} {fps:>8} {params:>12}")
    lines.append(sep)
    lines.append("\nNote: 'Not evaluated' entries indicate experiments not yet run.")

    content = "\n".join(lines)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Comparison table saved -> %s", save_path)
    return content
