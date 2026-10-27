"""
Training Pipeline
==================
Complete PyTorch training loop with:
  - GPU/CPU auto-detection
  - Mixed precision (AMP)
  - Configurable optimizer + scheduler
  - Early stopping
  - Checkpoint saving (best + periodic)
  - Resume from checkpoint
  - Reproducible seeding
  - Tensorboard-ready metrics
  - Per-epoch logging
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ── Seeding ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Optimizer / Scheduler ─────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["training"]
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("learning_rate", 1e-4)
    wd = train_cfg.get("weight_decay", 1e-4)

    if opt_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
) -> Optional[object]:
    train_cfg = cfg["training"]
    sched_name = train_cfg.get("scheduler", "cosine").lower()
    epochs = train_cfg.get("epochs", 50)

    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif sched_name == "step":
        return StepLR(
            optimizer,
            step_size=train_cfg.get("scheduler_step_size", 10),
            gamma=train_cfg.get("scheduler_gamma", 0.5),
        )
    elif sched_name == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="max", patience=train_cfg.get("scheduler_patience", 5),
            factor=0.5, verbose=True,
        )
    elif sched_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy."""
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    class_names: list[str],
    checkpoint_dir: str,
    is_best: bool = False,
    filename: Optional[str] = None,
) -> str:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    fname = filename or f"epoch_{epoch:03d}.pth"
    path = str(Path(checkpoint_dir) / fname)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        **metrics,
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved: %s", path)

    if is_best:
        best_path = str(Path(checkpoint_dir) / "best_model.pth")
        torch.save(payload, best_path)
        logger.info("Best model updated: %s", best_path)

    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %s)", path, ckpt.get("epoch", "?"))
    return ckpt


# ── One-epoch functions ───────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    cfg: dict,
    epoch: int,
) -> dict:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    clip_norm = cfg["training"].get("gradient_clip_norm", 1.0)

    for batch_idx, batch in enumerate(loader):
        frames = batch["frames"].to(device)      # (B, T, C, H, W)
        pose = batch["pose"].to(device)          # (B, T, 99)
        labels = batch["label"].to(device)       # (B,)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast("cuda"):
                logits = model(frames, pose)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames, pose)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_samples += bs

        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                "Epoch %d [%d/%d] loss=%.4f",
                epoch, batch_idx + 1, len(loader), loss.item()
            )

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return {"train_loss": avg_loss, "train_accuracy": avg_acc}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        pose = batch["pose"].to(device)
        labels = batch["label"].to(device)

        logits = model(frames, pose)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_samples += bs

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return {"val_loss": avg_loss, "val_accuracy": avg_acc}


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
    cfg: dict,
    results_dir: str = "../results",
) -> dict:
    """
    Main training loop.

    Returns:
        Training history dict with loss/accuracy curves.
    """
    train_cfg = cfg["training"]
    seed = cfg["project"].get("seed", 42)
    set_seed(seed)

    # Device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info("Training on: %s", device)
    model = model.to(device)

    # Loss
    label_smoothing = train_cfg.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer + Scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # AMP scaler
    use_amp = train_cfg.get("mixed_precision", True) and device_str == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("Mixed precision (AMP) enabled")

    # Resume
    start_epoch = 1
    resume_path = train_cfg.get("resume_from")
    if resume_path and Path(resume_path).exists():
        ckpt = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resuming from epoch %d", start_epoch)

    # Early stopping state
    best_val_acc = 0.0
    patience_counter = 0
    early_stop = train_cfg.get("early_stopping", True)
    patience = train_cfg.get("early_stopping_patience", 10)
    min_delta = train_cfg.get("early_stopping_min_delta", 0.001)

    checkpoint_dir = train_cfg.get("checkpoint_dir", "../checkpoints")
    save_every = train_cfg.get("save_every_n_epochs", 5)
    epochs = train_cfg.get("epochs", 50)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)

    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "learning_rates": [],
        "epoch_times": [],
    }

    logger.info("=" * 60)
    logger.info("Starting training: %d epochs, batch_size=%d",
                epochs, train_cfg.get("batch_size", 16))
    logger.info("=" * 60)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Warmup LR
        if epoch <= warmup_epochs:
            warmup_lr = train_cfg["learning_rate"] * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, cfg, epoch
        )

        # Validate
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        # Scheduler step
        if scheduler is not None and epoch > warmup_epochs:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["val_accuracy"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        # Record history
        history["train_loss"].append(train_metrics["train_loss"])
        history["train_accuracy"].append(train_metrics["train_accuracy"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_accuracy"].append(val_metrics["val_accuracy"])
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(epoch_time)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f | "
            "lr=%.2e | time=%.1fs",
            epoch, epochs,
            train_metrics["train_loss"], train_metrics["train_accuracy"],
            val_metrics["val_loss"], val_metrics["val_accuracy"],
            current_lr, epoch_time,
        )

        # Save periodic checkpoint
        if epoch % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {**train_metrics, **val_metrics, "best_val_accuracy": best_val_acc},
                class_names, checkpoint_dir,
                is_best=False,
            )

        # Save best checkpoint
        val_acc = val_metrics["val_accuracy"]
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch,
                {**train_metrics, **val_metrics, "best_val_accuracy": best_val_acc},
                class_names, checkpoint_dir,
                is_best=True,
            )
            logger.info("  [+] New best val_acc: %.4f", best_val_acc)
        else:
            patience_counter += 1

        # Early stopping
        if early_stop and patience_counter >= patience:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d)", epoch, patience
            )
            break

    # Save training history
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    history_path = str(Path(results_dir) / "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved -> %s", history_path)

    logger.info("Training complete. Best val_accuracy: %.4f", best_val_acc)
    return history
