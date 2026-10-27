#!/usr/bin/env python3
"""
Training Script — HAR Deep Vision
====================================
Usage:
    python train.py [--config path/to/config.yaml] [--classes cls1 cls2 ...]

Example (UCF101 subset):
    python train.py --config backend/config.yaml

Environment:
    PYTHONHOME must NOT be set when using the embedded Python distribution.
    Use run_backend.bat or set PYTHONHOME="" before running.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure backend is importable
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

import yaml
import torch

from models.har_model import HARModel
from preprocessing.dataset_loader import build_dataloaders
from training.trainer import train, set_seed
from training.evaluator import evaluate_model, plot_training_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR Training Script")
    p.add_argument("--config", default="backend/config.yaml", help="Path to config.yaml")
    p.add_argument("--classes", nargs="*", help="Override selected_classes list")
    p.add_argument("--epochs", type=int, help="Override training epochs")
    p.add_argument("--batch-size", type=int, help="Override batch size")
    p.add_argument("--lr", type=float, help="Override learning rate")
    p.add_argument("--backbone", choices=["mobilenet_v3_small", "efficientnet_b0", "resnet18"],
                   help="Override backbone")
    p.add_argument("--temporal", choices=["transformer", "lstm", "gru"],
                   help="Override temporal module type")
    p.add_argument("--no-pose", action="store_true", help="Disable pose branch")
    p.add_argument("--eval-only", action="store_true", help="Skip training, run evaluation only")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Normalize relative paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    path_keys = {"dataset_path", "processed_path", "class_names_file", "checkpoint_dir", "results_dir", "checkpoint_path"}
    for section in ["data", "training", "evaluation", "inference"]:
        if section in cfg:
            for k, v in list(cfg[section].items()):
                if k in path_keys and isinstance(v, str):
                    clean_rel = v[3:] if v.startswith("../") else v
                    cfg[section][k] = str((project_root / clean_rel).resolve())

    # CLI overrides
    if args.classes:
        cfg["data"]["selected_classes"] = args.classes
        logger.info("Using classes: %s", args.classes)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.temporal:
        cfg["model"]["temporal_type"] = args.temporal
    if args.no_pose:
        cfg["model"]["use_pose"] = False

    seed = cfg["project"].get("seed", 42)
    set_seed(seed)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info("Device: %s", device)
    if device_str == "cuda":
        logger.info("GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Build data loaders
    results_dir = cfg["evaluation"]["results_dir"]
    class_names_path = cfg["data"]["class_names_file"]

    logger.info("Loading dataset from: %s", cfg["data"]["dataset_path"])
    try:
        train_loader, val_loader, test_loader, class_names = build_dataloaders(
            cfg, class_names_output_path=class_names_path
        )
    except FileNotFoundError as e:
        logger.error(
            "\n\n%s\n\n"
            "Dataset not found. Please:\n"
            "  1. Download UCF101 from: https://www.crcv.ucf.edu/data/UCF101.php\n"
            "  2. Extract to: %s\n"
            "  3. Re-run this script.\n"
            "Or adjust 'data.dataset_path' in config.yaml.\n",
            e, cfg["data"]["dataset_path"]
        )
        sys.exit(1)
    except ValueError as e:
        logger.error("Dataset error: %s", e)
        sys.exit(1)

    logger.info("Classes (%d): %s", len(class_names), class_names)

    # Build model
    cfg["model"]["num_classes"] = len(class_names)
    model = HARModel.from_config(cfg)
    total_params = model.count_parameters()
    trainable_params = model.count_parameters(trainable_only=True)
    logger.info("Model: %s | Total params: %s | Trainable: %s",
                cfg["model"]["name"],
                f"{total_params:,}", f"{trainable_params:,}")

    if not args.eval_only:
        # Training
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING")
        logger.info("=" * 60)
        history = train(model, train_loader, val_loader, class_names, cfg, results_dir)

        # Plot curves
        plot_training_curves(history, save_dir=results_dir)
        logger.info("Training complete.")

    # Evaluation
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    # Load best checkpoint for evaluation
    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / "best_model.pth"
    if ckpt_path.exists():
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded best checkpoint from %s", ckpt_path)
    else:
        logger.warning("No checkpoint found. Evaluating with current (untrained) weights.")

    model = model.to(device)
    metrics = evaluate_model(
        model, test_loader, class_names, device,
        results_dir=results_dir,
        top_k_values=cfg["evaluation"].get("top_k", [1, 5]),
    )

    logger.info("\nFinal Evaluation Results:")
    logger.info("  Top-1 Accuracy: %.4f", metrics.get("top_1_accuracy", float("nan")))
    logger.info("  Macro F1:       %.4f", metrics.get("macro_f1", float("nan")))
    logger.info("  FPS (estimate): %.1f", metrics.get("fps_estimate", 0))
    logger.info("  Parameters:     %s", f"{metrics.get('total_parameters', 0):,}")
    logger.info("\nResults saved to: %s", results_dir)


if __name__ == "__main__":
    main()
