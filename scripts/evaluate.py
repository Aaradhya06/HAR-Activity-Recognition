#!/usr/bin/env python3
"""
Evaluation Script — HAR Deep Vision
====================================
Runs standalone evaluation against a test dataset split, computes
metrics (Top-1/Top-5 accuracy, precision, recall, F1, latency, FPS),
generates confusion matrix and classification report.

Usage:
    python scripts/evaluate.py [--config backend/config.yaml] [--checkpoint checkpoints/best_model.pth]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add backend to sys.path
_BACKEND = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

import yaml
import torch

from models.har_model import HARModel
from preprocessing.dataset_loader import build_dataloaders
from training.evaluator import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR Model Evaluation")
    p.add_argument("--config", default="backend/config.yaml", help="Path to config.yaml")
    p.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pth)")
    p.add_argument("--results-dir", default=None, help="Output directory for metrics and plots")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Configuration file not found: %s", cfg_path)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info("Evaluation device: %s", device)

    results_dir = args.results_dir or cfg.get("evaluation", {}).get("results_dir", "results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Dataloaders
    try:
        _, _, test_loader, class_names = build_dataloaders(
            cfg, class_names_output_path=cfg.get("data", {}).get("class_names_file")
        )
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    cfg["model"]["num_classes"] = len(class_names)
    model = HARModel.from_config(cfg)

    ckpt_path = args.checkpoint or str(Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints")) / "best_model.pth")
    if Path(ckpt_path).exists():
        logger.info("Loading checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        logger.warning("Checkpoint '%s' not found. Evaluating with random/initialized weights.", ckpt_path)

    model = model.to(device)

    metrics = evaluate_model(
        model,
        test_loader,
        class_names,
        device=device,
        results_dir=results_dir,
        top_k_values=cfg.get("evaluation", {}).get("top_k", [1, 5]),
    )

    logger.info("Evaluation complete! Results stored at: %s", results_dir)
    print("\n" + "=" * 50)
    print("HAR EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Top-1 Accuracy: {metrics.get('top_1_accuracy', 0):.4f}")
    print(f"Top-5 Accuracy: {metrics.get('top_5_accuracy', 0):.4f}")
    print(f"Macro F1-Score: {metrics.get('macro_f1', 0):.4f}")
    print(f"Latency (ms/sample): {metrics.get('avg_inference_ms_per_sample', 0):.2f} ms")
    print(f"Estimated FPS: {metrics.get('fps_estimate', 0):.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
