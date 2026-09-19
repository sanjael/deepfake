"""
Unified Academic Evaluation Script for Deepfake Detection
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

import sys
import os
import json
import argparse
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset_140k import create_dataloaders
from src.models.spatial_branch import EfficientNetB4SpatialBranch
from src.models.frequency_branch import FrequencyBranch
from src.models.fgsa_net import FGSANet
from src.evaluation.metrics import calculate_binary_metrics, print_metrics_summary


def build_model(config: dict) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    arch = model_cfg.get("architecture", "efficientnet_b4").lower()

    if arch in ["efficientnet_b4", "spatial"]:
        return EfficientNetB4SpatialBranch(
            pretrained=False,
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            num_classes=model_cfg.get("num_classes", 1),
            feature_dim=model_cfg.get("feature_dim", 1792)
        )
    elif arch in ["frequency_branch", "frequency"]:
        return FrequencyBranch(
            in_channels=model_cfg.get("in_channels", 12),
            feature_dim=model_cfg.get("feature_dim", 1792),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            num_classes=model_cfg.get("num_classes", 1)
        )
    elif arch in ["fgsa_net", "proposed"]:
        return FGSANet(
            feature_dim=model_cfg.get("feature_dim", 1792),
            pretrained_spatial=False,
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            num_classes=model_cfg.get("num_classes", 1)
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def plot_evaluation_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: dict,
    exp_name: str,
    output_path: Path
):
    """Plots ROC Curve, PR Curve, and Confusion Matrix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, color='#2563eb', lw=2.5, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
    axes[0].legend(loc='lower right', frameon=True)
    axes[0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec, prec, color='#16a34a', lw=2.5, label=f"PR (AUC = {metrics['pr_auc']:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontweight='bold')
    axes[1].set_ylabel('Precision', fontweight='bold')
    axes[1].set_title('Precision-Recall Curve', fontweight='bold')
    axes[1].legend(loc='lower left', frameon=True)
    axes[1].grid(True, alpha=0.3)

    # 3. Confusion Matrix
    cm = np.array([
        [metrics["confusion_matrix"]["true_negatives"], metrics["confusion_matrix"]["false_positives"]],
        [metrics["confusion_matrix"]["false_negatives"], metrics["confusion_matrix"]["true_positives"]]
    ])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[2],
        xticklabels=['Pred Fake', 'Pred Real'],
        yticklabels=['Actual Fake', 'Actual Real']
    )
    axes[2].set_title(f'Confusion Matrix (Acc: {metrics["accuracy"]*100:.2f}%)', fontweight='bold')

    plt.suptitle(f"Evaluation Analysis: {exp_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Evaluation plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Deepfake Model on Held-out Test Set")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Sub-sample test set for fast evaluation")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"Academic Evaluation: {exp_name}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Device: {device}")
    print("=" * 80)

    # Build Dataloader
    print("\n[1/3] Loading Test Partition...")
    _, _, test_loader = create_dataloaders(
        config=cfg,
        batch_size=args.batch_size,
        num_workers=4,
        img_size=cfg.get("dataset", {}).get("img_size", 256),
        max_eval_samples=args.max_test_samples
    )
    print(f"  - Test Set: {len(test_loader.dataset):,} samples ({len(test_loader)} batches)")

    # Build & Load Model
    print("\n[2/3] Loading Model Weights...")
    model = build_model(cfg)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Run Inference
    print("\n[3/3] Running Inference on Held-Out Test Set...")
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().flatten()
            
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_labels.extend(labels)
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    metrics = calculate_binary_metrics(y_true, y_prob)
    print_metrics_summary(metrics, title=f"Test Set Evaluation: {exp_name}")

    # Save metrics JSON
    eval_dir = PROJECT_ROOT / "results" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    json_path = eval_dir / f"{exp_name}_test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test metrics saved to: {json_path}")

    # Plot Curves
    fig_path = PROJECT_ROOT / "results" / "figures" / f"{exp_name}_evaluation_curves.png"
    plot_evaluation_curves(y_true, y_prob, metrics, exp_name, fig_path)


if __name__ == "__main__":
    main()
