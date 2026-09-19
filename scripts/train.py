"""
Unified Training Script for Deepfake Detection Baselines and Proposed Architecture
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

import sys
import os
import argparse
import random
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset_140k import create_dataloaders
from src.models.spatial_branch import EfficientNetB4SpatialBranch
from src.models.frequency_branch import FrequencyBranch
from src.models.fgsa_net import FGSANet
from src.training.losses import get_loss_function
from src.training.trainer import DeepfakeTrainer


def set_seed(seed: int = 42):
    """Ensures deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(config: dict) -> nn.Module:
    """Builds model based on config."""
    model_cfg = config.get("model", {})
    arch = model_cfg.get("architecture", "efficientnet_b4").lower()

    if arch in ["efficientnet_b4", "spatial"]:
        print("--> Building Baseline 1: Spatial Branch (EfficientNet-B4)")
        return EfficientNetB4SpatialBranch(
            pretrained=model_cfg.get("pretrained", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            num_classes=model_cfg.get("num_classes", 1),
            feature_dim=model_cfg.get("feature_dim", 1792)
        )
    elif arch in ["frequency_branch", "frequency"]:
        print("--> Building Baseline 2: Frequency Branch (2D-DWT ConvNet)")
        return FrequencyBranch(
            in_channels=model_cfg.get("in_channels", 12),
            feature_dim=model_cfg.get("feature_dim", 1792),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            num_classes=model_cfg.get("num_classes", 1)
        )
    elif arch in ["fgsa_net", "proposed"]:
        print("--> Building Proposed Architecture: FGSA-Net (Spatial + Frequency + Attention Fusion)")
        return FGSANet(
            feature_dim=model_cfg.get("feature_dim", 1792),
            pretrained_spatial=model_cfg.get("pretrained_spatial", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            num_classes=model_cfg.get("num_classes", 1)
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Models")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Sub-sample train set for fast validation")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Sub-sample eval set for fast validation")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Overrides
    train_cfg = cfg.get("training", {})
    num_epochs = args.epochs or train_cfg.get("num_epochs", 15)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    learning_rate = args.lr or train_cfg.get("learning_rate", 0.0003)
    max_train_samples = args.max_train_samples or cfg.get("dataset", {}).get("max_train_samples")
    max_eval_samples = args.max_eval_samples or cfg.get("dataset", {}).get("max_eval_samples")

    # Set seed
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    print("=" * 80)
    print(f"PhD Experiment Run: {cfg.get('experiment', {}).get('name', 'experiment')}")
    print(f"Description: {cfg.get('experiment', {}).get('description', '')}")
    print(f"Config File: {config_path}")
    print("=" * 80)

    # DataLoaders
    print("\n[1/4] Constructing DataLoaders...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        config=cfg,
        batch_size=batch_size,
        num_workers=train_cfg.get("num_workers", 4),
        img_size=cfg.get("dataset", {}).get("img_size", 256),
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples
    )
    print(f"  - Train Loader: {len(train_loader)} batches ({len(train_loader.dataset):,} samples)")
    print(f"  - Valid Loader: {len(valid_loader)} batches ({len(valid_loader.dataset):,} samples)")
    print(f"  - Test Loader:  {len(test_loader)} batches ({len(test_loader.dataset):,} samples)")

    # Model
    print("\n[2/4] Instantiating Model...")
    model = build_model(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  - Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Loss & Optimizer
    print("\n[3/4] Initializing Loss, Optimizer, and Scheduler...")
    criterion = get_loss_function(train_cfg.get("loss", "bce"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=train_cfg.get("weight_decay", 1e-5)
    )

    scheduler_type = train_cfg.get("scheduler", "cosine").lower()
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    else:
        scheduler = None

    # Trainer
    print("\n[4/4] Launching Training Loop...")
    paths_cfg = cfg.get("paths", {})
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_name=cfg.get("experiment", {}).get("name", "experiment"),
        checkpoints_dir=paths_cfg.get("checkpoints_dir", "results/checkpoints"),
        logs_dir=paths_cfg.get("logs_dir", "results/logs"),
        use_amp=train_cfg.get("use_amp", True),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 5)
    )

    history = trainer.fit(num_epochs=num_epochs)
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
