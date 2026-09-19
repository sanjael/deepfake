"""
PhD Paper I: Robustness Evaluation Engine
Tests Spatial, Frequency, and Proposed FGSA-Net against real-world corruptions:
- JPEG Compression (Quality: 90, 70, 50, 30)
- Gaussian Blur (Kernel: 3x3, 5x5, 7x7)
- Gaussian Noise (Std: 0.02, 0.05, 0.10)
Author: Ramesh Prabhakaran R.
"""

import sys
import io
import json
import argparse
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.spatial_branch import EfficientNetB4SpatialBranch
from src.models.frequency_branch import FrequencyBranch
from src.models.fgsa_net import FGSANet
from src.datasets.transforms import get_validation_transforms
from src.evaluation.metrics import calculate_binary_metrics


class PerturbedDeepfakeDataset(Dataset):
    """Applies controlled image corruptions to test forensic robustness."""
    def __init__(self, samples, perturbation_type: str = "clean", level: float = 1.0, img_size: int = 256):
        self.samples = samples
        self.perturbation_type = perturbation_type
        self.level = level
        self.transform = get_validation_transforms(img_size=img_size)

    def __len__(self):
        return len(self.samples)

    def apply_perturbation(self, img: Image.Image) -> Image.Image:
        if self.perturbation_type == "clean":
            return img

        if self.perturbation_type == "jpeg":
            quality = int(self.level)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")

        if self.perturbation_type == "blur":
            radius = float(self.level)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

        if self.perturbation_type == "noise":
            np_img = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, float(self.level), np_img.shape)
            noisy_img = np.clip(np_img + noise, 0.0, 1.0) * 255.0
            return Image.fromarray(noisy_img.astype(np.uint8))

        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as raw:
            img = raw.convert("RGB")

        perturbed_img = self.apply_perturbation(img)
        tensor = self.transform(perturbed_img)
        return tensor, torch.tensor(label, dtype=torch.float32)


def load_model(arch: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if arch == "spatial":
        model = EfficientNetB4SpatialBranch(pretrained=False, dropout_rate=0.3, num_classes=1, feature_dim=1792)
    elif arch == "frequency":
        model = FrequencyBranch(in_channels=12, feature_dim=1792, dropout_rate=0.3, num_classes=1)
    elif arch == "fgsa_net":
        model = FGSANet(feature_dim=1792, pretrained_spatial=False, dropout_rate=0.3, num_classes=1)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def evaluate_model_on_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for tensors, labels in loader:
            tensors = tensors.to(device, non_blocking=True)
            logits = model(tensors)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())
    return calculate_binary_metrics(all_labels, all_probs)


def main():
    parser = argparse.ArgumentParser(description="Robustness Benchmark across Perturbations")
    parser.add_argument("--test-samples", type=int, default=2000, help="Number of test samples for fast benchmark")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("PhD Paper I: Robustness & Generalization Benchmark")
    print(f"Device: {device} | Test Samples per Condition: {args.test_samples:,}")
    print("=" * 85)

    # 1. Load Test Metadata
    test_csv = PROJECT_ROOT / "daaset" / "test.csv"
    images_dir = PROJECT_ROOT / "daaset" / "real_vs_fake" / "real-vs-fake"
    import pandas as pd
    import os

    df_test = pd.read_csv(test_csv)
    df_real = df_test[df_test['label'] == 1].head(args.test_samples // 2)
    df_fake = df_test[df_test['label'] == 0].head(args.test_samples // 2)
    df_sub = pd.concat([df_real, df_fake]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    samples = []
    for _, row in df_sub.iterrows():
        p = str(images_dir / str(row['path']).replace('/', os.sep))
        samples.append((p, int(row['label'])))

    # 2. Load Models
    models_dict = {
        "Baseline 1 (Spatial)": load_model("spatial", PROJECT_ROOT / "results/checkpoints/baseline1_efficientnet_b4_spatial_best.pt", device),
        "Baseline 2 (Frequency)": load_model("frequency", PROJECT_ROOT / "results/checkpoints/baseline2_frequency_dwt_best.pt", device),
        "Proposed (FGSA-Net)": load_model("fgsa_net", PROJECT_ROOT / "results/checkpoints/proposed_fgsa_net_best.pt", device),
    }

    perturbation_scenarios = [
        ("Clean (No Distortion)", "clean", 0),
        ("JPEG Quality 90", "jpeg", 90),
        ("JPEG Quality 70", "jpeg", 70),
        ("JPEG Quality 50", "jpeg", 50),
        ("JPEG Quality 30", "jpeg", 30),
        ("Gaussian Blur (r=1)", "blur", 1.0),
        ("Gaussian Blur (r=2)", "blur", 2.0),
        ("Gaussian Noise (std=0.05)", "noise", 0.05),
        ("Gaussian Noise (std=0.10)", "noise", 0.10),
    ]

    results_table = {}
    for model_name in models_dict.keys():
        results_table[model_name] = {}

    for name, p_type, lvl in perturbation_scenarios:
        print(f"\n--> Evaluating Condition: {name}...")
        ds = PerturbedDeepfakeDataset(samples, perturbation_type=p_type, level=lvl)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

        for model_name, model in models_dict.items():
            metrics = evaluate_model_on_loader(model, loader, device)
            results_table[model_name][name] = {
                "accuracy": round(metrics["accuracy"] * 100, 2),
                "roc_auc": round(metrics["roc_auc"], 4),
                "f1_score": round(metrics["f1_score"], 4)
            }
            print(f"    {model_name:<24}: Accuracy = {metrics['accuracy']*100:.2f}%, ROC-AUC = {metrics['roc_auc']:.4f}")

    # Save JSON
    out_json = PROJECT_ROOT / "results/evaluation/robustness_benchmark_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nRobustness metrics saved to: {out_json}")

    # Plot Publication-Grade Bar Chart
    conditions = [s[0] for s in perturbation_scenarios]
    spatial_accs = [results_table["Baseline 1 (Spatial)"][c]["accuracy"] for c in conditions]
    freq_accs = [results_table["Baseline 2 (Frequency)"][c]["accuracy"] for c in conditions]
    fgsa_accs = [results_table["Proposed (FGSA-Net)"][c]["accuracy"] for c in conditions]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    rects1 = ax.bar(x - width, spatial_accs, width, label='Baseline 1: Spatial Only (EfficientNet-B4)', color='#3b82f6')
    rects2 = ax.bar(x, freq_accs, width, label='Baseline 2: Frequency Only (2D-DWT)', color='#f59e0b')
    rects3 = ax.bar(x + width, fgsa_accs, width, label='Proposed: FGSA-Net (Attention Fusion)', color='#10b981')

    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Forensic Robustness Benchmark: Model Resilience under Perturbations & Compressions\n(PhD Paper I — M4D-Net Framework)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, ha="right", fontsize=10, fontweight='bold')
    ax.set_ylim([40, 105])
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = PROJECT_ROOT / "results/figures/robustness_comparison_benchmark.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Robustness comparison plot saved to: {fig_path}")


if __name__ == "__main__":
    main()
