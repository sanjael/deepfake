"""
Pipeline Integrity & Architecture Unit Test
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.spatial_branch import EfficientNetB4SpatialBranch
from src.models.frequency_branch import FrequencyBranch
from src.models.fgsa_net import FGSANet
from src.datasets.transforms import get_training_transforms
from src.evaluation.metrics import calculate_binary_metrics


def run_unit_tests():
    print("=" * 80)
    print("Running Pipeline & Architecture Integrity Tests...")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    dummy_input = torch.randn(4, 3, 256, 256).to(device)
    dummy_labels = torch.tensor([1.0, 0.0, 1.0, 0.0]).to(device).unsqueeze(1)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 1. Test Spatial Branch (EfficientNet-B4)
    print("\n[Test 1/4] Testing Baseline 1: Spatial Branch (EfficientNet-B4)...")
    spatial_model = EfficientNetB4SpatialBranch(pretrained=False).to(device)
    logits, feat = spatial_model(dummy_input, return_features=True)
    assert logits.shape == (4, 1), f"Expected logits shape (4, 1), got {logits.shape}"
    assert feat.shape == (4, 1792), f"Expected feature shape (4, 1792), got {feat.shape}"
    loss = criterion(logits, dummy_labels)
    loss.backward()
    print("  -> Spatial Branch passed! (Logits: (4, 1), Features: (4, 1792), Backward OK)")

    # 2. Test Frequency Branch
    print("\n[Test 2/4] Testing Baseline 2: Frequency Branch (2D-DWT ConvNet)...")
    freq_model = FrequencyBranch(in_channels=12, feature_dim=1792).to(device)
    f_logits, f_feat = freq_model(dummy_input, return_features=True)
    assert f_logits.shape == (4, 1), f"Expected freq logits (4, 1), got {f_logits.shape}"
    assert f_feat.shape == (4, 1792), f"Expected freq features (4, 1792), got {f_feat.shape}"
    f_loss = criterion(f_logits, dummy_labels)
    f_loss.backward()
    print("  -> Frequency Branch passed! (Logits: (4, 1), Features: (4, 1792), Backward OK)")

    # 3. Test FGSA-Net (Proposed)
    print("\n[Test 3/4] Testing Proposed Model: FGSA-Net (Spatial + Frequency + Attention Fusion)...")
    fgsa_model = FGSANet(feature_dim=1792, pretrained_spatial=False).to(device)
    fgsa_logits, theta_s, theta_f = fgsa_model(dummy_input, return_attention_weights=True)
    assert fgsa_logits.shape == (4, 1), f"Expected FGSA logits (4, 1), got {fgsa_logits.shape}"
    assert theta_s.shape == (4, 1792), f"Expected theta_s (4, 1792), got {theta_s.shape}"
    assert theta_f.shape == (4, 1792), f"Expected theta_f (4, 1792), got {theta_f.shape}"
    
    # Check attention weights sum to 1
    attn_sum = theta_s + theta_f
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "Attention weights must sum to 1.0"
    
    fgsa_loss = criterion(fgsa_logits, dummy_labels)
    fgsa_loss.backward()
    print("  -> FGSA-Net passed! (Logits: (4, 1), Attn Weights: (4, 1792), Attn Sum == 1.0, Backward OK)")

    # 4. Test Metrics Calculation
    print("\n[Test 4/4] Testing Academic Metrics Engine...")
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_prob = [0.95, 0.12, 0.88, 0.74, 0.05, 0.40, 0.65, 0.20]
    metrics = calculate_binary_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1_score"] == 1.0
    print(f"  -> Metrics Calculation passed! (Accuracy: {metrics['accuracy']}, ROC-AUC: {metrics['roc_auc']})")

    print("\n" + "=" * 80)
    print("All Pipeline Integrity Tests PASSED Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()
