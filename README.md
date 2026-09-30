# M4D-Net — Paper I: Frequency-Guided Spatial Attention Network for Robust Deepfake Image Detection

**Author**: Ramesh Prabhakaran R.  
**Research Framework**: Unified Multimodal Deepfake Detection Framework (M4D-Net) — Paper I  
**Target Publication**: IEEE / ACM Transactions on Pattern Analysis & Machine Intelligence / Forensics

---

## 1. Research Overview

This repository implements the research pipeline for **Paper I: Frequency-Guided Spatial Attention Network with Discrete Wavelet Decomposition for Robust Deepfake Image Detection**.

```
INPUT IMAGE (256x256 RGB)
            │
    ┌───────┴────────────────────────┐
    │                                │
    ▼                                ▼
SPATIAL BRANCH               FREQUENCY BRANCH
(EfficientNet-B4)          (2D-DWT: LL, LH, HL, HH)
    │                                │
    ▼ (f_s: 1792-dim)                ▼ (f_f: 1792-dim)
    └───────────────┬────────────────┘
                    ▼
          CROSS-MODAL ATTENTION FUSION
          F_fused = θ_s · f_s + θ_f · f_f
                    │
                    ▼
          BINARY CLASSIFIER HEAD
                    │
                    ▼
            REAL / FAKE PREDICTION
```

---

## 2. Repository Structure

```
DEEPFAKE/
├── configs/                     # YAML Configuration files
│   ├── dataset_140k.yaml        # Dataset definition & paths
│   ├── baseline_spatial.yaml    # Baseline 1: Spatial EfficientNet-B4
│   ├── baseline_frequency.yaml  # Baseline 2: Frequency 2D-DWT
│   └── proposed_fgsa_net.yaml   # Proposed FGSA-Net
│
├── src/                         # Core implementation source code
│   ├── datasets/                # PyTorch dataset & data loaders
│   ├── frequency/               # DWT & DCT mathematical extractors
│   ├── models/                  # Neural network architectures
│   ├── fusion/                  # Cross-modal learnable attention gating
│   ├── training/                # AMP Trainer, losses, optimization
│   ├── evaluation/              # Academic metrics (ROC-AUC, EER, F1, etc.)
│   └── utils/                   # Verification tools & logging
│
├── scripts/                     # Executable command-line scripts
│   ├── run_dataset_verification.py   # Dataset verification audit
│   ├── test_pipeline_integrity.py    # Unit tests for models & shapes
│   ├── train.py                      # Universal model training script
│   └── evaluate.py                   # Model evaluation on held-out test set
│
├── results/                     # Experimental outputs & figures
│   ├── verification/            # Phase 1 dataset audit reports
│   ├── checkpoints/             # Trained PyTorch model checkpoints
│   ├── logs/                    # Training history JSON traces
│   └── figures/                 # Evaluation curves & visual figures
│
├── requirements.txt             # Pinned Python package dependencies
└── README.md                    # Research documentation
```

---

## 3. Quick Start & Execution Guide

### 1. Dataset Verification & Quality Audit
Run the exhaustive dataset verification across all 140,000 images:
```bash
python scripts/run_dataset_verification.py
```

### 2. Pipeline Integrity Unit Test
Run the architecture and tensor-flow unit test:
```bash
python scripts/test_pipeline_integrity.py
```

### 3. Training Baselines & Proposed Model

#### Train Baseline 1 (Spatial EfficientNet-B4):
```bash
python scripts/train.py --config configs/baseline_spatial.yaml
```

#### Train Baseline 2 (Frequency 2D-DWT):
```bash
python scripts/train.py --config configs/baseline_frequency.yaml
```

#### Train Proposed Model (FGSA-Net):
```bash
python scripts/train.py --config configs/proposed_fgsa_net.yaml
```

*Tip for quick validation on a subset:*
```bash
python scripts/train.py --config configs/baseline_spatial.yaml --max-train-samples 10000 --max-eval-samples 2000 --epochs 5
```

### 4. Evaluating on Held-Out Test Set (20,000 images)
```bash
python scripts/evaluate.py --config configs/baseline_spatial.yaml --checkpoint results/checkpoints/baseline1_efficientnet_b4_spatial_best.pt
```

---

## 4. Academic Evaluation Metrics Tracked
- **Accuracy**
- **ROC-AUC** (Area Under Receiver Operating Characteristic)
- **PR-AUC** (Area Under Precision-Recall Curve)
- **F1-Score**
- **Precision & Recall (Sensitivity / TPR)**
- **Specificity (TNR)**
- **Equal Error Rate (EER)**
- **Confusion Matrix (TP, TN, FP, FN)**
