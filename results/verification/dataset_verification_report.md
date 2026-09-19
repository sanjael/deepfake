# Dataset Verification & Integrity Audit Report
**PhD Project**: *Frequency-Guided Spatial Attention Network with Discrete Wavelet Decomposition for Robust Deepfake Image Detection*
**Principal Investigator / Author**: Ramesh Prabhakaran R.
**Verification Scope**: 140K Real and Fake Faces Dataset (`e:\DEEPFAKE\daaset`)

---

## 1. Executive Summary & Quality Gate Status

| Verification Gate | Result | Notes |
| :--- | :--- | :--- |
| **File Existence & Disk Integrity** | PASS | 140,000 / 140,000 files verified on disk |
| **Class Balance Ratio** | PASS (50:50) | 70,000 Real / 70,000 Fake (Imbalance Ratio: 1.000) |
| **Data Leakage & Split Independence** | PASS (0 Leakage) | Zero cross-split ID, path, or image hash overlap |
| **Image Integrity & Dimensionality** | PASS (100% Clean) | All images $256\times 256\times 3$ RGB JPEG |
| **Overall Verification Verdict** | **PASSED** | Ready for Baseline Modeling |

---

## 2. Dataset Distribution & Imbalance Audit

| Split | Total Samples | Real Count (Label 1) | Fake Count (Label 0) | Real % | Fake % | Imbalance Ratio | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | 100,000 | 50,000 | 50,000 | 50.0% | 50.0% | 1.0:1 | Perfect 50:50 Balance |
| **VALID** | 20,000 | 10,000 | 10,000 | 50.0% | 50.0% | 1.0:1 | Perfect 50:50 Balance |
| **TEST** | 20,000 | 10,000 | 10,000 | 50.0% | 50.0% | 1.0:1 | Perfect 50:50 Balance |
| **OVERALL** | **140,000** | **70,000** | **70,000** | **50.0%** | **50.0%** | **1.0:1** | **Perfect 50:50 Balance** |

> **Finding**: The dataset is **strictly balanced (50:50)** across all splits. Standard Binary Cross-Entropy (BCE) loss with default threshold (0.5) is mathematically well-calibrated without requiring class-reweighting or focal loss adjustments for imbalance.

---

## 3. Data Leakage & Split Independence Verification

To ensure scientific validity and prevent optimistic evaluation bias, we evaluated three independent leakage vectors:

1. **ID Metadata Overlap**:
   - Train $\cap$ Valid ID Overlap: **0**
   - Train $\cap$ Test ID Overlap: **0**
   - Valid $\cap$ Test ID Overlap: **0**

2. **Original Generator/FFHQ Source Path Overlap**:
   - Train $\cap$ Valid Source Overlap: **0**
   - Train $\cap$ Test Source Overlap: **0**
   - Valid $\cap$ Test Source Overlap: **0**

3. **Pixel Content (MD5 Hash) Collision Overlap**:
   - Train $\cap$ Valid Identical Images: **0**
   - Train $\cap$ Test Identical Images: **0**
   - Valid $\cap$ Test Identical Images: **0**
   - Intra-split duplicate hashes: Train: 0, Valid: 0, Test: 0

> **Leakage Verdict**: **ZERO LEAKAGE DETECTED**. The training, validation, and test partitions are completely disjoint.

---

## 4. Image Quality, Formats & Dimension Audit

| Split | Images Checked | Corrupted Files | Resolutions Detected | Color Mode | Mean Size (KB) | Health Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | 100,000 | 0 | (256, 256, 3): 100000 | RGB: 100000 | 28.02 KB | 100% Valid |
| **VALID** | 20,000 | 0 | (256, 256, 3): 20000 | RGB: 20000 | 27.97 KB | 100% Valid |
| **TEST** | 20,000 | 0 | (256, 256, 3): 20000 | RGB: 20000 | 28.01 KB | 100% Valid |

---

## 5. Next Steps (Proceeding to Phase 2: Baselines)

1. **Construct PyTorch Dataset & DataLoader** (`src/datasets/dataset_140k.py`) with GPU pinned memory and standard EfficientNet-B4 input transforms ($256 \times 256 \rightarrow 380 \times 380$ or $256 \times 256$ native).
2. **Implement Baseline 1 (Spatial Branch: EfficientNet-B4)** (`src/models/spatial_branch.py`).
3. **Implement Baseline 2 (Frequency Branch: DWT + DCT)** (`src/models/frequency_branch.py`).
4. **Implement Attention Fusion & FGSA-Net** (`src/models/fgsa_net.py`).