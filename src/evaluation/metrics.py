"""
Academic Evaluation Metrics for Deepfake Image Detection
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve
)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the optimal threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # Find point where FPR is closest to FNR
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    optimal_threshold = thresholds[idx]
    return float(eer), float(optimal_threshold)


def calculate_binary_metrics(
    y_true: Union[np.ndarray, torch.Tensor, list],
    y_prob: Union[np.ndarray, torch.Tensor, list],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculates comprehensive research-grade classification metrics:
    - Accuracy
    - Precision
    - Recall (Sensitivity / TPR)
    - Specificity (TNR)
    - F1-Score
    - ROC-AUC
    - PR-AUC
    - EER (Equal Error Rate)
    - Confusion Matrix (TP, TN, FP, FN)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_pred = (y_prob >= threshold).astype(np.int32)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Core Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC metrics
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = 0.5

    # EER
    try:
        eer, opt_thresh = compute_eer(y_true, y_prob)
    except Exception:
        eer, opt_thresh = 0.5, 0.5

    return {
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall_sensitivity": round(float(rec), 6),
        "specificity": round(float(spec), 6),
        "f1_score": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "eer": round(float(eer), 6),
        "optimal_threshold_eer": round(float(opt_thresh), 6),
        "confusion_matrix": {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "total_samples": int(len(y_true))
    }


def print_metrics_summary(metrics: Dict[str, Any], title: str = "Evaluation Metrics"):
    """Formatted terminal summary table."""
    cm = metrics["confusion_matrix"]
    print(f"\n{'=' * 65}")
    print(f"  {title.upper()}")
    print(f"{'=' * 65}")
    print(f"  Accuracy       : {metrics['accuracy'] * 100:.2f}%")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC         : {metrics['pr_auc']:.4f}")
    print(f"  F1-Score       : {metrics['f1_score']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall (Sens)  : {metrics['recall_sensitivity']:.4f}")
    print(f"  Specificity    : {metrics['specificity']:.4f}")
    print(f"  Equal Error (EER): {metrics['eer'] * 100:.2f}%")
    print(f"  Confusion Matrix: TP={cm['true_positives']:,} | TN={cm['true_negatives']:,} | FP={cm['false_positives']:,} | FN={cm['false_negatives']:,}")
    print(f"{'=' * 65}\n")
