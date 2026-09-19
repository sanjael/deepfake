"""
Video Deepfake Predictor / Frame-by-Frame Forensic Inference Engine
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

import sys
import argparse
from pathlib import Path
import cv2
import yaml
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.spatial_branch import EfficientNetB4SpatialBranch
from src.models.frequency_branch import FrequencyBranch
from src.models.fgsa_net import FGSANet
from src.datasets.transforms import get_validation_transforms


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    arch = model_cfg.get("architecture", "fgsa_net").lower()

    if arch in ["efficientnet_b4", "spatial"]:
        model = EfficientNetB4SpatialBranch(
            pretrained=False,
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            num_classes=1,
            feature_dim=1792
        )
    elif arch in ["frequency_branch", "frequency"]:
        model = FrequencyBranch(
            in_channels=12,
            feature_dim=1792,
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            num_classes=1
        )
    elif arch in ["fgsa_net", "proposed"]:
        model = FGSANet(
            feature_dim=1792,
            pretrained_spatial=False,
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            num_classes=1
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def predict_video(
    video_path: str,
    checkpoint_path: str = "results/checkpoints/baseline1_efficientnet_b4_spatial_best.pt",
    config_path: str = "configs/baseline_spatial.yaml",
    sample_rate_fps: int = 2
):
    v_path = Path(video_path)
    if not v_path.exists():
        raise FileNotFoundError(f"Video not found: {v_path}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg_path, ckpt_path, device)
    transform = get_validation_transforms(img_size=256)

    # Open video
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {v_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / sample_rate_fps))

    print(f"\nAnalyzing Video: {v_path.name}")
    print(f"Total Frames: {total_frames:,} | Native FPS: {video_fps:.1f} | Sampling 1 frame every {frame_interval} frames (~{sample_rate_fps} FPS)")

    frame_count = 0
    sampled_probs_real = []

    pbar = tqdm(total=total_frames, desc="Extracting & Evaluating Frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_tensor)
                prob_real = float(torch.sigmoid(logits).cpu().numpy().item())
                sampled_probs_real.append(prob_real)

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if not sampled_probs_real:
        raise ValueError("No frames could be extracted from video.")

    mean_prob_real = float(np.mean(sampled_probs_real))
    mean_prob_fake = 1.0 - mean_prob_real
    fake_frame_count = sum(1 for p in sampled_probs_real if p < 0.5)
    real_frame_count = len(sampled_probs_real) - fake_frame_count

    final_verdict = "REAL VIDEO" if mean_prob_real >= 0.5 else "FAKE / DEEPFAKE VIDEO"
    confidence = max(mean_prob_real, mean_prob_fake) * 100.0

    print("\n" + "=" * 70)
    print("  VIDEO DEEPFAKE FORENSIC INFERENCE RESULT")
    print("=" * 70)
    print(f"  Target Video          : {v_path.name}")
    print(f"  Final Video Verdict   : {final_verdict}")
    print(f"  Overall Confidence    : {confidence:.2f}%")
    print(f"  Real Probability      : {mean_prob_real * 100:.2f}%")
    print(f"  Fake Probability      : {mean_prob_fake * 100:.2f}%")
    print(f"  Frames Analyzed       : {len(sampled_probs_real):,} frames")
    print(f"  Classified Real Frames: {real_frame_count} ({(real_frame_count/len(sampled_probs_real))*100:.1f}%)")
    print(f"  Classified Fake Frames: {fake_frame_count} ({(fake_frame_count/len(sampled_probs_real))*100:.1f}%)")
    print("=" * 70 + "\n")

    return final_verdict, confidence, mean_prob_real, mean_prob_fake


def main():
    parser = argparse.ArgumentParser(description="Predict Deepfake on Any Video File (.mp4, .avi, etc.)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/baseline1_efficientnet_b4_spatial_best.pt", help="Model checkpoint")
    parser.add_argument("--config", type=str, default="configs/baseline_spatial.yaml", help="Model config YAML")
    parser.add_argument("--fps", type=int, default=2, help="Sample rate (frames per second to analyze)")
    args = parser.parse_args()

    predict_video(args.video, args.checkpoint, args.config, args.fps)


if __name__ == "__main__":
    main()
