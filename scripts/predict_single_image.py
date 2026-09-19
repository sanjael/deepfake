"""
Single Image Deepfake Predictor / Inference Engine with Auto Face Cropping
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch

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


def auto_detect_and_crop_face(img_rgb: Image.Image) -> Image.Image:
    """Detects frontal face and crops it; if no face detected, returns original."""
    np_img = np.array(img_rgb)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    if len(faces) > 0:
        # Choose the largest detected face
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces[0]
        # Add 15% margin
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(np_img.shape[1], x + w + margin_x)
        y2 = min(np_img.shape[0], y + h + margin_y)
        face_crop = np_img[y1:y2, x1:x2]
        return Image.fromarray(face_crop)

    return img_rgb


def predict_image(
    image_path: str,
    checkpoint_path: str = "results/checkpoints/proposed_fgsa_net_best.pt",
    config_path: str = "configs/proposed_fgsa_net.yaml",
    save_viz: bool = True,
    auto_crop: bool = True
):
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg_path, ckpt_path, device)

    # Preprocess
    with Image.open(img_path) as raw_img:
        img_rgb = raw_img.convert("RGB")

    face_img = auto_detect_and_crop_face(img_rgb) if auto_crop else img_rgb

    transform = get_validation_transforms(img_size=256)
    img_tensor = transform(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        prob_real = float(torch.sigmoid(logits).cpu().numpy().item())
        prob_fake = 1.0 - prob_real

    pred_label = "REAL FACE" if prob_real >= 0.5 else "FAKE / DEEPFAKE FACE"
    confidence = max(prob_real, prob_fake) * 100.0

    print("\n" + "=" * 65)
    print("  DEEPFAKE FORENSIC INFERENCE RESULT")
    print("=" * 65)
    print(f"  Target Image    : {img_path.name}")
    print(f"  Final Prediction: {pred_label}")
    print(f"  Confidence      : {confidence:.2f}%")
    print(f"  Real Probability: {prob_real * 100:.2f}%")
    print(f"  Fake Probability: {prob_fake * 100:.2f}%")
    print("=" * 65 + "\n")

    if save_viz:
        fig, ax = plt.subplots(figsize=(6, 7), dpi=200)
        ax.imshow(face_img)
        ax.axis("off")
        
        color = "#16a34a" if prob_real >= 0.5 else "#dc2626"
        title_text = f"Prediction: {pred_label}\nConfidence: {confidence:.1f}% (Real: {prob_real*100:.1f}%, Fake: {prob_fake*100:.1f}%)"
        ax.set_title(title_text, fontsize=12, fontweight="bold", color=color, pad=12)
        
        out_dir = PROJECT_ROOT / "results" / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{img_path.stem}_prediction.png"
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"Forensic visual card saved to: {out_path}")

    return pred_label, confidence, prob_real, prob_fake


def main():
    parser = argparse.ArgumentParser(description="Predict Deepfake on Any Single Face Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (.jpg, .png, etc.)")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/proposed_fgsa_net_best.pt", help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/proposed_fgsa_net.yaml", help="Model config YAML")
    args = parser.parse_args()

    predict_image(args.image, args.checkpoint, args.config)


if __name__ == "__main__":
    main()
