"""
2D Discrete Wavelet Transform (DWT) Extractor
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

import numpy as np
import pywt
import torch
import torch.nn as nn
from typing import Tuple, Dict, Union


class DWTExtractor2D:
    """
    Performs 2D Discrete Wavelet Transform decomposition on images.
    Decomposes an input image into 4 sub-bands:
    - LL: Low-frequency approximation (coarse facial structure)
    - LH: Horizontal detail (high-freq vertical edges)
    - HL: Vertical detail (high-freq horizontal edges)
    - HH: Diagonal detail (high-frequency corner & synthetic textures)
    """

    def __init__(self, wavelet: str = "haar", mode: str = "reflect"):
        self.wavelet = wavelet
        self.mode = mode

    def extract_numpy(self, img_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decomposes an RGB or grayscale numpy image (H, W, C) or (H, W).
        Returns dict containing LL, LH, HL, HH sub-bands.
        """
        if img_rgb.ndim == 2:
            cA, (cH, cV, cD) = pywt.dwt2(img_rgb, self.wavelet, mode=self.mode)
            return {"LL": cA, "LH": cH, "HL": cV, "HH": cD}
        
        # Multi-channel RGB
        h, w, c = img_rgb.shape
        subbands = {"LL": [], "LH": [], "HL": [], "HH": []}
        for ch in range(c):
            cA, (cH, cV, cD) = pywt.dwt2(img_rgb[:, :, ch], self.wavelet, mode=self.mode)
            subbands["LL"].append(cA)
            subbands["LH"].append(cH)
            subbands["HL"].append(cV)
            subbands["HH"].append(cD)

        return {k: np.stack(v, axis=-1) for k, v in subbands.items()}

    def extract_tensor_4ch(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Takes (B, C, H, W) or (C, H, W) PyTorch tensor, computes DWT per channel,
        and stacks subbands into a 4*C channel feature map of shape (B, 4*C, H/2, W/2).
        """
        is_batched = img_tensor.ndim == 4
        if not is_batched:
            img_tensor = img_tensor.unsqueeze(0)
            
        b, c, h, w = img_tensor.shape
        np_imgs = img_tensor.detach().cpu().numpy()
        
        batch_out = []
        for i in range(b):
            c_out = []
            for ch in range(c):
                cA, (cH, cV, cD) = pywt.dwt2(np_imgs[i, ch], self.wavelet, mode=self.mode)
                c_out.extend([cA, cH, cV, cD])
            batch_out.append(np.stack(c_out, axis=0))
            
        out_tensor = torch.tensor(np.stack(batch_out, axis=0), dtype=img_tensor.dtype, device=img_tensor.device)
        return out_tensor if is_batched else out_tensor.squeeze(0)
