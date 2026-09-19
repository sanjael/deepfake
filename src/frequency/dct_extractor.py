"""
2D Discrete Cosine Transform (DCT) Extractor
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

import numpy as np
from scipy.fftpack import dct, idct
import cv2
import torch
from typing import Tuple, Dict, Optional


class DCTExtractor2D:
    """
    Computes 2D Discrete Cosine Transform (DCT) representations:
    1. Full-image 2D DCT power spectrum (log-magnitude of frequency coefficients).
    2. 8x8 block-level DCT to isolate high-frequency AC components.
    """

    def __init__(self, block_size: int = 8):
        self.block_size = block_size

    @staticmethod
    def dct2(a: np.ndarray) -> np.ndarray:
        """2D DCT on a 2D numpy array."""
        return dct(dct(a.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def idct2(a: np.ndarray) -> np.ndarray:
        """2D Inverse DCT."""
        return idct(idct(a.T, norm='ortho').T, norm='ortho')

    def extract_full_dct_log_spectrum(self, img_rgb_or_gray: np.ndarray) -> np.ndarray:
        """
        Computes the log-magnitude 2D DCT spectrum for visualization / feature extraction.
        Returns normalized log spectrum in [0, 1].
        """
        if img_rgb_or_gray.ndim == 3:
            gray = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_rgb_or_gray

        dct_coeff = self.dct2(gray.astype(np.float32))
        log_spectrum = np.log(np.abs(dct_coeff) + 1e-6)
        
        # Normalize to [0, 1]
        s_min = log_spectrum.min()
        s_max = log_spectrum.max()
        if s_max - s_min > 1e-6:
            log_spectrum = (log_spectrum - s_min) / (s_max - s_min)
        else:
            log_spectrum = np.zeros_like(log_spectrum)

        return log_spectrum

    def extract_block_ac_features(self, img_rgb_or_gray: np.ndarray) -> np.ndarray:
        """
        Extracts 8x8 block-based DCT coefficients, separating DC from AC components.
        """
        if img_rgb_or_gray.ndim == 3:
            gray = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_rgb_or_gray

        h, w = gray.shape
        bs = self.block_size
        h_pad = (h // bs) * bs
        w_pad = (w // bs) * bs
        gray = gray[:h_pad, :w_pad].astype(np.float32)

        blocks_h = h_pad // bs
        blocks_w = w_pad // bs
        
        # Reshape to (blocks_h, bs, blocks_w, bs) -> (blocks_h, blocks_w, bs, bs)
        blocks = gray.reshape(blocks_h, bs, blocks_w, bs).transpose(0, 2, 1, 3)
        
        # Compute DCT on each block
        dct_blocks = np.zeros_like(blocks)
        for i in range(blocks_h):
            for j in range(blocks_w):
                dct_blocks[i, j] = self.dct2(blocks[i, j])

        return dct_blocks
