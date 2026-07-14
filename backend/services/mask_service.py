"""Mask conditioning (Doc 1 §2.7): SAM 2 masks jitter a pixel or two at
boundaries — invisible for blur, visible for glow/color-pop edges."""
from pathlib import Path

import cv2
import numpy as np

_KERNEL = np.ones((3, 3), np.uint8)


def stabilize_masks(masks_dir: Path, out_dir: Path | None = None) -> int:
    """Morphological close (3×3) + 2-frame EMA on mask alpha, rebinarized."""
    out_dir = out_dir or masks_dir
    paths = sorted(masks_dir.glob("mask_*.png"))
    prev = None
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _KERNEL)
        if prev is not None:
            m = 0.6 * m + 0.4 * prev
        prev = m
        binary = (m >= 0.5).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / p.name), binary)
    return len(paths)


def condition_single(mask_path: Path) -> None:
    """Close pinholes on one freshly-segmented mask (no EMA — needs neighbors)."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _KERNEL)
    cv2.imwrite(str(mask_path), m)


def load_mask_alpha(masks_dir: Path, frame_index: int, feather_px: int = 0) -> np.ndarray:
    """Mask as float alpha in [0,1]; optional Gaussian feather at the boundary only
    (hard core stays 1.0, far field stays 0.0)."""
    p = masks_dir / f"mask_{frame_index:04d}.png"
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    if feather_px > 0:
        k = feather_px * 2 + 1
        blurred = cv2.GaussianBlur(a, (k, k), 0)
        core = cv2.erode((a >= 0.5).astype(np.uint8), _KERNEL, iterations=feather_px)
        a = np.clip(blurred, 0.0, 1.0)
        a[core == 1] = 1.0
    return a
