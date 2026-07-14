"""Temporally-coherent inpainting (Doc 1 §4): source vacated/revealed background
from neighboring frames via cached flow — sourced, not re-guessed. TELEA only for
never-seen residual pixels, EMA'd across frames so the patch doesn't shimmer."""
from pathlib import Path

import cv2
import numpy as np
import torch

from services import flow_service
from services.propagation_service import _compose_chain


@torch.no_grad()
def inpaint_video(frames: dict[int, np.ndarray], hole_masks: dict[int, np.ndarray],
                  flows_dir: Path, device: torch.device,
                  max_donor_dist: int = 8) -> dict[int, np.ndarray]:
    indices = sorted(frames)
    lo, hi = indices[0], indices[-1]
    out: dict[int, np.ndarray] = {}
    prev_telea: np.ndarray | None = None
    prev_telea_mask: np.ndarray | None = None
    for t in indices:
        frame = frames[t].astype(np.float32)
        hole = hole_masks[t].astype(bool)
        acc = np.zeros_like(frame)
        wsum = np.zeros(frame.shape[:2], np.float32)
        if hole.any():
            for k in range(1, max_donor_dist + 1):
                for d in (t - k, t + k):
                    if d < lo or d > hi:
                        continue
                    # F_{t→d}: where each target pixel lives in the donor frame
                    f_td, f_dt = _compose_chain(flows_dir, d, t, device)
                    donor = torch.from_numpy(frames[d].astype(np.float32)) \
                        .permute(2, 0, 1)[None].to(device)
                    donor_ok = torch.from_numpy(
                        (~hole_masks[d].astype(bool)).astype(np.float32))[None, None].to(device)
                    warped = flow_service.warp(donor, f_td)[0].permute(1, 2, 0).cpu().numpy()
                    usable = flow_service.warp(donor_ok, f_td)[0, 0].cpu().numpy()
                    valid = flow_service.fb_check(f_td, f_dt)[0, 0].cpu().numpy()
                    w = (usable > 0.99).astype(np.float32) * valid / (k + 1e-3)
                    w = np.where(hole, w, 0.0)
                    acc += w[..., None] * warped
                    wsum += w
        filled = frame.copy()
        got = wsum > 0
        filled[got] = acc[got] / wsum[got][..., None]
        residual = hole & ~got
        if residual.any():
            telea = cv2.inpaint(np.clip(filled, 0, 255).astype(np.uint8),
                                residual.astype(np.uint8) * 255, 3,
                                cv2.INPAINT_TELEA).astype(np.float32)
            if prev_telea is not None and prev_telea_mask is not None:
                shared = residual & prev_telea_mask
                telea[shared] = 0.5 * telea[shared] + 0.5 * prev_telea[shared]  # EMA
            filled[residual] = telea[residual]
            prev_telea, prev_telea_mask = telea, residual
        else:
            prev_telea, prev_telea_mask = None, None
        out[t] = np.clip(filled, 0, 255).astype(np.uint8)
    return out
