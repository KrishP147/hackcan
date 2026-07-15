"""Temporally-coherent inpainting (Doc 1 §4): source vacated/revealed background
from neighboring frames via cached flow — sourced, not re-guessed. TELEA only for
never-seen residual pixels, EMA'd across frames so the patch doesn't shimmer."""
from pathlib import Path

import cv2
import numpy as np
import torch

from services import flow_service
from services.propagation_service import iter_chain


@torch.no_grad()
def inpaint_video(frames: dict[int, np.ndarray], hole_masks: dict[int, np.ndarray],
                  flows_dir: Path, device: torch.device,
                  max_donor_dist: int = 8, on_frame=None) -> dict[int, np.ndarray]:
    """on_frame(t, filled) fires as each frame completes (ascending order) so
    callers can save/report progressively instead of waiting for the whole clip."""
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
            # donor chains anchored at t, extended one pairwise flow per k —
            # iter_chain yields (d, F_{d→t}, F_{t→d}) walking outward
            chains = {
                +1: iter_chain(flows_dir, t, +1, min(max_donor_dist, hi - t), device),
                -1: iter_chain(flows_dir, t, -1, min(max_donor_dist, t - lo), device),
            }
            for k in range(1, max_donor_dist + 1):
                for s in (-1, +1):
                    d = t + s * k
                    if d < lo or d > hi:
                        continue
                    _, f_dt, f_td = next(chains[s])
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
                # nearest donors carry ~all the weight (1/k); once every hole
                # pixel is sourced, farther donors change nothing visible
                if (wsum[hole] > 0).all():
                    break
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
        if on_frame:
            on_frame(t, out[t])
    return out
