"""delete / resize / move (Doc 1 §5.2–§5.4) — deterministic transports riding
the §4 temporal inpainter. Cutouts come from the ORIGINAL frame per index, so
these tools cannot morph; only revealed background needs the inpainter."""
from pathlib import Path

import cv2
import numpy as np

from services import inpaint_service, mask_service


_DILATE = np.ones((5, 5), np.uint8)


def _grow(hole: np.ndarray, px: int = 2) -> np.ndarray:
    """Dilate a hole mask so JPEG/soft-edge contamination around the object
    boundary gets re-filled instead of smeared inward by the inpainter."""
    return cv2.dilate(hole.astype(np.uint8), _DILATE, iterations=px).astype(bool)


def _load(project_dir: Path, indices):
    frames = {t: cv2.imread(str(project_dir / "frames" / f"frame_{t:04d}.jpg"))
              for t in indices}
    masks = {t: mask_service.load_mask_alpha(project_dir / "masks", t) for t in indices}
    return frames, masks


def _save(project_dir: Path, frames: dict):
    for t, img in frames.items():
        cv2.imwrite(str(project_dir / "frames" / f"frame_{t:04d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


def apply_delete_range(project_dir: Path, frame_indices, device) -> None:
    frames, masks = _load(project_dir, frame_indices)
    holes = {t: _grow(masks[t] > 0.5) for t in frame_indices}
    _save(project_dir, inpaint_service.inpaint_video(frames, holes,
                                                     project_dir / "flows", device))


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    m = np.zeros_like(mask)
    h, w = mask.shape[:2]
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    if xs0 < xs1 and ys0 < ys1:
        m[ys0:ys1, xs0:xs1] = mask[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return m


def _shift_rgb(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.stack([_shift_mask(img[..., c].astype(np.float32), dx, dy)
                     for c in range(img.shape[2])], axis=2)


def apply_move_range(project_dir: Path, frame_indices, offsets: dict, device) -> None:
    frames, masks = _load(project_dir, frame_indices)
    cutouts = {t: frames[t].copy() for t in frame_indices}
    holes = {}
    for t in frame_indices:
        # plate = original with the WHOLE object removed; the moved cutout is
        # pasted on top, so the inpaint never borders contaminated object pixels
        holes[t] = _grow(masks[t] > 0.5)
    plates = inpaint_service.inpaint_video(frames, holes, project_dir / "flows", device)
    out = {}
    for t in frame_indices:
        dx, dy = offsets[t]
        alpha = mask_service.load_mask_alpha(project_dir / "masks", t, feather_px=3)
        moved_alpha = _shift_mask(alpha, dx, dy)[..., None]
        moved_cut = _shift_rgb(cutouts[t], dx, dy)
        plate = plates[t].astype(np.float32)
        out[t] = np.clip(moved_alpha * moved_cut + (1 - moved_alpha) * plate,
                         0, 255).astype(np.uint8)
    _save(project_dir, out)


def apply_resize_range(project_dir: Path, frame_indices, scale: float, device) -> None:
    frames, masks = _load(project_dir, frame_indices)
    holes = {}
    scaled = {}
    for t in frame_indices:
        m = (masks[t] > 0.5).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            holes[t] = np.zeros(m.shape, bool)
            scaled[t] = None
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        h, w = m.shape
        soft = mask_service.load_mask_alpha(project_dir / "masks", t, feather_px=3)
        s_alpha = cv2.warpAffine(soft, M, (w, h))
        s_rgb = cv2.warpAffine(frames[t], M, (w, h))
        scaled[t] = (s_rgb, s_alpha)
        holes[t] = _grow(m > 0) if scale < 1.0 else np.zeros(m.shape, bool)
    if any(hm.any() for hm in holes.values()):
        plates = inpaint_service.inpaint_video(frames, holes, project_dir / "flows", device)
    else:
        plates = {t: frames[t] for t in frame_indices}
    out = {}
    for t in frame_indices:
        if scaled[t] is None:
            out[t] = frames[t]
            continue
        s_rgb, s_alpha = scaled[t]
        a = s_alpha[..., None]
        out[t] = np.clip(a * s_rgb.astype(np.float32)
                         + (1 - a) * plates[t].astype(np.float32), 0, 255).astype(np.uint8)
    _save(project_dir, out)
