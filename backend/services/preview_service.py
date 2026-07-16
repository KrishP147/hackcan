"""Instant single-frame preview (the editor's first paint after a button press).

Applies the requested edit to ONE frame entirely in memory and returns JPEG
bytes — never writes into frames/. The durable, temporally-coherent result
comes from the background propagation (edit_dispatch); transports preview with
a single-frame TELEA plate, which the temporal inpainter later replaces.
"""
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from services import local_edit_service, mask_service, object_tools, project_manager

PREVIEWABLE = {"recolor", "blur_region", "color_pop", "glow",
               "resize", "delete", "move"}


def _resolve_mask(project_dir: Path, project_id: str, frame_index: int) -> Path:
    mask_path = project_dir / "masks" / f"mask_{frame_index:04d}.png"
    if mask_path.exists():
        return mask_path
    # before first propagation only the clicked anchor frame has a mask;
    # approximate with it — the background pass corrects every frame
    anchor = project_manager.get_status(project_id).get("anchor_frame")
    if anchor:
        anchor_mask = project_dir / "masks" / f"mask_{anchor:04d}.png"
        if anchor_mask.exists():
            return anchor_mask
    raise RuntimeError("No mask — click an object first")


def _preview_move(frame_path: Path, mask_path: Path, dx: int, dy: int) -> None:
    img = cv2.imread(str(frame_path))
    hard = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 128)
    hole = object_tools._grow(hard)
    plate = cv2.inpaint(img, hole.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    alpha = mask_service.load_mask_alpha(mask_path.parent,
                                         int(mask_path.stem.split("_")[1]), feather_px=3)
    moved_alpha = object_tools._shift_mask(alpha, dx, dy)[..., None]
    moved_cut = object_tools._shift_rgb(img, dx, dy)
    out = np.clip(moved_alpha * moved_cut + (1 - moved_alpha) * plate.astype(np.float32),
                  0, 255).astype(np.uint8)
    cv2.imwrite(str(frame_path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _preview_resize(frame_path: Path, mask_path: Path, scale: float) -> None:
    img = cv2.imread(str(frame_path))
    hard = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
    ys, xs = np.nonzero(hard)
    if len(xs) == 0:
        return
    cx, cy = float(xs.mean()), float(ys.mean())
    h, w = hard.shape
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    soft = mask_service.load_mask_alpha(mask_path.parent,
                                        int(mask_path.stem.split("_")[1]), feather_px=3)
    s_alpha = cv2.warpAffine(soft, M, (w, h))
    s_rgb = cv2.warpAffine(img, M, (w, h))
    if scale < 1.0:
        hole = object_tools._grow(hard > 0)
        plate = cv2.inpaint(img, hole.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    else:
        plate = img
    a = s_alpha[..., None]
    out = np.clip(a * s_rgb.astype(np.float32) + (1 - a) * plate.astype(np.float32),
                  0, 255).astype(np.uint8)
    cv2.imwrite(str(frame_path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])


def render_preview(project_id: str, frame_index: int, edit_type: str, *,
                   color: str | None = None, blur_strength: int | None = None,
                   scale: float | None = None, dx: int = 0, dy: int = 0) -> bytes:
    if edit_type not in PREVIEWABLE:
        raise ValueError(f"No instant preview for '{edit_type}'")
    project_dir = project_manager.get_project_dir(project_id)
    frame_path = project_dir / "frames" / f"frame_{frame_index:04d}.jpg"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame {frame_index} not found")
    mask_path = _resolve_mask(project_dir, project_id, frame_index)

    with tempfile.TemporaryDirectory() as td:
        # same in-place editors the propagation pass uses, on a throwaway copy —
        # preview pixels match the final result wherever no temporal fill is needed
        tmp = Path(td) / frame_path.name
        shutil.copy2(frame_path, tmp)
        if edit_type == "recolor":
            local_edit_service.apply_recolor(tmp, mask_path, color or "FF0000")
        elif edit_type == "blur_region":
            local_edit_service.apply_blur_region(tmp, mask_path, blur_strength or 10)
        elif edit_type == "color_pop":
            local_edit_service.apply_color_pop(tmp, mask_path)
        elif edit_type == "glow":
            local_edit_service.apply_glow(tmp, mask_path)
        elif edit_type == "delete":
            local_edit_service.apply_remove(tmp, mask_path)
        elif edit_type == "resize":
            _preview_resize(tmp, mask_path, scale or 1.5)
        elif edit_type == "move":
            _preview_move(tmp, mask_path, dx, dy)
        return tmp.read_bytes()
