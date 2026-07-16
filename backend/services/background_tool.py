"""§5.5 background replace: soft matte + camera-stabilized plate.
The plate is generated ONCE and carried (same anti-morph discipline as §5.6);
it tracks camera motion via the median scene flow outside the object mask."""
from pathlib import Path

import cv2
import numpy as np

from services import flow_service, mask_service

_KERNEL = np.ones((3, 3), np.uint8)


def soft_matte(frame: np.ndarray, mask_alpha: np.ndarray, radius: int = 8) -> np.ndarray:
    """Upgrade a hard SAM 2 mask to a soft matte so hair/motion-blur edges
    composite seam-free against a brand-new background."""
    guide = frame.astype(np.float32) / 255.0
    src = mask_alpha.astype(np.float32)
    try:
        from cv2 import ximgproc
        matte = ximgproc.guidedFilter(guide, src, radius, 1e-4)
    except Exception:
        matte = cv2.GaussianBlur(src, (radius * 2 + 1, radius * 2 + 1), 0)
        core = cv2.erode((src > 0.5).astype(np.uint8), _KERNEL,
                         iterations=max(1, radius // 2))
        matte = np.maximum(matte, core.astype(np.float32))
    return np.clip(matte, 0.0, 1.0)


def _scene_shift(flows_dir: Path, pair_index: int, obj_mask: np.ndarray) -> tuple[float, float]:
    """Global camera translation for pair t→t+1 = median flow OUTSIDE the object."""
    f = flow_service.load_flow(flows_dir, pair_index, "fwd")[0].numpy()
    bg = obj_mask < 0.5
    return float(np.median(f[0][bg])), float(np.median(f[1][bg]))


def _shift_plate(plate: np.ndarray, sx: float, sy: float) -> np.ndarray:
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(plate, M, (plate.shape[1], plate.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


async def _default_generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
    from services import gemini_service
    return await gemini_service.edit_frame(
        frame_path, f"replace the background with: {prompt}", mask_path=mask_path)


async def apply_background_replace_range(project_dir: Path, frame_indices, anchor_index: int,
                                         prompt: str, device, generate=None) -> None:
    generate = generate or _default_generate
    frames_dir = project_dir / "frames"
    masks_dir = project_dir / "masks"
    flows_dir = project_dir / "flows"
    frame_indices = sorted(frame_indices)

    raw = await generate(frames_dir / f"frame_{anchor_index:04d}.jpg", prompt,
                         mask_path=masks_dir / f"mask_{anchor_index:04d}.png")
    plate0 = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)  # generate ONCE

    # Accumulated scene shift per frame relative to the anchor.
    shift: dict[int, tuple[float, float]] = {anchor_index: (0.0, 0.0)}
    for t in (i for i in frame_indices if i > anchor_index):
        px, py = shift[t - 1]
        m = mask_service.load_mask_alpha(masks_dir, t - 1)
        dx, dy = _scene_shift(flows_dir, t - 1, m)
        shift[t] = (px + dx, py + dy)
    for t in sorted((i for i in frame_indices if i < anchor_index), reverse=True):
        px, py = shift[t + 1]
        m = mask_service.load_mask_alpha(masks_dir, t)
        dx, dy = _scene_shift(flows_dir, t, m)
        shift[t] = (px - dx, py - dy)

    for t in frame_indices:
        frame = cv2.imread(str(frames_dir / f"frame_{t:04d}.jpg"))
        matte = soft_matte(frame, mask_service.load_mask_alpha(masks_dir, t))[..., None]
        plate = _shift_plate(plate0, *shift[t]).astype(np.float32)
        out = matte * frame.astype(np.float32) + (1 - matte) * plate
        cv2.imwrite(str(frames_dir / f"frame_{t:04d}.jpg"),
                    np.clip(out, 0, 255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
