"""§5.6 replace: generate ONCE at the anchor → star-warp propagate →
warp-then-refine re-anchor. Never regenerate from scratch — a second
independent draw is exactly the morphing this architecture removes."""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from services import inpaint_service, mask_service
from services.propagation_service import (EditLayer, blend_bidirectional, composite,
                                          make_anchor_layer, needs_reanchor, star_warp)

CHAIN_CAP = 60  # frames (~2s @30fps) since last anchor

REFINE_PROMPT = "clean up warp artifacts; preserve this exact appearance"


def _decode(edited_bytes: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(edited_bytes, np.uint8), cv2.IMREAD_COLOR)


async def _default_generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
    from services import gemini_service
    return await gemini_service.edit_frame_with_reference(
        frame_path, prompt,
        reference_frame_path=reference_frame_path, mask_path=mask_path)


async def apply_replace_range(project_dir: Path, frame_indices, anchor_index: int,
                              prompt: str, device, generate=None) -> None:
    generate = generate or _default_generate
    frames_dir = project_dir / "frames"
    masks_dir = project_dir / "masks"
    flows_dir = project_dir / "flows"
    frame_indices = sorted(frame_indices)

    async def gen_layer(idx: int, gen_prompt: str, ref: Path | None,
                        init_frame: Path | None = None) -> EditLayer:
        src = init_frame or frames_dir / f"frame_{idx:04d}.jpg"
        edited = _decode(await generate(src, gen_prompt, reference_frame_path=ref,
                                        mask_path=masks_dir / f"mask_{idx:04d}.png"))
        alpha = mask_service.load_mask_alpha(masks_dir, idx, feather_px=5)
        return make_anchor_layer(edited, alpha)

    anchor_path = frames_dir / f"frame_{anchor_index:04d}.jpg"
    anchors: dict[int, EditLayer] = {anchor_index: await gen_layer(anchor_index, prompt, None)}
    anchor_area = float((mask_service.load_mask_alpha(masks_dir, anchor_index) > 0.5).sum())

    # Pass 1 — discover re-anchor points walking outward from the anchor.
    # Warp from the nearest anchor; when the carry degrades, refine the WARPED
    # frame (img2img-style seed through the API) rather than regenerating.
    for direction in (1, -1):
        cur = anchor_index
        walk = [t for t in frame_indices if (t - anchor_index) * direction > 0]
        if direction == -1:
            walk = walk[::-1]
        for t in walk:
            dist = abs(t - cur)
            layer = star_warp(anchors[cur], cur, [t], flows_dir, device)[t]
            mask_t = mask_service.load_mask_alpha(masks_dir, t)
            if dist >= CHAIN_CAP or needs_reanchor(layer, mask_t, anchor_area, dist):
                frame_t = cv2.imread(str(frames_dir / f"frame_{t:04d}.jpg"))
                warped_full = composite(frame_t, layer, mask_t)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    cv2.imwrite(tf.name, warped_full)
                    anchors[t] = await gen_layer(t, REFINE_PROMPT, anchor_path,
                                                 init_frame=Path(tf.name))
                cur = t

    # Pass 2 — final layers: bidirectional star-warp between bracketing anchors.
    anchor_idx_sorted = sorted(anchors)
    for t in frame_indices:
        past = max((a for a in anchor_idx_sorted if a <= t), default=None)
        fut = min((a for a in anchor_idx_sorted if a >= t), default=None)
        if t in anchors:
            layer = anchors[t]
        elif past is not None and fut is not None and past != fut:
            lp = star_warp(anchors[past], past, [t], flows_dir, device)[t]
            lf = star_warp(anchors[fut], fut, [t], flows_dir, device)[t]
            layer = blend_bidirectional(lp, lf, t - past, fut - t)
        else:
            a = past if past is not None else fut
            layer = star_warp(anchors[a], a, [t], flows_dir, device)[t]
        mask_t = mask_service.load_mask_alpha(masks_dir, t, feather_px=5)
        hole = (mask_t > 0.5) & (layer.validity < 0.5)
        frame_t = cv2.imread(str(frames_dir / f"frame_{t:04d}.jpg"))
        out = composite(frame_t, layer, mask_t)
        if hole.any():                                  # disocclusion inside the object
            out = inpaint_service.inpaint_video({t: out}, {t: hole},
                                                flows_dir, device)[t]
        cv2.imwrite(str(frames_dir / f"frame_{t:04d}.jpg"), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
