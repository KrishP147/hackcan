import numpy as np
from PIL import Image
from pathlib import Path
import torch

_image_predictor = None
_video_predictor = None

CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "sam2.1_hiera_large.pt"
CONFIG = "sam2.1/sam2.1_hiera_l"


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_image_predictor():
    global _image_predictor
    if _image_predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(CONFIG, str(CHECKPOINT), device=_device())
        _image_predictor = SAM2ImagePredictor(model)
    return _image_predictor


def get_video_predictor():
    global _video_predictor
    if _video_predictor is None:
        from sam2.build_sam import build_sam2_video_predictor
        _video_predictor = build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device=_device())
    return _video_predictor


def segment_frame(frame_path: Path, click_x: int, click_y: int) -> np.ndarray:
    """Segment object at click point in a single frame. Returns best mask."""
    predictor = get_image_predictor()
    image = np.array(Image.open(frame_path).convert("RGB"))
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([[click_x, click_y]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    return masks[np.argmax(scores)]


def propagate_masks(
    video_path: Path,
    anchor_index: int,
    click_x: int,
    click_y: int,
    masks_dir: Path,
) -> int:
    """
    Propagate object mask across the full video using SAM 2 video predictor.
    anchor_index is 1-based (matches our frame_XXXX.jpg naming).
    Propagates forward from anchor to end, then backward from anchor to start.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    predictor = get_video_predictor()

    # SAM 2 uses 0-based frame indices
    anchor_idx = anchor_index - 1

    inference_state = predictor.init_state(
        video_path=str(video_path),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    predictor.reset_state(inference_state)

    # Condition on click point at anchor frame
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=anchor_idx,
        obj_id=1,
        points=np.array([[click_x, click_y]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
        normalize_coords=False,
    )

    masks = {}

    # Forward: anchor → end
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
        mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy().astype(np.uint8) * 255
        masks[out_frame_idx] = mask

    # Backward: anchor → start
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        inference_state, reverse=True
    ):
        if out_frame_idx not in masks:
            mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy().astype(np.uint8) * 255
            masks[out_frame_idx] = mask

    # Save masks with 1-based filenames to match frame_XXXX.jpg naming
    for idx, mask in masks.items():
        Image.fromarray(mask).save(masks_dir / f"mask_{idx + 1:04d}.png")

    return len(masks)
