import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        base_dir = Path(__file__).resolve().parent.parent
        checkpoint = str(base_dir / "checkpoints" / "sam2_hiera_large.pt")
        model = build_sam2("sam2_hiera_l.yaml", checkpoint, device=device)
        _predictor = SAM2ImagePredictor(model)
    return _predictor


def segment_frame(frame_path: Path, click_x: int, click_y: int) -> np.ndarray:
    predictor = get_predictor()
    image = np.array(Image.open(frame_path).convert("RGB"))
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([[click_x, click_y]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    return masks[np.argmax(scores)]


def propagate_masks(frames_dir: Path, anchor_index: int, anchor_mask: np.ndarray, masks_dir: Path) -> int:
    """
    For hackathon: apply the anchor mask to all frames.
    Full SAM 2 video propagation is a stretch improvement.
    This saves the anchor mask for every frame as a starting point.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    mask_img = (anchor_mask * 255).astype(np.uint8)

    for i, frame_file in enumerate(frame_files, start=1):
        mask_path = masks_dir / f"mask_{i:04d}.png"
        Image.fromarray(mask_img).save(mask_path)

    return len(frame_files)
