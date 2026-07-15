import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import tempfile

# Enable MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Patch torch.jit.script to avoid SAM2 JIT compilation errors with newer PyTorch
import torch
torch.jit.script = lambda f=None, *args, **kwargs: f if f is not None else (lambda fn: fn)

_image_predictor = None
_video_predictor = None


class PropagationCancelled(RuntimeError):
    """Raised when a user cancels an in-flight mask propagation job."""


def reset_predictors():
    """Call after switching checkpoints to force reload."""
    global _image_predictor, _video_predictor
    _image_predictor = None
    _video_predictor = None


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_checkpoint_and_config():
    base_dir = Path(__file__).resolve().parent.parent
    checkpoint = str(base_dir / "checkpoints" / "sam2_hiera_tiny.pt")
    config = "sam2_hiera_t.yaml"
    return checkpoint, config


def get_image_predictor():
    """Get SAM2 image predictor for single-frame segmentation."""
    global _image_predictor
    if _image_predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = _get_device()
        checkpoint, config = _get_checkpoint_and_config()
        model = build_sam2(config, checkpoint, device=device)
        _image_predictor = SAM2ImagePredictor(model)
    return _image_predictor


def get_video_predictor():
    """Get SAM2 video predictor for multi-frame propagation."""
    global _video_predictor
    if _video_predictor is None:
        from sam2.build_sam import build_sam2_video_predictor

        device = _get_device()
        checkpoint, config = _get_checkpoint_and_config()
        _video_predictor = build_sam2_video_predictor(config, checkpoint, device=device)
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
    frames_dir: Path,
    anchor_index: int,
    anchor_mask: np.ndarray,
    masks_dir: Path,
    click_x: int = None,
    click_y: int = None,
    frame_step: int = 1,
    cancel_check=None,
) -> int:
    """
    Propagate segmentation masks across video frames using SAM 2 video predictor.

    frame_step: process every Nth frame. Production passes 1; larger values
                are an explicit degraded local-preview mode only.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    if not frame_files or click_x is None or click_y is None:
        # Fallback: copy anchor mask to all frames
        mask_img = (anchor_mask * 255).astype(np.uint8)
        for i, _ in enumerate(frame_files, start=1):
            if cancel_check and cancel_check():
                raise PropagationCancelled()
            mask_path = masks_dir / f"mask_{i:04d}.png"
            Image.fromarray(mask_img).save(mask_path)
        return len(frame_files)

    # Select frames to process (always include anchor frame)
    total = len(frame_files)
    anchor_idx_0based = anchor_index - 1
    selected_indices = set(range(0, total, frame_step))
    selected_indices.add(anchor_idx_0based)
    selected_indices = sorted(selected_indices)

    # Create temp dir with symlinks named <number>.jpg for SAM 2
    with tempfile.TemporaryDirectory() as tmp_dir:
        idx_map = {}  # sam2 index -> original index
        reverse_map = {}  # original index -> sam2 index
        for sam2_idx, orig_idx in enumerate(selected_indices):
            link_path = os.path.join(tmp_dir, f"{sam2_idx:05d}.jpg")
            os.symlink(str(frame_files[orig_idx].resolve()), link_path)
            idx_map[sam2_idx] = orig_idx
            reverse_map[orig_idx] = sam2_idx

        sam2_anchor = reverse_map[anchor_idx_0based]

        predictor = get_video_predictor()
        inference_state = predictor.init_state(
            video_path=tmp_dir,
            offload_video_to_cpu=True,
        )
        predictor.reset_state(inference_state)

        points = np.array([[click_x, click_y]], dtype=np.float32)
        labels = np.array([1], np.int32)

        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=sam2_anchor,
            obj_id=1,
            points=points,
            labels=labels,
        )

        # Collect propagated masks (keyed by original frame index)
        propagated = {}

        # Forward propagation
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            if cancel_check and cancel_check():
                raise PropagationCancelled()
            orig_idx = idx_map[frame_idx]
            propagated[orig_idx] = (mask_logits[0] > 0.0).cpu().numpy().squeeze()

        # Backward propagation
        if sam2_anchor > 0:
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                if cancel_check and cancel_check():
                    raise PropagationCancelled()
                orig_idx = idx_map[frame_idx]
                propagated[orig_idx] = (mask_logits[0] > 0.0).cpu().numpy().squeeze()

    # Save masks for ALL frames, using nearest propagated mask for skipped frames
    propagated_indices = sorted(propagated.keys())
    for i in range(total):
        if cancel_check and cancel_check():
            raise PropagationCancelled()
        mask_path = masks_dir / f"mask_{i + 1:04d}.png"
        if i in propagated:
            mask_data = propagated[i]
        else:
            nearest = min(propagated_indices, key=lambda x: abs(x - i))
            mask_data = propagated[nearest]
        mask_img = (mask_data.astype(np.uint8)) * 255
        Image.fromarray(mask_img).save(mask_path)

    return total
