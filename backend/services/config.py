import os

import torch

USE_SYNTH = os.getenv("FRAMESHIFT_USE_SYNTH", "0") == "1"        # Backend B (Doc 2)
USE_CUDA_KERNEL = os.getenv("FRAMESHIFT_USE_CUDA_KERNEL", "0") == "1"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_mask_frame_step() -> int:
    """Return the full-quality SAM2 stride, with an explicit dev override."""
    configured = os.getenv("FRAMESHIFT_MASK_FRAME_STEP")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    # Production always tracks every frame. A stride is available only as an
    # explicit local preview override; it must never be selected implicitly.
    return 1
