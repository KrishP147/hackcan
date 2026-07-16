import os
import threading
from contextlib import contextmanager

import torch

USE_SYNTH = os.getenv("FRAMESHIFT_USE_SYNTH", "0") == "1"        # Backend B (Doc 2)
USE_CUDA_KERNEL = os.getenv("FRAMESHIFT_USE_CUDA_KERNEL", "0") == "1"

# SAM2 and RAFT are both memory-hungry. Keep one process-wide GPU queue so an
# API request cannot start RAFT while SAM2 is still tracking another project.
# Modal is initially configured with one container, so this lock serializes all
# heavy neural stages in the deployment.
_GPU_JOB_LOCK = threading.Lock()


@contextmanager
def gpu_job(stage: str):
    """Serialize a named GPU stage within the backend process."""
    print(f"[GPU queue] waiting: {stage}")
    with _GPU_JOB_LOCK:
        print(f"[GPU queue] running: {stage}")
        try:
            yield
        finally:
            print(f"[GPU queue] finished: {stage}")


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


def should_precompute_flows() -> bool:
    """Whether extraction should immediately schedule full-clip RAFT.

    Production defaults to lazy flow computation so the intended order is
    keyframe SAM2 -> confirmed SAM2 tracking -> RAFT -> propagation.
    """
    return os.getenv("FRAMESHIFT_PRECOMPUTE_FLOWS", "0") == "1"


def get_raft_batch_size() -> int:
    """Number of adjacent frame pairs evaluated per RAFT forward pass."""
    return _int_env("FRAMESHIFT_RAFT_BATCH_SIZE", 1, 1)


def _int_env(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def get_synth_patch_radius() -> int:
    return _int_env("FRAMESHIFT_SYNTH_PATCH_RADIUS", 2, 0)


def get_synth_iterations() -> int:
    return _int_env("FRAMESHIFT_SYNTH_ITERATIONS", 4, 1)


def get_synth_pyramid_levels() -> int:
    return _int_env("FRAMESHIFT_SYNTH_PYRAMID_LEVELS", 4, 1)


def get_synth_max_working_size() -> int:
    return _int_env("FRAMESHIFT_SYNTH_MAX_SIZE", 256, 32)


def get_synth_seed() -> int:
    return _int_env("FRAMESHIFT_SYNTH_SEED", 0, 0)
