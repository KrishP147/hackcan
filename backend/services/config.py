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
