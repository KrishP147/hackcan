"""Optional CUDA extension for FrameShift Backend B.

Build from ``backend`` with::

    venv/bin/python ebsynth_synth/setup.py build_ext --inplace

Importing this package is always safe.  A missing or ABI-incompatible extension
reports unavailable and lets the product use the PyTorch reference.
"""
from __future__ import annotations

try:
    from . import _C
except (ImportError, OSError, RuntimeError):
    _C = None


def is_available() -> bool:
    return _C is not None


def patchmatch(source, target, weights, initial_nnf, patch_radius, iterations, seed):
    if _C is None:
        raise RuntimeError("ebsynth_synth CUDA extension is not built")
    return _C.patchmatch(
        source, target, weights, initial_nnf, patch_radius, iterations, seed
    )


def vote(source_edit, nnf, patch_radius):
    if _C is None:
        raise RuntimeError("ebsynth_synth CUDA extension is not built")
    return _C.vote(source_edit, nnf, patch_radius)


def synthesize(source_edit, source, target, weights, initial_nnf,
               patch_radius=2, iterations=4, seed=0):
    nnf, cost = patchmatch(
        source, target, weights, initial_nnf, patch_radius, iterations, seed
    )
    return vote(source_edit, nnf, patch_radius), nnf, cost
