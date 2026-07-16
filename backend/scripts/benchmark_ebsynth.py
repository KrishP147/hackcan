"""Benchmark and validate the guided-synthesis CPU/PyTorch and CUDA paths.

Usage from ``backend``::

    venv/bin/python scripts/benchmark_ebsynth.py \
        --source original_anchor.jpg --edited edited_anchor.jpg \
        --target original_target.jpg --source-mask anchor_mask.png \
        --target-mask target_mask.png

The positional seed defaults to identity. Pass ``--flow-to-anchor`` with a
``(2,H,W)`` NumPy flow file to benchmark motion-guided synthesis.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import guided_synthesis  # noqa: E402


def _image(path: Path, grayscale: bool = False) -> np.ndarray:
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise FileNotFoundError(path)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--edited", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-mask", type=Path, required=True)
    parser.add_argument("--target-mask", type=Path, required=True)
    parser.add_argument("--flow-to-anchor", type=Path)
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--patch-radius", type=int, default=2)
    args = parser.parse_args()

    source = _image(args.source)
    edited = _image(args.edited)
    target = _image(args.target)
    source_mask = _image(args.source_mask, grayscale=True)
    target_mask = _image(args.target_mask, grayscale=True)
    height, width = target.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    seed = np.stack((xx, yy), axis=-1)
    if args.flow_to_anchor:
        flow = np.load(args.flow_to_anchor).astype(np.float32)
        if flow.shape[0] == 2:
            flow = flow.transpose(1, 2, 0)
        seed += flow
    seed[..., 0] = np.clip(seed[..., 0], 0, width - 1)
    seed[..., 1] = np.clip(seed[..., 1], 0, height - 1)
    cfg = guided_synthesis.PatchMatchConfig(
        patch_radius=args.patch_radius,
        iterations=args.iterations,
        max_working_size=args.max_size,
    )

    reference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    reference, ref_meta = guided_synthesis.synthesize(
        edited, source, target, source_mask, target_mask, seed,
        device=reference_device, config=cfg, use_cuda_kernel=False)
    if reference_device.type == "cuda":
        torch.cuda.synchronize()
    ref_seconds = time.perf_counter() - start
    print(f"reference: {ref_seconds:.3f}s {ref_meta}")

    try:
        import ebsynth_synth
        cuda_ready = torch.cuda.is_available() and ebsynth_synth.is_available()
    except ImportError:
        cuda_ready = False
    if not cuda_ready:
        print("cuda: unavailable (build ebsynth_synth first)")
        return

    start = time.perf_counter()
    accelerated, cuda_meta = guided_synthesis.synthesize(
        edited, source, target, source_mask, target_mask, seed,
        device=torch.device("cuda"), config=cfg, use_cuda_kernel=True)
    torch.cuda.synchronize()
    cuda_seconds = time.perf_counter() - start
    print(f"cuda: {cuda_seconds:.3f}s {cuda_meta}")
    print(f"speedup: {ref_seconds / max(cuda_seconds, 1e-9):.2f}x")
    print(f"PSNR vs reference: {guided_synthesis.psnr(reference, accelerated):.3f} dB")
    print(f"SSIM vs reference: {guided_synthesis.ssim(reference, accelerated):.5f}")


if __name__ == "__main__":
    main()
