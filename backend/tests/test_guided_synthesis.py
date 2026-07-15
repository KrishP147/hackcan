import math

import numpy as np
import pytest
import torch

from services import guided_synthesis


def test_radius_zero_vote_is_exact_nnf_gather():
    source = torch.arange(3 * 6 * 8, dtype=torch.float32).reshape(3, 6, 8)
    nnf = guided_synthesis.identity_nnf(6, 8, torch.device("cpu"))
    nnf[..., 0] = (nnf[..., 0] - 1).clamp_min(0)

    voted = guided_synthesis.vote_reference(source, nnf, radius=0)
    expected = source[:, :, torch.arange(8).sub(1).clamp_min(0)]

    assert torch.equal(voted, expected)


def test_patchmatch_reference_is_deterministic():
    generator = torch.Generator().manual_seed(7)
    source = torch.rand((5, 12, 16), generator=generator)
    target = torch.roll(source, shifts=2, dims=2)
    weights = torch.ones(5)
    initial = guided_synthesis.identity_nnf(12, 16, torch.device("cpu"))

    first_nnf, first_cost = guided_synthesis.patchmatch_reference(
        source, target, weights, initial, radius=1, iterations=2, seed=123)
    second_nnf, second_cost = guided_synthesis.patchmatch_reference(
        source, target, weights, initial, radius=1, iterations=2, seed=123)

    assert torch.equal(first_nnf, second_nnf)
    assert torch.equal(first_cost, second_cost)


def test_guided_synthesis_carries_one_edited_appearance_with_flow_seed():
    height = width = 32
    source = np.zeros((height, width, 3), dtype=np.uint8)
    texture = np.indices((12, 10)).sum(axis=0) % 2
    source[9:21, 5:15] = np.where(texture[..., None] == 0,
                                  np.array([30, 160, 220], dtype=np.uint8),
                                  np.array([220, 80, 20], dtype=np.uint8))
    edited = source.copy()
    edited[9:21, 5:15] = np.where(texture[..., None] == 0,
                                  np.array([10, 10, 245], dtype=np.uint8),
                                  np.array([20, 220, 40], dtype=np.uint8))

    shift = 7
    target = np.zeros_like(source)
    target[:, shift:] = source[:, :-shift]
    source_mask = np.zeros((height, width), dtype=np.uint8)
    target_mask = np.zeros_like(source_mask)
    source_mask[9:21, 5:15] = 255
    target_mask[9:21, 5 + shift:15 + shift] = 255
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    seed = np.stack((np.clip(xx - shift, 0, width - 1), yy), axis=-1)

    result, metadata = guided_synthesis.synthesize(
        edited,
        source,
        target,
        source_mask,
        target_mask,
        seed,
        device=torch.device("cpu"),
        config=guided_synthesis.PatchMatchConfig(
            patch_radius=1,
            iterations=2,
            pyramid_levels=2,
            max_working_size=32,
            seed=11,
        ),
    )

    expected = np.zeros_like(edited)
    expected[:, shift:] = edited[:, :-shift]
    region = target_mask > 0
    mse = np.mean((result[region].astype(np.float32) - expected[region].astype(np.float32)) ** 2)
    psnr = math.inf if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))
    assert psnr > 18.0
    assert metadata["backend"] == "torch"


def test_optional_cuda_vote_matches_radius_zero_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    try:
        import ebsynth_synth
    except ImportError:
        pytest.skip("CUDA extension is not built")
    if not ebsynth_synth.is_available():
        pytest.skip("CUDA extension is not built")

    source = torch.rand((3, 16, 20), device="cuda")
    nnf = guided_synthesis.identity_nnf(16, 20, torch.device("cuda"))
    nnf[..., 0] = (nnf[..., 0] - 2).clamp_min(0)
    expected = guided_synthesis.vote_reference(source, nnf, radius=0)
    actual = ebsynth_synth.vote(source, nnf, 0)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_quality_metrics_are_exact_for_identical_images():
    image = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
    assert math.isinf(guided_synthesis.psnr(image, image))
    assert guided_synthesis.ssim(image, image) == pytest.approx(1.0, abs=1e-6)


def test_optional_cuda_synthesis_matches_reference_quality():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    try:
        import ebsynth_synth
    except ImportError:
        pytest.skip("CUDA extension is not built")
    if not ebsynth_synth.is_available():
        pytest.skip("CUDA extension is not built")

    height, width = 24, 32
    rng = np.random.default_rng(5)
    source = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    edited = np.clip(source.astype(np.int16) + np.array([15, -10, 20]), 0, 255).astype(np.uint8)
    target = np.roll(source, 2, axis=1)
    mask = np.full((height, width), 255, dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    seed = np.stack((np.clip(xx - 2, 0, width - 1), yy), axis=-1)
    cfg = guided_synthesis.PatchMatchConfig(
        patch_radius=1, iterations=2, pyramid_levels=2,
        max_working_size=32, seed=17)

    reference, _ = guided_synthesis.synthesize(
        edited, source, target, mask, mask, seed,
        device=torch.device("cuda"), config=cfg, use_cuda_kernel=False)
    accelerated, metadata = guided_synthesis.synthesize(
        edited, source, target, mask, mask, seed,
        device=torch.device("cuda"), config=cfg, use_cuda_kernel=True)

    assert metadata["backend"] == "cuda"
    assert guided_synthesis.psnr(reference, accelerated) > 18.0
    assert guided_synthesis.ssim(reference, accelerated) > 0.75
