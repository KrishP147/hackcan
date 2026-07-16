"""Guided PatchMatch synthesis used by Backend B.

The implementation follows docs/frameshift-cuda-track.md Track B:

* correspondences are found in the original/guide domain;
* pixels are always copied from the pristine edited anchor;
* jump-flood propagation removes scan-order dependencies;
* random search is deterministic for a fixed seed;
* a coarse-to-fine pyramid carries the NNF between resolutions; and
* reconstruction uses overlapping patch voting.

The PyTorch implementation is the portable correctness reference.  When the
optional ``ebsynth_synth`` CUDA extension is available, the same pyramid host
loop dispatches NNF search and voting to the custom kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchMatchConfig:
    patch_radius: int = 2
    iterations: int = 4
    pyramid_levels: int = 4
    max_working_size: int = 256
    seed: int = 0
    appearance_weight: float = 1.0
    edge_weight: float = 2.0
    position_weight: float = 0.35
    mask_weight: float = 4.0
    temporal_weight: float = 0.75


@dataclass(frozen=True)
class GuidePyramid:
    source: list[torch.Tensor]
    target: list[torch.Tensor]
    weights: torch.Tensor
    seed_nnf: list[torch.Tensor]


def _as_float_chw(image: np.ndarray, device: torch.device) -> torch.Tensor:
    if image.ndim == 2:
        image = image[..., None]
    arr = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32) / 255.0


def _edge_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    scale = float(np.percentile(magnitude, 99)) if magnitude.size else 0.0
    if scale > 1e-6:
        magnitude = magnitude / scale
    return np.clip(magnitude, 0.0, 1.0)[..., None]


def identity_nnf(height: int, width: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1).to(torch.int32)


def _resize_nnf(nnf: torch.Tensor, height: int, width: int) -> torch.Tensor:
    old_h, old_w = nnf.shape[:2]
    if old_h == height and old_w == width:
        return nnf.to(torch.int32)
    channels = nnf.permute(2, 0, 1)[None].to(torch.float32)
    resized = F.interpolate(channels, size=(height, width), mode="bilinear", align_corners=False)[0]
    resized[0] *= (width - 1) / max(old_w - 1, 1)
    resized[1] *= (height - 1) / max(old_h - 1, 1)
    resized[0].clamp_(0, width - 1)
    resized[1].clamp_(0, height - 1)
    return resized.round().permute(1, 2, 0).to(torch.int32)


def _working_shape(height: int, width: int, max_size: int) -> tuple[int, int]:
    longest = max(height, width)
    if longest <= max_size:
        return height, width
    scale = max_size / longest
    return max(8, round(height * scale)), max(8, round(width * scale))


def _pyramid_shapes(height: int, width: int, levels: int) -> list[tuple[int, int]]:
    shapes = []
    for exponent in range(max(levels, 1) - 1, -1, -1):
        divisor = 2**exponent
        shape = (max(8, round(height / divisor)), max(8, round(width / divisor)))
        if not shapes or shape != shapes[-1]:
            shapes.append(shape)
    return shapes


def _resize_feature(feature: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(feature[None], size=shape, mode="bilinear", align_corners=False)[0]


def build_guide_pyramid(
    source_original: np.ndarray,
    target_original: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    positional_seed: np.ndarray | None,
    temporal_source: np.ndarray | None,
    temporal_target: np.ndarray | None,
    device: torch.device,
    config: PatchMatchConfig,
) -> GuidePyramid:
    """Build appearance, edge, positional, mask, and optional temporal guides.

    ``positional_seed[y, x]`` stores the full-resolution anchor coordinate that
    target pixel ``(x, y)`` should sample according to composed optical flow.
    """
    height, width = target_original.shape[:2]
    work_h, work_w = _working_shape(height, width, config.max_working_size)

    src_parts = [_as_float_chw(source_original, device)]
    tgt_parts = [_as_float_chw(target_original, device)]
    channel_weights = [config.appearance_weight] * 3

    src_parts.append(_as_float_chw((_edge_map(source_original) * 255).astype(np.uint8), device))
    tgt_parts.append(_as_float_chw((_edge_map(target_original) * 255).astype(np.uint8), device))
    channel_weights.append(config.edge_weight)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    source_position = np.stack(
        (xx / max(width - 1, 1), yy / max(height - 1, 1)), axis=-1
    )
    if positional_seed is None:
        target_position = source_position
        positional_seed = np.stack((xx, yy), axis=-1)
    else:
        target_position = positional_seed.astype(np.float32).copy()
        target_position[..., 0] /= max(width - 1, 1)
        target_position[..., 1] /= max(height - 1, 1)
        target_position = np.clip(target_position, 0.0, 1.0)
    src_parts.append(torch.from_numpy(source_position.transpose(2, 0, 1)).to(device))
    tgt_parts.append(torch.from_numpy(target_position.transpose(2, 0, 1)).to(device))
    channel_weights.extend([config.position_weight, config.position_weight])

    src_mask = (source_mask > 0).astype(np.float32)[None]
    tgt_mask = (target_mask > 0).astype(np.float32)[None]
    src_parts.append(torch.from_numpy(src_mask).to(device))
    tgt_parts.append(torch.from_numpy(tgt_mask).to(device))
    channel_weights.append(config.mask_weight)

    if temporal_source is not None and temporal_target is not None:
        src_parts.append(_as_float_chw(temporal_source, device))
        tgt_parts.append(_as_float_chw(temporal_target, device))
        channel_weights.extend([config.temporal_weight] * 3)

    source_full = torch.cat(src_parts, dim=0).to(torch.float32)
    target_full = torch.cat(tgt_parts, dim=0).to(torch.float32)
    source_work = _resize_feature(source_full, (work_h, work_w))
    target_work = _resize_feature(target_full, (work_h, work_w))

    seed = torch.from_numpy(np.ascontiguousarray(positional_seed)).to(
        device=device, dtype=torch.float32
    )
    seed[..., 0] *= work_w / max(width, 1)
    seed[..., 1] *= work_h / max(height, 1)
    seed = F.interpolate(
        seed.permute(2, 0, 1)[None],
        size=(work_h, work_w),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    seed[..., 0].clamp_(0, work_w - 1)
    seed[..., 1].clamp_(0, work_h - 1)
    seed = seed.round().to(torch.int32)

    shapes = _pyramid_shapes(work_h, work_w, config.pyramid_levels)
    source_levels = [_resize_feature(source_work, shape) for shape in shapes]
    target_levels = [_resize_feature(target_work, shape) for shape in shapes]
    seed_levels = [_resize_nnf(seed, *shape) for shape in shapes]
    weights = torch.tensor(channel_weights, dtype=torch.float32, device=device)
    return GuidePyramid(source_levels, target_levels, weights, seed_levels)


def _shifted_target(feature: torch.Tensor, dx: int, dy: int) -> tuple[torch.Tensor, torch.Tensor]:
    channels, height, width = feature.shape
    shifted = torch.zeros_like(feature)
    valid = torch.zeros((height, width), dtype=torch.bool, device=feature.device)

    tx0, tx1 = max(0, -dx), min(width, width - dx)
    ty0, ty1 = max(0, -dy), min(height, height - dy)
    if tx0 >= tx1 or ty0 >= ty1:
        return shifted, valid
    shifted[:, ty0:ty1, tx0:tx1] = feature[:, ty0 + dy:ty1 + dy, tx0 + dx:tx1 + dx]
    valid[ty0:ty1, tx0:tx1] = True
    return shifted, valid


def patch_cost(
    source: torch.Tensor,
    target: torch.Tensor,
    candidates: torch.Tensor,
    weights: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Exact integer-coordinate patch SSD for one or more candidate NNFs.

    ``candidates`` is ``(K,H,W,2)`` and the result is ``(K,H,W)``.
    """
    if candidates.ndim == 3:
        candidates = candidates[None]
    count, height, width, _ = candidates.shape
    channels = source.shape[0]
    src_flat = source.reshape(channels, -1)
    costs = torch.zeros((count, height, width), device=source.device, dtype=torch.float32)
    valid_votes = torch.zeros_like(costs)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sx = candidates[..., 0].to(torch.int64) + dx
            sy = candidates[..., 1].to(torch.int64) + dy
            src_valid = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
            src_index = sy.clamp(0, height - 1) * width + sx.clamp(0, width - 1)
            sampled = src_flat[:, src_index.reshape(-1)].reshape(channels, count, height, width)
            shifted_target, target_valid = _shifted_target(target, dx, dy)
            valid = src_valid & target_valid[None]
            diff = sampled - shifted_target[:, None]
            weighted = (diff.square() * weights[:, None, None, None]).sum(dim=0)
            costs += torch.where(valid, weighted, torch.full_like(weighted, 10.0))
            valid_votes += valid

    return costs / valid_votes.clamp_min(1.0)


def _neighbor_candidate(nnf: torch.Tensor, dx: int, dy: int) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = nnf.shape[:2]
    candidate = torch.roll(nnf, shifts=(-dy, -dx), dims=(0, 1)).clone()
    candidate[..., 0] -= dx
    candidate[..., 1] -= dy

    valid = torch.ones((height, width), dtype=torch.bool, device=nnf.device)
    if dx > 0:
        valid[:, width - dx:] = False
    elif dx < 0:
        valid[:, : -dx] = False
    if dy > 0:
        valid[height - dy:, :] = False
    elif dy < 0:
        valid[: -dy, :] = False
    return candidate, valid


def _shift_nnf(nnf: torch.Tensor, dx: int, dy: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``nnf[p + (dx,dy)]`` at every target pixel ``p``."""
    height, width = nnf.shape[:2]
    shifted = torch.roll(nnf, shifts=(-dy, -dx), dims=(0, 1))
    valid = torch.ones((height, width), dtype=torch.bool, device=nnf.device)
    if dx > 0:
        valid[:, width - dx:] = False
    elif dx < 0:
        valid[:, : -dx] = False
    if dy > 0:
        valid[height - dy:, :] = False
    elif dy < 0:
        valid[: -dy, :] = False
    return shifted, valid


def _choose_best(
    source: torch.Tensor,
    target: torch.Tensor,
    nnf: torch.Tensor,
    candidates: Iterable[tuple[torch.Tensor, torch.Tensor]],
    weights: torch.Tensor,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = nnf.shape[:2]
    all_candidates = [nnf]
    all_valid = [torch.ones((height, width), dtype=torch.bool, device=nnf.device)]
    for candidate, valid in candidates:
        all_candidates.append(candidate)
        all_valid.append(valid)
    stacked = torch.stack(all_candidates)
    valid = torch.stack(all_valid)
    valid &= (
        (stacked[..., 0] >= 0)
        & (stacked[..., 0] < width)
        & (stacked[..., 1] >= 0)
        & (stacked[..., 1] < height)
    )
    costs = patch_cost(source, target, stacked, weights, radius)
    costs = torch.where(valid, costs, torch.full_like(costs, torch.inf))
    best_cost, best_idx = costs.min(dim=0)
    selected = torch.gather(
        stacked,
        0,
        best_idx[None, ..., None].expand(1, height, width, 2),
    )[0]
    return selected.to(torch.int32), best_cost


def _jump_flood(
    source: torch.Tensor,
    target: torch.Tensor,
    nnf: torch.Tensor,
    weights: torch.Tensor,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = nnf.shape[:2]
    step = 1 << max(0, math.ceil(math.log2(max(height, width))) - 1)
    best_cost = patch_cost(source, target, nnf, weights, radius)[0]
    while step >= 1:
        offsets = (
            (-step, 0), (step, 0), (0, -step), (0, step),
            (-step, -step), (step, step), (-step, step), (step, -step),
        )
        nnf, best_cost = _choose_best(
            source,
            target,
            nnf,
            (_neighbor_candidate(nnf, dx, dy) for dx, dy in offsets),
            weights,
            radius,
        )
        step //= 2
    return nnf, best_cost


def _random_search(
    source: torch.Tensor,
    target: torch.Tensor,
    nnf: torch.Tensor,
    weights: torch.Tensor,
    radius: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = nnf.shape[:2]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    search_radius = max(height, width)
    best_cost = patch_cost(source, target, nnf, weights, radius)[0]
    while search_radius >= 1:
        jitter = torch.randint(
            -search_radius,
            search_radius + 1,
            (height, width, 2),
            generator=generator,
            dtype=torch.int32,
        ).to(nnf.device)
        candidate = nnf + jitter
        candidate[..., 0].clamp_(0, width - 1)
        candidate[..., 1].clamp_(0, height - 1)
        nnf, best_cost = _choose_best(
            source,
            target,
            nnf,
            ((candidate, torch.ones((height, width), dtype=torch.bool, device=nnf.device)),),
            weights,
            radius,
        )
        search_radius //= 2
    return nnf, best_cost


def patchmatch_reference(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    initial_nnf: torch.Tensor,
    radius: int,
    iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nnf = initial_nnf.to(device=source.device, dtype=torch.int32).clone()
    height, width = nnf.shape[:2]
    nnf[..., 0].clamp_(0, width - 1)
    nnf[..., 1].clamp_(0, height - 1)
    cost = patch_cost(source, target, nnf, weights, radius)[0]
    for iteration in range(max(iterations, 1)):
        nnf, cost = _jump_flood(source, target, nnf, weights, radius)
        nnf, cost = _random_search(
            source, target, nnf, weights, radius, seed + iteration * 104729
        )
    return nnf, cost


def vote_reference(source_edit: torch.Tensor, nnf: torch.Tensor, radius: int) -> torch.Tensor:
    """Reconstruct a target by averaging source pixels from overlapping patches."""
    channels, height, width = source_edit.shape
    src_flat = source_edit.reshape(channels, -1)
    output = torch.zeros_like(source_edit)
    weight_sum = torch.zeros((height, width), dtype=torch.float32, device=source_edit.device)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            center, center_valid = _shift_nnf(nnf, dx, dy)
            sx = center[..., 0].to(torch.int64) - dx
            sy = center[..., 1].to(torch.int64) - dy
            valid = center_valid & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
            index = sy.clamp(0, height - 1) * width + sx.clamp(0, width - 1)
            sample = src_flat[:, index.reshape(-1)].reshape(channels, height, width)
            output += sample * valid[None]
            weight_sum += valid

    return output / weight_sum.clamp_min(1.0)[None]


def _load_cuda_extension():
    try:
        import ebsynth_synth

        if ebsynth_synth.is_available():
            return ebsynth_synth
    except (ImportError, OSError, RuntimeError):
        pass
    return None


def solve_nnf(
    guides: GuidePyramid,
    config: PatchMatchConfig,
    use_cuda_kernel: bool,
) -> tuple[torch.Tensor, bool]:
    extension = _load_cuda_extension() if use_cuda_kernel else None
    nnf = None
    used_cuda = extension is not None

    for level, (source, target, positional_seed) in enumerate(
        zip(guides.source, guides.target, guides.seed_nnf)
    ):
        initial = positional_seed if nnf is None else _resize_nnf(nnf, *source.shape[-2:])
        # Let the flow seed compete with the upsampled coarse solution.
        if nnf is not None:
            initial, _ = _choose_best(
                source,
                target,
                initial,
                ((positional_seed, torch.ones(source.shape[-2:], dtype=torch.bool, device=source.device)),),
                guides.weights,
                min(config.patch_radius, max(0, min(source.shape[-2:]) // 4)),
            )
        level_radius = min(config.patch_radius, max(0, min(source.shape[-2:]) // 4))
        if extension is not None:
            nnf, _ = extension.patchmatch(
                source.contiguous(),
                target.contiguous(),
                guides.weights.contiguous(),
                initial.contiguous(),
                level_radius,
                config.iterations,
                config.seed + level * 8191,
            )
        else:
            nnf, _ = patchmatch_reference(
                source,
                target,
                guides.weights,
                initial,
                level_radius,
                config.iterations,
                config.seed + level * 8191,
            )
    assert nnf is not None
    return nnf, used_cuda


def synthesize(
    source_edit: np.ndarray,
    source_original: np.ndarray,
    target_original: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    positional_seed: np.ndarray | None = None,
    temporal_target: np.ndarray | None = None,
    *,
    device: torch.device | None = None,
    config: PatchMatchConfig | None = None,
    use_cuda_kernel: bool = False,
) -> tuple[np.ndarray, dict]:
    """Synthesize one target frame from a single pristine edited anchor."""
    config = config or PatchMatchConfig()
    device = device or torch.device("cpu")
    if use_cuda_kernel and device.type != "cuda":
        use_cuda_kernel = False

    temporal_source = source_edit if temporal_target is not None else None
    guides = build_guide_pyramid(
        source_original,
        target_original,
        source_mask,
        target_mask,
        positional_seed,
        temporal_source,
        temporal_target,
        device,
        config,
    )
    nnf, used_cuda = solve_nnf(guides, config, use_cuda_kernel)

    full_h, full_w = target_original.shape[:2]
    full_nnf = _resize_nnf(nnf, full_h, full_w)
    edit_tensor = _as_float_chw(source_edit, device)
    extension = _load_cuda_extension() if used_cuda else None
    if extension is not None:
        voted = extension.vote(edit_tensor.contiguous(), full_nnf.contiguous(), config.patch_radius)
    else:
        voted = vote_reference(edit_tensor, full_nnf, config.patch_radius)
    image = voted.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return image, {
        "backend": "cuda" if used_cuda else "torch",
        "working_size": list(nnf.shape[:2]),
        "patch_radius": config.patch_radius,
        "iterations": config.iterations,
        "pyramid_levels": len(guides.source),
    }


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Peak signal-to-noise ratio used for CPU/CUDA quality validation."""
    delta = reference.astype(np.float32) - candidate.astype(np.float32)
    mse = float(np.mean(delta * delta))
    if mse <= 1e-12:
        return math.inf
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean local SSIM without adding a deployment dependency on scikit-image."""
    ref = reference.astype(np.float32)
    cand = candidate.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    scores = []
    for channel in range(ref.shape[2] if ref.ndim == 3 else 1):
        x = ref[..., channel] if ref.ndim == 3 else ref
        y = cand[..., channel] if cand.ndim == 3 else cand
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        scores.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(scores))
