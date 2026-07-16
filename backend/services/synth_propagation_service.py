"""Backend B — Ezsynth-inspired guided PatchMatch synthesis.

One generated anchor is matched in the pristine/original guide domain and its
edited pixels are voted into every target.  The portable PyTorch reference is
always available; the optional CUDA extension accelerates the same NNF/voting
operations and falls back cleanly when it cannot be imported.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from services import (config, flow_service, guided_synthesis, mask_service,
                      replace_tool)
from services.propagation_service import iter_chain


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Generated anchor could not be decoded")
    return image


async def _default_generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
    from services import gemini_service

    return await gemini_service.edit_frame_with_reference(
        frame_path,
        prompt,
        reference_frame_path=reference_frame_path,
        mask_path=mask_path,
    )


def _identity_seed(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack((xx, yy), axis=-1)


def _seed_from_flow(flow_to_anchor: torch.Tensor, height: int, width: int) -> np.ndarray:
    flow = flow_to_anchor[0].detach().to(dtype=torch.float32, device="cpu").permute(1, 2, 0).numpy()
    if flow.shape[:2] != (height, width):
        old_h, old_w = flow.shape[:2]
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= width / max(old_w, 1)
        flow[..., 1] *= height / max(old_h, 1)
    seed = _identity_seed(height, width) + flow
    seed[..., 0] = np.clip(seed[..., 0], 0, width - 1)
    seed[..., 1] = np.clip(seed[..., 1], 0, height - 1)
    return seed


def _load_mask(masks_dir: Path, index: int) -> np.ndarray:
    mask = cv2.imread(str(masks_dir / f"mask_{index:04d}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask {index} is missing")
    return mask


def _warp_temporal(
    previous: np.ndarray,
    flows_dir: Path,
    target_index: int,
    direction: int,
    device: torch.device,
) -> np.ndarray | None:
    pair_index = target_index - 1 if direction > 0 else target_index
    flow_direction = "bwd" if direction > 0 else "fwd"
    flow_path = flows_dir / f"flow_{flow_direction}_{pair_index:04d}.npy"
    if not flow_path.exists():
        return None
    flow = flow_service.load_flow(flows_dir, pair_index, flow_direction).to(device)
    source = torch.from_numpy(previous.astype(np.float32)).permute(2, 0, 1)[None].to(device)
    warped = flow_service.warp(source, flow)[0].permute(1, 2, 0)
    return warped.clamp(0, 255).byte().cpu().numpy()


def _patchmatch_config() -> guided_synthesis.PatchMatchConfig:
    return guided_synthesis.PatchMatchConfig(
        patch_radius=config.get_synth_patch_radius(),
        iterations=config.get_synth_iterations(),
        pyramid_levels=config.get_synth_pyramid_levels(),
        max_working_size=config.get_synth_max_working_size(),
        seed=config.get_synth_seed(),
    )


def _composite(target: np.ndarray, synthesized: np.ndarray, masks_dir: Path, index: int) -> np.ndarray:
    alpha = mask_service.load_mask_alpha(masks_dir, index, feather_px=5)[..., None]
    output = alpha * synthesized.astype(np.float32) + (1.0 - alpha) * target.astype(np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


async def _synthesize_target(
    source_edit: np.ndarray,
    source_original: np.ndarray,
    source_mask: np.ndarray,
    target_original: np.ndarray,
    target_mask: np.ndarray,
    positional_seed: np.ndarray,
    temporal_target: np.ndarray | None,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    return await asyncio.to_thread(
        guided_synthesis.synthesize,
        source_edit,
        source_original,
        target_original,
        source_mask,
        target_mask,
        positional_seed,
        temporal_target,
        device=device,
        config=_patchmatch_config(),
        use_cuda_kernel=config.USE_CUDA_KERNEL,
    )


async def apply_replace_range(
    project_dir: Path,
    frame_indices,
    anchor_index: int,
    prompt: str,
    device: torch.device,
    generate=None,
) -> None:
    """Generate once at ``anchor_index`` and guided-synthesize every target.

    Results are staged and committed only after the full range succeeds.  Any
    failure leaves pristine frames untouched and invokes Backend A, preserving
    Document 2's mandatory product-level fallback.
    """
    generate = generate or _default_generate
    project_dir = Path(project_dir)
    frames_dir = project_dir / "frames"
    masks_dir = project_dir / "masks"
    flows_dir = project_dir / "flows"
    indices = sorted(set(int(index) for index in frame_indices))
    if not indices:
        return
    if anchor_index not in indices:
        anchor_index = min(indices, key=lambda index: abs(index - anchor_index))

    anchor_frame_path = frames_dir / f"frame_{anchor_index:04d}.jpg"
    anchor_mask_path = masks_dir / f"mask_{anchor_index:04d}.png"
    source_original = cv2.imread(str(anchor_frame_path))
    if source_original is None:
        raise FileNotFoundError(f"Anchor frame {anchor_index} is missing")
    source_mask = _load_mask(masks_dir, anchor_index)
    anchor_bytes = await generate(
        anchor_frame_path,
        prompt,
        reference_frame_path=None,
        mask_path=anchor_mask_path,
    )
    source_edit = _decode(anchor_bytes)

    # Reuse the already-generated anchor if the mandatory Backend A fallback is
    # needed; never pay for a second independent draw.
    first_generation = True

    async def cached_generate(frame_path, gen_prompt, reference_frame_path=None, mask_path=None):
        nonlocal first_generation
        if first_generation and reference_frame_path is None:
            first_generation = False
            return anchor_bytes
        return await generate(
            frame_path,
            gen_prompt,
            reference_frame_path=reference_frame_path,
            mask_path=mask_path,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="frameshift-synth-") as tmp:
            staging = Path(tmp)
            anchor_output = _composite(source_original, source_edit, masks_dir, anchor_index)
            cv2.imwrite(str(staging / f"frame_{anchor_index:04d}.jpg"), anchor_output,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            height, width = source_original.shape[:2]
            metadata = {"backend": "anchor"}
            for direction in (1, -1):
                targets = [index for index in indices if (index - anchor_index) * direction > 0]
                if not targets:
                    continue
                if direction < 0:
                    targets.reverse()
                target_set = set(targets)
                previous_output = anchor_output
                max_distance = max(abs(index - anchor_index) for index in targets)
                for index, flow_to_anchor, _ in iter_chain(
                    flows_dir, anchor_index, direction, max_distance, device
                ):
                    if index not in target_set:
                        continue
                    target_path = frames_dir / f"frame_{index:04d}.jpg"
                    target_original = cv2.imread(str(target_path))
                    if target_original is None:
                        raise FileNotFoundError(f"Target frame {index} is missing")
                    target_mask = _load_mask(masks_dir, index)
                    seed = _seed_from_flow(flow_to_anchor, height, width)
                    temporal_target = _warp_temporal(
                        previous_output, flows_dir, index, direction, device
                    )
                    synthesized, metadata = await _synthesize_target(
                        source_edit,
                        source_original,
                        source_mask,
                        target_original,
                        target_mask,
                        seed,
                        temporal_target,
                        device,
                    )
                    output = _composite(target_original, synthesized, masks_dir, index)
                    cv2.imwrite(str(staging / f"frame_{index:04d}.jpg"), output,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    previous_output = output

            for index in indices:
                staged = staging / f"frame_{index:04d}.jpg"
                if not staged.exists():
                    raise RuntimeError(f"Backend B did not synthesize frame {index}")
            for index in indices:
                shutil.copy2(staging / f"frame_{index:04d}.jpg",
                             frames_dir / f"frame_{index:04d}.jpg")
            print(
                f"[synth] Backend B complete: {len(indices)} frames, "
                f"engine={metadata['backend']}, working_size={metadata.get('working_size')}"
            )
    except Exception as error:
        print(f"[synth] Backend B failed ({error}); falling back to Backend A")
        await replace_tool.apply_replace_range(
            project_dir,
            indices,
            anchor_index,
            prompt,
            device,
            generate=cached_generate,
        )
