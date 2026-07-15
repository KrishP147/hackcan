"""Backend A star-warp propagation (Doc 1 §2). Carry ONE appearance:
compose pairwise flow into a single A→t displacement, sample the pristine
anchor edit layer exactly once per target frame — nothing to morph between."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from services import flow_service


@dataclass
class EditLayer:
    rgb: np.ndarray        # (H,W,3) uint8
    alpha: np.ndarray      # (H,W) float32 [0,1]
    validity: np.ndarray   # (H,W) float32 [0,1]


def make_anchor_layer(edited_frame: np.ndarray, mask_alpha: np.ndarray) -> EditLayer:
    return EditLayer(rgb=edited_frame.copy(),
                     alpha=mask_alpha.astype(np.float32).copy(),
                     validity=(mask_alpha > 0).astype(np.float32))


def _compose_chain(flows_dir: Path, start: int, end: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Composed displacements between anchor `start` and target `end`.
    Returns (F_{end→start}, F_{start→end}): the first samples anchor content
    into the target frame; the second is its round-trip partner for FB gating.
    Composing resamples a smooth vector field per step (negligible blur);
    the single RGBA sample off the pristine anchor happens in star_warp."""
    if end > start:
        back = flow_service.load_flow(flows_dir, end - 1, "bwd").to(device)   # end→end-1
        for k in range(end - 1, start, -1):
            back = flow_service.compose_flow(
                back, flow_service.load_flow(flows_dir, k - 1, "bwd").to(device))
        fwd = flow_service.load_flow(flows_dir, start, "fwd").to(device)      # start→start+1
        for k in range(start + 1, end):
            fwd = flow_service.compose_flow(
                fwd, flow_service.load_flow(flows_dir, k, "fwd").to(device))
        return back, fwd
    # end < start: sample direction chains fwd pairs, gate partner chains bwd pairs
    back = flow_service.load_flow(flows_dir, end, "fwd").to(device)           # end→end+1
    for k in range(end + 1, start):
        back = flow_service.compose_flow(
            back, flow_service.load_flow(flows_dir, k, "fwd").to(device))
    fwd = flow_service.load_flow(flows_dir, start - 1, "bwd").to(device)      # start→start-1
    for k in range(start - 1, end, -1):
        fwd = flow_service.compose_flow(
            fwd, flow_service.load_flow(flows_dir, k - 1, "bwd").to(device))
    return back, fwd


def iter_chain(flows_dir: Path, anchor: int, step: int, count: int, device):
    """Walk away from `anchor` one frame at a time (step = +1 or -1), yielding
    (t, F_{t→anchor}, F_{anchor→t}) for t = anchor+step, anchor+2·step, …
    Each yield extends the previous composition by ONE pairwise flow — O(1)
    per step instead of recomposing the whole chain (O(k)) per target, which
    turned full-clip propagation into O(N²) flow loads."""
    f_ta = f_at = None
    for i in range(1, count + 1):
        t = anchor + step * i
        if step > 0:
            step_ta = flow_service.load_flow(flows_dir, t - 1, "bwd").to(device)  # t→t-1
            step_at = flow_service.load_flow(flows_dir, t - 1, "fwd").to(device)  # t-1→t
        else:
            step_ta = flow_service.load_flow(flows_dir, t, "fwd").to(device)      # t→t+1
            step_at = flow_service.load_flow(flows_dir, t, "bwd").to(device)      # t+1→t
        f_ta = step_ta if f_ta is None else flow_service.compose_flow(step_ta, f_ta)
        f_at = step_at if f_at is None else flow_service.compose_flow(f_at, step_at)
        yield t, f_ta, f_at


@torch.no_grad()
def star_warp(anchor: EditLayer, anchor_index: int, target_indices: list[int],
              flows_dir: Path, device: torch.device) -> dict[int, EditLayer]:
    payload = np.concatenate([anchor.rgb.astype(np.float32),
                              anchor.alpha[..., None] * 255.0,
                              anchor.validity[..., None] * 255.0], axis=2)  # (H,W,5)
    src = torch.from_numpy(payload).permute(2, 0, 1)[None].to(device)
    targets = set(target_indices)
    out: dict[int, EditLayer] = {}
    if anchor_index in targets:
        out[anchor_index] = EditLayer(anchor.rgb.copy(), anchor.alpha.copy(),
                                      anchor.validity.copy())
    for step in (1, -1):
        side = [t for t in targets if (t - anchor_index) * step > 0]
        if not side:
            continue
        count = max(abs(t - anchor_index) for t in side)
        for t, f_ta, f_at in iter_chain(flows_dir, anchor_index, step, count, device):
            if t not in targets:
                continue
            warped = flow_service.warp(src, f_ta)[0].permute(1, 2, 0).cpu().numpy()
            valid = flow_service.fb_check(f_ta, f_at)[0, 0].cpu().numpy()  # gate composed flows
            out[t] = EditLayer(
                rgb=np.clip(warped[..., :3], 0, 255).astype(np.uint8),
                alpha=np.clip(warped[..., 3] / 255.0, 0, 1),
                validity=np.clip(warped[..., 4] / 255.0, 0, 1) * valid,
            )
    return out


def blend_bidirectional(past: EditLayer, fut: EditLayer,
                        dist_past: int, dist_fut: int, eps: float = 1e-3) -> EditLayer:
    """§2.4: per-pixel blend weighted by temporal distance × validity.
    Something hidden looking forward is often visible looking back."""
    wp = past.validity / (dist_past + eps)
    wf = fut.validity / (dist_fut + eps)
    tot = wp + wf
    safe = np.where(tot > 0, tot, 1.0)[..., None]
    rgb = (wp[..., None] * past.rgb + wf[..., None] * fut.rgb) / safe
    alpha = (wp * past.alpha + wf * fut.alpha) / safe[..., 0]
    best = np.maximum(past.validity, fut.validity)
    validity = np.where(best >= 0.5, best, 0.0).astype(np.float32)
    return EditLayer(np.clip(rgb, 0, 255).astype(np.uint8),
                     np.clip(alpha, 0, 1).astype(np.float32), validity)


def needs_reanchor(layer: EditLayer, mask_alpha: np.ndarray,
                   anchor_mask_area: float, frames_since_anchor: int) -> bool:
    """§2.6 triggers: validity collapse, mask-area drift, chain-length cap."""
    inside = mask_alpha > 0.5
    if inside.any() and layer.validity[inside].mean() < 0.7:
        return True
    area = float(inside.sum())
    if anchor_mask_area > 0 and abs(area / anchor_mask_area - 1.0) > 0.3:
        return True
    return frames_since_anchor >= 60


def composite(frame: np.ndarray, layer: EditLayer, mask_alpha: np.ndarray) -> np.ndarray:
    """out = α·E + (1−α)·I with α = mask · layer alpha · validity gate (§2.5)."""
    a = (mask_alpha * layer.alpha * (layer.validity >= 0.5))[..., None].astype(np.float32)
    out = a * layer.rgb.astype(np.float32) + (1 - a) * frame.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
