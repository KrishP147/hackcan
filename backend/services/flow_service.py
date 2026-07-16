"""Backend A flow primitives (Doc 1 §2). Sampling convention pinned here once:
grid_sample bilinear, padding zeros, align_corners=False, pixel centers +0.5.
Doc 2 CUDA kernels must match this byte-for-byte."""
from pathlib import Path
import threading

import numpy as np
import torch
import torch.nn.functional as F


_raft_model = None
_raft_device = None
_raft_model_lock = threading.Lock()


@torch.no_grad()
def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward bilinear sample: out(x) = img(x + flow(x)). img (N,C,H,W), flow (N,2,H,W) px."""
    n, _, h, w = img.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=img.device),
                            torch.arange(w, device=img.device), indexing="ij")
    base = torch.stack((xx, yy)).float()
    src = base[None] + flow
    gx = 2 * (src[:, 0] + 0.5) / w - 1
    gy = 2 * (src[:, 1] + 0.5) / h - 1
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)


@torch.no_grad()
def compose_flow(f_na: torch.Tensor, f_step: torch.Tensor) -> torch.Tensor:
    """F_{N→k-1}(x) = F_{N→k}(x) + sample(F_{k→k-1}, x + F_{N→k}(x))."""
    return f_na + warp(f_step, f_na)


@torch.no_grad()
def fb_check(f_fwd: torch.Tensor, f_bwd: torch.Tensor) -> torch.Tensor:
    """Sundaram et al. forward-backward consistency → validity mask in {0,1}."""
    err = (f_fwd + warp(f_bwd, f_fwd)).pow(2).sum(1, keepdim=True)
    bound = 0.01 * (f_fwd.pow(2).sum(1, keepdim=True)
                    + f_bwd.pow(2).sum(1, keepdim=True)) + 0.5
    return (err <= bound).float()


def _build_raft(device):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    return raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()


def get_raft_model(device: torch.device):
    """Load RAFT once per process and retain it for later flow ranges."""
    global _raft_model, _raft_device
    device_key = str(device)
    with _raft_model_lock:
        if _raft_model is None or _raft_device != device_key:
            _raft_model = _build_raft(device)
            _raft_device = device_key
    return _raft_model


def reset_raft_model():
    """Drop the process cache (primarily useful for tests/device changes)."""
    global _raft_model, _raft_device
    with _raft_model_lock:
        _raft_model = None
        _raft_device = None


def _load_frame_tensor(path: Path, device) -> torch.Tensor:
    import cv2
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 127.5 - 1.0  # [-1,1]
    return t.to(device)


def _pad8(t: torch.Tensor):
    h, w = t.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    return F.pad(t, (0, pw, 0, ph)), h, w


@torch.no_grad()
def compute_flows(frames_dir: Path, flows_dir: Path,
                  device: torch.device | None = None,
                  num_flow_updates: int = 12,
                  start: int | None = None, end: int | None = None,
                  batch_size: int | None = None) -> int:
    """Pairwise RAFT flow on ORIGINAL footage, cached to disk once per project.
    flow_fwd_%04d.npy = F_{t→t+1}, flow_bwd_%04d.npy = F_{t+1→t}, indexed by t.
    start/end (1-based frame numbers) restrict computation to the pairs an edit
    actually needs; omitted → whole clip."""
    from services.config import get_device, get_raft_batch_size
    device = device or get_device()
    batch_size = max(1, batch_size or get_raft_batch_size())
    flows_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    pairs = 0
    pending = []
    for i in range(len(frames) - 1):
        idx = i + 1  # 1-based pair index = left frame number
        if (start is not None and idx < start) or (end is not None and idx >= end):
            continue
        fwd_p = flows_dir / f"flow_fwd_{idx:04d}.npy"
        bwd_p = flows_dir / f"flow_bwd_{idx:04d}.npy"
        pairs += 1
        if fwd_p.exists() and bwd_p.exists():
            continue
        pending.append((frames[i], frames[i + 1], fwd_p, bwd_p))

    if not pending:
        return pairs

    model = get_raft_model(device)
    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset:offset + batch_size]
        left, right, sizes = [], [], []
        for left_path, right_path, _, _ in chunk:
            a, h, w = _pad8(_load_frame_tensor(left_path, device))
            b, _, _ = _pad8(_load_frame_tensor(right_path, device))
            left.append(a)
            right.append(b)
            sizes.append((h, w))

        a_batch = torch.cat(left, dim=0)
        b_batch = torch.cat(right, dim=0)
        fwd_batch = model(
            a_batch, b_batch, num_flow_updates=num_flow_updates)[-1]
        bwd_batch = model(
            b_batch, a_batch, num_flow_updates=num_flow_updates)[-1]

        for j, (_, _, fwd_p, bwd_p) in enumerate(chunk):
            h, w = sizes[j]
            np.save(
                fwd_p,
                fwd_batch[j, ..., :h, :w].cpu().numpy().astype(np.float16),
            )
            np.save(
                bwd_p,
                bwd_batch[j, ..., :h, :w].cpu().numpy().astype(np.float16),
            )
    return pairs


def load_flow(flows_dir: Path, pair_index: int, direction: str) -> torch.Tensor:
    arr = np.load(flows_dir / f"flow_{direction}_{pair_index:04d}.npy").astype(np.float32)
    return torch.from_numpy(arr)[None]
