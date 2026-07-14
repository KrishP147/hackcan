"""Backend A flow primitives (Doc 1 §2). Sampling convention pinned here once:
grid_sample bilinear, padding zeros, align_corners=False, pixel centers +0.5.
Doc 2 CUDA kernels must match this byte-for-byte."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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
                  num_flow_updates: int = 12) -> int:
    """Pairwise RAFT flow on ORIGINAL footage, cached to disk once per project.
    flow_fwd_%04d.npy = F_{t→t+1}, flow_bwd_%04d.npy = F_{t+1→t}, indexed by t."""
    from services.config import get_device
    device = device or get_device()
    flows_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    model = None
    pairs = 0
    for i in range(len(frames) - 1):
        idx = i + 1  # 1-based pair index = left frame number
        fwd_p = flows_dir / f"flow_fwd_{idx:04d}.npy"
        bwd_p = flows_dir / f"flow_bwd_{idx:04d}.npy"
        pairs += 1
        if fwd_p.exists() and bwd_p.exists():
            continue
        if model is None:
            model = _build_raft(device)
        a, h, w = _pad8(_load_frame_tensor(frames[i], device))
        b, _, _ = _pad8(_load_frame_tensor(frames[i + 1], device))
        fwd = model(a, b, num_flow_updates=num_flow_updates)[-1][..., :h, :w]
        bwd = model(b, a, num_flow_updates=num_flow_updates)[-1][..., :h, :w]
        np.save(fwd_p, fwd[0].cpu().numpy().astype(np.float16))
        np.save(bwd_p, bwd[0].cpu().numpy().astype(np.float16))
    return pairs


def load_flow(flows_dir: Path, pair_index: int, direction: str) -> torch.Tensor:
    arr = np.load(flows_dir / f"flow_{direction}_{pair_index:04d}.npy").astype(np.float32)
    return torch.from_numpy(arr)[None]
