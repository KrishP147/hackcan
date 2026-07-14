"""Backend A flow primitives (Doc 1 §2). Sampling convention pinned here once:
grid_sample bilinear, padding zeros, align_corners=False, pixel centers +0.5.
Doc 2 CUDA kernels must match this byte-for-byte."""
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
