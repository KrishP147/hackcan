import numpy as np
import torch

from services.inpaint_service import inpaint_video

H, W = 48, 64


def _zero_flows(flows_dir, n_pairs):
    z = np.zeros((2, H, W), np.float16)
    for i in range(1, n_pairs + 1):
        np.save(flows_dir / f"flow_fwd_{i:04d}.npy", z)
        np.save(flows_dir / f"flow_bwd_{i:04d}.npy", z)


def test_flow_guided_fill_recovers_static_background(tmp_project):
    _zero_flows(tmp_project / "flows", 2)
    rng = np.random.default_rng(1)
    bg = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    frames, holes = {}, {}
    for t, x0 in [(1, 10), (2, 25), (3, 40)]:      # hole slides across static bg
        holes[t] = np.zeros((H, W), bool)
        holes[t][20:30, x0:x0 + 10] = True
        f = bg.copy()
        f[holes[t]] = 0
        frames[t] = f
    out = inpaint_video(frames, holes, tmp_project / "flows", torch.device("cpu"))
    # frame 2's hole is visible in frames 1 and 3 → recovered near-exactly
    assert np.abs(out[2][20:30, 25:35].astype(int) - bg[20:30, 25:35].astype(int)).mean() < 2


def test_never_seen_pixels_fall_back_to_telea(tmp_project):
    _zero_flows(tmp_project / "flows", 1)
    bg = np.full((H, W, 3), 128, np.uint8)
    hole = np.zeros((H, W), bool)
    hole[20:30, 20:30] = True                      # same hole every frame
    frames = {1: bg.copy(), 2: bg.copy()}
    for f in frames.values():
        f[hole] = 0
    out = inpaint_video(frames, {1: hole, 2: hole}, tmp_project / "flows", torch.device("cpu"))
    assert np.abs(out[1][22:28, 22:28].astype(int) - 128).mean() < 10   # TELEA on flat bg ≈ flat
