from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from services import flow_service
from services.flow_service import warp, compose_flow, fb_check


@pytest.fixture(autouse=True)
def reset_raft_cache():
    flow_service.reset_raft_model()
    yield
    flow_service.reset_raft_model()


def const_flow(dx, dy, h=32, w=48):
    f = torch.zeros(1, 2, h, w)
    f[:, 0] = dx
    f[:, 1] = dy
    return f


def test_warp_translation():
    img = torch.rand(1, 3, 32, 48)
    out = warp(img, const_flow(-5, 0))          # out(x) = img(x-5) → content shifts right
    assert torch.allclose(out[..., 5:], img[..., :-5], atol=1e-5)
    assert torch.all(out[..., :4].abs() < 1e-6)  # zeros padding at the seam


def test_warp_identity():
    img = torch.rand(1, 3, 32, 48)
    assert torch.allclose(warp(img, const_flow(0, 0)), img, atol=1e-5)


def test_compose_translations_add():
    f1, f2 = const_flow(3, 1), const_flow(2, -1)
    comp = compose_flow(f1, f2)
    assert torch.allclose(comp[:, 0, 8:-8, 8:-8], torch.tensor(5.0), atol=1e-4)
    assert torch.allclose(comp[:, 1, 8:-8, 8:-8], torch.tensor(0.0), atol=1e-4)


def test_fb_check_consistent_and_broken():
    v = fb_check(const_flow(5, 0), const_flow(-5, 0))
    assert v[:, :, 8:-8, 8:-8].min() == 1.0      # interior valid
    v_bad = fb_check(const_flow(5, 0), const_flow(5, 0))  # doesn't round-trip
    assert v_bad[:, :, 8:-8, 8:-8].max() == 0.0


def _write_frames(frames_dir: Path, n=3, h=64, w=96):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    for i in range(1, n + 1):
        cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), np.roll(base, i * 2, axis=1))


def test_compute_flows_caches_all_pairs(tmp_project, monkeypatch):
    _write_frames(tmp_project / "frames")

    class FakeRaft:  # returns constant 2px-right flow, matching the rolled frames
        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def __call__(self, a, b, num_flow_updates=12):
            n, _, h, w = a.shape
            f = torch.zeros(n, 2, h, w)
            f[:, 0] = -2.0
            return [f]

    monkeypatch.setattr(flow_service, "_build_raft", lambda device: FakeRaft())
    pairs = flow_service.compute_flows(tmp_project / "frames", tmp_project / "flows",
                                       device=torch.device("cpu"))
    assert pairs == 2
    assert (tmp_project / "flows" / "flow_fwd_0001.npy").exists()
    assert (tmp_project / "flows" / "flow_bwd_0002.npy").exists()
    f = flow_service.load_flow(tmp_project / "flows", 1, "fwd")
    assert f.shape[1] == 2 and abs(f[:, 0].mean().item() + 2.0) < 1e-3


def test_compute_flows_batches_adjacent_pairs(tmp_project, monkeypatch):
    _write_frames(tmp_project / "frames", n=5)
    batch_calls = []

    class FakeRaft:
        def __call__(self, a, b, num_flow_updates=12):
            batch_calls.append(a.shape[0])
            return [torch.zeros(a.shape[0], 2, a.shape[2], a.shape[3])]

    monkeypatch.setattr(flow_service, "_build_raft", lambda device: FakeRaft())
    pairs = flow_service.compute_flows(
        tmp_project / "frames",
        tmp_project / "flows",
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert pairs == 4
    assert batch_calls == [2, 2, 2, 2]  # forward + backward for two chunks


@pytest.mark.slow
def test_real_raft_smoke(tmp_project):
    # RAFT's correlation pyramid needs feature maps ≥ its floor: use 128×192, not 64×96
    _write_frames(tmp_project / "frames", h=128, w=192)
    pairs = flow_service.compute_flows(tmp_project / "frames", tmp_project / "flows",
                                       device=torch.device("cpu"))
    assert pairs == 2  # downloads raft_large weights on first run
