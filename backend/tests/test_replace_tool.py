import asyncio

import cv2
import numpy as np
import torch

from services import replace_tool

H, W = 48, 64


def _project(tmp_project, n_frames, shift=1):
    for i in range(1, n_frames):
        fwd = np.zeros((2, H, W), np.float16)
        fwd[0] = shift
        bwd = np.zeros((2, H, W), np.float16)
        bwd[0] = -shift
        np.save(tmp_project / "flows" / f"flow_fwd_{i:04d}.npy", fwd)
        np.save(tmp_project / "flows" / f"flow_bwd_{i:04d}.npy", bwd)
    for t in range(1, n_frames + 1):
        f = np.full((H, W, 3), 90, np.uint8)
        x0 = 10 + (t - 1) * shift
        f[20:30, x0:x0 + 10] = (0, 0, 255)
        cv2.imwrite(str(tmp_project / "frames" / f"frame_{t:04d}.jpg"), f)
        m = np.zeros((H, W), np.uint8)
        m[20:30, x0:x0 + 10] = 255
        cv2.imwrite(str(tmp_project / "masks" / f"mask_{t:04d}.png"), m)
    return tmp_project


def _green_generator(calls):
    async def fake_generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
        calls.append(prompt)
        img = cv2.imread(str(frame_path))
        m = cv2.imread(str(mask_path), 0)
        img[m > 127] = (0, 255, 0)                     # deterministic green object
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()
    return fake_generate


def test_replace_carries_single_appearance(tmp_project):
    p = _project(tmp_project, 6)
    calls = []
    asyncio.run(replace_tool.apply_replace_range(
        p, list(range(1, 7)), anchor_index=1, prompt="make it green",
        device=torch.device("cpu"), generate=_green_generator(calls)))
    assert len(calls) == 1                             # no re-anchor needed on easy clip
    out5 = cv2.imread(str(p / "frames" / "frame_0005.jpg"))
    x = 14 + 4                                          # box center followed the motion
    assert out5[24, x, 1] > 150 and out5[24, x, 2] < 120   # green carried, not red


def test_reanchor_fires_on_chain_cap(tmp_project, monkeypatch):
    monkeypatch.setattr(replace_tool, "CHAIN_CAP", 3)  # shrink 60-frame cap for the test
    p = _project(tmp_project, 8)
    calls = []
    asyncio.run(replace_tool.apply_replace_range(
        p, list(range(1, 9)), anchor_index=1, prompt="x",
        device=torch.device("cpu"), generate=_green_generator(calls)))
    assert len(calls) >= 2                             # anchor + at least one re-anchor
    out8 = cv2.imread(str(p / "frames" / "frame_0008.jpg"))
    x = 14 + 7                                          # appearance still green at the end
    assert out8[24, x, 1] > 150
