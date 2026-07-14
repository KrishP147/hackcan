import asyncio

import cv2
import numpy as np
import torch

from services import background_tool

H, W = 48, 64


def test_soft_matte_softens_boundary():
    frame = np.full((H, W, 3), 200, np.uint8)
    mask = np.zeros((H, W), np.float32)
    mask[10:38, 10:54] = 1.0
    matte = background_tool.soft_matte(frame, mask)
    assert matte[24, 30] > 0.9 and matte[2, 2] < 0.1
    band = ((matte > 0.1) & (matte < 0.9)).sum()
    assert band > 30                                    # soft edge exists


def test_bg_replace_tracks_camera_shift(tmp_project):
    # camera pans right 2px/frame: bg flow = -2 (content moves left)
    for i in (1, 2):
        fwd = np.zeros((2, H, W), np.float16)
        fwd[0] = -2
        bwd = np.zeros((2, H, W), np.float16)
        bwd[0] = 2
        np.save(tmp_project / "flows" / f"flow_fwd_{i:04d}.npy", fwd)
        np.save(tmp_project / "flows" / f"flow_bwd_{i:04d}.npy", bwd)
    for t in (1, 2, 3):
        cv2.imwrite(str(tmp_project / "frames" / f"frame_{t:04d}.jpg"),
                    np.full((H, W, 3), 90, np.uint8))
        m = np.zeros((H, W), np.uint8)
        m[20:30, 28:38] = 255
        cv2.imwrite(str(tmp_project / "masks" / f"mask_{t:04d}.png"), m)

    async def fake_generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
        plate = np.zeros((H, W, 3), np.uint8)
        plate[:, 8::16] = 255                           # vertical stripes at x=8,24,40,...
        ok, buf = cv2.imencode(".png", plate)
        return buf.tobytes()

    asyncio.run(background_tool.apply_background_replace_range(
        tmp_project, [1, 2, 3], anchor_index=1, prompt="stripes",
        device=torch.device("cpu"), generate=fake_generate))
    f1 = cv2.imread(str(tmp_project / "frames" / "frame_0001.jpg"))
    f3 = cv2.imread(str(tmp_project / "frames" / "frame_0003.jpg"))
    assert f1[5, 8, 0] > 150                            # stripe at x=8 in frame 1
    assert f3[5, 4, 0] > 150                            # stripes shifted −4px by frame 3
    assert f3[5, 8, 0] < 100                            # ...and gone from x=8
