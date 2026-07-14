import cv2
import numpy as np
import torch

from services import object_tools

H, W = 48, 64


def _project(tmp_project, n=3):
    z = np.zeros((2, H, W), np.float16)
    for i in range(1, n):
        np.save(tmp_project / "flows" / f"flow_fwd_{i:04d}.npy", z)
        np.save(tmp_project / "flows" / f"flow_bwd_{i:04d}.npy", z)
    bg = np.full((H, W, 3), 90, np.uint8)
    for t in range(1, n + 1):
        f = bg.copy()
        f[20:30, 10:20] = (0, 0, 255)                         # red box (BGR)
        cv2.imwrite(str(tmp_project / "frames" / f"frame_{t:04d}.jpg"), f)
        m = np.zeros((H, W), np.uint8)
        m[20:30, 10:20] = 255
        cv2.imwrite(str(tmp_project / "masks" / f"mask_{t:04d}.png"), m)
    return tmp_project


def test_move_transports_object_and_fills_vacated(tmp_project):
    p = _project(tmp_project)
    object_tools.apply_move_range(p, [1, 2, 3], {1: (20, 0), 2: (20, 0), 3: (20, 0)},
                                  torch.device("cpu"))
    out = cv2.imread(str(p / "frames" / "frame_0002.jpg"))
    assert out[24, 35, 2] > 180 and out[24, 35, 0] < 80       # object at new spot
    assert abs(int(out[24, 12, 2]) - 90) < 25                  # vacated ≈ background
    assert abs(int(out[24, 12, 0]) - 90) < 25


def test_delete_removes_object(tmp_project):
    p = _project(tmp_project)
    object_tools.apply_delete_range(p, [1, 2, 3], torch.device("cpu"))
    out = cv2.imread(str(p / "frames" / "frame_0002.jpg"))
    assert abs(int(out[24, 14, 2]) - 90) < 25                  # red box gone → bg


def test_resize_down_reveals_clean_background(tmp_project):
    p = _project(tmp_project)
    object_tools.apply_resize_range(p, [1, 2, 3], 0.5, torch.device("cpu"))
    out = cv2.imread(str(p / "frames" / "frame_0002.jpg"))
    assert out[24, 14, 2] > 150                                # shrunk object still at center
    assert abs(int(out[21, 11, 2]) - 90) < 40                  # revealed ring ≈ background
