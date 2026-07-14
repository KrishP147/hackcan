import cv2
import numpy as np

from services import local_edit_service


def _setup(tmp_path):
    frame = np.zeros((64, 64, 3), np.uint8)
    frame[:, :, 2] = 200                              # red-ish everywhere (BGR)
    fp = tmp_path / "frame_0001.jpg"
    cv2.imwrite(str(fp), frame)
    mask = np.zeros((64, 64), np.uint8)
    mask[20:44, 20:44] = 255
    mp = tmp_path / "mask_0001.png"
    cv2.imwrite(str(mp), mask)
    return fp, mp


def test_color_pop_desaturates_outside_only(tmp_path):
    fp, mp = _setup(tmp_path)
    local_edit_service.apply_color_pop(fp, mp)
    out = cv2.imread(str(fp))
    b, g, r = out[32, 32].astype(int)                 # inside: still colorful
    assert r - b > 100
    b, g, r = out[5, 5].astype(int)                   # outside: gray (channels equal-ish)
    assert abs(r - b) < 20 and abs(r - g) < 20


def test_glow_brightens_ring_outside_mask(tmp_path):
    fp, mp = _setup(tmp_path)
    before = cv2.imread(str(fp)).astype(int)
    local_edit_service.apply_glow(fp, mp)
    after = cv2.imread(str(fp)).astype(int)
    assert after[18, 32].sum() > before[18, 32].sum() + 30   # halo just outside mask
    assert abs(after[2, 2].sum() - before[2, 2].sum()) < 30  # far field untouched
