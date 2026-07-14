import cv2
import numpy as np

from services import mask_service


def _disk(h, w, cx, cy, r):
    yy, xx = np.mgrid[:h, :w]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8) * 255


def test_stabilize_closes_pinholes_and_smooths_jitter(tmp_project):
    masks = tmp_project / "masks"
    m1 = _disk(64, 64, 32, 32, 20)
    m1[32, 32] = 0                                             # pinhole
    m2 = _disk(64, 64, 33, 32, 20)                             # 1px jitter
    cv2.imwrite(str(masks / "mask_0001.png"), m1)
    cv2.imwrite(str(masks / "mask_0002.png"), m2)
    n = mask_service.stabilize_masks(masks)
    assert n == 2
    s1 = cv2.imread(str(masks / "mask_0001.png"), 0)
    assert s1[32, 32] == 255                                   # pinhole closed
    s2 = cv2.imread(str(masks / "mask_0002.png"), 0)
    assert set(np.unique(s2)) <= {0, 255}                      # still binary


def test_feathered_alpha_is_soft_at_boundary(tmp_project):
    masks = tmp_project / "masks"
    cv2.imwrite(str(masks / "mask_0001.png"), _disk(64, 64, 32, 32, 20))
    a = mask_service.load_mask_alpha(masks, 1, feather_px=5)
    assert a.max() == 1.0 and a.min() == 0.0
    assert ((a > 0.05) & (a < 0.95)).sum() > 50                # soft transition band exists
