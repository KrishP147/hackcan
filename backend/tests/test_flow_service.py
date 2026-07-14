import torch

from services.flow_service import warp, compose_flow, fb_check


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
