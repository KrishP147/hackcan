import numpy as np
import torch

from services.propagation_service import EditLayer, make_anchor_layer, star_warp, composite

H, W, SHIFT = 48, 64, 2  # scene translates right 2px/frame


def _write_const_flows(flows_dir, n_pairs, dx):
    for i in range(1, n_pairs + 1):
        fwd = np.zeros((2, H, W), np.float16)
        fwd[0] = dx                                           # F_{t→t+1}
        bwd = np.zeros((2, H, W), np.float16)
        bwd[0] = -dx                                          # F_{t+1→t}
        np.save(flows_dir / f"flow_fwd_{i:04d}.npy", fwd)
        np.save(flows_dir / f"flow_bwd_{i:04d}.npy", bwd)


def test_star_warp_carries_one_appearance(tmp_project):
    _write_const_flows(tmp_project / "flows", 4, SHIFT)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[20:30, 10:20] = (255, 0, 0)                           # red box at anchor
    alpha = np.zeros((H, W), np.float32)
    alpha[20:30, 10:20] = 1.0
    anchor = make_anchor_layer(rgb, alpha)
    layers = star_warp(anchor, anchor_index=1, target_indices=[2, 3, 5],
                       flows_dir=tmp_project / "flows", device=torch.device("cpu"))
    l5 = layers[5]                                            # 4 steps of +2px → box at x=[18,28)
    assert l5.rgb[24, 22, 0] > 200                            # red carried, not re-invented
    assert l5.alpha[24, 22] > 0.9
    assert l5.alpha[24, 12] < 0.1                             # old location vacated
    assert l5.validity[24, 22] > 0.9


def test_star_warp_backward_direction(tmp_project):
    _write_const_flows(tmp_project / "flows", 4, SHIFT)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[20:30, 18:28] = (0, 255, 0)                           # anchor at frame 5
    alpha = np.zeros((H, W), np.float32)
    alpha[20:30, 18:28] = 1.0
    anchor = make_anchor_layer(rgb, alpha)
    layers = star_warp(anchor, anchor_index=5, target_indices=[1],
                       flows_dir=tmp_project / "flows", device=torch.device("cpu"))
    l1 = layers[1]                                            # 4 steps of −2px → box back at x=[10,20)
    assert l1.rgb[24, 14, 1] > 200
    assert l1.alpha[24, 14] > 0.9
    assert l1.validity[24, 14] > 0.9


def test_composite_blends_only_valid_masked_pixels(tmp_project):
    frame = np.full((H, W, 3), 100, np.uint8)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, :, 2] = 255
    alpha = np.ones((H, W), np.float32)
    layer = EditLayer(rgb=rgb, alpha=alpha, validity=np.ones((H, W), np.float32))
    layer.validity[:, W // 2:] = 0.0                          # right half gated out
    mask = np.ones((H, W), np.float32)
    out = composite(frame, layer, mask)
    assert out[10, 5, 2] == 255                               # valid → edit shows
    assert out[10, W - 5, 2] == 100                           # gated → original footage


from services.propagation_service import blend_bidirectional, needs_reanchor


def _layer(val_left, val_right, red=255):
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, :, 0] = red
    alpha = np.ones((H, W), np.float32)
    v = np.zeros((H, W), np.float32)
    v[:, :W // 2] = val_left
    v[:, W // 2:] = val_right
    return EditLayer(rgb, alpha, v)


def test_bidirectional_fills_disocclusion_from_other_side():
    past = _layer(1.0, 0.0, red=200)   # right half occluded looking forward
    fut = _layer(1.0, 1.0, red=100)    # visible looking back
    out = blend_bidirectional(past, fut, dist_past=2, dist_fut=2)
    assert out.rgb[10, W - 5, 0] == 100          # right half sourced from future
    assert 100 < out.rgb[10, 5, 0] < 200         # left half blends both
    assert out.validity[10, W - 5] > 0.5


def test_reanchor_triggers():
    good = _layer(1.0, 1.0)
    mask = np.ones((H, W), np.float32)
    assert not needs_reanchor(good, mask, anchor_mask_area=mask.sum(), frames_since_anchor=10)
    assert needs_reanchor(_layer(0.4, 0.4), mask, mask.sum(), 10)          # validity collapse
    assert needs_reanchor(good, mask, anchor_mask_area=mask.sum() * 2, frames_since_anchor=10)  # area jump
    assert needs_reanchor(good, mask, mask.sum(), frames_since_anchor=60)  # chain cap
