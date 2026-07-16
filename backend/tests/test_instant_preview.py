"""Instant-preview path + progressive-sweep plumbing."""
import asyncio
from io import BytesIO

import cv2
import numpy as np
import pytest
import torch

from services import preview_service, project_manager
from services.inpaint_service import inpaint_video
from services.propagation_service import _compose_chain, iter_chain

H, W = 48, 64


# --- iter_chain must match the O(k) reference composition ---

def _write_const_flows(flows_dir, n_pairs, dx):
    for i in range(1, n_pairs + 1):
        fwd = np.zeros((2, H, W), np.float16)
        fwd[0] = dx
        bwd = np.zeros((2, H, W), np.float16)
        bwd[0] = -dx
        np.save(flows_dir / f"flow_fwd_{i:04d}.npy", fwd)
        np.save(flows_dir / f"flow_bwd_{i:04d}.npy", bwd)


@pytest.mark.parametrize("anchor,step,count", [(1, 1, 4), (5, -1, 4)])
def test_iter_chain_matches_compose_chain(tmp_project, anchor, step, count):
    _write_const_flows(tmp_project / "flows", 4, 2)
    dev = torch.device("cpu")
    for t, f_ta, f_at in iter_chain(tmp_project / "flows", anchor, step, count, dev):
        ref_ta, ref_at = _compose_chain(tmp_project / "flows", anchor, t, dev)
        assert torch.allclose(f_ta, ref_ta, atol=1e-3), f"f_ta mismatch at t={t}"
        assert torch.allclose(f_at, ref_at, atol=1e-3), f"f_at mismatch at t={t}"


# --- inpaint on_frame fires per frame with the returned pixels ---

def test_inpaint_on_frame_callback(tmp_project):
    z = np.zeros((2, H, W), np.float16)
    for i in (1, 2):
        np.save(tmp_project / "flows" / f"flow_fwd_{i:04d}.npy", z)
        np.save(tmp_project / "flows" / f"flow_bwd_{i:04d}.npy", z)
    bg = np.full((H, W, 3), 90, np.uint8)
    frames, holes = {}, {}
    for t, x0 in [(1, 10), (2, 25), (3, 40)]:
        holes[t] = np.zeros((H, W), bool)
        holes[t][20:30, x0:x0 + 10] = True
        f = bg.copy()
        f[holes[t]] = 0
        frames[t] = f
    seen = {}
    out = inpaint_video(frames, holes, tmp_project / "flows", torch.device("cpu"),
                        on_frame=lambda t, img: seen.__setitem__(t, img.copy()))
    assert sorted(seen) == [1, 2, 3]
    for t in seen:
        assert np.array_equal(seen[t], out[t])


# --- preview renders in memory and never touches frames/ on disk ---

@pytest.fixture
def preview_project(tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "BASE_DIR", tmp_path)
    pdir = tmp_path / "prev"
    for sub in ("frames", "masks"):
        (pdir / sub).mkdir(parents=True)
    frame = np.full((H, W, 3), 120, np.uint8)
    cv2.imwrite(str(pdir / "frames" / "frame_0002.jpg"), frame)
    mask = np.zeros((H, W), np.uint8)
    mask[20:30, 10:20] = 255
    cv2.imwrite(str(pdir / "masks" / "mask_0002.png"), mask)
    (pdir / "status.json").write_text('{"anchor_frame": 2}')
    return pdir


def test_preview_recolor_returns_jpeg_and_leaves_frame_untouched(preview_project):
    before = (preview_project / "frames" / "frame_0002.jpg").read_bytes()
    jpeg = preview_service.render_preview("prev", 2, "recolor", color="FF0000")
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (H, W, 3)
    assert img[24, 14, 2] > img[24, 40, 2] + 20     # masked region pushed red
    after = (preview_project / "frames" / "frame_0002.jpg").read_bytes()
    assert before == after                           # source frame untouched


def test_preview_falls_back_to_anchor_mask(preview_project):
    # frame 3 exists but has no mask → anchor (frame 2) mask is used
    frame = np.full((H, W, 3), 120, np.uint8)
    cv2.imwrite(str(preview_project / "frames" / "frame_0003.jpg"), frame)
    jpeg = preview_service.render_preview("prev", 3, "blur_region", blur_strength=8)
    assert jpeg[:2] == b"\xff\xd8"                   # valid JPEG out


def test_preview_rejects_generative_types(preview_project):
    with pytest.raises(ValueError):
        preview_service.render_preview("prev", 2, "replace")


# --- /edit backup now records which frames were backed up (undo works) ---

def test_background_edit_records_backup_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "BASE_DIR", tmp_path)
    pdir = tmp_path / "undoable"
    (pdir / "frames").mkdir(parents=True)
    for t in (1, 2, 3):
        cv2.imwrite(str(pdir / "frames" / f"frame_{t:04d}.jpg"),
                    np.full((H, W, 3), 90, np.uint8))
    (pdir / "status.json").write_text("{}")

    import main
    from unittest.mock import AsyncMock, patch
    rule = main.EditRule(edit_type="recolor", start_frame=1, end_frame=3)
    with patch("services.edit_dispatch.run_edit_rule", new_callable=AsyncMock):
        asyncio.run(main._background_edit("undoable", [rule]))

    status = project_manager.get_status("undoable")
    assert status["last_backup_frames"] == [1, 2, 3]
    assert status["edit_status"] == "done"
    backup_dir = pdir / "backups" / status["last_backup_timestamp"]
    assert sorted(p.name for p in backup_dir.glob("*.jpg")) == \
        ["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"]
