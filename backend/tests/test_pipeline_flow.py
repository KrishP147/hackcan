import asyncio
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from services import edit_dispatch, sam2_service


def _write_frame(path: Path, value: int = 80):
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def test_selection_builds_full_mask_cache_once(tmp_path):
    import main

    project_dir = tmp_path
    frames_dir = project_dir / "frames"
    masks_dir = project_dir / "masks"
    frames_dir.mkdir()
    masks_dir.mkdir()
    for index in range(1, 4):
        _write_frame(frames_dir / f"frame_{index:04d}.jpg", index * 40)

    statuses = []
    with patch.object(main.project_manager, "get_project_dir", return_value=project_dir), \
         patch.object(main.project_manager, "update_status", side_effect=lambda _id, **kw: statuses.append(kw)), \
         patch.object(main.project_manager, "get_status", return_value={}), \
         patch.object(main.sam2_service, "segment_frame", return_value=np.ones((16, 16), dtype=bool)), \
         patch.object(main.sam2_service, "propagate_masks", return_value=3) as propagate:
        asyncio.run(main._background_segment_and_propagate("pid", 1, 4, 4))

    propagate.assert_called_once()
    assert propagate.call_args.kwargs["frame_step"] == 1
    assert len([s for s in statuses if s.get("segment_status") == "done"]) == 1
    assert statuses[-1]["mask_count"] == 3


def test_cached_masks_mean_deterministic_edit_skips_sam2(tmp_project):
    for index in (1, 2, 3):
        _write_frame(tmp_project / "frames" / f"frame_{index:04d}.jpg")
        Image.fromarray(np.ones((16, 16), dtype=np.uint8) * 255).save(
            tmp_project / "masks" / f"mask_{index:04d}.png")

    from main import EditRule

    with patch.object(edit_dispatch, "_project_dir", return_value=tmp_project), \
         patch.object(edit_dispatch.project_manager, "get_status", return_value={"anchor_frame": 1}), \
         patch.object(sam2_service, "propagate_masks") as propagate, \
         patch.object(edit_dispatch.local_edit_service, "apply_blur_region") as blur:
        asyncio.run(edit_dispatch.run_edit_rule(
            "pid", EditRule(edit_type="blur_region", start_frame=1, end_frame=3)))

    propagate.assert_not_called()
    assert blur.call_count == 3


def test_deterministic_edit_cancels_between_frames(tmp_project):
    for index in (1, 2, 3):
        _write_frame(tmp_project / "frames" / f"frame_{index:04d}.jpg")
        Image.fromarray(np.ones((16, 16), dtype=np.uint8) * 255).save(
            tmp_project / "masks" / f"mask_{index:04d}.png")

    from main import EditRule

    with patch.object(edit_dispatch, "_project_dir", return_value=tmp_project), \
         patch.object(edit_dispatch.project_manager, "get_status", return_value={"anchor_frame": 1}), \
         patch.object(edit_dispatch.local_edit_service, "apply_blur_region") as blur:
        with pytest.raises(edit_dispatch.EditCancelled):
            asyncio.run(edit_dispatch.run_edit_rule(
                "pid", EditRule(edit_type="blur_region", start_frame=1, end_frame=3),
                cancel_check=lambda: True))

    blur.assert_not_called()
