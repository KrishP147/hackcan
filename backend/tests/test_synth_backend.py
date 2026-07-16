import asyncio
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from PIL import Image
import torch

from services import synth_propagation_service


def _write_frame(path, value):
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_mask(path):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    Image.fromarray(mask).save(path)


def _encoded_anchor(value=220):
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return data.tobytes()


def test_backend_b_generates_once_and_commits_staged_frames(tmp_project):
    _write_frame(tmp_project / "frames" / "frame_0001.jpg", 20)
    _write_frame(tmp_project / "frames" / "frame_0002.jpg", 40)
    _write_mask(tmp_project / "masks" / "mask_0001.png")
    _write_mask(tmp_project / "masks" / "mask_0002.png")
    generate = AsyncMock(return_value=_encoded_anchor())
    zero_flow = torch.zeros((1, 2, 16, 16), dtype=torch.float32)

    def chain(*_args, **_kwargs):
        yield 2, zero_flow, zero_flow

    def synth(source_edit, *_args, **_kwargs):
        return source_edit.copy(), {"backend": "torch", "working_size": [16, 16]}

    with patch.object(synth_propagation_service, "iter_chain", side_effect=chain), \
         patch.object(synth_propagation_service.guided_synthesis, "synthesize", side_effect=synth) as run:
        asyncio.run(synth_propagation_service.apply_replace_range(
            tmp_project, [1, 2], 1, "red robot", torch.device("cpu"), generate=generate))

    generate.assert_awaited_once()
    assert run.call_count == 1
    anchor = cv2.imread(str(tmp_project / "frames" / "frame_0001.jpg"))
    target = cv2.imread(str(tmp_project / "frames" / "frame_0002.jpg"))
    assert anchor[8, 8].mean() > 180
    assert target[8, 8].mean() > 180
    assert target[0, 0].mean() < 80


def test_backend_b_failure_falls_back_without_partial_commit(tmp_project):
    _write_frame(tmp_project / "frames" / "frame_0001.jpg", 20)
    _write_frame(tmp_project / "frames" / "frame_0002.jpg", 40)
    _write_mask(tmp_project / "masks" / "mask_0001.png")
    _write_mask(tmp_project / "masks" / "mask_0002.png")
    generate = AsyncMock(return_value=_encoded_anchor())
    zero_flow = torch.zeros((1, 2, 16, 16), dtype=torch.float32)

    def chain(*_args, **_kwargs):
        yield 2, zero_flow, zero_flow

    with patch.object(synth_propagation_service, "iter_chain", side_effect=chain), \
         patch.object(synth_propagation_service.guided_synthesis, "synthesize",
                      side_effect=RuntimeError("kernel failed")), \
         patch.object(synth_propagation_service.replace_tool, "apply_replace_range",
                      new_callable=AsyncMock) as fallback:
        asyncio.run(synth_propagation_service.apply_replace_range(
            tmp_project, [1, 2], 1, "red robot", torch.device("cpu"), generate=generate))

    fallback.assert_awaited_once()
    # Staged anchor was not committed before Backend A took over.
    anchor = cv2.imread(str(tmp_project / "frames" / "frame_0001.jpg"))
    assert anchor[8, 8].mean() < 80


def test_cached_anchor_is_reused_by_backend_a_fallback(tmp_project):
    _write_frame(tmp_project / "frames" / "frame_0001.jpg", 20)
    _write_frame(tmp_project / "frames" / "frame_0002.jpg", 40)
    _write_mask(tmp_project / "masks" / "mask_0001.png")
    _write_mask(tmp_project / "masks" / "mask_0002.png")
    anchor_bytes = _encoded_anchor()
    generate = AsyncMock(return_value=anchor_bytes)
    zero_flow = torch.zeros((1, 2, 16, 16), dtype=torch.float32)

    def chain(*_args, **_kwargs):
        yield 2, zero_flow, zero_flow

    reused = []

    async def fallback(*_args, generate, **_kwargs):
        reused.append(await generate(
            tmp_project / "frames" / "frame_0001.jpg", "x",
            reference_frame_path=None,
            mask_path=tmp_project / "masks" / "mask_0001.png"))

    with patch.object(synth_propagation_service, "iter_chain", side_effect=chain), \
         patch.object(synth_propagation_service.guided_synthesis, "synthesize",
                      side_effect=RuntimeError("kernel failed")), \
         patch.object(synth_propagation_service.replace_tool, "apply_replace_range",
                      side_effect=fallback):
        asyncio.run(synth_propagation_service.apply_replace_range(
            tmp_project, [1, 2], 1, "red robot", torch.device("cpu"), generate=generate))

    assert reused == [anchor_bytes]
    generate.assert_awaited_once()
