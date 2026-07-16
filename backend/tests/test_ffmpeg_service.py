import subprocess
from pathlib import Path

import pytest

from services import ffmpeg_service


@pytest.fixture
def clip_24fps(tmp_path: Path) -> Path:
    out = tmp_path / "in.mp4"
    subprocess.run(
        [ffmpeg_service._FFMPEG, "-y", "-f", "lavfi",
         "-i", "testsrc=duration=1:size=1920x1080:rate=24",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True)
    return out


def test_probe_reports_native_fps(clip_24fps):
    info = ffmpeg_service.probe_video(clip_24fps)
    assert abs(info["fps"] - 24.0) < 0.01
    assert info["width"] == 1920


def test_extract_native_fps_and_720p(clip_24fps, tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    count, used_fps = ffmpeg_service.extract_frames(clip_24fps, frames_dir, fps=None)
    assert abs(used_fps - 24.0) < 0.01
    assert count == 24                      # 1s @ native 24fps, not forced 30
    import cv2
    img = cv2.imread(str(frames_dir / "frame_0001.jpg"))
    assert img.shape[0] == 720              # downscaled working frames
