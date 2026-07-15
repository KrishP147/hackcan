from pathlib import Path

import pytest
from fastapi import HTTPException

import main


def test_upload_duration_accepts_video_under_six_seconds(monkeypatch, tmp_path: Path):
    video = tmp_path / "video.mp4"
    monkeypatch.setattr(
        main.ffmpeg_service,
        "probe_video",
        lambda _: {"duration": 5.99},
    )

    assert main._validate_upload_duration(video) == 5.99


@pytest.mark.parametrize("duration", [6.0, 6.01, 30.0])
def test_upload_duration_rejects_six_seconds_or_longer(
    monkeypatch, tmp_path: Path, duration: float
):
    video = tmp_path / "video.mp4"
    monkeypatch.setattr(
        main.ffmpeg_service,
        "probe_video",
        lambda _: {"duration": duration},
    )

    with pytest.raises(HTTPException) as error:
        main._validate_upload_duration(video)

    assert error.value.status_code == 400
    assert "under 6 seconds" in error.value.detail
