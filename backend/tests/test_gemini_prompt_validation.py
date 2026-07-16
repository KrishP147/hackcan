import asyncio
import json

from PIL import Image

from services import gemini_service


class _FakeResponse:
    def __init__(self, payload):
        self.text = json.dumps(payload)


class _FakeModels:
    def __init__(self, payload):
        self.payload = payload

    def generate_content(self, **_kwargs):
        return _FakeResponse(self.payload)


class _FakeClient:
    def __init__(self, payload):
        self.models = _FakeModels(payload)


def _visuals(tmp_path):
    frame = tmp_path / "frame.jpg"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (16, 16), "green").save(frame)
    Image.new("L", (16, 16), 255).save(mask)
    return frame, mask


def test_prompt_validator_accepts_relevant_prompt(tmp_path, monkeypatch):
    frame, mask = _visuals(tmp_path)
    monkeypatch.setattr(
        gemini_service,
        "_get_client",
        lambda: _FakeClient({"valid": True, "reason": "Concrete new scene."}),
    )

    result = asyncio.run(gemini_service.validate_edit_prompt(
        frame, mask, "bg_replace", "a rainy street in Tokyo at night"
    ))

    assert result.valid is True


def test_prompt_validator_rejects_unrelated_prompt(tmp_path, monkeypatch):
    frame, mask = _visuals(tmp_path)
    monkeypatch.setattr(
        gemini_service,
        "_get_client",
        lambda: _FakeClient({"valid": False, "reason": "This asks a question."}),
    )

    result = asyncio.run(gemini_service.validate_edit_prompt(
        frame, mask, "replace", "what time is it"
    ))

    assert result.valid is False
    assert result.reason == "This asks a question."


def test_prompt_validator_rejects_empty_prompt_without_api(tmp_path, monkeypatch):
    frame, mask = _visuals(tmp_path)
    monkeypatch.setattr(
        gemini_service,
        "_get_client",
        lambda: (_ for _ in ()).throw(AssertionError("API must not be called")),
    )

    result = asyncio.run(gemini_service.validate_edit_prompt(
        frame, mask, "replace", "  "
    ))

    assert result.valid is False
    assert "Describe" in result.reason
