"""Host-authored render contract fixtures; no model acceptance is implied."""

import json

import pytest
from pydantic import ValidationError

from clipperx.render_contracts import EncoderName, FFmpegRenderSpec, RenderProfile


def test_profile_defaults_and_rejects_unsupported_settings() -> None:
    assert RenderProfile().model_dump(mode="json") == {
        "width": 1080, "height": 1920, "fps": 60, "video_encoder": "h264_nvenc",
        "audio_codec": "aac", "audio_bitrate": "192k", "audio_rate": 48000,
        "pixel_format": "yuv420p",
    }
    for values in ({"width": 1079}, {"fps": 30}, {"video_encoder": "auto"}, {"extra": 1}):
        with pytest.raises(ValidationError):
            RenderProfile.model_validate(values)


def test_spec_detaches_and_freezes_collections_but_serializes_arrays() -> None:
    command = ["ffmpeg", "-version"]
    assets = ["brand.png"]
    profile = RenderProfile(video_encoder=EncoderName.LIBX264)
    spec = FFmpegRenderSpec.model_validate({
        "source": "source.mp4", "output": "output.mp4", "plan_hash": "host_hash",
        "command": command, "filter_complex": "", "expected_duration": 2,
        "profile": profile, "assets": assets,
    })
    command.append("changed")
    assets.clear()
    assert spec.command == ("ffmpeg", "-version")
    assert spec.assets == ("brand.png",)
    assert json.loads(spec.model_dump_json())["command"] == ["ffmpeg", "-version"]
    assert set(spec.model_dump()) == {
        "source", "output", "plan_hash", "command", "filter_complex",
        "expected_duration", "profile", "assets",
    }
    with pytest.raises(ValidationError):
        spec.expected_duration = 3
    with pytest.raises(ValidationError):
        spec.profile.fps = 60


@pytest.mark.parametrize("duration", [0, -1, 180, float("nan"), float("inf")])
def test_spec_rejects_invalid_duration(duration: float) -> None:
    with pytest.raises(ValidationError):
        FFmpegRenderSpec.model_validate({
            "source": "s", "output": "o", "plan_hash": "h", "command": [],
            "filter_complex": "", "expected_duration": duration,
            "profile": {}, "assets": [],
        })
