"""Output acceptance uses actual bounded observations."""

import importlib.util
from fractions import Fraction

import pytest
from typing import Any


def test_metrics_module_exists() -> None:
    assert importlib.util.find_spec("clipperx.render_metrics") is not None


def test_actual_output_validation_and_unmeasured_oracles() -> None:
    from clipperx.render_metrics import validate_output_observations
    metadata = {"streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920,
         "pix_fmt": "yuv420p", "avg_frame_rate": "60/1", "nb_read_frames": "2"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 1}],
        "format": {"start_time": "0", "duration": "0.034"}}
    metrics = validate_output_observations(metadata, (Fraction(0), Fraction(1, 60)),
        ((Fraction(0), 1024), (Fraction(1024, 48000), 1024)),
        expected_frames=2, expected_samples=1600, channels=1, actual_size=1234, faststart=True)
    assert metrics["pixel_oracle"]["status"] == "notmeasured"
    assert metrics["pixel_oracle"]["source_marker"] is None
    with pytest.raises(ValueError, match="PTS"):
        validate_output_observations(metadata, (Fraction(0), Fraction(1, 59)),
            ((Fraction(0), 1600),), expected_frames=2, expected_samples=1600,
            channels=1, actual_size=1234, faststart=True)


@pytest.mark.parametrize("change", ["count", "codec", "offset", "cap", "faststart"])
def test_observed_output_fails_closed(change: str) -> None:
    from clipperx.render_metrics import validate_output_observations
    data: dict[str, Any] = {"streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920,
         "pix_fmt": "yuv420p", "avg_frame_rate": "60/1", "nb_read_frames": "2"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 1}],
        "format": {"start_time": "0", "duration": "0.034"}}
    if change == "codec":
        data["streams"][0]["codec_name"] = "hevc"
    with pytest.raises(ValueError):
        validate_output_observations(data, (Fraction(0),) if change == "count" else
            (Fraction(0), Fraction(1, 60)), ((Fraction(1) if change == "offset" else Fraction(0), 1600),),
            expected_frames=2, expected_samples=1600, channels=1,
            actual_size=2**31+1 if change == "cap" else 1234, faststart=change != "faststart")
