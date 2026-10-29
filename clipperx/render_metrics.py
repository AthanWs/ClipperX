"""Bounded output observations, deliberately separate from compiler geometry."""

from fractions import Fraction
from typing import Any


def parse_frame_observations(raw: bytes, *, audio: bool) -> tuple[Any, ...]:
    """Parse ffprobe JSON without accepting absent/nonfinite timing facts."""
    import json
    if type(raw) is not bytes or len(raw) > 1_000_000:
        raise ValueError("bounded frame observation required")
    payload = json.loads(raw)
    frames = payload.get("frames")
    if type(frames) is not list or len(frames) > 20000:
        raise ValueError("bounded decoded frame list required")
    result: list[Any] = []
    for frame in frames:
        pts = Fraction(frame["best_effort_timestamp_time"])
        if abs(pts) > 1000:
            raise ValueError("bounded decoded PTS required")
        if audio:
            count = int(frame["nb_samples"])
            if not 1 <= count <= 65536:
                raise ValueError("bounded audio sample count required")
            result.append((pts, count))
        else:
            result.append(pts)
    return tuple(result)


def validate_output_observations(metadata: dict[str, Any], video_pts: tuple[Fraction, ...],
        audio_frames: tuple[tuple[Fraction, int], ...], *, expected_frames: int,
        expected_samples: int, channels: int, actual_size: int, faststart: bool) -> dict[str, Any]:
    if not 0 < actual_size <= 2147483648 or faststart is not True:
        raise ValueError("MP4 size cap/faststart failure")
    streams = metadata.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if len(streams) != 2 or len(video) != 1 or len(audio) != 1:
        raise ValueError("one H264 and one AAC stream required")
    v, a = video[0], audio[0]
    if (v.get("codec_name") != "h264" or (v.get("width"), v.get("height")) != (1080, 1920)
            or v.get("pix_fmt") != "yuv420p" or Fraction(v["avg_frame_rate"]) != 60
            or a.get("codec_name") != "aac" or int(a["sample_rate"]) != 48000
            or int(a["channels"]) != channels):
        raise ValueError("delivery codec/format mismatch")
    if len(video_pts) != expected_frames or int(v["nb_read_frames"]) != expected_frames:
        raise ValueError("natural decoded video count mismatch")
    # ffprobe's decimal rendering rounds to microseconds; compare to exact K/60.
    pts_error = max((abs(t-Fraction(k, 60)) for k, t in enumerate(video_pts)), default=Fraction(0))
    if pts_error > Fraction(1, 1_000_000):
        raise ValueError("continuous global video PTS mismatch")
    if not audio_frames or abs(audio_frames[0][0]) > Fraction(1, 48000):
        raise ValueError("audio presentation offset mismatch")
    samples = 0
    for pts, count in audio_frames:
        if abs(pts-Fraction(samples, 48000)) > Fraction(1, 1_000_000):
            raise ValueError("audio PTS discontinuity")
        samples += count
    if not 0 <= samples-expected_samples <= 1024:
        raise ValueError("AAC presented sample count mismatch")
    fmt = metadata["format"]
    tolerance = Fraction(1, 60)+Fraction(1024, 48000)
    if (abs(Fraction(fmt["start_time"])) > Fraction(1, 48000)
            or abs(Fraction(fmt["duration"])-Fraction(expected_samples, 48000)) > tolerance
            or abs(Fraction(expected_frames, 60)-Fraction(samples, 48000)) > tolerance):
        raise ValueError("mux presentation duration/sync mismatch")
    return {"status": "rendered", "user_approved": False, "full_qa_passed": False,
        "decoded_observations": {"video_frames": len(video_pts), "audio_samples": samples,
            "video_clock": "K/60", "audio_clock": "Qa/48000", "video_pts_max_delta": str(pts_error),
            "audio_sample_delta": samples-expected_samples, "actual_bytes": actual_size,
            "faststart": faststart, "mux_tolerance_seconds": str(tolerance), "streams": streams},
        "pixel_oracle": {"status": "notmeasured", "reason": "Requires independent pixel/source-marker and text oracle",
            "source_marker": None, "crop": None, "text": None, "color": None, "safe_box": None},
        "audio_oracle": {"status": "notmeasured", "reason": "No independent final loudness/flash-click oracle",
            "loudness": None, "peak": None, "flash_click_sync": None}}
