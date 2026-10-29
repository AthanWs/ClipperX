"""Media metadata probing without decoding the full source."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def probe_media(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate,format_name:stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,channels,sample_rate,start_time,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    payload = json.loads(result.stdout)
    video_stream: dict[str, Any] = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), {})
    audio_streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"]
    media_format = payload.get("format", {})
    return {
        "duration_seconds": _as_float(media_format.get("duration")),
        "size_bytes": _as_int(media_format.get("size")),
        "bit_rate": _as_int(media_format.get("bit_rate")),
        "format_name": media_format.get("format_name"),
        "video": {
            "codec": video_stream.get("codec_name"),
            "width": _as_int(video_stream.get("width")),
            "height": _as_int(video_stream.get("height")),
            "avg_frame_rate": video_stream.get("avg_frame_rate"),
            "real_frame_rate": video_stream.get("r_frame_rate"),
            "start_time": _as_float(video_stream.get("start_time")),
        },
        "audio_streams": [
            {
                "index": _as_int(stream.get("index")),
                "codec": stream.get("codec_name"),
                "channels": _as_int(stream.get("channels")),
                "sample_rate": _as_int(stream.get("sample_rate")),
                "start_time": _as_float(stream.get("start_time")),
            }
            for stream in audio_streams
        ],
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
