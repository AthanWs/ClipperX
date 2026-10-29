"""Audio preprocessing utilities."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
import tempfile
from typing import Sequence
from typing_extensions import TypedDict, Unpack

from .local_paths import paths


logger = logging.getLogger(__name__)


class _PreprocessOptions(TypedDict, total=False):
    output_path: Path | None
    loudnorm: bool
    channels: int
    sample_rate: int
    start: float | None
    end: float | None


def build_ffmpeg_cmd(
    input_path: Path,
    output_path: Path,
    *,
    loudnorm: bool = False,
    channels: int = 1,
    sample_rate: int = 16_000,
    start: float | None = None,
    end: float | None = None,
) -> list[str]:
    """Build ffmpeg command for audio normalization."""

    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if start is not None:
        cmd.extend(["-ss", f"{start:.6f}"])
    cmd.extend([
        "-i",
        str(input_path),
    ])
    if start is not None and end is not None:
        cmd.extend(["-t", f"{end - start:.6f}"])
    cmd.extend([
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
    ])
    if loudnorm:
        cmd.extend(["-af", "loudnorm"])
    cmd.append(str(output_path))
    return cmd


def preprocess_audio(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    loudnorm: bool = False,
    channels: int = 1,
    sample_rate: int = 16_000,
    start: float | None = None,
    end: float | None = None,
) -> Path:
    """Extract mono PCM audio ready for ASR."""

    if output_path is None:
        if output_dir is None:
            output_dir = paths.audio_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_{sample_rate//1000}k.wav"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[audio] normalizando", extra={"input": str(input_path), "output": str(output_path)})
    cmd = build_ffmpeg_cmd(
        input_path,
        output_path,
        loudnorm=loudnorm,
        channels=channels,
        sample_rate=sample_rate,
        start=start,
        end=end,
    )
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def _atomic_audio_extract(
    input_path: Path,
    output_path: Path,
    *,
    start: float | None = None,
    end: float | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix or ".flac",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        preprocess_audio(
            input_path,
            output_path=temporary,
            start=start,
            end=end,
        )
        temporary.replace(output_path)
        return output_path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_audio_checkpoint(
    input_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> Path:
    """Create or reuse a local audio checkpoint without settings/providers."""

    output_path = Path(output_path)
    if output_path.is_file() and not force:
        return output_path
    return _atomic_audio_extract(input_path, output_path)


def extract_audio_window(
    input_path: Path,
    output_path: Path,
    *,
    start: float,
    end: float,
) -> Path:
    """Decode only one validated source interval into a local audio file."""

    if start < 0 or end <= start:
        raise ValueError("audio window requires 0 <= start < end")
    return _atomic_audio_extract(
        input_path,
        output_path,
        start=float(start),
        end=float(end),
    )


def batch_preprocess(
    inputs: Sequence[Path],
    *,
    output_dir: Path | None = None,
    **kwargs: Unpack[_PreprocessOptions],
) -> list[Path]:
    """Normalize multiple inputs returning generated paths."""

    return [
        preprocess_audio(path, output_dir=output_dir, **kwargs)
        for path in inputs
    ]


