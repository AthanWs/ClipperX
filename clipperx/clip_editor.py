"""Post-render clip editing operations."""

from __future__ import annotations

from pathlib import Path

from ._media_types import load_ffmpeg
ffmpeg = load_ffmpeg()

from .logger import get_logger  # noqa: E402 - ffmpeg stays eager before logger

logger = get_logger(__name__)


def trim_clip(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
) -> Path:
    """Trim a clip to the range [start, end] seconds."""
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid trim range: start={start}, end={end}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = ffmpeg.input(str(input_path), ss=start, t=duration)
    out = ffmpeg.output(
        stream.video,
        stream.audio,
        str(output_path),
        vcodec="copy",
        acodec="copy",
        movflags="+faststart",
    )
    ffmpeg.run(out, quiet=True, overwrite_output=True)
    logger.info("[editor] trimmed", extra={"output": str(output_path)})
    return output_path


def split_clip(
    input_path: Path,
    output_dir: Path,
    split_at: float,
) -> tuple[Path, Path]:
    """Split a clip at *split_at* seconds into two files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    part1 = output_dir / f"{stem}_part1.mp4"
    part2 = output_dir / f"{stem}_part2.mp4"

    # Part 1: 0 -> split_at
    s1 = ffmpeg.input(str(input_path), t=split_at)
    ffmpeg.output(
        s1.video, s1.audio, str(part1),
        vcodec="copy", acodec="copy", movflags="+faststart",
    ).run(quiet=True, overwrite_output=True)

    # Part 2: split_at -> end
    s2 = ffmpeg.input(str(input_path), ss=split_at)
    ffmpeg.output(
        s2.video, s2.audio, str(part2),
        vcodec="copy", acodec="copy", movflags="+faststart",
    ).run(quiet=True, overwrite_output=True)

    logger.info("[editor] split", extra={"part1": str(part1), "part2": str(part2)})
    return part1, part2


def merge_clips(
    input_paths: list[Path],
    output_path: Path,
) -> Path:
    """Concatenate multiple clips into one file."""
    if not input_paths:
        raise ValueError("No input paths provided")
    if len(input_paths) == 1:
        import shutil
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_paths[0], output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = [ffmpeg.input(str(p)) for p in input_paths]
    concat_node = ffmpeg.concat(*inputs, v=1, a=1).node
    out = ffmpeg.output(
        concat_node[0],
        concat_node[1],
        str(output_path),
        vcodec="libx264",
        acodec="aac",
        crf=18,
        preset="fast",
        movflags="+faststart",
    )
    ffmpeg.run(out, quiet=True, overwrite_output=True)
    logger.info("[editor] merged", extra={"output": str(output_path), "inputs": len(input_paths)})
    return output_path


def adjust_speed(
    input_path: Path,
    output_path: Path,
    factor: float,
) -> Path:
    """Change playback speed by *factor* (e.g. 1.5 = 50% faster)."""
    if factor <= 0:
        raise ValueError(f"Speed factor must be positive, got {factor}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = ffmpeg.input(str(input_path))

    video = stream.video.filter("setpts", f"PTS/{factor}")

    # atempo only accepts 0.5-100.0; chain filters for extreme values
    audio = stream.audio
    remaining = factor
    while remaining > 2.0:
        audio = audio.filter("atempo", 2.0)
        remaining /= 2.0
    while remaining < 0.5:
        audio = audio.filter("atempo", 0.5)
        remaining /= 0.5
    audio = audio.filter("atempo", remaining)

    out = ffmpeg.output(
        video,
        audio,
        str(output_path),
        vcodec="libx264",
        acodec="aac",
        crf=18,
        preset="fast",
        movflags="+faststart",
    )
    ffmpeg.run(out, quiet=True, overwrite_output=True)
    logger.info("[editor] speed adjusted", extra={"factor": factor, "output": str(output_path)})
    return output_path


def adjust_volume(
    input_path: Path,
    output_path: Path,
    factor: float,
) -> Path:
    """Adjust audio volume by *factor* (e.g. 1.5 = 50% louder)."""
    if factor < 0:
        raise ValueError(f"Volume factor must be non-negative, got {factor}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = ffmpeg.input(str(input_path))
    out = ffmpeg.output(
        stream.video,
        stream.audio.filter("volume", factor),
        str(output_path),
        vcodec="copy",
        acodec="aac",
        movflags="+faststart",
    )
    ffmpeg.run(out, quiet=True, overwrite_output=True)
    logger.info("[editor] volume adjusted", extra={"factor": factor, "output": str(output_path)})
    return output_path
