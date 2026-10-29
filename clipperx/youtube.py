"""YouTube video download support via yt-dlp."""

from __future__ import annotations

import re
import time
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

_YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be)"
    r"(?:/(?:watch|embed|shorts|live|v))?(?:\?.*v=|/)?"
    r"[A-Za-z0-9_-]{11}"
)


def is_youtube_url(url: str) -> bool:
    """Check if a string is a YouTube URL."""
    return bool(_YOUTUBE_RE.match(url.strip()))


def get_video_info(url: str) -> dict:
    """Return basic metadata (title, duration, thumbnail) without downloading."""
    import yt_dlp  # type: ignore

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info is None:
        return {}
    return {
        "title": info.get("title", ""),
        "duration": info.get("duration", 0),
        "thumbnail": info.get("thumbnail", ""),
    }


def download_video(
    url: str,
    output_dir: Path,
    *,
    max_retries: int = 3,
    min_duration: int = 60,
    max_duration: int = 43_200,
) -> Path:
    """Download a YouTube video and return the local path.

    Uses yt-dlp as a library. Retries with exponential backoff on failure.
    Validates duration is between ``min_duration`` and ``max_duration`` seconds.
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check duration first
    info = get_video_info(url)
    duration = info.get("duration", 0)
    if duration and (duration < min_duration or duration > max_duration):
        raise ValueError(
            f"Video duration ({duration}s) outside allowed range "
            f"({min_duration}-{max_duration}s)"
        )

    logger.info(
        "[youtube] descargando video",
        extra={"url": url, "title": info.get("title", ""), "duration": duration},
    )

    outtmpl = str(output_dir / "%(title)s.%(ext)s")
    opts = {
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                result = ydl.extract_info(url, download=True)
            if result is None:
                raise RuntimeError("yt-dlp returned no result")
            # Locate the downloaded file
            filename = ydl.prepare_filename(result)
            # yt-dlp may change extension after merge
            downloaded = Path(filename).with_suffix(".mp4")
            if not downloaded.exists():
                downloaded = Path(filename)
            if not downloaded.exists():
                # Search output_dir for the most recent mp4
                mp4s = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
                if mp4s:
                    downloaded = mp4s[-1]
                else:
                    raise FileNotFoundError(f"Downloaded file not found: {filename}")
            logger.info("[youtube] descarga completada", extra={"path": str(downloaded)})
            return downloaded
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "[youtube] reintentando descarga",
                    extra={"attempt": attempt, "wait": wait, "error": str(exc)},
                )
                time.sleep(wait)

    raise RuntimeError(f"Failed to download after {max_retries} attempts: {last_exc}")
