"""Utility helpers to clear generated artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import paths


def safe_clear(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clear_generated() -> None:
    for directory in (paths.clips_dir, paths.subtitles_dir, paths.frames_dir):
        safe_clear(directory)


def cleanup_after_success(audio_path: Path | None = None) -> None:
    if audio_path is not None and audio_path.exists():
        audio_path.unlink()
    if paths.frames_dir.exists():
        safe_clear(paths.frames_dir)

