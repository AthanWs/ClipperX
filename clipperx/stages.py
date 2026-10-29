"""Reusable non-LLM pipeline stages backed by JobStore checkpoints."""

from __future__ import annotations

from pathlib import Path

from .audio import preprocess_audio
from .config import settings
from .jobs import JobStore


def transcription_cache_is_current(
    job: JobStore, audio_path: Path, *, language: str | None,
) -> bool:
    if not job.stage_complete("transcription", artifact=job.transcript_path):
        return False
    if settings.asr_backend != "parakeet":
        return True
    from .parakeet import transcript_is_current
    return transcript_is_current(job.transcript_path, audio_path, language=language)


def ensure_audio(video_path: Path, *, force: bool = False) -> tuple[JobStore, Path]:
    job = JobStore.for_video(video_path)
    if not force and job.stage_complete("audio", artifact=job.audio_path):
        return job, job.audio_path

    job.mark_started("audio")
    try:
        preprocess_audio(video_path, output_path=job.audio_path)
        job.mark_complete("audio", artifacts=[job.audio_path])
        return job, job.audio_path
    except Exception as exc:
        job.mark_failed("audio", str(exc))
        raise


def ensure_transcript(
    video_path: Path,
    *,
    language: str | None = None,
    force: bool = False,
) -> tuple[JobStore, Path]:
    from . import transcription

    job, audio_path = ensure_audio(video_path)
    effective_language=language if language is not None else settings.transcription_language
    if not force and transcription_cache_is_current(job, audio_path, language=effective_language):
        return job, job.transcript_path

    job.mark_started("transcription")
    try:
        transcription.transcribe_to_json(
            audio_path,
            output_dir=job.root,
            output_name="transcript",
            language=effective_language,
        )
        job.mark_complete("transcription", artifacts=[job.transcript_path])
        return job, job.transcript_path
    except Exception as exc:
        job.mark_failed("transcription", str(exc))
        raise
