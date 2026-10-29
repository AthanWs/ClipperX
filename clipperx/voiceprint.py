"""Persistent streamer voice enrollment and per-VOD speaker matching."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import paths, settings
from .diarization import _diarization_python
from .jobs import JobStore
from .stages import ensure_audio


def enroll_streamer_voice(
    video_path: Path,
    *,
    start: float,
    end: float,
    name: str,
) -> Path:
    if end - start < 3.0:
        raise ValueError("Usa al menos 3 segundos de voz limpia para el voiceprint")
    job, audio_path = ensure_audio(video_path)
    embedding_path = job.root / "voiceprint_enrollment.json"
    _run_worker(
        [
            "embed",
            "--audio",
            str(audio_path),
            "--output",
            str(embedding_path),
            "--model",
            settings.speaker_embedding_model,
            "--start",
            str(start),
            "--end",
            str(end),
        ]
    )
    embedding_payload = json.loads(embedding_path.read_text(encoding="utf-8"))
    profile_path = paths.profiles_dir / f"{_slug(name)}.voiceprint.json"
    profile = {
        "version": 1,
        "name": name,
        "model": settings.speaker_embedding_model,
        "embedding": embedding_payload["embedding"],
        "enrollment": {
            "source": str(job.source),
            "fingerprint": job.fingerprint,
            "start": start,
            "end": end,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(profile_path, profile)
    return profile_path


def match_streamer_voice(video_path: Path, profile_path: Path) -> Path:
    job = JobStore.for_video(video_path)
    if not job.diarization_path.exists():
        raise RuntimeError("Primero debes diarizar el VOD")
    _, audio_path = ensure_audio(video_path)
    output_path = job.root / "voiceprint_matches.json"
    _run_worker(
        [
            "match",
            "--audio",
            str(audio_path),
            "--diarization",
            str(job.diarization_path),
            "--profile",
            str(profile_path),
            "--output",
            str(output_path),
            "--model",
            settings.speaker_embedding_model,
        ]
    )
    matches = json.loads(output_path.read_text(encoding="utf-8"))
    if job.speaker_analysis_path.exists():
        analysis = json.loads(job.speaker_analysis_path.read_text(encoding="utf-8"))
        analysis["voiceprint"] = matches
        analysis["identity_status"] = "voiceprint_candidate"
        _write_json(job.speaker_analysis_path, analysis)
    return output_path


def _run_worker(arguments: list[str]) -> None:
    environment = os.environ.copy()
    if settings.hf_token:
        environment["HF_TOKEN"] = settings.hf_token
    project_root = str(paths.base_dir)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        project_root if not existing_pythonpath else os.pathsep.join([project_root, existing_pythonpath])
    )
    result = subprocess.run(
        [str(_diarization_python()), "-m", "clipperx.diarization_worker", *arguments],
        cwd=paths.base_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower() or "streamer"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
