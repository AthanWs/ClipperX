"""Speaker diarization and timestamp-based transcript attribution."""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .config import paths, settings
from .jobs import JobStore
from .speech_process import run_worker


@dataclass(slots=True)
class DiarizationTurn:
    start: float
    end: float
    speaker: str


def diarize_video(
    video_path: Path,
    *,
    transcript_path: Path | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    from .stages import ensure_transcript

    backend = settings.diarization_backend.strip().lower()
    if backend not in ("nemotron", "pyannote"):
        raise ValueError(f"Backend de diarización desconocido: {backend}")
    if backend == "nemotron":
        from .nemotron_worker import validate_speaker_bounds

        validate_speaker_bounds(min_speakers, max_speakers)
        model = str(_nemotron_model_path())
    else:
        model = settings.diarization_model
    job, default_transcript = ensure_transcript(video_path)
    transcript_path = transcript_path or default_transcript
    provenance = _provenance(job.audio_path, transcript_path, backend, model,
                             min_speakers, max_speakers)
    artifacts = [job.diarization_path, job.speaker_transcript_path, job.speaker_analysis_path]
    with job.stage_lock("diarization"):
        if not force and any(path.exists() for path in artifacts):
            try:
                cached = json.loads(job.diarization_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "La caché de diarización está incompleta o dañada. Usa force para "
                    "regenerarla conservando el historial."
                ) from exc
            if not isinstance(cached, dict) or cached.get("provenance") != provenance:
                raise RuntimeError(
                    "La caché de diarización corresponde a otro backend, modelo o transcript. "
                    "Usa force para regenerarla conservando el historial."
                )
            if job.stage_complete("diarization", artifact=job.diarization_path) and all(
                path.exists() for path in artifacts
            ):
                return job.diarization_path, job.speaker_transcript_path
            raise RuntimeError(
                "La etapa de diarización está incompleta. Usa force para regenerarla "
                "conservando el historial."
            )
        return _diarize_job(job, transcript_path, backend, provenance, artifacts,
                            min_speakers, max_speakers, force)


def _diarize_job(
    job: JobStore, transcript_path: Path, backend: str, provenance: dict[str, Any],
    artifacts: list[Path], min_speakers: int | None, max_speakers: int | None, force: bool,
) -> tuple[Path, Path]:
    if backend == "pyannote" and not settings.hf_token:
        raise RuntimeError(
            "Falta HF_TOKEN para descargar pyannote Community-1. "
            "La diarización no ejecuta ningún LLM."
        )
    if force and any(path.exists() for path in artifacts):
        archive = job.root / "history" / "diarization" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
        )
        archive.mkdir(parents=True, exist_ok=False)
        for path in artifacts:
            if path.exists():
                shutil.copy2(path, archive / path.name)
    job.mark_started("diarization")
    try:
        with tempfile.TemporaryDirectory(prefix=".diarization-", dir=job.root) as scratch:
            candidates = [Path(scratch) / path.name for path in artifacts]
            if backend == "nemotron":
                payload = _run_nemotron_worker(
                    job.audio_path, output_path=candidates[0], min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
            else:
                payload = _run_community_worker(
                    job.audio_path, token=settings.hf_token, model_name=settings.diarization_model,
                    output_path=candidates[0], min_speakers=min_speakers, max_speakers=max_speakers,
                )
            payload["backend"], payload["provenance"] = backend, provenance
            _write_json(candidates[0], payload)
            raw_turns = payload.get("exclusive_turns") or payload.get("turns") or []
            turns = [DiarizationTurn(**turn) for turn in raw_turns]
            assign_speakers_to_transcript(transcript_path, turns, candidates[1])
            from .speaker_identity import analyze_speakers

            analyze_speakers(candidates[0], candidates[2])
            for candidate, artifact in zip(candidates, artifacts):
                candidate.replace(artifact)
        job.mark_complete("diarization", artifacts=artifacts)
        return job.diarization_path, job.speaker_transcript_path
    except Exception as exc:
        job.mark_failed("diarization", str(exc))
        raise


def _provenance(
    audio: Path, transcript: Path, backend: str, model: str,
    minimum: int | None, maximum: int | None,
) -> dict[str, Any]:
    info = audio.stat()
    provenance: dict[str, Any] = {
        "backend": backend, "model": model,
        "audio": {"path": str(audio.resolve()), "size": info.st_size, "mtime_ns": info.st_mtime_ns},
        "transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest(),
        "min_speakers": minimum, "max_speakers": maximum,
        "attribution": "overlap_ambiguity_v1",
    }
    if backend == "nemotron":
        model_files: list[dict[str, Any]] = []
        for file in sorted(Path(model).rglob("*")):
            if not file.is_file() or ".cache" in file.parts or file.suffix not in (
                ".json", ".safetensors", ".bin", ".pt", ".nemo",
            ):
                continue
            metadata = file.stat()
            entry: dict[str, Any] = {"file": str(file.relative_to(model)), "size": metadata.st_size,
                                     "mtime_ns": metadata.st_mtime_ns}
            if file.suffix == ".json":
                entry["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            model_files.append(entry)
        provenance["model_files"] = model_files
    return provenance


def _nemotron_model_path() -> Path:
    path = Path(settings.nemotron_model_path)
    return (path if path.is_absolute() else paths.base_dir / path).resolve()


def _run_nemotron_worker(
    audio_path: Path, *, output_path: Path,
    min_speakers: int | None, max_speakers: int | None,
) -> dict[str, Any]:
    from .nemotron_worker import validate_speaker_bounds
    from ._media_types import load_soundfile

    validate_speaker_bounds(min_speakers, max_speakers)
    configured = settings.nvidia_speech_python.strip()
    python = Path(configured) if configured else (
        paths.base_dir / ".venv-nvidia-speech" / "Scripts" / "python.exe"
    )
    if not python.is_file():
        raise RuntimeError("Falta el runtime aislado NVIDIA speech para Nemotron")
    model = _nemotron_model_path()
    if not model.is_dir():
        raise RuntimeError(f"Falta el modelo Nemotron local: {model}")
    command = [str(python), "-m", "clipperx.nemotron_worker", "run", "--audio", str(audio_path),
               "--output", str(output_path), "--model", str(model)]
    for name, value in (("--min-speakers", min_speakers), ("--max-speakers", max_speakers)):
        if value is not None:
            command.extend([name, str(value)])
    environment = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        environment.pop(key, None)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    project_root = str(paths.base_dir)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = project_root if not existing else os.pathsep.join([project_root, existing])
    with load_soundfile().SoundFile(str(audio_path)) as audio:
        duration = audio.frames / audio.samplerate
    timeout = min(86400.0, max(400.0, duration * 4.0 + 120.0))
    try:
        result = run_worker(command, cwd=paths.base_dir, env=environment, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Nemotron worker excedió el límite de {timeout:.0f} segundos") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Nemotron worker falló: {detail}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def _run_community_worker(
    audio_path: Path,
    *,
    output_path: Path,
    token: str,
    model_name: str,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict[str, Any]:
    python = _diarization_python()
    command = [
        str(python),
        "-m",
        "clipperx.diarization_worker",
        "run",
        "--audio",
        str(audio_path),
        "--output",
        str(output_path),
        "--model",
        model_name,
    ]
    if min_speakers is not None:
        command.extend(["--min-speakers", str(min_speakers)])
    if max_speakers is not None:
        command.extend(["--max-speakers", str(max_speakers)])
    environment = os.environ.copy()
    environment["HF_TOKEN"] = token
    project_root = str(paths.base_dir)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        project_root if not existing_pythonpath else os.pathsep.join([project_root, existing_pythonpath])
    )
    result = subprocess.run(
        command,
        cwd=paths.base_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Community-1 worker falló: {detail}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def _diarization_python() -> Path:
    configured = settings.diarization_python.strip()
    python = Path(configured) if configured else paths.base_dir / ".venv-diarization" / "Scripts" / "python.exe"
    if not python.exists():
        raise RuntimeError(
            "Falta el runtime aislado de Community-1. Ejecuta "
            "scripts\\setup_diarization_env.ps1 antes de diarizar."
        )
    return python


def assign_speakers_to_transcript(
    transcript_path: Path,
    turns: Iterable[DiarizationTurn],
    output_path: Path,
) -> Path:
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    ordered_turns = sorted(turns, key=lambda turn: (turn.start, turn.end))
    starts = [turn.start for turn in ordered_turns]
    latest_ends: list[float] = []
    latest = -math.inf
    for turn in ordered_turns:
        latest = max(latest, turn.end)
        latest_ends.append(latest)
    for segment in transcript:
        _attribute(segment, ordered_turns, starts, latest_ends)
        for word in segment.get("words", []):
            _attribute(word, ordered_turns, starts, latest_ends)
    _write_json(output_path, transcript)
    return output_path


def _attribute(
    item: dict[str, Any], turns: list[DiarizationTurn], starts: list[float], latest_ends: list[float],
) -> None:
    start, end = item.get("start"), item.get("end")
    item["speaker"] = None
    item.pop("speaker_candidates", None)
    item.pop("speaker_status", None)
    if start is None or end is None:
        return
    start, end = float(start), float(end)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return
    left, right = bisect_right(latest_ends, start), bisect_left(starts, end)
    active = [turn for turn in turns[left:right] if turn.end > start]
    item["speaker"] = _best_speaker(start, end, active)
    ambiguous = any(
        first.speaker != second.speaker
        and max(first.start, second.start, start) < min(first.end, second.end, end)
        for index, first in enumerate(active) for second in active[index + 1:]
    )
    if ambiguous:
        item["speaker"] = None
        item["speaker_candidates"] = sorted({turn.speaker for turn in active})
        item["speaker_status"] = "overlap_ambiguous"


def _best_speaker(
    start: Any,
    end: Any,
    turns: list[DiarizationTurn],
) -> str | None:
    if start is None or end is None:
        return None
    start_value = float(start)
    end_value = float(end)
    if not math.isfinite(start_value) or not math.isfinite(end_value) or end_value <= start_value:
        return None
    overlaps: dict[str, float] = {}
    for turn in turns:
        if turn.start >= end_value:
            break
        if turn.end <= start_value:
            continue
        overlap = min(end_value, turn.end) - max(start_value, turn.start)
        if overlap > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap
    if not overlaps:
        return None
    return max(overlaps, key=overlaps.__getitem__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
