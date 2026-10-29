"""Speaker evidence and source-scoped streamer identity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TypedDict

from .jobs import JobStore


@dataclass(slots=True)
class SpeakerStats:
    speaker: str
    speaking_seconds: float
    speaking_share: float
    turn_count: int
    longest_turn_seconds: float
    first_seen: float
    last_seen: float
    active_sections: int
    section_coverage: float
    ranking_hint: float


class _NormalizedTurn(TypedDict):
    start: float
    end: float
    speaker: str


def analyze_speakers(diarization_path: Path, output_path: Path, *, sections: int = 12) -> Path:
    payload = json.loads(diarization_path.read_text(encoding="utf-8"))
    raw_turns = (
        payload.get("exclusive_turns") or payload.get("turns") or []
        if isinstance(payload, dict) else payload
    )
    stats = compute_speaker_stats(raw_turns, sections=sections)
    ranked = sorted(stats, key=lambda item: item.ranking_hint, reverse=True)
    analysis = {
        "version": 1,
        "identity_status": "unresolved",
        "note": (
            "El ranking resume presencia temporal; no identifica por sí solo al streamer. "
            "Las etiquetas SPEAKER_* solo son válidas dentro de este VOD."
        ),
        "candidate": ranked[0].speaker if ranked else None,
        "speakers": [asdict(item) for item in ranked],
    }
    _write_json(output_path, analysis)
    return output_path


def compute_speaker_stats(
    turns: Iterable[dict[str, Any]],
    *,
    sections: int = 12,
) -> list[SpeakerStats]:
    normalized: list[_NormalizedTurn] = [
        {
            "start": float(turn["start"]),
            "end": float(turn["end"]),
            "speaker": str(turn["speaker"]),
        }
        for turn in turns
        if float(turn["end"]) > float(turn["start"])
    ]
    if not normalized:
        return []
    sections = max(1, sections)
    stream_end = max(turn["end"] for turn in normalized)
    section_size = stream_end / sections if stream_end > 0 else 1.0
    grouped: dict[str, list[_NormalizedTurn]] = {}
    for turn in normalized:
        grouped.setdefault(turn["speaker"], []).append(turn)
    total_speech = sum(turn["end"] - turn["start"] for turn in normalized)
    stats: list[SpeakerStats] = []
    for speaker, speaker_turns in grouped.items():
        durations = [turn["end"] - turn["start"] for turn in speaker_turns]
        speaking_seconds = sum(durations)
        occupied_sections: set[int] = set()
        for turn in speaker_turns:
            first = min(sections - 1, int(turn["start"] / section_size))
            last = min(sections - 1, int(max(turn["start"], turn["end"] - 1e-6) / section_size))
            occupied_sections.update(range(first, last + 1))
        speaking_share = speaking_seconds / total_speech if total_speech else 0.0
        section_coverage = len(occupied_sections) / sections
        turn_density = min(1.0, len(speaker_turns) / max(len(normalized) * 0.35, 1.0))
        ranking_hint = 0.50 * section_coverage + 0.35 * speaking_share + 0.15 * turn_density
        stats.append(
            SpeakerStats(
                speaker=speaker,
                speaking_seconds=round(speaking_seconds, 3),
                speaking_share=round(speaking_share, 6),
                turn_count=len(speaker_turns),
                longest_turn_seconds=round(max(durations), 3),
                first_seen=round(min(turn["start"] for turn in speaker_turns), 3),
                last_seen=round(max(turn["end"] for turn in speaker_turns), 3),
                active_sections=len(occupied_sections),
                section_coverage=round(section_coverage, 6),
                ranking_hint=round(ranking_hint, 6),
            )
        )
    return stats


def confirm_streamer_speaker(
    video_path: Path,
    speaker: str,
    *,
    display_name: str | None = None,
) -> Path:
    job = JobStore.for_video(video_path)
    if not job.speaker_analysis_path.exists():
        raise RuntimeError("Primero debes completar la diarización del VOD")
    analysis = json.loads(job.speaker_analysis_path.read_text(encoding="utf-8"))
    available = {item["speaker"] for item in analysis.get("speakers", [])}
    if speaker not in available:
        raise ValueError(f"Speaker desconocido: {speaker}. Disponibles: {sorted(available)}")
    profile = {
        "version": 1,
        "scope": "source",
        "source": str(job.source),
        "fingerprint": job.fingerprint,
        "speaker": speaker,
        "display_name": display_name,
        "method": "manual_confirmation",
        "confidence": 1.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(job.speaker_profile_path, profile)
    analysis["identity_status"] = "confirmed"
    analysis["streamer"] = profile
    _write_json(job.speaker_analysis_path, analysis)
    return job.speaker_profile_path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
