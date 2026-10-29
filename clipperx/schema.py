"""Data structures shared across the ClipperX pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class Word:
    """Word-level timestamp information."""

    start: float
    end: float
    word: str
    probability: float | None = None
    speaker: str | None = None
    speaker_candidates: list[str] = field(default_factory=list)
    speaker_status: str | None = None


@dataclass(slots=True)
class TranscriptSegment:
    """Segment returned by the ASR engine."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        words = [
            Word(**word)
            for word in data.get("words", [])
            if word.get("start") is not None and word.get("end") is not None
        ]
        return cls(
            start=data["start"],
            end=data["end"],
            text=data["text"],
            words=words,
            speaker=data.get("speaker"),
        )


@dataclass(slots=True)
class ViralityScore:
    """Multi-dimensional virality scoring for a clip."""

    hook: int = 0           # 0-25
    engagement: int = 0     # 0-25
    value: int = 0          # 0-25
    shareability: int = 0   # 0-25
    total: int = 0          # 0-100
    hook_type: str = "none"  # question|statement|statistic|story|contrast|action|none
    reasoning: str = ""
    emotion: str = "none"   # alta activacion: humor|outrage|awe|suspense|curiosity|none


@dataclass(slots=True)
class BRollInsert:
    """Describes a B-roll insertion point within a clip."""

    timestamp: float       # Where to insert (relative to clip start)
    duration: float        # Duration of the B-roll
    search_term: str       # Keyword for Pexels search
    video_path: Path | None = None  # Local path after download


@dataclass(slots=True)
class ClipCandidate:
    """Candidate clip generated during segmentation."""

    start: float
    end: float
    text: str
    coherence: float
    headline: str | None = None
    keywords: list[str] = field(default_factory=list)
    segment_range: tuple[int, int] | None = None
    layout: str | None = None
    layout_segments: list[tuple[float, float, str]] | None = None
    time_ranges: list[tuple[float, float]] | None = None
    face_crop: tuple[int, int, int, int] | None = None
    webcam_region: tuple[int, int, int, int, str] | None = None  # x, y, w, h, corner
    segment_face_crops: dict[int, tuple[int, int, int, int] | None] | None = None  # explicit None rejects inherited crop
    virality: ViralityScore | None = None
    broll_inserts: list[BRollInsert] = field(default_factory=list)
    segment_content_crops: dict[int, tuple[int, int, int, int]] | None = None  # explicit original-source content ROI

    @property
    def duration(self) -> float:
        if self.time_ranges:
            return sum(max(0.0, end - start) for start, end in self.time_ranges)
        return self.end - self.start


def words_to_text(words: Iterable[Word]) -> str:
    """Join words into a sanitized string."""

    return "".join(word.word for word in words).replace("  ", " ").strip()


@dataclass(slots=True)
class SilenceWindow:
    start: float
    end: float
    reason: str | None = None
    relevant: bool = False
    layout: str | None = None
    description: str | None = None
    category: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start
