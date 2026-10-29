"""Shared utility functions used across multiple ClipperX modules."""

from __future__ import annotations

from typing import Sequence

from .schema import ClipCandidate, TranscriptSegment


def segment_overlaps_ranges(segment: TranscriptSegment, ranges: list[tuple[float, float]]) -> bool:
    """Check if a transcript segment overlaps with any of the given time ranges."""
    for start, end in ranges:
        if segment.end > start and segment.start < end:
            return True
    return False


def clip_text(clip: ClipCandidate, segments: Sequence[TranscriptSegment]) -> str:
    """Extract the full text of a clip from overlapping transcript segments."""
    ranges = clip.time_ranges or [(clip.start, clip.end)]
    parts: list[str] = []
    for segment in segments:
        if segment_overlaps_ranges(segment, ranges):
            if segment.text:
                parts.append(segment.text)
    return " ".join(parts).strip() or clip.text
