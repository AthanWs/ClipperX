"""Transcript overlap and clip text helpers."""

from clipperx.helpers import clip_text, segment_overlaps_ranges
from clipperx.schema import ClipCandidate, TranscriptSegment


def segment(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def test_overlap_is_strict_at_range_edges() -> None:
    assert segment_overlaps_ranges(segment(1.0, 3.0, "a"), [(2.0, 4.0)])
    assert segment_overlaps_ranges(segment(1.0, 5.0, "a"), [(2.0, 4.0)])
    assert not segment_overlaps_ranges(segment(0.0, 2.0, "a"), [(2.0, 4.0)])
    assert not segment_overlaps_ranges(segment(4.0, 6.0, "a"), [(2.0, 4.0)])
    assert not segment_overlaps_ranges(segment(1.0, 3.0, "a"), [])


def test_overlap_checks_every_range() -> None:
    ranges = [(0.0, 1.0), (10.0, 12.0)]
    assert segment_overlaps_ranges(segment(11.0, 13.0, "a"), ranges)
    assert not segment_overlaps_ranges(segment(1.0, 10.0, "a"), ranges)


def test_clip_text_joins_overlapping_segments_in_order() -> None:
    clip = ClipCandidate(start=1.0, end=5.0, text="fallback", coherence=1.0)
    segments = [segment(0.0, 1.0, "before"), segment(1.0, 2.0, "hola"),
                segment(2.0, 3.0, ""), segment(4.5, 6.0, "mundo"),
                segment(5.0, 7.0, "after")]
    assert clip_text(clip, segments) == "hola mundo"


def test_clip_text_uses_time_ranges_instead_of_bounds() -> None:
    clip = ClipCandidate(start=0.0, end=20.0, text="fallback", coherence=1.0,
                         time_ranges=[(0.0, 2.0), (10.0, 12.0)])
    segments = [segment(0.0, 2.0, "setup"), segment(5.0, 6.0, "cut"),
                segment(10.0, 12.0, "payoff")]
    assert clip_text(clip, segments) == "setup payoff"


def test_clip_text_falls_back_to_candidate_text() -> None:
    clip = ClipCandidate(start=1.0, end=2.0, text="fallback", coherence=1.0)
    assert clip_text(clip, [segment(5.0, 6.0, "far away")]) == "fallback"
    assert clip_text(clip, []) == "fallback"


def test_duration_sums_time_ranges() -> None:
    assert ClipCandidate(start=2.0, end=7.5, text="", coherence=0.0).duration == 5.5
    clip = ClipCandidate(start=0.0, end=20.0, text="", coherence=0.0,
                         time_ranges=[(0.0, 2.0), (10.0, 12.5), (15.0, 14.0)])
    assert clip.duration == 4.5
