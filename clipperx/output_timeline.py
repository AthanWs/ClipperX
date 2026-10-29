"""Exact, occurrence-aware source/output mapping without inference or I/O.

Supported compiler subset: duration equality and output continuity use a 1e-9
second representation epsilon, stricter than the unchanged v2 schema's 0.001s.
Schema validity alone therefore does not imply renderability. Reject mismatches
explicitly; never rescale source time or snap original source/output endpoints.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Literal

from .render_clock import exact
from .schema import TranscriptSegment
from .visual_contracts import VisualShot


def _operand(value: float) -> Fraction:
    """Retain each validated decimal operand before doing clock arithmetic."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Timeline coordinates must be finite nonnegative numbers")
    return exact(str(value))


def _fraction(value: Fraction) -> Fraction:
    if not isinstance(value, Fraction) or value < 0:
        raise ValueError("Time requires nonnegative Fraction metadata")
    return exact(value)


def _qv(value: Fraction) -> int:
    return (120 * value.numerator + value.denominator) // (2 * value.denominator)


def _exact_interval(value: tuple[Fraction, Fraction] | None) -> tuple[Fraction, Fraction] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Exact interval requires two Fraction endpoints")
    start, end = (_fraction(v) for v in value)
    if start >= end:
        raise ValueError("Exact interval must be positive and ordered")
    return start, end


def _exact_word_interval(value: tuple[Fraction, Fraction] | None
                         ) -> tuple[Fraction, Fraction] | None:
    """Native lexical anchors may be points; display/event intervals may not."""
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Exact word interval requires two Fraction endpoints")
    start, end = (_fraction(v) for v in value)
    if start > end:
        raise ValueError("Exact word interval must be ordered")
    return start, end


def _map_coordinate(timestamp: Fraction, source_start: Fraction,
                    output_start: Fraction) -> Fraction:
    return output_start + timestamp - source_start


@dataclass(frozen=True, slots=True)
class RasterSpan:
    occurrence_index: int
    shot_id: str
    source_start: Fraction
    source_end: Fraction
    output_start: Fraction
    output_end: Fraction
    k_start: int
    k_end: int

    def __post_init__(self) -> None:
        for name in ("occurrence_index", "k_start", "k_end"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("Raster indices require nonnegative integers")
        for name in ("source_start", "source_end", "output_start", "output_end"):
            object.__setattr__(self, name, _fraction(getattr(self, name)))
        if (type(self.shot_id) is not str or not self.shot_id
                or self.source_start >= self.source_end or self.output_start >= self.output_end
                or self.k_start != _qv(self.output_start) or self.k_end != _qv(self.output_end)
                or self.k_start >= self.k_end):
            raise ValueError(f"Invalid raster span {self.shot_id}: "
                             f"[{self.output_start}, {self.output_end}) allocates "
                             f"[{self.k_start}, {self.k_end})")

    @property
    def frame_count(self) -> int:
        return self.k_end - self.k_start

    @property
    def start_delta(self) -> Fraction:
        return Fraction(self.k_start, 60) - self.output_start

    @property
    def end_delta(self) -> Fraction:
        return Fraction(self.k_end, 60) - self.output_end

    @property
    def duration_delta(self) -> Fraction:
        return Fraction(self.frame_count, 60) - (self.output_end - self.output_start)


@dataclass(frozen=True, slots=True)
class VideoSampleRequest:
    global_k: int
    occurrence_index: int
    shot_id: str
    local_n: int
    requested_source: Fraction
    request_rule: Literal["declared_start_for_leading_raster_cell",
                          "inverse_declared_mapper_at_global_pts"]

    def __post_init__(self) -> None:
        for name in ("global_k", "occurrence_index", "local_n"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("Video request indices require nonnegative integers")
        object.__setattr__(self, "requested_source", _fraction(self.requested_source))
        if (type(self.shot_id) is not str or not self.shot_id or self.request_rule not in (
                "declared_start_for_leading_raster_cell", "inverse_declared_mapper_at_global_pts")):
            raise ValueError("Invalid video request identity/rule")


def join_words(words: Sequence[str]) -> str:
    """Join ASR tokens with punctuation spacing; keep explicit line breaks."""
    result = ""
    for raw in words:
        word = raw.strip()
        if not word:
            continue
        no_space = (not result or word[0] in ",.;:!?%)]}’'"
                    or result[-1] in "¿¡([{’'")
        result += ("" if no_space else " ") + word
    return result


@dataclass(frozen=True, slots=True)
class TimelineSpan:
    shot_id: str
    source_start: float
    source_end: float
    output_start: float
    output_end: float

    def __post_init__(self) -> None:
        values = (self.source_start, self.source_end, self.output_start, self.output_end)
        for value in values:
            _operand(value)
        if (type(self.shot_id) is not str or not self.shot_id
                or self.source_end <= self.source_start or self.output_end <= self.output_start
                or abs((self.source_end - self.source_start)
                       - (self.output_end - self.output_start)) > 1e-9):
            raise ValueError("Invalid timeline span; source/output durations must match exactly")


@dataclass(frozen=True, slots=True)
class MappedWord:
    start: float
    end: float
    text: str
    exact_interval: tuple[Fraction, Fraction] | None = None

    def __post_init__(self) -> None:
        interval = _exact_word_interval(self.exact_interval)
        if (interval is not None and interval[0] == interval[1]
                and (self.start, self.end) != (float(interval[0]), float(interval[1]))):
            raise ValueError("Native word point must match its acoustic float projection")
        object.__setattr__(self, "exact_interval", interval)


@dataclass(frozen=True, slots=True)
class MappedEvent:
    shot_id: str
    start: float
    end: float
    text: str
    words: tuple[MappedWord, ...] = ()
    exact_interval: tuple[Fraction, Fraction] | None = None
    # Exact owning shot limits, independent of the acoustic word intervals.
    output_bounds: tuple[Fraction, Fraction] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "words", tuple(self.words))
        object.__setattr__(self, "exact_interval", _exact_interval(self.exact_interval))
        object.__setattr__(self, "output_bounds", _exact_interval(self.output_bounds))


@dataclass(frozen=True, slots=True)
class OutputTimeline:
    spans: tuple[TimelineSpan, ...]

    def __post_init__(self) -> None:
        spans = tuple(self.spans)
        previous = 0.0
        seen: set[str] = set()
        if not spans:
            raise ValueError("Timeline requires shots")
        for span in spans:
            if span.shot_id in seen or abs(span.output_start - previous) > 1e-9:
                raise ValueError("Timeline requires unique shots and continuous output from zero")
            seen.add(span.shot_id)
            previous = span.output_end
        object.__setattr__(self, "spans", spans)

    @classmethod
    def from_shots(cls, shots: Sequence[VisualShot]) -> "OutputTimeline":
        """Validate the exact-anchor subset before downstream tools/assembly.

        Schema-valid duration/continuity differences above 1e-9 seconds are
        unsupported, even when below its 1ms tolerance. No timing is repaired.
        """
        return cls(tuple(TimelineSpan(s.shot_id, s.source.start, s.source.end,
                                      s.output.start, s.output.end) for s in shots))

    def source_to_outputs(self, timestamp: float, *, shot_id: str | None = None
                          ) -> tuple[float, ...]:
        time = _operand(timestamp)
        if shot_id is not None and not any(s.shot_id == shot_id for s in self.spans):
            raise ValueError("Unknown shot_id")
        outputs: list[float] = []
        for span in self.spans:
            if shot_id is None or span.shot_id == shot_id:
                s0, s1, o0 = (_operand(v) for v in (
                    span.source_start, span.source_end, span.output_start))
                if s0 <= time < s1:
                    outputs.append(float(_map_coordinate(time, s0, o0)))
        return tuple(outputs)

    def source_to_output(self, timestamp: float, *, shot_id: str | None = None) -> float | None:
        matches = self.source_to_outputs(timestamp, shot_id=shot_id)
        if len(matches) > 1:
            raise ValueError("Source timestamp has ambiguous output occurrences; supply shot_id")
        return matches[0] if matches else None

    def raster_spans(self) -> tuple[RasterSpan, ...]:
        spans: list[RasterSpan] = []
        previous = 0
        for index, span in enumerate(self.spans):
            s0, s1, o0, o1 = (_operand(value) for value in (
                span.source_start, span.source_end, span.output_start, span.output_end))
            raster = RasterSpan(index, span.shot_id, s0, s1, o0, o1, _qv(o0), _qv(o1))
            if raster.k_start != previous:
                raise ValueError(f"Exact raster continuity required at {span.shot_id}")
            spans.append(raster)
            previous = raster.k_end
        return tuple(spans)

    def video_request(self, global_k: int) -> VideoSampleRequest:
        if type(global_k) is not int or global_k < 0:
            raise ValueError("Global raster index requires a nonnegative integer")
        for span in self.raster_spans():
            if span.k_start <= global_k < span.k_end:
                pts = Fraction(global_k, 60)
                representative = max(pts, span.output_start)
                requested = span.source_start + representative - span.output_start
                if not span.source_start <= requested < span.source_end:
                    raise ValueError(f"Video request outside original source span {span.shot_id}")
                return VideoSampleRequest(global_k, span.occurrence_index, span.shot_id,
                    global_k - span.k_start, requested,
                    "declared_start_for_leading_raster_cell" if pts < span.output_start
                    else "inverse_declared_mapper_at_global_pts")
        raise ValueError("Global raster index outside output")

    def raster_window(self, start: Fraction, end: Fraction, *, shot_id: str) -> tuple[int, int]:
        start, end = _fraction(start), _fraction(end)
        for span in self.raster_spans():
            if span.shot_id == shot_id:
                k0, k1 = _qv(start), _qv(end)
                if (not span.output_start <= start < end <= span.output_end
                        or not span.k_start <= k0 < k1 <= span.k_end):
                    raise ValueError("Interval outside occurrence or collapses to zero raster cells")
                return k0, k1
        raise ValueError("Unknown shot_id")

    def overlay_output_interval(self, shot_id: str, local_start: float,
                                local_end: float) -> tuple[Fraction, Fraction]:
        start, end = _operand(local_start), _operand(local_end)
        for span in self.spans:
            if span.shot_id == shot_id:
                o0, o1 = _operand(span.output_start), _operand(span.output_end)
                if not start < end <= o1 - o0:
                    raise ValueError("Overlay interval outside declared occurrence")
                return o0 + start, o0 + end
        raise ValueError("Unknown shot_id")

    def remap(self, transcript: Sequence[TranscriptSegment] | None) -> tuple[MappedEvent, ...]:
        if transcript is None:
            raise ValueError("Missing subtitle data; use an empty sequence for known silence")
        previous = Fraction(-1)
        segments: list[tuple[Fraction, Fraction, str, tuple[tuple[Fraction, Fraction, str], ...]]] = []
        for segment in transcript:
            if (not _valid_range(segment.start, segment.end)
                    or not isinstance(segment.text, str)):
                raise ValueError("Invalid subtitle segment timing/text")
            a, b = _operand(segment.start), _operand(segment.end)
            if a < previous:
                raise ValueError("Invalid subtitle segment timing/text")
            previous = a
            word_start = a
            timed_words: list[tuple[Fraction, Fraction, str]] = []
            for word in segment.words:
                if (not _valid_word_range(word.start, word.end) or not isinstance(word.word, str)):
                    raise ValueError("Invalid subtitle word timing/text")
                left, right = _operand(word.start), _operand(word.end)
                if left < word_start or right > b:
                    raise ValueError("Invalid subtitle word timing/text")
                word_start = left
                timed_words.append((left, right, word.word))
            segments.append((a, b, segment.text, tuple(timed_words)))
        events: list[MappedEvent] = []
        for span in self.spans:
            s0, s1, o0 = (_operand(value) for value in (
                span.source_start, span.source_end, span.output_start))
            output_bounds = (o0, _operand(span.output_end))
            for a, b, text, timed in segments:
                start, end = max(a, s0), min(b, s1)
                words_list: list[MappedWord] = []
                for left, right, token in timed:
                    point = left == right and s0 <= left < s1
                    if token.strip() and (point or left < end and right > start):
                        interval = ((_map_coordinate(left, s0, o0),) * 2 if point else
                                    (_map_coordinate(max(left, start), s0, o0),
                                     _map_coordinate(min(right, end), s0, o0)))
                        words_list.append(MappedWord(float(interval[0]), float(interval[1]),
                                                     token, interval))
                words = tuple(words_list)
                if words:
                    intervals = tuple(w.exact_interval for w in words)
                    assert all(interval is not None for interval in intervals)
                    first = words[0].exact_interval
                    assert first is not None
                    interval = (first[0], max(i[1] for i in intervals if i is not None))
                    if interval[0] == interval[1]:
                        # A point-only transcript has no positive acoustic span.
                        # Retain its known segment envelope (or owning shot when
                        # its terminal anchor lies exactly at the source cut).
                        # Word anchors stay exact; display dwell is chosen later.
                        interval = ((_map_coordinate(start, s0, o0),
                                     _map_coordinate(end, s0, o0)) if start < end
                                    else output_bounds)
                    events.append(MappedEvent(span.shot_id, float(interval[0]), float(interval[1]),
                        join_words([w.text for w in words]), words, interval, output_bounds))
                elif start < end and not timed and text.strip():
                    interval = (_map_coordinate(start, s0, o0), _map_coordinate(end, s0, o0))
                    events.append(MappedEvent(span.shot_id, float(interval[0]), float(interval[1]),
                        text.strip(), exact_interval=interval, output_bounds=output_bounds))
        return tuple(events)


def _valid_range(start: float, end: float) -> bool:
    return (type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(start) and math.isfinite(end) and 0 <= start < end)


def _valid_word_range(start: float, end: float) -> bool:
    return (type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end)
