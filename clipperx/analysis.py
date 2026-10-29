"""Visual relevance analysis combining WhisperX VAD with Ollama vision."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, SupportsFloat, SupportsIndex, cast

import numpy as np

from .audio_features import AudioFeatureExtractor
from .config import paths
from .helpers import segment_overlaps_ranges, clip_text
from .llm import analyze_boundary_decision, analyze_gap_decision, analyze_visual_relevance
from .schema import ClipCandidate, SilenceWindow, TranscriptSegment
from .segmentation import extract_silences, load_embedder
from .logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class VisualDecision:
    silence: SilenceWindow
    relevant: bool
    reason: str | None = None
    layout: str | None = None
    description: str | None = None
    category: str | None = None


@dataclass(slots=True)
class RefineConfig:
    min_gap: float = 0.9
    max_duration: float = 179.0
    min_clip_duration: float = 12.0
    max_extend: float = 12.0
    context_window: float = 10.0
    boundary_audio_window: float = 2.0
    max_iterations: int = 2
    merge_gap: float = 2.0
    merge_similarity: float = 0.72


def _extract_frames(video_path: Path, silence: SilenceWindow, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    midpoint = (silence.start + silence.end) / 2
    frame_path = output_dir / f"frame_{midpoint:.2f}.jpg"
    _extract_frame_at(video_path, midpoint, frame_path)
    return frame_path


def _extract_frame_at(video_path: Path, timestamp: float, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def enrich_with_visual_context(
    video_path: Path,
    segments: Sequence[TranscriptSegment],
    clips: Sequence[ClipCandidate],
    *,
    min_gap: float = 1.5,
) -> None:
    silences = extract_silences(segments, min_gap=min_gap)
    if not silences:
        return

    for silence in silences:
        frame = _extract_frames(video_path, silence, paths.frames_dir)
        context = _context_for_silence(segments, silence)
        decision = analyze_visual_relevance(frame, context)

        silence.relevant = bool(decision.get("relevante"))
        silence.reason = decision.get("motivo")
        silence.layout = decision.get("layout_suggestion")
        silence.description = decision.get("description")
        silence.category = decision.get("category")

        logger.info(
            "[analysis] silencio evaluado",
            extra={
                "start": silence.start,
                "end": silence.end,
                "relevante": silence.relevant,
                "layout": silence.layout,
                "category": silence.category,
            },
        )

    _adjust_clips_for_silence(clips, silences)


def refine_clips(
    video_path: Path,
    audio_path: Path,
    segments: Sequence[TranscriptSegment],
    clips: Sequence[ClipCandidate],
    *,
    video_duration: float | None = None,
    config: RefineConfig | None = None,
) -> list[ClipCandidate]:
    cfg = config or RefineConfig()
    if not clips:
        return []

    extractor = AudioFeatureExtractor(audio_path)
    refined: list[ClipCandidate] = []

    # Normalize initial ranges
    for clip in clips:
        clip.time_ranges = _normalize_ranges(clip.time_ranges or [(clip.start, clip.end)])
        _update_clip_bounds(clip)

    # Gap audit (within clips)
    for idx, clip in enumerate(clips):
        refined.extend(_audit_clip_gaps(video_path, segments, clip, extractor, cfg, idx))

    # Boundary audit (iterative)
    refined = _audit_boundaries(video_path, segments, refined, extractor, cfg, video_duration)

    # Merge related clips (same topic, adjacent/overlapping)
    refined = _merge_related_clips(refined, segments, cfg)

    # Clamp durations and drop too short
    final: list[ClipCandidate] = []
    for clip in refined:
        _update_clip_bounds(clip)
        if clip.duration < cfg.min_clip_duration:
            continue
        if clip.duration > cfg.max_duration:
            _trim_to_max_duration(clip, cfg.max_duration, segments)
        final.append(clip)

    _assign_segment_ranges(final, segments)
    return final


def _audit_clip_gaps(
    video_path: Path,
    segments: Sequence[TranscriptSegment],
    clip: ClipCandidate,
    extractor: AudioFeatureExtractor,
    cfg: RefineConfig,
    clip_idx: int,
) -> list[ClipCandidate]:
    gaps = _find_internal_gaps(segments, clip, cfg.min_gap)
    if not gaps:
        return [clip]

    cuts: list[tuple[float, float]] = []
    split_points: list[float] = []

    gap_clip_headline = clip.headline or (clip.text[:80] if clip.text else "") or "sin título"

    for gap_idx, gap in enumerate(gaps):
        gap_duration = gap.end - gap.start
        t20 = gap.start + gap_duration * 0.2
        t50 = gap.start + gap_duration * 0.5
        t80 = gap.start + gap_duration * 0.8
        frames = [
            _extract_frame_at(video_path, t20, _frame_path(f"gap_{clip_idx}_{gap_idx}_20", t20)),
            _extract_frame_at(video_path, t50, _frame_path(f"gap_{clip_idx}_{gap_idx}_50", t50)),
            _extract_frame_at(video_path, t80, _frame_path(f"gap_{clip_idx}_{gap_idx}_80", t80)),
        ]
        before_text = _text_window(segments, gap.start - cfg.context_window, gap.start)
        after_text = _text_window(segments, gap.end, gap.end + cfg.context_window)
        features = extractor.window(gap.start, gap.end).to_dict()
        decision = analyze_gap_decision(
            frames,
            before_text=before_text,
            after_text=after_text,
            audio_features=features,
            gap_start=gap.start,
            gap_end=gap.end,
            clip_headline=gap_clip_headline,
        )
        action = str(decision.get("action", "keep")).lower()
        if action == "split":
            split_at = _safe_float(decision.get("split_at"), (gap.start + gap.end) / 2)
            split_points.append(split_at)
        elif action == "cut":
            cut_start = _safe_float(decision.get("cut_start"), gap.start)
            cut_end = _safe_float(decision.get("cut_end"), gap.end)
            cuts.append((cut_start, cut_end))

    if split_points:
        return _split_clip_at_points(clip, split_points, cfg)

    if cuts:
        ranges = clip.time_ranges or [(clip.start, clip.end)]
        for cut_start, cut_end in cuts:
            ranges = _subtract_range(ranges, cut_start, cut_end)
        clip.time_ranges = _normalize_ranges(ranges)
        _update_clip_bounds(clip)
    return [clip] if clip.duration >= cfg.min_clip_duration else []


def _audit_boundaries(
    video_path: Path,
    segments: Sequence[TranscriptSegment],
    clips: Sequence[ClipCandidate],
    extractor: AudioFeatureExtractor,
    cfg: RefineConfig,
    video_duration: float | None,
) -> list[ClipCandidate]:
    if not clips:
        return []

    refined = list(clips)
    for _ in range(cfg.max_iterations):
        changed = False
        for idx, clip in enumerate(refined):
            ranges = clip.time_ranges or [(clip.start, clip.end)]
            start = ranges[0][0]
            end = ranges[-1][1]

            # Start boundary
            clip_duration = end - start
            clip_headline = clip.headline or (clip.text[:80] if clip.text else "") or "sin título"
            before_text = _text_window(segments, start - cfg.context_window, start)
            after_text = _text_window(segments, start, start + cfg.context_window)
            features = extractor.window(max(0.0, start - cfg.boundary_audio_window), start + cfg.boundary_audio_window).to_dict()
            frame = _extract_frame_at(video_path, start, _frame_path(f"bound_start_{idx}", start))
            decision = analyze_boundary_decision(
                [frame],
                before_text=before_text,
                after_text=after_text,
                audio_features=features,
                boundary_side="inicio",
                clip_duration=clip_duration,
                clip_headline=clip_headline,
            )
            ranges, updated = _apply_boundary_decision(
                ranges,
                decision,
                is_start=True,
                cfg=cfg,
                video_duration=video_duration,
            )
            changed = changed or updated

            # End boundary
            start = ranges[0][0]
            end = ranges[-1][1]
            clip_duration = end - start
            before_text = _text_window(segments, end - cfg.context_window, end)
            after_text = _text_window(segments, end, end + cfg.context_window)
            features = extractor.window(max(0.0, end - cfg.boundary_audio_window), end + cfg.boundary_audio_window).to_dict()
            frame = _extract_frame_at(video_path, end, _frame_path(f"bound_end_{idx}", end))
            decision = analyze_boundary_decision(
                [frame],
                before_text=before_text,
                after_text=after_text,
                audio_features=features,
                boundary_side="fin",
                clip_duration=clip_duration,
                clip_headline=clip_headline,
            )
            ranges, updated_end = _apply_boundary_decision(
                ranges,
                decision,
                is_start=False,
                cfg=cfg,
                video_duration=video_duration,
            )
            changed = changed or updated_end

            clip.time_ranges = _normalize_ranges(ranges)
            _update_clip_bounds(clip)

        if not changed:
            break

    return refined


def _apply_boundary_decision(
    ranges: list[tuple[float, float]],
    decision: dict[str, object],
    *,
    is_start: bool,
    cfg: RefineConfig,
    video_duration: float | None,
) -> tuple[list[tuple[float, float]], bool]:
    extend_before = _clamp(_safe_float(decision.get("extend_before_sec"), 0.0), 0.0, cfg.max_extend)
    extend_after = _clamp(_safe_float(decision.get("extend_after_sec"), 0.0), 0.0, cfg.max_extend)
    trim_before = _clamp(_safe_float(decision.get("trim_before_sec"), 0.0), 0.0, cfg.max_extend)
    trim_after = _clamp(_safe_float(decision.get("trim_after_sec"), 0.0), 0.0, cfg.max_extend)

    updated = False
    if is_start:
        start, end = ranges[0]
        new_start = max(0.0, start - extend_before)
        if trim_before:
            new_start = min(new_start + trim_before, end)
        if new_start != start:
            ranges[0] = (new_start, end)
            updated = True
    else:
        start, end = ranges[-1]
        new_end = end + extend_after
        if video_duration is not None:
            new_end = min(new_end, video_duration)
        if trim_after:
            new_end = max(new_end - trim_after, start)
        if new_end != end:
            ranges[-1] = (start, new_end)
            updated = True

    return ranges, updated


def _merge_related_clips(
    clips: list[ClipCandidate],
    segments: Sequence[TranscriptSegment],
    cfg: RefineConfig,
) -> list[ClipCandidate]:
    if not clips:
        return []

    sorted_clips = sorted(clips, key=lambda c: c.start)
    texts = [_clip_text(clip, segments) for clip in sorted_clips]
    embedder = load_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    merged: list[ClipCandidate] = [sorted_clips[0]]
    for idx in range(1, len(sorted_clips)):
        current = sorted_clips[idx]
        last = merged[-1]
        gap = current.start - last.end
        similarity = float(np.dot(embeddings[idx], embeddings[idx - 1]))
        if gap <= cfg.merge_gap and similarity >= cfg.merge_similarity:
            combined_ranges = (last.time_ranges or [(last.start, last.end)]) + (current.time_ranges or [(current.start, current.end)])
            combined_ranges = _normalize_ranges(combined_ranges)
            combined = _clone_clip_base(last)
            combined.time_ranges = combined_ranges
            combined.headline = ""
            combined.keywords = []
            _update_clip_bounds(combined)
            if combined.duration <= cfg.max_duration:
                merged[-1] = combined
                continue
        merged.append(current)

    return merged


def _find_internal_gaps(
    segments: Sequence[TranscriptSegment],
    clip: ClipCandidate,
    min_gap: float,
) -> list[SilenceWindow]:
    ranges = clip.time_ranges or [(clip.start, clip.end)]
    gaps: list[SilenceWindow] = []

    for range_start, range_end in ranges:
        words = _collect_words(segments, range_start, range_end)
        if len(words) >= 2:
            prev_end = words[0][1]
            for start, end in words[1:]:
                gap = start - prev_end
                if gap >= min_gap:
                    gaps.append(SilenceWindow(start=prev_end, end=start))
                prev_end = end
        else:
            prev_end = range_start
            for seg in segments:
                if seg.start >= range_end or seg.end <= range_start:
                    continue
                if seg.start - prev_end >= min_gap:
                    gaps.append(SilenceWindow(start=prev_end, end=seg.start))
                prev_end = seg.end

    return gaps


def _collect_words(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    words: list[tuple[float, float]] = []
    for seg in segments:
        if seg.start >= end or seg.end <= start:
            continue
        for word in seg.words:
            if word.start is None or word.end is None:
                continue
            if word.end <= start or word.start >= end:
                continue
            words.append((float(word.start), float(word.end)))
    words.sort(key=lambda w: w[0])
    return words


_clip_text = clip_text
_segment_overlaps_ranges = segment_overlaps_ranges


def _split_clip_at_points(
    clip: ClipCandidate,
    split_points: list[float],
    cfg: RefineConfig,
) -> list[ClipCandidate]:
    split_points = sorted(p for p in split_points if clip.start < p < clip.end)
    if not split_points:
        return [clip]

    segments: list[tuple[float, float]] = []
    current_start = clip.start
    for point in split_points:
        segments.append((current_start, point))
        current_start = point
    segments.append((current_start, clip.end))

    clips: list[ClipCandidate] = []
    for start, end in segments:
        if end - start < cfg.min_clip_duration:
            continue
        new_clip = _clone_clip_base(clip)
        new_clip.time_ranges = _normalize_ranges([(start, end)])
        new_clip.headline = ""
        new_clip.keywords = []
        _update_clip_bounds(new_clip)
        clips.append(new_clip)
    return clips or [clip]


def _clone_clip_base(clip: ClipCandidate) -> ClipCandidate:
    return ClipCandidate(
        start=clip.start,
        end=clip.end,
        text=clip.text,
        coherence=clip.coherence,
        headline=clip.headline or "",
        keywords=list(clip.keywords),
        segment_range=None,
        layout=clip.layout,
        layout_segments=None,
        time_ranges=clip.time_ranges,
        face_crop=clip.face_crop,
    )


def _assign_segment_ranges(clips: Sequence[ClipCandidate], segments: Sequence[TranscriptSegment]) -> None:
    for clip in clips:
        ranges = clip.time_ranges or [(clip.start, clip.end)]
        indices = [idx for idx, seg in enumerate(segments) if _segment_overlaps_ranges(seg, ranges)]
        if indices:
            clip.segment_range = (min(indices), max(indices) + 1)
        else:
            clip.segment_range = None


def _normalize_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned = [(float(start), float(end)) for start, end in ranges if end > start]
    cleaned.sort(key=lambda r: r[0])
    merged: list[tuple[float, float]] = []
    for start, end in cleaned:
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _subtract_range(
    ranges: list[tuple[float, float]],
    cut_start: float,
    cut_end: float,
) -> list[tuple[float, float]]:
    updated: list[tuple[float, float]] = []
    for start, end in ranges:
        if cut_end <= start or cut_start >= end:
            updated.append((start, end))
            continue
        if cut_start > start:
            updated.append((start, cut_start))
        if cut_end < end:
            updated.append((cut_end, end))
    return updated


def _trim_to_max_duration(
    clip: ClipCandidate,
    max_duration: float,
    segments: Sequence[TranscriptSegment] = (),
) -> None:
    ranges = clip.time_ranges or [(clip.start, clip.end)]
    total = sum(end - start for start, end in ranges)
    if total <= max_duration:
        return

    # Compute the hard cut point first
    remaining = max_duration
    hard_cut_time = ranges[0][0]
    trimmed_ranges_count = 0
    for start, end in ranges:
        duration = end - start
        if duration <= remaining:
            remaining -= duration
            trimmed_ranges_count += 1
        else:
            hard_cut_time = start + remaining
            break

    # Search for a natural cut point near the hard cut (within a 10s window before it)
    search_window = 10.0
    search_start = max(ranges[0][0], hard_cut_time - search_window)
    best_cut = hard_cut_time

    # Prefer segment boundaries (end of a transcript segment = end of a sentence/phrase)
    best_distance = search_window + 1.0
    for seg in segments:
        if seg.end < search_start or seg.end > hard_cut_time:
            continue
        distance = hard_cut_time - seg.end
        if distance < best_distance:
            best_distance = distance
            best_cut = seg.end

    # Also check word boundaries with sentence-ending punctuation
    if best_cut == hard_cut_time and segments:
        for seg in segments:
            if seg.start > hard_cut_time or seg.end < search_start:
                continue
            for word in seg.words:
                if word.end is None or word.start is None:
                    continue
                if word.end < search_start or word.end > hard_cut_time:
                    continue
                w = word.word.strip()
                if w and w[-1] in ".!?;":
                    distance = hard_cut_time - word.end
                    if distance < best_distance:
                        best_distance = distance
                        best_cut = word.end

    # Rebuild trimmed ranges using the best cut point
    remaining = best_cut - ranges[0][0]
    trimmed: list[tuple[float, float]] = []
    for start, end in ranges:
        duration = end - start
        if duration <= remaining:
            trimmed.append((start, end))
            remaining -= duration
        else:
            if remaining > 0:
                trimmed.append((start, start + remaining))
            break

    clip.time_ranges = _normalize_ranges(trimmed)
    _update_clip_bounds(clip)


def _update_clip_bounds(clip: ClipCandidate) -> None:
    ranges = clip.time_ranges or [(clip.start, clip.end)]
    clip.start = ranges[0][0]
    clip.end = ranges[-1][1]


def _text_window(segments: Sequence[TranscriptSegment], start: float, end: float) -> str:
    if end <= start:
        return ""
    parts: list[str] = []
    for seg in segments:
        if seg.start >= end or seg.end <= start:
            continue
        if seg.text:
            parts.append(seg.text)
    return " ".join(parts).strip()


def _frame_path(prefix: str, timestamp: float) -> Path:
    return paths.frames_dir / f"{prefix}_{timestamp:.2f}.jpg"


def _safe_float(value: object, default: float) -> float:
    try:
        if value is None:
            return default
        # The conversion stays dynamic: unsupported values still take the fallback below.
        conversion_input = cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value)
        return float(conversion_input)
    except Exception:
        return default


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _context_for_silence(segments: Sequence[TranscriptSegment], silence: SilenceWindow) -> str:
    window_text = []
    for segment in segments:
        if segment.end < silence.start:
            window_text = [segment.text]
        elif segment.start > silence.end:
            window_text.append(segment.text)
            break
        elif silence.start <= segment.start <= silence.end:
            window_text.append(segment.text)
    return " ".join(window_text).strip()


def _adjust_clips_for_silence(clips: Sequence[ClipCandidate], silences: Iterable[SilenceWindow]) -> None:
    # Filter only relevant windows that serve as context
    relevant_windows = [silence for silence in silences if silence.relevant]
    if not relevant_windows:
        return

    for clip in clips:
        for silence in relevant_windows:
            # If the silence (visual check) overlaps with or is adjacent to the clip
            if clip.start <= silence.start <= clip.end or silence.start <= clip.start <= silence.end:
                clip.end = max(clip.end, silence.end)
                clip.start = min(clip.start, silence.start)

                if silence.layout and not clip.layout:
                    clip.layout = silence.layout


def analyze_clip_layout(video_path: Path, clips: Sequence[ClipCandidate]) -> None:
    """Determine visual layout for each clip using multi-frame sampling and recursive transition search."""
    if not clips:
        return

    logger.info("[analysis] iniciando análisis de layout dinámico", extra={"clips": len(clips)})

    for clip in clips:
        ranges = clip.time_ranges or [(clip.start, clip.end)]
        layout_segments: list[tuple[float, float, str]] = []
        for range_start, range_end in ranges:
            layout_segments.extend(_layout_segments_for_range(video_path, range_start, range_end))
        clip.layout_segments = layout_segments

        final_summary = [f"{seg[2]}({seg[1]-seg[0]:.1f}s)" for seg in clip.layout_segments] if clip.layout_segments else clip.layout
        logger.info(
            "[analysis] layout decidido",
            extra={
                "clip_start": clip.start,
                "segments": final_summary,
            },
        )


def _layout_segments_for_range(video_path: Path, start: float, end: float) -> list[tuple[float, float, str]]:
    if end <= start:
        return []

    start_layout = _get_layout_at(video_path, start)
    end_layout = _get_layout_at(video_path, end)

    if start_layout == end_layout:
        mid_ts = (start + end) / 2
        mid_layout = _get_layout_at(video_path, mid_ts)
        if mid_layout != start_layout:
            pivot1 = _find_transition_point(video_path, start, mid_ts, start_layout, mid_layout)
            pivot2 = _find_transition_point(video_path, mid_ts, end, mid_layout, end_layout)
            return [
                (start, pivot1, start_layout),
                (pivot1, pivot2, mid_layout),
                (pivot2, end, end_layout),
            ]
        return [(start, end, start_layout)]

    pivot = _find_transition_point(video_path, start, end, start_layout, end_layout)
    return [(start, pivot, start_layout), (pivot, end, end_layout)]


def _get_layout_at(video_path: Path, timestamp: float) -> str:
    """Helper to get layout at a specific timestamp."""
    frame_path = paths.frames_dir / f"check_{timestamp:.2f}.jpg"
    _extract_frame_at(video_path, timestamp, frame_path)
    if not frame_path.exists():
        return "SPLIT"
    decision = analyze_visual_relevance(frame_path, context="Analisis de layout.")
    raw_layout = (decision.get("layout_suggestion") or "SPLIT").upper()

    if "TALKING" in raw_layout or "FULL_CAM" in raw_layout:
        return "FULL_CAM"
    if "CONTENT" in raw_layout:
        return "FULL_CONTENT"
    return "SPLIT"


def _find_transition_point(video_path: Path, start: float, end: float, layout_start: str, layout_end: str, depth: int = 0) -> float:
    """Recursive binary search to find transition timestamp."""
    if (end - start) < 1.5 or depth > 4:
        return (start + end) / 2

    mid = (start + end) / 2
    layout_mid = _get_layout_at(video_path, mid)

    if layout_mid == layout_start:
        return _find_transition_point(video_path, mid, end, layout_mid, layout_end, depth + 1)
    return _find_transition_point(video_path, start, mid, layout_start, layout_mid, depth + 1)
