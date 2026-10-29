"""Sparse-to-dense, provider-free visual indexing for source streams."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
import json
import io
import math
from pathlib import Path
from typing import Any, Callable, Protocol, cast
from uuid import uuid4
import wave

import cv2
import numpy as np

from .audio import ensure_audio_checkpoint, extract_audio_window
from .audio_features import AudioFeatureExtractor, AudioWindowFeatures
from .face_identity import (
    FaceIdentityBackend,
    FaceObservation,
    FaceTrack,
    LandmarkedFace,
    OpenCVFaceIdentityBackend,
    StreamerIdentityResult,
    associate_face_tracks,
    resolve_streamer_identity,
)
from .jobs import JobStore
from .local_paths import (
    atomic_read_text_batch,
    atomic_write_text,
    global_streamer_identity_path,
    paths,
    source_fingerprint,
    source_job_root,
)
from .media import probe_media
from .model_assets import (
    FACE_MODELS,
    MANIFEST_FILENAME,
    ensure_face_models,
    load_face_manifest,
    sha256_file,
    validate_face_manifest,
    validate_model_file,
    ModelAssetError,
)
from .render_clock import SourceDecodeReceipt, SourceTimeOrigin, canonical, digest, exact, rational
from .render_runtime import RenderRuntime, qualify_render_runtime, resolve_render_runtime
from .source_decode import (
    AudioDecodeLimits, DecodeLimits, Window, decode_requirements, observe_audio_window,
    observe_windows, probe_source,
)
from .visual_provenance import (
    AudioUnavailableError, MAX_SCHEDULE_POINTS, MAX_CATALOG_ARTIFACTS, MAX_ARTIFACT_BYTES,
    MAX_GENERATION_BYTES, OBSERVED_ANALYZER_VERSION,
    file_sha256, generation_catalog, generation_file, load_visual_provenance,
    map_source_audio_window, read_generation_json, source_time, validate_observed_generation,
    visual_timeline_fingerprint, supported_unavailable_intervals, emitted_text_size,
)
from .planner_contracts import TimeRange
from .visual_contracts import (
    IdentityStatus,
    NormalizedRect,
    SourceState,
    SourceVisualSample,
    SourceVisualSpan,
    SourceVisualTimeline,
    WebcamTrack,
)
from .webcam_tracking import (
    WebcamRectangleTrack,
    WebcamRegionObservation,
    associate_webcam_tracks,
)


_CAMERA_MOVE_THRESHOLD = 0.08
_TEXT_BROWSER_THRESHOLD = 0.22
_VISUAL_CHANGE_THRESHOLD = 0.18
_TEXT_DENSITY_CHANGE_THRESHOLD = 0.12
_DENSE_RADIUS_SECONDS = 3.0
_ANALYZER_VERSION = OBSERVED_ANALYZER_VERSION
_PROFILE_COSINE_THRESHOLD = 0.55
_PROFILE_TRACK_MERGE_COSINE_THRESHOLD = 0.60
_PROFILE_FULL_CAMERA_FACE_AREA_MIN = 0.004
_PROFILE_FULL_CAMERA_FACE_AREA_MAX = 0.12
_FULL_CAMERA_BRIDGE_MAX_SECONDS = 5.0
_FULL_CAMERA_EDGE_HOLD_SECONDS = 1.0
_FULL_CAMERA_CONTINUITY_HISTOGRAM_MAX = 0.12
_FULL_CAMERA_CONTINUITY_STRUCTURAL_MAX = 0.15


@dataclass(frozen=True, slots=True)
class LocalFrameFeatures:
    """Provider-free evidence extracted from one decoded source frame."""

    histogram_change_score: float
    structural_change_score: float
    letterbox_bounds: NormalizedRect | None
    persistent_rectangle_candidates: tuple[NormalizedRect, ...]
    text_density: float
    text_regions: tuple[NormalizedRect, ...]
    brightness: float
    sharpness: float
    motion: float


class FrameDecoder(Protocol):
    """Seam for Task 4B decoders; implementations own media I/O."""

    def decode_frame(self, at: float) -> np.ndarray:
        """Decode a BGR frame at a source timestamp."""


class LocalVisualAnalyzer(Protocol):
    """Seam used by the future sparse-to-dense builder."""

    def analyze(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
    ) -> LocalFrameFeatures:
        """Extract deterministic local evidence without provider calls."""


def _validated_bgr(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame debe ser un ndarray no vacío")
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("frame debe ser gris, BGR o BGRA")
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_validated_bgr(frame), cv2.COLOR_BGR2GRAY)


def _matching_gray_frames(
    previous_frame: np.ndarray | None,
    current_frame: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    current = _gray(current_frame)
    if previous_frame is None:
        return None, current
    previous = _gray(previous_frame)
    if previous.shape != current.shape:
        raise ValueError("Los frames comparados deben tener las mismas dimensiones")
    return previous, current


def compute_histogram_change(
    previous_frame: np.ndarray | None,
    current_frame: np.ndarray,
) -> float:
    """Return normalized grayscale-distribution change in ``[0, 1]``."""

    previous, current = _matching_gray_frames(previous_frame, current_frame)
    if previous is None:
        return 0.0

    def normalized_histogram(gray: np.ndarray) -> np.ndarray:
        histogram = cv2.calcHist([gray], [0], None, [64], [0, 256])
        total = float(histogram.sum())
        return histogram / total if total else histogram

    distance = 0.5 * float(
        np.abs(normalized_histogram(previous) - normalized_histogram(current)).sum()
    )
    return float(np.clip(distance, 0.0, 1.0))


def compute_structural_change(
    previous_frame: np.ndarray | None,
    current_frame: np.ndarray,
) -> float:
    """Measure edge-map change independently from overall luminance."""

    previous, current = _matching_gray_frames(previous_frame, current_frame)
    if previous is None:
        return 0.0
    previous_edges = cv2.Canny(previous, 60, 160)
    current_edges = cv2.Canny(current, 60, 160)
    return float(np.mean(cv2.absdiff(previous_edges, current_edges)) / 255.0)


def compute_brightness(frame: np.ndarray) -> float:
    return float(np.mean(_gray(frame)) / 255.0)


def compute_sharpness(frame: np.ndarray) -> float:
    return float(cv2.Laplacian(_gray(frame), cv2.CV_64F).var())


def compute_motion(
    previous_frame: np.ndarray | None,
    current_frame: np.ndarray,
) -> float:
    previous, current = _matching_gray_frames(previous_frame, current_frame)
    if previous is None:
        return 0.0
    return float(np.mean(cv2.absdiff(previous, current)) / 255.0)


def _outer_run(mask: np.ndarray, *, reverse: bool = False) -> int:
    values = mask[::-1] if reverse else mask
    count = 0
    for value in values:
        if not bool(value):
            break
        count += 1
    return count


def detect_letterbox_bounds(frame: np.ndarray) -> NormalizedRect | None:
    """Detect uniform dark bars and return the remaining source content bounds."""

    gray = _gray(frame)
    height, width = gray.shape
    reference = float(np.percentile(gray, 75))
    if reference <= 1.0:
        return None
    dark_threshold = max(8.0, min(40.0, reference * 0.15))
    row_bars = (gray.mean(axis=1) <= dark_threshold) & (gray.std(axis=1) <= 8.0)
    column_bars = (gray.mean(axis=0) <= dark_threshold) & (gray.std(axis=0) <= 8.0)

    top = _outer_run(row_bars)
    bottom = _outer_run(row_bars, reverse=True)
    left = _outer_run(column_bars)
    right = _outer_run(column_bars, reverse=True)
    if max(top / height, bottom / height, left / width, right / width) < 0.015:
        return None
    content_width = width - left - right
    content_height = height - top - bottom
    if content_width <= 0 or content_height <= 0:
        return None
    return NormalizedRect(
        x=left / width,
        y=top / height,
        width=content_width / width,
        height=content_height / height,
    )


def _rect_iou(left: NormalizedRect, right: NormalizedRect) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _deduplicate_rectangles(
    rectangles: Sequence[NormalizedRect],
    *,
    iou_threshold: float = 0.80,
) -> list[NormalizedRect]:
    ordered = sorted(
        rectangles,
        key=lambda rect: rect.width * rect.height,
        reverse=True,
    )
    kept: list[NormalizedRect] = []
    for rectangle in ordered:
        if all(_rect_iou(rectangle, existing) < iou_threshold for existing in kept):
            kept.append(rectangle)
    return kept


def detect_straight_edge_rectangles(frame: np.ndarray) -> list[NormalizedRect]:
    """Find sizeable four-edge overlays without deciding that they are webcams."""

    gray = _gray(frame)
    height, width = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(width * height)
    candidates: list[NormalizedRect] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, box_width, box_height = cv2.boundingRect(polygon)
        area_ratio = box_width * box_height / frame_area
        if not 0.01 <= area_ratio <= 0.90 or box_width < 12 or box_height < 12:
            continue
        contour_area = abs(float(cv2.contourArea(polygon)))
        rectangularity = contour_area / max(1.0, box_width * box_height)
        if rectangularity < 0.75:
            continue
        candidates.append(
            NormalizedRect(
                x=x / width,
                y=y / height,
                width=min(box_width / width, 1.0 - x / width),
                height=min(box_height / height, 1.0 - y / height),
            )
        )
    return _deduplicate_rectangles(candidates)


def detect_text_regions(frame: np.ndarray) -> tuple[float, list[NormalizedRect]]:
    """Estimate text density and boxes with OpenCV MSER."""

    gray = _gray(frame)
    height, width = gray.shape
    frame_area = width * height
    mser_factory = cast(Callable[[], cv2.MSER], getattr(cv2, "MSER_create"))
    detector = mser_factory()
    detector.setMinArea(max(4, int(frame_area * 0.00002)))
    detector.setMaxArea(max(20, int(frame_area * 0.08)))
    _, boxes = detector.detectRegions(gray)
    mask = np.zeros((height, width), dtype=np.uint8)
    candidates: list[NormalizedRect] = []
    for x, y, box_width, box_height in boxes:
        if box_width < 2 or box_height < 3:
            continue
        aspect = box_width / box_height
        area_ratio = box_width * box_height / frame_area
        if not 0.08 <= aspect <= 15.0 or not 0.00002 <= area_ratio <= 0.08:
            continue
        x2 = min(width, x + box_width)
        y2 = min(height, y + box_height)
        mask[y:y2, x:x2] = 1
        candidates.append(
            NormalizedRect(
                x=x / width,
                y=y / height,
                width=(x2 - x) / width,
                height=(y2 - y) / height,
            )
        )
    density = float(np.count_nonzero(mask) / frame_area)
    return float(np.clip(density, 0.0, 1.0)), _deduplicate_rectangles(
        candidates,
        iou_threshold=0.70,
    )


class OpenCVLocalVisualAnalyzer:
    """Default pure local analyzer for generated or already-decoded BGR frames."""

    def analyze(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
    ) -> LocalFrameFeatures:
        normalized = _validated_bgr(frame)
        previous = _validated_bgr(previous_frame) if previous_frame is not None else None
        text_density, text_regions = detect_text_regions(normalized)
        return LocalFrameFeatures(
            histogram_change_score=compute_histogram_change(previous, normalized),
            structural_change_score=compute_structural_change(previous, normalized),
            letterbox_bounds=detect_letterbox_bounds(normalized),
            persistent_rectangle_candidates=tuple(
                detect_straight_edge_rectangles(normalized)[:32]
            ),
            text_density=text_density,
            text_regions=tuple(text_regions[:128]),
            brightness=compute_brightness(normalized),
            sharpness=compute_sharpness(normalized),
            motion=compute_motion(previous, normalized),
        )


def _dense_change_detected(
    previous: SourceVisualSample,
    current: SourceVisualSample,
) -> bool:
    if max(current.histogram_change_score, current.structural_change_score) >= (
        _VISUAL_CHANGE_THRESHOLD
    ):
        return True
    if (previous.webcam_rect is None) != (current.webcam_rect is None):
        return True
    if _webcam_moved(previous.webcam_rect, current.webcam_rect):
        return True
    if set(previous.face_track_ids) != set(current.face_track_ids):
        return True
    if _rect_changed(previous.letterbox_bounds, current.letterbox_bounds):
        return True
    return (
        abs(current.text_density - previous.text_density) + 1e-12
        >= _TEXT_DENSITY_CHANGE_THRESHOLD
    )


def _dense_trigger_times(
    samples: Sequence[SourceVisualSample],
    editorial_boundaries: Sequence[float],
) -> list[float]:
    triggers = {float(value) for value in editorial_boundaries if float(value) >= 0.0}
    ordered = sorted(samples, key=lambda item: item.at)
    for previous, current in zip(ordered, ordered[1:]):
        if _dense_change_detected(previous, current):
            triggers.add(current.at)
    return sorted(triggers)


def _dense_change_brackets(
    samples: Sequence[SourceVisualSample],
) -> list[tuple[float, float]]:
    """Return every sparse interval whose endpoints show a material change.

    The event can occur anywhere between the endpoints, so dense inspection must
    cover the whole bracket rather than only a radius around the later sample.
    """

    ordered = sorted(samples, key=lambda item: item.at)
    return [
        (previous.at, current.at)
        for previous, current in zip(ordered, ordered[1:])
        if current.at > previous.at and _dense_change_detected(previous, current)
    ]


def _dense_windows(
    trigger_times: Sequence[float],
    *,
    duration_seconds: float,
    change_brackets: Sequence[tuple[float, float]] = (),
) -> list[tuple[float, float]]:
    if duration_seconds <= 0:
        return []
    expanded = [
        (
            max(0.0, float(timestamp) - _DENSE_RADIUS_SECONDS),
            min(duration_seconds, float(timestamp) + _DENSE_RADIUS_SECONDS),
        )
        for timestamp in trigger_times
        if 0.0 <= float(timestamp) <= duration_seconds
    ]
    for raw_start, raw_end in change_brackets:
        start = max(0.0, min(duration_seconds, float(raw_start)))
        end = max(0.0, min(duration_seconds, float(raw_end)))
        if end > start:
            expanded.append((start, end))
    expanded.sort()
    merged: list[tuple[float, float]] = []
    for start, end in expanded:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged

def _timestamps_for_windows(
    windows: Sequence[tuple[float, float]],
    *,
    interval: float,
    existing_timestamps: set[float],
) -> list[float]:
    if interval <= 0:
        raise ValueError("dense_interval debe ser positivo")
    existing = {round(float(value), 6) for value in existing_timestamps}
    generated: set[float] = set()
    for start, end in windows:
        if (not all(math.isfinite(v) for v in (start, end, interval))
                or end < start or math.floor((end - start) / interval) + 1 > MAX_SCHEDULE_POINTS):
            raise ValueError("Dense scheduled sample point budget exceeded")
        timestamp = float(start)
        while timestamp <= end + 1e-9:
            rounded = round(timestamp, 6)
            if rounded not in existing:
                generated.add(rounded)
                if len(generated) + len(existing) > MAX_SCHEDULE_POINTS:
                    raise ValueError("Cumulative dense sample point budget exceeded")
            timestamp += interval
        rounded_end = round(float(end), 6)
        if rounded_end not in existing:
            generated.add(rounded_end)
            if len(generated) + len(existing) > MAX_SCHEDULE_POINTS:
                raise ValueError("Cumulative dense sample point budget exceeded")
    return sorted(generated)


def classify_source_state(
    sample: SourceVisualSample,
) -> tuple[SourceState, float, list[str]]:
    """Classify only evidence visible in one already-associated sample."""

    reasons: list[str] = []
    tracked_webcam = (
        sample.webcam_track_id is not None and sample.webcam_rect is not None
    )
    persistent_webcam = sample.webcam_rect is not None and any(
        _rect_iou(sample.webcam_rect, rectangle) >= 0.75
        for rectangle in sample.persistent_rectangles
    )
    has_webcam = tracked_webcam or persistent_webcam
    face_count = len(set(sample.face_track_ids))

    inferred_full_camera = "inferred_full_camera" in sample.evidence_ids
    if inferred_full_camera and not has_webcam and face_count <= 1:
        reasons.append("profile_or_transition_full_camera")
        return SourceState.FULL_CAMERA, 0.90, reasons

    if has_webcam:
        if tracked_webcam:
            reasons.append("tracked_webcam")
        if persistent_webcam:
            reasons.append("persistent_webcam_rectangle")
        assert sample.webcam_rect is not None
        if face_count == 0:
            reasons.append("webcam_without_face")
        elif face_count == 1:
            reasons.append("single_face_in_webcam")
        else:
            reasons.append("multiple_faces_in_webcam")
        if sample.webcam_rect.width * sample.webcam_rect.height >= 0.65:
            return SourceState.FULL_CAMERA, 0.86 if face_count == 1 else 0.68, reasons
        return SourceState.CONTENT_WITH_CAMERA, 0.88 if face_count == 1 else 0.66, reasons

    if face_count:
        reasons.append("face_without_webcam_evidence")
    if sample.letterbox_bounds is not None and face_count:
        reasons.append("letterboxed_face_content")
        return SourceState.REACTED_VIDEO, 0.72, reasons
    if sample.text_density >= _TEXT_BROWSER_THRESHOLD:
        reasons.append("high_text_density")
        return SourceState.BROWSER_TEXT, 0.70, reasons
    if face_count > 1:
        reasons.append("multiple_unresolved_faces")
        return SourceState.MULTI_PERSON, 0.58, reasons
    if not sample.evidence_ids:
        return SourceState.UNKNOWN, 0.0, ["missing_evidence"]
    reasons.append("content_without_camera")
    return SourceState.CONTENT_ONLY, 0.62 if face_count == 0 else 0.54, reasons


def segment_visual_states(
    samples: Sequence[SourceVisualSample],
    *,
    minimum_span_seconds: float = 1.0,
) -> list[SourceVisualSpan]:
    """Segment ordered samples, placing visual changes between observations."""

    if minimum_span_seconds <= 0:
        raise ValueError("minimum_span_seconds debe ser positivo")
    ordered = sorted(samples, key=lambda item: item.at)
    if not ordered:
        return []
    if any(right.at <= left.at for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Los timestamps visuales deben ser únicos y crecientes")

    classifications = [classify_source_state(sample) for sample in ordered]
    groups: list[tuple[int, int, float, list[str]]] = []
    group_start = 0
    group_source_start = ordered[0].at
    group_reasons = ["initial_state"]
    for index in range(1, len(ordered)):
        reasons = _sample_change_reasons(
            ordered[index - 1],
            ordered[index],
            classifications[index - 1][0],
            classifications[index][0],
        )
        if reasons:
            midpoint = (ordered[index - 1].at + ordered[index].at) / 2.0
            if midpoint - group_source_start < minimum_span_seconds:
                group_reasons = list(
                    dict.fromkeys(
                        [*group_reasons, *reasons, "short_span_merged"]
                    )
                )
                continue
            groups.append((group_start, index, group_source_start, group_reasons))
            group_start = index
            group_source_start = midpoint
            group_reasons = reasons
    groups.append((group_start, len(ordered), group_source_start, group_reasons))

    final_end = max(ordered[-1].at, groups[-1][2] + minimum_span_seconds)
    spans: list[SourceVisualSpan] = []
    for group_index, (start_index, stop_index, source_start, change_reasons) in enumerate(groups):
        source_end = groups[group_index + 1][2] if group_index + 1 < len(groups) else final_end
        if source_end - source_start < minimum_span_seconds and spans:
            previous = spans[-1]
            spans[-1] = previous.model_copy(
                update={
                    "source": TimeRange(start=previous.source.start, end=source_end),
                    "change_reasons": list(
                        dict.fromkeys([*previous.change_reasons, *change_reasons, "short_span_merged"])
                    ),
                }
            )
            continue
        group_samples = ordered[start_index:stop_index]
        group_classifications = classifications[start_index:stop_index]
        state, confidence = _representative_state(
            group_classifications,
            sample_times=[item.at for item in group_samples],
            source_start=source_start,
            source_end=source_end,
        )
        webcam_ids = {
            item.webcam_track_id for item in group_samples if item.webcam_track_id is not None
        }
        face_ids = {track_id for item in group_samples for track_id in item.face_track_ids}
        streamer_track_id, identity_status = _local_identity(
            state=state,
            webcam_ids=webcam_ids,
            face_ids=face_ids,
        )
        important_regions = _unique_rectangles(
            [
                rect
                for item in group_samples
                for rect in ([item.webcam_rect] if item.webcam_rect is not None else [])
            ]
        )
        text_regions = _unique_rectangles(
            [rect for item in group_samples for rect in item.text_regions]
        )
        evidence_ids = list(
            dict.fromkeys(evidence for item in group_samples for evidence in item.evidence_ids)
        )
        spans.append(
            SourceVisualSpan(
                span_id=f"visual_span_{len(spans) + 1:04d}",
                source=TimeRange(start=source_start, end=source_end),
                state=state,
                confidence=confidence,
                streamer_track_id=streamer_track_id,
                identity_status=identity_status,
                webcam_track_id=next(iter(webcam_ids)) if len(webcam_ids) == 1 else None,
                important_regions=important_regions,
                text_regions=text_regions,
                evidence_ids=evidence_ids,
                change_reasons=change_reasons,
            )
        )
    return spans


def _sample_change_reasons(
    previous: SourceVisualSample,
    current: SourceVisualSample,
    previous_state: SourceState,
    current_state: SourceState,
) -> list[str]:
    reasons: list[str] = []
    if previous_state is not current_state:
        reasons.append("state_changed")
    if _webcam_moved(previous.webcam_rect, current.webcam_rect):
        reasons.append("webcam_moved")
    if previous.webcam_track_id != current.webcam_track_id:
        reasons.append("webcam_track_changed")
    if set(previous.face_track_ids) != set(current.face_track_ids):
        reasons.append("face_tracks_changed")
    if _rect_changed(previous.letterbox_bounds, current.letterbox_bounds):
        reasons.append("letterbox_changed")
    return reasons


def _rect_changed(
    previous: NormalizedRect | None,
    current: NormalizedRect | None,
    *,
    tolerance: float = 0.005,
) -> bool:
    if previous is None or current is None:
        return previous is not current
    return max(
        abs(previous.x - current.x),
        abs(previous.y - current.y),
        abs(previous.width - current.width),
        abs(previous.height - current.height),
    ) > tolerance


def _webcam_moved(
    previous: NormalizedRect | None,
    current: NormalizedRect | None,
) -> bool:
    if previous is None or current is None:
        return False
    previous_center = (
        previous.x + previous.width / 2.0,
        previous.y + previous.height / 2.0,
    )
    current_center = (
        current.x + current.width / 2.0,
        current.y + current.height / 2.0,
    )
    return (
        abs(previous_center[0] - current_center[0]) > _CAMERA_MOVE_THRESHOLD + 1e-9
        or abs(previous_center[1] - current_center[1]) > _CAMERA_MOVE_THRESHOLD + 1e-9
    )


def _representative_state(
    classifications: Sequence[tuple[SourceState, float, list[str]]],
    *,
    sample_times: Sequence[float],
    source_start: float,
    source_end: float,
) -> tuple[SourceState, float]:
    if len(classifications) != len(sample_times) or not classifications:
        raise ValueError("classifications y sample_times deben tener la misma longitud")
    scores: dict[SourceState, tuple[float, float]] = {}
    for index, (state, confidence, _) in enumerate(classifications):
        left = (
            source_start
            if index == 0
            else (sample_times[index - 1] + sample_times[index]) / 2.0
        )
        right = (
            source_end
            if index == len(sample_times) - 1
            else (sample_times[index] + sample_times[index + 1]) / 2.0
        )
        weight = max(1e-9, right - left)
        total_weight, weighted_confidence = scores.get(state, (0.0, 0.0))
        scores[state] = (
            total_weight + weight,
            weighted_confidence + confidence * weight,
        )
    state = max(
        scores,
        key=lambda item: (scores[item][0], scores[item][1], item.value),
    )
    total_weight, weighted_confidence = scores[state]
    return state, weighted_confidence / total_weight


def _local_identity(
    *,
    state: SourceState,
    webcam_ids: set[str],
    face_ids: set[str],
) -> tuple[str | None, IdentityStatus]:
    has_camera_evidence = state in {SourceState.FULL_CAMERA, SourceState.CONTENT_WITH_CAMERA}
    if has_camera_evidence and len(webcam_ids) == 1 and len(face_ids) == 1:
        return next(iter(face_ids)), IdentityStatus.CONFIDENT
    if has_camera_evidence and len(face_ids) > 1:
        return None, IdentityStatus.AMBIGUOUS
    return None, IdentityStatus.UNAVAILABLE


def _unique_rectangles(rectangles: Sequence[NormalizedRect]) -> list[NormalizedRect]:
    unique: dict[tuple[float, float, float, float], NormalizedRect] = {}
    for rect in rectangles:
        key = (rect.x, rect.y, rect.width, rect.height)
        unique.setdefault(key, rect)
    return list(unique.values())


@dataclass(slots=True)
class _DecodedFrame:
    at: float
    independent_features: LocalFrameFeatures
    gray_thumbnail: np.ndarray
    histogram: np.ndarray
    faces: list[LandmarkedFace]


class _AudioFeatureWindow(Protocol):
    def window(self, start: float, end: float) -> AudioWindowFeatures: ...


class _VisualArtifactContext(Protocol):
    @property
    def fingerprint(self) -> str: ...
    @property
    def transcript_path(self) -> Path: ...
    @property
    def diarization_path(self) -> Path: ...
    @property
    def streamer_identity_path(self) -> Path: ...


@dataclass(frozen=True, slots=True)
class _ReadOnlyVisualContext:
    fingerprint: str
    transcript_path: Path
    diarization_path: Path
    streamer_identity_path: Path


@dataclass(slots=True)
class _BuilderRuntime:
    duration_seconds: float
    decoder: FrameDecoder
    local_analyzer: LocalVisualAnalyzer
    face_backend: FaceIdentityBackend
    model_hashes: dict[str, str]
    audio_extractor: _AudioFeatureWindow | None
    window_start: float = 0.0
    window_end: float | None = None


@dataclass(slots=True)
class _SourceTimeAudioExtractor:
    delegate: AudioFeatureExtractor
    source_offset: float
    local_duration: float

    def window(self, start: float, end: float) -> AudioWindowFeatures:
        local_start = max(0.0, float(start) - self.source_offset)
        local_end = min(self.local_duration, float(end) - self.source_offset)
        if local_end <= local_start:
            local_end = min(self.local_duration, local_start + 0.001)
        return self.delegate.window(local_start, local_end)


class _OpenCVFrameDecoder:
    def __init__(self, video_path: Path) -> None:
        self._capture = cv2.VideoCapture(str(video_path))
        if not self._capture.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def decode_frame(self, at: float) -> np.ndarray:
        requested = float(at)
        if not math.isfinite(requested) or requested < 0:
            raise ValueError("frame timestamp must be finite and non-negative")
        if math.isfinite(self._fps) and self._fps > 0 and self._frame_count > 0:
            target_frame = min(
                self._frame_count - 1,
                max(0, int(math.floor(requested * self._fps + 1e-9))),
            )
            positioned = self._capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                float(target_frame),
            )
        else:
            positioned = self._capture.set(
                cv2.CAP_PROP_POS_MSEC,
                requested * 1000.0,
            )
        if not positioned:
            raise RuntimeError(f"No se pudo posicionar el frame en {requested:.3f}s")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"No se pudo decodificar el frame en {requested:.3f}s")
        return _validated_bgr(frame)

    def close(self) -> None:
        self._capture.release()

class _SequentialOpenCVFrameDecoder:
    """Seek once, then decode monotonically inside a bounded source window."""

    def __init__(self, video_path: Path, *, start_at: float) -> None:
        if not math.isfinite(start_at) or start_at < 0:
            raise ValueError("start_at must be finite and non-negative")
        self._capture = cv2.VideoCapture(str(video_path))
        if not self._capture.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self._fps) or self._fps <= 0:
            self._capture.release()
            raise RuntimeError(f"El video no reportó un FPS válido: {video_path}")
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(
            0,
            int(math.floor(float(start_at) * self._fps + 1e-9)),
        )
        if self._frame_count > 0:
            start_frame = min(self._frame_count - 1, start_frame)
        if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame)):
            self._capture.release()
            raise RuntimeError(
                f"No se pudo posicionar el video en {start_at:.3f}s: {video_path}"
            )
        self._next_frame_index = start_frame
        self._last_requested_at: float | None = None
        self._last_frame: np.ndarray | None = None

    def decode_frame(self, at: float) -> np.ndarray:
        requested = float(at)
        if not math.isfinite(requested) or requested < 0:
            raise ValueError("frame timestamp must be finite and non-negative")
        if (
            self._last_requested_at is not None
            and requested < self._last_requested_at - 1e-9
        ):
            raise ValueError("sequential frame timestamps must be monotonic")
        target_frame = max(
            0,
            int(math.floor(requested * self._fps + 1e-9)),
        )
        if self._frame_count > 0:
            target_frame = min(self._frame_count - 1, target_frame)
        decoded: np.ndarray | None = None
        while self._next_frame_index <= target_frame:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"No se pudo decodificar el frame en {requested:.3f}s"
                )
            decoded = _validated_bgr(frame)
            self._next_frame_index += 1
        if decoded is not None:
            self._last_frame = decoded
        if self._last_frame is None:
            raise RuntimeError(
                f"No se pudo decodificar el frame en {requested:.3f}s"
            )
        self._last_requested_at = requested
        return self._last_frame

    def close(self) -> None:
        self._capture.release()


def _face_model_directory() -> Path:
    return paths.face_model_dir


def _current_model_hashes() -> dict[str, str]:
    """Read verified hashes without instantiating OpenCV models or using network."""

    model_dir = _face_model_directory()
    manifest_path = model_dir / MANIFEST_FILENAME
    try:
        manifest = load_face_manifest(manifest_path)
        if validate_face_manifest(model_dir, manifest):
            return {}
        models = manifest["models"]
        assert isinstance(models, dict)
        return {
            name: str(models[name]["sha256"])
            for name in FACE_MODELS
            if isinstance(models.get(name), dict)
        }
    except Exception:
        return {}


def _create_builder_runtime(
    video_path: Path,
    job: JobStore,
    *,
    allow_model_download: bool,
    force: bool,
) -> _BuilderRuntime:
    metadata = probe_media(video_path)
    duration = float(metadata.get("duration_seconds") or 0.0)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("ffprobe no devolvió una duración válida")

    model_dir = _face_model_directory()
    model_paths = ensure_face_models(model_dir, allow_download=allow_model_download)
    model_hashes = {name: sha256_file(path) for name, path in model_paths.items()}
    audio_path = ensure_audio_checkpoint(
        video_path, job.audio_path, force=force
    )
    return _BuilderRuntime(
        duration_seconds=duration,
        decoder=_OpenCVFrameDecoder(video_path),
        local_analyzer=OpenCVLocalVisualAnalyzer(),
        face_backend=OpenCVFaceIdentityBackend(
            str(model_paths["yunet"]),
            str(model_paths["sface"]),
        ),
        model_hashes=model_hashes,
        audio_extractor=AudioFeatureExtractor(audio_path),
    )


def _create_window_runtime(
    video_path: Path,
    *,
    duration_seconds: float,
    start: float,
    end: float,
    allow_model_download: bool,
    audio_path: Path,
) -> _BuilderRuntime:
    model_dir = _face_model_directory()
    model_paths = ensure_face_models(
        model_dir,
        allow_download=allow_model_download,
    )
    model_hashes = {
        name: sha256_file(path) for name, path in model_paths.items()
    }
    extract_audio_window(
        video_path,
        audio_path,
        start=start,
        end=end,
    )
    return _BuilderRuntime(
        duration_seconds=duration_seconds,
        decoder=_SequentialOpenCVFrameDecoder(video_path, start_at=start),
        local_analyzer=OpenCVLocalVisualAnalyzer(),
        face_backend=OpenCVFaceIdentityBackend(
            str(model_paths["yunet"]),
            str(model_paths["sface"]),
        ),
        model_hashes=model_hashes,
        audio_extractor=_SourceTimeAudioExtractor(
            delegate=AudioFeatureExtractor(audio_path),
            source_offset=start,
            local_duration=end - start,
        ),
        window_start=start,
        window_end=end,
    )


def _sample_times(start: float, end: float, interval: float) -> list[float]:
    if not all(math.isfinite(v) for v in (start, end, interval)) or interval <= 0:
        raise ValueError("sample interval must be positive")
    if end <= start or math.ceil((end - start) / interval) + 1 > MAX_SCHEDULE_POINTS:
        raise ValueError("Scheduled sample point budget exceeded or empty scope")
    times = list(np.arange(start, end, interval, dtype=float))
    final = max(start, end - min(0.25, interval / 2.0))
    if not times or final - times[-1] > max(0.1, interval * 0.1):
        times.append(final)
    return [round(float(value), 6) for value in times]

def _normalized_boundaries(values: Sequence[float]) -> tuple[float, ...]:
    normalized: set[float] = set()
    for raw in values:
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError("editorial_boundaries debe contener tiempos finitos no negativos")
        normalized.add(round(value, 6))
    return tuple(sorted(normalized))


def _sampling_config(
    *,
    sparse_interval: float,
    dense_interval: float,
    editorial_boundaries: Sequence[float],
) -> dict[str, float]:
    config = {
        "sparse_interval_seconds": float(sparse_interval),
        "dense_interval_seconds": float(dense_interval),
        "dense_radius_seconds": _DENSE_RADIUS_SECONDS,
        "editorial_boundary_count": float(len(editorial_boundaries)),
    }
    config.update(
        {
            f"editorial_boundary_{index:04d}": float(value)
            for index, value in enumerate(editorial_boundaries)
        }
    )
    return config


def _artifact_cache_config(job: JobStore) -> dict[str, float]:
    config: dict[str, float] = {}
    for label, path in (
        ("transcript", job.transcript_path),
        ("diarization", job.diarization_path),
        ("streamer_identity", job.streamer_identity_path),
        ("global_streamer_identity", global_streamer_identity_path()),
    ):
        if not path.is_file():
            config[f"{label}_present"] = 0.0
            config[f"{label}_bytes"] = 0.0
            config[f"{label}_digest_a"] = 0.0
            config[f"{label}_digest_b"] = 0.0
            continue
        digest = sha256_file(path)
        config[f"{label}_present"] = 1.0
        config[f"{label}_bytes"] = float(path.stat().st_size)
        config[f"{label}_digest_a"] = float(int(digest[:12], 16))
        config[f"{label}_digest_b"] = float(int(digest[12:24], 16))
    return config


def _cached_timeline_matches(
    path: Path,
    *,
    source_fingerprint: str,
    model_hashes: dict[str, str],
    sampling_config: dict[str, float],
) -> bool:
    if not path.is_file() or set(model_hashes) != set(FACE_MODELS):
        return False
    try:
        timeline = SourceVisualTimeline.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        timeline.source_fingerprint == source_fingerprint
        and timeline.analyzer_version == _ANALYZER_VERSION
        and timeline.model_hashes == model_hashes
        and timeline.sampling_config == sampling_config
    )


def _decode_timestamp(timestamp: float, duration: float) -> float:
    if duration <= 0:
        raise ValueError("duration debe ser positiva")
    latest = max(0.0, duration - min(0.001, duration / 1000.0))
    return round(min(max(0.0, float(timestamp)), latest), 6)


def _decode_new_frames(
    runtime: _BuilderRuntime,
    timestamps: Sequence[float | Fraction],
    records: dict[float, _DecodedFrame],
) -> None:
    if isinstance(runtime.decoder, _ObservedSession):
        runtime.decoder.decode(runtime, timestamps, records)
        return
    for raw_timestamp in timestamps:
        timestamp = _decode_timestamp(float(raw_timestamp), runtime.duration_seconds)
        if timestamp in records:
            continue
        frame = _analysis_frame_for_index(runtime.decoder.decode_frame(timestamp))
        independent_features = runtime.local_analyzer.analyze(
            frame,
            previous_frame=None,
        )
        gray_thumbnail = cv2.resize(
            _gray(frame),
            (96, 54),
            interpolation=cv2.INTER_AREA,
        )
        histogram = cv2.calcHist(
            [gray_thumbnail], [0], None, [64], [0, 256]
        ).reshape(-1)
        total = float(histogram.sum())
        if total:
            normalized_histogram = histogram / total
        else:
            normalized_histogram = histogram
        faces = sorted(
            runtime.face_backend.detect_and_embed(frame),
            key=lambda face: face.confidence,
            reverse=True,
        )[:16]
        records[timestamp] = _DecodedFrame(
            at=timestamp,
            independent_features=independent_features,
            gray_thumbnail=gray_thumbnail,
            histogram=np.asarray(normalized_histogram, dtype=np.float32),
            faces=faces,
        )


def _decode_dense_windows(
    video_path: Path,
    runtime: _BuilderRuntime,
    windows: Sequence[tuple[float, float]],
    *,
    interval: float,
    records: dict[float, _DecodedFrame],
) -> None:
    if not isinstance(runtime.decoder, _OpenCVFrameDecoder):
        timestamps = _timestamps_for_windows(
            windows,
            interval=interval,
            existing_timestamps=set(records),
        )
        _decode_new_frames(runtime, timestamps, records)
        return

    runtime.decoder.close()
    for window in windows:
        timestamps = _timestamps_for_windows(
            [window],
            interval=interval,
            existing_timestamps=set(records),
        )
        if not timestamps:
            continue
        decoder = _SequentialOpenCVFrameDecoder(
            video_path,
            start_at=timestamps[0],
        )
        runtime.decoder = decoder
        try:
            _decode_new_frames(runtime, timestamps, records)
        finally:
            decoder.close()


def _analysis_frame_for_index(frame: np.ndarray, *, maximum_width: int = 960) -> np.ndarray:
    normalized = _validated_bgr(frame)
    height, width = normalized.shape[:2]
    if width <= maximum_width:
        return normalized
    scale = maximum_width / float(width)
    return cv2.resize(
        normalized,
        (maximum_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _relative_local_features(
    previous: _DecodedFrame | None,
    current: _DecodedFrame,
) -> LocalFrameFeatures:
    base = current.independent_features
    if previous is None:
        histogram_change = 0.0
        structural_change = 0.0
        motion = 0.0
    else:
        histogram_change = float(
            np.clip(
                0.5 * np.abs(previous.histogram - current.histogram).sum(),
                0.0,
                1.0,
            )
        )
        structural_change = compute_structural_change(
            previous.gray_thumbnail,
            current.gray_thumbnail,
        )
        motion = compute_motion(
            previous.gray_thumbnail,
            current.gray_thumbnail,
        )
    return LocalFrameFeatures(
        histogram_change_score=histogram_change,
        structural_change_score=structural_change,
        letterbox_bounds=base.letterbox_bounds,
        persistent_rectangle_candidates=base.persistent_rectangle_candidates,
        text_density=base.text_density,
        text_regions=base.text_regions,
        brightness=base.brightness,
        sharpness=base.sharpness,
        motion=motion,
    )


def _point_inside(rectangle: NormalizedRect, x: float, y: float) -> bool:
    return (
        rectangle.x <= x <= rectangle.x + rectangle.width
        and rectangle.y <= y <= rectangle.y + rectangle.height
    )


def _face_inside_rectangle(face: LandmarkedFace, rectangle: NormalizedRect) -> bool:
    center_x = face.box.x + face.box.width / 2.0
    center_y = face.box.y + face.box.height / 2.0
    return _point_inside(rectangle, center_x, center_y)


def _serializable_face_tracks(tracks: Sequence[FaceTrack]) -> list[FaceTrack]:
    return [track for track in tracks if track.embedding_centroid() is not None]


def _is_webcam_seed(rectangle: NormalizedRect) -> bool:
    area = rectangle.width * rectangle.height
    aspect_ratio = rectangle.width / rectangle.height
    horizontal_edge = rectangle.x <= 0.04 or rectangle.x + rectangle.width >= 0.96
    vertical_edge = rectangle.y <= 0.04 or rectangle.y + rectangle.height >= 0.96
    return (
        area <= 0.12
        and 0.65 <= aspect_ratio <= 1.80
        and horizontal_edge
        and vertical_edge
    )


def _persistent_webcam_tracks(
    tracks: Sequence[WebcamRectangleTrack],
    *,
    face_evidence_ids: set[str],
    require_persistence: bool,
) -> list[WebcamRectangleTrack]:
    minimum_observations = 2 if require_persistence else 1
    return [
        track
        for track in tracks
        if len({round(item.at, 6) for item in track.observations})
        >= minimum_observations
        and any(item.evidence_id in face_evidence_ids for item in track.observations)
        and any(_is_webcam_seed(item.rect) for item in track.observations)
    ]



def _profile_embedding_from_text(
    text: str,
    expected_model_hashes: dict[str, str],
    *,
    expected_source_fingerprint: str | None = None,
) -> np.ndarray | None:
    allowed_keys = {
        "contract_version",
        "profile_type",
        "name",
        "embedding",
        "track_id",
        "source_fingerprint",
        "source_reference",
        "source_range",
        "observations",
        "model_hashes",
        "created_at",
    }
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict) or set(payload) != allowed_keys:
            return None
        if (
            payload.get("contract_version") != 1
            or payload.get("profile_type") != "streamer_face"
            or payload.get("model_hashes") != expected_model_hashes
        ):
            return None
        required_text = (
            payload.get("name"),
            payload.get("track_id"),
            payload.get("source_fingerprint"),
            payload.get("source_reference"),
            payload.get("created_at"),
        )
        if any(not isinstance(value, str) or not value.strip() for value in required_text):
            return None
        source_value = payload["source_fingerprint"]
        if (
            expected_source_fingerprint is not None
            and source_value != expected_source_fingerprint
        ):
            return None
        created_at = str(payload["created_at"])
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if parsed_created_at.tzinfo is None or parsed_created_at.utcoffset() is None:
            return None
        source_range = payload.get("source_range")
        if not isinstance(source_range, dict) or set(source_range) != {"start", "end"}:
            return None
        raw_start = source_range.get("start")
        raw_end = source_range.get("end")
        raw_observations = payload.get("observations")
        if (
            isinstance(raw_start, bool)
            or not isinstance(raw_start, (int, float))
            or isinstance(raw_end, bool)
            or not isinstance(raw_end, (int, float))
            or isinstance(raw_observations, bool)
            or not isinstance(raw_observations, int)
        ):
            return None
        range_start = float(raw_start)
        range_end = float(raw_end)
        if not all(math.isfinite(value) for value in (range_start, range_end)):
            return None
        if range_start < 0 or range_end <= range_start or raw_observations < 2:
            return None
        raw_embedding = payload.get("embedding")
        if not isinstance(raw_embedding, list) or not raw_embedding:
            return None
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_embedding
        ):
            return None
        embedding = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
    except (ValueError, TypeError, json.JSONDecodeError, AttributeError):
        return None
    if embedding.size == 0 or not np.isfinite(embedding).all():
        return None
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 1e-6 else None


def _load_profile_embedding(
    path: Path,
    expected_model_hashes: dict[str, str],
    *,
    expected_source_fingerprint: str | None = None,
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return _profile_embedding_from_text(
        text,
        expected_model_hashes,
        expected_source_fingerprint=expected_source_fingerprint,
    )


def _load_saved_profile_embedding(
    job: _VisualArtifactContext,
    model_hashes: dict[str, str],
) -> np.ndarray | None:
    job_path = job.streamer_identity_path
    global_path = global_streamer_identity_path()
    try:
        job_text, global_text = atomic_read_text_batch((job_path, global_path))
    except (OSError, ValueError, RuntimeError):
        return None
    if job_text is not None:
        embedding = _profile_embedding_from_text(
            job_text,
            model_hashes,
            expected_source_fingerprint=job.fingerprint,
        )
        if embedding is not None:
            return embedding
    if global_text is not None:
        return _profile_embedding_from_text(global_text, model_hashes)
    return None

def _profile_matches_track(
    track: FaceTrack,
    profile: np.ndarray | None,
) -> bool:
    if profile is None:
        return False
    centroid = track.embedding_centroid()
    if centroid is None or centroid.size != profile.size:
        return False
    similarity = float(np.dot(centroid, profile))
    return similarity >= _PROFILE_COSINE_THRESHOLD


def _merge_profile_matching_face_tracks(
    tracks: Sequence[FaceTrack],
    profile: np.ndarray | None,
) -> list[FaceTrack]:
    """Merge layout-split streamer tracks, never simultaneous matching faces."""

    if profile is None:
        return list(tracks)
    normalized_profile = np.asarray(profile, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(normalized_profile))
    if norm <= 1e-6 or not np.isfinite(normalized_profile).all():
        return list(tracks)
    normalized_profile = normalized_profile / norm
    matching: list[FaceTrack] = []
    for track in tracks:
        centroid = track.embedding_centroid()
        if centroid is None or centroid.size != normalized_profile.size:
            continue
        if float(np.dot(centroid, normalized_profile)) >= _PROFILE_TRACK_MERGE_COSINE_THRESHOLD:
            matching.append(track)
    if len(matching) < 2:
        return list(tracks)

    groups: list[tuple[list[FaceTrack], set[float]]] = []
    for track in matching:
        times = {round(item.at, 6) for item in track.observations}
        for group, observed_times in groups:
            if not observed_times.intersection(times):
                group.append(track)
                observed_times.update(times)
                break
        else:
            groups.append(([track], times))

    replacements: dict[int, FaceTrack] = {}
    for group, _ in groups:
        merged = (
            FaceTrack(
                track_id=group[0].track_id,
                observations=sorted(
                    [item for track in group for item in track.observations],
                    key=lambda item: item.at,
                ),
            )
            if len(group) > 1 else group[0]
        )
        for track in group:
            replacements[id(track)] = merged
    result: list[FaceTrack] = []
    inserted: set[int] = set()
    for track in tracks:
        replacement = replacements.get(id(track), track)
        if id(replacement) not in inserted:
            result.append(replacement)
            inserted.add(id(replacement))
    return result


def _profile_anchored_webcam_tracks(
    face_tracks: Sequence[FaceTrack],
    *,
    identity: StreamerIdentityResult,
    saved_profile_embedding: np.ndarray | None,
    sample_timestamps: Sequence[float] = (),
    excluded_hold_times: set[float] | None = None,
) -> list[WebcamRectangleTrack]:
    """Infer a borderless corner webcam only from a verified streamer track."""

    if identity.status is not IdentityStatus.CONFIDENT or identity.track_id is None:
        return []
    streamer = next(
        (track for track in face_tracks if track.track_id == identity.track_id),
        None,
    )
    if streamer is None or not _profile_matches_track(
        streamer,
        saved_profile_embedding,
    ):
        return []

    observations: list[WebcamRegionObservation] = []
    for observation in streamer.observations:
        box = observation.face.box
        face_area = box.width * box.height
        center_x = box.x + box.width / 2.0
        center_y = box.y + box.height / 2.0
        horizontal_anchor = (
            "left" if center_x <= 0.35 else "right" if center_x >= 0.65 else None
        )
        vertical_anchor = (
            "top" if center_y <= 0.40 else "bottom" if center_y >= 0.60 else None
        )
        if (
            horizontal_anchor is None
            or vertical_anchor is None
            or not 0.002 <= face_area <= 0.04
        ):
            continue
        width = float(np.clip(box.width * 3.5, 0.20, 0.36))
        height = float(np.clip(box.height * 2.2, 0.24, 0.42))
        x = 0.0 if horizontal_anchor == "left" else 1.0 - width
        y = 0.0 if vertical_anchor == "top" else 1.0 - height
        rectangle = NormalizedRect(x=x, y=y, width=width, height=height)
        if not _face_inside_rectangle(observation.face, rectangle):
            continue
        observations.append(
            WebcamRegionObservation(
                at=observation.at,
                rect=rectangle,
                confidence=0.84,
                evidence_id=f"profile-webcam:{observation.at:.6f}",
                face_track_ids=(streamer.track_id,),
            )
        )
    if len({round(item.at, 6) for item in observations}) < 2:
        return []
    corner_seeds = tuple(sorted(observations, key=lambda item: item.at))
    corner_times = {round(item.at, 6) for item in corner_seeds}
    streamer_times = {
        round(item.at, 6)
        for item in streamer.observations
    }
    for raw_at in sorted({round(float(value), 6) for value in sample_timestamps}):
        if (
            raw_at in corner_times
            or raw_at in streamer_times
            or raw_at in (excluded_hold_times or ())
        ):
            continue
        previous = next(
            (item for item in reversed(corner_seeds) if item.at < raw_at),
            None,
        )
        following = next(
            (item for item in corner_seeds if item.at > raw_at),
            None,
        )
        bridged = (
            previous is not None
            and following is not None
            and following.at - previous.at <= _FULL_CAMERA_BRIDGE_MAX_SECONDS
            and _rect_iou(previous.rect, following.rect) >= 0.50
        )
        edge_held = (
            previous is not None
            and raw_at - previous.at <= _FULL_CAMERA_EDGE_HOLD_SECONDS
        )
        if not bridged and not edge_held:
            continue
        nearest = min(
            (item for item in (previous, following) if item is not None),
            key=lambda item: abs(item.at - raw_at),
        )
        observations.append(
            WebcamRegionObservation(
                at=raw_at,
                rect=nearest.rect,
                confidence=0.72,
                evidence_id=f"profile-webcam-hold:{raw_at:.6f}",
                face_track_ids=(streamer.track_id,),
            )
        )
    observations.sort(key=lambda item: item.at)
    associated = associate_webcam_tracks(
        observations,
        sample_timestamps=sample_timestamps,
    )
    result: list[WebcamRectangleTrack] = []
    for track in associated:
        direct_observations = [
            item
            for item in track.observations
            if not item.evidence_id.startswith("profile-webcam-hold:")
        ]
        if len({round(item.at, 6) for item in direct_observations}) < 2:
            continue
        result.append(
            WebcamRectangleTrack(
                track_id=f"webcam_profile_{len(result) + 1:03d}",
                observations=list(track.observations),
            )
        )
    return result

def _role_verified_webcam_tracks(
    tracks: Sequence[WebcamRectangleTrack],
    *,
    face_tracks: Sequence[FaceTrack],
    identity: StreamerIdentityResult,
    saved_profile_embedding: np.ndarray | None,
) -> list[WebcamRectangleTrack]:
    if identity.status is not IdentityStatus.CONFIDENT or identity.track_id is None:
        return []
    streamer = next(
        (track for track in face_tracks if track.track_id == identity.track_id),
        None,
    )
    if streamer is None:
        return []
    verified: list[WebcamRectangleTrack] = []
    for track in tracks:
        if identity.track_id not in track.face_track_ids:
            continue
        rectangles_by_time = {
            round(observation.at, 6): observation.rect
            for observation in track.observations
        }
        seen_inside = False
        seen_outside = False
        for observation in streamer.observations:
            rectangle = rectangles_by_time.get(round(observation.at, 6))
            if rectangle is not None and _face_inside_rectangle(
                observation.face, rectangle
            ):
                seen_inside = True
            else:
                seen_outside = True
        profile_verified = seen_inside and _profile_matches_track(
            streamer, saved_profile_embedding
        )
        transition_verified = seen_inside and seen_outside
        if profile_verified or transition_verified:
            verified.append(track)
    return verified


def _full_camera_times(
    *,
    face_tracks: Sequence[FaceTrack],
    identity: StreamerIdentityResult,
    verified_webcams: Sequence[WebcamRectangleTrack],
    local_features: dict[float, LocalFrameFeatures],
    saved_profile_embedding: np.ndarray | None,
) -> dict[float, tuple[str, ...]]:
    """Map camera geometry times to their supporting face tracks, including holds."""
    if identity.status is not IdentityStatus.CONFIDENT or identity.track_id is None:
        return {}
    streamer = next(
        (track for track in face_tracks if track.track_id == identity.track_id),
        None,
    )
    if streamer is None:
        return {}
    profile_matches = _profile_matches_track(streamer, saved_profile_embedding)
    if not verified_webcams and not profile_matches:
        return {}
    persistent_profile_face = profile_matches and len(
        {round(item.at, 6) for item in streamer.observations}
    ) >= 2
    webcam_rectangles_by_time: dict[float, list[NormalizedRect]] = {}
    for webcam_track in verified_webcams:
        for webcam_observation in webcam_track.observations:
            webcam_rectangles_by_time.setdefault(round(webcam_observation.at, 6), []).append(
                webcam_observation.rect
            )
    result: dict[float, tuple[str, ...]] = {}
    profile_candidates = [
        track for track in face_tracks
        if track is not streamer
        and _profile_matches_track(track, saved_profile_embedding)
        and len({round(item.at, 6) for item in track.observations}) >= 2
    ]
    active_face_ids: dict[float, set[str]] = {}
    for track in face_tracks:
        for item in track.observations:
            active_face_ids.setdefault(round(item.at, 6), set()).add(track.track_id)
    for track, observation in (
        (track, observation)
        for track in (streamer, *profile_candidates)
        for observation in track.observations
    ):
        at = round(observation.at, 6)
        features = local_features.get(at)
        if features is None:
            continue
        # A later independent profile match supports camera geometry, not a
        # reassignment of the global identity or a merge of simultaneous faces.
        if track is not streamer and active_face_ids.get(at) != {track.track_id}:
            continue
        inside_webcam = any(
            _face_inside_rectangle(observation.face, rectangle)
            for rectangle in webcam_rectangles_by_time.get(at, [])
        )
        face_area = observation.face.box.width * observation.face.box.height
        face_center_x = observation.face.box.x + observation.face.box.width / 2.0
        face_center_y = observation.face.box.y + observation.face.box.height / 2.0
        small_centered_profile_face = (
            (persistent_profile_face if track is streamer else True)
            and not webcam_rectangles_by_time.get(at)
            and not features.persistent_rectangle_candidates
            and _PROFILE_FULL_CAMERA_FACE_AREA_MIN
            <= face_area
            <= _PROFILE_FULL_CAMERA_FACE_AREA_MAX
            and 0.25 <= face_center_x <= 0.75
            and 0.20 <= face_center_y <= 0.80
        )
        legacy_large_face = (
            face_area >= 0.08
            and features.text_density < _TEXT_BROWSER_THRESHOLD
        )
        if (
            not inside_webcam
            and features.letterbox_bounds is None
            and (legacy_large_face or small_centered_profile_face)
        ):
            result[at] = (track.track_id,)

    if not persistent_profile_face or not result:
        return result

    seed_times = tuple(sorted(result))
    all_face_times = {
        round(observation.at, 6)
        for track in face_tracks
        for observation in track.observations
    }

    def continuity_safe(at: float) -> bool:
        features = local_features[at]
        return (
            at not in all_face_times
            and not webcam_rectangles_by_time.get(at)
            and not features.persistent_rectangle_candidates
            and features.letterbox_bounds is None
            and features.histogram_change_score
            <= _FULL_CAMERA_CONTINUITY_HISTOGRAM_MAX
            and features.structural_change_score
            <= _FULL_CAMERA_CONTINUITY_STRUCTURAL_MAX
        )

    for at in sorted(local_features):
        if at in result or not continuity_safe(at):
            continue
        previous_seed = next(
            (seed for seed in reversed(seed_times) if seed < at),
            None,
        )
        next_seed = next((seed for seed in seed_times if seed > at), None)
        bridged = (
            previous_seed is not None
            and next_seed is not None
            and next_seed - previous_seed <= _FULL_CAMERA_BRIDGE_MAX_SECONDS
        )
        edge_held = (
            previous_seed is not None
            and at - previous_seed <= _FULL_CAMERA_EDGE_HOLD_SECONDS
        )
        if bridged and previous_seed is not None and next_seed is not None:
            # Both bounding seeds support a bridge. Preserve distinct track IDs
            # rather than treating stable geometry as global identity continuity.
            result[at] = tuple(sorted(set(result[previous_seed]) | set(result[next_seed])))
        elif edge_held and previous_seed is not None:
            result[at] = result[previous_seed]
    return result


def _sample_window(
    timestamps: Sequence[float],
    index: int,
    duration: float,
) -> TimeRange:
    start = float(timestamps[index])
    if index + 1 < len(timestamps):
        end = min(duration, float(timestamps[index + 1]))
    else:
        end = duration
    if end <= start:
        end = min(duration, start + min(0.001, max(duration - start, 0.001)))
    return TimeRange(start=start, end=end)


def _analyze_records(
    records: dict[float, _DecodedFrame],
    *,
    runtime: _BuilderRuntime,
    job: _VisualArtifactContext,
    require_webcam_persistence: bool = True,
) -> tuple[
    list[SourceVisualSample],
    list[FaceTrack],
    list[WebcamRectangleTrack],
    dict[float, tuple[str, ...]],
    StreamerIdentityResult,
]:
    ordered = [records[key] for key in sorted(records)]
    if not ordered:
        raise RuntimeError("No se pudo decodificar ninguna muestra visual")

    local_features: dict[float, LocalFrameFeatures] = {}
    previous_record: _DecodedFrame | None = None
    for record in ordered:
        local_features[record.at] = _relative_local_features(previous_record, record)
        previous_record = record

    face_observations: list[FaceObservation] = []
    observations_by_time: dict[float, list[FaceObservation]] = {}
    for record in ordered:
        current: list[FaceObservation] = []
        for index, face in enumerate(record.faces):
            observation = FaceObservation(
                at=record.at,
                face=face,
                source_state=SourceState.UNKNOWN,
                evidence_id=f"face:{record.at:.6f}:{index:03d}",
                embedding_quality=max(0.05, face.confidence),
            )
            face_observations.append(observation)
            current.append(observation)
        observations_by_time[record.at] = current

    all_face_tracks = associate_face_tracks(
        face_observations,
        sample_timestamps=[record.at for record in ordered],
    )
    face_tracks = _serializable_face_tracks(all_face_tracks)
    saved_profile_embedding = _load_saved_profile_embedding(job, runtime.model_hashes)
    face_tracks = _merge_profile_matching_face_tracks(
        face_tracks, saved_profile_embedding
    )
    face_id_by_evidence = {
        observation.evidence_id: track.track_id
        for track in face_tracks
        for observation in track.observations
    }

    webcam_observations: list[WebcamRegionObservation] = []
    webcam_face_evidence_ids: set[str] = set()
    for record in ordered:
        features = local_features[record.at]
        for index, rectangle in enumerate(features.persistent_rectangle_candidates):
            if (
                features.letterbox_bounds is not None
                and _rect_iou(rectangle, features.letterbox_bounds) >= 0.70
            ):
                continue
            evidence_id = f"rect:{record.at:.6f}:{index:03d}"
            if any(
                _face_inside_rectangle(observation.face, rectangle)
                for observation in observations_by_time[record.at]
            ):
                webcam_face_evidence_ids.add(evidence_id)
            contained_ids = tuple(
                sorted(
                    {
                        face_id_by_evidence[observation.evidence_id]
                        for observation in observations_by_time[record.at]
                        if observation.evidence_id in face_id_by_evidence
                        and _face_inside_rectangle(observation.face, rectangle)
                    }
                )
            )
            webcam_observations.append(
                WebcamRegionObservation(
                    at=record.at,
                    rect=rectangle,
                    confidence=0.75,
                    evidence_id=evidence_id,
                    face_track_ids=contained_ids,
                )
            )

    persistent_rectangle_tracks = _persistent_webcam_tracks(
        associate_webcam_tracks(
            webcam_observations,
            sample_timestamps=[record.at for record in ordered],
        ),
        face_evidence_ids=webcam_face_evidence_ids,
        require_persistence=require_webcam_persistence,
    )
    identity = resolve_streamer_identity(
        face_tracks,
        saved_profile_embedding=saved_profile_embedding,
    )

    if not any(
        identity.track_id in track.face_track_ids
        for track in persistent_rectangle_tracks
    ):
        profile_full_camera_times = _full_camera_times(
            face_tracks=face_tracks,
            identity=identity,
            verified_webcams=[],
            local_features=local_features,
            saved_profile_embedding=saved_profile_embedding,
        )
        inferred_webcams = _profile_anchored_webcam_tracks(
            face_tracks,
            identity=identity,
            saved_profile_embedding=saved_profile_embedding,
            sample_timestamps=[record.at for record in ordered],
            excluded_hold_times=set(profile_full_camera_times),
        )
        persistent_rectangle_tracks.extend(inferred_webcams)
    webcam_tracks = (
        _role_verified_webcam_tracks(
            persistent_rectangle_tracks,
            face_tracks=face_tracks,
            identity=identity,
            saved_profile_embedding=saved_profile_embedding,
        )
        if require_webcam_persistence
        else persistent_rectangle_tracks
    )
    full_camera_times = (
        _full_camera_times(
            face_tracks=face_tracks,
            identity=identity,
            verified_webcams=webcam_tracks,
            local_features=local_features,
            saved_profile_embedding=saved_profile_embedding,
        )
        if require_webcam_persistence
        else {}
    )
    persistent_at: dict[float, list[NormalizedRect]] = {}
    for webcam_track in persistent_rectangle_tracks:
        for webcam_observation in webcam_track.observations:
            persistent_at.setdefault(webcam_observation.at, []).append(webcam_observation.rect)
    webcam_at: dict[float, list[tuple[WebcamRectangleTrack, WebcamRegionObservation]]] = {}
    for webcam_track in webcam_tracks:
        for webcam_observation in webcam_track.observations:
            webcam_at.setdefault(webcam_observation.at, []).append((webcam_track, webcam_observation))

    timestamps = [record.at for record in ordered]
    runtime_window_end = getattr(runtime, "window_end", None)
    analysis_end = runtime.duration_seconds if runtime_window_end is None else runtime_window_end
    transcript_exists = job.transcript_path.is_file()
    diarization_exists = job.diarization_path.is_file()
    samples: list[SourceVisualSample] = []
    webcam_face_ids_by_time: dict[float, tuple[str, ...]] = {}
    for index, record in enumerate(ordered):
        features = local_features[record.at]
        face_ids = sorted(
            {
                face_id_by_evidence[observation.evidence_id]
                for observation in observations_by_time[record.at]
                if observation.evidence_id in face_id_by_evidence
            }
        )
        candidates = webcam_at.get(record.at, [])
        chosen = sorted(
            candidates,
            key=lambda item: (-len(item[0].face_track_ids), item[0].track_id),
        )[0] if candidates else None
        is_full_camera = chosen is None and record.at in full_camera_times
        rectangle_candidates = _unique_rectangles(
            list(features.persistent_rectangle_candidates)
        )
        persistent_rectangles = _unique_rectangles(persistent_at.get(record.at, []))
        camera_face_ids: tuple[str, ...] = ()
        if chosen is not None:
            camera_face_ids = tuple(chosen[1].face_track_ids)
        elif is_full_camera and identity.track_id is not None:
            if full_camera_times[record.at] == (identity.track_id,):
                camera_face_ids = (identity.track_id,)
        webcam_face_ids_by_time[record.at] = camera_face_ids
        sample_window = _sample_window(timestamps, index, analysis_end)
        audio_features: dict[str, float] = {}
        audio_window: TimeRange | None = None
        if runtime.audio_extractor is not None:
            audio_features = runtime.audio_extractor.window(
                sample_window.start,
                sample_window.end,
            ).to_dict()
            audio_window = sample_window
        evidence_ids = [f"frame:{record.at:.6f}"]
        evidence_ids.extend(
            observation.evidence_id for observation in observations_by_time[record.at]
        )
        if chosen is not None:
            evidence_ids.append(chosen[1].evidence_id)
        if is_full_camera:
            evidence_ids.append("inferred_full_camera")
        samples.append(
            SourceVisualSample(
                sample_id=f"sample_{int(round(record.at * 1_000_000)):012d}",
                at=record.at,
                histogram_change_score=features.histogram_change_score,
                structural_change_score=features.structural_change_score,
                letterbox_bounds=features.letterbox_bounds,
                persistent_rectangles=persistent_rectangles,
                rectangle_candidates=rectangle_candidates,
                text_density=features.text_density,
                text_regions=list(features.text_regions),
                face_track_ids=face_ids,
                webcam_track_id=chosen[0].track_id if chosen is not None else None,
                webcam_rect=chosen[1].rect if chosen is not None else None,
                brightness=features.brightness,
                sharpness=features.sharpness,
                motion=features.motion,
                audio_features=audio_features,
                audio_window=audio_window,
                transcript_window=sample_window if transcript_exists else None,
                diarization_window=sample_window if diarization_exists else None,
                evidence_ids=list(dict.fromkeys(evidence_ids)),
            )
        )

    state_by_time = {
        sample.at: classify_source_state(sample)[0]
        for sample in samples
    }
    for track in face_tracks:
        for observation in track.observations:
            observation.source_state = state_by_time.get(observation.at, SourceState.UNKNOWN)
    return samples, face_tracks, webcam_tracks, webcam_face_ids_by_time, identity


def _apply_identity_and_duration(
    spans: Sequence[SourceVisualSpan],
    *,
    samples: Sequence[SourceVisualSample],
    duration: float,
    identity_status: IdentityStatus,
    webcam_face_ids_by_time: dict[float, tuple[str, ...]],
    streamer_track_id: str | None,
    webcam_tracks: Sequence[WebcamTrack],
    window_start: float = 0.0,
    window_end: float | None = None,
) -> list[SourceVisualSpan]:
    if not spans:
        return []
    webcam_by_id = {track.track_id: track for track in webcam_tracks}
    finalized: list[SourceVisualSpan] = []
    effective_end = duration if window_end is None else window_end
    previous_end = window_start
    for index, span in enumerate(spans):
        start = window_start if index == 0 else previous_end
        end = effective_end if index == len(spans) - 1 else min(effective_end, span.source.end)
        if end <= start:
            continue
        status = IdentityStatus.UNAVAILABLE
        resolved_track: str | None = None
        webcam = webcam_by_id.get(span.webcam_track_id or "")
        has_camera = span.state in {SourceState.FULL_CAMERA, SourceState.CONTENT_WITH_CAMERA}
        samples_in_span = [
            sample
            for sample in samples
            if start <= sample.at < end
            and (
                span.state is SourceState.FULL_CAMERA
                or sample.webcam_track_id == span.webcam_track_id
            )
        ]
        streamer_is_current = bool(samples_in_span) and all(
            streamer_track_id in webcam_face_ids_by_time.get(sample.at, ())
            for sample in samples_in_span
        )
        role_support = span.state is SourceState.FULL_CAMERA or (
            webcam is not None
            and streamer_track_id is not None
            and streamer_track_id in webcam.face_track_ids
        )
        if (
            has_camera
            and identity_status is IdentityStatus.CONFIDENT
            and streamer_track_id is not None
            and role_support
            and streamer_is_current
        ):
            status = IdentityStatus.CONFIDENT
            resolved_track = streamer_track_id
        elif has_camera and identity_status is IdentityStatus.AMBIGUOUS:
            status = IdentityStatus.AMBIGUOUS
        payload = span.model_dump()
        payload.update(
            {
                "source": TimeRange(start=start, end=end),
                "streamer_track_id": resolved_track,
                "identity_status": status,
            }
        )
        finalized.append(SourceVisualSpan.model_validate(payload))
        previous_end = end
    if finalized and finalized[-1].source.end != effective_end:
        payload = finalized[-1].model_dump()
        payload["source"] = TimeRange(
            start=finalized[-1].source.start,
            end=effective_end,
        )
        finalized[-1] = SourceVisualSpan.model_validate(payload)
    return finalized


def _close_runtime(runtime: object | None) -> None:
    if runtime is None:
        return
    close = getattr(getattr(runtime, "decoder", None), "close", None)
    if callable(close):
        close()




def build_source_visual_timeline(
    video_path: Path,
    *,
    editorial_boundaries: Sequence[float] = (),
    sparse_interval: float = 10.0,
    dense_interval: float = 0.5,
    allow_model_download: bool = False,
    force: bool = False,
) -> Path:
    """Serialize one visual-index build per source job across processes."""

    return build_source_visual_timeline_result(
        video_path, editorial_boundaries=editorial_boundaries, sparse_interval=sparse_interval,
        dense_interval=dense_interval, allow_model_download=allow_model_download, force=force,
    ).artifact_path


_MAX_INSPECTION_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class VisualTimelineBuildResult:
    artifact_path: Path
    cache_hit: bool
    indexed_samples_this_run: int


class ObservedPublicationError(RuntimeError):
    """An immutable completion exists; publication/bookkeeping needs recovery."""

    def __init__(self, phase: str, artifact_path: Path, complete_sha256: str) -> None:
        self.phase = phase
        self.artifact_path = artifact_path
        self.complete_sha256 = complete_sha256
        super().__init__(f"{phase}: committed observed generation {artifact_path} ({complete_sha256})")


def _resolve_observed_source(video_path: Path) -> tuple[RenderRuntime, SourceTimeOrigin]:
    pinned = resolve_render_runtime(decode_requirements())
    return pinned, probe_source(video_path, pinned)


def _observed_face_backend(*, allow_model_download: bool) -> FaceIdentityBackend:
    model_dir = _face_model_directory()
    if allow_model_download:
        model_paths = ensure_face_models(model_dir, allow_download=True)
    else:
        manifest = load_face_manifest(model_dir / MANIFEST_FILENAME)
        failures = validate_face_manifest(model_dir, manifest)
        if failures:
            raise ModelAssetError(f"Cached face manifest invalid: {failures}")
        model_paths = {name: model_dir / asset.filename for name, asset in FACE_MODELS.items()}
        if any(not validate_model_file(model_paths[name], asset) for name, asset in FACE_MODELS.items()):
            raise ModelAssetError("Cached face model invalid; no maintenance/download permitted")
    return OpenCVFaceIdentityBackend(str(model_paths["yunet"]), str(model_paths["sface"]))


def _observed_config(context: _VisualArtifactContext, sampling: dict[str, float],
                     model_hashes: dict[str, str]) -> dict[str, Any]:
    legacy: dict[str, dict[str, str | None]] = {}
    for label, path in (("transcript", context.transcript_path), ("diarization", context.diarization_path),
                        ("streamer_identity", context.streamer_identity_path),
                        ("global_streamer_identity", global_streamer_identity_path())):
        legacy[label] = {"status": "unverified_legacy" if path.is_file() else "missing",
                         "sha256": file_sha256(path) if path.is_file() else None}
    return {"sampling": sampling, "model_hashes": model_hashes,
            "dependencies": {"opencv": cv2.__version__, "numpy": np.__version__}, "legacy_context": legacy}


def _write_observed_json(path: Path, value: object) -> None:
    encoded = canonical(value) + "\n"
    if emitted_text_size(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("Generation JSON byte budget exceeded")
    atomic_write_text(path, encoded)


class _ObservedSession:
    """Process and persist one bounded selection at a time; retain no full pixels."""

    def __init__(self, pinned: RenderRuntime, origin: SourceTimeOrigin, generation: Path,
                 start: float | Fraction, end: float | Fraction) -> None:
        self.pinned, self.origin, self.generation = pinned, origin, generation
        self.start, self.end = source_time(start), source_time(end)
        self.attempted: set[Fraction] = set()
        self.observations: list[dict[str, Any]] = []
        self.associations: list[dict[str, Any]] = []
        self.audio_associations: list[dict[str, Any]] = []
        self.unavailable: list[dict[str, Any]] = []
        self.intervals: list[tuple[Fraction, Fraction, str]] = []
        self.created_files = 2  # reserve the timeline and companion clock JSON
        self.created_bytes = 0
        self._metadata_slot_bytes = MAX_ARTIFACT_BYTES
        self._metadata_reserved = 3 * MAX_ARTIFACT_BYTES  # timeline, clock, completion (not self-catalogued)
        self._work_reserved = 0
        if self._metadata_reserved > MAX_GENERATION_BYTES:
            raise ValueError("Final generation metadata byte budget exceeded before processing")

    def _reserve_files(self, count: int) -> None:
        if self.created_files + count > MAX_CATALOG_ARTIFACTS:
            raise ValueError("Cumulative generation catalog budget exceeded before processing")

    def _check_work_budget(self, bound: int) -> None:
        if self.created_bytes + self._metadata_reserved + bound > MAX_GENERATION_BYTES:
            raise ValueError("Next work plus final metadata exceeds generation byte budget before processing")

    def _begin_work(self, bound: int) -> None:
        if self._work_reserved:
            raise RuntimeError("Previous observed work reservation was not reconciled")
        self._check_work_budget(bound)
        self._work_reserved = bound

    def _finish_work(self) -> None:
        self._work_reserved = 0  # unused estimate is released, actual written bytes remain charged

    def _check_write_size(self, size: int, *, metadata: bool = False) -> None:
        reservation = self._metadata_slot_bytes if metadata else self._work_reserved
        if size > MAX_ARTIFACT_BYTES or size > reservation or self.created_bytes + size > MAX_GENERATION_BYTES:
            raise ValueError("Serialized artifact exceeds reserved byte budget before file write")

    def _record_write(self, path: Path, size: int, *, metadata: bool = False) -> None:
        if path.stat().st_size != size:
            raise ValueError("Actual emitted file bytes disagree with checked serialization")
        self.created_bytes += size
        if metadata:
            self._metadata_reserved -= self._metadata_slot_bytes
        else:
            self.created_files += 1
            self._work_reserved -= size

    def _write_text(self, path: Path, text: str) -> None:
        size = emitted_text_size(text)
        self._check_write_size(size)
        atomic_write_text(path, text)
        self._record_write(path, size)

    def _write_binary(self, path: Path, body: bytes) -> None:
        size = len(body)
        self._check_write_size(size)
        with path.open("xb") as handle:
            handle.write(body)
        self._record_write(path, size)

    def write_metadata(self, path: Path, value: object) -> None:
        size = emitted_text_size(canonical(value) + "\n")
        self._check_write_size(size, metadata=True)
        _write_observed_json(path, value)
        self._record_write(path, size, metadata=True)

    def _video_capacity(self) -> int:
        video = next(s for s in self.origin.streams if s.kind == "video")
        raw_bytes, pcm_reserve = 3 * video.width * video.height, 3 * 16000 * 2 + 8
        capacity = min((64_000_000 - pcm_reserve) // raw_bytes,
                       (192_000_000 - 4 * raw_bytes - pcm_reserve) // raw_bytes, 1200)
        if capacity < 1:
            raise ValueError("Observed source dimensions exceed bounded BGR working/payload budgets")
        return capacity

    def _video_byte_bound(self, count: int) -> int:
        video = next(s for s in self.origin.streams if s.kind == "video")
        width = min(960, video.width)
        height = video.height if video.width <= 960 else max(1, round(video.height * 960 / video.width))
        # Deliberately conservative JPEG bound: padded8x8 blocks, three8-bit
        # components,16bytes/coefficient plus64KiB headers. Valid files are also
        # capped by the unchanged per-artifact ceiling; actual encoded size wins.
        jpeg_bound = min(MAX_ARTIFACT_BYTES, math.ceil(width / 8) * math.ceil(height / 8) * 64 * 3 * 16 + 65536)
        return MAX_ARTIFACT_BYTES + count * jpeg_bound  # observed receipt + only this next batch

    def decode_frame(self, at: float) -> np.ndarray:
        raise RuntimeError("Observed frames must be decoded in bounded batches")

    def decode(self, runtime: _BuilderRuntime, timestamps: Sequence[float | Fraction], records: dict[float, _DecodedFrame]) -> None:
        points = sorted({source_time(t) for t in timestamps} - self.attempted)
        if len(points) + len(self.attempted) > MAX_SCHEDULE_POINTS:
            raise ValueError("Cumulative scheduled sample point budget exceeded")
        if any(not self.start <= t < self.end for t in points):
            raise ValueError("Observed sample outside declared analyzer scope")
        # Selected bytes plus combined analysis PCM <= decoder retention. Reserve
        # predecessor/lookahead and two array/analysis copies separately.
        capacity = self._video_capacity()
        offset = 0
        while offset < len(points):
            batch = points[offset:offset + capacity]
            batch = [t for t in batch if t - batch[0] <= 2]
            offset += len(batch)
            self._reserve_files(1 + len(batch))  # worst case: all scheduled points covered
            self._begin_work(self._video_byte_bound(len(batch)))
            end = min(self.end, batch[-1] + Fraction(1, 1000))
            window = Window(batch[0], end)
            limits = DecodeLimits(max_window=Fraction(3))
            decoded = observe_windows(self.pinned, self.origin, (window,), tuple(batch),
                                      audio_policy="analysis16k-mono", limits=limits)
            if len(decoded) != 1:
                raise ValueError("One analyzer batch requires exactly one observed window")
            result = decoded[0]
            result.receipt.validate_origin(self.origin)
            payload = json.loads(result.receipt.payload_json)
            if [exact(o["requested"]) for o in payload["outcomes"]] != batch:
                raise ValueError("Observed outcomes do not match scheduled requests")
            name = f"video-{len(self.observations):06d}.json"
            self._write_text(self.generation / name, result.receipt.to_json() + "\n")
            self.observations.append({"path": name, "kind": "video", "wav": None,
                                      "observation_id": json.loads(result.receipt.to_json())["sha256"]})
            frames = {f.tag.n: f for f in result.frames}
            for outcome in payload["outcomes"]:
                requested = exact(outcome["requested"])
                self.attempted.add(requested)
                if outcome["status"] != "covered":
                    self.unavailable.append({"requested": rational(requested), "status": outcome["status"], "receipt": name})
                    continue
                observed = frames.get(outcome["frame_n"])
                if observed is None or not observed.raw_start <= self.origin.to_raw(requested) < observed.raw_end:
                    raise ValueError("Selected image does not cover requested sample")
                frame = _analysis_frame_for_index(observed.array_copy())
                independent = runtime.local_analyzer.analyze(frame, previous_frame=None)
                thumbnail = cv2.resize(_gray(frame), (96, 54), interpolation=cv2.INTER_AREA)
                histogram: np.ndarray = cv2.calcHist([thumbnail], [0], None, [64], [0, 256]).reshape(-1)
                if float(histogram.sum()):
                    histogram = histogram / float(histogram.sum())
                faces = sorted(runtime.face_backend.detect_and_embed(frame), key=lambda face: face.confidence,
                               reverse=True)[:16]
                timestamp = float(requested)
                records[timestamp] = _DecodedFrame(timestamp, independent, thumbnail,
                                                     np.asarray(histogram, dtype=np.float32), faces)
                preview = f"sample-{int(round(timestamp * 1_000_000)):012d}.jpg"
                encoded_ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not encoded_ok:
                    raise RuntimeError("Could not persist observed analysis preview")
                self._write_binary(self.generation / preview, jpeg.tobytes())
                self.associations.append({"sample_id": f"sample_{int(round(timestamp * 1_000_000)):012d}",
                    "requested": rational(requested), "receipt": name, "frame_n": observed.tag.n,
                    "raw_pts": observed.tag.pts, "time_base": rational(observed.tag.time_base),
                    "coverage_start": rational(observed.raw_start - self.origin.origin),
                    "coverage_end": rational(observed.raw_end - self.origin.origin),
                    "raw_bgr_sha256": observed.sha256, "analysis_sha256": digest(frame.tobytes()),
                    "analysis_width": frame.shape[1], "analysis_height": frame.shape[0],
                    "transform": {"maximum_width": 960, "resize": "INTER_AREA", "preview": "JPEG90"},
                    "preview": preview, "preview_sha256": file_sha256(self.generation / preview)})
                del frame
                del observed
            # No decoded batches or raw pixel bytes survive this iteration.
            del frames, result, decoded
            self._finish_work()
    def video_receipts(self) -> Iterator[SourceDecodeReceipt]:
        for row in self.observations:
            if row["kind"] == "video":
                envelope = read_generation_json(self.generation / row["path"])
                yield SourceDecodeReceipt.from_json(canonical(envelope), origin=self.origin, policy=envelope["policy"])

    def attach_audio(self, samples: list[SourceVisualSample]) -> list[SourceVisualSample]:
        groups: list[list[tuple[int, TimeRange]]] = []
        timestamps = [s.at for s in samples]
        for index in range(len(samples)):
            interval = _sample_window(timestamps, index, float(self.end))
            effective_start = max(Fraction(0), source_time(interval.start) - Fraction(1, 2))
            effective_end = min(self.origin.duration, source_time(interval.end) + Fraction(1, 4))
            if effective_end - effective_start > 300:
                raise ValueError("A complete audio feature interval exceeds continuous 300s policy")
            if (not groups or effective_end - max(Fraction(0),
                    source_time(groups[-1][0][1].start) - Fraction(1, 2)) > 300):
                groups.append([])
            groups[-1].append((index, interval))
        for group_index, group in enumerate(groups):
            self._reserve_files(2)
            group_end = self.end if group[-1][0] == len(samples) - 1 else source_time(group[-1][1].end)
            window = Window(source_time(group[0][1].start), group_end)
            effective_duration = min(self.origin.duration, window.end + Fraction(1, 4)) - max(Fraction(0), window.start - Fraction(1, 2))
            wav_bound = min(MAX_ARTIFACT_BYTES, math.ceil(effective_duration * 16000) * 2 + 8 + 44)
            self._begin_work(MAX_ARTIFACT_BYTES + wav_bound)
            observed = observe_audio_window(self.pinned, self.origin, window,
                audio_policy="analysis16k-mono", limits=AudioDecodeLimits())
            observed.receipt.validate_origin(self.origin)
            name, wav_name = f"audio-{group_index:06d}.json", f"audio-{group_index:06d}.wav"
            audio = json.loads(observed.receipt.payload_json)["audio"][0]
            self._write_text(self.generation / name, observed.receipt.to_json() + "\n")
            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(observed.pcm)
            self._write_binary(self.generation / wav_name, wav_bytes.getvalue())
            self.observations.append({"path": name, "kind": "audio", "wav": wav_name,
                "observation_id": json.loads(observed.receipt.to_json())["sha256"]})
            if audio["policy"] != "analysis16k-mono":
                raise ValueError("Analyzer audio requires observed 16k mono policy")
            extractor = AudioFeatureExtractor(self.generation / wav_name)
            for index, interval in group:
                sample = samples[index]
                try:
                    intended_end = self.end if index == len(samples) - 1 else source_time(interval.end)
                    mapped = map_source_audio_window(observed.receipt, self.origin, interval.start, intended_end)
                except AudioUnavailableError as exc:
                    self.audio_associations.append({"sample_id": sample.sample_id, "status": "unavailable",
                        "receipt": name, "slice": None, "reason": str(exc)})
                    continue
                features = extractor.window_samples(mapped.first, mapped.count, observation_id=mapped.observation_id,
                    source_start=mapped.included_start, source_end=mapped.included_end)
                samples[index] = SourceVisualSample.model_validate({**sample.model_dump(),
                    "audio_features": features.to_dict(), "audio_window": interval})
                self.audio_associations.append({"sample_id": sample.sample_id, "status": "covered",
                    "receipt": name, "slice": mapped.to_record(), "reason": None})
            del observed
            self._finish_work()
        return samples


def _mask_unavailable_spans(
    spans: list[SourceVisualSpan], session: _ObservedSession, *, no_samples: bool,
) -> list[SourceVisualSpan]:
    merged = supported_unavailable_intervals(session.origin, session.start, session.end,
                                             session.video_receipts(), no_samples=no_samples)
    session.intervals = list(merged)
    # The public v1 contract has float span endpoints; receipt boundaries remain
    # exact. Collapse only identical float representations to avoid zero-width
    # contract spans when a rational boundary (e.g.1/3) is not representable.
    by_contract_time = {float(value): value for value in (
        *(source_time(s.source.start) for s in spans), *(source_time(s.source.end) for s in spans),
        *(a for a, _, _ in merged), *(b for _, b, _ in merged), session.start, session.end)}
    cuts = [by_contract_time[t] for t in sorted(by_contract_time)]
    result: list[SourceVisualSpan] = []
    for start, end in zip(cuts, cuts[1:]):
        unavailable = next((reason for a, b, reason in merged if a <= start < end <= b), None)
        previous = next((s for s in spans if s.source.start <= float(start) < float(end) <= s.source.end), None)
        if unavailable is not None or previous is None:
            span = SourceVisualSpan(span_id=f"observed_span_{len(result):06d}", source=TimeRange(start=float(start), end=float(end)),
                state=SourceState.UNKNOWN, confidence=0, streamer_track_id=None, identity_status=IdentityStatus.UNAVAILABLE,
                webcam_track_id=None, important_regions=[], text_regions=[], evidence_ids=[],
                change_reasons=["observed_unavailable:" + (unavailable or "no_observed_samples")])
        else:
            span = SourceVisualSpan.model_validate({**previous.model_dump(), "span_id": f"observed_span_{len(result):06d}",
                                                    "source": {"start": float(start), "end": float(end)}})
        result.append(span)
    return result


def _create_observed_runtime(
    pinned: RenderRuntime, origin: SourceTimeOrigin, generation: Path, *,
    start: float, end: float, model_hashes: dict[str, str], allow_model_download: bool,
    session: _ObservedSession | None = None,
) -> _BuilderRuntime:
    session = session or _ObservedSession(pinned, origin, generation, start, end)
    qualify_render_runtime(pinned, decode_requirements())
    backend = _observed_face_backend(allow_model_download=allow_model_download)
    # Hash again after backend creation, so loading cannot silently bind other bytes.
    actual_hashes = _current_model_hashes()
    if not actual_hashes or (model_hashes and actual_hashes != model_hashes):
        raise ValueError("Face model identity changed or cached manifest is unavailable")
    return _BuilderRuntime(float(origin.duration), session,
        OpenCVLocalVisualAnalyzer(), backend, actual_hashes, None, start, end)


def _build_observed_generation(
    video_path: Path, context: _VisualArtifactContext, *, pinned: RenderRuntime, origin: SourceTimeOrigin,
    generation: Path, scope: dict[str, str], config: dict[str, Any], allow_model_download: bool,
    boundaries: tuple[float, ...] = (), dense_interval: float | None = None,
) -> VisualTimelineBuildResult:
    start, end = float(exact(scope["start"])), float(exact(scope["end"]))
    session = _ObservedSession(pinned, origin, generation, exact(scope["start"]), exact(scope["end"]))
    interval = config["sampling"].get("sparse_interval_seconds", config["sampling"].get("inspection_interval_seconds"))
    initial_count_bound = min(session._video_capacity(), math.ceil(2 / interval) + 2,
                              math.ceil((end - start) / interval) + 2)
    session._reserve_files(1 + initial_count_bound)
    session._check_work_budget(session._video_byte_bound(initial_count_bound))
    runtime = _create_observed_runtime(pinned, origin, generation, start=start, end=end,
        model_hashes=config["model_hashes"], allow_model_download=allow_model_download, session=session)
    if allow_model_download and not config["model_hashes"]:
        config["model_hashes"] = dict(runtime.model_hashes)
    assert runtime.decoder is session
    try:
        records: dict[float, _DecodedFrame] = {}
        sparse: list[float | Fraction] = list(_sample_times(start, end, interval))
        video = next(s for s in origin.streams if s.kind == "video")
        if video.start is not None:
            first = video.start - origin.origin
            if source_time(start) <= first < source_time(end):
                sparse = sorted(set(sparse) | {first})
        _decode_new_frames(runtime, sparse, records)
        if dense_interval is not None and records:
            sparse_samples, _, _, _, _ = _analyze_records(records, runtime=runtime, job=context,
                                                          require_webcam_persistence=False)
            windows = _dense_windows(_dense_trigger_times(sparse_samples, boundaries), duration_seconds=float(origin.duration),
                                    change_brackets=_dense_change_brackets(sparse_samples))
            points = _timestamps_for_windows(windows, interval=dense_interval, existing_timestamps=set(records))
            _decode_new_frames(runtime, [t for t in points if start <= t < end], records)
        if records:
            samples, face_tracks, webcam_tracks, camera_ids, identity = _analyze_records(records, runtime=runtime, job=context)
            samples = session.attach_audio(samples)
            webcams = [track.to_contract() for track in webcam_tracks]
            spans = _apply_identity_and_duration(segment_visual_states(samples), samples=samples,
                duration=float(origin.duration), webcam_face_ids_by_time=camera_ids, identity_status=identity.status,
                streamer_track_id=identity.track_id, webcam_tracks=webcams, window_start=start, window_end=end)
            faces = [track.to_summary() for track in face_tracks]
        else:
            samples, spans, faces, webcams = [], [], [], []
        spans = _mask_unavailable_spans(spans, session, no_samples=not samples)
        if _observed_config(context, config["sampling"], runtime.model_hashes) != config:
            raise ValueError("Analyzer legacy context/models changed during generation")
        timeline = SourceVisualTimeline(source_fingerprint=origin.fingerprint, duration_seconds=float(origin.duration),
            analyzer_version=OBSERVED_ANALYZER_VERSION, model_hashes=runtime.model_hashes, sampling_config=config["sampling"],
            samples=samples, spans=spans, face_tracks=faces, webcam_tracks=webcams)
        destination = generation / "source_visual_timeline.json"
        session.write_metadata(destination, timeline.model_dump(mode="json"))
        clock = {"version": 1, "kind": "observed_visual_evidence", "generation_id": generation.name,
            "origin": origin.record(), "origin_id": origin.identity, "scope": scope, "config": config,
            "observations": session.observations, "sample_associations": sorted(session.associations, key=lambda row: exact(row["requested"])),
            "audio_associations": session.audio_associations, "unavailable_attempts": session.unavailable,
            "unavailable_intervals": [{"start": rational(a), "end": rational(b), "reason": reason} for a, b, reason in session.intervals]}
        session.write_metadata(generation / "clock-receipt.json", clock)
        marker = {"version": 1, "kind": "observed_visual_generation", "generation_id": generation.name,
            "source_fingerprint": origin.fingerprint, "origin_id": origin.identity, "scope": scope, "config": config,
            "analyzer_version": OBSERVED_ANALYZER_VERSION, "timeline_fingerprint": visual_timeline_fingerprint(timeline),
            "timeline_sha256": file_sha256(destination), "clock_receipt_sha256": file_sha256(generation / "clock-receipt.json"),
            "catalog": generation_catalog(generation), "creation_work": {"indexed_samples": len(samples)}}
        marker["sha256"] = digest(canonical(marker))
        validate_observed_generation(destination, source_video=video_path, current_origin=origin, completion=marker)
        # Content commitment is the last generation write. No replacement/reuse.
        session.write_metadata(generation / "complete.json", marker)
        return VisualTimelineBuildResult(destination, False, len(samples))
    except Exception as exc:
        if (generation / "complete.json").is_file():
            committed = read_generation_json(generation / "complete.json")
            raise ObservedPublicationError("COMPLETE_UNSELECTED", generation / "source_visual_timeline.json",
                                           committed["sha256"]) from exc
        raise
    finally:
        _close_runtime(runtime)


def _select_observed_generation(job: JobStore, path: Path, marker: dict[str, Any]) -> None:
    _write_observed_json(job.source_visual_observed_latest_path,
        {"version": 1, "generation_id": path.parent.name, "complete_sha256": marker["sha256"]})


def _reconcile_observed_generation(job: JobStore, path: Path) -> None:
    job.mark_complete(f"source_visual_observed:{path.parent.name}", artifacts=[path, path.parent / "complete.json"])


def _selection_failure_phase(job: JobStore, path: Path, marker: dict[str, Any]) -> str:
    try:
        latest = read_generation_json(job.source_visual_observed_latest_path)
    except (OSError, ValueError):
        return "COMPLETE_UNSELECTED"
    return ("COMMITTED_BOOKKEEPING_FAILED" if latest == {"version": 1, "generation_id": path.parent.name,
            "complete_sha256": marker["sha256"]} else "COMPLETE_UNSELECTED")


def build_source_visual_timeline_result(
    video_path: Path, *, editorial_boundaries: Sequence[float] = (), sparse_interval: float = 10.0,
    dense_interval: float = 0.5, allow_model_download: bool = False, force: bool = False,
) -> VisualTimelineBuildResult:
    video_path = Path(video_path).resolve()
    if not all(math.isfinite(v) and v > 0 for v in (sparse_interval, dense_interval)):
        raise ValueError("Sampling intervals must be finite and positive")
    boundaries = _normalized_boundaries(editorial_boundaries)
    job = JobStore.for_video(video_path)
    with job.stage_lock("source_visual_timeline"):
        pinned, origin = _resolve_observed_source(video_path)
        if origin.fingerprint != job.fingerprint or source_fingerprint(video_path) != origin.fingerprint:
            raise ValueError("Current source changed across job/origin resolution")
        config = _observed_config(job, _sampling_config(sparse_interval=sparse_interval,
            dense_interval=dense_interval, editorial_boundaries=boundaries), _current_model_hashes())
        scope = {"kind": "full", "start": "0/1", "end": rational(origin.duration)}
        if not force and job.source_visual_observed_latest_path.exists():
            latest = read_generation_json(job.source_visual_observed_latest_path)
            if (set(latest) != {"version", "generation_id", "complete_sha256"} or type(latest["version"]) is not int
                    or latest["version"] != 1 or type(latest["generation_id"]) is not str
                    or len(latest["generation_id"]) != 32 or any(c not in "0123456789abcdef" for c in latest["generation_id"])):
                raise ValueError("Invalid observed latest pointer")
            destination = generation_file(job.root / "visual_observed" / "generations",
                                          latest["generation_id"] + "/source_visual_timeline.json")
            provenance = load_visual_provenance(destination, source_video=video_path, current_origin=origin)
            marker = read_generation_json(destination.parent / "complete.json")
            if (provenance.to_record()["complete_sha256"] != latest["complete_sha256"]
                    or marker["scope"] != scope or marker["config"] != config):
                raise ValueError("Observed latest scope/config/marker conflict; explicit force required")
            return VisualTimelineBuildResult(destination, True, 0)
        # Reject schedule size before model construction or an allocation-heavy arange.
        if math.ceil(float(origin.duration) / sparse_interval) + 1 > MAX_SCHEDULE_POINTS:
            raise ValueError("Scheduled sample point budget exceeded")
        generation = job.new_observed_generation()
        stage = f"source_visual_observed:{generation.name}"
        job.mark_started(stage)
        try:
            result = _build_observed_generation(video_path, job, pinned=pinned, origin=origin, generation=generation,
                scope=scope, config=config, allow_model_download=allow_model_download,
                boundaries=boundaries, dense_interval=dense_interval)
        except Exception as exc:
            if (generation / "complete.json").is_file():
                marker = read_generation_json(generation / "complete.json")
                raise ObservedPublicationError("COMPLETE_UNSELECTED", generation / "source_visual_timeline.json", marker["sha256"]) from exc
            job.mark_failed(stage, str(exc))
            raise
        marker = read_generation_json(generation / "complete.json")
        try:
            _select_observed_generation(job, result.artifact_path, marker)
        except Exception as exc:
            raise ObservedPublicationError(_selection_failure_phase(job, result.artifact_path, marker),
                                           result.artifact_path, marker["sha256"]) from exc
        try:
            _reconcile_observed_generation(job, result.artifact_path)
        except Exception as exc:
            raise ObservedPublicationError("COMMITTED_BOOKKEEPING_FAILED", result.artifact_path, marker["sha256"]) from exc
        return result


def recover_observed_generation(video_path: Path, committed_path: Path, *, select: bool = True) -> Path:
    """Explicit known-path recovery; never discover/guess the newest directory."""
    video_path, committed_path = Path(video_path).resolve(), Path(committed_path).resolve()
    job = JobStore.for_video(video_path)
    with job.stage_lock("source_visual_timeline"):
        expected_parent = (job.root / "visual_observed" / "generations").resolve()
        if committed_path.name != "source_visual_timeline.json" or committed_path.parent.parent != expected_parent:
            raise ValueError("Recovery requires an explicit committed path in this source namespace")
        _, origin = _resolve_observed_source(video_path)
        load_visual_provenance(committed_path, source_video=video_path, current_origin=origin)
        marker = read_generation_json(committed_path.parent / "complete.json")
        current_config = _observed_config(job, marker["config"]["sampling"], _current_model_hashes())
        if current_config != marker["config"] or marker["scope"]["kind"] != "full":
            raise ValueError("Committed generation is not currently applicable")
        if select:
            try:
                _select_observed_generation(job, committed_path, marker)
            except Exception as exc:
                raise ObservedPublicationError(_selection_failure_phase(job, committed_path, marker),
                                               committed_path, marker["sha256"]) from exc
            try:
                _reconcile_observed_generation(job, committed_path)
            except Exception as exc:
                raise ObservedPublicationError("COMMITTED_BOOKKEEPING_FAILED", committed_path, marker["sha256"]) from exc
        return committed_path


def _inspection_filename_token(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _validated_inspection_duration(
    video_path: Path,
    *,
    start: float,
    end: float,
    sample_interval: float,
) -> float:
    values = (float(start), float(end), float(sample_interval))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("inspection range and interval must be finite")
    if start < 0 or end <= start:
        raise ValueError("inspection requires 0 <= start < end")
    if end - start > _MAX_INSPECTION_SECONDS:
        raise ValueError("visual inspection is limited to 300 seconds")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")

    metadata = probe_media(video_path)
    duration = float(metadata.get("duration_seconds") or 0.0)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("ffprobe did not return a valid source duration")
    if start >= duration or end > duration:
        raise ValueError(
            f"requested range {start:.1f}-{end:.1f} exceeds source duration "
            f"{duration:.1f}"
        )
    return duration


def build_source_visual_timeline_window(
    video_path: Path,
    *,
    start: float,
    end: float,
    sample_interval: float = 0.5,
    allow_model_download: bool = False,
    output_dir: Path | None = None,
) -> Path:
    """Decode and analyze one immutable source-time window without JobStore mutation."""
    video_path = Path(video_path).resolve()
    if (not all(math.isfinite(v) for v in (start, end, sample_interval))
            or not 0 <= start < end or end - start > _MAX_INSPECTION_SECONDS or sample_interval <= 0):
        raise ValueError("inspection requires finite 0 <= start < end, <=300 seconds and positive interval")
    pinned, origin = _resolve_observed_source(video_path)
    if source_time(end) > origin.duration:
        raise ValueError(f"requested range exceeds source duration {float(origin.duration):.1f}")
    if math.ceil((end - start) / sample_interval) + 1 > MAX_SCHEDULE_POINTS:
        raise ValueError("Scheduled sample point budget exceeded")
    root = source_job_root(video_path)
    context = _ReadOnlyVisualContext(origin.fingerprint, root / "transcript.json",
                                    root / "diarization.json", root / "streamer_identity.json")
    destination_dir = Path(output_dir) if output_dir is not None else root / "visual_inspections"
    generation = destination_dir / "visual_observed" / uuid4().hex
    generation.mkdir(parents=True, exist_ok=False)
    sampling = {"inspection_start_seconds": float(start), "inspection_end_seconds": float(end),
                "inspection_interval_seconds": float(sample_interval)}
    config = _observed_config(context, sampling, _current_model_hashes())
    return _build_observed_generation(video_path, context, pinned=pinned, origin=origin,
        generation=generation, scope={"kind": "window", "start": rational(source_time(start)),
                                     "end": rational(source_time(end))},
        config=config, allow_model_download=allow_model_download).artifact_path
