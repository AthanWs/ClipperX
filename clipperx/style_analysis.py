"""Deterministic local style evidence. No providers, OCR, identity or narrative inference.

Frames use RGB uint8 HxWx3 arrays. All positional overlay roles, panel rectangles,
blur and motion detections are candidates, not semantic claims. JPEG paths are
portable relative references resolved against the caller's style_library_root.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Literal, TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from clipperx.jobs import _InterProcessFileLock
from clipperx.local_paths import atomic_write_text, paths, source_fingerprint
from clipperx.media import probe_media
from clipperx.style_contracts import BackgroundTreatment, StyleKeyframe
from clipperx.visual_contracts import NormalizedRect

ANALYZER_VERSION = "objective-style-1.2"
MAX_KEYFRAMES = 8
MAX_TEMPORAL_SAMPLES = 180
MAX_CLIP_DURATION = 600.0
MAX_DECODED_FRAMES = 360_000
MAX_DECODE_SECONDS = 180.0
RGBFrame: TypeAlias = NDArray[np.uint8]


class _EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RegionEvidence(_EvidenceModel):
    kind: Literal["black", "solid", "blur_candidate", "panel_candidate"]
    rect: NormalizedRect
    persistence: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class OverlayEvidence(_EvidenceModel):
    role: Literal["title_candidate", "subtitle_candidate", "brand_candidate"]
    rect: NormalizedRect
    persistence: float = Field(ge=0, le=1)
    observed_timestamps: list[float]
    palette: list[str]
    glyph_height_pixels_estimate: float = Field(ge=0)

    @field_validator("observed_timestamps")
    @classmethod
    def validate_observations(cls, values: list[float]) -> list[float]:
        if not values or any(not math.isfinite(t) or t < 0 for t in values):
            raise ValueError("overlay observations must be finite and nonnegative")
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError("overlay observations must be strictly increasing")
        return values


class MotionEvidence(_EvidenceModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    kind: Literal["pan_candidate", "zoom_candidate"]
    scale_estimate: float = Field(gt=0)
    translation_fraction: float = Field(ge=0)
    inlier_fraction: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_interval(self) -> MotionEvidence:
        if self.end <= self.start:
            raise ValueError("motion end must be after start")
        return self


class StyleFrameAnalysis(_EvidenceModel):
    """Measured frame evidence plus explicitly named visual candidates.

    visual_quality is a heuristic technical score, never narrative approval.
    camera_regions remains empty without identity/camera evidence. Panel/content
    rectangles denote stable boundaries only; role cannot be established locally.
    """

    background_treatment: BackgroundTreatment
    background_regions: list[RegionEvidence]
    panel_candidates: list[RegionEvidence]
    content_regions: list[NormalizedRect]
    camera_regions: list[NormalizedRect] = Field(default_factory=list)
    title_regions: list[NormalizedRect]
    brand_regions: list[NormalizedRect]
    subtitle_regions: list[NormalizedRect]
    subtitle_palette: list[str]
    overlay_candidates: list[OverlayEvidence]
    motion_candidates: list[MotionEvidence]
    cut_rate_per_minute: float = Field(ge=0)
    zoom_count: int = Field(ge=0)
    pan_count: int = Field(ge=0)
    visual_quality: float = Field(ge=0, le=1)
    narrative_quality: float | None = Field(default=None, ge=0, le=1)
    quality_metrics: dict[str, float | int | None]
    limitations: list[str]


class StyleEvidence(StyleFrameAnalysis):
    """JSON-ready on-disk evidence; keyframes/evidence_path are library relative.

    clip_fingerprint uses host source identity (resolved path, size and mtime),
    matching local_paths.source_fingerprint; it is not a content hash.
    """

    analyzer_version: str
    clip_fingerprint: str
    duration_seconds: float = Field(gt=0)
    media_properties: dict[str, Any]
    sampled_timestamps: list[float]
    scene_change_timestamps: list[float]
    keyframes: list[StyleKeyframe] = Field(min_length=1, max_length=8)
    keyframe_sha256: dict[str, str]
    evidence_path: str

    @model_validator(mode="after")
    def validate_source_times(self) -> StyleEvidence:
        groups = [self.sampled_timestamps, self.scene_change_timestamps,
                  [frame.timestamp for frame in self.keyframes]]
        if not self.sampled_timestamps:
            raise ValueError("evidence must include sampled timestamps")
        groups.extend(item.observed_timestamps for item in self.overlay_candidates)
        groups.extend([item.start, item.end] for item in self.motion_candidates)
        for values in groups:
            if any(not math.isfinite(t) or not 0 <= t <= self.duration_seconds for t in values):
                raise ValueError("evidence timestamps must be finite and within source duration")
            if any(b <= a for a, b in zip(values, values[1:])):
                raise ValueError("evidence timestamps must be strictly increasing")
        sampled = set(self.sampled_timestamps)
        if any(t not in sampled for item in self.overlay_candidates for t in item.observed_timestamps):
            raise ValueError("overlay observations must reference sampled frames")
        if any(item.start not in sampled or item.end not in sampled for item in self.motion_candidates):
            raise ValueError("motion intervals must reference sampled frames")
        return self


def _rect(x: int, y: int, width: int, height: int, shape: tuple[int, ...]) -> NormalizedRect:
    h, w = shape[:2]
    return NormalizedRect(x=x / w, y=y / h, width=width / w, height=height / h)


def _small(frame: RGBFrame, bound: int = 480) -> RGBFrame:
    h, w = frame.shape[:2]
    ratio = min(1.0, bound / max(h, w))
    return cast(RGBFrame, cv2.resize(frame, (max(1, round(w * ratio)), max(1, round(h * ratio)))))


def _runs(mask: NDArray[np.bool_], minimum: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    return [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
        if b - a >= minimum
    ]


def _background(frame: RGBFrame) -> list[RegionEvidence]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    result: list[RegionEvidence] = []
    # Row/column uniformity detects full bars rather than treating low Laplacian
    # energy alone as blur. Inspect border-connected strips only.
    for axis in (0, 1):
        # axis 1 => horizontal bars; axis 0 => vertical bars.
        deviation = frame.astype(np.float32).std(axis=axis).mean(axis=1)
        extent = h if axis == 1 else w
        for start, end in _runs(deviation < 0.75, max(2, int(extent * 0.025))):
            if start > extent * 0.05 and end < extent * 0.95:
                continue
            region = frame[start:end] if axis == 1 else frame[:, start:end]
            kind: Literal["black", "solid"] = "black" if np.percentile(region, 95) < 18 else "solid"
            rect = (
                _rect(0, start, w, end - start, frame.shape)
                if axis == 1
                else _rect(start, 0, end - start, h, frame.shape)
            )
            result.append(RegionEvidence(kind=kind, rect=rect, persistence=1, confidence=0.95))
    if result:
        return result
    # Non-uniform border strips with much less high-frequency energy than the
    # center are blur candidates. A naturally soft scene can also satisfy this.
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    center = lap[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    center_energy = float(center.var())
    for start, end in ((0, h // 5), (4 * h // 5, h)):
        gray_region = gray[start:end]
        if gray_region.size and gray_region.std() > 0.75 and center_energy > 80:
            ratio = float(lap[start:end].var()) / center_energy
            if ratio < 0.12:
                result.append(
                    RegionEvidence(
                        kind="blur_candidate",
                        rect=_rect(0, start, w, end - start, frame.shape),
                        persistence=1,
                        confidence=round(1 - ratio, 4),
                    )
                )
    return result


def _overlay_regions(frame: RGBFrame) -> list[tuple[NormalizedRect, list[str], float]]:
    h, w = frame.shape[:2]
    rgb = frame.astype(np.int16)
    # Bright neutral and saturated pixels are potential text/logo strokes.
    bright = (rgb.min(axis=2) > 205) | (
        (rgb.max(axis=2) > 180) & (rgb.max(axis=2) - rgb.min(axis=2) > 95)
    )
    mask = bright.astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    strokes = np.zeros_like(mask)
    heights: list[int] = []
    for index in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[index])
        if 2 <= bh <= h * 0.12 and 1 <= bw <= w * 0.3 and 4 <= area <= w * h * 0.035:
            if area / (bw * bh) >= 0.12:
                component = labels[y : y + bh, x : x + bw] == index
                pixels = frame[y : y + bh, x : x + bw][component]
                _, color_counts = np.unique(pixels // 16, axis=0, return_counts=True)
                if color_counts.max() / len(pixels) >= 0.25:
                    strokes[y : y + bh, x : x + bw][component] = 255
                    heights.append(bh)
    merged = cv2.dilate(strokes, np.ones((3, max(3, w // 30)), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    result = []
    for index in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[index])
        if bw < w * 0.07 or bh > h * 0.15 or area < 20:
            continue
        pixels = frame[y : y + bh, x : x + bw][strokes[y : y + bh, x : x + bw] > 0]
        if pixels.size == 0:
            continue
        colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
        order = np.argsort(counts)[::-1][:6]
        palette = [
            "#" + "".join(f"{int(v):02X}" for v in colors[i])
            for i in order
            if counts[i] >= max(2, len(pixels) * 0.01)
        ]
        if not palette:
            continue
        result.append(
            (
                _rect(x, y, bw, bh, frame.shape),
                palette,
                float(np.median(heights)) if heights else 0.0,
            )
        )
    return result


def _overlap(a: NormalizedRect, b: NormalizedRect) -> float:
    intersection = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x)) * max(
        0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    )
    return intersection / max(1e-9, min(a.width * a.height, b.width * b.height))


def _candidate_palette(colors: Sequence[str], limit: int) -> list[str]:
    """Keep bright overlay colors when temporal candidates exceed the bound."""

    def priority(color: str) -> tuple[int, int, int]:
        rgb = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
        value, saturation = max(rgb), max(rgb) - min(rgb)
        tier = 2 if value >= 235 and saturation >= 60 else 1 if value >= 220 else 0
        return tier, value, saturation

    return sorted(dict.fromkeys(colors), key=priority, reverse=True)[:limit]


def _stable_panels(frames: Sequence[RGBFrame]) -> list[RegionEvidence]:
    """Stable textured horizontal blocks and their column boundaries, no roles."""
    h, w = frames[0].shape[:2]
    texture = np.stack([f.astype(np.float32).std(axis=1).mean(axis=1) for f in frames])
    row_activity = np.median(texture, axis=0) > 12
    row_activity = (
        cv2.morphologyEx(
            row_activity.astype(np.uint8)[:, None],
            cv2.MORPH_CLOSE,
            np.ones((max(1, h // 100), 1), np.uint8),
        )
        .ravel()
        .astype(bool)
    )
    panels = []
    for start, end in _runs(row_activity, max(8, h // 10)):
        col_texture = np.stack(
            [f[start:end].astype(np.float32).std(axis=0).mean(axis=1) for f in frames]
        )
        columns = np.flatnonzero(np.median(col_texture, axis=0) > 6)
        if len(columns) < w * 0.15:
            continue
        left, right = int(columns[0]), int(columns[-1]) + 1
        persistence = float((texture[:, start:end].mean(axis=1) > 12).mean())
        panels.append(
            RegionEvidence(
                kind="panel_candidate",
                rect=_rect(left, start, right - left, end - start, frames[0].shape),
                persistence=persistence,
                confidence=0.6,
            )
        )
    return panels


def _motion(frames: Sequence[RGBFrame], timestamps: Sequence[float]) -> list[MotionEvidence]:
    detector = cv2.ORB.create(nfeatures=350)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    output: list[MotionEvidence] = []
    for previous, current, start, end in zip(frames, frames[1:], timestamps, timestamps[1:]):
        a = cv2.cvtColor(_small(previous, 256), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(_small(current, 256), cv2.COLOR_RGB2GRAY)
        ka, da = detector.detectAndCompute(a, None)
        kb, db = detector.detectAndCompute(b, None)
        if da is None or db is None or len(ka) < 12 or len(kb) < 12:
            continue
        pairs = matcher.knnMatch(da, db, k=2)
        matches = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
        ]
        if len(matches) < 12:
            continue
        src = np.asarray([ka[m.queryIdx].pt for m in matches], dtype=np.float32)
        dst = np.asarray([kb[m.trainIdx].pt for m in matches], dtype=np.float32)
        affine, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0
        )
        if affine is None or inliers is None:
            continue
        fraction = float(inliers.mean())
        scale = float(math.hypot(affine[0, 0], affine[0, 1]))
        translation = float(math.hypot(affine[0, 2] / a.shape[1], affine[1, 2] / a.shape[0]))
        if fraction < 0.65 or not 0.75 < scale < 1.3:
            continue
        detections: tuple[tuple[Literal["zoom_candidate", "pan_candidate"], bool], ...] = (
            ("zoom_candidate", abs(scale - 1) > 0.025),
            ("pan_candidate", translation > 0.015 and abs(scale - 1) < 0.025),
        )
        for kind, detected in detections:
            if detected:
                output.append(
                    MotionEvidence(
                        start=start,
                        end=end,
                        kind=kind,
                        scale_estimate=scale,
                        translation_fraction=translation,
                        inlier_fraction=fraction,
                    )
                )
    return output


def analyze_style_frames(
    frames: Sequence[RGBFrame],
    *,
    timestamps: Sequence[float] | None = None,
    duration_seconds: float | None = None,
) -> StyleFrameAnalysis:
    """Analyze RGB uint8 frames without I/O; timestamps are sorted seconds.

    Same frame dimensions are required. Omitted timestamps use frame indices;
    omitted duration spans those indices (minimum one second). Measurements are
    sample based and cannot establish OCR text, readability or face identity.
    """
    if not frames:
        raise ValueError("at least one RGB frame is required")
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frames must be RGB uint8 HxWx3 arrays")
        if min(frame.shape[:2]) < 8 or frame.shape != frames[0].shape:
            raise ValueError("frames must share nonempty dimensions of at least 8 pixels")
    times = list(timestamps) if timestamps is not None else [float(i) for i in range(len(frames))]
    if (
        len(times) != len(frames)
        or any(not math.isfinite(t) or t < 0 for t in times)
        or any(b <= a for a, b in zip(times, times[1:]))
    ):
        raise ValueError("timestamps must match frames and be finite, nonnegative and increasing")
    duration = duration_seconds if duration_seconds is not None else max(1.0, times[-1])
    if not math.isfinite(duration) or duration <= 0 or times[-1] > duration:
        raise ValueError("duration must be finite, positive and include all timestamps")
    reduced = [_small(frame) for frame in frames]
    background_groups: list[tuple[RegionEvidence, set[int]]] = []
    overlays: list[tuple[OverlayEvidence, set[int]]] = []
    frame_treatments = []
    black_count = 0
    clipped_pixels = []
    for index, frame in enumerate(reduced):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        black_count += int(float((gray < 18).mean()) > 0.98)
        clipped_pixels.append(float(((gray <= 2) | (gray >= 253)).mean()))
        regions = _background(frame)
        kinds = {region.kind for region in regions}
        treatment = (
            BackgroundTreatment.BLACK
            if "black" in kinds
            else BackgroundTreatment.SOLID
            if "solid" in kinds
            else BackgroundTreatment.CONTAIN_BLUR
            if "blur_candidate" in kinds
            else BackgroundTreatment.COVER
        )
        frame_treatments.append(treatment)
        for region in regions:
            group = next(
                (
                    g
                    for g in background_groups
                    if g[0].kind == region.kind and _overlap(g[0].rect, region.rect) > 0.7
                ),
                None,
            )
            if group is None:
                background_groups.append((region, {index}))
            else:
                group[1].add(index)
        for rect, palette, height in _overlay_regions(frame):
            role: Literal["title_candidate", "brand_candidate", "subtitle_candidate"] = (
                "title_candidate"
                if rect.y < 0.25
                else "brand_candidate"
                if rect.y > 0.82
                else "subtitle_candidate"
            )
            overlay_group = next(
                (g for g in overlays if g[0].role == role and _overlap(g[0].rect, rect) > 0.5), None
            )
            if overlay_group is None:
                overlays.append(
                    (
                        OverlayEvidence(
                            role=role,
                            rect=rect,
                            palette=palette,
                            persistence=1 / len(frames),
                            observed_timestamps=[times[index]],
                            glyph_height_pixels_estimate=height
                            * frames[0].shape[0]
                            / frame.shape[0],
                        ),
                        {index},
                    )
                )
            else:
                if index not in overlay_group[1]:
                    overlay_group[0].observed_timestamps.append(times[index])
                overlay_group[1].add(index)
                overlay_group[0].palette = _candidate_palette(overlay_group[0].palette + palette, 12)
    backgrounds = [
        item.model_copy(update={"persistence": len(indices) / len(frames)})
        for item, indices in background_groups
    ]
    overlay_evidence = [
        item.model_copy(update={"persistence": len(indices) / len(frames)})
        for item, indices in overlays
    ]
    # Strip boundaries give a stable geometric content candidate, not a camera.
    persistent = [item for item in backgrounds if item.persistence >= 0.6]
    horizontal = sorted(
        (item.rect for item in persistent if item.rect.width > 0.9), key=lambda r: r.y
    )
    top = max((r.y + r.height for r in horizontal if r.y < 0.05), default=0.0)
    bottom = min((r.y for r in horizontal if r.y + r.height > 0.95), default=1.0)
    panels = _stable_panels(reduced)
    if not panels and bottom - top > 0.05 and (top > 0 or bottom < 1):
        panels = [
            RegionEvidence(
                kind="panel_candidate",
                rect=NormalizedRect(x=0, y=top, width=1, height=bottom - top),
                persistence=min((r.persistence for r in persistent), default=1.0),
                confidence=0.6,
            )
        ]
    treatment = (
        frame_treatments[0] if len(set(frame_treatments)) == 1 else BackgroundTreatment.MIXED
    )
    motions = _motion(reduced, times)
    edge_overlays = sum(
        item.rect.x < 0.005
        or item.rect.y < 0.005
        or item.rect.x + item.rect.width > 0.995
        or item.rect.y + item.rect.height > 0.995
        for item in overlay_evidence
    )
    tiny = sum(item.glyph_height_pixels_estimate < 8 for item in overlay_evidence)
    consistency = max(frame_treatments.count(t) for t in set(frame_treatments)) / len(frames)
    black_fraction = black_count / len(frames)
    score = max(
        0.0,
        1
        - 0.6 * black_fraction
        - 0.15 * (1 - consistency)
        - 0.1 * min(1, tiny / max(1, len(overlay_evidence)))
        - 0.15 * min(1, edge_overlays / max(1, len(overlay_evidence))),
    )
    # Frame-only discontinuities are coarse candidates. Clip analysis replaces
    # this rate with actual decoder scene-change timestamps.
    changes = sum(
        float(np.mean(np.abs(a.astype(float) - b.astype(float)))) > 55
        for a, b in zip(reduced, reduced[1:])
    )
    return StyleFrameAnalysis(
        background_treatment=treatment,
        background_regions=backgrounds,
        panel_candidates=panels,
        content_regions=[p.rect for p in panels],
        title_regions=[o.rect for o in overlay_evidence if o.role == "title_candidate"],
        brand_regions=[o.rect for o in overlay_evidence if o.role == "brand_candidate"],
        subtitle_regions=[o.rect for o in overlay_evidence if o.role == "subtitle_candidate"],
        subtitle_palette=_candidate_palette(
            [c for o in overlay_evidence if o.role == "subtitle_candidate" for c in o.palette], 24
        ),
        overlay_candidates=overlay_evidence,
        motion_candidates=motions,
        cut_rate_per_minute=changes * 60 / duration,
        zoom_count=sum(m.kind == "zoom_candidate" for m in motions),
        pan_count=sum(m.kind == "pan_candidate" for m in motions),
        visual_quality=round(score, 4),
        quality_metrics={
            "black_frame_fraction": black_fraction,
            "luminance_extreme_pixel_fraction": float(np.mean(clipped_pixels)),
            "edge_overlay_candidates": edge_overlays,
            "tiny_overlay_candidates": tiny,
            "background_consistency": consistency,
            "severe_face_cuts": None,
            "unreadable_overlays": None,
        },
        limitations=[
            "Overlay roles are positional text-like candidates; no OCR or semantic classification.",
            "Blur candidates may be naturally soft regions; no original source comparison.",
            "Camera identity and severe face cuts are unmeasured; camera_regions remains empty.",
            "Tiny/edge overlay candidates do not establish unreadability or clipping of actual text.",
            "Motion counts are sampled affine intervals, not unique edit events; in-content motion can match.",
            "Technical quality is a sampled heuristic, not aesthetic or narrative approval.",
        ],
    )


def _scene_changes(path: Path) -> list[float]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-an",
            "-vf",
            "scale=160:-2,select='gt(scene,0.30)',showinfo",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    return sorted(set(float(value) for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)))


def _decode_samples(
    capture: Any, requested: Sequence[float], representatives: Sequence[float], duration: float,
) -> tuple[dict[float, RGBFrame], dict[float, RGBFrame]]:
    """Decode to EOF using observed presentation times, with bounded retained images.

    Keep the first frame at or after each request, plus the actual first/final
    frames. No timestamp seek or frame-count/FPS endpoint inference is used.
    """
    sampled: dict[float, RGBFrame] = {}
    keyframes: dict[float, RGBFrame] = {}
    sample_index = keyframe_index = 0
    previous = -1.0
    last: RGBFrame | None = None
    deadline = time.monotonic() + MAX_DECODE_SECONDS
    for _ in range(MAX_DECODED_FRAMES + 1):
        if time.monotonic() >= deadline:
            raise RuntimeError("video decode exceeded style analysis time budget")
        ok, bgr = capture.read()
        if not ok:
            break
        if _ == MAX_DECODED_FRAMES:
            raise RuntimeError("video decode exceeded style analysis frame budget")
        measured = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        if (
            not math.isfinite(measured) or measured < 0 or measured <= previous
            or measured > duration + 1e-6
        ):
            raise RuntimeError("video backend returned invalid/non-increasing presentation timestamp")
        timestamp = min(duration, round(measured, 6))
        last = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if not sampled or (sample_index < len(requested) and measured >= requested[sample_index]):
            sampled[timestamp] = _small(last)
        if not keyframes or (
            keyframe_index < len(representatives) and measured >= representatives[keyframe_index]
        ):
            keyframes[timestamp] = _small(last, 960)
        while sample_index < len(requested) and measured >= requested[sample_index]:
            sample_index += 1
        while keyframe_index < len(representatives) and measured >= representatives[keyframe_index]:
            keyframe_index += 1
        previous = measured
    if last is None:
        raise ValueError("input has no decodable video frames")
    final_timestamp = min(duration, round(previous, 6))
    sampled[final_timestamp] = _small(last)
    keyframes[final_timestamp] = _small(last, 960)
    return sampled, keyframes


def _atomic_write_jpeg(target: Path, blob: bytes) -> None:
    """Publish a complete JPEG with a unique sibling temporary and atomic replace."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cache(root: Path, fingerprint: str, evidence_path: str) -> StyleEvidence | None:
    try:
        result = StyleEvidence.model_validate_json(
            (root / evidence_path).read_text(encoding="utf-8")
        )
        if (
            result.analyzer_version != ANALYZER_VERSION
            or result.clip_fingerprint != fingerprint
            or result.evidence_path != evidence_path
            or result.narrative_quality is not None
            or set(result.keyframe_sha256) != {frame.path for frame in result.keyframes}
            or any(
                not math.isfinite(t) or t < 0 or t > result.duration_seconds
                for t in result.sampled_timestamps + result.scene_change_timestamps
            )
            or any(b <= a for a, b in zip(result.sampled_timestamps, result.sampled_timestamps[1:]))
        ):
            return None
        expected_prefix = f"evidence/{fingerprint}/frames/"
        for frame in result.keyframes:
            if (
                not frame.path.startswith(expected_prefix)
                or frame.timestamp > result.duration_seconds
            ):
                return None
            jpeg = root / frame.path
            if not jpeg.resolve().is_relative_to(root.resolve()):
                return None
            if hashlib.sha256(jpeg.read_bytes()).hexdigest() != result.keyframe_sha256[frame.path]:
                return None
            if cv2.imread(str(jpeg)) is None:
                return None
        return result
    except (OSError, ValueError, TypeError):
        return None


def analyze_edited_clip(
    path: Path,
    *,
    style_library_root: Path | None = None,
    narrative_quality: float | None = None,
    use_cache: bool = True,
) -> StyleEvidence:
    """Probe/fingerprint read-only media and persist local evidence plus 1..8 JPEGs.

    style_library_root defaults to workspace data/processed/style_library. References
    in the returned StyleEvidence resolve against that exact root. narrative_quality
    is an optional caller-owned [0,1] score; never inferred or persisted by analysis.
    use_cache=False forces rebuilding. Cache checks version, fingerprint, schema,
    timestamps and JPEG SHA-256/decodability. Clips over 600 seconds are rejected to
    keep this edited-reference operation bounded. Input MP4s are never copied.
    """
    if narrative_quality is not None and (
        not math.isfinite(narrative_quality) or not 0 <= narrative_quality <= 1
    ):
        raise ValueError("narrative_quality must be None or a finite score in [0,1]")
    source = Path(path).resolve(strict=True)
    fingerprint = source_fingerprint(source)
    root = Path(style_library_root or paths.data_dir / "processed" / "style_library").resolve()
    directory = root / "evidence" / fingerprint
    evidence_path = f"evidence/{fingerprint}/style_evidence.json"
    with _InterProcessFileLock(directory / ".analysis.lock"):
        if source_fingerprint(source) != fingerprint:
            raise RuntimeError("source changed while waiting for style analysis lock")
        cached = _cache(root, fingerprint, evidence_path) if use_cache else None
        if cached is not None:
            if source_fingerprint(source) != fingerprint:
                raise RuntimeError("source changed during style cache validation")
            return cached.model_copy(update={"narrative_quality": narrative_quality})
        properties = probe_media(source)
        duration = float(properties.get("duration_seconds") or 0)
        if not math.isfinite(duration) or not 0 < duration <= MAX_CLIP_DURATION:
            raise ValueError("edited clip duration must be between 0 and 600 seconds")
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise ValueError("input has no decodable video stream")
            scenes = [t for t in _scene_changes(source) if 0 < t <= duration]
            # Preserve endpoints and spread selected actual scene candidates across
            # the clip before filling remaining slots with even timestamps.
            selected = {0.0, duration}
            if len(scenes) > MAX_KEYFRAMES - len(selected):
                indices = np.linspace(0, len(scenes) - 1, MAX_KEYFRAMES - len(selected), dtype=int)
                selected.update(scenes[i] for i in indices)
            else:
                selected.update(scenes)
            for at in np.linspace(0, duration, MAX_KEYFRAMES):
                if len(selected) >= MAX_KEYFRAMES:
                    break
                selected.add(float(at))
            temporal_requested = sorted(
                set(
                    float(t)
                    for t in np.linspace(
                        0, duration, min(MAX_TEMPORAL_SAMPLES, max(2, math.ceil(duration * 2)))
                    )
                )
                | selected
            )
            sampled, decoded_keyframes = _decode_samples(
                capture, temporal_requested, sorted(selected), duration,
            )
            times = sorted(sampled)
            scenes = [t for t in scenes if t <= times[-1]]
            analysis = analyze_style_frames(
                [sampled[t] for t in times], timestamps=times, duration_seconds=duration
            )
            keyframes_by_time: dict[float, StyleKeyframe] = {}
            hashes: dict[str, str] = {}
            frames_dir = directory / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for measured, frame in sorted(decoded_keyframes.items()):
                ref = f"evidence/{fingerprint}/frames/{len(keyframes_by_time):02d}.jpg"
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(_small(frame, 960), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                blob = jpeg.tobytes()
                _atomic_write_jpeg(root / ref, blob)
                hashes[ref] = hashlib.sha256(blob).hexdigest()
                keyframes_by_time[measured] = StyleKeyframe(timestamp=measured, path=ref)
            if source_fingerprint(source) != fingerprint:
                raise RuntimeError("source changed during style analysis; evidence not published")
            evidence = StyleEvidence(
                **analysis.model_dump(exclude={"cut_rate_per_minute"}),
                cut_rate_per_minute=len(scenes) * 60 / duration,
                analyzer_version=ANALYZER_VERSION,
                clip_fingerprint=fingerprint,
                duration_seconds=duration,
                media_properties=properties,
                sampled_timestamps=times,
                scene_change_timestamps=scenes,
                keyframes=list(keyframes_by_time.values()),
                keyframe_sha256=hashes,
                evidence_path=evidence_path,
            )
            atomic_write_text(root / evidence_path, evidence.model_dump_json(indent=2))
            return evidence.model_copy(update={"narrative_quality": narrative_quality})
        finally:
            capture.release()
