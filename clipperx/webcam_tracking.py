"""Persistent webcam rectangles kept distinct from facial identity tracks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from ._vision_types import linear_sum_assignment

from .planner_contracts import TimeRange
from .visual_contracts import NormalizedRect, TemporalRectObservation, WebcamTrack


_MOTION_GATE = 0.45
_EPSILON = 1e-6


@dataclass(slots=True)
class WebcamRegionObservation:
    at: float
    rect: NormalizedRect
    confidence: float
    evidence_id: str
    face_track_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.at) or self.at < 0:
            raise ValueError("El timestamp de webcam debe ser finito y no negativo")
        if not 0.0 <= self.confidence <= 1.0 or not self.evidence_id:
            raise ValueError("La observación de webcam requiere confianza y evidencia válidas")


def _iou(left: NormalizedRect, right: NormalizedRect) -> float:
    x1, y1 = max(left.x, right.x), max(left.y, right.y)
    x2, y2 = min(left.x + left.width, right.x + right.width), min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0


def _center_motion(left: NormalizedRect, right: NormalizedRect) -> float:
    return min(1.0, float(np.hypot(left.x + left.width / 2 - right.x - right.width / 2, left.y + left.height / 2 - right.y - right.height / 2)))


@dataclass(slots=True)
class WebcamRectangleTrack:
    track_id: str
    observations: list[WebcamRegionObservation] = field(default_factory=list)
    face_track_ids: set[str] = field(default_factory=set)
    missed_samples: int = 0

    def __post_init__(self) -> None:
        if self.observations:
            self.face_track_ids.update(face_id for item in self.observations for face_id in item.face_track_ids)

    @property
    def last_observation(self) -> WebcamRegionObservation:
        return self.observations[-1]

    def add(self, observation: WebcamRegionObservation) -> None:
        self.observations.append(observation)
        self.face_track_ids.update(observation.face_track_ids)
        self.missed_samples = 0

    def to_contract(self) -> WebcamTrack:
        first = self.observations[0].at
        last = max(self.observations[-1].at, first + _EPSILON)
        return WebcamTrack(
            track_id=self.track_id,
            source_ranges=[TimeRange(start=first, end=last)],
            observations=[TemporalRectObservation(at=item.at, rect=item.rect, confidence=item.confidence, evidence_id=item.evidence_id) for item in self.observations],
            face_track_ids=sorted(self.face_track_ids),
            confidence=float(np.mean([item.confidence for item in self.observations])),
            evidence_ids=list(dict.fromkeys(item.evidence_id for item in self.observations)),
        )


def _association_cost(track: WebcamRectangleTrack, observation: WebcamRegionObservation) -> tuple[float, bool]:
    shared_face = bool(track.face_track_ids.intersection(observation.face_track_ids))
    overlap = _iou(track.last_observation.rect, observation.rect)
    motion = _center_motion(track.last_observation.rect, observation.rect)
    if not shared_face and overlap <= 0.0 and motion > _MOTION_GATE:
        return 1_000_000.0, False
    return (0.0 if shared_face else 1.0 - overlap + motion), True


def associate_webcam_tracks(
    observations: Sequence[WebcamRegionObservation], *, sample_timestamps: Sequence[float] = (), max_missed_samples: int = 3
) -> list[WebcamRectangleTrack]:
    """Associate rectangles independently of whether a face is temporarily visible."""
    if max_missed_samples < 0:
        raise ValueError("max_missed_samples no puede ser negativo")
    grouped: dict[float, list[WebcamRegionObservation]] = {}
    for observation in observations:
        grouped.setdefault(float(observation.at), []).append(observation)
    tracks: list[WebcamRectangleTrack] = []
    next_id = 1
    for timestamp in sorted({*grouped, *(float(value) for value in sample_timestamps)}):
        current = grouped.get(timestamp, [])
        active = [track for track in tracks if track.missed_samples < max_missed_samples]
        matched_tracks: set[int] = set()
        matched_observations: set[int] = set()
        if active and current:
            costs = np.empty((len(active), len(current)), dtype=float)
            allowed: dict[tuple[int, int], bool] = {}
            for row, track in enumerate(active):
                for column, item in enumerate(current):
                    costs[row, column], allowed[(row, column)] = _association_cost(track, item)
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows.tolist(), columns.tolist()):
                if allowed[(row, column)]:
                    active[row].add(current[column])
                    matched_tracks.add(id(active[row]))
                    matched_observations.add(column)
        for track in active:
            if id(track) not in matched_tracks:
                track.missed_samples += 1
        for index, item in enumerate(current):
            if index not in matched_observations:
                tracks.append(WebcamRectangleTrack(track_id=f"webcam_{next_id:03d}", observations=[item]))
                next_id += 1
    return tracks
