"""YuNet/SFace facial observations, temporal tracks, and streamer identity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence, cast

import numpy as np
from ._vision_types import linear_sum_assignment

from .visual_contracts import FaceTrackSummary, IdentityStatus, NormalizedRect, SourceState


class _FaceDetector(Protocol):
    def setInputSize(self, size: tuple[int, int]) -> None: ...

    def detect(self, frame: np.ndarray) -> tuple[object, np.ndarray | None]: ...


class _FaceRecognizer(Protocol):
    def alignCrop(self, frame: np.ndarray, detection: np.ndarray) -> np.ndarray: ...

    def feature(self, aligned: np.ndarray) -> np.ndarray: ...


_COSINE_GATE = 0.32
_MOTION_GATE = 0.45
_EPSILON = 1e-6


def _normalized(vector: np.ndarray | Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("El embedding debe ser finito y no vacío")
    norm = float(np.linalg.norm(value))
    if norm <= _EPSILON:
        raise ValueError("El embedding no puede tener norma cero")
    return value / norm


@dataclass(slots=True)
class LandmarkedFace:
    """A frame-local YuNet result with five normalized landmarks."""

    box: NormalizedRect
    landmarks: tuple[tuple[float, float], ...]
    confidence: float
    embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.landmarks) != 5:
            raise ValueError("LandmarkedFace requiere exactamente cinco landmarks")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("La confianza facial debe estar entre 0 y 1")
        for point in self.landmarks:
            if len(point) != 2 or not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in point):
                raise ValueError("Los landmarks deben ser pares normalizados")
        if self.embedding is not None:
            self.embedding = _normalized(self.embedding)


class FaceIdentityBackend(Protocol):
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]: ...

    def cosine(self, left: np.ndarray, right: np.ndarray) -> float: ...


class OpenCVFaceIdentityBackend:
    """Small OpenCV adapter; model acquisition stays in :mod:`model_assets`."""

    def __init__(
        self,
        yunet_path: str | None = None,
        sface_path: str | None = None,
        *,
        detector: _FaceDetector | None = None,
        recognizer: _FaceRecognizer | None = None,
    ) -> None:
        if detector is None or recognizer is None:
            if not yunet_path or not sface_path:
                raise ValueError("Se requieren rutas YuNet y SFace o dobles OpenCV inyectados")
            import cv2

            detector_factory = cast(
                Callable[[str, str, tuple[int, int]], _FaceDetector],
                getattr(cv2, "FaceDetectorYN_create"),
            )
            recognizer_factory = cast(
                Callable[[str, str], _FaceRecognizer],
                getattr(cv2, "FaceRecognizerSF_create"),
            )
            detector = detector_factory(yunet_path, "", (320, 320))
            recognizer = recognizer_factory(sface_path, "")
        self._detector = detector
        self._recognizer = recognizer

    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        if frame.ndim < 2:
            raise ValueError("El frame debe tener alto y ancho")
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return []
        self._detector.setInputSize((int(width), int(height)))
        _, detected = self._detector.detect(frame)
        if detected is None:
            return []
        rows = np.asarray(detected)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        faces: list[LandmarkedFace] = []
        for row in rows:
            if np.asarray(row).size != 15:
                continue
            detection = np.asarray(row, dtype=np.float32).reshape(15)
            candidate = self._face_from_yunet_row(frame, detection, width, height)
            if candidate is not None:
                faces.append(candidate)
        return faces

    def _face_from_yunet_row(
        self, frame: np.ndarray, detection: np.ndarray, width: int, height: int
    ) -> LandmarkedFace | None:
        x, y, box_width, box_height = (float(value) for value in detection[:4])
        left, top = max(0.0, x), max(0.0, y)
        right, bottom = min(float(width), x + box_width), min(float(height), y + box_height)
        if right <= left or bottom <= top:
            return None
        landmarks = tuple(
            (float(detection[index]) / width, float(detection[index + 1]) / height)
            for index in range(4, 14, 2)
        )
        if any(not (0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0) for x_value, y_value in landmarks):
            return None
        embedding: np.ndarray | None = None
        try:
            aligned = self._recognizer.alignCrop(frame, detection)
            feature = self._recognizer.feature(aligned)
            embedding = _normalized(feature)
        except Exception:
            # Geometry remains useful for temporal/webcam evidence when SFace
            # cannot align a particular detection.
            embedding = None
        return LandmarkedFace(
            box=NormalizedRect(x=left / width, y=top / height, width=(right - left) / width, height=(bottom - top) / height),
            landmarks=landmarks,
            confidence=float(detection[14]),
            embedding=embedding,
        )

    @staticmethod
    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.clip(np.dot(_normalized(left), _normalized(right)), -1.0, 1.0))


@dataclass(slots=True)
class FaceObservation:
    at: float
    face: LandmarkedFace
    source_state: SourceState
    evidence_id: str
    embedding_quality: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.at) or self.at < 0:
            raise ValueError("El timestamp facial debe ser finito y no negativo")
        if not self.evidence_id:
            raise ValueError("FaceObservation requiere evidence_id")
        if not np.isfinite(self.embedding_quality) or self.embedding_quality <= 0:
            raise ValueError("embedding_quality debe ser positivo")


@dataclass(slots=True)
class FaceTrack:
    track_id: str
    observations: list[FaceObservation] = field(default_factory=list)
    missed_samples: int = 0

    @property
    def last_observation(self) -> FaceObservation:
        return self.observations[-1]

    def add(self, observation: FaceObservation) -> None:
        self.observations.append(observation)
        self.missed_samples = 0

    def embedding_centroid(self) -> np.ndarray | None:
        embeddings = [(item.face.embedding, item.embedding_quality) for item in self.observations if item.face.embedding is not None]
        if not embeddings:
            return None
        weighted = sum((embedding * quality for embedding, quality in embeddings), np.zeros_like(embeddings[0][0]))
        return _normalized(weighted)

    def duration_seconds(self) -> float:
        if not self.observations:
            return 0.0
        return max(0.0, self.observations[-1].at - self.observations[0].at)

    def recurrence_count(self) -> int:
        return len({item.source_state for item in self.observations})

    def full_camera_ratio(self) -> float:
        if not self.observations:
            return 0.0
        return sum(item.source_state is SourceState.FULL_CAMERA for item in self.observations) / len(self.observations)

    def to_summary(self) -> FaceTrackSummary:
        centroid = self.embedding_centroid()
        if centroid is None:
            raise ValueError("No se puede serializar un track sin embeddings")
        first_seen = self.observations[0].at
        last_seen = max(self.observations[-1].at, first_seen + _EPSILON)
        states = list(dict.fromkeys(item.source_state for item in self.observations))
        evidence_ids = list(dict.fromkeys(item.evidence_id for item in self.observations))
        duration = self.duration_seconds()
        return FaceTrackSummary(
            track_id=self.track_id,
            first_seen=first_seen,
            last_seen=last_seen,
            observation_count=len(self.observations),
            embedding_centroid=centroid.tolist(),
            positions=[item.face.box for item in self.observations],
            recurrent_states=states,
            full_camera_seconds=duration * self.full_camera_ratio(),
            mean_confidence=float(np.mean([item.face.confidence for item in self.observations])),
            evidence_ids=evidence_ids,
        )


@dataclass(frozen=True, slots=True)
class StreamerIdentityResult:
    status: IdentityStatus
    track_id: str | None
    score: float | None
    candidate_scores: dict[str, float]


def _iou(left: NormalizedRect, right: NormalizedRect) -> float:
    x1, y1 = max(left.x, right.x), max(left.y, right.y)
    x2, y2 = min(left.x + left.width, right.x + right.width), min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _center_motion(left: NormalizedRect, right: NormalizedRect) -> float:
    left_center = (left.x + left.width / 2, left.y + left.height / 2)
    right_center = (right.x + right.width / 2, right.y + right.height / 2)
    return min(1.0, float(np.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])))


def _cosine_track_observation(track: FaceTrack, observation: FaceObservation) -> float:
    centroid = track.embedding_centroid()
    embedding = observation.face.embedding
    if centroid is None or embedding is None or centroid.size != embedding.size:
        return -1.0
    return float(np.dot(centroid, embedding))


def _association_cost(track: FaceTrack, observation: FaceObservation) -> tuple[float, bool]:
    previous = track.last_observation
    similarity = _cosine_track_observation(track, observation)
    motion = _center_motion(previous.face.box, observation.face.box)
    if similarity < _COSINE_GATE or motion > _MOTION_GATE:
        return 1_000_000.0, False
    cost = 0.55 * (1.0 - similarity) + 0.30 * (1.0 - _iou(previous.face.box, observation.face.box)) + 0.15 * motion
    return cost, True


def associate_face_tracks(
    observations: Sequence[FaceObservation], *, sample_timestamps: Sequence[float] = (), max_missed_samples: int = 3
) -> list[FaceTrack]:
    """Associate per-sample faces with global Hungarian matching and hard gates."""
    if max_missed_samples < 0:
        raise ValueError("max_missed_samples no puede ser negativo")
    grouped: dict[float, list[FaceObservation]] = {}
    for observation in observations:
        grouped.setdefault(float(observation.at), []).append(observation)
    timestamps = sorted({*grouped, *(float(value) for value in sample_timestamps)})
    tracks: list[FaceTrack] = []
    next_id = 1
    for timestamp in timestamps:
        current = grouped.get(timestamp, [])
        active = [track for track in tracks if track.missed_samples < max_missed_samples]
        matched_tracks: set[int] = set()
        matched_observations: set[int] = set()
        if active and current:
            costs = np.empty((len(active), len(current)), dtype=float)
            allowed: dict[tuple[int, int], bool] = {}
            for row, track in enumerate(active):
                for column, item in enumerate(current):
                    cost, accepted = _association_cost(track, item)
                    costs[row, column] = cost
                    allowed[(row, column)] = accepted
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
                tracks.append(FaceTrack(track_id=f"face_{next_id:03d}", observations=[item]))
                next_id += 1
    return tracks


def resolve_streamer_identity(
    tracks: Sequence[FaceTrack], *, saved_profile_embedding: np.ndarray | None = None, ambiguity_margin: float = 0.12
) -> StreamerIdentityResult:
    candidates = [
        track for track in tracks if track.observations and track.embedding_centroid() is not None
    ]
    if not candidates:
        return StreamerIdentityResult(IdentityStatus.UNAVAILABLE, None, None, {})
    profile = _normalized(saved_profile_embedding) if saved_profile_embedding is not None else None
    maximum_duration = max((track.duration_seconds() for track in candidates), default=0.0)
    maximum_recurrence = max((track.recurrence_count() for track in candidates), default=0)
    scores: dict[str, float] = {}
    for track in candidates:
        duration = track.duration_seconds() / maximum_duration if maximum_duration > _EPSILON else 1.0
        recurrence = track.recurrence_count() / maximum_recurrence if maximum_recurrence else 1.0
        terms = [(0.35, duration), (0.20, recurrence), (0.20, track.full_camera_ratio())]
        if profile is not None:
            centroid = track.embedding_centroid()
            cosine = float(np.dot(centroid, profile)) if centroid is not None and centroid.size == profile.size else -1.0
            terms.append((0.25, (cosine + 1.0) / 2.0))
        active_weight = sum(weight for weight, _ in terms)
        scores[track.track_id] = sum(weight * value for weight, value in terms) / active_weight
    ranked = sorted(candidates, key=lambda track: (-scores[track.track_id], track.track_id))
    winner = ranked[0]
    if len(ranked) > 1 and scores[winner.track_id] - scores[ranked[1].track_id] < ambiguity_margin:
        return StreamerIdentityResult(IdentityStatus.AMBIGUOUS, None, scores[winner.track_id], scores)
    return StreamerIdentityResult(IdentityStatus.CONFIDENT, winner.track_id, scores[winner.track_id], scores)
