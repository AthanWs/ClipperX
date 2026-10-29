"""Face detection helpers for cropping.

Uses MediaPipe Face Detection as the primary backend with Haar Cascade as fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Protocol, cast

import cv2
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


class _MediaPipeBox(Protocol):
    origin_x: int
    origin_y: int
    width: int
    height: int


class _MediaPipeDetection(Protocol):
    bounding_box: _MediaPipeBox


class _MediaPipeResult(Protocol):
    detections: Iterable[_MediaPipeDetection]


class _MediaPipeFaceDetector(Protocol):
    def detect(self, image: object) -> _MediaPipeResult: ...
    def close(self) -> None: ...


class _MediaPipeDetectorFactory(Protocol):
    def create_from_options(self, options: object) -> _MediaPipeFaceDetector: ...


class _MediaPipeVision(Protocol):
    FaceDetector: _MediaPipeDetectorFactory

    def FaceDetectorOptions(self, *, base_options: object,
                            min_detection_confidence: float) -> object: ...


class _MediaPipeTasks(Protocol):
    vision: _MediaPipeVision

    def BaseOptions(self, *, model_asset_path: str) -> object: ...


class _MediaPipeImageFormat(Protocol):
    SRGB: object


class _MediaPipeModule(Protocol):
    tasks: _MediaPipeTasks
    ImageFormat: _MediaPipeImageFormat

    def Image(self, *, image_format: object, data: np.ndarray) -> object: ...


# Try to import MediaPipe; fall back to Haar Cascade if unavailable
try:
    mp = cast(_MediaPipeModule, import_module("mediapipe"))
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False
    logger.info("[face] MediaPipe no disponible, usando Haar Cascade como fallback")

_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "blaze_face_short_range.tflite"


@dataclass(slots=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


class _MediaPipeDetector:
    """Wraps MediaPipe Face Detection (Tasks API) for a consistent interface."""

    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or _MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Modelo MediaPipe no encontrado: {path}")
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            min_detection_confidence=0.5,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._detector.detect(mp_image)
        if not results.detections:
            return []
        h, w = image.shape[:2]
        boxes: list[FaceBox] = []
        for detection in results.detections:
            bbox = detection.bounding_box
            x = max(0, bbox.origin_x)
            y = max(0, bbox.origin_y)
            bw = min(bbox.width, w - x)
            bh = min(bbox.height, h - y)
            if bw > 0 and bh > 0:
                boxes.append(FaceBox(x=x, y=y, width=bw, height=bh))
        return boxes

    def close(self) -> None:
        self._detector.close()


class _HaarData(Protocol):
    haarcascades: str


class _HaarDetector:
    """Wraps OpenCV Haar Cascade for a consistent interface."""

    def __init__(self) -> None:
        cascade_path = cast(_HaarData, getattr(cv2, "data")).haarcascades + "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(cascade_path)
        if self._classifier.empty():
            raise RuntimeError("No se pudo cargar el modelo de detección facial Haar.")

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self._classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64),
        )
        return [
            FaceBox(x=int(x), y=int(y), width=int(w), height=int(h))
            for (x, y, w, h) in detections
        ]

    def close(self) -> None:
        pass


# Type alias for the detector (either backend)
FaceDetector = Any


def load_face_model() -> FaceDetector:
    """Load the face detection model. Returns MediaPipe if available, else Haar Cascade."""
    if _MEDIAPIPE_AVAILABLE:
        try:
            detector = _MediaPipeDetector()
            logger.info("[face] usando MediaPipe Face Detection")
            return detector
        except Exception as exc:
            logger.warning("[face] MediaPipe falló, usando Haar Cascade", extra={"error": str(exc)})
    logger.info("[face] usando Haar Cascade (fallback)")
    return _HaarDetector()


def detect_faces(image: np.ndarray, classifier: FaceDetector | None = None) -> list[FaceBox]:
    """Detect faces in an image using the provided detector."""
    detector = classifier or load_face_model()
    return detector.detect(image)


def track_video_face(
    video_path: Path,
    *,
    stride: int = 12,
    warmup: int = 24,
    classifier: FaceDetector | None = None,
) -> FaceBox | None:
    detector = classifier or load_face_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    boxes: list[FaceBox] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx < warmup:
            frame_idx += 1
            continue
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        detections = detect_faces(frame, detector)
        if detections:
            boxes.append(_largest_box(detections))
        frame_idx += 1

    cap.release()
    if not boxes:
        return None
    return _smooth_boxes(boxes)


def track_video_face_window(
    video_path: Path,
    *,
    start: float,
    end: float,
    stride: int = 12,
    max_samples: int = 60,
    classifier: FaceDetector | None = None,
) -> FaceBox | None:
    detector = classifier or load_face_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    start = max(0.0, float(start))
    end = max(start, float(end))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    boxes: list[FaceBox] = []
    frame_idx = 0
    samples = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if current_ms / 1000.0 > end:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        detections = detect_faces(frame, detector)
        if detections:
            boxes.append(_largest_box(detections))
        frame_idx += 1
        samples += 1
        if samples >= max_samples:
            break

    cap.release()
    if not boxes:
        return None
    return _smooth_boxes(boxes)


def _largest_box(boxes: Iterable[FaceBox]) -> FaceBox:
    return max(boxes, key=lambda box: box.width * box.height)


def _smooth_boxes(boxes: list[FaceBox]) -> FaceBox:
    xs = np.array([box.x for box in boxes], dtype=float)
    ys = np.array([box.y for box in boxes], dtype=float)
    ws = np.array([box.width for box in boxes], dtype=float)
    hs = np.array([box.height for box in boxes], dtype=float)

    return FaceBox(
        x=int(np.percentile(xs, 20)),
        y=int(np.percentile(ys, 20)),
        width=int(np.percentile(ws, 80)),
        height=int(np.percentile(hs, 80)),
    )


@dataclass(slots=True)
class WebcamRegion:
    """Detected webcam overlay position within a frame."""
    x: int
    y: int
    width: int
    height: int
    corner: str  # "top-left", "top-right", "bottom-left", "bottom-right", "center"
    relative_area: float  # fraction of total frame area

    def as_tuple(self) -> tuple[int, int, int, int, str]:
        return self.x, self.y, self.width, self.height, self.corner


def detect_webcam_region(
    video_path: Path,
    *,
    stride: int = 30,
    max_samples: int = 20,
    start: float | None = None,
    end: float | None = None,
    classifier: FaceDetector | None = None,
) -> WebcamRegion | None:
    """Estimate a persistent corner face crop, rejecting ambiguous/moving windows.

    Corner-local inputs recover small faces without preferring a large face in
    the content. This estimates a crop, not identity or exact overlay boundaries;
    full-camera classification remains a separate concern.
    """
    owns_detector = classifier is None
    detector = load_face_model() if classifier is None else classifier
    cap = None
    observations: list[dict[str, FaceBox]] = []
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_w <= 0 or frame_h <= 0 or max_samples <= 0:
            return None

        if start is not None or end is not None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if total_frames > 0 else 0.0
            window_start = max(0.0, float(start or 0.0))
            window_end = min(duration, float(end if end is not None else duration))
            if window_end <= window_start:
                window_end = min(duration, window_start + 1.0)
            for timestamp in np.linspace(window_start, window_end, max_samples):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
                ok, frame = cap.read()
                if ok:
                    observations.append(_detect_corner_faces(frame, detector))
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 4))
            frame_idx = 0
            while len(observations) < max_samples:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % stride == 0:
                    observations.append(_detect_corner_faces(frame, detector))
                frame_idx += 1
    finally:
        try:
            if cap is not None:
                cap.release()
        finally:
            if owns_detector:
                detector.close()

    corners = {corner for sample in observations for corner in sample}
    supported = {
        corner: [sample[corner] for sample in observations if corner in sample]
        for corner in corners
        if sum(corner in sample for sample in observations) >= 3
    }
    # Multiple supported positions must be split upstream, never averaged here.
    if len(supported) != 1:
        return None
    corner, face_positions = next(iter(supported.items()))
    if len(face_positions) / len(observations) < 0.6:
        return None
    # A short relocation may have fewer than three samples in its new position.
    # Reject two consecutive alternate-only observations; ignore isolated noise.
    for previous, current in zip(observations, observations[1:]):
        if corner not in previous and corner not in current and previous.keys() & current.keys():
            return None

    smoothed = _smooth_boxes(face_positions)
    wc_x, wc_y, wc_w, wc_h = _estimate_webcam_bounds(smoothed, frame_w, frame_h)
    relative_area = (wc_w * wc_h) / (frame_w * frame_h) if frame_w * frame_h > 0 else 0.0

    region = WebcamRegion(
        x=wc_x, y=wc_y, width=wc_w, height=wc_h,
        corner=corner, relative_area=relative_area,
    )
    logger.info(
        "[face] webcam detectada",
        extra={"corner": corner, "relative_area": f"{relative_area:.2%}", "region": (wc_x, wc_y, wc_w, wc_h)},
    )
    return region


def _detect_corner_faces(image: np.ndarray, detector: FaceDetector) -> dict[str, FaceBox]:
    """Translate unambiguous 40%-width/50%-height corner observations to source pixels."""
    height, width = image.shape[:2]
    tile_w, tile_h = max(1, round(width * 0.4)), max(1, round(height * 0.5))
    positions = {
        "top-left": (0, 0),
        "top-right": (width - tile_w, 0),
        "bottom-left": (0, height - tile_h),
        "bottom-right": (width - tile_w, height - tile_h),
    }
    observations: dict[str, FaceBox] = {}
    for corner, (x, y) in positions.items():
        boxes = detect_faces(image[y:y + tile_h, x:x + tile_w], detector)
        if len(boxes) == 1:
            box = boxes[0]
            observations[corner] = FaceBox(x + box.x, y + box.y, box.width, box.height)
    return observations


def _estimate_webcam_bounds(
    face: FaceBox,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Expand a face to the likely fixed webcam rectangle without cutting the head."""

    near_left = face.x < frame_width * 0.15
    near_right = face.x + face.width > frame_width * 0.85
    near_top = face.y < frame_height * 0.15
    near_bottom = face.y + face.height > frame_height * 0.85

    left = 0 if near_left else max(0, round(face.x - face.width * 0.45))
    right = (
        frame_width
        if near_right
        else min(frame_width, round(face.x + face.width * 1.40))
    )
    top = 0 if near_top else max(0, round(face.y - face.height * 0.55))
    bottom = (
        frame_height
        if near_bottom
        else min(frame_height, round(face.y + face.height * 1.25))
    )
    return left, top, max(2, right - left), max(2, bottom - top)
