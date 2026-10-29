import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from clipperx import face

from clipperx.face import FaceBox, _estimate_webcam_bounds, _largest_box, detect_webcam_region


def test_corner_webcam_expands_to_frame_edges_without_cutting_head() -> None:
    bounds = _estimate_webcam_bounds(FaceBox(94, 51, 177, 177), 1280, 720)

    assert bounds == (0, 0, 342, 272)


def test_legacy_webcam_helpers_remain_isolated_from_v2_tracking() -> None:
    source = inspect.getsource(detect_webcam_region) + inspect.getsource(_largest_box)
    assert "face_identity" not in source
    assert "webcam_tracking" not in source


class _SampleCapture:
    """Video I/O boundary double; selection still runs through the real entry point."""

    def __init__(self, count: int, *, opened: bool = True) -> None:
        self.count = count
        self.opened = opened
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, key: int) -> float:
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 1000,
            cv2.CAP_PROP_FRAME_HEIGHT: 600,
            cv2.CAP_PROP_FPS: 1,
            cv2.CAP_PROP_FRAME_COUNT: self.count,
        }.get(key, 0)

    def set(self, _key: int, _value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= self.count:
            return False, None
        frame = np.zeros((600, 1000, 3), dtype=np.uint8)
        frame[:, :, 0] = self.index
        frame[:, :400, 1] = 1
        frame[:, 600:, 1] = 2
        frame[300:, :, 2] = 1
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class _TileDetector:
    """Deterministic face observations, without depending on model availability."""

    def __init__(self, observations: list[set[str]], *, global_face: bool = False) -> None:
        self.observations = observations
        self.global_face = global_face
        self.closed = False

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        if image.shape[:2] == (600, 1000):
            return [FaceBox(450, 100, 250, 250)] if self.global_face else []
        index, horizontal, vertical = (int(value) for value in image[0, 0])
        corner = ("bottom" if vertical else "top") + ("-left" if horizontal == 1 else "-right")
        if corner not in self.observations[index]:
            return []
        return [FaceBox(100 if horizontal == 1 else 230, 160 if vertical else 60, 70, 70)]

    def close(self) -> None:
        self.closed = True


def _detect_samples(
    monkeypatch: pytest.MonkeyPatch,
    observations: list[set[str]],
    *,
    global_face: bool = False,
    windowed: bool = True,
) -> tuple[face.WebcamRegion | None, _SampleCapture, _TileDetector]:
    capture = _SampleCapture(len(observations))
    detector = _TileDetector(observations, global_face=global_face)
    monkeypatch.setattr(face.cv2, "VideoCapture", lambda _path: capture)
    kwargs = {"start": 0.0, "end": 8.0} if windowed else {}
    result = detect_webcam_region(
        Path("controlled-observations.mp4"),
        classifier=detector,
        max_samples=8,
        stride=1,
        **kwargs,
    )
    return result, capture, detector


@pytest.mark.parametrize("corner", ["top-left", "top-right", "bottom-left", "bottom-right"])
@pytest.mark.parametrize("windowed", [True, False])
def test_webcam_recovers_persistent_corner_faces_and_translates_coordinates(
    monkeypatch: pytest.MonkeyPatch, corner: str, windowed: bool,
) -> None:
    region, capture, detector = _detect_samples(
        monkeypatch, [{corner}] * 8, windowed=windowed,
    )
    assert region is not None
    assert region.corner == corner
    assert (region.x >= 500) == corner.endswith("right")
    assert (region.y >= 300) == corner.startswith("bottom")
    assert capture.released
    assert not detector.closed


def test_webcam_ignores_large_global_content_face_and_isolated_other_corner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = [{"top-left", "bottom-right"}] + [{"top-left"}] * 7
    region, _, _ = _detect_samples(monkeypatch, observations, global_face=True)
    assert region is not None
    assert region.corner == "top-left"
    assert region.x == 0
    assert region.width < 400


@pytest.mark.parametrize(
    "observations",
    [
        [{"top-left"}] * 3 + [set()] * 5,
        [{"top-left"}] * 2,
        [{"top-left", "top-right"}] * 8,
        [{"top-left"}] * 4 + [{"top-right"}] * 4,
        [{"top-left"}] * 5 + [{"top-right"}] * 3,
        [{"top-left"}] * 6 + [{"top-right"}] * 2,
    ],
    ids=["under-60-percent", "under-three-samples", "two-persistent-corners",
         "equal-relocation", "partial-relocation-three", "partial-relocation-two"],
)
def test_webcam_rejects_insufficient_persistence_or_relocation(
    monkeypatch: pytest.MonkeyPatch, observations: list[set[str]],
) -> None:
    region, _, _ = _detect_samples(monkeypatch, observations, global_face=True)
    assert region is None


def test_webcam_persistence_uses_decoded_frames_instead_of_requested_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region, _, _ = _detect_samples(monkeypatch, [{"top-left"}] * 3 + [set()] * 2)
    assert region is not None


@pytest.mark.parametrize("owned", [True, False])
def test_webcam_releases_capture_on_detector_exception_and_closes_only_owned_detector(
    monkeypatch: pytest.MonkeyPatch, owned: bool,
) -> None:
    capture = _SampleCapture(8)
    detector = _TileDetector([])
    monkeypatch.setattr(face.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(face, "load_face_model", lambda: detector)

    def fail(_image: np.ndarray) -> list[FaceBox]:
        raise RuntimeError("inference failure")

    monkeypatch.setattr(detector, "detect", fail)
    with pytest.raises(RuntimeError, match="inference failure"):
        detect_webcam_region(
            Path("controlled-observations.mp4"), start=0, end=8,
            classifier=None if owned else detector,
        )
    assert capture.released
    assert detector.closed is owned


def test_webcam_releases_unopened_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _SampleCapture(0, opened=False)
    detector = _TileDetector([])
    monkeypatch.setattr(face.cv2, "VideoCapture", lambda _path: capture)
    assert detect_webcam_region(Path("missing.mp4"), classifier=detector) is None
    assert capture.released
    assert not detector.closed
