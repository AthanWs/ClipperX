from __future__ import annotations

from clipperx.webcam_tracking import WebcamRegionObservation, associate_webcam_tracks
from clipperx.visual_contracts import NormalizedRect


def rect(x: float, y: float, width: float = 0.25, height: float = 0.25) -> NormalizedRect:
    return NormalizedRect(x=x, y=y, width=width, height=height)


def observation(at: float, rectangle: NormalizedRect, *, faces: tuple[str, ...] = ()) -> WebcamRegionObservation:
    return WebcamRegionObservation(at=at, rect=rectangle, confidence=0.9, evidence_id=f"webcam_{at}", face_track_ids=faces)


def test_webcam_rectangle_persists_when_face_evidence_is_absent() -> None:
    tracks = associate_webcam_tracks([observation(0, rect(0.0, 0.0), faces=("face_001",)), observation(1, rect(0.01, 0.01))], sample_timestamps=(0, 1))
    assert len(tracks) == 1
    assert tracks[0].face_track_ids == {"face_001"}
    assert len(tracks[0].observations) == 2


def test_shared_face_evidence_can_relocate_a_webcam_rectangle() -> None:
    tracks = associate_webcam_tracks([observation(0, rect(0.0, 0.0), faces=("face_001",)), observation(1, rect(0.7, 0.7), faces=("face_001",))], sample_timestamps=(0, 1))
    assert len(tracks) == 1


def test_webcam_detection_after_three_misses_starts_new_rectangle_track() -> None:
    tracks = associate_webcam_tracks([observation(0, rect(0.0, 0.0)), observation(4, rect(0.0, 0.0))], sample_timestamps=(0, 1, 2, 3, 4))
    assert len(tracks) == 2


def test_webcam_detection_before_three_misses_keeps_rectangle_track() -> None:
    tracks = associate_webcam_tracks(
        [observation(0, rect(0.0, 0.0)), observation(3, rect(0.01, 0.0))],
        sample_timestamps=(0, 1, 2, 3),
    )
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 2


def test_bounded_center_motion_can_associate_rectangles_without_overlap() -> None:
    tracks = associate_webcam_tracks(
        [observation(0, rect(0.0, 0.0, 0.1, 0.1)), observation(1, rect(0.2, 0.0, 0.1, 0.1))],
        sample_timestamps=(0, 1),
    )
    assert len(tracks) == 1


def test_unrelated_rectangle_beyond_motion_gate_starts_new_track() -> None:
    tracks = associate_webcam_tracks(
        [observation(0, rect(0.0, 0.0, 0.1, 0.1)), observation(1, rect(0.7, 0.7, 0.1, 0.1))],
        sample_timestamps=(0, 1),
    )
    assert len(tracks) == 2


def test_empty_webcam_sample_timestamps_do_not_create_implicit_misses() -> None:
    tracks = associate_webcam_tracks([observation(0, rect(0.0, 0.0)), observation(20, rect(0.01, 0.0))], sample_timestamps=())
    assert len(tracks) == 1


def test_webcam_track_serializes_rectangle_and_face_ids_separately() -> None:
    track = associate_webcam_tracks([observation(0, rect(0.0, 0.0), faces=("face_001",)), observation(1, rect(0.0, 0.0))], sample_timestamps=(0, 1))[0]
    contract = track.to_contract()
    assert contract.face_track_ids == ["face_001"]
    assert contract.observations[0].rect == rect(0.0, 0.0)
