from __future__ import annotations

import numpy as np
import pytest

import clipperx.face_identity as face_identity
from clipperx.face_identity import FaceObservation, FaceTrack, LandmarkedFace, OpenCVFaceIdentityBackend, associate_face_tracks, resolve_streamer_identity
from clipperx.visual_contracts import IdentityStatus, NormalizedRect, SourceState


def unit(index: int, size: int = 3) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector

def direction(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    return np.array([np.cos(radians), np.sin(radians)], dtype=np.float32)





def rect(x: float, y: float, width: float = 0.15, height: float = 0.2) -> NormalizedRect:
    return NormalizedRect(x=x, y=y, width=width, height=height)


def face(box: NormalizedRect, embedding: np.ndarray | None = None, confidence: float = 0.9) -> LandmarkedFace:
    return LandmarkedFace(box=box, landmarks=((0.1, 0.1), (0.2, 0.1), (0.15, 0.15), (0.1, 0.2), (0.2, 0.2)), confidence=confidence, embedding=embedding)


def obs(at: float, box: NormalizedRect, embedding: np.ndarray | None, *, state: SourceState = SourceState.CONTENT_WITH_CAMERA, quality: float = 1.0, evidence_id: str | None = None) -> FaceObservation:
    return FaceObservation(at=at, face=face(box, embedding), source_state=state, evidence_id=evidence_id or f"evidence_{at}", embedding_quality=quality)


@pytest.mark.parametrize(
    ("landmarks", "embedding"),
    [(((0.1, 0.1),) * 4, unit(0)), (((1.1, 0.1),) * 5, unit(0)), (((0.1, 0.1),) * 5, np.zeros(2, dtype=np.float32))],
)
def test_landmarked_face_rejects_invalid_shape_or_non_normalized_data(landmarks: tuple[tuple[float, float], ...], embedding: np.ndarray) -> None:
    with pytest.raises(ValueError):
        LandmarkedFace(box=rect(0.1, 0.1), landmarks=landmarks, confidence=0.9, embedding=embedding)


def test_opencv_backend_uses_full_yunet_row_for_exact_alignment_and_l2_embedding() -> None:
    row = np.array([[10, 20, 30, 40, 12, 25, 32, 25, 22, 35, 15, 48, 30, 48, 0.95]], dtype=np.float32)

    class Detector:
        input_size: tuple[int, int] | None = None
        def setInputSize(self, size: tuple[int, int]) -> None: self.input_size = size
        def detect(self, frame: np.ndarray) -> tuple[None, np.ndarray]: return None, row

    class Recognizer:
        aligned_row: np.ndarray | None = None
        def alignCrop(self, frame: np.ndarray, detection: np.ndarray) -> np.ndarray:
            self.aligned_row = detection
            return frame
        def feature(self, aligned: np.ndarray) -> np.ndarray: return np.array([[3.0, 4.0]], dtype=np.float32)

    detector, recognizer = Detector(), Recognizer()
    faces = OpenCVFaceIdentityBackend(detector=detector, recognizer=recognizer).detect_and_embed(np.zeros((100, 200, 3), dtype=np.uint8))
    assert detector.input_size == (200, 100)
    assert recognizer.aligned_row is not None
    assert np.array_equal(recognizer.aligned_row, row[0])
    assert len(faces) == 1
    assert faces[0].embedding is not None
    assert np.allclose(faces[0].embedding, np.array([0.6, 0.8], dtype=np.float32))


def test_opencv_backend_ignores_non_yunet_detection_rows() -> None:
    class Detector:
        def setInputSize(self, size: tuple[int, int]) -> None: pass
        def detect(self, frame: np.ndarray) -> tuple[None, np.ndarray]: return None, np.zeros((1, 14), dtype=np.float32)
    class Recognizer:
        def alignCrop(self, frame: np.ndarray, detection: np.ndarray) -> np.ndarray: raise AssertionError("malformed row")
        def feature(self, aligned: np.ndarray) -> np.ndarray: raise AssertionError("malformed row")
    assert OpenCVFaceIdentityBackend(detector=Detector(), recognizer=Recognizer()).detect_and_embed(np.zeros((100, 100, 3), dtype=np.uint8)) == []


def test_association_uses_global_hungarian_assignment_not_rowwise_greedy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[np.ndarray] = []
    def global_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(cost.copy())
        return np.array([0, 1]), np.array([1, 0])
    monkeypatch.setattr(face_identity, "linear_sum_assignment", global_assignment)
    tracks = associate_face_tracks([
        obs(0, rect(0.10, 0.1), unit(0), evidence_id="a0"), obs(0, rect(0.45, 0.1), unit(0), evidence_id="b0"),
        obs(1, rect(0.12, 0.1), unit(0), evidence_id="a1"), obs(1, rect(0.47, 0.1), unit(0), evidence_id="b1"),
    ], sample_timestamps=(0, 1))
    assert calls
    assert [track.observations[-1].evidence_id for track in tracks] == ["b1", "a1"]


def test_association_cost_uses_exact_contract_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[np.ndarray] = []
    def solve(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        captured.append(cost.copy())
        return np.array([0]), np.array([0])
    monkeypatch.setattr(face_identity, "linear_sum_assignment", solve)

    associate_face_tracks(
        [
            obs(0, rect(0.0, 0.0, 0.2, 0.2), direction(0)),
            obs(1, rect(0.1, 0.0, 0.2, 0.2), direction(60)),
        ], sample_timestamps=(0, 1),
    )

    assert captured[0][0, 0] == pytest.approx(0.55 * 0.5 + 0.30 * (2 / 3) + 0.15 * 0.1)

@pytest.mark.parametrize("candidate", [obs(1, rect(0.12, 0.1), unit(1)), obs(1, rect(0.70, 0.1), unit(0))], ids=["cosine_gate", "motion_gate"])
def test_association_gates_start_new_track(candidate: FaceObservation) -> None:
    assert len(associate_face_tracks([obs(0, rect(0.10, 0.1), unit(0)), candidate], sample_timestamps=(0, 1))) == 2

def test_association_uses_historical_embedding_centroid() -> None:
    tracks = associate_face_tracks(
        [
            obs(0, rect(0.1, 0.1), direction(0), evidence_id="front"),
            obs(1, rect(0.1, 0.1), direction(70), evidence_id="turned"),
            obs(2, rect(0.1, 0.1), direction(-20), evidence_id="returned"),
        ],
        sample_timestamps=(0, 1, 2),
    )
    assert len(tracks) == 1
    assert [item.evidence_id for item in tracks[0].observations] == ["front", "turned", "returned"]



def test_empty_sample_timestamps_process_observations_without_implicit_misses() -> None:
    tracks = associate_face_tracks([obs(0, rect(0.1, 0.1), unit(0)), obs(10, rect(0.11, 0.1), unit(0))], sample_timestamps=())
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 2


def test_detection_after_three_missing_samples_starts_new_track() -> None:
    tracks = associate_face_tracks([obs(0, rect(0.1, 0.1), unit(0)), obs(4, rect(0.11, 0.1), unit(0))], sample_timestamps=(0, 1, 2, 3, 4))
    assert len(tracks) == 2


def test_detection_before_three_missing_samples_keeps_existing_track() -> None:
    tracks = associate_face_tracks(
        [obs(0, rect(0.1, 0.1), unit(0)), obs(3, rect(0.11, 0.1), unit(0))],
        sample_timestamps=(0, 1, 2, 3),
    )
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 2


def test_quality_weighted_centroid_is_normalized_at_serialization_boundary() -> None:
    tracks = associate_face_tracks([obs(0, rect(0.1, 0.1), unit(0), quality=3.0), obs(1, rect(0.11, 0.1), unit(1), quality=1.0)], sample_timestamps=(0, 1))
    centroid = np.asarray(tracks[0].to_summary().embedding_centroid)
    assert np.isclose(np.linalg.norm(centroid), 1.0)
    assert centroid[0] > centroid[1]


def test_larger_foreign_face_does_not_replace_persistent_streamer() -> None:
    tracks = associate_face_tracks([
        obs(0, rect(0.02, 0.02, 0.18, 0.25), unit(0)), obs(5, rect(0.02, 0.02, 0.18, 0.25), unit(0)),
        obs(10, rect(0.30, 0.05, 0.60, 0.85), unit(1)), obs(10, rect(0.02, 0.02, 0.18, 0.25), unit(0)),
    ], sample_timestamps=(0, 5, 10))
    persistent = next(
        track for track in tracks
        if (embedding := track.observations[0].face.embedding) is not None and embedding[0] == 1.0
    )
    assert resolve_streamer_identity(tracks).track_id == persistent.track_id


def test_saved_profile_can_rank_smaller_streamer_without_using_face_area() -> None:
    tracks = associate_face_tracks([
        obs(0, rect(0.0, 0.0, 0.15, 0.2), unit(0)), obs(2, rect(0.0, 0.0, 0.15, 0.2), unit(0)),
        obs(0, rect(0.2, 0.0, 0.7, 0.8), unit(1)), obs(1, rect(0.2, 0.0, 0.7, 0.8), unit(1)), obs(2, rect(0.2, 0.0, 0.7, 0.8), unit(1)),
    ], sample_timestamps=(0, 1, 2))
    small = next(track for track in tracks if track.observations[0].face.box.width == 0.15)
    identity = resolve_streamer_identity(tracks, saved_profile_embedding=unit(0))
    assert identity.status is IdentityStatus.CONFIDENT
    assert identity.track_id == small.track_id


def test_identity_score_uses_exact_active_weights_and_never_face_area() -> None:
    smaller_persistent = FaceTrack(
        track_id="face_small",
        observations=[
            obs(0, rect(0.0, 0.0, 0.1, 0.1), unit(0), state=SourceState.FULL_CAMERA, evidence_id="small_0"),
            obs(10, rect(0.0, 0.0, 0.1, 0.1), unit(0), state=SourceState.CONTENT_WITH_CAMERA, evidence_id="small_10"),
        ],
    )
    larger_brief = FaceTrack(
        track_id="face_large",
        observations=[
            obs(0, rect(0.2, 0.0, 0.7, 0.8), unit(1), evidence_id="large_0"),
            obs(5, rect(0.2, 0.0, 0.7, 0.8), unit(1), evidence_id="large_5"),
        ],
    )

    identity = resolve_streamer_identity([larger_brief, smaller_persistent])

    expected_small = (0.35 * 1.0 + 0.20 * 1.0 + 0.20 * 0.5) / 0.75
    expected_large = (0.35 * 0.5 + 0.20 * 0.5 + 0.20 * 0.0) / 0.75
    assert identity.status is IdentityStatus.CONFIDENT
    assert identity.track_id == "face_small"
    assert identity.candidate_scores["face_small"] == pytest.approx(expected_small)
    assert identity.candidate_scores["face_large"] == pytest.approx(expected_large)


def test_single_valid_track_is_confident() -> None:
    track = FaceTrack(
        track_id="face_only",
        observations=[
            obs(0, rect(0.1, 0.1), unit(0), evidence_id="only_0"),
            obs(1, rect(0.1, 0.1), unit(0), evidence_id="only_1"),
        ],
    )
    identity = resolve_streamer_identity([track])
    assert identity.status is IdentityStatus.CONFIDENT
    assert identity.track_id == "face_only"
def test_close_identity_scores_return_ambiguous() -> None:
    tracks = associate_face_tracks([obs(0, rect(0.0, 0.0), unit(0)), obs(1, rect(0.01, 0.0), unit(0)), obs(0, rect(0.4, 0.0), unit(1)), obs(1, rect(0.41, 0.0), unit(1))], sample_timestamps=(0, 1))
    identity = resolve_streamer_identity(tracks)
    assert identity.status is IdentityStatus.AMBIGUOUS
    assert identity.track_id is None


def test_no_tracks_returns_unavailable_identity() -> None:
    assert resolve_streamer_identity([]).status is IdentityStatus.UNAVAILABLE


def test_sface_failure_isolated_per_detection_and_keeps_geometry() -> None:
    rows = np.array(
        [
            [10, 10, 20, 20, 12, 14, 24, 14, 18, 20, 13, 27, 23, 27, 0.9],
            [50, 10, 20, 20, 52, 14, 64, 14, 58, 20, 53, 27, 63, 27, 0.8],
        ],
        dtype=np.float32,
    )

    class Detector:
        def setInputSize(self, size: tuple[int, int]) -> None:
            pass

        def detect(self, frame: np.ndarray) -> tuple[None, np.ndarray]:
            return None, rows

    class Recognizer:
        def alignCrop(self, frame: np.ndarray, detection: np.ndarray) -> np.ndarray:
            if detection[0] == 10:
                raise RuntimeError("first face cannot be aligned")
            return detection

        def feature(self, aligned: np.ndarray) -> np.ndarray:
            return np.array([[3.0, 4.0]], dtype=np.float32)

    faces = OpenCVFaceIdentityBackend(
        detector=Detector(), recognizer=Recognizer()
    ).detect_and_embed(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(faces) == 2
    assert faces[0].embedding is None
    assert faces[1].embedding is not None
    assert np.allclose(faces[1].embedding, np.array([0.6, 0.8], dtype=np.float32))


def test_empty_track_is_not_a_valid_identity_candidate() -> None:
    identity = resolve_streamer_identity([FaceTrack(track_id="empty")])

    assert identity.status is IdentityStatus.UNAVAILABLE
    assert identity.track_id is None
    assert identity.candidate_scores == {}


def test_track_without_embedding_is_not_a_valid_identity_candidate() -> None:
    track = FaceTrack(
        track_id="geometry_only",
        observations=[obs(0, rect(0.1, 0.1), None, evidence_id="geometry")],
    )

    identity = resolve_streamer_identity([track])

    assert identity.status is IdentityStatus.UNAVAILABLE
    assert identity.track_id is None
    assert identity.candidate_scores == {}


def test_embedding_quality_changes_association_through_weighted_centroid() -> None:
    tracks = associate_face_tracks(
        [
            obs(0, rect(0.1, 0.1), direction(0), quality=10.0, evidence_id="strong"),
            obs(1, rect(0.1, 0.1), direction(60), quality=1.0, evidence_id="weak"),
            obs(2, rect(0.1, 0.1), direction(-55), quality=1.0, evidence_id="return"),
        ],
        sample_timestamps=(0, 1, 2),
    )

    assert len(tracks) == 1
    assert [item.evidence_id for item in tracks[0].observations] == [
        "strong",
        "weak",
        "return",
    ]
