from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NoReturn, Sequence, TypedDict

import cv2
import numpy as np
import pytest

from clipperx.face_identity import FaceTrack, LandmarkedFace
from clipperx.jobs import JobStore
from clipperx.render_clock import SourceTimeOrigin
from clipperx.render_runtime import RenderRuntime
from clipperx.source_decode import AudioDecodeLimits, DecodeLimits, Window
from clipperx.visual_contracts import (
    IdentityStatus,
    NormalizedRect,
    SourceState,
    SourceVisualSample,
    SourceVisualTimeline,
)
import clipperx.visual_timeline as visual_timeline
from clipperx.visual_timeline import (
    LocalFrameFeatures,
    OpenCVLocalVisualAnalyzer,
    _dense_trigger_times,
    _dense_windows,
    _timestamps_for_windows,
    compute_brightness,
    compute_histogram_change,
    compute_motion,
    compute_sharpness,
    compute_structural_change,
    build_source_visual_timeline,
    classify_source_state,
    detect_letterbox_bounds,
    detect_straight_edge_rectangles,
    detect_text_regions,
    segment_visual_states,
)


def _rect(x: float, y: float, width: float = 0.25, height: float = 0.30) -> NormalizedRect:
    return NormalizedRect(x=x, y=y, width=width, height=height)


def _sample(
    at: float,
    *,
    webcam: NormalizedRect | None = None,
    webcam_track_id: str | None = None,
    faces: tuple[str, ...] = (),
    letterbox: NormalizedRect | None = None,
    text_density: float = 0.0,
) -> SourceVisualSample:
    return SourceVisualSample(
        sample_id=f"sample_{int(at * 1000):010d}",
        at=at,
        histogram_change_score=0.0,
        structural_change_score=0.0,
        letterbox_bounds=letterbox,
        persistent_rectangles=[webcam] if webcam is not None else [],
        text_density=text_density,
        text_regions=[],
        face_track_ids=list(faces),
        webcam_track_id=webcam_track_id,
        webcam_rect=webcam,
        brightness=0.5,
        sharpness=20.0,
        motion=0.0,
        evidence_ids=[f"frame:{at:.3f}"],
    )

def _profile_payload(
    embedding: tuple[float, ...] | list[float],
    *,
    model_hashes: dict[str, str] | None = None,
    source_fingerprint: str = "source-fingerprint",
) -> dict[str, object]:
    return {
        "contract_version": 1,
        "profile_type": "streamer_face",
        "name": "Streamer",
        "embedding": list(embedding),
        "track_id": "face_001",
        "source_fingerprint": source_fingerprint,
        "source_reference": "source.mp4",
        "source_range": {"start": 0.0, "end": 5.0},
        "observations": 2,
        "model_hashes": model_hashes or {"yunet": "a" * 64, "sface": "b" * 64},
        "created_at": "2026-07-26T00:00:00Z",
    }

def test_camera_move_splits_source_state_near_event() -> None:
    samples = [
        _sample(0, webcam=_rect(0.00, 0.00), webcam_track_id="webcam_001", faces=("face_001",)),
        _sample(10, webcam=_rect(0.00, 0.00), webcam_track_id="webcam_001", faces=("face_001",)),
        _sample(20, webcam=_rect(0.72, 0.00), webcam_track_id="webcam_001", faces=("face_001",)),
        _sample(30, webcam=_rect(0.72, 0.00), webcam_track_id="webcam_001", faces=("face_001",)),
    ]

    spans = segment_visual_states(samples)

    assert [span.source.start for span in spans] == [0, 15]
    assert "webcam_moved" in spans[1].change_reasons


def test_person_inside_reacted_video_is_not_labeled_webcam() -> None:
    content = _rect(0.0, 0.1, 1.0, 0.8)
    samples = [
        _sample(0, faces=("face_guest",), letterbox=content),
        _sample(10, faces=("face_guest",), letterbox=content),
    ]

    spans = segment_visual_states(samples)

    assert all(span.state is not SourceState.CONTENT_WITH_CAMERA for span in spans)
    assert all(span.webcam_track_id is None for span in spans)


def test_face_without_rectangle_never_creates_confident_streamer_identity() -> None:
    sample = _sample(2, faces=("face_only",))

    state, confidence, reasons = classify_source_state(sample)
    span = segment_visual_states([sample], minimum_span_seconds=1.5)[0]

    assert state in {SourceState.CONTENT_ONLY, SourceState.UNKNOWN}
    assert 0.0 <= confidence <= 1.0
    assert "face_without_webcam_evidence" in reasons
    assert span.identity_status is IdentityStatus.UNAVAILABLE
    assert span.streamer_track_id is None
    assert span.source.start == 2
    assert span.source.end == 3.5


def test_exactly_one_face_inside_tracked_webcam_can_be_confident() -> None:
    webcam = _rect(0.70, 0.05)
    sample = _sample(
        0,
        webcam=webcam,
        webcam_track_id="webcam_001",
        faces=("face_streamer",),
    )

    state, confidence, reasons = classify_source_state(sample)
    span = segment_visual_states([sample])[0]

    assert state is SourceState.CONTENT_WITH_CAMERA
    assert confidence > 0.5
    assert "tracked_webcam" in reasons
    assert span.identity_status is IdentityStatus.CONFIDENT
    assert span.streamer_track_id == "face_streamer"


def test_multiple_faces_in_webcam_remain_ambiguous() -> None:
    webcam = _rect(0.70, 0.05)
    sample = _sample(
        0,
        webcam=webcam,
        webcam_track_id="webcam_001",
        faces=("face_a", "face_b"),
    )

    span = segment_visual_states([sample])[0]

    assert span.identity_status is IdentityStatus.AMBIGUOUS
    assert span.streamer_track_id is None


def test_dense_triggers_use_exact_visual_and_text_thresholds() -> None:
    below = _sample(0, text_density=0.0).model_copy(
        update={"histogram_change_score": 0.179999}
    )
    exact_visual = _sample(10, text_density=0.119999).model_copy(
        update={"histogram_change_score": 0.18}
    )
    exact_text_delta = _sample(20, text_density=0.239999).model_copy(
        update={"histogram_change_score": 0.0}
    )

    assert _dense_trigger_times([below], ()) == []
    assert _dense_trigger_times([below, exact_visual], ()) == [10]
    assert _dense_trigger_times([exact_visual, exact_text_delta], ()) == [20]


def test_dense_triggers_cover_webcam_face_letterbox_and_editorial_events() -> None:
    left = _sample(
        0,
        webcam=_rect(0.0, 0.0),
        webcam_track_id="webcam_001",
        faces=("face_001",),
    )
    exact_eight_percent = _sample(
        10,
        webcam=_rect(0.08, 0.0),
        webcam_track_id="webcam_001",
        faces=("face_001",),
    )
    moved = _sample(
        20,
        webcam=_rect(0.160001, 0.0),
        webcam_track_id="webcam_001",
        faces=("face_002",),
        letterbox=_rect(0.0, 0.1, 1.0, 0.8),
    )

    assert _dense_trigger_times([left, exact_eight_percent], ()) == []
    assert _dense_trigger_times([left, exact_eight_percent, moved], (7.25,)) == [
        7.25,
        20,
    ]


def test_dense_windows_expand_clamp_and_merge_before_sampling() -> None:
    windows = _dense_windows([1.0, 2.5, 9.0], duration_seconds=12.0)

    assert windows == [(0.0, 5.5), (6.0, 12.0)]


@pytest.mark.parametrize("event_at", [1.0, 5.0, 9.0])
def test_dense_windows_cover_entire_changed_sparse_bracket(event_at: float) -> None:
    previous = _sample(0.0)
    changed = _sample(10.0).model_copy(
        update={"histogram_change_score": 0.18}
    )

    brackets = visual_timeline._dense_change_brackets([previous, changed])
    windows = _dense_windows(
        _dense_trigger_times([previous, changed], ()),
        duration_seconds=20.0,
        change_brackets=brackets,
    )

    assert brackets == [(0.0, 10.0)]
    assert any(start <= event_at <= end for start, end in windows)
    assert windows == [(0.0, 13.0)]


def test_dense_timestamps_reuse_existing_sparse_decodes() -> None:
    timestamps = _timestamps_for_windows(
        [(0.0, 2.0), (5.0, 6.0)],
        interval=0.5,
        existing_timestamps={0.0, 1.0, 5.5},
    )

    assert timestamps == [0.5, 1.5, 2.0, 5.0, 6.0]


def test_dense_windows_use_one_sequential_decoder_per_merged_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSparseDecoder(_FakeDecoder):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeSequentialDecoder(_FakeDecoder):
        def __init__(self, path: Path, *, start_at: float) -> None:
            super().__init__()
            self.path = path
            self.start_at = start_at
            self.closed = False

        def close(self) -> None:
            self.closed = True

    sparse = FakeSparseDecoder()
    sequential: list[FakeSequentialDecoder] = []

    def create_sequential(path: Path, *, start_at: float) -> FakeSequentialDecoder:
        decoder = FakeSequentialDecoder(path, start_at=start_at)
        sequential.append(decoder)
        return decoder

    monkeypatch.setattr(visual_timeline, "_OpenCVFrameDecoder", FakeSparseDecoder)
    monkeypatch.setattr(
        visual_timeline,
        "_SequentialOpenCVFrameDecoder",
        create_sequential,
    )
    runtime = visual_timeline._BuilderRuntime(
        duration_seconds=20.0, decoder=sparse,
        local_analyzer=OpenCVLocalVisualAnalyzer(), face_backend=_NoFaces(),
        model_hashes={}, audio_extractor=None,
    )
    records: dict[float, visual_timeline._DecodedFrame] = {}

    visual_timeline._decode_dense_windows(
        Path("source.mp4"),
        runtime,
        [(2.0, 3.0), (12.0, 13.0)],
        interval=0.5,
        records=records,
    )

    assert sparse.closed is True
    assert [decoder.start_at for decoder in sequential] == [2.0, 12.0]
    assert [decoder.calls for decoder in sequential] == [
        [2.0, 2.5, 3.0],
        [12.0, 12.5, 13.0],
    ]
    assert all(decoder.closed for decoder in sequential)
    assert sorted(records) == [2.0, 2.5, 3.0, 12.0, 12.5, 13.0]


def test_generated_frames_produce_bounded_change_and_quality_features() -> None:
    blank = np.zeros((120, 200, 3), dtype=np.uint8)
    changed = blank.copy()
    cv2.rectangle(changed, (40, 25), (160, 95), (255, 255, 255), -1)

    assert compute_histogram_change(blank, blank) == pytest.approx(0.0)
    assert compute_structural_change(blank, blank) == pytest.approx(0.0)
    assert compute_motion(blank, blank) == pytest.approx(0.0)
    assert 0.0 < compute_histogram_change(blank, changed) <= 1.0
    assert 0.0 < compute_structural_change(blank, changed) <= 1.0
    assert 0.0 < compute_motion(blank, changed) <= 1.0
    assert compute_brightness(blank) == pytest.approx(0.0)
    assert compute_brightness(changed) > 0.0
    assert compute_sharpness(changed) > compute_sharpness(blank)


def test_letterbox_and_pillarbox_bounds_are_measured_in_source_coordinates() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[10:90, 20:180] = 180

    bounds = detect_letterbox_bounds(frame)

    assert bounds is not None
    assert bounds.x == pytest.approx(0.10, abs=0.015)
    assert bounds.y == pytest.approx(0.10, abs=0.015)
    assert bounds.width == pytest.approx(0.80, abs=0.02)
    assert bounds.height == pytest.approx(0.80, abs=0.02)
    assert detect_letterbox_bounds(np.full_like(frame, 100)) is None


def test_straight_edge_detector_finds_generated_overlay_rectangle() -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (190, 20), (310, 100), (255, 255, 255), 3)

    rectangles = detect_straight_edge_rectangles(frame)

    assert any(
        rect.x < 190 / 320 < rect.x + rect.width
        and rect.y < 20 / 180 < rect.y + rect.height
        and rect.width > 0.30
        and rect.height > 0.35
        for rect in rectangles
    )


def test_mser_text_features_return_density_and_normalized_boxes() -> None:
    frame = np.full((180, 420, 3), 255, dtype=np.uint8)
    cv2.putText(
        frame,
        "STREAM EN VIVO",
        (15, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    density, regions = detect_text_regions(frame)

    assert 0.0 < density <= 1.0
    assert regions
    assert all(
        0.0 <= region.x <= 1.0
        and 0.0 <= region.y <= 1.0
        and region.x + region.width <= 1.000001
        and region.y + region.height <= 1.000001
        for region in regions
    )


def test_local_analyzer_composes_all_pure_frame_features() -> None:
    previous = np.zeros((120, 200, 3), dtype=np.uint8)
    current = previous.copy()
    cv2.rectangle(current, (50, 25), (175, 100), (220, 220, 220), 2)
    cv2.putText(
        current,
        "LIVE",
        (65, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    features = OpenCVLocalVisualAnalyzer().analyze(current, previous_frame=previous)

    assert isinstance(features, LocalFrameFeatures)
    assert features.histogram_change_score > 0
    assert features.structural_change_score > 0
    assert features.persistent_rectangle_candidates
    assert features.text_density > 0
    assert features.text_regions
    assert features.brightness > 0
    assert features.sharpness > 0
    assert features.motion > 0


def test_persistent_rectangle_is_webcam_evidence_but_face_alone_is_not() -> None:
    rectangle = _rect(0.70, 0.05)
    with_rectangle = _sample(0, webcam=rectangle, faces=("face_streamer",))
    face_only = _sample(1, faces=("face_streamer",))

    rectangle_state, _, rectangle_reasons = classify_source_state(with_rectangle)
    face_state, _, face_reasons = classify_source_state(face_only)

    assert rectangle_state is SourceState.CONTENT_WITH_CAMERA
    assert "persistent_webcam_rectangle" in rectangle_reasons
    assert face_state is not SourceState.CONTENT_WITH_CAMERA
    assert "face_without_webcam_evidence" in face_reasons


def test_segment_spans_are_ordered_non_overlapping_and_merge_tiny_changes() -> None:
    left = _rect(0.0, 0.0)
    right = _rect(0.72, 0.0)
    samples = [
        _sample(0.0, webcam=left, webcam_track_id="webcam_001", faces=("face_001",)),
        _sample(0.5, webcam=right, webcam_track_id="webcam_001", faces=("face_001",)),
        _sample(3.0, webcam=right, webcam_track_id="webcam_001", faces=("face_001",)),
    ]

    spans = segment_visual_states(samples, minimum_span_seconds=1.0)

    assert all(span.source.end - span.source.start >= 1.0 for span in spans)
    assert all(
        left_span.source.end <= right_span.source.start
        for left_span, right_span in zip(spans, spans[1:])
    )
    assert [span.source.start for span in spans] == sorted(
        span.source.start for span in spans
    )


class _FakeDecoder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def decode_frame(self, at: float) -> np.ndarray:
        rounded = round(float(at), 6)
        self.calls.append(rounded)
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        frame[:, :, 0] = min(255, int(round(rounded * 10)))
        return frame

    def close(self) -> None:
        return None


class _NoFaces:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        del frame
        return []

    def cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        raise AssertionError("No face embeddings exist in this fake")


class _GeometryOnlyFaces:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        del frame
        return [
            LandmarkedFace(
                box=_rect(0.70, 0.05, 0.20, 0.25),
                landmarks=(
                    (0.74, 0.12),
                    (0.82, 0.12),
                    (0.78, 0.17),
                    (0.75, 0.23),
                    (0.81, 0.23),
                ),
                confidence=0.9,
                embedding=None,
            )
        ]


class _FaceOnlyAtStart:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        if int(frame[0, 0, 0]) != 0:
            return []
        return [
            LandmarkedFace(
                box=_rect(0.70, 0.05, 0.20, 0.25),
                landmarks=(
                    (0.74, 0.12),
                    (0.82, 0.12),
                    (0.78, 0.17),
                    (0.75, 0.23),
                    (0.81, 0.23),
                ),
                confidence=0.9,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]


class _PersistentRectangleAnalyzer:
    def analyze(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
    ) -> LocalFrameFeatures:
        del frame, previous_frame
        rectangle = _rect(0.65, 0.0, 0.35, 0.34)
        return LocalFrameFeatures(
            histogram_change_score=0.0,
            structural_change_score=0.0,
            letterbox_bounds=None,
            persistent_rectangle_candidates=(rectangle,),
            text_density=0.0,
            text_regions=(),
            brightness=0.5,
            sharpness=10.0,
            motion=0.0,
        )


def _install_fake_builder_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    face_backend: object | None = None,
    saved_profile_embedding: tuple[float, ...] | None = None,
    local_analyzer: object | None = None,
) -> tuple[Path, _FakeDecoder, list[bool]]:
    decoder = _FakeDecoder()
    video, _, calls = _install_observed_boundary(monkeypatch, tmp_path, duration=20.,
        pixel_decoder=decoder, face_backend=face_backend, local_analyzer=local_analyzer)
    if saved_profile_embedding is not None:
        job = JobStore.for_video(video)
        job.streamer_identity_path.write_text(json.dumps(_profile_payload(
            saved_profile_embedding, model_hashes={"yunet": "a" * 64, "sface": "b" * 64},
            source_fingerprint=job.fingerprint)), encoding="utf-8")
    return video, decoder, calls["models"]


def test_builder_uses_sparse_then_dense_without_duplicate_decodes_and_not_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, decoder, created = _install_fake_builder_runtime(monkeypatch, tmp_path)

    def legacy_failure(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("el builder v2 no puede usar selectores legacy")

    monkeypatch.setattr("clipperx.face.detect_webcam_region", legacy_failure)
    monkeypatch.setattr("clipperx.face._largest_box", legacy_failure)
    webcam_calls: list[int] = []
    real_associate = visual_timeline.associate_webcam_tracks

    def spy_associate(
        observations: Sequence[visual_timeline.WebcamRegionObservation],
        *, sample_timestamps: Sequence[float] = (), max_missed_samples: int = 3,
    ) -> list[visual_timeline.WebcamRectangleTrack]:
        webcam_calls.append(len(observations))
        return real_associate(
            observations, sample_timestamps=sample_timestamps,
            max_missed_samples=max_missed_samples,
        )

    monkeypatch.setattr(visual_timeline, "associate_webcam_tracks", spy_associate)

    output = build_source_visual_timeline(
        video,
        editorial_boundaries=(5.0,),
        sparse_interval=10.0,
        dense_interval=0.5,
    )
    timeline = SourceVisualTimeline.model_validate_json(output.read_text(encoding="utf-8"))

    assert created == [True]
    assert webcam_calls
    assert len(decoder.calls) == len(set(decoder.calls))
    assert timeline.samples == sorted(timeline.samples, key=lambda sample: sample.at)
    assert timeline.spans[0].source.start == 0.0
    assert timeline.spans[-1].source.end == timeline.duration_seconds == 20.0
    assert all(
        left.source.end == right.source.start
        for left, right in zip(timeline.spans, timeline.spans[1:])
    )


def test_builder_cache_hit_skips_runtime_and_boundary_change_invalidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, created = _install_fake_builder_runtime(monkeypatch, tmp_path)
    first = build_source_visual_timeline(video, editorial_boundaries=(5.0,))
    assert created == [True]

    monkeypatch.setattr(
        visual_timeline,
        "_create_observed_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache hit no debe cargar modelos ni decoder")
        ),
    )
    assert build_source_visual_timeline(video, editorial_boundaries=(5.0,)) == first

    with pytest.raises(ValueError, match="config"):
        build_source_visual_timeline(video, editorial_boundaries=(6.0,))


def test_builder_rejects_tampered_generation_instead_of_implicit_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, created = _install_fake_builder_runtime(monkeypatch, tmp_path)
    output = build_source_visual_timeline(video)
    previous = json.loads(output.read_text(encoding="utf-8"))
    previous["analyzer_version"] = "source-visual-local-v4"
    output.write_text(json.dumps(previous), encoding="utf-8")

    with pytest.raises(ValueError, match="Catalog"):
        build_source_visual_timeline(video)
    assert created == [True]


def test_geometry_only_faces_do_not_create_unserializable_track_references(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=_GeometryOnlyFaces(),
    )

    output = build_source_visual_timeline(video, sparse_interval=10.0)
    timeline = SourceVisualTimeline.model_validate_json(output.read_text(encoding="utf-8"))

    assert timeline.face_tracks == []
    assert all(sample.face_track_ids == [] for sample in timeline.samples)
    assert all(span.streamer_track_id is None for span in timeline.spans)


def test_persistent_webcam_rectangle_survives_temporary_face_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=_FaceOnlyAtStart(),
        local_analyzer=_PersistentRectangleAnalyzer(),
        saved_profile_embedding=(1.0, 0.0),
    )

    output = build_source_visual_timeline(video, sparse_interval=10.0)
    timeline = SourceVisualTimeline.model_validate_json(output.read_text(encoding="utf-8"))

    assert len(timeline.webcam_tracks) == 1
    webcam = timeline.webcam_tracks[0]
    assert webcam.face_track_ids == ["face_001"]
    assert len(webcam.observations) >= 2
    assert any(
        sample.webcam_track_id == webcam.track_id and not sample.face_track_ids
        for sample in timeline.samples
    )


def test_wrong_saved_profile_does_not_promote_picture_in_picture_webcam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=_LeftCornerActorFaces(),
        local_analyzer=_CornerReactedPlayerAnalyzer(),
        saved_profile_embedding=(0.0, 1.0),
    )

    output = build_source_visual_timeline(video, sparse_interval=10.0)
    timeline = SourceVisualTimeline.model_validate_json(output.read_text(encoding="utf-8"))

    assert timeline.webcam_tracks == []
    assert all(sample.webcam_track_id is None for sample in timeline.samples)


def _embedded_face(vector: tuple[float, float]) -> LandmarkedFace:
    return LandmarkedFace(
        box=_rect(0.70, 0.05, 0.20, 0.25),
        landmarks=(
            (0.74, 0.12),
            (0.82, 0.12),
            (0.78, 0.17),
            (0.75, 0.23),
            (0.81, 0.23),
        ),
        confidence=0.9,
        embedding=np.array(vector, dtype=np.float32),
    )


class _StableEmbeddedFaces:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        del frame
        return [_embedded_face((1.0, 0.0))]


class _BriefFaceAtTenSeconds:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        return [_embedded_face((1.0, 0.0))] if int(frame[0, 0, 0]) == 100 else []


class _SwitchingFaces:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        if int(frame[0, 0, 0]) < 150:
            return [_embedded_face((1.0, 0.0))]
        return [_embedded_face((0.0, 1.0))]


class _CenteredReactedPlayerAnalyzer:
    def analyze(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
    ) -> LocalFrameFeatures:
        del frame, previous_frame
        player = _rect(0.10, 0.10, 0.80, 0.80)
        return LocalFrameFeatures(
            histogram_change_score=0.0,
            structural_change_score=0.0,
            letterbox_bounds=None,
            persistent_rectangle_candidates=(player,),
            text_density=0.0,
            text_regions=(),
            brightness=0.5,
            sharpness=10.0,
            motion=0.0,
        )


class _BriefCameraAnalyzer:
    def analyze(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
    ) -> LocalFrameFeatures:
        del previous_frame
        rectangles = (
            (_rect(0.65, 0.0, 0.35, 0.34),)
            if int(frame[0, 0, 0]) == 100
            else ()
        )
        return LocalFrameFeatures(
            histogram_change_score=0.0,
            structural_change_score=0.0,
            letterbox_bounds=None,
            persistent_rectangle_candidates=rectangles,
            text_density=0.0,
            text_regions=(),
            brightness=0.5,
            sharpness=10.0,
            motion=0.0,
        )


def test_builder_retains_only_compact_descriptors_instead_of_source_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(monkeypatch, tmp_path)

    build_source_visual_timeline(video, editorial_boundaries=(5.0,))

    fields = visual_timeline._DecodedFrame.__dataclass_fields__
    assert "gray_thumbnail" in fields
    assert "histogram" in fields
    assert "independent_features" in fields
    assert "frame" not in visual_timeline._DecodedFrame.__dataclass_fields__


def test_centered_reacted_player_with_actor_is_not_promoted_to_webcam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=_StableEmbeddedFaces(),
        local_analyzer=_CenteredReactedPlayerAnalyzer(),
    )

    output = build_source_visual_timeline(video)
    timeline = SourceVisualTimeline.model_validate_json(output.read_text(encoding="utf-8"))

    assert timeline.webcam_tracks == []
    assert all(sample.webcam_track_id is None for sample in timeline.samples)
    assert all(span.state is not SourceState.CONTENT_WITH_CAMERA for span in timeline.spans)


def test_yunet_geometry_is_preserved_without_claiming_identity_or_webcam_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=_GeometryOnlyFaces(),
        local_analyzer=_PersistentRectangleAnalyzer(),
    )

    output = build_source_visual_timeline(video)
    timeline = SourceVisualTimeline.model_validate_json(output.read_text(encoding="utf-8"))

    assert timeline.face_tracks == []
    assert timeline.webcam_tracks == []
    assert any(sample.rectangle_candidates for sample in timeline.samples)
    assert any(sample.persistent_rectangles for sample in timeline.samples)
    assert all(sample.webcam_track_id is None for sample in timeline.samples)
    assert all(span.identity_status is IdentityStatus.UNAVAILABLE for span in timeline.spans)


def test_single_sparse_camera_observation_still_triggers_dense_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, decoder, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=_BriefFaceAtTenSeconds(),
        local_analyzer=_BriefCameraAnalyzer(),
    )

    build_source_visual_timeline(video, sparse_interval=10.0, dense_interval=0.5)

    assert 9.5 in decoder.calls
    assert 10.5 in decoder.calls
    assert len(decoder.calls) == len(set(decoder.calls))


def test_cache_invalidates_when_transcript_artifact_appears(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(monkeypatch, tmp_path)
    build_source_visual_timeline(video)
    JobStore.for_video(video).transcript_path.write_text(
        '{"segments":[{"start":0,"end":1,"text":"secreto"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        visual_timeline,
        "_create_builder_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("el transcript nuevo debe invalidar la caché")
        ),
    )

    with pytest.raises(ValueError, match="config"):
        build_source_visual_timeline(video)


def test_streamer_identity_is_rechecked_inside_each_webcam_span(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=_SwitchingFaces(),
        local_analyzer=_PersistentRectangleAnalyzer(),
        saved_profile_embedding=(1.0, 0.0),
    )

    output = build_source_visual_timeline(video, sparse_interval=10.0)
    timeline = SourceVisualTimeline.model_validate_json(output.read_text(encoding="utf-8"))
    guest_sample = next(
        sample for sample in reversed(timeline.samples) if sample.face_track_ids == ["face_002"]
    )
    guest_span = next(
        span
        for span in timeline.spans
        if span.source.start <= guest_sample.at <= span.source.end
    )

    assert any(span.streamer_track_id == "face_001" for span in timeline.spans)
    assert guest_span.streamer_track_id is None
    assert guest_span.identity_status is IdentityStatus.UNAVAILABLE


def test_small_letterbox_jitter_does_not_trigger_dense_window() -> None:
    first = _sample(0.0, letterbox=_rect(0.0, 0.10, 1.0, 0.80))
    jittered = _sample(10.0, letterbox=_rect(0.0, 0.102, 1.0, 0.798))

    assert _dense_trigger_times([first, jittered], ()) == []


class _LeftCornerActorFaces:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        del frame
        return [
            LandmarkedFace(
                box=_rect(0.05, 0.05, 0.18, 0.22),
                landmarks=(
                    (0.08, 0.10),
                    (0.16, 0.10),
                    (0.12, 0.15),
                    (0.09, 0.21),
                    (0.16, 0.21),
                ),
                confidence=0.9,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]


class _CornerReactedPlayerAnalyzer:
    def analyze(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
    ) -> LocalFrameFeatures:
        del frame, previous_frame
        player = _rect(0.0, 0.0, 0.30, 0.30)
        return LocalFrameFeatures(
            histogram_change_score=0.0,
            structural_change_score=0.0,
            letterbox_bounds=None,
            persistent_rectangle_candidates=(player,),
            text_density=0.0,
            text_regions=(),
            brightness=0.5,
            sharpness=10.0,
            motion=0.0,
        )


def test_corner_reacted_player_is_not_promoted_to_webcam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=_LeftCornerActorFaces(),
        local_analyzer=_CornerReactedPlayerAnalyzer(),
    )

    output = build_source_visual_timeline(video)
    timeline = SourceVisualTimeline.model_validate_json(
        output.read_text(encoding="utf-8")
    )

    assert timeline.webcam_tracks == []
    assert all(sample.webcam_track_id is None for sample in timeline.samples)


class _ExternalStreamerWithGuestInWebcam:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        streamer = LandmarkedFace(
            box=_rect(0.30, 0.45, 0.18, 0.25),
            landmarks=(
                (0.33, 0.50),
                (0.40, 0.50),
                (0.36, 0.56),
                (0.34, 0.62),
                (0.40, 0.62),
            ),
            confidence=0.95,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        if int(frame[0, 0, 0]) < 150:
            return [streamer]
        guest = _embedded_face((0.0, 1.0))
        return [streamer, guest]


def test_streamer_visible_outside_webcam_cannot_claim_guest_webcam_span(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=_ExternalStreamerWithGuestInWebcam(),
        local_analyzer=_PersistentRectangleAnalyzer(),
        saved_profile_embedding=(1.0, 0.0),
    )

    output = build_source_visual_timeline(video, sparse_interval=10.0)
    timeline = SourceVisualTimeline.model_validate_json(
        output.read_text(encoding="utf-8")
    )
    guest_sample = next(
        sample
        for sample in reversed(timeline.samples)
        if "face_002" in sample.face_track_ids
    )
    guest_span = next(
        span
        for span in timeline.spans
        if span.source.start <= guest_sample.at <= span.source.end
    )

    assert guest_span.streamer_track_id is None
    assert guest_span.identity_status is not IdentityStatus.CONFIDENT


class _LargeStreamerFaces:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        del frame
        return [
            LandmarkedFace(
                box=_rect(0.25, 0.15, 0.45, 0.55),
                landmarks=(
                    (0.35, 0.30),
                    (0.58, 0.30),
                    (0.47, 0.42),
                    (0.38, 0.57),
                    (0.57, 0.57),
                ),
                confidence=0.95,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]


class _NoRectangleAnalyzer:
    def analyze(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
    ) -> LocalFrameFeatures:
        del frame, previous_frame
        return LocalFrameFeatures(
            histogram_change_score=0.0,
            structural_change_score=0.0,
            letterbox_bounds=None,
            persistent_rectangle_candidates=(),
            text_density=0.0,
            text_regions=(),
            brightness=0.5,
            sharpness=10.0,
            motion=0.0,
        )


def test_saved_streamer_profile_enables_full_camera_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=_LargeStreamerFaces(),
        local_analyzer=_NoRectangleAnalyzer(),
        saved_profile_embedding=(1.0, 0.0),
    )

    output = build_source_visual_timeline(video)
    timeline = SourceVisualTimeline.model_validate_json(
        output.read_text(encoding="utf-8")
    )

    assert any(span.state is SourceState.FULL_CAMERA for span in timeline.spans)
    assert any(
        span.state is SourceState.FULL_CAMERA
        and span.streamer_track_id == "face_001"
        and span.identity_status is IdentityStatus.CONFIDENT
        for span in timeline.spans
    )


def test_profile_embedding_rejects_wrong_contract_type_and_model_hashes(
    tmp_path: Path,
) -> None:
    expected = {"yunet": "a" * 64, "sface": "b" * 64}
    profile = tmp_path / "streamer_identity.json"

    wrong_contract = _profile_payload([1.0, 0.0], model_hashes=expected)
    wrong_contract["profile_type"] = "speaker_voice"
    profile.write_text(json.dumps(wrong_contract), encoding="utf-8")
    assert visual_timeline._load_profile_embedding(profile, expected) is None

    wrong_models = _profile_payload(
        [1.0, 0.0],
        model_hashes={"yunet": "c" * 64, "sface": "d" * 64},
    )
    profile.write_text(json.dumps(wrong_models), encoding="utf-8")
    assert visual_timeline._load_profile_embedding(profile, expected) is None

    string_observations = _profile_payload([1.0, 0.0], model_hashes=expected)
    string_observations["observations"] = "3"
    profile.write_text(json.dumps(string_observations), encoding="utf-8")
    assert visual_timeline._load_profile_embedding(profile, expected) is None

    extra_field = _profile_payload([1.0, 0.0], model_hashes=expected)
    extra_field["untrusted"] = True
    profile.write_text(json.dumps(extra_field), encoding="utf-8")
    assert visual_timeline._load_profile_embedding(profile, expected) is None

    invalid_timestamp = _profile_payload([1.0, 0.0], model_hashes=expected)
    invalid_timestamp["created_at"] = "not-a-timestamp"
    profile.write_text(json.dumps(invalid_timestamp), encoding="utf-8")
    assert visual_timeline._load_profile_embedding(profile, expected) is None

    wrong_source = _profile_payload(
        [1.0, 0.0],
        model_hashes=expected,
        source_fingerprint="wrong-source",
    )
    profile.write_text(json.dumps(wrong_source), encoding="utf-8")
    assert visual_timeline._load_profile_embedding(
        profile,
        expected,
        expected_source_fingerprint="expected-source",
    ) is None


    valid = _profile_payload([3.0, 4.0], model_hashes=expected)
    profile.write_text(json.dumps(valid), encoding="utf-8")
    embedding = visual_timeline._load_profile_embedding(profile, expected)

    assert embedding is not None
    assert np.allclose(embedding, np.array([0.6, 0.8], dtype=np.float32))


def test_saved_profile_prefers_job_copy_and_falls_back_to_global(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, _ = _install_fake_builder_runtime(monkeypatch, tmp_path)
    model_hashes = {"yunet": "a" * 64, "sface": "b" * 64}
    profiles_dir = tmp_path / "profiles"
    monkeypatch.setattr(visual_timeline.paths, "profiles_dir", profiles_dir)
    profiles_dir.mkdir(parents=True)
    global_profile = profiles_dir / "streamer_identity.json"
    global_profile.write_text(
        json.dumps(_profile_payload([0.0, 5.0], model_hashes=model_hashes)),
        encoding="utf-8",
    )
    job = visual_timeline.JobStore.for_video(video)

    fallback = visual_timeline._load_saved_profile_embedding(job, model_hashes)
    assert fallback is not None
    assert np.allclose(fallback, np.array([0.0, 1.0], dtype=np.float32))

    wrong_job_payload = _profile_payload(
        [5.0, 0.0],
        model_hashes=model_hashes,
        source_fingerprint="not-the-job-fingerprint",
    )
    job.streamer_identity_path.write_text(
        json.dumps(wrong_job_payload),
        encoding="utf-8",
    )
    rejected_job = visual_timeline._load_saved_profile_embedding(job, model_hashes)
    assert rejected_job is not None
    assert np.allclose(rejected_job, np.array([0.0, 1.0], dtype=np.float32))

    job.streamer_identity_path.write_text(
        json.dumps(
            _profile_payload(
                [3.0, 4.0],
                model_hashes=model_hashes,
                source_fingerprint=job.fingerprint,
            )
        ),
        encoding="utf-8",
    )
    preferred = visual_timeline._load_saved_profile_embedding(job, model_hashes)
    assert preferred is not None
    assert np.allclose(preferred, np.array([0.6, 0.8], dtype=np.float32))


def test_saved_job_profile_loads_across_different_global_profile_drive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if tmp_path.anchor.lower() == visual_timeline.paths.profiles_dir.anchor.lower():
        pytest.skip("requires job and global profile paths on different volumes")
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch, tmp_path, saved_profile_embedding=(3.0, 4.0)
    )
    job = visual_timeline.JobStore.for_video(video)
    expected = {"yunet": "a" * 64, "sface": "b" * 64}

    embedding = visual_timeline._load_saved_profile_embedding(job, expected)

    assert embedding is not None
    assert np.allclose(embedding, np.array([0.6, 0.8], dtype=np.float32))


def test_global_profile_change_invalidates_canonical_timeline_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profiles_dir = tmp_path / "profiles"
    monkeypatch.setattr(visual_timeline.paths, "profiles_dir", profiles_dir)
    video, _, _ = _install_fake_builder_runtime(monkeypatch, tmp_path)
    build_source_visual_timeline(video)
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "streamer_identity.json").write_text(
        json.dumps({"embedding": [1.0, 0.0]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        visual_timeline,
        "_create_builder_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global profile must invalidate the canonical cache")
        ),
    )

    with pytest.raises(ValueError, match="config"):
        build_source_visual_timeline(video)


class _SmallCenteredStreamerFaces:
    def __init__(self, embedding: tuple[float, float] = (1.0, 0.0)) -> None:
        self.embedding = embedding

    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        del frame
        return [
            LandmarkedFace(
                box=_rect(0.46, 0.44, 0.08, 0.08),
                landmarks=(
                    (0.478, 0.465),
                    (0.522, 0.465),
                    (0.500, 0.485),
                    (0.482, 0.505),
                    (0.518, 0.505),
                ),
                confidence=0.95,
                embedding=np.array(self.embedding, dtype=np.float32),
            )
        ]


class _MediumCenteredStreamerFaces:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        del frame
        return [
            LandmarkedFace(
                box=_rect(0.42, 0.30, 0.22, 0.24),
                landmarks=(
                    (0.47, 0.37),
                    (0.59, 0.37),
                    (0.53, 0.42),
                    (0.49, 0.48),
                    (0.57, 0.48),
                ),
                confidence=0.95,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]


class _SmallCornerStreamerFaces:
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        del frame
        return [
            LandmarkedFace(
                box=_rect(0.02, 0.02, 0.08, 0.08),
                landmarks=(
                    (0.038, 0.045),
                    (0.082, 0.045),
                    (0.060, 0.065),
                    (0.042, 0.085),
                    (0.078, 0.085),
                ),
                confidence=0.95,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]


class _HighChatWithoutWebcamRectangleAnalyzer:
    def analyze(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
    ) -> LocalFrameFeatures:
        del frame, previous_frame
        return LocalFrameFeatures(
            histogram_change_score=0.0,
            structural_change_score=0.0,
            letterbox_bounds=None,
            persistent_rectangle_candidates=(),
            text_density=0.75,
            text_regions=(_rect(0.76, 0.05, 0.20, 0.80),),
            brightness=0.5,
            sharpness=10.0,
            motion=0.0,
        )


@pytest.mark.parametrize(
    ("face_backend", "profile", "expected_full_camera", "expected_webcam"),
    [
        (_SmallCenteredStreamerFaces(), (1.0, 0.0), True, False),
        (_MediumCenteredStreamerFaces(), (1.0, 0.0), True, False),
        (_SmallCenteredStreamerFaces(), (0.0, 1.0), False, False),
        (_SmallCornerStreamerFaces(), (1.0, 0.0), False, True),
    ],
)
def test_small_profile_face_requires_match_and_center_for_full_camera(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    face_backend: object,
    profile: tuple[float, float],
    expected_full_camera: bool,
    expected_webcam: bool,
) -> None:
    video, _, _ = _install_fake_builder_runtime(
        monkeypatch,
        tmp_path,
        face_backend=face_backend,
        local_analyzer=_HighChatWithoutWebcamRectangleAnalyzer(),
        saved_profile_embedding=profile,
    )

    output = build_source_visual_timeline(video, sparse_interval=5.0)
    timeline = SourceVisualTimeline.model_validate_json(
        output.read_text(encoding="utf-8")
    )
    full_camera_spans = [
        span for span in timeline.spans if span.state is SourceState.FULL_CAMERA
    ]

    assert bool(full_camera_spans) is expected_full_camera
    assert bool(timeline.webcam_tracks) is expected_webcam
    assert any(
        span.state is SourceState.CONTENT_WITH_CAMERA
        for span in timeline.spans
    ) is expected_webcam
    if expected_webcam:
        assert all("face_001" in track.face_track_ids for track in timeline.webcam_tracks)
    assert all(sample.letterbox_bounds is None for sample in timeline.samples)
    if expected_full_camera:
        assert all(sample.webcam_rect is None for sample in timeline.samples)
        assert any(sample.text_density >= 0.75 for sample in timeline.samples)


def test_profile_matching_tracks_merge_only_when_they_do_not_overlap() -> None:
    def track(
        track_id: str,
        times: tuple[float, ...],
        embedding: tuple[float, float],
    ) -> FaceTrack:
        face = LandmarkedFace(
            box=_rect(0.1, 0.1, 0.08, 0.12),
            landmarks=(
                (0.118, 0.13),
                (0.162, 0.13),
                (0.140, 0.16),
                (0.122, 0.19),
                (0.158, 0.19),
            ),
            confidence=0.95,
            embedding=np.array(embedding, dtype=np.float32),
        )
        return visual_timeline.FaceTrack(
            track_id=track_id,
            observations=[
                visual_timeline.FaceObservation(
                    at=at,
                    face=face,
                    source_state=SourceState.UNKNOWN,
                    evidence_id=f"{track_id}:{at:.1f}",
                )
                for at in times
            ],
        )

    webcam = track("face_001", (0.0, 1.0), (0.8, 0.6))
    full_camera = track("face_002", (2.0, 3.0), (0.82, 0.58))
    guest = track("face_003", (1.0, 2.0), (0.0, 1.0))

    merged = visual_timeline._merge_profile_matching_face_tracks(
        [webcam, full_camera, guest],
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert [item.track_id for item in merged] == ["face_001", "face_003"]
    streamer = merged[0]
    assert [item.at for item in streamer.observations] == [0.0, 1.0, 2.0, 3.0]

    simultaneous = track("face_004", (1.0, 1.5), (0.81, 0.59))
    unsafe = visual_timeline._merge_profile_matching_face_tracks(
        [webcam, simultaneous],
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert [item.track_id for item in unsafe] == ["face_001", "face_004"]

    boundary_conflict = track("face_005", (3.0, 4.0), (0.81, 0.59))
    later = track("face_006", (4.0, 5.0), (0.83, 0.57))
    recovered = visual_timeline._merge_profile_matching_face_tracks(
        [webcam, full_camera, boundary_conflict, later, guest],
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert [item.track_id for item in recovered] == [
        "face_001",
        "face_005",
        "face_003",
    ]
    assert [item.at for item in recovered[0].observations] == [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0
    ]

    long_before = track(
        "face_007", tuple(float(value) for value in range(11)), (0.84, 0.56)
    )
    long_after = track(
        "face_008", tuple(float(value) for value in range(11, 21)), (0.85, 0.55)
    )
    scene_cut_transition = visual_timeline._merge_profile_matching_face_tracks(
        [long_before, long_after],
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert len(scene_cut_transition) == 1
    assert len(scene_cut_transition[0].observations) == 21


@pytest.mark.parametrize("long_overlap", [False, True])
def test_profile_track_merges_preserve_simultaneous_faces_and_all_evidence(
    long_overlap: bool,
) -> None:
    face = LandmarkedFace(
        box=_rect(0.1, 0.1, 0.08, 0.12),
        landmarks=((0.12, 0.13), (0.16, 0.13), (0.14, 0.16),
                   (0.12, 0.19), (0.16, 0.19)),
        confidence=0.95,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    def track(track_id: str, times: tuple[float, ...]) -> visual_timeline.FaceTrack:
        return visual_timeline.FaceTrack(
            track_id=track_id,
            observations=[
                visual_timeline.FaceObservation(
                    at=at, face=face, source_state=SourceState.UNKNOWN,
                    evidence_id=f"{track_id}:{at:.6f}",
                )
                for at in times
            ],
        )

    if long_overlap:
        tracks = [
            track("face_001", tuple(float(at) for at in range(11))),
            track("face_002", tuple(float(at) for at in range(10, 21))),
        ]
        expected_ids = ["face_001", "face_002"]
    else:
        tracks = [
            track("face_001", (0.0, 1.0)),
            track("face_002", (0.0, 1.0)),
            track("face_003", (2.0, 3.0)),
            track("face_004", (2.0, 3.0)),
        ]
        expected_ids = ["face_001", "face_002"]
    original_observations = [item for candidate in tracks for item in candidate.observations]

    merged = visual_timeline._merge_profile_matching_face_tracks(
        tracks, np.array([1.0, 0.0], dtype=np.float32)
    )

    assert [candidate.track_id for candidate in merged] == expected_ids
    assert sorted(id(item) for candidate in merged for item in candidate.observations) == (
        sorted(id(item) for item in original_observations)
    )
    for candidate in merged:
        times = [round(item.at, 6) for item in candidate.observations]
        assert len(times) == len(set(times))


def test_profile_anchored_webcam_holds_one_short_face_dropout() -> None:
    face = LandmarkedFace(
        box=_rect(0.09, 0.08, 0.06, 0.13),
        landmarks=(
            (0.103, 0.12),
            (0.137, 0.12),
            (0.120, 0.15),
            (0.107, 0.18),
            (0.133, 0.18),
        ),
        confidence=0.95,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    streamer = visual_timeline.FaceTrack(
        track_id="face_001",
        observations=[
            visual_timeline.FaceObservation(
                at=at,
                face=face,
                source_state=SourceState.UNKNOWN,
                evidence_id=f"face:{at:.1f}",
            )
            for at in (0.0, 1.0, 3.0)
        ],
    )
    identity = visual_timeline.StreamerIdentityResult(
        IdentityStatus.CONFIDENT,
        "face_001",
        1.0,
        {"face_001": 1.0},
    )

    webcams = visual_timeline._profile_anchored_webcam_tracks(
        [streamer],
        identity=identity,
        saved_profile_embedding=np.array([1.0, 0.0], dtype=np.float32),
        sample_timestamps=(0.0, 1.0, 2.0, 3.0),
    )

    assert len(webcams) == 1
    webcam = webcams[0]
    assert [item.at for item in webcam.observations] == [0.0, 1.0, 2.0, 3.0]
    held = next(item for item in webcam.observations if item.at == 2.0)
    assert held.confidence < 0.84
    assert held.evidence_id == "profile-webcam-hold:2.000000"


def test_profile_anchored_webcam_never_edge_holds_from_future_only() -> None:
    face = LandmarkedFace(
        box=_rect(0.09, 0.08, 0.06, 0.13),
        landmarks=(
            (0.103, 0.12),
            (0.137, 0.12),
            (0.120, 0.15),
            (0.107, 0.18),
            (0.133, 0.18),
        ),
        confidence=0.95,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    streamer = visual_timeline.FaceTrack(
        track_id="face_001",
        observations=[
            visual_timeline.FaceObservation(
                at=at,
                face=face,
                source_state=SourceState.UNKNOWN,
                evidence_id=f"face:{at:.1f}",
            )
            for at in (1.0, 2.0)
        ],
    )
    identity = visual_timeline.StreamerIdentityResult(
        IdentityStatus.CONFIDENT,
        "face_001",
        1.0,
        {"face_001": 1.0},
    )

    webcams = visual_timeline._profile_anchored_webcam_tracks(
        [streamer],
        identity=identity,
        saved_profile_embedding=np.array([1.0, 0.0], dtype=np.float32),
        sample_timestamps=(0.5, 1.0, 2.0),
    )

    assert len(webcams) == 1
    assert [item.at for item in webcams[0].observations] == [1.0, 2.0]


def test_profile_anchored_webcam_splits_long_absence_into_distinct_tracks() -> None:
    face = LandmarkedFace(
        box=_rect(0.09, 0.08, 0.06, 0.13),
        landmarks=((0.103, 0.12), (0.137, 0.12), (0.120, 0.15), (0.107, 0.18), (0.133, 0.18)),
        confidence=0.95,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    streamer = visual_timeline.FaceTrack(
        track_id="face_001",
        observations=[
            visual_timeline.FaceObservation(
                at=at,
                face=face,
                source_state=SourceState.UNKNOWN,
                evidence_id=f"face:{at:.1f}",
            )
            for at in (0.0, 1.0, 7200.0, 7201.0)
        ],
    )
    identity = visual_timeline.StreamerIdentityResult(
        IdentityStatus.CONFIDENT,
        "face_001",
        1.0,
        {"face_001": 1.0},
    )

    tracks = visual_timeline._profile_anchored_webcam_tracks(
        [streamer],
        identity=identity,
        saved_profile_embedding=np.array([1.0, 0.0], dtype=np.float32),
        sample_timestamps=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7200.0, 7201.0),
    )

    assert len(tracks) == 2
    assert [[item.at for item in track.observations] for track in tracks] == [
        [0.0, 1.0, 2.0],
        [7200.0, 7201.0],
    ]
    contracts = [track.to_contract() for track in tracks]
    assert contracts[0].source_ranges[0].end < contracts[1].source_ranges[0].start


def test_profile_verified_full_camera_bridges_short_face_detector_dropouts() -> None:
    face = LandmarkedFace(
        box=_rect(0.46, 0.44, 0.08, 0.08),
        landmarks=(
            (0.478, 0.465),
            (0.522, 0.465),
            (0.500, 0.485),
            (0.482, 0.505),
            (0.518, 0.505),
        ),
        confidence=0.95,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    seed_times = (600.0, 601.0, 607.0, 609.0, 612.0, 616.0, 618.0, 619.0)
    streamer = visual_timeline.FaceTrack(
        track_id="face_001",
        observations=[
            visual_timeline.FaceObservation(
                at=at,
                face=face,
                source_state=SourceState.UNKNOWN,
                evidence_id=f"face:{at:.1f}",
            )
            for at in seed_times
        ],
    )
    timestamps = (
        600.0,
        599.0,
        601.0,
        608.0,
        607.0,
        609.0,
        610.0,
        611.0,
        612.0,
        617.0,
        618.0,
        619.0,
        619.75,
        620.0,
    )
    local_features = {
        at: LocalFrameFeatures(
            histogram_change_score=0.20 if at == 620.0 else 0.03,
            structural_change_score=0.20 if at == 620.0 else 0.06,
            letterbox_bounds=None,
            persistent_rectangle_candidates=(),
            text_density=0.75,
            text_regions=(_rect(0.76, 0.05, 0.20, 0.80),),
            brightness=0.5,
            sharpness=10.0,
            motion=0.02,
        )
        for at in timestamps
    }
    identity = visual_timeline.StreamerIdentityResult(
        IdentityStatus.CONFIDENT,
        "face_001",
        1.0,
        {"face_001": 1.0},
    )

    inferred = visual_timeline._full_camera_times(
        face_tracks=[streamer],
        identity=identity,
        verified_webcams=[],
        local_features=local_features,
        saved_profile_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert {608.0, 610.0, 611.0, 619.75}.issubset(inferred)
    assert {599.0, 617.0, 620.0}.isdisjoint(inferred)
    state, _, _ = classify_source_state(
        _sample(610.0, faces=()).model_copy(
            update={"evidence_ids": ["frame:610.000000", "inferred_full_camera"]}
        )
    )
    assert state is SourceState.FULL_CAMERA


@pytest.fixture
def reverse_dissolve_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> tuple[
    dict[float, visual_timeline._DecodedFrame],
    visual_timeline._BuilderRuntime,
    visual_timeline._ReadOnlyVisualContext,
]:
    def face(box: NormalizedRect) -> LandmarkedFace:
        return LandmarkedFace(
            box=box,
            landmarks=tuple((box.x + box.width * x, box.y + box.height * y)
                            for x, y in ((.3, .3), (.7, .3), (.5, .5), (.3, .7), (.7, .7))),
            confidence=.95, embedding=np.array([1.0, 0.0], dtype=np.float32),
        )

    corner = face(_rect(.09, .08, .06, .13))
    centered = face(_rect(.45, .30, .14, .35))
    features = LocalFrameFeatures(
        histogram_change_score=0, structural_change_score=0,
        letterbox_bounds=None, persistent_rectangle_candidates=(),
        text_density=0, text_regions=(), brightness=.5, sharpness=10, motion=0,
    )
    records: dict[float, visual_timeline._DecodedFrame] = {}
    for at in range(10):
        faces = ([corner] if at < 6 else [corner, centered] if at == 6 else [centered])
        records[float(at)] = visual_timeline._DecodedFrame(
            at=float(at), independent_features=features,
            gray_thumbnail=np.zeros((54, 96), dtype=np.uint8),
            histogram=np.ones(64, dtype=np.float32) / 64, faces=faces,
        )
    profile = np.array([1.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(visual_timeline, "_load_saved_profile_embedding", lambda *args: profile)
    runtime = visual_timeline._BuilderRuntime(
        10., _FakeDecoder(), OpenCVLocalVisualAnalyzer(), _NoFaces(), {}, None,
    )
    job = visual_timeline._ReadOnlyVisualContext(
        "fixture-source", tmp_path / "transcript.json", tmp_path / "diarization.json",
        tmp_path / "streamer-identity.json",
    )
    return records, runtime, job


@pytest.mark.parametrize("check_webcam_hold", [False, True])
def test_saved_profile_full_camera_after_reverse_dissolve_without_merging_faces(
    reverse_dissolve_records: tuple[dict[float, visual_timeline._DecodedFrame], visual_timeline._BuilderRuntime, visual_timeline._ReadOnlyVisualContext],
    check_webcam_hold: bool,
) -> None:
    records, runtime, job = reverse_dissolve_records
    samples, tracks, webcams, camera_ids, identity = visual_timeline._analyze_records(
        records, runtime=runtime, job=job,
    )
    assert identity.track_id == "face_001"
    assert len(tracks) == 2
    assert [o.at for o in tracks[0].observations] == list(map(float, range(7)))
    assert [o.at for o in tracks[1].observations] == [6., 7., 8., 9.]
    if check_webcam_hold:
        assert all(s.webcam_rect is None for s in samples if s.at >= 7)
        return
    assert all(classify_source_state(s)[0] is SourceState.FULL_CAMERA
               for s in samples if s.at >= 7)
    spans = visual_timeline._apply_identity_and_duration(
        segment_visual_states(samples), samples=samples, duration=10,
        identity_status=identity.status, streamer_track_id=identity.track_id,
        webcam_face_ids_by_time=camera_ids, webcam_tracks=[t.to_contract() for t in webcams],
    )
    recovered = next(s for s in spans if s.source.start <= 8 < s.source.end)
    assert recovered.state is SourceState.FULL_CAMERA
    assert recovered.identity_status is IdentityStatus.UNAVAILABLE
    assert recovered.streamer_track_id is None


@pytest.mark.parametrize("held_at", [8.0, 10.0], ids=["bridge", "edge_hold"])
def test_independent_full_camera_hold_keeps_identity_unavailable(
    reverse_dissolve_records: tuple[dict[float, visual_timeline._DecodedFrame], visual_timeline._BuilderRuntime, visual_timeline._ReadOnlyVisualContext],
    held_at: float,
) -> None:
    records, runtime, job = reverse_dissolve_records
    records[held_at] = visual_timeline._DecodedFrame(
        at=held_at, independent_features=records[9.0].independent_features,
        gray_thumbnail=records[9.0].gray_thumbnail,
        histogram=records[9.0].histogram, faces=[],
    )
    runtime.duration_seconds = 11
    samples, tracks, webcams, camera_ids, identity = visual_timeline._analyze_records(
        records, runtime=runtime, job=job,
    )
    assert len(tracks) == 2
    held = next(sample for sample in samples if sample.at == held_at)
    assert classify_source_state(held)[0] is SourceState.FULL_CAMERA
    spans = visual_timeline._apply_identity_and_duration(
        segment_visual_states(samples), samples=samples, duration=11,
        identity_status=identity.status, streamer_track_id=identity.track_id,
        webcam_face_ids_by_time=camera_ids, webcam_tracks=[t.to_contract() for t in webcams],
    )
    recovered = next(span for span in spans if span.source.start <= held_at < span.source.end)
    assert recovered.state is SourceState.FULL_CAMERA
    assert recovered.identity_status is IdentityStatus.UNAVAILABLE
    assert recovered.streamer_track_id is None
    assert camera_ids[held_at] == ()


@pytest.mark.parametrize("held_at", [8.0, 10.0], ids=["bridge", "edge_hold"])
def test_global_full_camera_hold_keeps_confident_identity(
    reverse_dissolve_records: tuple[dict[float, visual_timeline._DecodedFrame], visual_timeline._BuilderRuntime, visual_timeline._ReadOnlyVisualContext],
    held_at: float,
) -> None:
    records, runtime, job = reverse_dissolve_records
    centered = records[9.0].faces[0]
    for record in records.values():
        record.faces = [centered]
    records[held_at] = visual_timeline._DecodedFrame(
        at=held_at, independent_features=records[9.0].independent_features,
        gray_thumbnail=records[9.0].gray_thumbnail,
        histogram=records[9.0].histogram, faces=[],
    )
    runtime.duration_seconds = 11
    samples, _, webcams, camera_ids, identity = visual_timeline._analyze_records(
        records, runtime=runtime, job=job,
    )
    held = next(sample for sample in samples if sample.at == held_at)
    assert classify_source_state(held)[0] is SourceState.FULL_CAMERA
    assert camera_ids[held_at] == (identity.track_id,)
    spans = visual_timeline._apply_identity_and_duration(
        segment_visual_states(samples), samples=samples, duration=11,
        identity_status=identity.status, streamer_track_id=identity.track_id,
        webcam_face_ids_by_time=camera_ids, webcam_tracks=[t.to_contract() for t in webcams],
    )
    recovered = next(span for span in spans if span.source.start <= held_at < span.source.end)
    assert recovered.state is SourceState.FULL_CAMERA
    assert recovered.identity_status is IdentityStatus.CONFIDENT
    assert recovered.streamer_track_id == identity.track_id


def test_full_camera_holds_retain_global_independent_and_mixed_supports() -> None:
    face = LandmarkedFace(
        box=_rect(.45, .30, .14, .35),
        landmarks=((.48, .40), (.54, .40), (.51, .47), (.48, .54), (.54, .54)),
        confidence=.95, embedding=np.array([1., 0.], dtype=np.float32),
    )
    tracks = [visual_timeline.FaceTrack(track_id=track_id, observations=[
        visual_timeline.FaceObservation(at=at, face=face, source_state=SourceState.UNKNOWN,
                                       evidence_id=f"{track_id}:{at}")
        for at in times
    ]) for track_id, times in (("global", (0., 2.)), ("local", (2., 4., 5.)))]
    features = {float(at): LocalFrameFeatures(0, 0, None, (), 0, (), .5, 10, 0)
                for at in range(7)}
    identity = visual_timeline.StreamerIdentityResult(IdentityStatus.CONFIDENT, "global", 1., {})
    result = visual_timeline._full_camera_times(
        face_tracks=tracks, identity=identity, verified_webcams=[], local_features=features,
        saved_profile_embedding=np.array([1., 0.], dtype=np.float32),
    )
    assert result == {
        0.: ("global",), 1.: ("global",), 2.: ("global",),
        3.: ("global", "local"), 4.: ("local",), 5.: ("local",), 6.: ("local",),
    }


@pytest.mark.parametrize("guard", ["single_observation", "wrong_profile", "simultaneous"])
def test_independent_full_camera_requires_persistent_sole_profile_match(guard: str) -> None:
    def track(track_id: str, times: tuple[float, ...], embedding: tuple[float, float]) -> FaceTrack:
        face = LandmarkedFace(
            box=_rect(.45, .30, .14, .35),
            landmarks=((.48, .40), (.54, .40), (.51, .47), (.48, .54), (.54, .54)),
            confidence=.95, embedding=np.array(embedding, dtype=np.float32),
        )
        return visual_timeline.FaceTrack(track_id=track_id, observations=[
            visual_timeline.FaceObservation(at=at, face=face, source_state=SourceState.UNKNOWN,
                                            evidence_id=f"{track_id}:{at}") for at in times
        ])
    tracks = [track("global", (0., 1.), (1., 0.)),
              track("local", (4.,) if guard == "single_observation" else (4., 5.),
                    (0., 1.) if guard == "wrong_profile" else (1., 0.))]
    if guard == "simultaneous":
        tracks.append(track("other", (4., 5.), (1., 0.)))
    features = {at: LocalFrameFeatures(0, 0, None, (), 0, (), .5, 10, 0)
                for at in (0., 1., 4., 5.)}
    identity = visual_timeline.StreamerIdentityResult(IdentityStatus.CONFIDENT, "global", 1., {})
    result = visual_timeline._full_camera_times(
        face_tracks=tracks, identity=identity, verified_webcams=[], local_features=features,
        saved_profile_embedding=np.array([1., 0.], dtype=np.float32),
    )
    assert {4., 5.}.isdisjoint(result)


def test_full_camera_identity_checks_samples_after_dissolve_webcam_disappears() -> None:
    corner = _sample(43.5, webcam=_rect(0., 0.), webcam_track_id="webcam_001",
                     faces=("face_001", "face_004"))
    samples = [corner, *[_sample(at, faces=("face_004",)).model_copy(
        update={"evidence_ids": [f"frame:{at}", "inferred_full_camera"]}
    ) for at in (44., 44.5, 45., 45.5)]]
    span = visual_timeline.SourceVisualSpan(
        span_id="reverse_dissolve", source=visual_timeline.TimeRange(start=43.25, end=46),
        state=SourceState.FULL_CAMERA, confidence=.9, webcam_track_id="webcam_001",
        identity_status=IdentityStatus.UNAVAILABLE, evidence_ids=["frame:43.5"],
        streamer_track_id=None, important_regions=[], text_regions=[], change_reasons=[],
    )
    result = visual_timeline._apply_identity_and_duration(
        [span], samples=samples, duration=46, window_start=43.25,
        identity_status=IdentityStatus.CONFIDENT, streamer_track_id="face_001",
        webcam_tracks=[], webcam_face_ids_by_time={43.5: ("face_001",)},
    )
    assert result[0].identity_status is IdentityStatus.UNAVAILABLE
    assert result[0].streamer_track_id is None



def test_profile_verified_full_camera_bridges_stable_four_second_dropout() -> None:
    face = LandmarkedFace(
        box=_rect(0.46, 0.44, 0.08, 0.08),
        landmarks=(
            (0.478, 0.465),
            (0.522, 0.465),
            (0.500, 0.485),
            (0.482, 0.505),
            (0.518, 0.505),
        ),
        confidence=0.95,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    streamer = visual_timeline.FaceTrack(
        track_id="face_001",
        observations=[
            visual_timeline.FaceObservation(
                at=at,
                face=face,
                source_state=SourceState.UNKNOWN,
                evidence_id=f"face:{at:.1f}",
            )
            for at in (0.0, 4.5)
        ],
    )
    local_features = {
        at: LocalFrameFeatures(
            histogram_change_score=0.03,
            structural_change_score=0.06,
            letterbox_bounds=None,
            persistent_rectangle_candidates=(),
            text_density=0.75,
            text_regions=(_rect(0.76, 0.05, 0.20, 0.80),),
            brightness=0.5,
            sharpness=10.0,
            motion=0.02,
        )
        for at in (0.0, 2.0, 4.0, 4.5)
    }
    identity = visual_timeline.StreamerIdentityResult(
        IdentityStatus.CONFIDENT,
        "face_001",
        1.0,
        {"face_001": 1.0},
    )

    inferred = visual_timeline._full_camera_times(
        face_tracks=[streamer],
        identity=identity,
        verified_webcams=[],
        local_features=local_features,
        saved_profile_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert {0.0, 2.0, 4.0, 4.5}.issubset(inferred)
def test_cache_hit_preserves_interrupted_legacy_stage_state(

    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video, _, created = _install_fake_builder_runtime(monkeypatch, tmp_path)
    output = build_source_visual_timeline(video)
    job = visual_timeline.JobStore.for_video(video)
    job.mark_failed("source_visual_timeline", "interrupción simulada")

    assert build_source_visual_timeline(video) == output

    assert created == [True]
    assert not job.stage_complete("source_visual_timeline", artifact=output)
    stage = job.load()["stages"]["source_visual_timeline"]
    assert stage["status"] == "failed" and "failed_at" in stage
    assert stage["error"] == "interrupción simulada"


def _install_fake_window_runtime(
    monkeypatch: pytest.MonkeyPatch, *, video: Path,
    duration_seconds: float, decoder: _FakeDecoder,
) -> None:
    _install_observed_boundary(monkeypatch, video.parent, duration=duration_seconds,
        pixel_decoder=decoder, video_path=video, face_backend=_LargeStreamerFaces(),
        local_analyzer=_NoRectangleAnalyzer())


def test_isolated_window_never_uses_jobstore_or_decodes_outside_requested_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    job_root = tmp_path / "jobs" / "source-fingerprint"
    job_root.mkdir(parents=True)
    canonical = job_root / "source_visual_timeline.json"
    state = job_root / "state.json"
    canonical.write_text("canonical-sentinel", encoding="utf-8")
    state.write_text("state-sentinel", encoding="utf-8")
    decoder = _FakeDecoder()
    _install_fake_window_runtime(
        monkeypatch,
        video=video,
        duration_seconds=120.0,
        decoder=decoder,
    )
    monkeypatch.setattr(
        visual_timeline,
        "source_job_root",
        lambda source: job_root,
        raising=False,
    )
    monkeypatch.setattr(
        visual_timeline.JobStore,
        "for_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("isolated inspection cannot create or mutate JobStore")
        ),
    )

    output = visual_timeline.build_source_visual_timeline_window(
        video,
        start=40.0,
        end=42.0,
        sample_interval=0.5,
    )
    timeline = SourceVisualTimeline.model_validate_json(
        output.read_text(encoding="utf-8")
    )

    assert canonical.read_text(encoding="utf-8") == "canonical-sentinel"
    assert state.read_text(encoding="utf-8") == "state-sentinel"
    assert decoder.calls
    assert all(40.0 <= timestamp <= 42.0 for timestamp in decoder.calls)
    assert timeline.duration_seconds == 120.0
    assert all(40.0 <= sample.at <= 42.0 for sample in timeline.samples)
    assert timeline.face_tracks
    assert all(
        40.0 <= track.first_seen <= track.last_seen <= 42.0
        and track.observation_count <= len(timeline.samples)
        for track in timeline.face_tracks
    )
    assert all(
        40.0 <= span.source.start < span.source.end <= 42.0
        for span in timeline.spans
    )


def test_isolated_window_rejects_end_past_real_duration_before_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    _install_observed_boundary(monkeypatch, tmp_path, duration=60, video_path=video)
    monkeypatch.setattr(
        visual_timeline,
        "probe_media",
        lambda video_path: {"duration_seconds": 60.0},
    )
    monkeypatch.setattr(
        visual_timeline,
        "_create_window_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid range must fail before runtime/model/audio")
        ),
        raising=False,
    )

    with pytest.raises(ValueError, match="source duration 60.0"):
        visual_timeline.build_source_visual_timeline_window(
            video,
            start=59.0,
            end=61.0,
        )


def test_window_runtime_extracts_only_requested_audio_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    audio_path = tmp_path / "window.flac"
    observed: dict[str, object] = {}
    model_paths = {"yunet": tmp_path / "yunet.onnx", "sface": tmp_path / "sface.onnx"}

    monkeypatch.setattr(visual_timeline, "ensure_face_models", lambda *args, **kwargs: model_paths)
    monkeypatch.setattr(visual_timeline, "sha256_file", lambda path: "a" * 64)
    decoder_calls: dict[str, object] = {}

    def fake_window_decoder(path: Path, *, start_at: float) -> _FakeDecoder:
        decoder_calls.update(path=path, start_at=start_at)
        return _FakeDecoder()

    monkeypatch.setattr(visual_timeline, "_SequentialOpenCVFrameDecoder", fake_window_decoder)
    monkeypatch.setattr(visual_timeline, "OpenCVFaceIdentityBackend", lambda *args: _NoFaces())
    monkeypatch.setattr(visual_timeline, "AudioFeatureExtractor", lambda path: SimpleNamespace(window=lambda start, end: None))

    def fake_extract(video_path: Path, output_path: Path, *, start: float, end: float) -> Path:
        observed.update(video_path=video_path, output_path=output_path, start=start, end=end)
        return output_path

    monkeypatch.setattr(visual_timeline, "extract_audio_window", fake_extract, raising=False)

    visual_timeline._create_window_runtime(
        video,
        duration_seconds=100.0,
        start=40.0,
        end=42.0,
        allow_model_download=False,
        audio_path=audio_path,
    )
    assert decoder_calls == {"path": video, "start_at": 40.0}
    assert observed == {"video_path": video, "output_path": audio_path, "start": 40.0, "end": 42.0}


def test_sequential_window_decoder_seeks_once_and_decodes_monotonically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.position = 0
            self.set_calls: list[tuple[int, float]] = []
            self.read_calls = 0
            self.released = False

        def isOpened(self) -> bool:
            return True

        def set(self, property_id: int, value: float) -> bool:
            self.set_calls.append((property_id, value))
            if property_id == cv2.CAP_PROP_POS_FRAMES:
                self.position = int(value)
            return True

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FPS:
                return 10.0
            if property_id == cv2.CAP_PROP_POS_FRAMES:
                return float(self.position)
            return 0.0

        def read(self) -> tuple[bool, np.ndarray]:
            frame = np.full((4, 4, 3), self.position, dtype=np.uint8)
            self.position += 1
            self.read_calls += 1
            return True, frame

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    decoder = visual_timeline._SequentialOpenCVFrameDecoder(
        Path("source.mp4"),
        start_at=4.0,
    )

    first = decoder.decode_frame(4.0)
    second = decoder.decode_frame(4.2)

    assert int(first[0, 0, 0]) == 40
    assert int(second[0, 0, 0]) == 42
    assert capture.set_calls == [(cv2.CAP_PROP_POS_FRAMES, 40.0)]
    assert capture.read_calls == 3
    with pytest.raises(ValueError, match="monotonic"):
        decoder.decode_frame(4.1)
    decoder.close()
    assert capture.released is True


@pytest.mark.parametrize(
    ("decoder_factory", "expected_seek_calls"),
    [
        (
            lambda: visual_timeline._OpenCVFrameDecoder(Path("source.mp4")),
            [(cv2.CAP_PROP_POS_FRAMES, 899.0)],
        ),
        (
            lambda: visual_timeline._SequentialOpenCVFrameDecoder(
                Path("source.mp4"),
                start_at=29.999,
            ),
            [(cv2.CAP_PROP_POS_FRAMES, 899.0)],
        ),
    ],
)
def test_opencv_decoders_clamp_exact_duration_to_last_decodable_frame(
    monkeypatch: pytest.MonkeyPatch,
    decoder_factory: Callable[[], visual_timeline._OpenCVFrameDecoder | visual_timeline._SequentialOpenCVFrameDecoder],
    expected_seek_calls: list[tuple[int, float]],
) -> None:
    class EndBoundedCapture:
        def __init__(self) -> None:
            self.position = 0
            self.set_calls: list[tuple[int, float]] = []

        def isOpened(self) -> bool:
            return True

        def set(self, property_id: int, value: float) -> bool:
            self.set_calls.append((property_id, value))
            if property_id == cv2.CAP_PROP_POS_FRAMES:
                self.position = int(value)
            return True

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FPS:
                return 30.0
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 900.0
            return 0.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.position >= 900:
                return False, None
            frame = np.full((4, 4, 3), self.position % 255, dtype=np.uint8)
            self.position += 1
            return True, frame

        def release(self) -> None:
            return None

    capture = EndBoundedCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    decoder = decoder_factory()

    frame = decoder.decode_frame(29.999)

    assert int(frame[0, 0, 0]) == 899 % 255
    assert capture.set_calls == expected_seek_calls
    decoder.close()


def test_canonical_builder_public_signature_is_unchanged() -> None:
    import inspect

    assert str(inspect.signature(build_source_visual_timeline)) == (
        "(video_path: 'Path', *, editorial_boundaries: 'Sequence[float]' = (), "
        "sparse_interval: 'float' = 10.0, dense_interval: 'float' = 0.5, "
        "allow_model_download: 'bool' = False, force: 'bool' = False) -> 'Path'"
    )


class _ObservedCalls(TypedDict):
    video: list[tuple[tuple[Window, ...], tuple[F, ...], DecodeLimits]]
    audio: list[Window]
    qualify: list[bool]
    models: list[bool]


def _observed_work_counts(calls: _ObservedCalls) -> dict[str, int]:
    return {"video": len(calls["video"]), "audio": len(calls["audio"]),
            "models": len(calls["models"]), "qualify": len(calls["qualify"])}


def _install_observed_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, prefix: float | F = 0., duration: float | F = 4.,
    terminal: bool = False, pixel_decoder: visual_timeline.FrameDecoder | None = None, video_path: Path | None = None,
    face_backend: object | None = None, local_analyzer: object | None = None,
    audio_prefix: F = F(0),
    late_window_at: F | None = None,
) -> tuple[Path, SourceTimeOrigin, _ObservedCalls]:
    """Fake only tools/models; retain real receipt, WAV/JPEG and analyzer I/O."""
    from fractions import Fraction as F
    from clipperx.render_clock import SourceTimeOrigin, StreamClock, SourceAudioDecodeReceipt
    from clipperx.render_clock import SourceDecodeReceipt, audio_decode_argv, digest, rational
    from clipperx.source_decode import DecodedAudioWindow, DecodedWindow, AudioDecodeLimits
    from clipperx.source_decode import VideoTag, select_frames
    import zlib

    video = video_path or tmp_path / "source.mp4"
    if not video.exists():
        video.write_bytes(b"HOST synthetic source identity")
    width, height = (120, 80) if pixel_decoder is not None else (96, 54)
    video_base = F(1, 3000000) if isinstance(prefix, F) else F(1, 1000000)
    stat = video.stat()
    origin = SourceTimeOrigin(str(video.resolve()), stat.st_size, stat.st_mtime_ns,
        visual_timeline.source_fingerprint(video), F(0), F(str(duration)), (
            StreamClock(0, "video", video_base, F(str(prefix)), F(str(duration)), width, height),
            StreamClock(1, "audio", F(1, 48000), F(0), F(str(duration)),
                        sample_rate=48000, channels=1, sample_format="s16")), "HOST-observed", {})
    calls: _ObservedCalls = {"video": [], "audio": [], "qualify": [], "models": []}
    original = JobStore.for_video
    monkeypatch.setattr(visual_timeline.JobStore, "for_video",
        staticmethod(lambda path: original(path, jobs_root=tmp_path / "jobs")))
    monkeypatch.setattr(visual_timeline, "_resolve_observed_source", lambda path: (SimpleNamespace(), origin), raising=False)
    monkeypatch.setattr(visual_timeline, "qualify_render_runtime", lambda *args, **kwargs: calls["qualify"].append(True), raising=False)
    monkeypatch.setattr(visual_timeline, "_current_model_hashes", lambda: {"yunet": "a" * 64, "sface": "b" * 64})

    def backend(*args: object, **kwargs: object) -> object:
        calls["models"].append(True)
        return face_backend or _NoFaces()
    monkeypatch.setattr(visual_timeline, "_observed_face_backend", backend, raising=False)
    if local_analyzer is not None:
        monkeypatch.setattr(visual_timeline, "OpenCVLocalVisualAnalyzer", lambda: local_analyzer)

    def audio_payload(window: Window, limits: AudioDecodeLimits | DecodeLimits) -> tuple[dict[str, Any], bytes]:
        start = max(F(0), window.start - limits.preroll)
        end = min(origin.duration, window.end + limits.postroll)
        decoder_start = start
        start = max(start, audio_prefix)
        count_in, count_out = int((end - start) * 48000), int((end - start) * 16000)
        pcm = bytes(count_out * 2)
        first: dict[str, Any] = {"n": 0, "pts": int(start * 48000), "time_base": "1/48000", "rate": 48000,
                 "channels": 1, "samples": count_in, "format": "s16", "checksum": "00000000", "stage": "cxain"}
        output: dict[str, Any] = {**first, "pts": int(start * 16000), "rate": 16000, "time_base": "1/16000",
                  "samples": count_out, "bytes": len(pcm), "sha256": digest(pcm), "stage": "cxaout"}
        phase = F(output["pts"], 16000)
        covered = phase <= window.start and phase + F(count_out, 16000) >= window.end
        payload: dict[str, Any] = {"windows": [[rational(window.start), rational(window.end)]],
            "limits": {**limits.record(), "decoder_start": rational(decoder_start), "decoder_end": rational(end)},
            "audio": [{"policy": "analysis16k-mono", "input": [first], "output": [output], "samples": count_out,
                "bytes": len(pcm), "pcm_sha256": digest(pcm), "sample_grid_origin": rational(phase),
                "source_phase": rational(phase), "coverage": [{"window": [rational(window.start), rational(window.end)],
                                                              "status": "covered" if covered else "unavailable"}]}],
            "argv": [audio_decode_argv("HOST-ffmpeg", origin, decoder_start, end, "analysis16k-mono")]}
        return payload, pcm

    def observe_audio(runtime: RenderRuntime, selected_origin: SourceTimeOrigin, window: Window, *, audio_policy: str,
                      limits: AudioDecodeLimits = AudioDecodeLimits()) -> DecodedAudioWindow:
        calls["audio"].append(window)
        payload, pcm = audio_payload(window, limits)
        receipt = SourceAudioDecodeReceipt(origin.identity, origin.decoder_id, "HOST;audio-only;v1", payload)
        receipt.validate_origin(origin)
        return DecodedAudioWindow(receipt, pcm)

    def observe_video(runtime: RenderRuntime, selected_origin: SourceTimeOrigin, windows: tuple[Window, ...], requests: tuple[F, ...], *,
                      audio_policy: str, limits: DecodeLimits) -> tuple[DecodedWindow, ...]:
        calls["video"].append((tuple(windows), tuple(requests), limits))
        window = windows[0]
        payload, pcm = audio_payload(window, limits)
        instants = sorted({max(F(str(prefix)), t) for t in requests} |
                          {min(origin.duration - F(1, 1000000), max(F(str(prefix)), max(requests)) + F(1, 25))})
        if late_window_at is not None and window.start == late_window_at:
            instants = sorted({max(window.start + F(1, 50), t) for t in instants})
        if terminal:
            instants = [t for t in instants if t < origin.duration - F(1, 2)]
        pairs = []
        for n, t in enumerate(instants):
            if pixel_decoder is not None and t in requests:
                pixels = pixel_decoder.decode_frame(float(t)).tobytes()
            elif pixel_decoder is not None:
                pixels = bytes(width * height * 3)
            else:
                pixels = bytes([60 + n % 100]) * (width * height * 3)
            tag = VideoTag(n, int(t / video_base), video_base, width, height, zlib.adler32(pixels, 0))
            pairs.append((tag, pixels))
        selection = select_frames(pairs, origin, requests, limits)
        payload.update(video=[json.loads(s) for s in selection.observations],
                       outcomes=[o.record() for o in selection.outcomes], argv=[["HOST-video"], payload["argv"][0]])
        receipt = SourceDecodeReceipt(origin.identity, origin.decoder_id, "HOST-observed;v1", payload)
        receipt.validate_origin(origin)
        return (DecodedWindow(receipt, selection.frames, pcm),)

    monkeypatch.setattr(visual_timeline, "observe_audio_window", observe_audio, raising=False)
    monkeypatch.setattr(visual_timeline, "observe_windows", observe_video, raising=False)
    return video, origin, calls


def test_observed_generations_preserve_legacy_and_cache_without_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path)
    job = JobStore.for_video(video)
    job.source_visual_timeline_path.write_bytes(b"original v7")
    job.audio_path.write_bytes(b"original FLAC")
    job.mark_review_required("old_review", reason="keep", receipt=job.source_visual_timeline_path)
    old_stages = job.load()["stages"]
    assert hasattr(visual_timeline, "build_source_visual_timeline_result"), "invocation result seam absent"
    first = visual_timeline.build_source_visual_timeline_result(video, sparse_interval=1)
    assert first.cache_hit is False and first.indexed_samples_this_run > 0
    assert first.artifact_path.parent.parent.name == "generations"
    from clipperx.visual_provenance import load_visual_provenance
    context = load_visual_provenance(first.artifact_path, source_video=video, current_origin=origin)
    assert context.status == "observed_samples"
    work = _observed_work_counts(calls)
    hit = visual_timeline.build_source_visual_timeline_result(video, sparse_interval=1)
    assert hit.artifact_path == first.artifact_path and hit.cache_hit and hit.indexed_samples_this_run == 0
    assert _observed_work_counts(calls) == work
    forced = visual_timeline.build_source_visual_timeline_result(video, sparse_interval=1, force=True)
    assert forced.artifact_path != first.artifact_path and first.artifact_path.is_file()
    assert job.source_visual_timeline_path.read_bytes() == b"original v7"
    assert job.audio_path.read_bytes() == b"original FLAC"
    assert all(job.load()["stages"][key] == value for key, value in old_stages.items())


@pytest.mark.parametrize("prefix", [.015, .059])
def test_observed_unavailable_prefix_masked_after_finalize_no_dense_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prefix: float,
) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path, prefix=prefix)
    assert hasattr(visual_timeline, "build_source_visual_timeline_result"), "observed builder absent"
    result = visual_timeline.build_source_visual_timeline_result(video, sparse_interval=1, editorial_boundaries=(0.,))
    timeline = SourceVisualTimeline.model_validate_json(result.artifact_path.read_text())
    assert timeline.samples[0].at == prefix
    assert timeline.spans[0].source.start == 0 and timeline.spans[0].source.end == prefix
    assert timeline.spans[0].state is SourceState.UNKNOWN and timeline.spans[0].confidence == 0
    requests = [float(t) for _, ts, _ in calls["video"] for t in ts]
    assert requests.count(0.) == 1 and requests.count(prefix) == 1
    assert all(s.at != 0 for s in timeline.samples)


def test_observed_bounded_inspection_creates_no_job_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path)
    monkeypatch.setattr(visual_timeline.JobStore, "for_video", lambda *args, **kwargs: pytest.fail("bounded creates JobStore"))
    result = visual_timeline.build_source_visual_timeline_window(video, start=1, end=3, output_dir=tmp_path / "proof")
    assert result.parent.parent == tmp_path / "proof" / "visual_observed"
    assert (result.parent / "complete.json").is_file()
    assert not list(tmp_path.rglob("state.json")) and not list(tmp_path.rglob("latest.json"))


@pytest.mark.parametrize("fault", ["timeline", "preview", "receipt", "source", "config", "catalog"])
def test_observed_cache_corruption_fails_without_reanalysis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path)
    assert hasattr(visual_timeline, "build_source_visual_timeline_result"), "observed builder absent"
    result = visual_timeline.build_source_visual_timeline_result(video)
    generation = result.artifact_path.parent
    if fault == "source":
        video.write_bytes(b"changed source")
    elif fault == "config":
        with pytest.raises(ValueError):
            visual_timeline.build_source_visual_timeline_result(video, sparse_interval=5)
        return
    elif fault == "catalog":
        marker = generation / "complete.json"
        data = json.loads(marker.read_text())
        data["catalog"].append(data["catalog"][0])
        marker.write_text(json.dumps(data))
    else:
        target = result.artifact_path if fault == "timeline" else next(generation.glob("*.jpg" if fault == "preview" else "video-*.json"))
        target.write_bytes(b"corrupt")
    counts = _observed_work_counts(calls)
    with pytest.raises(ValueError):
        visual_timeline.build_source_visual_timeline_result(video)
    assert counts == _observed_work_counts(calls)


@pytest.mark.parametrize("phase", ["precommit", "select", "bookkeeping"])
def test_observed_publication_failures_preserve_commit_and_explicit_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str,
) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path)
    first = visual_timeline.build_source_visual_timeline_result(video)
    job = JobStore.for_video(video)
    latest_before = job.source_visual_observed_latest_path.read_bytes()
    old_stages = job.load()["stages"]
    writer = visual_timeline._write_observed_json
    selector = visual_timeline._select_observed_generation
    reconciler = visual_timeline._reconcile_observed_generation
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("HOST injected publication failure")
    if phase == "precommit":
        def before_marker(path: Path, value: object) -> None:
            if path.name == "complete.json":
                raise OSError("HOST precommit")
            writer(path, value)
        monkeypatch.setattr(visual_timeline, "_write_observed_json", before_marker)
        with pytest.raises(OSError, match="precommit"):
            visual_timeline.build_source_visual_timeline_result(video, force=True)
        assert job.source_visual_observed_latest_path.read_bytes() == latest_before
        incomplete = [p for p in first.artifact_path.parent.parent.iterdir() if not (p / "complete.json").is_file()]
        assert len(incomplete) == 1 and (incomplete[0] / "source_visual_timeline.json").is_file()
        return
    monkeypatch.setattr(visual_timeline,
        "_select_observed_generation" if phase == "select" else "_reconcile_observed_generation", fail)
    with pytest.raises(visual_timeline.ObservedPublicationError) as failure:
        visual_timeline.build_source_visual_timeline_result(video, force=True)
    error = failure.value
    assert error.phase == ("COMPLETE_UNSELECTED" if phase == "select" else "COMMITTED_BOOKKEEPING_FAILED")
    assert (error.artifact_path.parent / "complete.json").is_file()
    assert first.artifact_path.is_file()
    if phase == "select":
        assert job.source_visual_observed_latest_path.read_bytes() == latest_before
    else:
        assert json.loads(job.source_visual_observed_latest_path.read_text())["generation_id"] == error.artifact_path.parent.name
    assert all(job.load()["stages"][k] == value for k, value in old_stages.items())
    monkeypatch.setattr(visual_timeline, "_select_observed_generation", selector)
    monkeypatch.setattr(visual_timeline, "_reconcile_observed_generation", reconciler)
    counts = _observed_work_counts(calls)
    # Recovery is tied to the explicit path, even with a newer incomplete folder.
    job.new_observed_generation()
    assert visual_timeline.recover_observed_generation(video, error.artifact_path) == error.artifact_path
    assert counts == _observed_work_counts(calls)
    assert job.stage_complete("source_visual_observed:" + error.artifact_path.parent.name)


def test_observed_unknown_only_has_no_fake_samples_or_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path, duration=.1, terminal=True)
    result = visual_timeline.build_source_visual_timeline_result(video)
    timeline = SourceVisualTimeline.model_validate_json(result.artifact_path.read_text())
    assert timeline.samples == timeline.face_tracks == timeline.webcam_tracks == []
    assert len(timeline.spans) == 1 and timeline.spans[0].state is SourceState.UNKNOWN
    assert timeline.spans[0].confidence == 0 and result.indexed_samples_this_run == 0
    assert calls["audio"] == [] and not list(result.artifact_path.parent.glob("*.jpg"))


def test_observed_dense_schedule_budget_is_checked_before_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(visual_timeline, "MAX_SCHEDULE_POINTS", 3)
    with pytest.raises(ValueError, match="budget"):
        visual_timeline._timestamps_for_windows([(0., 2.)], interval=.5, existing_timestamps=set())


def test_observed_publication_error_after_latest_replace_reports_selected_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path)
    selector = visual_timeline._select_observed_generation
    def select_then_fail(job: JobStore, path: Path, marker: dict[str, object]) -> None:
        selector(job, path, marker)
        raise OSError("HOST acknowledgement failed after replace")
    monkeypatch.setattr(visual_timeline, "_select_observed_generation", select_then_fail)
    with pytest.raises(visual_timeline.ObservedPublicationError) as failure:
        visual_timeline.build_source_visual_timeline_result(video)
    error = failure.value
    assert error.phase == "COMMITTED_BOOKKEEPING_FAILED"
    latest = json.loads(JobStore.for_video(video).source_visual_observed_latest_path.read_text())
    assert latest["generation_id"] == error.artifact_path.parent.name


def test_observed_no_download_backend_uses_only_read_only_model_primitives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(visual_timeline, "_face_model_directory", lambda: tmp_path)
    monkeypatch.setattr(visual_timeline, "load_face_manifest", lambda path: {"models": {}})
    monkeypatch.setattr(visual_timeline, "validate_face_manifest", lambda *args: [])
    monkeypatch.setattr(visual_timeline, "validate_model_file", lambda *args: True)
    monkeypatch.setattr(visual_timeline, "OpenCVFaceIdentityBackend", lambda *args: _NoFaces())
    monkeypatch.setattr(visual_timeline, "ensure_face_models", lambda *args, **kwargs: pytest.fail("maintenance touched"))
    assert isinstance(visual_timeline._observed_face_backend(allow_model_download=False), _NoFaces)


def test_observed_audio_missing_prefix_is_empty_metadata_not_silence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path, audio_prefix=F(1, 16000))
    result = visual_timeline.build_source_visual_timeline_result(video, sparse_interval=1)
    timeline = SourceVisualTimeline.model_validate_json(result.artifact_path.read_text())
    assert timeline.samples[0].at == 0 and timeline.samples[0].audio_features == {}
    assert timeline.samples[0].audio_window is None
    assert timeline.samples[1].audio_features["start"] == 1.
    clock = json.loads((result.artifact_path.parent / "clock-receipt.json").read_text())
    assert clock["audio_associations"][0]["status"] == "unavailable"
    assert clock["audio_associations"][0]["slice"] is None


def test_observed_audio_groups_keep_complete_features_and_reject_oversize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path, duration=600)
    result = visual_timeline.build_source_visual_timeline_result(video, sparse_interval=100)
    assert len(calls["audio"]) == 3
    for window in calls["audio"]:
        assert min(origin.duration, window.end + F(1, 4)) - max(F(0), window.start - F(1, 2)) <= 300
    clock = json.loads((result.artifact_path.parent / "clock-receipt.json").read_text())
    assert len(clock["audio_associations"]) == result.indexed_samples_this_run
    assert all(row["status"] == "covered" for row in clock["audio_associations"])
    previous = JobStore.for_video(video).source_visual_observed_latest_path.read_bytes()
    counts = len(calls["audio"])
    with pytest.raises(ValueError, match="complete audio feature interval"):
        visual_timeline.build_source_visual_timeline_result(video, sparse_interval=301, force=True)
    assert len(calls["audio"]) == counts
    assert JobStore.for_video(video).source_visual_observed_latest_path.read_bytes() == previous


def test_observed_batches_release_analysis_pixels_and_respect_time_byte_budgets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import weakref
    references: list[Any] = []
    class Analyzer:
        def analyze(self, frame: np.ndarray, *, previous_frame: np.ndarray | None = None) -> LocalFrameFeatures:
            references.append(weakref.ref(frame))
            return LocalFrameFeatures(0, 0, None, (), 0, (), .5, 10, 0)
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path, duration=8., local_analyzer=Analyzer())
    observe = visual_timeline.observe_windows
    def checked(*args: Any, **kwargs: Any) -> Any:
        assert all(reference() is None for reference in references)
        window, requests = args[2][0], args[3]
        assert max(requests) - min(requests) <= 2
        assert min(origin.duration, window.end + F(1, 4)) - max(F(0), window.start - F(1, 2)) <= 3
        frame_bytes = 3 * 96 * 54
        assert len(requests) * frame_bytes + 96000 <= 64000000
        assert (len(requests) + 4) * frame_bytes + 96000 <= 192000000
        return observe(*args, **kwargs)
    monkeypatch.setattr(visual_timeline, "observe_windows", checked)
    visual_timeline.build_source_visual_timeline_result(video, sparse_interval=.1)
    assert len(calls["video"]) >= 4 and all(reference() is None for reference in references)


def test_observed_concurrent_invocations_report_one_build_one_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: visual_timeline.build_source_visual_timeline_result(video), range(2)))
    assert sorted(r.cache_hit for r in results) == [False, True]
    assert results[0].artifact_path == results[1].artifact_path
    assert next(r for r in results if r.cache_hit).indexed_samples_this_run == 0
    assert len(calls["models"]) == 1


def test_observed_catalog_budget_checked_before_batch_processing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path)
    monkeypatch.setattr(visual_timeline, "MAX_CATALOG_ARTIFACTS", 4, raising=False)
    with pytest.raises(ValueError, match="catalog"):
        visual_timeline.build_source_visual_timeline_result(video, sparse_interval=1)
    assert calls["video"] == []


def test_observed_bounded_postcommit_failure_carries_committed_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path)
    writer = visual_timeline._write_observed_json
    def write_then_fail(path: Path, value: object) -> None:
        writer(path, value)
        if path.name == "complete.json":
            raise OSError("HOST postcommit acknowledgement")
    monkeypatch.setattr(visual_timeline, "_write_observed_json", write_then_fail)
    with pytest.raises(visual_timeline.ObservedPublicationError) as failure:
        visual_timeline.build_source_visual_timeline_window(video, start=1, end=3, output_dir=tmp_path / "bounded")
    assert failure.value.artifact_path.is_file()
    assert (failure.value.artifact_path.parent / "complete.json").is_file()


def test_observed_first_eligible_rational_is_not_rounded_before_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path, prefix=F(1, 3))
    result = visual_timeline.build_source_visual_timeline_result(video, sparse_interval=1)
    timeline = SourceVisualTimeline.model_validate_json(result.artifact_path.read_text())
    assert timeline.samples[0].at == float(F(1, 3))
    clock = json.loads((result.artifact_path.parent / "clock-receipt.json").read_text())
    assert clock["sample_associations"][0]["requested"] == "1/3"
    assert clock["sample_associations"][0]["coverage_start"] == "1/3"


def _review_byte_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, Any, _ObservedCalls, Path]:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path, local_analyzer=_NoRectangleAnalyzer())
    generation = tmp_path / "byte-generation"
    generation.mkdir()
    pinned = RenderRuntime(Path("HOST-ffmpeg"), Path("HOST-ffprobe"), "HOST", "0" * 64, "0" * 64,
                           None, None, "HOST-graph", None)
    class NoFaces(_NoFaces):
        def cosine(self, left: np.ndarray, right: np.ndarray) -> float:
            raise AssertionError("No face arrays exist in this byte-budget test")
    session = visual_timeline._ObservedSession(pinned, origin, generation, 0., 4.)
    runtime = visual_timeline._BuilderRuntime(4., session, _NoRectangleAnalyzer(), NoFaces(), {}, None, 0., 4.)
    return session, runtime, calls, generation


def test_review_i3_reserves_final_metadata_before_models_observers_or_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path)
    monkeypatch.setattr(visual_timeline, "MAX_GENERATION_BYTES", 1000)
    with pytest.raises(ValueError, match="budget"):
        visual_timeline.build_source_visual_timeline_window(video, start=1., end=3., output_dir=tmp_path / "bounded")
    assert _observed_work_counts(calls) == {"video": 0, "audio": 0, "models": 0, "qualify": 0}
    assert not list((tmp_path / "bounded").rglob("*.json"))


def test_review_i3_next_batch_fails_before_observer_or_file_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session, runtime, calls, generation = _review_byte_session(monkeypatch, tmp_path)
    session.decode(runtime, [0.], {})
    before = {p.name: p.read_bytes() for p in generation.iterdir()}
    observed = len(calls["video"])
    monkeypatch.setattr(visual_timeline, "MAX_GENERATION_BYTES", session.created_bytes + 1)
    with pytest.raises(ValueError, match="budget"):
        session.decode(runtime, [1.], {})
    assert len(calls["video"]) == observed
    assert {p.name: p.read_bytes() for p in generation.iterdir()} == before


def test_review_i3_next_audio_group_fails_before_observer_or_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session, runtime, calls, generation = _review_byte_session(monkeypatch, tmp_path)
    monkeypatch.setattr(visual_timeline, "MAX_GENERATION_BYTES", session.created_bytes + 1)
    with pytest.raises(ValueError, match="budget"):
        session.attach_audio([_sample(0.)])
    assert calls["audio"] == [] and list(generation.iterdir()) == []


def test_review_i3_reconciles_next_only_reservation_to_actual_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(visual_timeline, "MAX_GENERATION_BYTES", 300_000_000)
    session, runtime, calls, generation = _review_byte_session(monkeypatch, tmp_path)
    records: dict[float, Any] = {}
    session.decode(runtime, [0.], records)
    session.decode(runtime, [1.], records)
    assert len(calls["video"]) == 2 and len(records) == 2
    assert session.created_bytes == sum(p.stat().st_size for p in generation.iterdir())


@pytest.mark.parametrize("kind", ["receipt", "jpeg", "wav"])
def test_review_i3_checks_encoded_artifact_size_before_file_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str,
) -> None:
    session, runtime, calls, generation = _review_byte_session(monkeypatch, tmp_path)
    monkeypatch.setattr(visual_timeline, "MAX_ARTIFACT_BYTES", 100 if kind == "receipt" else 128000)
    if kind == "jpeg":
        monkeypatch.setattr(cv2, "imencode", lambda *args, **kwargs: (True, np.zeros(128001, dtype=np.uint8)))
    with pytest.raises(ValueError, match="budget"):
        if kind == "wav":
            session.attach_audio([_sample(0.)])
        else:
            session.decode(runtime, [0.], {})
    suffix = ".json" if kind == "receipt" else ".jpg" if kind == "jpeg" else ".wav"
    assert not list(generation.glob("*" + suffix))


def test_review_i3_json_limit_counts_actual_windows_text_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os
    text = visual_timeline.canonical({"text": "sí"}) + "\n"
    emitted = text.replace("\n", os.linesep).encode("utf-8")
    target = tmp_path / "metadata.json"
    monkeypatch.setattr(visual_timeline, "MAX_ARTIFACT_BYTES", len(emitted) - 1)
    with pytest.raises(ValueError, match="budget"):
        visual_timeline._write_observed_json(target, {"text": "sí"})
    assert not target.exists()
    monkeypatch.setattr(visual_timeline, "MAX_ARTIFACT_BYTES", len(emitted))
    visual_timeline._write_observed_json(target, {"text": "sí"})
    assert target.read_bytes() == emitted


def test_review_i3_final_timeline_clock_and_complete_are_charged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video, origin, calls = _install_observed_boundary(monkeypatch, tmp_path)
    captured: list[Any] = []
    mask = visual_timeline._mask_unavailable_spans
    def capture(spans: Any, session: Any, *, no_samples: bool) -> Any:
        captured.append(session)
        return mask(spans, session, no_samples=no_samples)
    monkeypatch.setattr(visual_timeline, "_mask_unavailable_spans", capture)
    result = visual_timeline.build_source_visual_timeline_result(video)
    assert captured[0].created_bytes == sum(p.stat().st_size for p in result.artifact_path.parent.iterdir() if p.is_file())
