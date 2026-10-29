from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from typing import TypedDict

from typing_extensions import NotRequired

import pytest

from clipperx.visual_contracts import SourceVisualTimeline
from scripts import validate_source_timeline as validation


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_PATH = ROOT / "data" / "validation_visual" / "scenarios.json"
EXPECTED_IDS = {
    "stable_gameplay_webcam",
    "full_camera",
    "full_camera_to_browser",
    "browser_to_gameplay",
    "foreign_faces_in_content",
    "multi_person",
    "no_webcam",
    "reacted_vertical_video",
    "browser_small_text",
    "webcam_move_or_disappear",
}
ASSERTION_KEYS = {
    "expected_states",
    "ordered_states",
    "event_transition",
    "forbid_streamer_switch",
    "forbid_webcam_track",
    "require_text_regions",
    "require_multiple_faces",
    "require_vertical_region",
}


class _SamplePayload(TypedDict):
    sample_id: str
    at: float
    evidence_ids: list[str]
    face_track_ids: NotRequired[list[str]]
    webcam_track_id: NotRequired[str]
    webcam_rect: NotRequired[dict[str, float]]
    text_density: NotRequired[float]
    text_regions: NotRequired[list[dict[str, float]]]
    histogram_change_score: NotRequired[float]
    rectangle_candidates: NotRequired[list[dict[str, float]]]
    structural_change_score: NotRequired[float]
    brightness: NotRequired[float]
    sharpness: NotRequired[float]
    motion: NotRequired[float]


class _SampleBase(TypedDict):
    histogram_change_score: float
    structural_change_score: float
    text_density: float
    brightness: float
    sharpness: float
    motion: float


class _SpanPayload(TypedDict):
    span_id: str
    source: dict[str, float]
    state: str
    streamer_track_id: str | None
    identity_status: str
    webcam_track_id: str | None
    evidence_ids: list[str]
    change_reasons: list[str]
    confidence: NotRequired[float]
    important_regions: NotRequired[list[dict[str, float]]]
    text_regions: NotRequired[list[dict[str, float]]]


class _SpanBase(TypedDict):
    confidence: float
    important_regions: list[dict[str, float]]
    text_regions: list[dict[str, float]]
    change_reasons: list[str]


class _AllPassSpanBase(_SpanBase):
    streamer_track_id: None
    identity_status: str
    webcam_track_id: None


class _TimelinePayload(TypedDict):
    contract_version: int
    source_fingerprint: str
    duration_seconds: float
    analyzer_version: str
    model_hashes: dict[str, str]
    sampling_config: dict[str, float]
    samples: list[_SamplePayload]
    spans: list[_SpanPayload]
    face_tracks: list[dict[str, object]]
    webcam_tracks: list[dict[str, object]]


class _MappingEntry(TypedDict):
    scenario_id: str
    source: str
    start: float
    end: float
    event_at: NotRequired[float]


class _MappingPayload(TypedDict):
    version: int
    mappings: list[_MappingEntry]


def _assertion(report: dict[str, object], index: int) -> dict[str, object]:
    assertions = report["assertions"]
    assert isinstance(assertions, list)
    assertion = assertions[index]
    assert isinstance(assertion, dict)
    assert all(isinstance(key, str) for key in assertion)
    return assertion


def _actual_mapping(assertion: dict[str, object]) -> dict[str, object]:
    actual = assertion["actual"]
    assert isinstance(actual, dict)
    assert all(isinstance(key, str) for key in actual)
    return actual


def _counts(summary: dict[str, object]) -> dict[str, object]:
    counts = summary["counts"]
    assert isinstance(counts, dict)
    assert all(isinstance(key, str) for key in counts)
    return counts


def _number(value: object) -> int | float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return value

@pytest.fixture(autouse=True)
def _synthetic_media_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        validation,
        "probe_media",
        lambda source: {"duration_seconds": 40.0},
        raising=False,
    )


def _published_generation(output: Path) -> tuple[dict[str, object], Path]:
    latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    generation = output / str(latest["generation"])
    assert generation.is_dir()
    return latest, generation


def _report(output: Path, scenario_id: str) -> dict[str, object]:
    _, generation = _published_generation(output)
    return json.loads(
        (generation / f"{scenario_id}.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("hit,count,force", [(True, 0, False), (False, 7, False), (False, 3, True)])
def test_observed_default_runner_uses_locked_invocation_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hit: bool, count: int, force: bool,
) -> None:
    from clipperx.visual_timeline import VisualTimelineBuildResult
    source = tmp_path / "source.mp4"
    source.write_bytes(b"HOST source")
    path = tmp_path / "immutable-generation" / "source_visual_timeline.json"
    def build(*args: object, **kwargs: object) -> VisualTimelineBuildResult:
        assert kwargs["force"] is force
        return VisualTimelineBuildResult(path, hit, count)
    monkeypatch.setattr(validation, "build_source_visual_timeline_result", build, raising=False)
    monkeypatch.setattr(validation, "_file_snapshot", lambda *args: pytest.fail("default runner guesses from historical files"))
    metadata = validation._invoke_builder(validation.build_source_visual_timeline, source,
        boundaries=(), sparse_interval=10, dense_interval=.5, force=force)
    assert metadata.artifact_path == path and metadata.cache_hit is hit and metadata.indexed_samples_this_run == count


def test_runner_json_writer_handles_transient_target_denial_without_losing_latest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    target = tmp_path / "latest.json"
    target.write_text('{"run":"old"}\n', encoding="utf-8")
    original = Path.replace
    attempts: list[Path] = []
    def denied_once(path: Path, destination: Path) -> Path:
        if destination == target:
            attempts.append(path)
            if len(attempts) == 1:
                assert target.read_text(encoding="utf-8") == '{"run":"old"}\n'
                error = PermissionError("HOST transient target denial")
                error.winerror = 5
                raise error
        return original(path, destination)
    monkeypatch.setattr(Path, "replace", denied_once)
    validation._atomic_write_json(target, {"run": "nuevo", "texto": "sí"})
    assert len(attempts) == 2
    assert target.read_text(encoding="utf-8") == '{\n  "run": "nuevo",\n  "texto": "sí"\n}\n'
    assert not list(tmp_path.glob(".latest*.tmp"))


def test_runner_json_writer_preserves_latest_and_propagates_permanent_denial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    target = tmp_path / "latest.json"
    old = b'{"run":"old"}\n'
    target.write_bytes(old)
    attempts: list[Path] = []
    def permanently_denied(path: Path, destination: Path) -> Path:
        attempts.append(path)
        error = PermissionError("HOST permanent target denial")
        error.winerror = 5
        raise error
    monkeypatch.setattr(Path, "replace", permanently_denied)
    with pytest.raises(PermissionError, match="HOST permanent"):
        validation._atomic_write_json(target, {"run": "new"})
    assert target.read_bytes() == old and not list(tmp_path.glob(".latest*.tmp"))
    assert len(attempts) <= 6


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _timeline_payload() -> _TimelinePayload:
    face_rect = {"x": 0.02, "y": 0.02, "width": 0.2, "height": 0.25}
    text_rect = {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.1}
    sample_base: _SampleBase = {
        "histogram_change_score": 0.1,
        "structural_change_score": 0.1,
        "text_density": 0.0,
        "brightness": 0.5,
        "sharpness": 1.0,
        "motion": 0.1,
    }
    span_base: _SpanBase = {
        "confidence": 0.9,
        "important_regions": [],
        "text_regions": [],
        "change_reasons": [],
    }
    return {
        "contract_version": 1,
        "source_fingerprint": "synthetic-source",
        "duration_seconds": 40.0,
        "analyzer_version": "visual-test-v1",
        "model_hashes": {"yunet": "yunet-test", "sface": "sface-test"},
        "sampling_config": {"sparse_interval": 10.0, "dense_interval": 0.5},
        "samples": [
            {
                **sample_base,
                "sample_id": "sample_5",
                "at": 5.0,
                "face_track_ids": ["face_streamer"],
                "webcam_track_id": "webcam_main",
                "webcam_rect": face_rect,
                "evidence_ids": ["sample_5"],
            },
            {
                **sample_base,
                "sample_id": "sample_15",
                "at": 15.0,
                "face_track_ids": ["face_streamer"],
                "evidence_ids": ["sample_15"],
            },
            {
                **sample_base,
                "sample_id": "sample_25",
                "at": 25.0,
                "text_density": 0.8,
                "text_regions": [text_rect],
                "evidence_ids": ["sample_25"],
            },
            {
                **sample_base,
                "sample_id": "sample_35",
                "at": 35.0,
                "evidence_ids": ["sample_35"],
            },
        ],
        "spans": [
            {
                **span_base,
                "span_id": "span_gameplay",
                "source": {"start": 0.0, "end": 10.0},
                "state": "content_with_camera",
                "streamer_track_id": "face_streamer",
                "identity_status": "confident",
                "webcam_track_id": "webcam_main",
                "evidence_ids": ["span_gameplay"],
            },
            {
                **span_base,
                "span_id": "span_camera",
                "source": {"start": 10.0, "end": 20.0},
                "state": "full_camera",
                "streamer_track_id": "face_streamer",
                "identity_status": "confident",
                "webcam_track_id": None,
                "evidence_ids": ["span_camera"],
                "change_reasons": ["layout_changed"],
            },
            {
                **span_base,
                "span_id": "span_browser",
                "source": {"start": 20.0, "end": 30.0},
                "state": "browser_text",
                "streamer_track_id": None,
                "identity_status": "ambiguous",
                "webcam_track_id": None,
                "text_regions": [text_rect],
                "evidence_ids": ["span_browser"],
                "change_reasons": ["layout_changed"],
            },
            {
                **span_base,
                "span_id": "span_content",
                "source": {"start": 30.0, "end": 40.0},
                "state": "content_only",
                "streamer_track_id": None,
                "identity_status": "unavailable",
                "webcam_track_id": None,
                "evidence_ids": ["span_content"],
                "change_reasons": ["layout_changed"],
            },
        ],
        "face_tracks": [
            {
                "track_id": "face_streamer",
                "first_seen": 1.0,
                "last_seen": 19.0,
                "observation_count": 3,
                "embedding_centroid": [1.0, 0.0],
                "positions": [face_rect],
                "recurrent_states": ["content_with_camera", "full_camera"],
                "full_camera_seconds": 10.0,
                "mean_confidence": 0.9,
                "evidence_ids": ["face_streamer"],
            }
        ],
        "webcam_tracks": [
            {
                "track_id": "webcam_main",
                "source_ranges": [{"start": 0.0, "end": 10.0}],
                "observations": [
                    {
                        "at": 5.0,
                        "rect": face_rect,
                        "confidence": 0.9,
                        "evidence_id": "webcam_5",
                    }
                ],
                "face_track_ids": ["face_streamer"],
                "confidence": 0.9,
                "evidence_ids": ["webcam_main"],
            }
        ],
    }


def _write_timeline(path: Path) -> Path:
    payload = _timeline_payload()
    SourceVisualTimeline.model_validate(payload)
    return _write_json(path, payload)


def _mapping(source: Path, *windows: tuple[str, float, float]) -> _MappingPayload:
    return {
        "version": 1,
        "mappings": [
            {
                "scenario_id": scenario_id,
                "source": str(source),
                "start": start,
                "end": end,
            }
            for scenario_id, start, end in windows
        ],
    }


def test_portable_manifest_has_exactly_ten_executable_scenarios() -> None:
    raw = SCENARIOS_PATH.read_text(encoding="utf-8")
    manifest = validation.load_scenario_manifest(SCENARIOS_PATH)

    assert {scenario.scenario_id for scenario in manifest.scenarios} == EXPECTED_IDS
    assert len(manifest.scenarios) == 10
    assert all(scenario.assertions for scenario in manifest.scenarios)
    assert all(set(scenario.assertions) <= ASSERTION_KEYS for scenario in manifest.scenarios)
    assert "R:\\" not in raw
    assert "O:\\" not in raw
    assert "C:\\" not in raw


def test_manifest_rejects_non_executable_expectations(tmp_path: Path) -> None:
    invalid = {
        "version": 1,
        "scenarios": [{"id": "stable_gameplay_webcam", "human_note": "looks good"}],
    }
    with pytest.raises(ValueError, match="assertion ejecutable"):
        validation.load_scenario_manifest(_write_json(tmp_path / "bad.json", invalid))


    generic_change_count = {
        "version": 1,
        "scenarios": [
            {"id": "stable_gameplay_webcam", "minimum_state_changes": 1}
        ],
    }
    with pytest.raises(ValueError, match="assertion ejecutable"):
        validation.load_scenario_manifest(_write_json(tmp_path / "generic.json", generic_change_count))

@pytest.mark.parametrize(
    "mapping, message",
    [
        ({"scenario_id": "unknown", "source": "vod.mp4", "start": 0, "end": 1}, "scenario_id"),
        ({"scenario_id": "full_camera", "source": "", "start": 0, "end": 1}, "source"),
        ({"scenario_id": "full_camera", "source": "vod.mp4", "start": 2, "end": 1}, "end"),
    ],
)
def test_local_mapping_validates_scenario_source_and_range(
    tmp_path: Path, mapping: dict[str, object], message: str
) -> None:
    manifest = validation.load_scenario_manifest(SCENARIOS_PATH)
    payload = {"version": 1, "mappings": [mapping]}
    with pytest.raises(ValueError, match=message):
        validation.load_source_mappings(
            _write_json(tmp_path / "sources.json", payload), manifest
        )


def test_runner_groups_by_source_and_combines_all_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline(tmp_path / "timeline.json")
    sources = _write_json(
        tmp_path / "sources.json",
        _mapping(
            source,
            ("stable_gameplay_webcam", 0.0, 10.0),
            ("full_camera", 10.0, 20.0),
            ("browser_small_text", 20.0, 30.0),
            ("no_webcam", 30.0, 40.0),
        ),
    )
    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_builder(video_path: Path, **kwargs: object) -> Path:
        calls.append((video_path, kwargs))
        return timeline

    monkeypatch.setattr(validation, "build_source_visual_timeline", fake_builder)
    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    assert len(calls) == 1
    assert calls[0][0] == source
    assert calls[0][1]["editorial_boundaries"] == (0.0, 10.0, 20.0, 30.0, 40.0)
    assert calls[0][1]["allow_model_download"] is False
    assert result.exit_code == 2
    assert result.summary["gate_passed"] is False
    assert result.summary["counts"] == {
        "pass": 4,
        "fail": 0,
        "ambiguous": 0,
        "unavailable": 6,
    }


def test_runner_includes_human_event_in_editorial_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline(tmp_path / "timeline.json")
    mapping_payload = _mapping(source, ("browser_to_gameplay", 20.0, 40.0))
    mapping_payload["mappings"][0]["event_at"] = 30.0
    sources = _write_json(tmp_path / "sources.json", mapping_payload)
    calls: list[dict[str, object]] = []

    def fake_builder(video_path: Path, **kwargs: object) -> Path:
        del video_path
        calls.append(kwargs)
        return timeline

    monkeypatch.setattr(validation, "build_source_visual_timeline", fake_builder)
    validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    assert calls[0]["editorial_boundaries"] == (20.0, 30.0, 40.0)


def test_runner_writes_complete_reports_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline_payload = _timeline_payload()
    timeline_payload["spans"][2]["state"] = "content_with_camera"
    timeline = _write_timeline_payload(tmp_path / "timeline.json", timeline_payload)
    mapping_payload = _mapping(source, ("full_camera_to_browser", 10.0, 30.0))
    mapping_payload["mappings"][0]["event_at"] = 20.0
    sources = _write_json(tmp_path / "sources.json", mapping_payload)
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: timeline
    )
    output = tmp_path / "run"
    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=output,
    )

    latest, generation = _published_generation(output)
    report = json.loads((generation / "full_camera_to_browser.json").read_text())
    summary = json.loads((generation / "summary.json").read_text())
    assert result.exit_code == 2
    assert len(list(generation.glob("*.json"))) == 11
    assert latest["run_id"] == summary["run_id"]
    assert report["status"] == "pass"
    assert report["source_reference"] == str(source)
    assert report["requested_window"] == {"start": 10.0, "end": 30.0}
    assert _number(report["source_build_runtime_seconds"]) >= 0
    assert _number(report["scenario_evaluation_runtime_seconds"]) >= 0
    assert report["source_indexed_sample_count"] == 4
    assert report["window_indexed_sample_count"] == 2
    assert report["indexed_samples_this_run"] is None
    assert report["cache_hit"] is False
    assert "decoded_frame_count" not in report
    assert "runtime_seconds" not in report
    assert report["model_hashes"] == {"yunet": "yunet-test", "sface": "sface-test"}
    assert [span["state"] for span in report["relevant_state_spans"]] == [
        "full_camera",
        "content_with_camera",
    ]
    assert report["identity_statuses"] == ["ambiguous", "confident"]
    assert _assertion(report, 0)["passed"] is True
    assert report["failures"] == []
    assert summary["counts"]["unavailable"] == 9


def test_same_state_identity_boundary_does_not_satisfy_main_content_event() -> None:
    payload = _timeline_payload()
    payload["spans"][2] = {
        **payload["spans"][2],
        "state": "content_with_camera",
        "streamer_track_id": "face_streamer",
        "identity_status": "confident",
        "webcam_track_id": "webcam_main",
        "change_reasons": ["webcam_track_changed"],
    }
    payload["spans"][3] = {
        **payload["spans"][3],
        "state": "content_with_camera",
        "streamer_track_id": "face_streamer",
        "identity_status": "confident",
        "webcam_track_id": "webcam_main",
        "change_reasons": ["identity_changed"],
    }
    timeline = SourceVisualTimeline.model_validate(payload)
    mapping = validation.SourceMapping(
        scenario_id="browser_to_gameplay",
        source_reference="source.mp4",
        source_path=Path("source.mp4"),
        start=20.0,
        end=40.0,
        event_at=30.0,
    )
    expectation = {
        "kind": "main_content",
        "within_seconds": 0.1,
        "minimum_change_score": 0.18,
    }

    report = validation.evaluate_scenario(
        validation.Scenario(
            scenario_id="browser_to_gameplay",
            assertions={"event_transition": expectation},
        ),
        mapping,
        timeline,
        source_build_runtime_seconds=0.0,
        build_metadata=validation.BuildMetadata(
            artifact_path=Path("timeline.json"),
            cache_hit=True,
            indexed_samples_this_run=0,
        ),
    )

    assert report["status"] == "fail"
    assert _actual_mapping(_assertion(report, 0))["nearest_qualifying_event"] is None


def test_main_content_change_near_event_satisfies_semantic_transition() -> None:
    payload = _timeline_payload()
    payload["samples"].insert(
        3,
        {
            **payload["samples"][2],
            "sample_id": "sample_30",
            "at": 30.0,
            "histogram_change_score": 0.31,
            "evidence_ids": ["sample_30"],
        },
    )
    timeline = SourceVisualTimeline.model_validate(payload)
    mapping = validation.SourceMapping(
        scenario_id="browser_to_gameplay",
        source_reference="source.mp4",
        source_path=Path("source.mp4"),
        start=20.0,
        end=40.0,
        event_at=30.0,
    )
    expectation = {
        "kind": "main_content",
        "within_seconds": 0.1,
        "minimum_change_score": 0.18,
    }

    result = validation._assertion_result(
        "event_transition",
        expectation,
        mapping=mapping,
        spans=[span for span in timeline.spans if span.source.start >= 20.0],
        samples=[sample for sample in timeline.samples if sample.at >= 20.0],
        webcam_tracks=[],
    )

    assert result["status"] == "pass"
    assert _actual_mapping(result)["nearest_qualifying_event"] == 30.0


def test_expected_states_are_exclusive_and_ordered_states_require_sequence() -> None:
    timeline = SourceVisualTimeline.model_validate(_timeline_payload())
    mapping = validation.SourceMapping(
        scenario_id="full_camera_to_browser",
        source_reference="source.mp4",
        source_path=Path("source.mp4"),
        start=10.0,
        end=30.0,
    )
    spans, samples, webcam_tracks = validation._relevant_timeline_data(timeline, mapping)

    strict = validation._assertion_result(
        "expected_states",
        ["full_camera"],
        mapping=mapping,
        spans=spans,
        samples=samples,
        webcam_tracks=webcam_tracks,
    )
    ordered = validation._assertion_result(
        "ordered_states",
        ["full_camera", "browser_text"],
        mapping=mapping,
        spans=spans,
        samples=samples,
        webcam_tracks=webcam_tracks,
    )
    reversed_order = validation._assertion_result(
        "ordered_states",
        ["browser_text", "full_camera"],
        mapping=mapping,
        spans=spans,
        samples=samples,
        webcam_tracks=webcam_tracks,
    )

    assert strict["status"] == "fail"
    assert ordered["status"] == "pass"
    assert reversed_order["status"] == "fail"
    prefixed_mapping = validation.SourceMapping(
        scenario_id="full_camera_to_browser",
        source_reference="source.mp4",
        source_path=Path("source.mp4"),
        start=0.0,
        end=30.0,
    )
    prefixed_spans, prefixed_samples, prefixed_webcams = validation._relevant_timeline_data(
        timeline, prefixed_mapping
    )
    unexpected_prefix = validation._assertion_result(
        "ordered_states",
        ["full_camera", "browser_text"],
        mapping=prefixed_mapping,
        spans=prefixed_spans,
        samples=prefixed_samples,
        webcam_tracks=prefixed_webcams,
    )
    assert unexpected_prefix["status"] == "fail"


def test_failed_required_assertion_sets_exit_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline(tmp_path / "timeline.json")
    sources = _write_json(
        tmp_path / "sources.json", _mapping(source, ("no_webcam", 0.0, 10.0))
    )
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: timeline
    )
    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    assert result.exit_code == 1
    assert _counts(result.summary)["fail"] == 1
    report = _report(tmp_path / "run", "no_webcam")
    assert report["status"] == "fail"
    assert report["failures"]


def test_vertical_region_requires_evidence_in_two_distinct_samples() -> None:
    payload = _timeline_payload()
    payload["samples"][3] = {
        **payload["samples"][3],
        "rectangle_candidates": [
            {"x": 0.35, "y": 0.15, "width": 0.23, "height": 0.62}
        ],
    }
    timeline = SourceVisualTimeline.model_validate(payload)
    mapping = validation.SourceMapping(
        scenario_id="reacted_vertical_video",
        source_reference="source.mp4",
        source_path=Path("source.mp4"),
        start=30.0,
        end=40.0,
    )

    report = validation.evaluate_scenario(
        validation.Scenario(
            scenario_id="reacted_vertical_video",
            assertions={"require_vertical_region": True},
        ),
        mapping,
        timeline,
        source_build_runtime_seconds=0.0,
        build_metadata=validation.BuildMetadata(
            artifact_path=Path("timeline.json"),
            cache_hit=True,
            indexed_samples_this_run=0,
        ),
    )

    assert report["status"] == "fail"
    assert _actual_mapping(_assertion(report, 0))["persistent_sample_count"] == 1


def test_missing_local_sources_records_ten_unavailable_reports(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=tmp_path / "missing.json",
        output_dir=output,
    )

    assert result.exit_code == 2
    assert result.summary["gate_passed"] is False
    assert result.summary["counts"] == {
        "pass": 0,
        "fail": 0,
        "ambiguous": 0,
        "unavailable": 10,
    }
    _, generation = _published_generation(output)
    assert len(list(generation.glob("*.json"))) == 11
    report = _report(output, "stable_gameplay_webcam")
    assert report["status"] == "unavailable"
    assert report["failures"] == []
    assert report["unavailable_reason"] == "sources_mapping_missing"


def _write_timeline_payload(path: Path, payload: _TimelinePayload) -> Path:
    SourceVisualTimeline.model_validate(payload)
    return _write_json(path, payload)


def _all_pass_timeline_payload() -> _TimelinePayload:
    payload = _timeline_payload()
    payload["spans"][2]["state"] = "content_with_camera"
    payload["duration_seconds"] = 60.0
    samples = payload["samples"]
    sample_5 = samples[0].copy()
    samples.insert(0, {**sample_5, "sample_id": "sample_2", "at": 2.0})
    samples.insert(2, {**sample_5, "sample_id": "sample_8", "at": 8.0})
    sample_30: _SamplePayload = {
        **samples[4],
        "sample_id": "sample_30",
        "at": 30.0,
        "histogram_change_score": 0.31,
        "evidence_ids": ["sample_30"],
    }
    samples.insert(5, sample_30)
    sample_base: _SampleBase = {
        "histogram_change_score": 0.1,
        "structural_change_score": 0.1,
        "text_density": 0.0,
        "brightness": 0.5,
        "sharpness": 1.0,
        "motion": 0.1,
    }
    samples.extend(
        [
            {
                **sample_base,
                "sample_id": "sample_45",
                "at": 45.0,
                "face_track_ids": ["face_streamer", "face_guest"],
                "evidence_ids": ["sample_45"],
            },
            {
                **sample_base,
                "sample_id": "sample_54",
                "at": 54.0,
                "rectangle_candidates": [
                    {"x": 0.35, "y": 0.15, "width": 0.23, "height": 0.62}
                ],
                "evidence_ids": ["sample_54"],
            },
            {
                **sample_base,
                "sample_id": "sample_56",
                "at": 56.0,
                "rectangle_candidates": [
                    {"x": 0.36, "y": 0.15, "width": 0.23, "height": 0.62}
                ],
                "evidence_ids": ["sample_56"],
            },
        ]
    )
    span_base: _AllPassSpanBase = {
        "confidence": 0.9,
        "streamer_track_id": None,
        "identity_status": "unavailable",
        "webcam_track_id": None,
        "important_regions": [],
        "text_regions": [],
        "change_reasons": ["layout_changed"],
    }
    payload["spans"].extend(
        [
            {
                **span_base,
                "span_id": "span_multi",
                "source": {"start": 40.0, "end": 50.0},
                "state": "multi_person",
                "evidence_ids": ["span_multi"],
            },
            {
                **span_base,
                "span_id": "span_reacted",
                "source": {"start": 50.0, "end": 60.0},
                "state": "reacted_video",
                "evidence_ids": ["span_reacted"],
            },
        ]
    )
    guest_track = dict(payload["face_tracks"][0])
    guest_track.update(
        {
            "track_id": "face_guest",
            "first_seen": 44.0,
            "last_seen": 46.0,
            "observation_count": 2,
            "recurrent_states": ["multi_person"],
            "evidence_ids": ["face_guest"],
        }
    )
    payload["face_tracks"].append(guest_track)
    return payload


def _all_windows(source: Path) -> _MappingPayload:
    payload = _mapping(
        source,
        ("stable_gameplay_webcam", 0.0, 10.0),
        ("full_camera", 10.0, 20.0),
        ("full_camera_to_browser", 10.0, 30.0),
        ("browser_to_gameplay", 20.0, 40.0),
        ("foreign_faces_in_content", 0.0, 10.0),
        ("multi_person", 40.0, 50.0),
        ("no_webcam", 30.0, 40.0),
        ("reacted_vertical_video", 50.0, 60.0),
        ("browser_small_text", 20.0, 30.0),
        ("webcam_move_or_disappear", 0.0, 20.0),
    )

    events = {
        "full_camera_to_browser": 20.0,
        "browser_to_gameplay": 30.0,
        "webcam_move_or_disappear": 10.0,
    }
    for mapping in payload["mappings"]:
        event_at = events.get(mapping["scenario_id"])
        if event_at is not None:
            mapping["event_at"] = event_at
    return payload

def test_gate_passes_only_when_all_ten_scenarios_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline_payload(
        tmp_path / "timeline.json", _all_pass_timeline_payload()
    )
    sources = _write_json(tmp_path / "sources.json", _all_windows(source))
    monkeypatch.setattr(validation, "probe_media", lambda path: {"duration_seconds": 60.0})
    monkeypatch.setattr(
        validation,
        "build_source_visual_timeline",
        lambda *args, **kwargs: validation.BuildMetadata(
            artifact_path=timeline,
            cache_hit=False,
            indexed_samples_this_run=17,
        ),
    )

    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    assert result.exit_code == 0
    assert result.summary["gate_passed"] is True
    assert result.summary["counts"] == {
        "pass": 10,
        "fail": 0,
        "ambiguous": 0,
        "unavailable": 0,
    }


def test_window_end_is_exclusive_for_samples() -> None:
    payload = _timeline_payload()
    boundary_sample: _SamplePayload = {
        **payload["samples"][0],
        "sample_id": "sample_40",
        "at": 40.0,
        "evidence_ids": ["sample_40"],
    }
    payload["samples"].append(boundary_sample)
    timeline = SourceVisualTimeline.model_validate(payload)
    mapping = validation.SourceMapping(
        scenario_id="no_webcam",
        source_reference="source.mp4",
        source_path=Path("source.mp4"),
        start=30.0,
        end=40.0,
    )

    report = validation.evaluate_scenario(
        validation.Scenario(
            scenario_id="no_webcam",
            assertions={"forbid_webcam_track": True},
        ),
        mapping,
        timeline,
        source_build_runtime_seconds=0.0,
        build_metadata=validation.BuildMetadata(
            artifact_path=Path("timeline.json"),
            cache_hit=True,
            indexed_samples_this_run=0,
        ),
    )

    assert report["status"] == "pass"
    assert report["window_indexed_sample_count"] == 1

def test_no_webcam_forbid_is_unavailable_without_indexed_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    payload = _timeline_payload()
    payload["samples"] = [sample for sample in payload["samples"] if sample["at"] != 35.0]
    timeline = _write_timeline_payload(tmp_path / "timeline.json", payload)
    sources = _write_json(
        tmp_path / "sources.json", _mapping(source, ("no_webcam", 30.0, 40.0))
    )
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: timeline
    )

    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    report = _report(tmp_path / "run", "no_webcam")
    assert result.exit_code == 2
    assert report["status"] == "unavailable"
    assert _assertion(report, 0)["status"] == "unavailable"


def test_streamer_switch_forbid_requires_identity_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline(tmp_path / "timeline.json")
    sources = _write_json(
        tmp_path / "sources.json",
        _mapping(source, ("foreign_faces_in_content", 30.0, 40.0)),
    )
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: timeline
    )

    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    report = _report(tmp_path / "run", "foreign_faces_in_content")
    assert result.exit_code == 2
    assert report["status"] == "unavailable"
    assert _assertion(report, 0)["status"] == "unavailable"


def test_streamer_switch_forbid_requires_identity_coverage_for_full_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    payload = _timeline_payload()
    payload["spans"][1] = {
        **payload["spans"][1],
        "streamer_track_id": None,
        "identity_status": "unavailable",
    }
    timeline = _write_timeline_payload(tmp_path / "timeline.json", payload)
    sources = _write_json(
        tmp_path / "sources.json",
        _mapping(source, ("foreign_faces_in_content", 0.0, 20.0)),
    )
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: timeline
    )

    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    report = _report(tmp_path / "run", "foreign_faces_in_content")
    assert result.exit_code == 2
    assert report["status"] == "unavailable"
    assert _assertion(report, 0)["status"] == "unavailable"


def test_streamer_switch_forbid_propagates_identity_ambiguity_only_for_that_assertion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline(tmp_path / "timeline.json")
    sources = _write_json(
        tmp_path / "sources.json",
        _mapping(
            source,
            ("foreign_faces_in_content", 20.0, 30.0),
            ("browser_small_text", 20.0, 30.0),
        ),
    )
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: timeline
    )

    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    identity_report = _report(tmp_path / "run", "foreign_faces_in_content")
    text_report = _report(tmp_path / "run", "browser_small_text")
    assert result.exit_code == 2
    assert identity_report["status"] == "ambiguous"
    assert _assertion(identity_report, 0)["status"] == "ambiguous"
    assert text_report["status"] == "pass"


def test_probe_media_runs_once_and_invalid_windows_do_not_enter_build_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline(tmp_path / "timeline.json")
    sources = _write_json(
        tmp_path / "sources.json",
        _mapping(
            source,
            ("full_camera", 10.0, 20.0),
            ("multi_person", 45.0, 55.0),
        ),
    )
    probes: list[Path] = []
    builds: list[dict[str, object]] = []

    def fake_probe(path: Path) -> dict[str, object]:
        probes.append(path)
        return {"duration_seconds": 40.0}

    def fake_builder(path: Path, **kwargs: object) -> Path:
        builds.append(kwargs)
        return timeline

    monkeypatch.setattr(validation, "probe_media", fake_probe)
    monkeypatch.setattr(validation, "build_source_visual_timeline", fake_builder)
    validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    assert probes == [source]
    assert len(builds) == 1
    assert builds[0]["editorial_boundaries"] == (10.0, 20.0)
    assert _report(tmp_path / "run", "multi_person")["status"] == "unavailable"


def test_all_out_of_duration_windows_skip_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    sources = _write_json(
        tmp_path / "sources.json",
        _mapping(source, ("multi_person", 45.0, 55.0)),
    )
    builds: list[Path] = []
    monkeypatch.setattr(validation, "probe_media", lambda path: {"duration_seconds": 40.0})
    monkeypatch.setattr(
        validation,
        "build_source_visual_timeline",
        lambda path, **kwargs: builds.append(path),
    )

    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    assert result.exit_code == 2
    assert builds == []
    assert _report(tmp_path / "run", "multi_person")["unavailable_reason"] == "window_outside_media"


def test_builder_metadata_separates_cache_and_index_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline(tmp_path / "timeline.json")
    sources = _write_json(
        tmp_path / "sources.json", _mapping(source, ("full_camera", 10.0, 20.0))
    )
    monkeypatch.setattr(
        validation,
        "build_source_visual_timeline",
        lambda *args, **kwargs: validation.BuildMetadata(
            artifact_path=timeline,
            cache_hit=True,
            indexed_samples_this_run=0,
        ),
    )

    validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    report = _report(tmp_path / "run", "full_camera")
    assert report["source_indexed_sample_count"] == 4
    assert report["window_indexed_sample_count"] == 1
    assert report["indexed_samples_this_run"] == 0
    assert report["cache_hit"] is True
    assert _number(report["source_build_runtime_seconds"]) >= 0
    assert _number(report["scenario_evaluation_runtime_seconds"]) >= 0


def test_transactional_publication_keeps_complete_unique_generations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline(tmp_path / "timeline.json")
    sources = _write_json(
        tmp_path / "sources.json", _mapping(source, ("full_camera", 10.0, 20.0))
    )
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: timeline
    )
    output = tmp_path / "run"

    first = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH, sources_path=sources, output_dir=output
    )
    second = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH, sources_path=sources, output_dir=output
    )

    generations = list((output / "generations").iterdir())
    assert first.summary["run_id"] != second.summary["run_id"]
    assert len(generations) == 2
    assert all(len(list(path.glob("*.json"))) == 11 for path in generations)
    latest, generation = _published_generation(output)
    assert latest["run_id"] == second.summary["run_id"]
    assert generation in generations
    assert not list(output.glob("*scenario*.json"))


def test_concurrent_publications_never_mix_generations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline(tmp_path / "timeline.json")
    sources = _write_json(
        tmp_path / "sources.json", _mapping(source, ("full_camera", 10.0, 20.0))
    )
    barrier = threading.Barrier(2)

    def synchronized_builder(*args: object, **kwargs: object) -> Path:
        barrier.wait(timeout=5)
        return timeline

    monkeypatch.setattr(validation, "build_source_visual_timeline", synchronized_builder)
    output = tmp_path / "run"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                validation.run_validation,
                scenario_manifest_path=SCENARIOS_PATH,
                sources_path=sources,
                output_dir=output,
            )
            for _ in range(2)
        ]
        results = [future.result(timeout=15) for future in futures]

    generations = list((output / "generations").iterdir())
    assert len({result.summary["run_id"] for result in results}) == 2
    assert len(generations) == 2
    assert all(len(list(path.glob("*.json"))) == 11 for path in generations)
    latest, generation = _published_generation(output)
    assert latest["run_id"] in {result.summary["run_id"] for result in results}
    assert len(list(generation.glob("*.json"))) == 11


def test_failed_generation_does_not_replace_previous_latest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    timeline = _write_timeline(tmp_path / "timeline.json")
    sources = _write_json(
        tmp_path / "sources.json", _mapping(source, ("full_camera", 10.0, 20.0))
    )
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: timeline
    )
    output = tmp_path / "run"
    validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH, sources_path=sources, output_dir=output
    )
    previous_latest = (output / "latest.json").read_bytes()
    real_write = validation._atomic_write_json

    def fail_on_summary(path: Path, payload: object) -> None:
        if path.name == "summary.json":
            raise OSError("simulated publication failure")
        real_write(path, payload)

    monkeypatch.setattr(validation, "_atomic_write_json", fail_on_summary)
    with pytest.raises(OSError, match="simulated publication failure"):
        validation.run_validation(
            scenario_manifest_path=SCENARIOS_PATH,
            sources_path=sources,
            output_dir=output,
        )

    assert (output / "latest.json").read_bytes() == previous_latest
    _, generation = _published_generation(output)
    assert len(list(generation.glob("*.json"))) == 11


def test_style_audit_records_real_gate_and_remaining_long_vod_limit() -> None:
    audit = (ROOT / "docs" / "style-corpus-audit.md").read_text(encoding="utf-8")

    assert "Task 6" in audit
    assert "GATE LOCAL PHASE 1 APROBADO" in audit
    assert "local_sources.json" in audit
    assert "10/0/0/0" in audit
    assert "gate_passed: true" in audit
    assert "347" in audit
    assert "645,567 s" in audit


def test_forbid_requires_complete_temporal_span_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    payload = _timeline_payload()
    payload["spans"] = [span for span in payload["spans"] if span["span_id"] != "span_content"]
    timeline = _write_timeline_payload(tmp_path / "timeline.json", payload)
    sources = _write_json(
        tmp_path / "sources.json", _mapping(source, ("no_webcam", 30.0, 40.0))
    )
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: timeline
    )

    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    report = _report(tmp_path / "run", "no_webcam")
    assert result.exit_code == 2
    assert report["status"] == "unavailable"
    assert _assertion(report, 0)["message"] == "incomplete temporal span coverage"


def test_streamer_switch_with_sufficient_identity_evidence_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    payload = _timeline_payload()
    payload["spans"][1]["streamer_track_id"] = "face_guest"
    payload["samples"][1]["face_track_ids"] = ["face_guest"]
    payload["face_tracks"].append(
        {
            "track_id": "face_guest",
            "first_seen": 11.0,
            "last_seen": 19.0,
            "observation_count": 2,
            "embedding_centroid": [0.0, 1.0],
            "positions": [{"x": 0.6, "y": 0.1, "width": 0.2, "height": 0.25}],
            "recurrent_states": ["full_camera"],
            "full_camera_seconds": 5.0,
            "mean_confidence": 0.9,
            "evidence_ids": ["face_guest"],
        }
    )
    timeline = _write_timeline_payload(tmp_path / "timeline.json", payload)
    sources = _write_json(
        tmp_path / "sources.json",
        _mapping(source, ("foreign_faces_in_content", 0.0, 20.0)),
    )
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: timeline
    )

    result = validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
    )

    report = _report(tmp_path / "run", "foreign_faces_in_content")
    assert result.exit_code == 1
    assert report["status"] == "fail"
    assert _assertion(report, 0)["actual"] == ["face_guest", "face_streamer"]


def test_force_run_is_never_inferred_as_cache_hit_from_unchanged_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    job_root = tmp_path / "job"
    artifact = _write_timeline(job_root / "source_visual_timeline.json")
    sources = _write_json(
        tmp_path / "sources.json", _mapping(source, ("full_camera", 10.0, 20.0))
    )
    monkeypatch.setattr(validation, "source_job_root", lambda path: job_root)
    monkeypatch.setattr(
        validation, "build_source_visual_timeline", lambda *args, **kwargs: artifact
    )

    validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
        force=True,
    )

    report = _report(tmp_path / "run", "full_camera")
    assert report["cache_hit"] is False
    assert report["indexed_samples_this_run"] is None


def test_rebuilt_canonical_artifact_reports_decoded_sample_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"synthetic")
    job_root = tmp_path / "job"
    artifact = job_root / "source_visual_timeline.json"
    sources = _write_json(
        tmp_path / "sources.json", _mapping(source, ("full_camera", 10.0, 20.0))
    )

    def rebuild(*args: object, **kwargs: object) -> Path:
        return _write_timeline(artifact)

    monkeypatch.setattr(validation, "source_job_root", lambda path: job_root)
    monkeypatch.setattr(validation, "build_source_visual_timeline", rebuild)

    validation.run_validation(
        scenario_manifest_path=SCENARIOS_PATH,
        sources_path=sources,
        output_dir=tmp_path / "run",
        force=True,
    )

    report = _report(tmp_path / "run", "full_camera")
    assert report["cache_hit"] is False
    assert report["indexed_samples_this_run"] == 4
