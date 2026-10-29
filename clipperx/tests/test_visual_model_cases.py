from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from clipperx.structured_client import StructuredResult
from clipperx.tests.test_structured_client import (
    DiagnosticResponse, DiagnosticSession, PLANTED_SECRET, diagnostic_client,
)
from clipperx.tests.test_visual_director import ready
from clipperx.visual_contracts import SourceVisualTimeline
from scripts import benchmark_visual_models as benchmark


ROOT = Path(__file__).resolve().parents[2]
pytest_plugins = ["clipperx.tests.test_visual_director"]
ModelT = TypeVar("ModelT", bound=BaseModel)


class FakeClient:
    """Stateless fake implementing the actual optional-image client protocol."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []
        self.images: list[list[Path]] = []

    def generate(self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
                 images: list[Path] | None = None, temperature: float = .2
                 ) -> StructuredResult[ModelT]:
        payload = json.loads(user_prompt)
        self.payloads.append(payload)
        self.images.append(images or [])
        response = self.responses.pop(0)
        if callable(response):
            response = response(payload)
        return StructuredResult(output_model.model_validate(response), {"total_tokens": 10}, {})


def _prepare_cli_fake_results(
    scene: Any, tmp_path: Path, monkeypatch: Any,
) -> tuple[FakeClient, list[dict[str, Any]], list[str], Path]:
    """Generate the three real validated fake results before any cold aggregate probe."""
    video, clip, timeline, library = scene
    cases: list[dict[str, Any]] = [
        {"case_id": case_id, "expected": {"source_ranges": [[60, 62], [62, 64]]}}
        for case_id in ["one", "two", "three"]]
    monkeypatch.setattr(benchmark, "load_cases", lambda path: cases)
    monkeypatch.setattr(benchmark, "prepare_case", lambda *args: (video, clip, timeline))
    monkeypatch.setattr(benchmark, "StyleLibrary", lambda path: library)
    client = FakeClient([ready, ready, ready])
    monkeypatch.setattr(benchmark, "make_client", lambda *args: client)
    monkeypatch.setattr(benchmark, "read_key", lambda: "fake")
    audit = tmp_path / "audit.md"
    audit.write_text("Offline test\n")
    output = tmp_path / "selection.json"
    common = ["--models", benchmark.OWNER_MODEL, "--run-dir", str(tmp_path / "run"),
              "--selection-output", str(output), "--audit-output", str(audit)]
    for case in cases:
        assert benchmark.main(common + ["--case-id", case["case_id"], "--confirm-inference"]) == 0
        assert json.loads(output.read_text())["selected_model"] is None
    assert len(client.payloads) == 3 and client.responses == []
    return client, cases, common, output


def _saved_fake_case_artifacts(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for case_id in ("one", "two", "three")
            for path in (root / case_id).rglob("*") if path.is_file()}


def test_three_portable_cases_are_bounded() -> None:
    cases = benchmark.load_cases(ROOT / "data/model_validation/visual_director_cases.json")
    assert len(cases) == 3
    assert [c["expected"]["source_ranges"] for c in cases] == [
        [[0, 10]], [[0.5, 6.5]], [[24, 30.75], [30.75, 38]]]
    assert all(len(c["image_paths"]) <= 8 for c in cases)


def test_confirmation_precedes_credentials_and_client(monkeypatch: Any) -> None:
    monkeypatch.setattr(benchmark, "read_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(benchmark, "make_client", lambda *a: pytest.fail("client created"))
    with pytest.raises(ValueError, match="confirm-inference"):
        benchmark.main(["--models", "gpt-6-luna(low)"])


@pytest.mark.parametrize("operation", ["import", "help", "unconfirmed", "aggregate"])
def test_fresh_benchmark_entrypoints_block_configuration_before_its_loader(
    tmp_path: Path, operation: str, request: pytest.FixtureRequest,
) -> None:
    saved_cases: dict[str, bytes] = {}
    if operation == "aggregate":
        scene = request.getfixturevalue("scene")
        with pytest.MonkeyPatch.context() as patch:
            _, cases, common, _ = _prepare_cli_fake_results(scene, tmp_path, patch)
        video, clip, timeline, library = scene
        benchmark.write_json(tmp_path / "aggregate-inputs.json", {
            "cases": cases, "argv": common + ["--aggregate"], "source": str(video),
            "clip": clip.model_dump(mode="json"), "timeline": timeline.model_dump(mode="json"),
            "style_root": str(library.root),
        })
        saved_cases = _saved_fake_case_artifacts(tmp_path / "run")
    probe = dedent('''
        import importlib.abc
        import importlib.machinery
        import json
        from pathlib import Path
        import sys
        from types import ModuleType
        from collections.abc import Sequence

        blocked_actions = []

        class BlockConfiguration(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname: str, path: Sequence[str] | None = None,
                          target: ModuleType | None = None
                          ) -> importlib.machinery.ModuleSpec | None:
                if fullname == "clipperx.config":
                    raise AssertionError("configuration loader BLOCKED before .env or credentials")
                if fullname == "pytest" and sys.argv[1] == "aggregate":
                    blocked_actions.append("nested pytest")
                    raise AssertionError("nested pytest forbidden in aggregate child")
                return None

        sys.meta_path.insert(0, BlockConfiguration())
        from clipperx import style_retrieval, visual_director
        from scripts import benchmark_visual_models as benchmark
        operation, temporary = sys.argv[1:]
        if operation == "help":
            try:
                benchmark.main(["--help"])
            except SystemExit as error:
                assert error.code == 0
            else:
                raise AssertionError("help did not exit")
        elif operation == "unconfirmed":
            root = Path(temporary)
            try:
                benchmark.main(["--models", benchmark.OWNER_MODEL,
                    "--base-url", "http://localhost:8317/v1",
                    "--cases", str(root / "must-not-read-cases.json"),
                    "--local-mapping", str(root / "must-not-read-mapping.json"),
                    "--style-library", str(root / "styles"),
                    "--run-dir", str(root / "run"), "--case-id", "synthetic",
                    "--selection-output", str(root / "selection.json"),
                    "--audit-output", str(root / "audit.md")])
            except ValueError as error:
                assert "confirm-inference" in str(error)
            else:
                raise AssertionError("unconfirmed command was accepted")
            assert not list(root.iterdir())
        elif operation == "aggregate":
            prepared = json.loads((Path(temporary) / "aggregate-inputs.json").read_text())
            video = Path(prepared["source"])
            clip = benchmark.ClipPlan.model_validate(prepared["clip"])
            timeline = benchmark.SourceVisualTimeline.model_validate(prepared["timeline"])
            library = benchmark.StyleLibrary(Path(prepared["style_root"]))
            reused_cases = []

            # Match the CLI fixture's existing input adapters. No case is
            # generated here; the production aggregate/signature checks stay real.
            benchmark.load_cases = lambda path: prepared["cases"]
            def reuse_prepared_case(case, mapping):
                assert case in prepared["cases"]
                reused_cases.append(case["case_id"])
                return video, clip, timeline
            benchmark.prepare_case = reuse_prepared_case
            benchmark.StyleLibrary = lambda path: library

            def forbidden(action):
                def deny(*args, **kwargs):
                    blocked_actions.append(action)
                    raise AssertionError(action)
                return deny

            boundaries = [
                (benchmark, "run_case"), (benchmark, "run_visual_director"),
                (benchmark, "read_key"), (benchmark, "make_client"),
                (visual_director, "run_visual_director"),
                (visual_director, "collect_visual_evidence"),
                (visual_director.observed_evidence, "prepare_visual_evidence_context"),
            ]
            for module, name in boundaries:
                setattr(module, name, forbidden(name))
                try:
                    getattr(module, name)()
                except AssertionError as error:
                    assert str(error) == name
                else:
                    raise AssertionError("inactive generation sentinel: " + name)
            try:
                __import__("pytest")
            except AssertionError as error:
                assert str(error) == "nested pytest forbidden in aggregate child"
            else:
                raise AssertionError("inactive nested pytest sentinel")
            assert blocked_actions == [name for _, name in boundaries] + ["nested pytest"]
            blocked_actions.clear()

            assert benchmark.main(prepared["argv"]) == 0
            assert reused_cases == ["one", "two", "three"]
            assert blocked_actions == [] and "pytest" not in sys.modules
            print("aggregate reused saved cases: no generation or nested pytest")
        assert "clipperx.config" not in sys.modules
        print("configuration never loaded: " + operation)
    ''')
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe, operation, str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=40, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"configuration never loaded: {operation}" in result.stdout
    if operation == "aggregate":
        assert "aggregate reused saved cases: no generation or nested pytest" in result.stdout
        assert _saved_fake_case_artifacts(tmp_path / "run") == saved_cases
        aggregate = json.loads((tmp_path / "selection.json").read_text())
        assert aggregate["selected_model"] == benchmark.OWNER_MODEL
        assert len(aggregate["results"]) == 3 and aggregate["cheapest_passing_model"] is None


def test_discovery_only_inventories_and_keeps_owner_target() -> None:
    class Inventory:
        def get(self, url: str, **kwargs: Any) -> Any:
            assert url.endswith("/models")
            assert kwargs["headers"]["Authorization"] == "Bearer ephemeral"
            class Response:
                def raise_for_status(self) -> None:
                    pass

                def json(self) -> Any:
                    return {"data": [{"id": "gpt-6-luna"}, {"id": "gemini-3"},
                                     {"id": "unselected-expensive"}]}
            return Response()

    data = benchmark.discover_models("http://localhost:8317/v1", "ephemeral", Inventory())
    assert data["models"] == ["gpt-6-luna(low)"]
    assert data["inference_calls"] == 0
    assert "gemini-3" not in data["inventory_model_ids"]
    assert data["billing"] == "unpriced"


def test_unknown_inventory_does_not_infer_alias() -> None:
    class Inventory:
        def get(self, *args: Any, **kwargs: Any) -> Any:
            class Response:
                def raise_for_status(self) -> None:
                    pass

                def json(self) -> Any:
                    return {"data": [{"id": "another-model"}]}
            return Response()
    assert benchmark.discover_models("http://localhost:8317/v1", "x", Inventory())["models"] == []


def test_local_subset_preserves_face_time_bijection(scene: Any) -> None:
    from clipperx.evidence import associated_face_positions
    from clipperx.tests.test_visual_contracts import make_face_track
    _, _, timeline, _ = scene
    data = timeline.model_dump()
    data["face_tracks"] = [make_face_track(first_seen=1, last_seen=80,
        observation_count=3, positions=[dict(x=.1, y=.1, width=.2, height=.2)] * 3,
        evidence_ids=["f1", "f60", "f80"]).model_dump()]
    sample = dict(sample_id="s1", at=1, histogram_change_score=0, structural_change_score=0,
        text_density=0, brightness=.5, sharpness=1, motion=0, face_track_ids=["face_main"],
        evidence_ids=["f1"])
    data["samples"] = [dict(sample, sample_id=f"s{at}", at=at, evidence_ids=[eid])
                       for at, eid in [(1, "f1"), (60, "f60"), (80, "f80")]]
    subset = benchmark.timeline_subset(SourceVisualTimeline.model_validate(data), 45, 90)
    face = subset.face_tracks[0]
    associated = associated_face_positions(subset, face)
    assert associated is not None
    assert [row[0] for row in associated] == [60, 80]
    assert face.observation_count == len(face.positions) == len(face.evidence_ids) == 2
    assert subset.duration_seconds == timeline.duration_seconds
    assert subset.source_fingerprint == timeline.source_fingerprint


def test_public_director_passing_case_records_measured_metrics(scene: Any, tmp_path: Path) -> None:
    video, clip, timeline, library = scene
    result = benchmark.run_case({"case_id": "synthetic", "expected": {
        "source_ranges": [[60, 62], [62, 64]]}}, video, clip, timeline, library,
        FakeClient([ready]), tmp_path)
    assert result["passed"] is True
    assert result["calls"] == 1
    assert result["schema_success_rate"] == result["invariant_pass_rate"] == 1
    assert result["estimated_provider_cost"] is None
    assert result["prompt_tokens"] is None  # Fake client reports no raw provider usage.
    assert result["evidence_iterations"] == 0
    assert Path(result["artifact"]).name == "visual_plan.json"
    assert not {"failure_stage", "provider_http_status", "provider_error_code"}.intersection(
        result["call_metrics"][0])


def test_valid_json_with_wrong_identity_fails_host_gate(scene: Any, tmp_path: Path) -> None:
    video, clip, timeline, library = scene
    def wrong(payload: Any) -> Any:
        response = ready(payload)
        response["plan"]["shots"][0]["layers"][0].update(
            kind="streamer_camera", source_ref="foreign_face")
        return response
    result = benchmark.run_case({"case_id": "bad", "expected": {
        "source_ranges": [[60, 62], [62, 64]]}}, video, clip, timeline, library,
        FakeClient([wrong]), tmp_path)
    assert result["schema_success_rate"] == 1
    assert result["passed"] is False
    assert result["invariant_pass_rate"] < 1
    assert Path(result["artifact"]).name == "receipt.json"


def test_observed_source_video_reference_fails_public_host_without_rewrite(
    scene: Any, tmp_path: Path,
) -> None:
    video, clip, timeline, library = scene
    responses: list[dict[str, Any]] = []

    def wrong(payload: dict[str, Any]) -> dict[str, Any]:
        response = ready(payload)
        response["plan"]["shots"][0]["layers"][0]["source_ref"] = "source_video"
        responses.append(response)
        return response

    result = benchmark.run_case({"case_id": "wrong_source_ref", "expected": {
        "source_ranges": [[60, 62], [62, 64]]}}, video, clip, timeline, library,
        FakeClient([wrong]), tmp_path)
    assert result["schema_success_rate"] == 1
    assert result["passed"] is False and result["calls"] == 1
    assert "Only streamer_camera" in result["failure_reason"]
    assert responses[0]["plan"]["shots"][0]["layers"][0]["source_ref"] == "source_video"
    assert Path(result["artifact"]).name == "receipt.json"


@pytest.mark.parametrize(("status", "code", "transport", "stage", "expected_code"), [
    (429, "insufficient_quota", False, "provider_http", "insufficient_quota"),
    (429, "rate_limit_exceeded", False, "provider_http", "rate_limit_exceeded"),
    (429, PLANTED_SECRET, False, "provider_http", None),
    (503, PLANTED_SECRET, False, "provider_http", None),
    (503, None, True, "transport", None),
    (200, None, False, "structured_schema", None),
])
def test_public_failure_metrics_categorize_without_persisting_secrets(
    scene: Any, tmp_path: Path, status: int, code: object, transport: bool,
    stage: str, expected_code: str | None,
) -> None:
    video, clip, timeline, library = scene
    session = DiagnosticSession(DiagnosticResponse(status, code), transport=transport)
    result = benchmark.run_case({"case_id": "diagnostic", "expected": {
        "source_ranges": [[60, 62], [62, 64]]}}, video, clip, timeline, library,
        diagnostic_client(session), tmp_path)
    assert result["passed"] is False and result["schema_success_rate"] == 0
    assert result["calls"] == session.calls == 1 and result["evidence_iterations"] == 0
    metric = result["call_metrics"][0]
    assert metric["failure_stage"] == stage
    assert metric.get("provider_http_status") == (status if stage == "provider_http" else None)
    assert metric.get("provider_error_code") == expected_code
    assert all(result[key] is None for key in (
        "prompt_tokens", "completion_tokens", "total_tokens", "estimated_provider_cost"))
    artifact = Path(result["artifact"])
    persisted = "".join(path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json"))
    assert PLANTED_SECRET not in json.dumps(result) + persisted
    assert "private.invalid" not in persisted
    if expected_code is None:
        assert "insufficient_quota" not in persisted and "rate_limit_exceeded" not in persisted
    assert artifact.name == "receipt.json"


def test_measured_input_budget_failure_is_separate_and_prevents_post(tmp_path: Path) -> None:
    from clipperx.planner_contracts import DiscoveryBatchResponse
    from clipperx.structured_client import InputBudgetError

    session = DiagnosticSession(DiagnosticResponse(503, PLANTED_SECRET))
    measured = benchmark.MeasuredClient(diagnostic_client(session, small_context=True))
    with pytest.raises(InputBudgetError):
        measured.generate(DiscoveryBatchResponse, system_prompt=PLANTED_SECRET,
            user_prompt=json.dumps({"frame_manifest": [], "text": PLANTED_SECRET * 20_000}))
    record = measured.records[0]
    assert record["failure_stage"] == "input_budget"
    assert record["schema_valid"] is False and record["total_tokens"] is None
    assert session.calls == 0
    path = tmp_path / "metrics.json"
    benchmark.write_json(path, measured.records)
    assert PLANTED_SECRET not in path.read_text()


@pytest.mark.parametrize(("content", "envelope", "stage", "kind"), [
    (None, False, "structured_schema", "empty_content"),
    ("{", False, "structured_schema", "json_parse"),
    ('{"status":"' + PLANTED_SECRET + '"}', False, "structured_schema", "model_validation"),
    (None, True, "response_envelope", "response_envelope"),
])
def test_measured_safe_failure_metadata_survives_serialization(
    tmp_path: Path, content: object, envelope: bool, stage: str, kind: str,
) -> None:
    from clipperx.structured_client import StructuredResponseError
    from clipperx.tests.test_structured_client import ProtocolResponse
    from clipperx.visual_contracts import VisualDirectorResponse

    session = DiagnosticSession(ProtocolResponse(content, envelope_error=envelope, usage={
        "prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125,
        "completion_tokens_details": {"reasoning_tokens": 10}}))
    measured = benchmark.MeasuredClient(diagnostic_client(session))
    with pytest.raises(StructuredResponseError):
        measured.generate(VisualDirectorResponse, system_prompt="s",
                          user_prompt=json.dumps({"frame_manifest": []}))
    record = measured.records[0]
    assert record["failure_stage"] == stage and record["failure_kind"] == kind
    assert record["finish_reason"] == (None if envelope else "length")
    assert record["total_tokens"] == (None if envelope else 125)
    assert record.get("reasoning_tokens") == (None if envelope else 10)
    assert record["schema_valid"] is False and session.calls == 1
    path = tmp_path / "safe-metrics.json"
    benchmark.write_json(path, measured.records)
    assert PLANTED_SECRET not in path.read_text() + repr(record)


def test_measured_requests_json_decode_is_not_model_schema_failure() -> None:
    from requests.exceptions import JSONDecodeError
    from clipperx.planner_contracts import DiscoveryBatchResponse

    class DecodeFailureSession(DiagnosticSession):
        def post(self, *args: object, **kwargs: object) -> DiagnosticResponse:
            raise JSONDecodeError(PLANTED_SECRET, PLANTED_SECRET, 0)

    measured = benchmark.MeasuredClient(diagnostic_client(
        DecodeFailureSession(DiagnosticResponse(200))))
    with pytest.raises(JSONDecodeError):
        measured.generate(DiscoveryBatchResponse, system_prompt="s",
                          user_prompt=json.dumps({"frame_manifest": []}))
    assert measured.records[0]["failure_stage"] == "response_envelope"
    assert measured.records[0]["total_tokens"] is None
    assert PLANTED_SECRET not in json.dumps(measured.records)


def test_measured_unknown_legacy_failure_remains_unknown() -> None:
    from clipperx.planner_contracts import DiscoveryBatchResponse

    class UnknownFailureSession(DiagnosticSession):
        def post(self, *args: object, **kwargs: object) -> DiagnosticResponse:
            raise RuntimeError(PLANTED_SECRET)

    measured = benchmark.MeasuredClient(diagnostic_client(
        UnknownFailureSession(DiagnosticResponse(200))))
    with pytest.raises(RuntimeError):
        measured.generate(DiscoveryBatchResponse, system_prompt="s",
                          user_prompt=json.dumps({"frame_manifest": []}))
    record = measured.records[0]
    assert not {"failure_stage", "failure_kind", "finish_reason", "validation_issues"}.intersection(record)
    assert all(record[key] is None for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
    assert PLANTED_SECRET not in json.dumps(record)


def test_benchmark_limits_initial_plus_one_followup(scene: Any, tmp_path: Path) -> None:
    from clipperx.tests.test_visual_director import more
    video, clip, timeline, library = scene
    client = FakeClient([more(1), more(2), ready])
    result = benchmark.run_case({"case_id": "bounded", "expected": {
        "source_ranges": [[60, 62], [62, 64]]}}, video, clip, timeline, library, client, tmp_path)
    assert result["calls"] == len(client.payloads) == 2
    assert result["evidence_iterations"] == 1
    assert result["passed"] is False
    assert all(len(images) <= 8 for images in client.images)


def test_unpriced_selection_does_not_claim_cheapest() -> None:
    result = benchmark.select_model([{"model": "gpt-6-luna(low)", "passed": True,
                                      "estimated_provider_cost": None}])
    assert result["selected_model"] == "gpt-6-luna(low)"
    assert result["cheapest_passing_model"] is None
    assert result["selection_basis"] == "owner_selected_only_passing_target_unpriced"
    assert benchmark.select_model([])["status"] == "none_passed"


def test_portable_cases_reject_absolute_or_transcript(tmp_path: Path) -> None:
    data = json.loads((ROOT / "data/model_validation/visual_director_cases.json").read_text())
    data["cases"][0]["image_paths"] = ["C:/private/frame.jpg"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="portable"):
        benchmark.load_cases(path)


def test_selected_metadata_is_honest_until_actual_execution() -> None:
    data = json.loads((ROOT / "data/model_validation/selected_visual_model.json").read_text())
    if data["status"] == "not_run":
        assert data["selected_model"] is None
        assert data["results"] == []
    elif data["selected_model"] is not None:
        assert len(data["results"]) == 3
        assert all(row["passed"] and Path(row["artifact"]).is_file() for row in data["results"])
        assert data["run_signature"]
    else:
        assert data["status"] in {"incomplete", "none_passed"}


def test_aggregation_requires_all_three_matching_cases(tmp_path: Path) -> None:
    ids = ["stable_gameplay_webcam", "full_camera", "full_camera_to_browser"]
    for case_id in ids[:2]:
        benchmark.write_json(tmp_path / case_id / "result.json", {
            "case_id": case_id, "run_signature": "same", "model": benchmark.OWNER_MODEL,
            "passed": True})
    with pytest.raises(ValueError, match="three"):
        benchmark.aggregate_cases(tmp_path, ids, "same")
    benchmark.write_json(tmp_path / ids[2] / "result.json", {
        "case_id": ids[2], "run_signature": "different", "model": benchmark.OWNER_MODEL,
        "passed": True})
    with pytest.raises(ValueError, match="signature"):
        benchmark.aggregate_cases(tmp_path, ids, "same")


def test_run_configuration_refuses_mixed_inputs(tmp_path: Path) -> None:
    benchmark.ensure_run_config(tmp_path, {"model": benchmark.OWNER_MODEL, "input": "one"})
    with pytest.raises(ValueError, match="configuration"):
        benchmark.ensure_run_config(tmp_path, {"model": benchmark.OWNER_MODEL, "input": "two"})


def test_aggregation_never_reads_credentials(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(benchmark, "read_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(benchmark, "prepare_case", lambda *args: (Path("source"), None, None))
    with pytest.raises(ValueError, match="run-dir"):
        benchmark.main(["--aggregate", "--models", benchmark.OWNER_MODEL])


def test_actual_cached_subsets_keep_source_and_local_face_authority() -> None:
    from clipperx.evidence import associated_face_positions
    mapping = ROOT / "data/validation_visual/local_sources.json"
    if not mapping.is_file():
        pytest.skip("Machine-local Phase1 source mapping absent")
    for case in benchmark.load_cases(ROOT / "data/model_validation/visual_director_cases.json"):
        video, clip, timeline = benchmark.prepare_case(case, mapping)
        assert video.is_file()
        assert timeline.source_fingerprint == case["timeline_subset"]["source_fingerprint"]
        assert timeline.duration_seconds <= 60
        assert clip.duration <= 14
        assert all(associated_face_positions(timeline, face) is not None
                   for face in timeline.face_tracks)


def test_client_configuration_is_explicit() -> None:
    client = benchmark.make_client(benchmark.OWNER_MODEL, "http://localhost:8317/v1", "fake")
    assert client.config.context_window_tokens == 65536
    assert client.config.max_output_tokens == 4096
    assert benchmark.LIMITS.max_calls_per_clip == 2
    assert benchmark.LIMITS.max_followups_per_segment == 1


def test_aggregate_reads_saved_fake_results_without_reinference(scene: Any, tmp_path: Path) -> None:
    video, clip, timeline, library = scene
    ids = ["one", "two", "three"]
    client = FakeClient([ready, ready, ready])
    for case_id in ids:
        case = {"case_id": case_id, "expected": {"source_ranges": [[60, 62], [62, 64]]}}
        result = benchmark.run_case(case, video, clip, timeline, library, client,
                                    tmp_path / case_id / "jobs")
        result.update(model=benchmark.OWNER_MODEL, run_signature="same")
        benchmark.write_json(tmp_path / case_id / "result.json", result)
    results = benchmark.aggregate_cases(tmp_path, ids, "same")
    assert len(results) == 3
    assert all(r["passed"] for r in results)
    assert len(client.payloads) == 3


def test_audit_records_exact_metrics_once(tmp_path: Path) -> None:
    path = tmp_path / "audit.md"
    path.write_text("Existing evidence\n", encoding="utf-8")
    data = {"run_signature": "same", "completed_utc": "2026-10-14T00:00:00Z",
        "status": "none_passed", "selected_model": None, "run_root": "local-run",
        "results": [{"case_id": "transition", "status": "fail", "source_ranges": [[24, 38]],
            "schema_success_rate": 1, "invariant_pass_rate": .5, "calls": 1,
            "evidence_iterations": 0, "prompt_tokens": 123, "completion_tokens": 45,
            "total_tokens": 168, "elapsed_seconds": 2.5,
            "call_metrics": [{"latency_seconds": 2}]}]}
    benchmark.append_audit(path, data)
    benchmark.append_audit(path, data)
    text = path.read_text(encoding="utf-8")
    assert text.count("<!-- visual-benchmark:same -->") == 1
    assert "123/45/168" in text and "none_passed" in text
    assert "unpriced" in text and "Existing evidence" in text


def test_cli_runs_one_case_at_a_time_then_aggregates_without_calls(
    scene: Any, tmp_path: Path, monkeypatch: Any,
) -> None:
    client, _, common, output = _prepare_cli_fake_results(scene, tmp_path, monkeypatch)
    monkeypatch.setattr(benchmark, "read_key", lambda: pytest.fail("aggregation credential read"))
    with pytest.raises(ValueError, match="already started"):
        benchmark.main(common + ["--case-id", "one", "--confirm-inference"])
    assert benchmark.main(common + ["--aggregate"]) == 0
    result = json.loads(output.read_text())
    assert result["selected_model"] == benchmark.OWNER_MODEL
    assert len(result["results"]) == len(client.payloads) == 3
    assert result["cheapest_passing_model"] is None


@pytest.mark.parametrize("change", ["case_path", "nested_transcript"])
def test_portable_cases_reject_unsafe_case_path_or_nested_transcript(
    tmp_path: Path, change: str,
) -> None:
    data = json.loads((ROOT / "data/model_validation/visual_director_cases.json").read_text())
    if change == "case_path":
        data["cases"][0]["case_id"] = "../other"
    else:
        data["cases"][0]["timeline_subset"]["transcript"] = "Full source text"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="case_id|transcript"):
        benchmark.load_cases(path)


def test_prepare_case_checks_cached_states() -> None:
    mapping = ROOT / "data/validation_visual/local_sources.json"
    if not mapping.is_file():
        pytest.skip("Machine-local Phase1 mapping absent")
    case = benchmark.load_cases(ROOT / "data/model_validation/visual_director_cases.json")[0]
    case["timeline_subset"]["states"] = ["full_camera"]
    with pytest.raises(ValueError, match="states"):
        benchmark.prepare_case(case, mapping)


def test_aggregation_rejects_fabricated_pass_without_public_artifacts(tmp_path: Path) -> None:
    for case_id in ["one", "two", "three"]:
        benchmark.write_json(tmp_path / case_id / "result.json", {"case_id": case_id,
            "run_signature": "same", "model": benchmark.OWNER_MODEL, "passed": True})
    with pytest.raises(ValueError, match="artifact|metrics"):
        benchmark.aggregate_cases(tmp_path, ["one", "two", "three"], "same")


@pytest.mark.parametrize("ref", ["/outside/cache.json", "\\outside\\cache.json"])
def test_portable_rejects_windows_root_relative_references(ref: str) -> None:
    assert benchmark.portable(ref) is False
    assert benchmark.portable("data/processed/cache.json") is True
    assert benchmark.portable("data\\processed\\cache.json") is True


@pytest.fixture
def isolated_code_tree(tmp_path: Path) -> Path:
    root = tmp_path / "isolated_code"
    (root / "clipperx").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts/benchmark_visual_models.py").write_text("# runner\n")
    for name in ["evidence", "visual_prompts", "visual_contracts", "style_retrieval", "structured_client"]:
        (root / f"clipperx/{name}.py").write_text(f"# {name}\n")
    return root


@pytest.mark.parametrize("dependency", [
    "evidence", "visual_prompts", "visual_contracts", "style_retrieval", "structured_client",
])
def test_pipeline_provenance_changes_for_dependency_only_edit(
    isolated_code_tree: Path, dependency: str,
) -> None:
    before = benchmark._pipeline_code_provenance(isolated_code_tree)
    assert before == benchmark._pipeline_code_provenance(isolated_code_tree)
    paths = [row["path"] for row in before["files"]]
    assert paths == sorted(paths)
    assert "scripts/benchmark_visual_models.py" in paths
    assert f"clipperx/{dependency}.py" in paths
    (isolated_code_tree / f"clipperx/{dependency}.py").write_text("# changed dependency only\n")
    after = benchmark._pipeline_code_provenance(isolated_code_tree)
    assert after["digest_sha256"] != before["digest_sha256"]


def test_pipeline_provenance_ignores_noncode_and_cache_files(isolated_code_tree: Path) -> None:
    before = benchmark._pipeline_code_provenance(isolated_code_tree)
    cache = isolated_code_tree / "clipperx/__pycache__"
    cache.mkdir()
    (cache / "generated.py").write_text("# cache is not pipeline source\n")
    (cache / "generated.pyc").write_bytes(b"cached")
    (isolated_code_tree / "clipperx/output.json").write_text("{}")
    (isolated_code_tree / ".env").write_text("UNUSED_TEST_VALUE=not-a-credential\n")
    assert benchmark._pipeline_code_provenance(isolated_code_tree) == before


@pytest.mark.parametrize("operation", ["reuse", "aggregate"])
def test_dependency_only_change_rejects_before_credentials_or_client(
    scene: Any, tmp_path: Path, monkeypatch: Any, isolated_code_tree: Path, operation: str,
) -> None:
    video, clip, timeline, library = scene
    cases: list[dict[str, Any]] = [
        {"case_id": case_id, "expected": {"source_ranges": [[60, 62], [62, 64]]}}
        for case_id in ["one", "two", "three"]]
    monkeypatch.setattr(benchmark, "load_cases", lambda path: cases)
    monkeypatch.setattr(benchmark, "prepare_case", lambda *args: (video, clip, timeline))
    monkeypatch.setattr(benchmark, "StyleLibrary", lambda path: library)
    original_provenance = getattr(benchmark, "_pipeline_code_provenance", None)
    def provenance() -> Any:
        if original_provenance is None:
            return {"probe": (isolated_code_tree / "clipperx/evidence.py").read_text()}
        return original_provenance(isolated_code_tree)
    monkeypatch.setattr(benchmark, "_pipeline_code_provenance", provenance, raising=False)
    client = FakeClient([ready, ready, ready])
    monkeypatch.setattr(benchmark, "make_client", lambda *args: client)
    monkeypatch.setattr(benchmark, "read_key", lambda: "fake")
    run = tmp_path / "run"
    output = tmp_path / "partial.json"
    audit = tmp_path / "audit.md"
    audit.write_text("Fake test evidence\n")
    common = ["--models", benchmark.OWNER_MODEL, "--run-dir", str(run),
              "--selection-output", str(output), "--audit-output", str(audit)]
    initial_ids = ["one", "two", "three"] if operation == "aggregate" else ["one"]
    for case_id in initial_ids:
        assert benchmark.main(common + ["--case-id", case_id, "--confirm-inference"]) == 0
    old_config = (run / "run_config.json").read_bytes()
    old_result = (run / "one/result.json").read_bytes()
    (isolated_code_tree / "clipperx/evidence.py").write_text("# changed dependency only\n")
    monkeypatch.setattr(benchmark, "read_key", lambda: pytest.fail("credential read after code drift"))
    monkeypatch.setattr(benchmark, "make_client", lambda *args: pytest.fail("client after code drift"))
    args = ["--aggregate"] if operation == "aggregate" else [
        "--case-id", "two", "--confirm-inference"]
    with pytest.raises(ValueError, match="configuration"):
        benchmark.main(common + args)
    assert (run / "run_config.json").read_bytes() == old_config
    assert (run / "one/result.json").read_bytes() == old_result
    if operation == "reuse":
        assert not (run / "two").exists()
    assert len(client.payloads) == len(initial_ids)
