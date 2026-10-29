import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from clipperx.planner_contracts import (
    BoundaryReviewResponse,
    CandidateSpan,
    ClipPlan,
    DiscoveryBatchResponse,
    EditSegment,
    EvidenceKind,
    EvidenceRequest,
    GlobalCompositionResponse,
    HeadlinePlan,
    ProbeDirection,
    ReviewAction,
    TimeRange,
)
from clipperx.planner_engine import PlannerRunLimits, run_editorial_planner
from clipperx.structured_client import StructuredResult

ModelT = TypeVar("ModelT", bound=BaseModel)


class FakeClient:
    def generate(
        self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
        images: list[Path] | None = None, temperature: float = 0.2,
    ) -> StructuredResult[ModelT]:
        value: BaseModel
        if output_model is DiscoveryBatchResponse:
            user = json.loads(user_prompt)
            value = DiscoveryBatchResponse(
                batch_id=user["batch_id"],
                cursor_before=user["cursor_before"],
                cursor_after=user["cursor_after"],
            )
        elif output_model is GlobalCompositionResponse:
            value = GlobalCompositionResponse(
                clips=[
                    ClipPlan(
                        clip_id="clip-1",
                        topic_id="topic-1",
                        headline=HeadlinePlan(overlay_text="Hook", platform_title="Título"),
                        segments=[
                            EditSegment(
                                source=TimeRange(start=0, end=2),
                                output_order=0,
                                narrative_role="hook",
                                layout="full_content",
                                reason="test",
                            )
                        ],
                        reasoning="test",
                    )
                ]
            )
        else:
            raise AssertionError(f"Modelo de salida inesperado: {output_model}")
        return StructuredResult(
            value=output_model.model_validate(value),
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw_response={},
        )


class IterativeFakeClient(FakeClient):
    def __init__(self) -> None:
        self.boundary_calls = 0

    def generate(
        self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
        images: list[Path] | None = None, temperature: float = 0.2,
    ) -> StructuredResult[ModelT]:
        value: BaseModel
        if output_model is DiscoveryBatchResponse:
            user = json.loads(user_prompt)
            value = DiscoveryBatchResponse(
                batch_id=user["batch_id"],
                cursor_before=user["cursor_before"],
                cursor_after=user["cursor_after"],
                candidates=[
                    CandidateSpan(
                        candidate_id="candidate-1",
                        topic_id="topic-1",
                        transcript_ranges=[TimeRange(start=0, end=2)],
                        hook="hook",
                        payoff="payoff",
                        context_risk="visual",
                        confidence=0.8,
                        evidence_requests=[_evidence_request(iteration=1, end=2)],
                    )
                ],
            )
        elif output_model is BoundaryReviewResponse:
            self.boundary_calls += 1
            request = json.loads(user_prompt)["request"]
            if request["iteration"] == 1:
                value = BoundaryReviewResponse(
                    candidate_id="candidate-1",
                    action=ReviewAction.NEED_MORE_EVIDENCE,
                    evidence_requests=[_evidence_request(iteration=2, end=3)],
                    reasoning="revisar más",
                    confidence=0.5,
                )
            else:
                value = BoundaryReviewResponse(
                    candidate_id="candidate-1",
                    action=ReviewAction.ACCEPT,
                    selected_ranges=[TimeRange(start=0, end=2)],
                    reasoning="suficiente",
                    confidence=0.9,
                )
        else:
            return super().generate(
                output_model, system_prompt=system_prompt, user_prompt=user_prompt,
                images=images, temperature=temperature,
            )
        return StructuredResult(
            value=output_model.model_validate(value),
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw_response={},
        )


def _evidence_request(*, iteration: int, end: float) -> EvidenceRequest:
    return EvidenceRequest(
        request_id="same-request-id",
        candidate_id="candidate-1",
        kind=EvidenceKind.VISUAL_REACTION,
        direction=ProbeDirection.INSIDE,
        window=TimeRange(start=0, end=end),
        question="¿Qué ocurre en pantalla?",
        iteration=iteration,
    )


def test_planner_engine_runs_discovery_and_composition_without_evidence(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"source")
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.0, "text": "hola", "words": []},
                {"start": 1.0, "end": 2.0, "text": "mundo", "words": []},
            ]
        ),
        encoding="utf-8",
    )

    result = run_editorial_planner(
        video,
        transcript,
        discovery_client=FakeClient(),
        evidence_client=None,
        model_name="gpt-test",
        context_window_tokens=8192,
        max_output_tokens=2048,
        jobs_root=tmp_path / "jobs",
    )

    payload = json.loads(Path(result).read_text(encoding="utf-8"))
    assert payload["clips"][0]["clip_id"] == "clip-1"
    manifest_path = next((tmp_path / "jobs").glob("*/planning/runs/*/batches/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["context_budget"]["reserved_output_tokens"] == 2048
    assert manifest["context_budget"]["overlap_tokens"] < manifest["input_token_budget"]


def test_planner_engine_rechecks_changed_request_and_uses_explicit_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"source")
    transcript = tmp_path / "external-transcript.json"
    transcript.write_text(
        json.dumps([{"start": 0.0, "end": 2.0, "text": "reacción", "words": []}]),
        encoding="utf-8",
    )
    collected: list[tuple[int, Path | None]] = []

    def fake_collect_evidence(
        video_path: Path, request: EvidenceRequest, *, transcript_path: Path | None = None,
        force: bool = False,
    ) -> Path:
        collected.append((request.iteration, transcript_path))
        output = tmp_path / f"evidence-{request.iteration}.json"
        output.write_text(
            json.dumps({"samples": [], "transcript": [], "visual_summary": {}}),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("clipperx.planner_engine.collect_evidence", fake_collect_evidence)
    client = IterativeFakeClient()

    result = run_editorial_planner(
        video,
        transcript,
        discovery_client=client,
        evidence_client=client,
        model_name="gpt-test",
        context_window_tokens=8192,
        max_output_tokens=2048,
        jobs_root=tmp_path / "jobs",
    )

    assert Path(result).exists()
    assert client.boundary_calls == 2
    assert collected == [(1, transcript), (2, transcript)]
    review_files = list((Path(result).parent / "reviews").glob("*.json"))
    assert len(review_files) == 2


def test_planner_engine_stops_boundary_search_at_run_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"source")
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps([{"start": 0.0, "end": 2.0, "text": "reacción", "words": []}]),
        encoding="utf-8",
    )

    def fake_collect_evidence(
        video_path: Path, request: EvidenceRequest, *, transcript_path: Path | None = None,
        force: bool = False,
    ) -> Path:
        output = tmp_path / f"evidence-{request.iteration}.json"
        output.write_text(
            json.dumps({"samples": [], "transcript": [], "visual_summary": {}}),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("clipperx.planner_engine.collect_evidence", fake_collect_evidence)
    client = IterativeFakeClient()
    result = run_editorial_planner(
        video,
        transcript,
        discovery_client=client,
        evidence_client=client,
        model_name="gpt-test",
        context_window_tokens=8192,
        max_output_tokens=2048,
        limits=PlannerRunLimits(
            target_clips=1,
            max_candidates=1,
            max_initial_evidence_requests=1,
            max_boundary_calls=1,
            max_images_per_request=1,
            max_total_reported_tokens=10_000,
        ),
        jobs_root=tmp_path / "jobs",
    )

    summary = json.loads((Path(result).parent / "run_summary.json").read_text(encoding="utf-8"))
    assert client.boundary_calls == 1
    assert summary["unresolved_due_to_limits"] == 1


class RecordingClient(FakeClient):
    def __init__(self) -> None:
        self.calls: list[type[BaseModel]] = []

    def generate(
        self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
        images: list[Path] | None = None, temperature: float = 0.2,
    ) -> StructuredResult[ModelT]:
        self.calls.append(output_model)
        return super().generate(output_model, system_prompt=system_prompt,
            user_prompt=user_prompt, images=images, temperature=temperature)


def _planner_inputs(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"source")
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps([
        {"start": 0.0, "end": 2.0, "text": "reaccion", "words": []},
    ]), encoding="utf-8")
    return video, transcript


def _run_recording_planner(
    tmp_path: Path, client: RecordingClient, *, token_limit: int = 60_000,
    force: bool = False,
) -> Path:
    return run_editorial_planner(tmp_path / "video.mp4", tmp_path / "transcript.json",
        discovery_client=client, evidence_client=None, model_name="gpt-test",
        context_window_tokens=8192, max_output_tokens=2048,
        limits=PlannerRunLimits(max_total_reported_tokens=token_limit),
        force=force, jobs_root=tmp_path / "jobs")


def test_identical_resume_reuses_composition_even_at_total_token_ceiling(tmp_path: Path) -> None:
    _planner_inputs(tmp_path)
    client = RecordingClient()
    plan = _run_recording_planner(tmp_path, client, token_limit=30)
    original = plan.read_bytes()
    usage = (plan.parent / "usage.json").read_bytes()
    resumed = RecordingClient()

    assert _run_recording_planner(tmp_path, resumed, token_limit=30) == plan

    assert resumed.calls == []
    assert plan.read_bytes() == original
    assert (plan.parent / "usage.json").read_bytes() == usage
    assert client.calls == [DiscoveryBatchResponse, GlobalCompositionResponse]


def test_composition_stops_before_new_call_when_discovery_consumes_token_budget(tmp_path: Path) -> None:
    _planner_inputs(tmp_path)
    client = RecordingClient()

    with pytest.raises(RuntimeError, match="budget.*composition"):
        _run_recording_planner(tmp_path, client, token_limit=15)

    assert client.calls == [DiscoveryBatchResponse]
    root = next((tmp_path / "jobs").glob("*/planning/runs/*"))
    assert not (root / "composition_plan.json").exists()
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "budget_exhausted"
    assert summary["stage"] == "composition"
    assert summary["reported_tokens"] == 15


def test_discovery_stops_with_pending_cursor_when_total_token_budget_is_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _planner_inputs(tmp_path)

    def prepare_batches(transcript: Path, output_dir: Path, **kwargs: Any) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        batches = []
        for index in range(2):
            batch_id = f"batch-{index}"
            path = output_dir / f"{batch_id}.json"
            payload = {"batch_id": batch_id, "cursor_before": index, "cursor_after": index + 1,
                "context_start_index": index, "new_start_index": index,
                "new_end_index": index + 1, "text": f"part {index}"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            batches.append({"batch_id": batch_id, "cursor_before": index,
                "cursor_after": index + 1, "path": str(path)})
        manifest = output_dir / "manifest.json"
        manifest.write_text(json.dumps({"batches": batches}), encoding="utf-8")
        return manifest

    monkeypatch.setattr("clipperx.planner_engine.prepare_transcript_batches", prepare_batches)
    client = RecordingClient()

    with pytest.raises(RuntimeError, match="budget.*discovery"):
        _run_recording_planner(tmp_path, client, token_limit=15)

    assert client.calls == [DiscoveryBatchResponse]
    root = next((tmp_path / "jobs").glob("*/planning/runs/*"))
    ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["next_cursor"] == 1
    assert ledger["batches"]["batch-0"]["status"] == "complete"
    assert ledger["batches"]["batch-1"]["status"] == "pending"
    assert not (root / "composition_plan.json").exists()
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "budget_exhausted"
    assert summary["stage"] == "discovery"


@pytest.mark.parametrize("ceiling", ("tokens", "calls"))
def test_cached_review_chain_is_loaded_before_new_call_budgets(
    tmp_path: Path, ceiling: str,
) -> None:
    from clipperx import planner_engine as engine

    initial = _evidence_request(iteration=1, end=2)
    follow_up = _evidence_request(iteration=2, end=3)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    cached = [BoundaryReviewResponse(candidate_id="candidate-1",
        action=ReviewAction.NEED_MORE_EVIDENCE, evidence_requests=[follow_up],
        reasoning="more", confidence=0.5),
        BoundaryReviewResponse(candidate_id="candidate-1", action=ReviewAction.ACCEPT,
            selected_ranges=[TimeRange(start=0, end=2)], reasoning="enough", confidence=0.9)]
    for request, review in zip((initial, follow_up), cached):
        path = reviews_dir / f"{engine._safe_name(request.request_id)}-{engine._request_key(request)[:12]}.json"
        path.write_text(review.model_dump_json(), encoding="utf-8")
    records = [{"stage": "boundary", "total_tokens": 15}]
    client = RecordingClient()

    reviews, unresolved = engine._run_evidence_reviews(tmp_path / "source.mp4",
        tmp_path / "transcript.json", [{"evidence_requests": [initial.model_dump(mode="json")]}],
        evidence_client=client, output_dir=reviews_dir, usage_records=records,
        usage_path=tmp_path / "usage.json", limits=PlannerRunLimits(
            max_boundary_calls=1 if ceiling == "calls" else 8,
            max_total_reported_tokens=15 if ceiling == "tokens" else 60_000), force=False)

    assert reviews == [review.model_dump(mode="json") for review in cached]
    assert unresolved == []
    assert client.calls == []
    assert records == [{"stage": "boundary", "total_tokens": 15}]


def test_uncached_review_does_not_hide_later_cached_review_at_budget_limit(tmp_path: Path) -> None:
    from clipperx import planner_engine as engine

    missing = _evidence_request(iteration=1, end=2)
    cached_request = missing.model_copy(update={"request_id": "cached", "candidate_id": "candidate-2"})
    cached = BoundaryReviewResponse(candidate_id="candidate-2", action=ReviewAction.ACCEPT,
        selected_ranges=[TimeRange(start=0, end=2)], reasoning="cached", confidence=0.9)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    path = reviews_dir / f"cached-{engine._request_key(cached_request)[:12]}.json"
    path.write_text(cached.model_dump_json(), encoding="utf-8")
    client = RecordingClient()

    reviews, unresolved = engine._run_evidence_reviews(tmp_path / "source.mp4",
        tmp_path / "transcript.json", [{"evidence_requests": [
            missing.model_dump(mode="json"), cached_request.model_dump(mode="json")]}],
        evidence_client=client, output_dir=reviews_dir,
        usage_records=[{"stage": "boundary", "total_tokens": 15}],
        usage_path=tmp_path / "usage.json", limits=PlannerRunLimits(max_boundary_calls=1), force=False)

    assert reviews == [cached.model_dump(mode="json")]
    assert unresolved == [missing.model_dump(mode="json")]
    assert client.calls == []


def test_cached_boundary_review_requires_matching_candidate_even_at_budget_limit(tmp_path: Path) -> None:
    from clipperx import planner_engine as engine

    request = _evidence_request(iteration=1, end=2)
    cached = BoundaryReviewResponse(candidate_id="wrong-candidate", action=ReviewAction.ACCEPT,
        selected_ranges=[TimeRange(start=0, end=2)], reasoning="wrong", confidence=0.9)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    path = reviews_dir / f"{request.request_id}-{engine._request_key(request)[:12]}.json"
    path.write_text(cached.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate"):
        engine._run_evidence_reviews(tmp_path / "source.mp4", tmp_path / "transcript.json",
            [{"evidence_requests": [request.model_dump(mode="json")]}],
            evidence_client=RecordingClient(), output_dir=reviews_dir,
            usage_records=[{"stage": "boundary", "total_tokens": 15}],
            usage_path=tmp_path / "usage.json", limits=PlannerRunLimits(max_boundary_calls=1), force=False)


@pytest.mark.parametrize("change", ("prompt", "identity", "legacy", "force"))
def test_changed_composition_regenerates_and_preserves_previous_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    from clipperx import planner_engine as engine

    _planner_inputs(tmp_path)
    plan = _run_recording_planner(tmp_path, RecordingClient())
    original = plan.read_bytes()
    prior_usage = (plan.parent / "usage.json").read_bytes()
    if change == "prompt":
        monkeypatch.setattr(engine, "COMPOSITION_SYSTEM_PROMPT", engine.COMPOSITION_SYSTEM_PROMPT + " changed")
    elif change == "identity":
        job = engine.JobStore.for_video(tmp_path / "video.mp4", jobs_root=tmp_path / "jobs")
        job.speaker_analysis_path.write_text(json.dumps({"identity": "confirmed"}), encoding="utf-8")
    elif change == "legacy":
        (plan.parent / "composition_receipt.json").unlink()
    resumed = RecordingClient()

    assert _run_recording_planner(tmp_path, resumed, force=change == "force") == plan

    expected = ([DiscoveryBatchResponse, GlobalCompositionResponse] if change == "force"
        else [GlobalCompositionResponse])
    assert resumed.calls == expected
    history = list((plan.parent / "composition_history").glob("*/composition_plan.json"))
    assert len(history) == 1
    assert history[0].read_bytes() == original
    assert (history[0].parent / "usage.json").read_bytes() == prior_usage


def test_tampered_composition_is_rejected_without_inference(tmp_path: Path) -> None:
    _planner_inputs(tmp_path)
    plan = _run_recording_planner(tmp_path, RecordingClient())
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["clips"][0]["headline"]["overlay_text"] = "tampered"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    resumed = RecordingClient()

    with pytest.raises(ValueError, match="composition.*(hash|digest|integrity)"):
        _run_recording_planner(tmp_path, resumed)

    assert resumed.calls == []


def test_changed_composition_at_budget_ceiling_preserves_prior_plan_without_new_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clipperx import planner_engine as engine

    _planner_inputs(tmp_path)
    plan = _run_recording_planner(tmp_path, RecordingClient(), token_limit=30)
    original = plan.read_bytes()
    receipt = (plan.parent / "composition_receipt.json").read_bytes()
    monkeypatch.setattr(engine, "EDIT_DIRECTION_RULES", engine.EDIT_DIRECTION_RULES + " changed")
    resumed = RecordingClient()

    with pytest.raises(engine.PlannerBudgetExceeded) as failure:
        _run_recording_planner(tmp_path, resumed, token_limit=30)

    assert failure.value.stage == "composition"
    assert failure.value.reported_tokens == failure.value.limit == 30
    assert resumed.calls == []
    assert plan.read_bytes() == original
    assert (plan.parent / "composition_receipt.json").read_bytes() == receipt


def test_composition_cache_keeps_179_second_contract_even_with_matching_plan_hash(tmp_path: Path) -> None:
    _planner_inputs(tmp_path)
    plan = _run_recording_planner(tmp_path, RecordingClient())
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["clips"][0]["segments"][0]["source"]["end"] = 180
    plan.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path = plan.parent / "composition_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    resumed = RecordingClient()

    with pytest.raises(ValueError, match="179"):
        _run_recording_planner(tmp_path, resumed)

    assert resumed.calls == []
