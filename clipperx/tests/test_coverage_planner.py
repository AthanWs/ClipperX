"""Coverage orchestration with real batching, contracts, ledgers and disk receipts."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from clipperx import planner_engine as engine
from clipperx.planner_contracts import DiscoveryBatchResponse, GlobalCompositionResponse
from clipperx.planner_engine import PlannerRunLimits, run_editorial_planner
from clipperx.structured_client import StructuredResult


def test_null_target_enables_coverage_and_keeps_bounded_default() -> None:
    PlannerRunLimits(target_clips=None).validate()
    assert PlannerRunLimits().target_clips == 3


@pytest.mark.parametrize("field", (
    "target_clips", "max_candidates", "max_initial_evidence_requests",
    "max_boundary_calls", "max_images_per_request", "max_total_reported_tokens",
))
@pytest.mark.parametrize("value", (0, -1, True, 1.5, "2", None))
def test_only_target_accepts_none_and_integer_limits_stay_positive(field, value) -> None:
    if field == "target_clips" and value is None:
        return
    with pytest.raises(ValueError):
        replace(PlannerRunLimits(), **{field: value}).validate()


class CoverageClient:
    def __init__(self, *, hook_words=1, connected=False, bad=None, tokens=10):
        self.calls = []
        self.hook_words = hook_words
        self.connected = connected
        self.bad = bad
        self.tokens = tokens

    def generate(self, output_model, *, system_prompt, user_prompt, **kwargs):
        user = json.loads(user_prompt)
        self.calls.append((output_model.__name__, user))
        if output_model is DiscoveryBatchResponse:
            indices = range(user["new_start_index"], min(user["new_end_index"],
                user["new_start_index"] + user["run_limits"]["max_candidates"]))
            candidates = [dict(candidate_id=f"candidate-{index:04d}",
                topic_id="connected" if self.connected else f"topic-{index:04d}",
                transcript_ranges=[dict(start=index * 10, end=index * 10 + 2)],
                hook="contexto " * self.hook_words, payoff="desenlace",
                context_risk="solo transcript; imagen sin confirmar", confidence=0.4)
                for index in indices]
            if self.bad == "discovery_conflict":
                for candidate in candidates:
                    candidate["candidate_id"] = "conflict"
            value = output_model(batch_id=user["batch_id"], cursor_before=user["cursor_before"],
                cursor_after=user["cursor_after"], candidates=candidates)
        else:
            assert output_model is GlobalCompositionResponse
            clips = [dict(clip_id="local-clip" if self.bad != "duplicate_id" else "same",
                topic_id=candidate["topic_id"], headline=dict(overlay_text="Hook",
                    platform_title="Título"), segments=[dict(
                    source=candidate["transcript_ranges"][0], output_order=0,
                    narrative_role="hook", layout="full_content", reason="Evidencia de voz")],
                reasoning="Imagen sin confirmar; no se afirma intención")
                for candidate in user["candidates"]]
            if self.bad != "duplicate_id":
                for index, clip in enumerate(clips):
                    clip["clip_id"] = f"local-{index}"
            if self.bad == "duplicate_source" and len(clips) > 1:
                clips[1]["segments"] = clips[0]["segments"]
            if self.bad == "oversized":
                clips[0]["segments"][0]["source"]["end"] += 180
                value = output_model.model_construct(clips=clips)
            elif self.bad == "malformed":
                value = output_model.model_construct(clips=[{}])
            else:
                value = output_model(clips=clips)
        return StructuredResult(value=value, usage={"total_tokens": self.tokens}, raw_response={})


def inputs(tmp_path, *, segments=4, words=2):
    video = tmp_path / "video.mp4"
    transcript = tmp_path / "transcript.json"
    if not video.exists():
        video.write_bytes(b"original source")
        transcript.write_text(json.dumps([dict(start=i * 10, end=i * 10 + 2,
            text="historia " * words, words=[]) for i in range(segments)]), encoding="utf-8")
    return video, transcript


def run(tmp_path, client, *, context=8192, budget=60000, force=False):
    video, transcript = inputs(tmp_path)
    return run_editorial_planner(video, transcript, discovery_client=client,
        evidence_client=None, model_name="gpt-test", context_window_tokens=context,
        max_output_tokens=512, limits=PlannerRunLimits(target_clips=None,
            max_total_reported_tokens=budget), jobs_root=tmp_path / "jobs", force=force)


def summary(path):
    return json.loads((path.parent / "run_summary.json").read_text(encoding="utf-8"))


def test_coverage_exceeds_three_clips_across_real_batches_and_bounded_groups(tmp_path):
    inputs(tmp_path, segments=120, words=70)
    client = CoverageClient()
    path = run(tmp_path, client)
    plan = GlobalCompositionResponse.model_validate_json(path.read_text(encoding="utf-8"))
    discovery = [payload for kind, payload in client.calls if kind == "DiscoveryBatchResponse"]
    compositions = [payload for kind, payload in client.calls if kind == "GlobalCompositionResponse"]
    assert len(discovery) > 1
    assert len(plan.clips) > 3
    assert len(compositions) > 1
    assert all(len(payload["candidates"]) <= 12 for payload in compositions)
    assert len({clip.clip_id for clip in plan.clips}) == len(plan.clips)
    assert summary(path)["status"] == "complete"


def test_small_context_splits_independent_candidates_without_loss(tmp_path):
    client = CoverageClient(hook_words=220)
    path = run(tmp_path, client, context=4096)
    compositions = [payload for kind, payload in client.calls if kind == "GlobalCompositionResponse"]
    assert len(compositions) > 1
    assert len(GlobalCompositionResponse.model_validate_json(path.read_text(encoding="utf-8")).clips) == 4
    assert sum(len(payload["candidates"]) for payload in compositions) == 4


def test_oversized_topic_component_is_deferred_without_inference_or_truncation(tmp_path):
    client = CoverageClient(hook_words=1400, connected=True)
    path = run(tmp_path, client, context=4096)
    assert [kind for kind, _ in client.calls] == ["DiscoveryBatchResponse"]
    assert GlobalCompositionResponse.model_validate_json(path.read_text(encoding="utf-8")).clips == []
    report = summary(path)
    assert report["status"] == "partial"
    assert report["deferred_groups"] == 1
    manifest = json.loads((path.parent / "coverage_groups.json").read_text(encoding="utf-8"))
    group = manifest["groups"][0]
    assert group["status"] == "deferred"
    request = json.loads(Path(group["request_path"]).read_text(encoding="utf-8"))
    assert len(request["payload"]["candidates"]) == 4
    assert request["estimated_input_tokens"] > request["input_token_limit"]


def test_budget_exhaustion_preserves_completed_group_then_resume_makes_zero_duplicate_calls(tmp_path):
    client = CoverageClient(hook_words=220)
    with pytest.raises(engine.PlannerBudgetExceeded):
        run(tmp_path, client, context=4096, budget=20)
    root = next((tmp_path / "jobs").glob("*/planning/runs/*"))
    manifest = json.loads((root / "coverage_groups.json").read_text(encoding="utf-8"))
    assert sum(group["status"] == "complete" for group in manifest["groups"]) == 1
    assert summary(root / "composition_plan.json")["completed_groups"] == 1
    saved = next((root / "composition_groups").glob("*/plan.json")).read_bytes()
    resumed = CoverageClient(hook_words=220)
    with pytest.raises(engine.PlannerBudgetExceeded):
        run(tmp_path, resumed, context=4096, budget=20)
    assert resumed.calls == []
    assert next((root / "composition_groups").glob("*/plan.json")).read_bytes() == saved


def test_complete_coverage_resume_at_budget_ceiling_reuses_every_call(tmp_path):
    client = CoverageClient()
    path = run(tmp_path, client, budget=20)
    before = path.read_bytes()
    resumed = CoverageClient()
    assert run(tmp_path, resumed, budget=20) == path
    assert resumed.calls == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ("prompt", "identity", "transcript", "group"))
def test_changed_input_does_not_reuse_stale_group(tmp_path, monkeypatch, change):
    path = run(tmp_path, CoverageClient())
    if change == "prompt":
        monkeypatch.setattr(engine, "COMPOSITION_SYSTEM_PROMPT", engine.COMPOSITION_SYSTEM_PROMPT + " cambio")
    elif change == "identity":
        job = engine.JobStore.for_video(tmp_path / "video.mp4", jobs_root=tmp_path / "jobs")
        job.speaker_analysis_path.write_text('{"identity":"revised"}', encoding="utf-8")
    elif change == "transcript":
        transcript = tmp_path / "transcript.json"
        transcript.write_text(transcript.read_text(encoding="utf-8").replace("historia", "cambio"), encoding="utf-8")
    else:
        manifest = json.loads((path.parent / "coverage_groups.json").read_text(encoding="utf-8"))
        request = Path(manifest["groups"][0]["request_path"])
        payload = json.loads(request.read_text(encoding="utf-8"))
        payload["payload"]["candidates"][0]["hook"] = "tampered"
        request.write_text(json.dumps(payload), encoding="utf-8")
    resumed = CoverageClient()
    if change == "group":
        with pytest.raises(ValueError, match="request|identity|hash"):
            run(tmp_path, resumed)
        assert resumed.calls == []
    else:
        replacement = run(tmp_path, resumed)
        assert any(kind == "GlobalCompositionResponse" for kind, _ in resumed.calls)
        assert path.exists()
        if change != "transcript":
            assert replacement == path
            assert list((path.parent / "composition_history").glob("*/composition_plan.json"))


@pytest.mark.parametrize("bad", ("duplicate_id", "duplicate_source", "oversized", "malformed"))
def test_invalid_composition_output_fails_without_publishing_global_plan(tmp_path, bad):
    with pytest.raises(ValueError):
        run(tmp_path, CoverageClient(bad=bad))
    root = next((tmp_path / "jobs").glob("*/planning/runs/*"))
    assert not (root / "composition_plan.json").exists()


def test_conflicting_discovery_ids_are_not_overwritten(tmp_path):
    with pytest.raises(ValueError, match="candidate_id"):
        run(tmp_path, CoverageClient(bad="discovery_conflict"))
    root = next((tmp_path / "jobs").glob("*/planning/runs/*"))
    assert json.loads((root / "ledger.json").read_text(encoding="utf-8"))["candidates"] == {}


def test_cached_group_still_validates_179_seconds_after_matching_hash(tmp_path):
    path = run(tmp_path, CoverageClient())
    manifest = json.loads((path.parent / "coverage_groups.json").read_text(encoding="utf-8"))
    group = manifest["groups"][0]
    plan_path = Path(group["plan_path"])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["clips"][0]["segments"][0]["source"]["end"] += 180
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path = plan_path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    resumed = CoverageClient()
    with pytest.raises(ValueError, match="179"):
        run(tmp_path, resumed)
    assert resumed.calls == []


def test_changed_discovery_prompt_invalidates_discovery_and_groups(tmp_path, monkeypatch):
    path = run(tmp_path, CoverageClient())
    before = (path.parent / "ledger.json").read_bytes()
    monkeypatch.setattr(engine, "DISCOVERY_SYSTEM_PROMPT", engine.DISCOVERY_SYSTEM_PROMPT + " otra instrucción")
    resumed = CoverageClient()
    assert run(tmp_path, resumed) == path
    assert [kind for kind, _ in resumed.calls] == ["DiscoveryBatchResponse", "GlobalCompositionResponse"]
    assert any(item.read_bytes() == before for item in
        (path.parent / "discovery_history").glob("*/ledger.json"))


def test_changed_cached_discovery_candidate_is_rejected_without_repurchase(tmp_path):
    path = run(tmp_path, CoverageClient())
    ledger_path = path.parent / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    next(iter(ledger["candidates"].values()))["hook"] = "untrusted modified hook"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    resumed = CoverageClient()
    with pytest.raises(ValueError, match="discovery|candidate|ledger"):
        run(tmp_path, resumed)
    assert resumed.calls == []


def test_review_cache_identity_keeps_matching_cache_and_defers_stale_cache_at_ceiling(tmp_path):
    from clipperx.planner_contracts import (BoundaryReviewResponse, EvidenceRequest)

    request = EvidenceRequest.model_validate(dict(request_id="review", candidate_id="c1",
        kind="context", direction="inside", window=dict(start=0, end=2), question="¿Qué sucede?"))
    candidates = [dict(candidate_id="c1", transcript_ranges=[dict(start=0, end=2)],
        evidence_requests=[request.model_dump(mode="json")])]
    identity = {"source_fingerprint": "source", "transcript_sha256": "transcript", "model": "fake"}
    key = engine._review_cache_key(request, candidates, identity)
    output = tmp_path / "reviews" / f"review-{key[:12]}.json"
    output.parent.mkdir()
    review = BoundaryReviewResponse.model_validate(dict(candidate_id="c1", action="accept",
        selected_ranges=[dict(start=0, end=2)], reasoning="evidence", confidence=0.5))
    output.write_text(review.model_dump_json(), encoding="utf-8")
    receipt = {"version": 1, "kind": "editorial_review_cache", "request_sha256": key,
        "plan_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    client = CoverageClient()
    kwargs = dict(evidence_client=client, output_dir=output.parent,
        usage_records=[dict(stage="boundary", total_tokens=1)], usage_path=tmp_path / "usage.json",
        limits=PlannerRunLimits(target_clips=None, max_total_reported_tokens=1), force=False)
    reviews, unresolved = engine._run_evidence_reviews(tmp_path / "source.mp4",
        tmp_path / "transcript.json", candidates, cache_identity=identity, **kwargs)
    assert reviews == [review.model_dump(mode="json")]
    assert unresolved == []
    reviews, unresolved = engine._run_evidence_reviews(tmp_path / "source.mp4",
        tmp_path / "transcript.json", candidates,
        cache_identity={**identity, "transcript_sha256": "changed"}, **kwargs)
    assert reviews == []
    assert unresolved == [request.model_dump(mode="json")]
    assert client.calls == []


def test_related_candidates_and_shared_topics_remain_one_component():
    from clipperx.composition_groups import connected_components

    candidates = [dict(candidate_id="a", topic_id="t1"), dict(candidate_id="b", topic_id="t2"),
        dict(candidate_id="c", topic_id="t2"), dict(candidate_id="d", topic_id="t3")]
    reviews = [dict(candidate_id="a", related_candidate_ids=["b"])]
    assert [[item["candidate_id"] for item in component] for component in
        connected_components(candidates, reviews)] == [["a", "b", "c"], ["d"]]


def test_group_candidate_bound_defers_whole_related_component():
    from clipperx.composition_groups import build_composition_groups

    candidates = [dict(candidate_id=f"c{i:02d}", topic_id=f"t{i}",
        transcript_ranges=[dict(start=i * 10, end=i * 10 + 2)]) for i in range(13)]
    reviews = [dict(candidate_id=f"c{i:02d}", related_candidate_ids=[f"c{i + 1:02d}"])
        for i in range(12)]
    groups = build_composition_groups(candidates, reviews, [], system_prompt="test",
        identity={}, run_limits={"target_clips": None}, streamer_identity=None,
        context_window_tokens=100000, max_output_tokens=512, model="gpt-test")
    assert len(groups) == 1
    assert groups[0]["deferred_reason"] == "candidate_bound"
    assert len(groups[0]["request"]["payload"]["candidates"]) == 13


def test_persisted_request_count_matches_actual_system_user_and_strict_schema(tmp_path):
    from clipperx.context_budget import TiktokenCounter
    from clipperx.structured_client import _strict_json_schema

    path = run(tmp_path, CoverageClient(hook_words=220), context=4096)
    counter = TiktokenCounter("gpt-test")
    requests = list((path.parent / "composition_groups").glob("*/request.json"))
    assert len(requests) > 1
    for output in requests:
        request = json.loads(output.read_text(encoding="utf-8"))
        schema = json.dumps(_strict_json_schema(GlobalCompositionResponse), ensure_ascii=False)
        tokens = counter.count(f'{request["system_prompt"]}\n{request["user_prompt"]}\n{schema}')
        assert tokens == request["estimated_input_tokens"]
        assert tokens <= request["input_token_limit"]


def test_injected_extra_candidate_in_discovery_ledger_is_rejected(tmp_path):
    path = run(tmp_path, CoverageClient())
    ledger_path = path.parent / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    candidate = dict(next(iter(ledger["candidates"].values())))
    candidate["candidate_id"] = "injected"
    ledger["candidates"]["injected"] = candidate
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    client = CoverageClient()
    with pytest.raises(ValueError, match="ledger|candidate|discovery"):
        run(tmp_path, client)
    assert client.calls == []


def test_recomposition_history_keeps_original_usage_and_summary(tmp_path, monkeypatch):
    path = run(tmp_path, CoverageClient())
    usage = (path.parent / "usage.json").read_bytes()
    prior_summary = (path.parent / "run_summary.json").read_bytes()
    monkeypatch.setattr(engine, "COMPOSITION_SYSTEM_PROMPT", engine.COMPOSITION_SYSTEM_PROMPT + " edit")
    run(tmp_path, CoverageClient())
    history = next((path.parent / "composition_history").glob("*/composition_plan.json")).parent
    assert (history / "usage.json").read_bytes() == usage
    assert (history / "run_summary.json").read_bytes() == prior_summary


def test_missing_related_candidate_defers_group_instead_of_claiming_coverage():
    from clipperx.composition_groups import build_composition_groups

    groups = build_composition_groups([dict(candidate_id="a", topic_id="topic",
        transcript_ranges=[dict(start=0, end=2)])],
        [dict(candidate_id="a", related_candidate_ids=["unavailable"])], [],
        system_prompt="test", identity={}, run_limits={"target_clips": None},
        streamer_identity=None, context_window_tokens=8192, max_output_tokens=512, model="gpt-test")
    assert groups[0]["deferred_reason"] == "missing_related_candidates"
    assert groups[0]["request"]["missing_related_candidate_ids"] == ["unavailable"]


def test_saturated_discovery_splits_and_covers_all_segments(tmp_path):
    inputs(tmp_path, segments=10)
    client = CoverageClient()
    path = run(tmp_path, client)
    plan = GlobalCompositionResponse.model_validate_json(path.read_text(encoding="utf-8"))
    assert len(plan.clips) == 10
    calls = [payload for kind, payload in client.calls if kind == "DiscoveryBatchResponse"]
    assert [(c["new_start_index"], c["new_end_index"]) for c in calls] == [(0, 10), (0, 5), (5, 10)]
    assert summary(path)["status"] == "complete"
    resumed = CoverageClient()
    assert run(tmp_path, resumed) == path
    assert resumed.calls == []


def test_saturated_discovery_budget_resume_does_not_repeat_calls(tmp_path):
    inputs(tmp_path, segments=10)
    client = CoverageClient()
    with pytest.raises(engine.PlannerBudgetExceeded, match="discovery"):
        run(tmp_path, client, budget=20)
    assert len(client.calls) == 2
    resumed = CoverageClient()
    with pytest.raises(engine.PlannerBudgetExceeded, match="discovery"):
        run(tmp_path, resumed, budget=20)
    assert resumed.calls == []


def test_indivisible_saturated_discovery_is_explicitly_partial(tmp_path):
    class Saturated(CoverageClient):
        def generate(self, output_model, *, system_prompt, user_prompt, **kwargs):
            if output_model is not DiscoveryBatchResponse:
                return super().generate(output_model, system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
            user = json.loads(user_prompt)
            self.calls.append((output_model.__name__, user))
            candidates = [dict(candidate_id=f"item-{i}", topic_id=f"topic-{i}",
                transcript_ranges=[dict(start=0, end=2)], hook="hook", payoff="payoff",
                context_risk="unknown", confidence=0.5) for i in range(6)]
            return StructuredResult(value=output_model(batch_id=user["batch_id"],
                cursor_before=user["cursor_before"], cursor_after=user["cursor_after"],
                candidates=candidates), usage={"total_tokens": 10}, raw_response={})
    inputs(tmp_path, segments=1)
    client = Saturated()
    path = run(tmp_path, client)
    assert summary(path)["status"] == "partial"
    assert summary(path)["discovery_deferred"][0]["reason"] == "saturated_indivisible"
    state = json.loads((path.parent / "ledger.json").read_text(encoding="utf-8"))
    assert state["next_cursor"] == 0
    resumed = Saturated()
    run(tmp_path, resumed)
    assert resumed.calls == []


def test_unassigned_candidates_are_review_pending_without_losing_clips(tmp_path):
    class Incomplete(CoverageClient):
        def generate(self, output_model, *, system_prompt, user_prompt, **kwargs):
            result = super().generate(output_model, system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
            if output_model is GlobalCompositionResponse:
                result.value.clips = result.value.clips[:1]
            return result
    client = Incomplete()
    path = run(tmp_path, client)
    plan = GlobalCompositionResponse.model_validate_json(path.read_text(encoding="utf-8"))
    assert len(plan.clips) == 1
    report = summary(path)
    assert report["status"] == "partial"
    assert set(report["unassigned_candidate_ids"]) == {"candidate-0001", "candidate-0002", "candidate-0003"}
    resumed = Incomplete()
    run(tmp_path, resumed)
    assert resumed.calls == []


def test_completed_discovery_missing_receipt_refuses_repurchase(tmp_path):
    path = run(tmp_path, CoverageClient())
    for item in (path.parent / "discovery").glob("*/response.json"):
        item.unlink()
        (item.parent / "receipt.json").unlink()
    resumed = CoverageClient()
    with pytest.raises(ValueError, match="cache|receipt"):
        run(tmp_path, resumed)
    assert resumed.calls == []


def test_saturated_discovery_interruption_reuses_completed_leaves(tmp_path):
    class Interrupt(CoverageClient):
        def generate(self, output_model, *, system_prompt, user_prompt, **kwargs):
            user = json.loads(user_prompt)
            if output_model is DiscoveryBatchResponse and user["new_start_index"] == 5:
                raise RuntimeError("simulated interruption")
            return super().generate(output_model, system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
    inputs(tmp_path, segments=10)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(tmp_path, Interrupt())
    resumed = CoverageClient()
    path = run(tmp_path, resumed)
    discovery = [item for kind, item in resumed.calls if kind == "DiscoveryBatchResponse"]
    assert [(item["new_start_index"], item["new_end_index"]) for item in discovery] == [(5, 10)]
    assert len(GlobalCompositionResponse.model_validate_json(path.read_text(encoding="utf-8")).clips) == 10


def test_conflicting_leaf_ids_leave_parent_uncommitted(tmp_path):
    class Collision(CoverageClient):
        def generate(self, output_model, *, system_prompt, user_prompt, **kwargs):
            result = super().generate(output_model, system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
            user = json.loads(user_prompt)
            if output_model is DiscoveryBatchResponse and user["new_start_index"] == 5:
                result.value.candidates[0].candidate_id = "candidate-0000"
            return result
    inputs(tmp_path, segments=10)
    with pytest.raises(ValueError, match="candidate_id"):
        run(tmp_path, Collision())
    root = next((tmp_path / "jobs").glob("*/planning/runs/*"))
    state = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert state["next_cursor"] == 0 and not state["candidates"]
    retry = Collision()
    with pytest.raises(ValueError, match="candidate_id"):
        run(tmp_path, retry)
    assert retry.calls == []


def test_recorded_discovery_without_response_refuses_repurchase(tmp_path, monkeypatch):
    original_write = engine._write_json
    def interrupt(path, payload):
        if path.name == "response.json" and path.parent.parent.name == "discovery":
            raise OSError("simulated interrupted persistence")
        return original_write(path, payload)
    monkeypatch.setattr(engine, "_write_json", interrupt)
    with pytest.raises(OSError, match="interrupted persistence"):
        run(tmp_path, CoverageClient())
    monkeypatch.setattr(engine, "_write_json", original_write)
    resumed = CoverageClient()
    with pytest.raises(ValueError, match="cache|receipt"):
        run(tmp_path, resumed)
    assert resumed.calls == []


def test_increasing_token_budget_resumes_completed_group_without_repurchase(tmp_path):
    client = CoverageClient(hook_words=220)
    with pytest.raises(engine.PlannerBudgetExceeded):
        run(tmp_path, client, context=4096, budget=20)
    root = next((tmp_path / "jobs").glob("*/planning/runs/*"))
    before = json.loads((root / "coverage_groups.json").read_text(encoding="utf-8"))
    first = next(item for item in before["groups"] if item["status"] == "complete")
    completed_bytes = Path(first["plan_path"]).read_bytes()
    resumed = CoverageClient(hook_words=220)
    path = run(tmp_path, resumed, context=4096, budget=100)
    assert path.parent == root
    assert not [kind for kind, _ in resumed.calls if kind == "DiscoveryBatchResponse"]
    assert len(resumed.calls) == sum(item["status"] == "pending" for item in before["groups"])
    assert Path(first["plan_path"]).read_bytes() == completed_bytes
    assert len(GlobalCompositionResponse.model_validate_json(path.read_text(encoding="utf-8")).clips) == 4
    retry = CoverageClient(hook_words=220)
    assert run(tmp_path, retry, context=4096, budget=100) == path
    assert retry.calls == []


def test_decreasing_coverage_budget_refuses_before_config_changes(tmp_path):
    path = run(tmp_path, CoverageClient(), budget=100)
    config = (path.parent / "run_config.json").read_bytes()
    resumed = CoverageClient()
    with pytest.raises(ValueError, match="budget|limit"):
        run(tmp_path, resumed, budget=90)
    assert (path.parent / "run_config.json").read_bytes() == config
    assert resumed.calls == []


def test_increasing_budget_above_consumption_preserves_run_identity(tmp_path):
    client = CoverageClient()
    path = run(tmp_path, client, budget=20)
    resumed = CoverageClient()
    assert run(tmp_path, resumed, budget=100) == path
    assert resumed.calls == []


def test_raise_boundary_limit_reuses_unchanged_group_and_review(tmp_path, monkeypatch):
    from clipperx.planner_contracts import BoundaryReviewResponse
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"samples": [], "transcript": [], "visual_summary": {}}', encoding="utf-8")
    monkeypatch.setattr(engine, "collect_evidence", lambda *args, **kwargs: evidence)
    class WithReviews(CoverageClient):
        def generate(self, output_model, *, system_prompt, user_prompt, **kwargs):
            user = json.loads(user_prompt)
            if output_model is BoundaryReviewResponse:
                self.calls.append((output_model.__name__, user))
                request = user["request"]
                return StructuredResult(value=output_model(candidate_id=request["candidate_id"],
                    action="accept", selected_ranges=[request["window"]], reasoning="reviewed", confidence=0.8),
                    usage={"total_tokens": 10}, raw_response={})
            result = super().generate(output_model, system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
            if output_model is DiscoveryBatchResponse:
                for candidate in result.value.candidates[:2]:
                    item = candidate.transcript_ranges[0]
                    from clipperx.planner_contracts import EvidenceRequest
                    candidate.evidence_requests = [EvidenceRequest(request_id="r-" + candidate.candidate_id,
                        candidate_id=candidate.candidate_id, kind="context", direction="inside",
                        window=item, question="What happened?")]
            return result
    video, transcript = inputs(tmp_path)
    def execute(client, calls):
        return run_editorial_planner(video, transcript, discovery_client=client, evidence_client=client,
            model_name="gpt-test", context_window_tokens=4096, max_output_tokens=512,
            limits=PlannerRunLimits(target_clips=None, max_boundary_calls=calls), jobs_root=tmp_path / "jobs")
    first = WithReviews(hook_words=220)
    path = execute(first, 1)
    manifest = json.loads((path.parent / "coverage_groups.json").read_text(encoding="utf-8"))
    old_groups = {item["group_id"] for item in manifest["groups"]}
    assert sum(kind == "BoundaryReviewResponse" for kind, _ in first.calls) == 1
    assert summary(path)["status"] == "partial"
    assert summary(path)["unresolved_due_to_limits"] == 1
    second = WithReviews(hook_words=220)
    assert execute(second, 2) == path
    assert sum(kind == "BoundaryReviewResponse" for kind, _ in second.calls) == 1
    assert not any(kind == "DiscoveryBatchResponse" for kind, _ in second.calls)
    new_manifest = json.loads((path.parent / "coverage_groups.json").read_text(encoding="utf-8"))
    new_groups = {item["group_id"] for item in new_manifest["groups"]}
    assert len(old_groups & new_groups) == 1
    assert sum(kind == "GlobalCompositionResponse" for kind, _ in second.calls) == 1
    assert summary(path)["status"] == "complete"
    third = WithReviews(hook_words=220)
    assert execute(third, 2) == path and third.calls == []


def test_coverage_force_cannot_erase_spent_budget(tmp_path):
    path = run(tmp_path, CoverageClient(), budget=20)
    old_usage = (path.parent / "usage.json").read_bytes()
    client = CoverageClient()
    with pytest.raises(engine.PlannerBudgetExceeded):
        run(tmp_path, client, budget=20, force=True)
    assert client.calls == []
    assert (path.parent / "usage.json").read_bytes() == old_usage


def test_bounded_run_id_retains_legacy_budget_identity(tmp_path):
    transcript = tmp_path / "transcript.json"
    transcript.write_text("[]", encoding="utf-8")
    limits = PlannerRunLimits(max_total_reported_tokens=100)
    import dataclasses
    payload = json.dumps({"transcript_hash": hashlib.sha256(transcript.read_bytes()).hexdigest()[:16],
        "model": "gpt-test", "context_window_tokens": 8192, "max_output_tokens": 512,
        "limits": dataclasses.asdict(limits)}, sort_keys=True, separators=(",", ":"))
    expected = "transcript-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    assert engine._planning_run_id(transcript, model_name="gpt-test", context_window_tokens=8192,
        max_output_tokens=512, limits=limits) == expected
    assert engine._planning_run_id(transcript, model_name="gpt-test", context_window_tokens=8192,
        max_output_tokens=512, limits=replace(limits, max_total_reported_tokens=200)) != expected


def test_budget_raise_must_cover_reported_spend_even_after_call_crossed_limit(tmp_path):
    client = CoverageClient(tokens=50)
    with pytest.raises(engine.PlannerBudgetExceeded):
        run(tmp_path, client, budget=20)
    root = next((tmp_path / "jobs").glob("*/planning/runs/*"))
    old_config = (root / "run_config.json").read_bytes()
    resumed = CoverageClient(tokens=50)
    with pytest.raises(ValueError, match="budget|limit"):
        run(tmp_path, resumed, budget=30)
    assert resumed.calls == []
    assert (root / "run_config.json").read_bytes() == old_config
    with pytest.raises(engine.PlannerBudgetExceeded):
        run(tmp_path, resumed, budget=20)
    assert resumed.calls == []
    path = run(tmp_path, resumed, budget=60)
    assert path.parent == root
    assert [kind for kind, _ in resumed.calls] == ["GlobalCompositionResponse"]


def test_coverage_run_id_distinguishes_semantic_changes(tmp_path):
    transcript = tmp_path / "transcript.json"
    transcript.write_text("[]", encoding="utf-8")
    limits = PlannerRunLimits(target_clips=None)
    def run_id(**kwargs):
        values = dict(model_name="gpt-test", context_window_tokens=8192,
            max_output_tokens=512, limits=limits)
        values.update(kwargs)
        return engine._planning_run_id(transcript, **values)
    baseline = run_id()
    assert run_id(limits=replace(limits, max_total_reported_tokens=100000,
        max_boundary_calls=40)) == baseline
    assert run_id(limits=replace(limits, max_candidates=7)) != baseline
    assert run_id(limits=replace(limits, max_images_per_request=3)) != baseline
    assert run_id(model_name="another-model") != baseline
    assert run_id(context_window_tokens=16384) != baseline
    transcript.write_text('[{"text": "changed"}]', encoding="utf-8")
    assert run_id() != baseline
