"""One-operation coordination, persisted resume and truthful terminal status."""
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipperx import local_paths
from clipperx.render_assets import RenderAssetRegistry


def _module():
    assert importlib.util.find_spec("clipperx.autonomous_pipeline") is not None, "autonomous operation missing"
    return importlib.import_module("clipperx.autonomous_pipeline")


@dataclass
class Library:
    root: Path
    fingerprint: str = "style-test-fingerprint"


class Client:
    def generate(self, *args, **kwargs):
        pytest.fail("coordination test must not invoke a model")


@pytest.fixture
def flow(tmp_path, monkeypatch):
    module = _module()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"Source identity, not media for this boundary test")
    transcript = tmp_path / "transcript.json"
    transcript.write_text('[{"start":0,"end":2,"text":"Hola","words":[]}]', encoding="utf-8")
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps({"contract_version": 1,
        "source_fingerprint": local_paths.source_fingerprint(source), "duration_seconds": 2,
        "analyzer_version": "test", "model_hashes": {}, "sampling_config": {},
        "samples": [], "spans": [], "face_tracks": [], "webcam_tracks": []}), encoding="utf-8")
    monkeypatch.setattr(local_paths.paths, "jobs_dir", tmp_path / "jobs")
    monkeypatch.setattr(module, "_implementation_signature", lambda: "stable-coordinator-contract")
    counts = {name: 0 for name in ("transcript", "timeline", "editorial", "visual", "export")}
    observed = {}

    def prepare_transcript(video):
        counts["transcript"] += 1
        return transcript

    def prepare_timeline(video):
        counts["timeline"] += 1
        return timeline

    def editorial(video, text, **kwargs):
        counts["editorial"] += 1
        observed["editorial_limits"] = kwargs["limits"]
        target = tmp_path / f"editorial-{counts['editorial']}.json"
        target.write_text('{"contract_version":1,"clips":[],"rejected_candidate_ids":[],"unresolved_requests":[]}', encoding="utf-8")
        return target

    def visual(video, plan, **kwargs):
        counts["visual"] += 1
        observed["transcript_path"] = kwargs["transcript_path"]
        observed["visual_limits"] = kwargs["limits"]
        path = tmp_path / f"visual-{counts['visual']}.json"
        path.write_text(json.dumps({"manifest_version": 1, "status": "ready", "clips": []}), encoding="utf-8")
        return SimpleNamespace(visual_plan_manifest=path)

    def export(video, visual_path, destination, **kwargs):
        counts["export"] += 1
        observed["export_transcript"] = kwargs["transcript_path"]
        destination.mkdir(parents=True, exist_ok=True)
        media = destination / "clip.mp4"
        if not media.exists():
            media.write_bytes(b"External renderer output")
        manifest = destination / "export.json"
        summary = destination / "export-summary.json"
        manifest.write_text('{"version":1,"status":"rendered_unreviewed"}', encoding="utf-8")
        summary.write_text('{"status":"rendered_unreviewed","rendered_unreviewed_clips":1}', encoding="utf-8")
        return SimpleNamespace(manifest=manifest, summary=summary, clips=(media,))

    monkeypatch.setattr(module, "_ensure_transcript", prepare_transcript)
    monkeypatch.setattr(module, "_ensure_timeline", prepare_timeline)
    monkeypatch.setattr(module, "_run_editorial", editorial)
    monkeypatch.setattr(module, "_run_visual", visual)
    monkeypatch.setattr(module, "_export", export)
    return SimpleNamespace(module=module, source=source, transcript=transcript, timeline=timeline,
        output=tmp_path / "exports", library=Library(tmp_path / "styles"), client=Client(),
        counts=counts, observed=observed)


def execute(flow, **kwargs):
    return flow.module.run_autonomous_pipeline(flow.source, flow.output,
        client=flow.client, model_name="gpt-6-luna(low)", style_library=flow.library,
        assets=RenderAssetRegistry({}, {}, {}), **kwargs)


def test_single_operation_prepares_plans_and_exports_without_claiming_quality(flow):
    result = execute(flow)
    summary = json.loads(result.summary_path.read_text())
    assert summary["status"] == "rendered_unreviewed"
    assert summary["full_qa_passed"] is False and summary["user_approved"] is False
    assert result.clips and result.clips[0].is_file()
    assert flow.counts == dict(transcript=1, timeline=1, editorial=1, visual=1, export=1)
    assert flow.observed["editorial_limits"].target_clips is None
    assert flow.observed["visual_limits"].require_creator_overlays is True
    assert flow.observed["transcript_path"] == flow.observed["export_transcript"] == flow.transcript


def test_resume_reuses_unchanged_model_stages_but_revalidates_exports(flow):
    first = execute(flow)
    second = execute(flow)
    assert first.state_path == second.state_path and first.clips == second.clips
    assert flow.counts["editorial"] == flow.counts["visual"] == 1
    assert flow.counts["export"] == 2


def test_provided_analysis_is_not_recomputed_or_copied_over_canonical_asr(flow):
    before = flow.transcript.read_bytes()
    execute(flow, transcript_path=flow.transcript, source_timeline_path=flow.timeline)
    assert flow.counts["transcript"] == flow.counts["timeline"] == 0
    assert flow.transcript.read_bytes() == before


def test_changed_transcript_invalidates_model_stage_reuse(flow):
    execute(flow)
    flow.transcript.write_text('[{"start":0,"end":2,"text":"Distinto","words":[]}]', encoding="utf-8")
    execute(flow)
    assert flow.counts["editorial"] == flow.counts["visual"] == 2


def test_altered_cached_visual_manifest_is_not_silently_replaced(flow):
    result = execute(flow)
    result.visual_manifest.write_text('{"forged":true}', encoding="utf-8")
    with pytest.raises(ValueError):
        execute(flow)
    assert flow.counts["visual"] == 1


def test_stage_failure_persists_error_and_resumes_only_pending_stage(flow, monkeypatch):
    original = flow.module._run_visual
    def failure(*args, **kwargs):
        raise RuntimeError("PLANTED_PROVIDER_SECRET_MUST_NOT_BE_PERSISTED")
    monkeypatch.setattr(flow.module, "_run_visual", failure)
    with pytest.raises(RuntimeError):
        execute(flow)
    summaries = list((flow.source.parent / "jobs").rglob("autonomous-summary.json"))
    assert len(summaries) == 1
    assert "PLANTED_PROVIDER_SECRET" not in summaries[0].read_text()
    assert json.loads(summaries[0].read_text())["status"] == "error"
    monkeypatch.setattr(flow.module, "_run_visual", original)
    execute(flow)
    assert flow.counts["editorial"] == 1 and flow.counts["export"] == 1


def test_source_changed_during_planning_cannot_reach_export(flow, monkeypatch):
    original = flow.module._run_editorial
    def mutate(*args, **kwargs):
        path = original(*args, **kwargs)
        flow.source.write_bytes(b"Changed source during processing")
        return path
    monkeypatch.setattr(flow.module, "_run_editorial", mutate)
    with pytest.raises(ValueError):
        execute(flow)
    assert flow.counts["visual"] == flow.counts["export"] == 0


def test_wrong_source_timeline_cannot_trigger_editorial_inference(flow):
    timeline = json.loads(flow.timeline.read_text())
    timeline["source_fingerprint"] = "incorrect-source"
    flow.timeline.write_text(json.dumps(timeline), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        execute(flow)
    assert flow.counts["editorial"] == flow.counts["visual"] == flow.counts["export"] == 0


def test_budget_exhaustion_reports_values_and_keeps_later_stages_pending(flow, monkeypatch):
    from clipperx.planner_engine import PlannerBudgetExceeded, PlannerRunLimits
    def exhausted(*args, **kwargs):
        raise PlannerBudgetExceeded("discovery", 61000, 60000)
    monkeypatch.setattr(flow.module, "_run_editorial", exhausted)
    result = execute(flow, editorial_limits=PlannerRunLimits(target_clips=None, max_total_reported_tokens=60000))
    summary = json.loads(result.summary_path.read_text())
    assert summary["status"] == "budget_exhausted"
    assert summary["budget"] == {"stage": "discovery", "reported_tokens": 61000, "limit": 60000}
    assert not result.clips and flow.counts["visual"] == flow.counts["export"] == 0


def test_automatic_budget_expands_and_resumes_without_user_restart(flow, monkeypatch):
    from clipperx.planner_engine import PlannerBudgetExceeded
    original = flow.module._run_editorial
    budgets = []
    def first_exhausted(*args, **kwargs):
        budgets.append(kwargs["limits"].max_total_reported_tokens)
        if len(budgets) == 1:
            raise PlannerBudgetExceeded("discovery", 61000, 60000)
        return original(*args, **kwargs)
    monkeypatch.setattr(flow.module, "_run_editorial", first_exhausted)
    result = execute(flow)
    assert budgets == [60000, 120000]
    assert json.loads(result.summary_path.read_text())["status"] == "rendered_unreviewed"
    assert flow.counts["visual"] == flow.counts["export"] == 1


def test_automatic_boundary_budget_expands_only_for_unresolved_limits(flow, monkeypatch):
    original = flow.module._run_editorial
    budgets = []
    def partial_then_complete(*args, **kwargs):
        budgets.append(kwargs["limits"].max_boundary_calls)
        path = original(*args, **kwargs)
        count = len(budgets)
        (path.parent / "usage.json").write_text(json.dumps([
            {"stage": "boundary", "total_tokens": 10} for _ in range(budgets[-1])]), encoding="utf-8")
        (path.parent / "run_summary.json").write_text(json.dumps({
            "status": "partial" if count == 1 else "complete",
            "unresolved_due_to_limits": 1 if count == 1 else 0,
            "reported_tokens": budgets[-1] * 10,
        }), encoding="utf-8")
        return path
    monkeypatch.setattr(flow.module, "_run_editorial", partial_then_complete)
    result = execute(flow)
    assert budgets == [8, 16]
    assert json.loads(result.summary_path.read_text())["status"] == "rendered_unreviewed"


def test_raising_coverage_budget_keeps_completed_plans_without_new_model_work(flow):
    from clipperx.planner_engine import PlannerRunLimits
    low = PlannerRunLimits(target_clips=None, max_total_reported_tokens=60000, max_boundary_calls=8)
    high = PlannerRunLimits(target_clips=None, max_total_reported_tokens=120000, max_boundary_calls=16)
    first = execute(flow, editorial_limits=low)
    second = execute(flow, editorial_limits=high)
    assert first.state_path == second.state_path
    assert flow.counts["editorial"] == flow.counts["visual"] == 1
    assert flow.counts["export"] == 2


def test_lower_coverage_budget_is_rejected_without_mutating_existing_state(flow):
    from clipperx.planner_engine import PlannerRunLimits
    high = PlannerRunLimits(target_clips=None, max_total_reported_tokens=120000, max_boundary_calls=16)
    low = PlannerRunLimits(target_clips=None, max_total_reported_tokens=60000, max_boundary_calls=8)
    first = execute(flow, editorial_limits=high)
    before = first.state_path.read_bytes()
    with pytest.raises(ValueError, match="budget"):
        execute(flow, editorial_limits=low)
    assert first.state_path.read_bytes() == before


def test_raising_boundary_budget_retries_previously_unresolved_editorial_work(flow, monkeypatch):
    from clipperx.planner_engine import PlannerRunLimits
    original = flow.module._run_editorial
    def partial_then_complete(*args, **kwargs):
        path = original(*args, **kwargs)
        count = flow.counts["editorial"]
        (path.parent / "run_summary.json").write_text(json.dumps({
            "status": "partial" if count == 1 else "complete",
            "unresolved_due_to_limits": 1 if count == 1 else 0,
        }), encoding="utf-8")
        return path
    monkeypatch.setattr(flow.module, "_run_editorial", partial_then_complete)
    low = PlannerRunLimits(target_clips=None, max_total_reported_tokens=60000, max_boundary_calls=8)
    high = PlannerRunLimits(target_clips=None, max_total_reported_tokens=120000, max_boundary_calls=16)
    first = execute(flow, editorial_limits=low)
    assert json.loads(first.summary_path.read_text())["status"] == "partial"
    second = execute(flow, editorial_limits=high)
    assert first.state_path == second.state_path
    assert flow.counts["editorial"] == 2
    assert json.loads(second.summary_path.read_text())["status"] == "rendered_unreviewed"


def test_cancellation_retains_completed_editorial_stage(flow, monkeypatch):
    original = flow.module._run_visual
    monkeypatch.setattr(flow.module, "_run_visual", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        execute(flow)
    summaries = list((flow.source.parent / "jobs").rglob("autonomous-summary.json"))
    assert json.loads(summaries[0].read_text())["status"] == "cancelled"
    monkeypatch.setattr(flow.module, "_run_visual", original)
    execute(flow)
    assert flow.counts["editorial"] == 1
