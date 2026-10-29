from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar,Any

from pydantic import BaseModel

import pytest
from typer.testing import CliRunner

from clipperx import cli, planner_engine as engine, visual_director
from clipperx.local_paths import source_fingerprint
from clipperx.planner_contracts import ClipPlan, DiscoveryBatchResponse, GlobalCompositionResponse
from clipperx.structured_client import StructuredResult
from clipperx.style_library import StyleLibrary
from clipperx.visual_contracts import SourceVisualTimeline
from clipperx.visual_director import VisualDirectorLimits
from clipperx.tests.test_evidence import install_observer


def canonical(value: BaseModel) -> str:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


def digest(value: BaseModel) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass
class Inputs:
    video: Path
    transcript: Path
    composition: GlobalCompositionResponse
    plan: Path
    timeline: SourceVisualTimeline
    timeline_path: Path
    library: StyleLibrary
    root: Path


class UnusedClient:
    def generate(
        self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
        images: list[Path] | None = None, temperature: float = 0.2,
    ) -> StructuredResult[ModelT]:
        raise AssertionError("test must not request inference")


@pytest.fixture
def inputs(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> Inputs:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"synthetic source identity only")
    install_observer(monkeypatch,video,duration=2)
    from clipperx.tests.test_style_retrieval import FakeEncoder
    monkeypatch.setattr("clipperx.style_retrieval.LocalStyleEncoder",FakeEncoder)
    transcript = tmp_path / "transcript.json"
    transcript.write_text('[{"start":0,"end":2,"text":"local narrative","words":[]}]')
    clip = ClipPlan.model_validate({
        "clip_id": "../../unsafe/clip", "topic_id": "topic",
        "headline": {"overlay_text": "Hook", "platform_title": "Title"},
        "segments": [{"source": {"start": 0, "end": 2}, "output_order": 0,
                      "narrative_role": "hook", "layout": "adaptive", "reason": "test"}],
        "reasoning": "test",
    })
    composition = GlobalCompositionResponse(clips=[clip])
    plan_path = tmp_path / "composition_plan.json"
    plan_path.write_text(composition.model_dump_json())
    timeline = SourceVisualTimeline.model_validate({
        "source_fingerprint": source_fingerprint(video), "duration_seconds": 2,
        "analyzer_version": "synthetic", "model_hashes": {}, "sampling_config": {},
        "samples": [], "spans": [], "face_tracks": [], "webcam_tracks": [],
    })
    timeline_path = tmp_path / "source_visual_timeline.json"
    timeline_path.write_text(timeline.model_dump_json())
    library = StyleLibrary(tmp_path / "styles")
    library.root.mkdir()
    return Inputs(video=video, transcript=transcript, composition=composition,
                           plan=plan_path, timeline=timeline, timeline_path=timeline_path,
                           library=library, root=tmp_path)


def review(path: Path, clip: ClipPlan, timeline: SourceVisualTimeline,
           transcript_path: Path | None = None) -> Path:
    path.mkdir(parents=True)
    receipt = path / "receipt.json"
    receipt.write_text(json.dumps({"status": "review_required", "reason": "needs evidence",
                                  "clip_id": clip.clip_id,
                                  "source_fingerprint": timeline.source_fingerprint, "calls": 0}))
    protocol(path,clip,timeline,transcript_path=transcript_path)
    return receipt


def protocol(path: Path,clip: ClipPlan,timeline: SourceVisualTimeline,
             transcript_path: Path | None = None) -> None:
    """Fake outputs obey the real host protocol; image/receipt/file work remains real."""
    import clipperx.evidence as e
    module=visual_director
    source=path.parent/"source.mp4"
    library=StyleLibrary(path.parent/"styles")
    context=e.prepare_visual_evidence_context(source)
    text = (dict(status="unverified_legacy", **module._file_record(transcript_path.resolve()))
            if transcript_path is not None else dict(status="missing",path=None,sha256=None,bytes=None))
    host=dict(origin=context.origin.record(),origin_id=context.origin.identity,
        upstream=module._upstream(source,timeline,path.parent/"source_visual_timeline.json",context),
        text=text)
    authority=dict(source=str(source.resolve()),job_root=str(path.parent/"jobs"),run_id=path.name,run_path=str(path),
        limits=module.asdict(VisualDirectorLimits()),
        fingerprints=module._fingerprints(clip,timeline,library.fingerprint),timeline=timeline.model_dump(mode="json"),
        editorial_clip=clip.model_dump(mode="json"),host_context=host,runtime=module._runtime_record(context),
        style_library_root=str(library.root.resolve()),style_images=[],selected_styles=[])
    module._write(path/"authority.json",authority)
    receipt=json.loads((path/"receipt.json").read_text())
    companion=dict(version=1,kind="visual_plan_provenance",run_id=path.name,clip_id=clip.clip_id,status=receipt["status"],host_context=host,
        source=str(source.resolve()),source_fingerprint=timeline.source_fingerprint,
        authority=module._file_record(path/"authority.json"),bundles=[],calls=[])
    if receipt["status"]=="ready":
        windows=[s.source for s in clip.segments]
        bundle_path=e.collect_visual_evidence(source,windows,timeline=timeline,context=context,
            output_path=path/"evidence"/"initial.json",max_images=5)
        bundle=json.loads(bundle_path.read_text())
        companion["bundles"]=[module._bundle_record(bundle_path,bundle,5)]
        manifest,_=module._manifest([dict(bundle,bundle_path=str(bundle_path))],[],library,VisualDirectorLimits())
        for row in manifest:
            row["upstream_context_sha256"]=module._hash(host["upstream"])
        module._write(path/"call-01"/"images.json",manifest)
        companion["calls"]=[dict(module._file_record(path/"call-01"/"images.json"),call=1,presented_assets=manifest,
            bundle_count=1,segment_indices=list(range(len(clip.segments))))]
        plan_path=path/"visual_plan.json"
        companion["plan"]=dict(module._file_record(plan_path),fingerprint=module._hash(json.loads(plan_path.read_text())))
        receipt["plan_sha256"]=module.file_sha256(plan_path)
    else:
        companion.update(failed_check="needs evidence",validation="last_validated_facts")
    module._write(path/"provenance.json",companion)
    receipt.update(version=1,kind="visual_director_receipt",run_id=path.name,source=str(source.resolve()),host_context=host,
        companion_path=str(path/"provenance.json"),companion_sha256=module.file_sha256(path/"provenance.json"))
    module._write(path/"receipt.json",receipt)


def ready(path: Path, clip: ClipPlan, timeline: SourceVisualTimeline, library: StyleLibrary) -> Path:
    path.mkdir(parents=True)
    plan = path / "visual_plan.json"
    identity = {
        "face_tracks": [track.model_dump(mode="json") for track in timeline.face_tracks],
        "webcam_tracks": [track.model_dump(mode="json") for track in timeline.webcam_tracks],
        "local_identity": [(span.span_id, span.identity_status.value, span.streamer_track_id,
                            span.webcam_track_id) for span in timeline.spans],
    }
    plan.write_text(json.dumps({
        "contract_version": 2, "clip_id": clip.clip_id,
        "source_fingerprint": timeline.source_fingerprint,
        "composition_fingerprint": digest(clip), "source_timeline_fingerprint": digest(timeline),
        "identity_fingerprint": hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                                                          sort_keys=True).encode()).hexdigest(),
        "style_library_fingerprint": library.fingerprint,
        "shots": [{"shot_id": "shot", "source": {"start": 0, "end": 2},
                   "output": {"start": 0, "end": 2}, "primary_source": "content",
                   "reason": "test", "confidence": 1, "evidence_ids": ["synthetic-evidence"],
                   "layers": [{"layer_id": "source", "kind": "source_video",
                               "source_ref": "source", "source_crop": {
                                   "x": 0, "y": 0, "width": 1, "height": 1},
                               "destination": {"x": 0, "y": 0, "width": 1, "height": 1},
                               "fill": "contain_blur", "blur_sigma": 12, "z_index": 0}]}],
    }))
    (path / "receipt.json").write_text(json.dumps({"status": "ready", "plan_path": str(plan),
        "clip_id": clip.clip_id, "source_fingerprint": timeline.source_fingerprint, "calls": 1}))
    protocol(path,clip,timeline)
    return plan


def visual_run(inputs: Inputs) -> engine.PlannerRunArtifacts:
    return engine.run_visual_planner(inputs.video, inputs.plan,
        source_timeline_path=inputs.timeline_path, client=UnusedClient(),
        style_library=inputs.library, limits=VisualDirectorLimits(), jobs_root=inputs.root / "jobs")


def test_planner_passes_exact_pinned_timeline_and_records_companion(inputs: Inputs,monkeypatch: pytest.MonkeyPatch) -> None:
    def director(video: Path,clip: ClipPlan,timeline: SourceVisualTimeline,**kwargs: Any) -> Path:
        assert kwargs.get("source_timeline_path")==inputs.timeline_path.resolve()
        return ready(inputs.root/"ready",clip,timeline,inputs.library)
    monkeypatch.setattr(visual_director,"run_visual_director",director)
    result=visual_run(inputs)
    entry=json.loads(result.visual_plan_manifest.read_text())["clips"][0]
    assert entry["provenance"] and entry["provenance_sha256"]


def test_planner_refuses_missing_host_companion(inputs: Inputs,monkeypatch: pytest.MonkeyPatch) -> None:
    def director(video: Path,clip: ClipPlan,timeline: SourceVisualTimeline,**kwargs: Any) -> Path:
        path=ready(inputs.root/"ready",clip,timeline,inputs.library)
        (path.parent/"provenance.json").unlink()
        return path
    monkeypatch.setattr(visual_director,"run_visual_director",director)
    with pytest.raises((ValueError,visual_director.DirectorPublicationError)):
        visual_run(inputs)


def test_planner_retains_verified_ready_on_committed_bookkeeping_error(inputs: Inputs,monkeypatch: pytest.MonkeyPatch) -> None:
    def director(video: Path,clip: ClipPlan,timeline: SourceVisualTimeline,**kwargs: Any) -> Path:
        path=ready(inputs.root/"committed",clip,timeline,inputs.library)
        raise visual_director.DirectorPublicationError("COMMITTED_BOOKKEEPING_FAILED",path.parent,path)
    monkeypatch.setattr(visual_director,"run_visual_director",director)
    result=visual_run(inputs)
    entry=json.loads(result.visual_plan_manifest.read_text())["clips"][0]
    assert entry["status"]=="ready" and entry["phase"]=="bookkeeping_error"
    assert Path(entry["visual_plan"]).is_file()


@pytest.mark.parametrize("target",["bundle","jpeg","authority","upstream"])
def test_planner_drift_cannot_be_published_ready(inputs: Inputs,monkeypatch: pytest.MonkeyPatch,target: str) -> None:
    def director(video: Path,clip: ClipPlan,timeline: SourceVisualTimeline,**kwargs: Any) -> Path:
        path=ready(inputs.root/"ready",clip,timeline,inputs.library)
        if target=="bundle":
            file=path.parent/"evidence"/"initial.json"
        elif target=="jpeg":
            file=next(path.parent.glob("evidence/initial/*.jpg"))
        elif target=="authority":
            file=path.parent/"authority.json"
        else:
            file=inputs.timeline_path
        file.write_bytes(file.read_bytes()+b" ")
        return path
    monkeypatch.setattr(visual_director,"run_visual_director",director)
    with pytest.raises((ValueError,visual_director.DirectorPublicationError)):
        visual_run(inputs)
    manifest=json.loads(next(inputs.root.glob("visual_runs/*/visual_plan_manifest.json")).read_text())
    assert manifest["status"]=="error" and manifest["clips"][0]["status"]=="error"


def integrated(inputs: Inputs) -> engine.PlannerRunArtifacts:
    return engine.run_editorial_and_visual_planner(inputs.video, inputs.transcript,
        editorial_client=EditorialClient(inputs.composition), visual_client=UnusedClient(),
        model_name="test-model", context_window_tokens=8192, max_output_tokens=2048,
        source_timeline_path=inputs.timeline_path, style_library=inputs.library,
        editorial_limits=engine.PlannerRunLimits(), visual_limits=VisualDirectorLimits(),
        jobs_root=inputs.root / "jobs")


class EditorialClient:
    def __init__(self, composition: GlobalCompositionResponse) -> None:
        self.composition = composition

    def generate(
        self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
        images: list[Path] | None = None, temperature: float = 0.2,
    ) -> StructuredResult[ModelT]:
        if output_model is DiscoveryBatchResponse:
            prompt = json.loads(user_prompt)
            value = output_model.model_validate({"batch_id": prompt["batch_id"],
                                 "cursor_before": prompt["cursor_before"],
                                 "cursor_after": prompt["cursor_after"]})
        else:
            assert output_model is GlobalCompositionResponse
            value = output_model.model_validate(self.composition.model_dump())
        return StructuredResult(value, {"total_tokens": 1}, {})


def test_visual_planning_consumes_persisted_composition_without_mutating_it(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    before = canonical(inputs.composition)

    def director(video: Path, clip: ClipPlan, timeline: SourceVisualTimeline,
                 **kwargs: object) -> Path:
        transcript = kwargs.get("transcript_path")
        receipt = review(inputs.root / "review", clip, timeline,
                         transcript_path=transcript if isinstance(transcript, Path) else None)
        clip.segments[0].source.end = 999
        clip.headline.overlay_text = "MUTATED"
        return receipt

    monkeypatch.setattr(visual_director, "run_visual_director", director)
    result = integrated(inputs)
    persisted = GlobalCompositionResponse.model_validate_json(result.composition_plan.read_text())
    assert canonical(persisted) == before == canonical(inputs.composition)


def test_engine_calls_visual_director_only_after_composition_path_exists(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    def director(video: Path, clip: ClipPlan, timeline: SourceVisualTimeline,
                 **kwargs: object) -> Path:
        paths = list((inputs.root / "jobs").glob("*/planning/runs/*/composition_plan.json"))
        assert len(paths) == 1
        persisted = GlobalCompositionResponse.model_validate_json(paths[0].read_text())
        assert canonical(persisted.clips[0]) == canonical(clip)
        transcript = kwargs.get("transcript_path")
        return review(inputs.root / "review", clip, timeline,
                      transcript_path=transcript if isinstance(transcript, Path) else None)

    monkeypatch.setattr(visual_director, "run_visual_director", director)
    assert integrated(inputs).visual_plan_manifest.is_file()


def test_wrapper_rejects_missing_persisted_composition(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "run_editorial_planner", lambda *a, **k: inputs.root / "missing.json")
    monkeypatch.setattr(visual_director, "run_visual_director", lambda *a, **k: pytest.fail("director"))
    with pytest.raises((FileNotFoundError, ValueError)):
        integrated(inputs)


def test_manifest_preserves_partial_success_and_links_both_versions(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs.composition.clips.append(inputs.composition.clips[0].model_copy(
        update={"clip_id": "other"}, deep=True))
    inputs.plan.write_text(inputs.composition.model_dump_json())

    def director(video: Path, clip: ClipPlan, timeline: SourceVisualTimeline,
                 **kwargs: object) -> Path:
        if clip.clip_id == "other":
            return review(inputs.root / "pending", clip, timeline)
        return ready(inputs.root / "success", clip, timeline, inputs.library)

    monkeypatch.setattr(visual_director, "run_visual_director", director)
    result = visual_run(inputs)
    manifest = json.loads(result.visual_plan_manifest.read_text())
    assert manifest["status"] == "partial"
    assert manifest["editorial_plan"] == str(inputs.plan.resolve())
    assert manifest["editorial_contract_version"] == 1
    assert manifest["visual_contract_version"] == 2
    assert [entry["status"] for entry in manifest["clips"]] == ["ready", "review_required"]
    assert Path(manifest["clips"][0]["visual_plan"]).is_file()
    assert manifest["clips"][1]["visual_plan"] is None
    assert Path(manifest["clips"][1]["receipt"]).is_file()
    assert json.loads(result.run_summary.read_text())["review_required_clips"] == 1
    assert result.run_summary.parent == result.visual_plan_manifest.parent


def test_empty_composition_has_empty_manifest_without_inference(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs.plan.write_text(GlobalCompositionResponse(clips=[]).model_dump_json())
    monkeypatch.setattr(visual_director, "run_visual_director", lambda *a, **k: pytest.fail("director"))
    result = visual_run(inputs)
    assert json.loads(result.visual_plan_manifest.read_text())["status"] == "empty"
    assert json.loads(result.visual_plan_manifest.read_text())["clips"] == []


@pytest.mark.parametrize("malformation", ["unknown", "fake_ready", "wrong_source", "wrong_clip",
                                          "missing_receipt", "wrong_fingerprint", "missing_version"])
def test_unknown_or_malformed_director_output_is_an_error(inputs: Inputs, monkeypatch: pytest.MonkeyPatch, malformation: str) -> None:
    def director(video: Path, clip: ClipPlan, timeline: SourceVisualTimeline,
                 **kwargs: object) -> Path:
        if malformation == "unknown":
            path = inputs.root / "unknown.json"
            path.write_text('{"status":"something"}')
            return path
        path = ready(inputs.root / "ready", clip, timeline, inputs.library)
        receipt = path.parent / "receipt.json"
        if malformation == "fake_ready":
            path.write_text('{"status":"ready"}')
        elif malformation == "missing_receipt":
            receipt.unlink()
        elif malformation == "missing_version":
            payload = json.loads(path.read_text())
            payload.pop("contract_version")
            path.write_text(json.dumps(payload))
        else:
            payload = json.loads(path.read_text())
            payload[{"wrong_source": "source_fingerprint", "wrong_clip": "clip_id",
                     "wrong_fingerprint": "composition_fingerprint"}[malformation]] = "wrong"
            path.write_text(json.dumps(payload))
        return path

    monkeypatch.setattr(visual_director, "run_visual_director", director)
    with pytest.raises((ValueError, FileNotFoundError,visual_director.DirectorPublicationError)):
        visual_run(inputs)


def cli_args(inputs: Inputs) -> list[str]:
    return ["plan-visual", str(inputs.video), str(inputs.plan), str(inputs.timeline_path),
            "--style-library", str(inputs.library.root), "--model", "test-model",
            "--context-window", "8192"]


def forbid_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*a: object, **k: object) -> None:
        pytest.fail("client constructed without confirmation")
    monkeypatch.setattr("clipperx.structured_client.OpenAICompatibleStructuredClient", forbidden)


def test_visual_plan_command_requires_confirmation_for_inference(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    forbid_client(monkeypatch)
    result = CliRunner().invoke(cli.app, cli_args(inputs))
    assert result.exit_code != 0
    assert "--confirm-inference" in result.output
    assert "No such command" not in result.output


def test_plan_visual_cli_invokes_post_composition_only(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.config import settings
    monkeypatch.setattr(settings, "llm_provider", "cliproxy")
    monkeypatch.setattr("clipperx.structured_client.OpenAICompatibleStructuredClient", lambda c: object())
    monkeypatch.setattr(engine, "run_editorial_planner", lambda *a, **k: pytest.fail("editorial"))
    monkeypatch.setattr(visual_director, "run_visual_director",
                        lambda v, c, t, **k: review(inputs.root / "review", c, t))
    result = CliRunner().invoke(cli.app, cli_args(inputs) + ["--confirm-inference"])
    assert result.exit_code == 0, result.exception
    assert "Visual plan manifest:" in result.output
    assert "review_required" in result.output


def test_local_style_extraction_never_constructs_client_even_with_configured_model(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.config import settings
    monkeypatch.setattr(settings, "llm_provider", "cliproxy")
    monkeypatch.setattr(settings, "openai_model", "configured-but-not-explicit")
    forbid_client(monkeypatch)
    observed: list[tuple[Path, dict[str, object]]] = []

    def analyze(path: Path, **kwargs: object) -> SimpleNamespace:
        observed.append((path, kwargs))
        return SimpleNamespace(clip_fingerprint=source_fingerprint(path),
                               evidence_path="evidence/local.json")

    monkeypatch.setattr("clipperx.style_analysis.analyze_edited_clip", analyze)
    result = CliRunner().invoke(cli.app, ["build-style-library", str(inputs.root),
                                         "--output-dir", str(inputs.library.root)])
    assert result.exit_code == 0, result.exception
    assert observed == [(inputs.video, {"style_library_root": inputs.library.root})]
    assert "local_only" in result.output


@pytest.mark.parametrize("extra", [["--model", "test", "--context-window", "8192"],
                                    ["--model", "test", "--confirm-inference"]])
def test_style_enrichment_requires_confirmation_and_explicit_context(inputs: Inputs, monkeypatch: pytest.MonkeyPatch, extra: list[str]) -> None:
    forbid_client(monkeypatch)
    result = CliRunner().invoke(cli.app, ["build-style-library", str(inputs.root),
                              "--output-dir", str(inputs.library.root)] + extra)
    assert result.exit_code != 0
    assert "No such command" not in result.output
    assert "--confirm-inference" in result.output or "--context-window" in result.output


def test_style_enrichment_cache_usage_counts_zero_new_tokens(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.config import settings
    monkeypatch.setattr(settings, "llm_provider", "cliproxy")
    monkeypatch.setattr("clipperx.structured_client.OpenAICompatibleStructuredClient", lambda c: object())
    monkeypatch.setattr("clipperx.style_analysis.analyze_edited_clip", lambda p, **k:
                        SimpleNamespace(clip_fingerprint=source_fingerprint(p), evidence_path="evidence/local.json"))
    monkeypatch.setattr("clipperx.style_library.build_style_card", lambda *a, **k:
                        SimpleNamespace(card=SimpleNamespace(card_id="style"),
                                        usage={"total_tokens": 99}, cache_hit=True))
    result = CliRunner().invoke(cli.app, ["build-style-library", str(inputs.root),
        "--output-dir", str(inputs.library.root), "--model", "test",
        "--context-window", "8192", "--confirm-inference"])
    assert result.exit_code == 0, result.exception
    manifest = json.loads((inputs.library.root / "build_manifest.json").read_text())
    assert manifest["new_inference_tokens"] == 0
    assert manifest["new_inference_calls"] == 0
    assert manifest["clips"][0]["historical_usage"] == {"total_tokens": 99}


def test_all_ready_runs_have_distinct_manifest_and_director_artifacts(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def director(video: Path, clip: ClipPlan, timeline: SourceVisualTimeline,
                 **kwargs: object) -> Path:
        calls.append(len(calls))
        return ready(inputs.root / f"director-{calls[-1]}", clip, timeline, inputs.library)

    monkeypatch.setattr(visual_director, "run_visual_director", director)
    first, second = visual_run(inputs), visual_run(inputs)
    assert first.visual_plan_manifest != second.visual_plan_manifest
    manifests = [json.loads(run.visual_plan_manifest.read_text()) for run in (first, second)]
    assert [manifest["status"] for manifest in manifests] == ["ready", "ready"]
    assert manifests[0]["clips"][0]["visual_plan"] != manifests[1]["clips"][0]["visual_plan"]
    assert "unsafe" not in str(first.visual_plan_manifest)


def test_error_manifest_preserves_success_and_remaining_pending(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs.composition.clips.extend([
        inputs.composition.clips[0].model_copy(update={"clip_id": name}, deep=True)
        for name in ("broken", "remaining")])
    inputs.plan.write_text(inputs.composition.model_dump_json())

    def director(video: Path, clip: ClipPlan, timeline: SourceVisualTimeline,
                 **kwargs: object) -> Path:
        if clip.clip_id == "broken":
            path = inputs.root / "unknown.json"
            path.write_text('{}')
            return path
        assert clip.clip_id != "remaining"
        return ready(inputs.root / "success", clip, timeline, inputs.library)

    monkeypatch.setattr(visual_director, "run_visual_director", director)
    with pytest.raises(ValueError, match="Unknown visual director artifact"):
        visual_run(inputs)
    manifest_path = next(inputs.root.glob("visual_runs/*/visual_plan_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "error"
    assert [item["status"] for item in manifest["clips"]] == ["ready", "error", "pending"]
    assert Path(manifest["clips"][0]["visual_plan"]).is_file()


def test_timeline_is_detached_and_input_drift_is_an_error(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    before = canonical(inputs.timeline)

    def director(video: Path, clip: ClipPlan, timeline: SourceVisualTimeline,
                 **kwargs: object) -> Path:
        result = review(inputs.root / "review", clip, timeline)
        timeline.duration_seconds = 999
        assert canonical(SourceVisualTimeline.model_validate_json(
            inputs.timeline_path.read_text())) == before
        inputs.plan.write_text(GlobalCompositionResponse(clips=[]).model_dump_json())
        return result

    monkeypatch.setattr(visual_director, "run_visual_director", director)
    with pytest.raises(ValueError, match="inputs changed"):
        visual_run(inputs)
    manifest = json.loads(next(inputs.root.glob("visual_runs/*/visual_plan_manifest.json")).read_text())
    assert manifest["status"] == "error"


@pytest.mark.parametrize("defect", ["wrong_source", "duplicate_clip"])
def test_invalid_input_associations_fail_before_director(inputs: Inputs, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    if defect == "wrong_source":
        inputs.timeline.source_fingerprint = "wrong"
        inputs.timeline_path.write_text(inputs.timeline.model_dump_json())
    else:
        inputs.composition.clips *= 2
        inputs.plan.write_text(inputs.composition.model_dump_json())
    monkeypatch.setattr(visual_director, "run_visual_director", lambda *a, **k: pytest.fail("director"))
    with pytest.raises(ValueError):
        visual_run(inputs)


@pytest.mark.parametrize("defect", ["status", "reason", "calls", "association"])
def test_review_receipts_must_be_well_formed(inputs: Inputs, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    def director(video: Path, clip: ClipPlan, timeline: SourceVisualTimeline,
                 **kwargs: object) -> Path:
        path = review(inputs.root / "review", clip, timeline)
        payload = json.loads(path.read_text())
        field, value = {"status": ("status", "ready"), "reason": ("reason", ""),
                        "calls": ("calls", True), "association": ("clip_id", "other")}[defect]
        payload[field] = value
        path.write_text(json.dumps(payload))
        return path

    monkeypatch.setattr(visual_director, "run_visual_director", director)
    with pytest.raises((ValueError,visual_director.DirectorPublicationError)):
        visual_run(inputs)


def test_style_evidence_cannot_escape_output_directory(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    forbid_client(monkeypatch)
    monkeypatch.setattr("clipperx.style_analysis.analyze_edited_clip", lambda p, **k:
                        SimpleNamespace(clip_fingerprint=source_fingerprint(p),
                                        evidence_path="../../outside.json"))
    result = CliRunner().invoke(cli.app, ["build-style-library", str(inputs.root),
        "--output-dir", str(inputs.library.root)])
    assert result.exit_code != 0
    assert "escapes output directory" in result.output


@pytest.mark.parametrize("target", ["composition", "timeline"])
def test_inputs_require_explicit_contract_versions(inputs: Inputs, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    path = inputs.plan if target == "composition" else inputs.timeline_path
    payload = json.loads(path.read_text())
    payload.pop("contract_version")
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(visual_director, "run_visual_director", lambda *a, **k: pytest.fail("director"))
    with pytest.raises(ValueError, match="contract_version"):
        visual_run(inputs)


@pytest.mark.parametrize("payload", [{"contract_version": 1, "other": "unknown"},
                                     {"contract_version": 1},
                                     {"contract_version": 1, "clips": [], "other": "unknown"}])
def test_unknown_composition_is_not_silently_an_empty_composition(inputs: Inputs, monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> None:
    inputs.plan.write_text(json.dumps(payload))
    monkeypatch.setattr(visual_director, "run_visual_director", lambda *a, **k: pytest.fail("director"))
    with pytest.raises(ValueError, match="composition"):
        visual_run(inputs)


def test_wrapper_runs_actual_director_with_synthetic_media_and_fake_clients(inputs: Inputs,monkeypatch: pytest.MonkeyPatch) -> None:
    video = inputs.root / "synthetic.avi"
    video.write_bytes(b"HOST identity placeholder, fake observation boundary")
    install_observer(monkeypatch,video,duration=3)
    inputs.video = video
    inputs.timeline.source_fingerprint = source_fingerprint(video)
    inputs.timeline.duration_seconds = 3
    inputs.timeline = SourceVisualTimeline.model_validate({
        **inputs.timeline.model_dump(mode="json"),
        "spans": [{"span_id": "span", "source": {"start": 0, "end": 2},
                   "state": "content_only", "confidence": 1, "identity_status": "unavailable",
                   "streamer_track_id": None, "webcam_track_id": None,
                   "important_regions": [{"x": 0, "y": 0, "width": 1, "height": 1}],
                   "text_regions": [], "evidence_ids": ["synthetic-evidence"],
                   "change_reasons": []}],
    })
    inputs.timeline_path.write_text(inputs.timeline.model_dump_json())

    class VisualClient:
        def generate(
            self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
            images: list[Path] | None = None, temperature: float = 0.2,
        ) -> StructuredResult[ModelT]:
            payload = json.loads(user_prompt)
            host = payload["clip"]["host_context"]
            assert images is not None
            assert len(images) <= 5
            assert all(path.is_file() for path in images)
            # Model response only; the Director itself publishes the host protocol.
            plan=dict(contract_version=2,clip_id=inputs.composition.clips[0].clip_id,**host["fingerprints"],
                shots=[dict(shot_id="shot",source=dict(start=0,end=2),output=dict(start=0,end=2),
                    primary_source="content",reason="test",confidence=1,evidence_ids=["synthetic-evidence"],
                    layers=[dict(layer_id="source",kind="source_video",source_ref="source",
                        source_crop=dict(x=0,y=0,width=1,height=1),destination=dict(x=0,y=0,width=1,height=1),
                        fill="contain_blur",blur_sigma=12,z_index=0)])])
            assert all(plan[name] == value for name, value in host["fingerprints"].items())
            return StructuredResult(output_model.model_validate({
                "status": "ready", "plan": plan, "reasoning": "synthetic evidence"}),
                {"total_tokens": 12}, {})

    before = canonical(inputs.composition)
    result = engine.run_editorial_and_visual_planner(
        inputs.video, inputs.transcript, editorial_client=EditorialClient(inputs.composition),
        visual_client=VisualClient(), model_name="test-model", context_window_tokens=8192,
        max_output_tokens=2048, source_timeline_path=inputs.timeline_path,
        style_library=inputs.library, editorial_limits=engine.PlannerRunLimits(),
        visual_limits=VisualDirectorLimits(), jobs_root=inputs.root / "jobs")
    manifest = json.loads(result.visual_plan_manifest.read_text())
    assert manifest["status"] == "ready", manifest
    assert canonical(GlobalCompositionResponse.model_validate_json(
        result.composition_plan.read_text())) == before
    entry = manifest["clips"][0]
    assert json.loads(Path(entry["visual_plan"]).read_text())["contract_version"] == 2
    assert json.loads(Path(entry["receipt"]).read_text())["status"] == "ready"
