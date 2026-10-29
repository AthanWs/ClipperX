from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from clipperx.jobs import JobStore
from clipperx.planner_contracts import ClipPlan
from clipperx.structured_client import StructuredResult
from clipperx.style_library import StyleLibrary
from clipperx.style_contracts import StyleCard
from clipperx.visual_contracts import SourceState, SourceVisualTimeline
from clipperx.visual_director import VisualDirectorLimits, run_visual_director


ModelT = TypeVar("ModelT", bound=BaseModel)


def manifest_frame(at: int,path: Path) -> dict[str,Any]:
    return dict(at=at,path=str(path),frame_id=f"frame-{at}",evidence_id=f"e-{at}",
        raw_pts=at,time_base="1/1",sample_refs=[dict(requested=f"{at}/1",window_indices=[0],
                                                  receipt="observation.json",frame_n=at)])

@pytest.fixture
def scene(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ClipPlan, SourceVisualTimeline, StyleLibrary]:
    from clipperx.tests.test_evidence import install_observer
    from clipperx.tests.test_style_retrieval import FakeEncoder
    video = tmp_path / "synthetic.avi"
    video.write_bytes(b"HOST identity only; never a media fixture")
    install_observer(monkeypatch,video)
    monkeypatch.setattr("clipperx.style_retrieval.LocalStyleEncoder",FakeEncoder)
    job = JobStore.for_video(video, jobs_root=tmp_path / "jobs")
    job.transcript_path.write_text(json.dumps([
        {"start": 60, "end": 61, "text": "local payoff", "speaker": "A"},
        {"start": 5, "end": 6, "text": "DISTANT_SECRET_SENTINEL"},
    ]), encoding="utf-8")
    clip = ClipPlan.model_validate({
        "clip_id": "clip-0001", "topic_id": "topic", "headline": {
            "overlay_text": "Local headline", "platform_title": "Wrong title"},
        "segments": [{"source": {"start": 60, "end": 64}, "output_order": 0,
                      "narrative_role": "payoff", "layout": "adaptive", "reason": "payoff"}],
        "reasoning": "preserve narrative",
    })
    timeline = SourceVisualTimeline.model_validate({
        "source_fingerprint": job.fingerprint, "duration_seconds": 100,
        "analyzer_version": "synthetic", "model_hashes": {}, "sampling_config": {},
        "samples": [], "face_tracks": [], "webcam_tracks": [],
        "spans": [dict(span_id=f"span-{i}", source={"start": start, "end": end},
                       state="content_only", confidence=1, streamer_track_id=None,
                       identity_status="unavailable", webcam_track_id=None,
                       important_regions=[dict(x=0, y=0, width=1, height=1)],
                       text_regions=[], evidence_ids=[f"e-{i}"], change_reasons=["cut"])
                  for i, (start, end) in enumerate([(0, 62), (62, 100)])],
    })
    return video, clip, timeline, StyleLibrary(tmp_path / "styles")


def ready(payload: dict[str, Any]) -> dict[str, Any]:
    host = payload["clip"]["host_context"]
    ranges = host["required_shots"]
    shots = []
    output = 0.0
    for i, source in enumerate(ranges):
        duration = source["end"] - source["start"]
        shots.append(dict(
            shot_id=f"shot-{i}", source=source,
            output=dict(start=output, end=output + duration), primary_source="content",
            reason="preserve payoff", layers=[dict(
                layer_id="source", kind="source_video", source_ref="source",
                source_crop=dict(x=0, y=0, width=1, height=1),
                destination=dict(x=0, y=0, width=1, height=1), fill="contain_blur",
                blur_sigma=12, z_index=0)], evidence_ids=["e-0" if source["start"] < 62 else "e-1"],
            confidence=1))
        output += duration
    return dict(status="ready", plan=dict(clip_id=payload["clip"]["clip_id"],
                **host["fingerprints"], shots=shots), reasoning="bounded evidence")


def more(index: int, start: float = 59, end: float = 63) -> dict[str, Any]:
    return dict(status="need_more_evidence", reasoning="verify boundary", evidence_requests=[dict(
        request_id=f"r-{index}", clip_id="clip-0001", source=dict(start=start, end=end),
        question=f"Boundary {index}?", sample_interval_seconds=0.5,
        reason="camera_boundary", iteration=index)])


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []
        self.images: list[list[Path]] = []
        self.output_models: list[type[BaseModel]] = []

    def generate(self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
                 images: list[Path] | None = None, temperature: float = 0.2) -> StructuredResult[ModelT]:
        payload = json.loads(user_prompt)
        self.output_models.append(output_model)
        self.payloads.append(payload)
        self.images.append(images or [])
        response = self.responses.pop(0)
        response = response(payload) if callable(response) else response
        return StructuredResult(output_model.model_validate(response), {"total_tokens": 10}, {})


def run(scene: Any, tmp_path: Path, client: FakeClient, **kwargs: Any) -> Path:
    video, clip, timeline, library = scene
    return run_visual_director(video, clip, timeline, client=client, style_library=library,
                               jobs_root=tmp_path / "jobs", **kwargs)


def reseal_terminal(root: Path,receipt: dict[str,Any],companion: dict[str,Any]) -> None:
    import clipperx.visual_director as module
    module._write(root/"provenance.json",companion)
    receipt["companion_sha256"]=module.file_sha256(root/"provenance.json")
    module._write(root/"receipt.json",receipt)


def pending_review_candidates(scene: Any,tmp_path: Path,monkeypatch: pytest.MonkeyPatch,
                              *, phase: str = "own_ready_finals") -> Path:
    """Real known JSON/partials, ordinary fake observation/client boundaries only."""
    import clipperx.visual_director as module
    write=module.atomic_write_text
    def interrupted(path: Path,text: str) -> Any:
        if path.name=="receipt.json" or (phase!="published_review_companion" and path.name=="provenance.json"
                and json.loads(text)["status"]=="review_required"):
            raise OSError("HOST interrupted final publication")
        return write(path,text)
    with monkeypatch.context() as patch:
        patch.setattr(module,"atomic_write_text",interrupted)
        with pytest.raises(module.DirectorPublicationError) as caught:
            run(scene,tmp_path,FakeClient([ready]))
    root=caught.value.run_path
    assert not (root/"receipt.json").exists()
    assert (root/"review-provenance.candidate.json").is_file() and (root/"review-receipt.candidate.json").is_file()
    if phase=="own_ready_finals":
        (root/"uncommitted-plan.json").replace(root/"visual_plan.json")
        (root/"uncommitted-ready-provenance.json").replace(root/"provenance.json")
    return root


def recovery_files(root: Path) -> dict[str,bytes]:
    return {path.relative_to(root).as_posix():path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("defect",["validation","failed_check","call_count","call_bool","context",
    "missing_receipt_version","receipt_version_bool","receipt_kind","companion_version","missing_companion_kind",
    "final_path","missing_final_path","forbidden_plan","receipt_source","noncanonical_companion","noncanonical_receipt"])
def test_review_candidate_recovery_rejects_before_any_mutation(scene: Any,tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,defect: str) -> None:
    import clipperx.visual_director as module
    root=pending_review_candidates(scene,tmp_path,monkeypatch)
    companion_path=root/"review-provenance.candidate.json"
    receipt_path=root/"review-receipt.candidate.json"
    companion=json.loads(companion_path.read_text())
    receipt=json.loads(receipt_path.read_text())
    if defect=="validation":
        companion["validation"]="INVALID CURRENT VALIDATION"
    elif defect=="failed_check":
        companion["failed_check"]="different failure"
    elif defect=="call_count":
        receipt["calls"]=2
    elif defect=="call_bool":
        companion["calls"][0]["call"]=True
    elif defect=="context":
        receipt["host_context"]=companion["host_context"]=None
    elif defect=="missing_receipt_version":
        receipt.pop("version")
    elif defect=="receipt_version_bool":
        receipt["version"]=True
    elif defect=="receipt_kind":
        receipt["kind"]="wrong-receipt-kind"
    elif defect=="companion_version":
        companion["version"]=2
    elif defect=="missing_companion_kind":
        companion.pop("kind")
    elif defect=="final_path":
        receipt["companion_path"]=str(root/"wrong-final.json")
    elif defect=="missing_final_path":
        receipt.pop("companion_path")
    elif defect=="forbidden_plan":
        receipt["plan_path"]=str(root/"visual_plan.json")
    elif defect=="receipt_source":
        receipt["source"]=str(tmp_path/"wrong.identity")
    if defect=="noncanonical_companion":
        companion_path.write_text(json.dumps(companion,ensure_ascii=False,indent=2),encoding="utf-8")
    else:
        module._write(companion_path,companion)
    receipt["companion_sha256"]=module.file_sha256(companion_path)
    if defect=="noncanonical_receipt":
        receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    else:
        module._write(receipt_path,receipt)
    before=recovery_files(root)
    with pytest.raises(module.DirectorPublicationError):
        module.recover_visual_director_publication(scene[0],root,jobs_root=tmp_path/"jobs")
    assert not (root/"receipt.json").exists(),"invalid recovery candidate materialized a terminal marker"
    assert recovery_files(root)==before,"candidate rejection moved or rewrote known partials"


@pytest.mark.parametrize("missing",["review-provenance.candidate.json","review-receipt.candidate.json"])
def test_review_candidate_recovery_requires_both_candidates_before_mutation(scene: Any,tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,missing: str) -> None:
    import clipperx.visual_director as module
    root=pending_review_candidates(scene,tmp_path,monkeypatch)
    (root/missing).unlink()
    before=recovery_files(root)
    with pytest.raises(module.DirectorPublicationError):
        module.recover_visual_director_publication(scene[0],root,jobs_root=tmp_path/"jobs")
    assert not (root/"receipt.json").exists() and recovery_files(root)==before


def test_review_candidate_recovery_preflights_all_preservation_targets(scene: Any,tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as module
    root=pending_review_candidates(scene,tmp_path,monkeypatch)
    (root/"provenance.json").write_bytes(b"HOST conflicting existing target")
    before=recovery_files(root)
    with pytest.raises(module.DirectorPublicationError):
        module.recover_visual_director_publication(scene[0],root,jobs_root=tmp_path/"jobs")
    assert not (root/"receipt.json").exists() and recovery_files(root)==before


@pytest.mark.parametrize("phase",["unpublished","own_ready_finals","published_review_companion"])
def test_review_candidate_recovery_valid_partials_publish_only_review(scene: Any,tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,phase: str) -> None:
    import clipperx.visual_director as module
    root=pending_review_candidates(scene,tmp_path,monkeypatch,phase=phase)
    candidates={name:(root/name).read_bytes() for name in ("review-provenance.candidate.json","review-receipt.candidate.json")}
    monkeypatch.setattr(module,"_check_dependencies",lambda *a:pytest.fail("current dependencies on REVIEW recovery"))
    monkeypatch.setattr(module.observed_evidence,"prepare_visual_evidence_context",lambda *a:pytest.fail("qualification"))
    monkeypatch.setattr(module.observed_evidence,"observe_windows",lambda *a,**k:pytest.fail("decode"))
    path=module.recover_visual_director_publication(scene[0],root,jobs_root=tmp_path/"jobs")
    assert path==root/"receipt.json" and json.loads(path.read_text())["status"]=="review_required"
    assert not (root/"visual_plan.json").exists()
    assert (root/"uncommitted-plan.json").is_file() and (root/"uncommitted-ready-provenance.json").is_file()
    assert all((root/name).read_bytes()==content for name,content in candidates.items())
    assert (root/"provenance.json").read_bytes()==candidates["review-provenance.candidate.json"]


@pytest.mark.parametrize("defect",["source","clip","null_authority","authority_path","context",
    "missing_bundles","bad_bundle","bundle_bool","missing_failed_check","failed_check","validation",
    "calls_bool","call_count","call_budget","call_sequence","call_bool","bad_call","missing_source"])
def test_review_terminal_rejects_resealed_captured_contradictions(scene: Any,tmp_path: Path,defect: str) -> None:
    import clipperx.visual_director as module
    path=run(scene,tmp_path,FakeClient([dict(status="review_required",reasoning="bounded review")]))
    root=path.parent
    receipt=json.loads(path.read_text())
    companion=json.loads((root/"provenance.json").read_text())
    if defect=="source":
        companion["source"]="WRONG SOURCE"
    elif defect=="clip":
        companion["clip_id"]="other"
    elif defect=="null_authority":
        companion["authority"]=None
    elif defect=="authority_path":
        companion["authority"]["path"]=str(root/"other.json")
    elif defect=="context":
        companion["host_context"]=receipt["host_context"]=None
    elif defect=="missing_bundles":
        companion.pop("bundles")
    elif defect=="bad_bundle":
        companion["bundles"]=[{}]
    elif defect=="bundle_bool":
        companion["bundles"][0]["max_images"]=True
    elif defect=="missing_failed_check":
        companion.pop("failed_check")
    elif defect=="failed_check":
        companion["failed_check"]="different reason"
    elif defect=="validation":
        companion["validation"]="NOT LAST VALIDATED FACTS"
    elif defect=="calls_bool":
        receipt["calls"]=True
    elif defect=="call_count":
        receipt["calls"]=2
    elif defect=="call_budget":
        base=companion["calls"][0]
        companion["calls"]=[dict(base,call=i,path=str(root/f"call-{i:02d}"/"images.json")) for i in range(1,8)]
        receipt["calls"]=7
    elif defect=="call_sequence":
        companion["calls"][0]["call"]=2
    elif defect=="call_bool":
        companion["calls"][0]["call"]=True
    elif defect=="bad_call":
        companion["calls"][0].pop("path")
    else:
        receipt.pop("source",None)
    reseal_terminal(root,receipt,companion)
    with pytest.raises(module.DirectorPublicationError):
        module._load_terminal(root)


def test_review_initialization_failure_is_explicit_null_zero_call(scene: Any,tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as module
    monkeypatch.setattr(module.observed_evidence,"prepare_visual_evidence_context",
        lambda *a:(_ for _ in ()).throw(ValueError("HOST initialization unavailable")))
    client=FakeClient([ready])
    path=run(scene,tmp_path,client)
    assert not client.payloads and not (path.parent/"authority.json").exists()
    terminal=module._load_terminal(path.parent)
    assert terminal is not None and terminal[1]["calls"]==0
    assert terminal[2]["authority"] is None and terminal[2]["host_context"] is None
    assert terminal[2]["calls"]==[] and terminal[2]["bundles"]==[]
    before=path.read_bytes()
    assert module.recover_visual_director_publication(scene[0],path.parent,jobs_root=tmp_path/"jobs")==path
    assert path.read_bytes()==before


def test_review_cannot_relabel_lost_captured_authority_as_initialization_failure(scene: Any,tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as module
    def failed_collection(*args: Any,**kwargs: Any) -> Path:
        root=kwargs["output_path"].parent.parent
        (root/"authority.json").unlink()
        raise ValueError("HOST unavailable collection")
    monkeypatch.setattr(module,"collect_visual_evidence",failed_collection)
    with pytest.raises(module.DirectorPublicationError):
        run(scene,tmp_path,FakeClient([ready]))
    roots=list((tmp_path/"jobs").glob("*/planning/visual/*/*"))
    assert len(roots)==1 and not (roots[0]/"receipt.json").exists()


@pytest.mark.parametrize("defect",["frame_type","outcome","reference","catalog","observation"])
def test_review_zero_call_rejects_resealed_malformed_snapshot_metadata(scene: Any,tmp_path: Path,defect: str) -> None:
    import clipperx.visual_director as module
    import hashlib
    # Text parsing fails after authority and valid tiny B images were captured.
    job=JobStore.for_video(scene[0],jobs_root=tmp_path/"jobs")
    job.transcript_path.write_text("{}")
    path=run(scene,tmp_path,FakeClient([ready]))
    receipt=json.loads(path.read_text())
    companion=json.loads((path.parent/"provenance.json").read_text())
    assert receipt["calls"]==0 and companion["authority"] is not None
    record=companion["bundles"][0]
    snapshot=record["captured"]
    if defect=="frame_type":
        snapshot["frames"]=[True]
    elif defect=="outcome":
        snapshot["outcomes"][0]["frame_id"]=None
    elif defect=="reference":
        snapshot["frames"][0]["sample_refs"][0]["receipt"]="missing.json"
    elif defect=="catalog":
        snapshot["catalog"][0]["bytes"]=True
    else:
        snapshot["observations"][0]["observation_id"]=True
    encoded=module.canonical(snapshot).encode()
    record.update(bytes=len(encoded),sha256=hashlib.sha256(encoded).hexdigest())
    reseal_terminal(path.parent,receipt,companion)
    with pytest.raises(module.DirectorPublicationError):
        module._load_terminal(path.parent)


def test_recovery_review_source_drift_uses_captured_existing_namespace(scene: Any,tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as module
    def drift(payload: Any) -> Any:
        scene[0].write_bytes(scene[0].read_bytes()+b"source changed")
        return ready(payload)
    path=run(scene,tmp_path,FakeClient([drift]))
    assert json.loads(path.read_text())["status"]=="review_required"
    jobs=tmp_path/"jobs"
    before={p.name for p in jobs.iterdir()}
    terminal=path.read_bytes()
    monkeypatch.setattr(module,"_check_dependencies",lambda *a:pytest.fail("current validity on drifted REVIEW"))
    monkeypatch.setattr(JobStore,"for_video",lambda *a,**k:pytest.fail("new JobStore initialization during recovery"))
    assert module.recover_visual_director_publication(scene[0],path.parent,jobs_root=jobs)==path
    assert path.read_bytes()==terminal and {p.name for p in jobs.iterdir()}==before


@pytest.mark.parametrize("defect",["source","jobs_root","missing_state","state_fingerprint","missing_stage"])
def test_recovery_review_namespace_conflicts_fail_without_creating_state(scene: Any,tmp_path: Path,defect: str) -> None:
    import clipperx.visual_director as module
    path=run(scene,tmp_path,FakeClient([dict(status="review_required",reasoning="bounded review")]))
    jobs=tmp_path/"jobs"
    state_path=path.parent.parents[3]/"state.json"
    source=scene[0]
    if defect=="source":
        source=tmp_path/"wrong.identity"
        source.write_bytes(b"HOST wrong source identity")
    elif defect=="jobs_root":
        jobs=tmp_path/"unrelated-jobs"
        jobs.mkdir()
    elif defect=="missing_state":
        state_path.unlink()
    elif defect=="state_fingerprint":
        state=json.loads(state_path.read_text())
        state["fingerprint"]="f"*16
        state_path.write_text(json.dumps(state))
    else:
        state=json.loads(state_path.read_text())
        state["stages"]={}
        state_path.write_text(json.dumps(state))
    before={p.name for p in jobs.iterdir()}
    state_before=state_path.read_bytes() if state_path.exists() else None
    with pytest.raises((ValueError,module.DirectorPublicationError)):
        module.recover_visual_director_publication(source,path.parent,jobs_root=jobs)
    assert {p.name for p in jobs.iterdir()}==before
    assert (state_path.read_bytes() if state_path.exists() else None)==state_before


def test_recovery_ready_keeps_current_source_gate_and_never_downgrades(scene: Any,tmp_path: Path) -> None:
    import clipperx.visual_director as module
    path=run(scene,tmp_path,FakeClient([ready]))
    receipt=path.parent/"receipt.json"
    before=receipt.read_bytes()
    jobs=tmp_path/"jobs"
    namespaces={p.name for p in jobs.iterdir()}
    scene[0].write_bytes(scene[0].read_bytes()+b"changed after READY")
    with pytest.raises(module.DirectorPublicationError):
        module.recover_visual_director_publication(scene[0],path.parent,jobs_root=jobs)
    assert receipt.read_bytes()==before and path.is_file()
    assert {p.name for p in jobs.iterdir()}==namespaces


@pytest.mark.parametrize("defect",["prefix","prefix_bool","segment_bool","segment_budget"])
def test_terminal_calls_bind_actual_prefix_and_segment_budget(scene: Any,tmp_path: Path,defect: str) -> None:
    import clipperx.visual_director as module
    path=run(scene,tmp_path,FakeClient([more(1,60,64),ready]))
    root=path.parent
    receipt=json.loads((root/"receipt.json").read_text())
    companion=json.loads((root/"provenance.json").read_text())
    if defect=="prefix":
        companion["calls"][0]["bundle_count"]=2
    elif defect=="prefix_bool":
        companion["calls"][0]["bundle_count"]=True
    elif defect=="segment_bool":
        companion["calls"][0]["segment_indices"]=[True]
    else:
        base=companion["calls"][-1]
        for index in (3,4):
            record=dict(base,call=index,path=str(root/f"call-{index:02d}"/"images.json"))
            module._write(Path(record["path"]),record["presented_assets"])
            record.update(module._file_record(Path(record["path"])))
            companion["calls"].append(record)
        receipt["calls"]=4
    reseal_terminal(root,receipt,companion)
    with pytest.raises(module.DirectorPublicationError):
        module._load_terminal(root)


@pytest.mark.parametrize("target",["source","jpeg","bundle","text","timeline"])
def test_review_drift_validates_captured_facts_without_current_dependencies(scene: Any,tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,target: str) -> None:
    import clipperx.visual_director as module
    pinned=tmp_path/"pinned.json"
    pinned.write_text(scene[2].model_dump_json())
    def drift(payload: Any) -> Any:
        root=next((tmp_path/"jobs").glob("*/planning/visual/*/*"))
        if target=="source":
            file=scene[0]
        elif target=="jpeg":
            file=next(root.glob("evidence/initial/*.jpg"))
        elif target=="bundle":
            file=root/"evidence"/"initial.json"
        elif target=="text":
            file=JobStore.for_video(scene[0],jobs_root=tmp_path/"jobs").transcript_path
        else:
            file=pinned
        file.write_bytes(file.read_bytes()+b" ")
        return ready(payload)
    path=run(scene,tmp_path,FakeClient([drift]),source_timeline_path=pinned)
    assert json.loads(path.read_text())["status"]=="review_required"
    monkeypatch.setattr(module,"_check_dependencies",lambda *a:pytest.fail("current dependencies on REVIEW"))
    terminal=module._load_terminal(path.parent)
    assert terminal is not None and terminal[2]["validation"]=="last_validated_facts"


@pytest.mark.parametrize("defect",["frame_id","aliases","bundle_path","receipt","sample_refs","upstream",
    "image_index","empty","source_budget","call_budget"])
def test_ready_terminal_joins_resealed_presented_rows_to_qualified_bundles(scene: Any,tmp_path: Path,defect: str) -> None:
    import clipperx.visual_director as module
    path=run(scene,tmp_path,FakeClient([ready]))
    assert path.name=="visual_plan.json",path.read_text()
    root=path.parent
    receipt=json.loads((root/"receipt.json").read_text())
    companion=json.loads((root/"provenance.json").read_text())
    call=companion["calls"][-1]
    row=call["presented_assets"][0]
    if defect=="frame_id":
        row["frame_id"]="f"*64
    elif defect=="aliases":
        row["aliases"]=["unqualified-alias"]
    elif defect=="bundle_path":
        row["sample_refs"][0].pop("bundle_path")
    elif defect=="receipt":
        row["sample_refs"][0]["receipt_path"]=str(root/"unrelated.json")
    elif defect=="sample_refs":
        row["sample_refs"][0]["requested"]="60/1"
    elif defect=="upstream":
        row["upstream_context_sha256"]="f"*64
    elif defect=="image_index":
        row["image_index"]=True
    elif defect=="empty":
        call["presented_assets"]=[]
    elif defect=="source_budget":
        call["presented_assets"]=[dict(row,image_index=i) for i in range(6)]
    else:
        companion["calls"]=[dict(call,call=i,path=str(root/f"call-{i:02d}"/"images.json")) for i in range(1,8)]
        receipt["calls"]=7
    for captured in companion["calls"]:
        manifest_path=Path(captured["path"])
        module._write(manifest_path,captured["presented_assets"])
        captured.update(module._file_record(manifest_path))
    reseal_terminal(root,receipt,companion)
    with pytest.raises(module.DirectorPublicationError):
        module._load_terminal(root)


def test_ready_final_stateless_call_rejects_omitted_image_citation(scene: Any,tmp_path: Path) -> None:
    from fractions import Fraction
    witnesses: list[str]=[]
    def omitted(payload: Any) -> Any:
        shown={alias for row in payload["frame_manifest"] if row["kind"]=="source" for alias in row["aliases"]}
        value=ready(payload)
        candidates=[(shot,frame) for shot in value["plan"]["shots"]
            for bundle in payload["clip"]["host_context"]["evidence_bundles"] for frame in bundle["frames"]
            if frame["evidence_id"] not in shown and any(Fraction(str(shot["source"]["start"]))<=Fraction(ref["requested"])
                <Fraction(str(shot["source"]["end"])) for ref in frame["sample_refs"])]
        assert candidates,"fixture must expose a local decoded but omitted image"
        shot,frame=candidates[0]
        witnesses.append(frame["evidence_id"])
        shot["evidence_ids"]=[frame["evidence_id"]]
        return value
    client=FakeClient([more(1,59,60.5),omitted])
    path=run(scene,tmp_path,client)
    assert len(client.payloads)==2 and witnesses
    assert json.loads(path.read_text()).get("status")=="review_required"
    assert not (path.parent/"visual_plan.json").exists()


@pytest.mark.parametrize("defect",["card","path","style_budget"])
def test_ready_terminal_joins_resealed_style_rows_to_selected_authority(scene: Any,tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,defect: str) -> None:
    from PIL import Image
    import clipperx.visual_director as module
    from clipperx.tests.test_style_retrieval import card
    library=scene[3]
    library.root.mkdir(parents=True)
    image=library.root/"chosen.jpg"
    Image.new("RGB",(16,16)).save(image)
    selected=card("chosen",keyframes=[dict(timestamp=0,path="chosen.jpg")])
    monkeypatch.setattr(module,"_select_styles",lambda *a:([selected],[]))
    path=run(scene,tmp_path,FakeClient([ready]))
    assert path.name=="visual_plan.json",path.read_text()
    root=path.parent
    receipt=json.loads((root/"receipt.json").read_text())
    companion=json.loads((root/"provenance.json").read_text())
    call=companion["calls"][-1]
    row=next(r for r in call["presented_assets"] if r["kind"]=="style")
    if defect=="card":
        row["card_id"]="not-selected"
    elif defect=="path":
        other=library.root/"same-bytes-unselected.jpg"
        other.write_bytes(image.read_bytes())
        row["path"]=str(other)
    else:
        start=len([r for r in call["presented_assets"] if r["kind"]=="source"])
        call["presented_assets"]=[r for r in call["presented_assets"] if r["kind"]=="source"]+[
            dict(row,image_index=start+i) for i in range(4)]
    module._write(Path(call["path"]),call["presented_assets"])
    call.update(module._file_record(Path(call["path"])))
    reseal_terminal(root,receipt,companion)
    with pytest.raises(module.DirectorPublicationError):
        module._load_terminal(root)


def test_presented_final_cross_bundle_alias_preserves_earlier_prefix(scene: Any,tmp_path: Path) -> None:
    aliases: list[str]=[]
    def final(payload: Any) -> Any:
        rows=[r for r in payload["frame_manifest"] if r["kind"]=="source"]
        row=next(r for r in rows if len(r["aliases"])==2 and any(ref["requested"]=="60/1" for ref in r["sample_refs"]))
        alias=next(a for a in row["aliases"] if a.startswith("followup"))
        aliases.append(alias)
        value=ready(payload)
        value["plan"]["shots"][0]["evidence_ids"]=[alias]
        return value
    client=FakeClient([more(1,60,64),final])
    path=run(scene,tmp_path,client)
    assert path.name=="visual_plan.json" and aliases
    first=[r for r in client.payloads[0]["frame_manifest"] if r["kind"]=="source"]
    assert all(len(r["aliases"])==1 for r in first)
    import clipperx.visual_director as module
    assert module._load_terminal(path.parent) is not None


def test_ready_terminal_rejects_resealed_plan_citing_final_omitted_image(scene: Any,tmp_path: Path) -> None:
    from fractions import Fraction
    import clipperx.visual_director as module
    path=run(scene,tmp_path,FakeClient([more(1,59,60.5),ready]))
    assert path.name=="visual_plan.json",path.read_text()
    root=path.parent
    receipt=json.loads((root/"receipt.json").read_text())
    companion=json.loads((root/"provenance.json").read_text())
    plan=json.loads(path.read_text())
    shown={alias for row in companion["calls"][-1]["presented_assets"] if row["kind"]=="source" for alias in row["aliases"]}
    omitted=[(shot,frame) for shot in plan["shots"] for record in companion["bundles"]
        for frame in json.loads(Path(record["path"]).read_text())["frames"]
        if frame["evidence_id"] not in shown and any(Fraction(str(shot["source"]["start"]))<=Fraction(ref["requested"])
            <Fraction(str(shot["source"]["end"])) for ref in frame["sample_refs"])]
    assert omitted
    omitted[0][0]["evidence_ids"]=[omitted[0][1]["evidence_id"]]
    module._write(path,plan)
    companion["plan"]={**module._file_record(path),"fingerprint":module._hash(plan)}
    receipt["plan_sha256"]=companion["plan"]["sha256"]
    reseal_terminal(root,receipt,companion)
    with pytest.raises(module.DirectorPublicationError):
        module._load_terminal(root)


def test_director_preserves_editorial_ranges_when_splitting_visual_shots(scene: Any,
                                                                      tmp_path: Path) -> None:
    client = FakeClient([ready])
    path = run(scene, tmp_path, client)
    plan = json.loads(path.read_text())
    assert plan["contract_version"] == 2
    assert [shot["source"] for shot in plan["shots"]] == [
        {"start": 60.0, "end": 62.0}, {"start": 62.0, "end": 64.0}]
    assert len(client.images[0]) <= 5
    assert all(path.is_file() for path in client.images[0])
    assert json.loads((path.parent / "receipt.json").read_text())["status"] == "ready"


def test_director_pins_exact_path_and_propagates_legacy_host_context(scene: Any,tmp_path: Path) -> None:
    _,_,timeline,_=scene
    pinned=tmp_path/"exact-timeline.json"
    pinned.write_text(timeline.model_dump_json(),encoding="utf-8")
    client=FakeClient([ready])
    path=run(scene,tmp_path,client,source_timeline_path=pinned)
    assert path.name=="visual_plan.json",path.read_text()
    host=client.payloads[0]["clip"]["host_context"]["provenance"]
    assert host["upstream"]["path"]==str(pinned.resolve())
    assert host["upstream"]["context"]["status"]=="unverified_legacy"
    assert host["text"]["status"]=="unverified_legacy"
    companion=json.loads((path.parent/"provenance.json").read_text())
    assert companion["kind"]=="visual_plan_provenance" and companion["status"]=="ready"
    assert companion["host_context"]==host
    assert companion["calls"][0]["presented_assets"]


@pytest.mark.parametrize("target",["jpeg","receipt","bundle","text","timeline","source"])
def test_director_checks_captured_dependencies_after_response(scene: Any,tmp_path: Path,target: str) -> None:
    pinned=tmp_path/"pinned.json"
    pinned.write_text(scene[2].model_dump_json())
    def drift(payload: Any) -> Any:
        run_root=next((tmp_path/"jobs").glob("*/planning/visual/*/*"))
        if target=="jpeg":
            file=next(run_root.glob("evidence/initial/*.jpg"))
        elif target=="receipt":
            file=next(run_root.glob("evidence/initial/observation*.json"))
        elif target=="bundle":
            file=run_root/"evidence/initial.json"
        elif target=="text":
            file=JobStore.for_video(scene[0],jobs_root=tmp_path/"jobs").transcript_path
        elif target=="timeline":
            file=pinned
        else:
            file=scene[0]
        file.write_bytes(file.read_bytes()+b" ")
        return ready(payload)
    client=FakeClient([drift])
    path=run(scene,tmp_path,client,source_timeline_path=pinned)
    assert path.name=="receipt.json" and json.loads(path.read_text())["status"]=="review_required"
    assert not (path.parent/"visual_plan.json").exists()
    assert len(client.payloads)==1


def test_director_marked_corrupt_upstream_calls_zero(scene: Any,tmp_path: Path) -> None:
    pinned=tmp_path/"visual_observed"/"generation"/"timeline.json"
    pinned.parent.mkdir(parents=True)
    pinned.write_text(scene[2].model_dump_json())
    client=FakeClient([ready])
    path=run(scene,tmp_path,client,source_timeline_path=pinned)
    assert json.loads(path.read_text())["status"]=="review_required" and not client.payloads


@pytest.mark.parametrize("target",["visual_plan.json","provenance.json","receipt.json"])
@pytest.mark.parametrize("after",[False,True])
def test_director_publication_known_target_fault_table(scene: Any,tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,target: str,after: bool) -> None:
    import clipperx.visual_director as module
    original=module.atomic_write_text
    fired: list[bool]=[]
    def write(path: Path,text: str) -> None:
        if path.name==target and not fired:
            fired.append(True)
            if after:
                original(path,text)
            raise OSError("HOST injected publication fault")
        original(path,text)
    monkeypatch.setattr(module,"atomic_write_text",write)
    path=run(scene,tmp_path,FakeClient([ready]))
    assert fired
    receipt=json.loads((path.parent/"receipt.json").read_text())
    assert receipt["status"]==("ready" if after else "review_required")
    if after:
        assert path.name=="visual_plan.json"
    else:
        assert not (path.parent/"visual_plan.json").exists()
        if target!="visual_plan.json":
            assert (path.parent/"uncommitted-plan.json").is_file()


def test_director_committed_ready_survives_bookkeeping_failure_and_recovery(scene: Any,tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as module
    assert hasattr(module,"DirectorPublicationError"),"typed publication error absent"
    original=JobStore.mark_complete
    monkeypatch.setattr(JobStore,"mark_complete",lambda *a,**k:(_ for _ in ()).throw(OSError("HOST")))
    with pytest.raises(module.DirectorPublicationError) as caught:
        run(scene,tmp_path,FakeClient([ready]))
    error=caught.value
    assert error.phase=="COMMITTED_BOOKKEEPING_FAILED"
    assert error.committed_path is not None
    assert error.committed_path.name=="visual_plan.json"
    terminal=error.run_path/"receipt.json"
    before=terminal.read_bytes()
    monkeypatch.setattr(JobStore,"mark_complete",original)
    monkeypatch.setattr(module.observed_evidence,"prepare_visual_evidence_context",lambda *a:pytest.fail("qualify"))
    monkeypatch.setattr(module.observed_evidence,"observe_windows",lambda *a,**k:pytest.fail("decode"))
    path=module.recover_visual_director_publication(scene[0],error.run_path,jobs_root=tmp_path/"jobs")
    assert path==error.committed_path and terminal.read_bytes()==before


def test_director_ambiguous_target_never_invents_review(scene: Any,tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as module
    assert hasattr(module,"DirectorPublicationError"),"typed publication error absent"
    original=module.atomic_write_text
    def write(path: Path,text: str) -> None:
        if path.name=="visual_plan.json":
            path.write_bytes(b"conflicting bytes")
            raise OSError("HOST")
        original(path,text)
    monkeypatch.setattr(module,"atomic_write_text",write)
    with pytest.raises(module.DirectorPublicationError) as caught:
        run(scene,tmp_path,FakeClient([ready]))
    assert caught.value.phase=="AMBIGUOUS_PUBLICATION"
    assert not (caught.value.run_path/"receipt.json").exists()
    assert (caught.value.run_path/"visual_plan.json").read_bytes()==b"conflicting bytes"


@pytest.mark.parametrize("after",[False,True])
def test_director_preservation_move_postconditions(scene: Any,tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch,after: bool) -> None:
    import clipperx.visual_director as module
    write=module.atomic_write_text
    replace=Path.replace
    def fail_receipt(path: Path,text: str) -> Any:
        if path.name=="receipt.json" and json.loads(text)["status"]=="ready":
            raise OSError("HOST")
        return write(path,text)
    def move(path: Path,target: Path) -> Any:
        if path.name=="visual_plan.json":
            if after:
                replace(path,target)
            raise OSError("HOST move")
        return replace(path,target)
    monkeypatch.setattr(module,"atomic_write_text",fail_receipt)
    monkeypatch.setattr(Path,"replace",move)
    if after:
        path=run(scene,tmp_path,FakeClient([ready]))
        assert json.loads(path.read_text())["status"]=="review_required"
        assert (path.parent/"uncommitted-plan.json").is_file()
    else:
        with pytest.raises(module.DirectorPublicationError) as caught:
            run(scene,tmp_path,FakeClient([ready]))
        assert caught.value.phase=="PRECOMMIT_PRESERVATION_FAILED"
        assert not (caught.value.run_path/"receipt.json").exists()
        assert (caught.value.run_path/"visual_plan.json").is_file()


@pytest.mark.parametrize("mode",["no_materialization","mismatch","unreadable","late_plan_drift"])
def test_director_publication_ambiguous_postconditions(scene: Any,tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,mode: str) -> None:
    import clipperx.visual_director as module
    write=module.atomic_write_text
    original_open=Path.open
    blocked: list[Path]=[]
    def open_file(path: Path,*args: Any,**kwargs: Any) -> Any:
        if path in blocked:
            raise PermissionError("HOST")
        return original_open(path,*args,**kwargs)
    def writer(path: Path,text: str) -> Any:
        if path.name=="visual_plan.json":
            if mode=="no_materialization":
                return path
            if mode=="mismatch":
                write(path,"{}")
                return path
            result=write(path,text)
            if mode=="unreadable":
                blocked.append(path)
            return result
        result=write(path,text)
        if mode=="late_plan_drift" and path.name=="provenance.json" and json.loads(text)["status"]=="ready":
            (path.parent/"visual_plan.json").write_bytes(b"changed after plan ack")
        return result
    monkeypatch.setattr(Path,"open",open_file)
    monkeypatch.setattr(module,"atomic_write_text",writer)
    with pytest.raises(module.DirectorPublicationError) as caught:
        run(scene,tmp_path,FakeClient([ready]))
    assert caught.value.phase=="AMBIGUOUS_PUBLICATION"
    assert not (caught.value.run_path/"receipt.json").exists()


def test_explicit_precommit_recovery_finishes_review_without_promotion(scene: Any,tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as module
    write=module.atomic_write_text
    def fail(path: Path,text: str) -> Any:
        if path.name=="receipt.json":
            raise OSError("HOST no terminal")
        return write(path,text)
    monkeypatch.setattr(module,"atomic_write_text",fail)
    with pytest.raises(module.DirectorPublicationError) as caught:
        run(scene,tmp_path,FakeClient([ready]))
    root=caught.value.run_path
    assert (root/"review-provenance.candidate.json").exists()
    monkeypatch.setattr(module,"atomic_write_text",write)
    monkeypatch.setattr(module.observed_evidence,"observe_windows",lambda *a,**k:pytest.fail("decode"))
    monkeypatch.setattr(module.observed_evidence,"prepare_visual_evidence_context",lambda *a:pytest.fail("qualify"))
    result=module.recover_visual_director_publication(scene[0],root,jobs_root=tmp_path/"jobs")
    assert json.loads(result.read_text())["status"]=="review_required"
    assert not (root/"visual_plan.json").exists()
    assert (root/"uncommitted-plan.json").exists()


def test_recovery_rejects_damaged_known_candidate(scene: Any,tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as module
    write=module.atomic_write_text
    def fail(path: Path,text: str) -> Any:
        if path.name=="receipt.json":
            raise OSError("HOST")
        return write(path,text)
    monkeypatch.setattr(module,"atomic_write_text",fail)
    with pytest.raises(module.DirectorPublicationError) as caught:
        run(scene,tmp_path,FakeClient([ready]))
    root=caught.value.run_path
    (root/"visual_plan.candidate.json").write_text("{}")
    monkeypatch.setattr(module,"atomic_write_text",write)
    with pytest.raises(module.DirectorPublicationError):
        module.recover_visual_director_publication(scene[0],root,jobs_root=tmp_path/"jobs")
    assert not (root/"receipt.json").exists()


def test_director_preserves_old_stage_flags(scene: Any,tmp_path: Path) -> None:
    job=JobStore.for_video(scene[0],jobs_root=tmp_path/"jobs")
    old="visual_director:clip-0001"
    job.mark_review_required(old,reason="old flag",receipt=job.transcript_path)
    before=job.load()["stages"][old]
    path=run(scene,tmp_path,FakeClient([ready]))
    stages=job.load()["stages"]
    assert stages[old]==before and stages[f"{old}:{path.parent.name}"]["status"]=="complete"


def test_director_ignores_unrelated_latest_and_prepares_cpu_once(scene: Any,tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.evidence as e
    pinned=tmp_path/"exact.json"
    pinned.write_text(scene[2].model_dump_json())
    qualify: list[bool]=[]
    monkeypatch.setattr(e,"qualify_render_runtime",lambda *a:qualify.append(True))
    def latest(payload: Any) -> Any:
        job=JobStore.for_video(scene[0],jobs_root=tmp_path/"jobs")
        unrelated=job.source_visual_observed_latest_path
        unrelated.parent.mkdir(parents=True,exist_ok=True)
        unrelated.write_text('{"unrelated":"changed"}')
        return more(1)
    client=FakeClient([latest,ready])
    path=run(scene,tmp_path,client,source_timeline_path=pinned)
    assert path.name=="visual_plan.json" and qualify==[True] and len(client.payloads)==2


def test_director_consumes_only_the_captured_transcript_path(scene: Any,tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as module
    collect=module.collect_visual_evidence
    def collecting(*args: Any,**kwargs: Any) -> Path:
        result=collect(*args,**kwargs)
        job=JobStore.for_video(scene[0],jobs_root=tmp_path/"jobs")
        job.speaker_transcript_path.write_text(json.dumps([dict(start=60,end=61,text="UNTRACKED_TEXT")]))
        return result
    monkeypatch.setattr(module,"collect_visual_evidence",collecting)
    client=FakeClient([ready])
    run(scene,tmp_path,client)
    assert "local payoff" in json.dumps(client.payloads)
    assert "UNTRACKED_TEXT" not in json.dumps(client.payloads)


def test_execution_and_recovery_use_the_same_run_lock(scene: Any,tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager
    import clipperx.visual_director as module
    original=JobStore.stage_lock
    events: list[str]=[]
    active: list[str]=[]
    @contextmanager
    def locking(job: Any,stage: str) -> Any:
        events.append(stage)
        with original(job,stage):
            active.append(stage)
            try:
                yield
            finally:
                active.pop()
    monkeypatch.setattr(JobStore,"stage_lock",locking)
    def response(payload: Any) -> Any:
        assert len(active)==1 and active[0].startswith("visual_director:")
        return ready(payload)
    path=run(scene,tmp_path,FakeClient([response]))
    recovered=module.recover_visual_director_publication(scene[0],path.parent,jobs_root=tmp_path/"jobs")
    assert recovered==path and len(events)==2 and events[0]==events[1] and not active


def test_run_lock_retains_entire_uuid_for_long_clip_ids(scene: Any,tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager
    from clipperx.local_paths import safe_slug
    original=JobStore.stage_lock
    names: list[str]=[]
    @contextmanager
    def locking(job: Any,stage: str) -> Any:
        names.append(safe_slug(stage))
        with original(job,stage):
            yield
    monkeypatch.setattr(JobStore,"stage_lock",locking)
    video,clip,timeline,library=scene
    clip=clip.model_copy(update={"clip_id":"long-clip-"*30})
    paths=[run((video,clip,timeline,library),tmp_path,FakeClient([ready])) for _ in range(2)]
    assert len(set(names))==2 and all(path.parent.name in label for path,label in zip(paths,names))


def test_manifest_rejects_conflicting_same_pts_across_bundles(tmp_path: Path) -> None:
    from PIL import Image
    from clipperx.visual_director import _manifest
    image=tmp_path/"view.jpg"
    Image.new("RGB",(16,16)).save(image)
    first=dict(manifest_frame(1,image),raw_bgr_sha256="a"*64,observed_source_start="1/1",observed_source_end="2/1")
    conflicting=dict(first,frame_id="different-pixels-same-pts",raw_bgr_sha256="b"*64)
    with pytest.raises(ValueError,match="Conflicting"):
        _manifest([dict(frames=[first]),dict(frames=[conflicting])],[],StyleLibrary(tmp_path/"styles"),VisualDirectorLimits())


@pytest.mark.parametrize("conflict",[None,"pixels","coverage"])
def test_manifest_equivalent_raw_representations_are_one_temporal_instant(tmp_path: Path,conflict: str | None) -> None:
    from PIL import Image
    from clipperx.visual_director import _manifest
    image=tmp_path/"view.jpg"
    Image.new("RGB",(16,16)).save(image)
    first=dict(manifest_frame(0,image),frame_id="clock-a",raw_pts=1,time_base="1/10",raw_bgr_sha256="a"*64,
        at=.15,observed_source_start="1/10",observed_source_end="1/5",
        sample_refs=[dict(requested="3/20",window_indices=[0],receipt="a.json",frame_n=0)])
    equivalent=dict(first,frame_id="clock-b",evidence_id="alias-b",raw_pts=100,time_base="1/1000")
    if conflict=="pixels":
        equivalent["raw_bgr_sha256"]="b"*64
    if conflict=="coverage":
        equivalent["observed_source_end"]="3/10"
    bundles: list[dict[str,Any]]=[dict(frames=[first]),dict(frames=[first,equivalent],request=dict(
        source=dict(start=0,end=1),reason="camera_boundary"))]
    with pytest.raises(ValueError,match="Conflicting" if conflict else "distinct temporal"):
        _manifest(bundles,[],StyleLibrary(tmp_path/"styles"),VisualDirectorLimits())
    if conflict is None:
        # Equivalent representations alias one actual frame even without a request.
        bundles[-1].pop("request")
        manifest,images=_manifest(bundles,[],StyleLibrary(tmp_path/"styles"),VisualDirectorLimits())
        assert len(manifest)==len(images)==1 and manifest[0]["aliases"]==["e-0","alias-b"]


def test_resealed_real_receipt_equivalent_clock_cannot_create_second_view(monkeypatch: pytest.MonkeyPatch,
                                                                       tmp_path: Path) -> None:
    import clipperx.evidence as e
    from clipperx.tests.test_evidence import observed_case
    from clipperx.render_clock import SourceDecodeReceipt,canonical,digest
    from clipperx.planner_contracts import TimeRange
    from clipperx.visual_director import _manifest
    video,timeline,_,context,_,_=observed_case(monkeypatch,tmp_path)
    windows=[TimeRange(start=.07,end=.1)]
    path=e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"one.json",max_images=3)
    bundle=json.loads(path.read_text())
    assert len(bundle["frames"])==1
    receipt=json.loads((path.parent/path.stem/bundle["observations"][0]["path"]).read_text())
    for row in receipt["payload"]["video"]:
        assert row["pts"]%1000==0
        row.update(pts=row["pts"]//1000,time_base="1/1000")
    receipt["sha256"]=digest(canonical({k:v for k,v in receipt.items() if k!="sha256"}))
    validated=SourceDecodeReceipt.from_json(canonical(receipt),origin=context.origin,policy=e.DECODE_POLICY)
    raw=json.loads(validated.payload_json)["video"][0]
    frame_id,_=e._frame_record(raw,context.origin)
    original=bundle["frames"][0]
    equivalent=dict(original,frame_id=frame_id,evidence_id="alternate-clock-alias",raw_pts=raw["pts"],time_base=raw["time_base"])
    assert original["frame_id"]!=frame_id
    combined=dict(bundle,frames=[original,equivalent],request=dict(source=dict(start=.07,end=.1),reason="camera_boundary"))
    with pytest.raises(ValueError,match="distinct temporal"):
        _manifest([combined],[],StyleLibrary(tmp_path/"styles"),VisualDirectorLimits())


def test_director_json_quota_is_checked_before_write(monkeypatch: pytest.MonkeyPatch,tmp_path: Path) -> None:
    import clipperx.visual_director as module
    writes: list[Path]=[]
    monkeypatch.setattr(module,"MAX_ARTIFACT_BYTES",80)
    monkeypatch.setattr(module,"atomic_write_text",lambda path,text:writes.append(path))
    with pytest.raises(ValueError,match="quota"):
        module._write(tmp_path/"oversize.json",dict(value="x"*100))
    assert not writes


def test_manifest_unions_real_bundle_aliases_and_receipt_namespaces(monkeypatch: pytest.MonkeyPatch,tmp_path: Path) -> None:
    import clipperx.evidence as e
    from clipperx.tests.test_evidence import observed_case
    from clipperx.visual_director import _manifest
    video,timeline,windows,context,_,_=observed_case(monkeypatch,tmp_path)
    bundles=[]
    for name in ("initial","followup"):
        path=e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                      output_path=tmp_path/f"{name}.json",max_images=3)
        bundles.append(dict(json.loads(path.read_text()),bundle_path=str(path)))
    manifest,images=_manifest(bundles,[],StyleLibrary(tmp_path/"styles"),VisualDirectorLimits())
    assert len(manifest)==len(images)==2
    assert all(len(frame["aliases"])==2 for frame in manifest)
    assert all(len({ref["bundle_path"] for ref in frame["sample_refs"]})==2 for frame in manifest)
    assert all(Path(ref["receipt_path"]).is_file() for frame in manifest for ref in frame["sample_refs"])


def test_director_entry_image_corruption_calls_zero(scene: Any,tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as module
    collect=module.collect_visual_evidence
    def corrupt(*args: Any,**kwargs: Any) -> Path:
        path=collect(*args,**kwargs)
        bundle=json.loads(path.read_text())
        Path(bundle["frames"][0]["path"]).write_bytes(b"corrupt before call")
        return path
    monkeypatch.setattr(module,"collect_visual_evidence",corrupt)
    client=FakeClient([ready])
    path=run(scene,tmp_path,client)
    assert json.loads(path.read_text())["status"]=="review_required" and not client.payloads


def test_director_no_usable_observed_initial_image_calls_zero(scene: Any,tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    from fractions import Fraction
    from clipperx.tests.test_evidence import install_observer
    install_observer(monkeypatch,scene[0],prefix=Fraction(65))
    client=FakeClient([ready])
    path=run(scene,tmp_path,client)
    assert json.loads(path.read_text())["status"]=="review_required" and not client.payloads
    bundle=json.loads((path.parent/"evidence"/"initial.json").read_text())
    assert not bundle["frames"] and all(o["status"]!="covered" for o in bundle["outcomes"])


def test_director_unicode_pinned_fingerprint_and_object_only_status(scene: Any,tmp_path: Path) -> None:
    import clipperx.visual_director as module
    video,clip,timeline,library=scene
    timeline=timeline.model_copy(update={"analyzer_version":"sintético-acción"})
    pinned=tmp_path/"índice.json"
    pinned.write_text(timeline.model_dump_json(),encoding="utf-8")
    client=FakeClient([ready])
    path=run((video,clip,timeline,library),tmp_path,client,source_timeline_path=pinned)
    assert path.name=="visual_plan.json"
    host=client.payloads[0]["clip"]["host_context"]
    assert host["fingerprints"]["source_timeline_fingerprint"]==module.visual_timeline_fingerprint(timeline)
    legacy=FakeClient([ready])
    run((video,clip,timeline,library),tmp_path,legacy)
    assert legacy.payloads[0]["clip"]["host_context"]["provenance"]["upstream"]["context"]["reason"]=="object_only_no_path"


@pytest.mark.parametrize("drift",[False,True])
def test_director_consumes_detached_observed_A_context_api(scene: Any,tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,drift: bool) -> None:
    """A is independently tested: fake that public API, retain the real B gate."""
    import clipperx.visual_director as module
    from clipperx.visual_provenance import VisualProvenanceContext
    pinned=tmp_path/"pinned.json"
    pinned.write_text(scene[2].model_dump_json())
    changed: list[bool]=[]
    calls=[]
    def upstream(path: Path,*,source_video: Path,expected_timeline_fingerprint: str,current_origin: Any) -> Any:
        assert path==pinned and source_video==scene[0]
        calls.append(True)
        record=dict(status="observed_samples",source_fingerprint=scene[2].source_fingerprint,
            timeline_fingerprint=expected_timeline_fingerprint,generation_id="fixed-generation",
            scope=dict(kind="samples",start="0/1",end="100/1"),origin_id=current_origin.identity,
            complete_sha256=("b" if changed else "a")*64,clock_receipt_sha256="c"*64,
            timeline_sha256=module.file_sha256(pinned))
        return VisualProvenanceContext("observed_samples",json.dumps(record))
    monkeypatch.setattr(module,"load_visual_provenance",upstream)
    def response(payload: Any) -> Any:
        assert payload["clip"]["host_context"]["provenance"]["upstream"]["context"]["status"]=="observed_samples"
        if drift:
            changed.append(True)
        return ready(payload)
    path=run(scene,tmp_path,FakeClient([response]),source_timeline_path=pinned)
    assert path.name==("receipt.json" if drift else "visual_plan.json") and len(calls)>=4


def test_director_rechecks_captured_style_image(scene: Any,tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image
    import clipperx.visual_director as module
    from clipperx.tests.test_style_retrieval import card
    library=scene[3]
    library.root.mkdir(parents=True)
    image=library.root/"view.jpg"
    Image.new("RGB",(16,16)).save(image)
    chosen=card("chosen",keyframes=[dict(timestamp=0,path="view.jpg")])
    monkeypatch.setattr(module,"_select_styles",lambda *a:([chosen],[]))
    def drift(payload: Any) -> Any:
        image.write_bytes(b"changed style")
        return ready(payload)
    client=FakeClient([drift])
    path=run(scene,tmp_path,client)
    assert json.loads(path.read_text())["status"]=="review_required" and len(client.payloads)==1


@pytest.mark.parametrize("target",["jpeg","receipt","text","timeline"])
def test_director_rechecks_dependencies_immediately_before_publication(scene: Any,tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,target: str) -> None:
    import clipperx.visual_director as module
    pinned=tmp_path/"pinned.json"
    pinned.write_text(scene[2].model_dump_json())
    validate=module._validate_final
    def mutate(*args: Any,**kwargs: Any) -> Any:
        value=validate(*args,**kwargs)
        root=next((tmp_path/"jobs").glob("*/planning/visual/*/*"))
        if target=="jpeg":
            file=next(root.glob("evidence/initial/*.jpg"))
        elif target=="receipt":
            file=next(root.glob("evidence/initial/observation*.json"))
        elif target=="text":
            file=JobStore.for_video(scene[0],jobs_root=tmp_path/"jobs").transcript_path
        else:
            file=pinned
        file.write_bytes(file.read_bytes()+b" ")
        return value
    monkeypatch.setattr(module,"_validate_final",mutate)
    path=run(scene,tmp_path,FakeClient([ready]),source_timeline_path=pinned)
    assert json.loads(path.read_text())["status"]=="review_required"
    assert not (path.parent/"visual_plan.json").exists()


@pytest.mark.parametrize("inside",[True,False])
def test_final_image_locality_uses_covered_ref_not_representative(scene: Any,inside: bool) -> None:
    from clipperx.visual_contracts import VisualDirectorResponse
    from clipperx.visual_director import _fingerprints,_required_shots,_validate_final
    _,clip,timeline,library=scene
    fingerprints=_fingerprints(clip,timeline,library.fingerprint)
    payload={"clip":{"clip_id":clip.clip_id,"host_context":{
        "fingerprints":fingerprints,"required_shots":_required_shots(clip,timeline)}}}
    data=ready(payload)
    data["plan"]["shots"][0]["evidence_ids"]=["alias"]
    response=VisualDirectorResponse.model_validate(data)
    bundles=[dict(frames=[dict(evidence_id="alias",at=59 if inside else 61,
        sample_refs=[dict(requested="61/1" if inside else "62/1")])])]
    if inside:
        _validate_final(response,clip,timeline,fingerprints,[],bundles,192,108,
            presented_assets=[dict(bundles[0]["frames"][0],kind="source",aliases=["alias"])])
    else:
        with pytest.raises(ValueError,match="nonlocal"):
            _validate_final(response,clip,timeline,fingerprints,[],bundles,192,108,
                presented_assets=[dict(bundles[0]["frames"][0],kind="source",aliases=["alias"])])


@pytest.mark.parametrize("cite_previous_shot", [False, True])
def test_director_payload_exposes_exact_transition_image_eligibility(
    scene: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cite_previous_shot: bool,
) -> None:
    from fractions import Fraction

    from clipperx.planner_contracts import TimeRange
    from clipperx.tests.test_evidence import install_observer
    from clipperx.visual_director import _presented_local_ids

    video, clip, timeline, library = scene
    install_observer(monkeypatch, video, step=Fraction(1, 100))
    clip_data = clip.model_dump()
    clip_data["segments"][0]["source"] = dict(start=24, end=38)
    clip = ClipPlan.model_validate(clip_data)
    timeline_data = timeline.model_dump()
    timeline_data["spans"][0]["source"] = dict(start=0, end=30.75)
    timeline_data["spans"][1]["source"] = dict(start=30.75, end=100)
    timeline = SourceVisualTimeline.model_validate(timeline_data)
    boundary_ids: list[str] = []

    def cite_boundary(payload: dict[str, Any]) -> dict[str, Any]:
        boundary = next(row for row in payload["frame_manifest"]
                        if row["kind"] == "source"
                        and {ref["requested"] for ref in row["sample_refs"]} == {"123/4"})
        boundary_ids.append(boundary["evidence_id"])
        response = ready(payload)
        for i, shot in enumerate(response["plan"]["shots"]):
            shot["evidence_ids"] = [f"e-{i}"]
        response["plan"]["shots"][0 if cite_previous_shot else 1]["evidence_ids"] = boundary_ids[:]
        return response

    client = FakeClient([cite_boundary])
    path = run((video, clip, timeline, library), tmp_path, client)
    assert len(client.payloads) == 1 and len(boundary_ids) == 1
    payload = client.payloads[0]
    host = payload["clip"]["host_context"]
    required = [dict(start=24.0, end=30.75), dict(start=30.75, end=38.0)]
    assert host["required_shots"] == required
    table = host["local_source_image_ids_by_shot"]
    assert table == [dict(source=part, image_evidence_ids=sorted(
        _presented_local_ids(payload["frame_manifest"], TimeRange.model_validate(part))))
        for part in required]
    assert boundary_ids[0] not in table[0]["image_evidence_ids"]
    assert boundary_ids[0] in table[1]["image_evidence_ids"]
    assert len(client.images[0]) == len(payload["frame_manifest"]) <= 8
    assert sum(row["kind"] == "source" for row in payload["frame_manifest"]) <= 5
    if cite_previous_shot:
        receipt = json.loads(path.read_text())
        assert receipt["status"] == "review_required" and "nonlocal" in receipt["reason"]
        assert not (path.parent / "visual_plan.json").exists()
        saved = json.loads((path.parent / "call-01" / "response.json").read_text())
        assert saved["plan"]["shots"][0]["evidence_ids"] == boundary_ids
    else:
        assert path.name == "visual_plan.json"
        assert json.loads(path.read_text())["shots"][1]["evidence_ids"] == boundary_ids


def test_presented_local_ids_include_captured_requests_and_all_aliases_only() -> None:
    from clipperx.planner_contracts import TimeRange
    from clipperx.visual_director import _presented_local_ids

    manifest = [
        dict(kind="source", at=24, aliases=["boundary"],
             sample_refs=[dict(requested="123/4")]),
        dict(kind="source", at=37, aliases=["start-z", "start-a"],
             sample_refs=[dict(requested="24/1")]),
        dict(kind="source", at=99, aliases=["shared-z", "shared-a"],
             sample_refs=[dict(requested="24/1"), dict(requested="123/4")]),
        dict(kind="style", at=24, aliases=["style-only"],
             sample_refs=[dict(requested="24/1")]),
    ]
    before = json.dumps(manifest, sort_keys=True)
    assert _presented_local_ids(manifest, TimeRange(start=24, end=30.75)) == {
        "start-z", "start-a", "shared-z", "shared-a"}
    assert _presented_local_ids(manifest, TimeRange(start=30.75, end=38)) == {
        "boundary", "shared-z", "shared-a"}
    assert _presented_local_ids(manifest, TimeRange(start=38, end=40)) == set()
    assert json.dumps(manifest, sort_keys=True) == before


def test_director_rebuilds_image_eligibility_for_actual_followup_manifest(scene: Any, tmp_path: Path) -> None:
    from clipperx.planner_contracts import TimeRange
    from clipperx.visual_contracts import VisualDirectorResponse
    from clipperx.visual_director import _presented_local_ids

    base_schema = VisualDirectorResponse.model_json_schema()
    client = FakeClient([more(1), ready])
    path = run(scene, tmp_path, client)
    assert path.name == "visual_plan.json" and len(client.payloads) == 2
    tables = []
    for i, payload in enumerate(client.payloads):
        host = payload["clip"]["host_context"]
        table = host["local_source_image_ids_by_shot"]
        assert table == [dict(source=part, image_evidence_ids=sorted(
            _presented_local_ids(payload["frame_manifest"], TimeRange.model_validate(part))))
            for part in host["required_shots"]]
        saved = json.loads((path.parent / f"call-{i + 1:02d}" / "request.json").read_text())
        assert saved["user_payload"] == payload
        tables.append(table)
        assert len(client.images[i]) == len(payload["frame_manifest"]) <= 8
        assert sum(row["kind"] == "source" for row in payload["frame_manifest"]) <= 5
    assert tables[0] != tables[1]
    first, second = [payload["clip"]["host_context"] for payload in client.payloads]
    assert first["required_shots"] == second["required_shots"] == [
        dict(start=60.0, end=62.0), dict(start=62.0, end=64.0)]
    assert first["fingerprints"] == second["fingerprints"] and len(first["fingerprints"]) == 5
    assert client.output_models[0] is client.output_models[1]
    assert VisualDirectorResponse.model_json_schema() == base_schema
    assert len(second["evidence_bundles"]) == 2
    assert len(list((path.parent / "evidence").glob("followup*.json"))) == 1
    sources = [row for row in client.payloads[1]["frame_manifest"] if row["kind"] == "source"]
    shared = [row for row in sources if len(row["aliases"]) > 1]
    assert shared
    eligible = {alias for row in tables[1] for alias in row["image_evidence_ids"]}
    assert any(set(row["aliases"]).issubset(eligible) for row in shared)


def test_director_rechecks_camera_boundary_then_finishes(scene: Any, tmp_path: Path) -> None:
    client = FakeClient([more(1), ready])
    path = run(scene, tmp_path, client)
    assert len(client.payloads) == 2
    bundles = list((path.parent / "evidence").glob("followup*.json"))
    assert len(bundles) == 1
    assert json.loads(bundles[0].read_text())["windows"] == [{"start": 59.0, "end": 63.0}]
    assert client.payloads[1]["clip"]["host_context"]["evidence_bundles"]


def test_director_stops_after_three_evidence_iterations(scene: Any, tmp_path: Path) -> None:
    client = FakeClient([more(1), more(2, 58, 63), more(3, 57, 63), ready])
    path = run(scene, tmp_path, client)
    assert json.loads(path.read_text())["status"] == "review_required"
    assert len(client.payloads) == 3
    assert not (path.parent / "visual_plan.json").exists()


def test_director_never_receives_full_transcript(scene: Any, tmp_path: Path) -> None:
    client = FakeClient([more(1), ready])
    run(scene, tmp_path, client)
    assert all("DISTANT_SECRET_SENTINEL" not in json.dumps(p) for p in client.payloads)
    assert all("local payoff" in json.dumps(p) for p in client.payloads)


@pytest.mark.parametrize("field", ["source_fingerprint", "composition_fingerprint",
    "source_timeline_fingerprint", "identity_fingerprint", "style_library_fingerprint"])
def test_director_rejects_fingerprint_forgery(scene: Any, tmp_path: Path, field: str) -> None:
    def forged(payload: Any) -> Any:
        response = ready(payload)
        response["plan"][field] = "forged"
        return response
    path = run(scene, tmp_path, FakeClient([forged]))
    assert json.loads(path.read_text())["status"] == "review_required"


def test_director_rejects_repeated_signature_under_new_request_id(scene: Any,
                                                                tmp_path: Path) -> None:
    repeated = more(1)
    repeated["evidence_requests"][0]["request_id"] = "different-id"
    path = run(scene, tmp_path, FakeClient([more(1), repeated]))
    assert "signature" in json.loads(path.read_text())["reason"]
    assert len(list((path.parent / "evidence").glob("followup*.json"))) == 1


@pytest.mark.parametrize("mutation", ["gap", "transition", "cover", "identity", "style", "cue",
                                      "title", "evidence", "primary"])
def test_director_host_rejects_invalid_plan(scene: Any, tmp_path: Path, mutation: str) -> None:
    def invalid(payload: Any) -> Any:
        response = ready(payload)
        plan = response["plan"]
        first = plan["shots"][0]
        if mutation == "gap":
            plan["shots"].pop()
        elif mutation == "transition":
            plan["shots"] = [dict(first, source=dict(start=60, end=64), output=dict(start=0, end=4))]
        elif mutation == "cover":
            first["layers"][0]["fill"] = "cover"
        elif mutation == "identity":
            first["layers"][0].update(kind="streamer_camera", source_ref="face_elsewhere")
        elif mutation == "style":
            response["style_card_ids"] = ["absent"]
        elif mutation == "cue":
            first["audio"] = dict(cues=[dict(asset_id="random_boom", start=0, end=1)])
        elif mutation == "title":
            first["title"] = dict(text="Wrong title", start=0, end=1,
                destination=dict(x=.1, y=.1, width=.8, height=.1), z_index=2)
        elif mutation == "evidence":
            first["evidence_ids"] = ["foreign-evidence"]
        elif mutation == "primary":
            first["primary_source"] = "streamer"
        return response
    path = run(scene, tmp_path, FakeClient([invalid]))
    assert json.loads(path.read_text())["status"] == "review_required"


def test_director_source_drift_is_review_receipt(scene: Any, tmp_path: Path) -> None:
    video, *_ = scene
    def drift(payload: Any) -> Any:
        video.touch()
        return ready(payload)
    path = run(scene, tmp_path, FakeClient([drift]))
    assert "source" in json.loads(path.read_text())["reason"].lower()


def test_director_unique_run_never_returns_stale_success(scene: Any, tmp_path: Path) -> None:
    first = run(scene, tmp_path, FakeClient([ready]))
    second = run(scene, tmp_path, FakeClient([dict(status="review_required", reasoning="unknown")]))
    assert first.parent != second.parent
    assert json.loads(first.read_text())["contract_version"] == 2
    assert json.loads(second.read_text())["status"] == "review_required"


def test_director_rejects_limits_that_bypass_hard_image_caps() -> None:
    with pytest.raises(ValueError):
        VisualDirectorLimits(max_source_images_per_call=6)


def test_director_client_failure_receipt_never_exposes_config(scene: Any, tmp_path: Path) -> None:
    class BrokenClient(FakeClient):
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("Authorization: Bearer PRIVATE_API_SECRET")
    path = run(scene, tmp_path, BrokenClient([]))
    assert "PRIVATE_API_SECRET" not in path.read_text()
    assert json.loads(path.read_text())["status"] == "review_required"


def test_director_tracks_followup_budget_per_segment(scene: Any, tmp_path: Path) -> None:
    video, clip, timeline, library = scene
    second = clip.segments[0].model_copy(update={"source": type(clip.segments[0].source)(start=80, end=84),
                                               "output_order": 1})
    clip = ClipPlan.model_validate(dict(clip.model_dump(), segments=[
        clip.segments[0].model_dump(), second.model_dump()]))
    requests = [more(1), more(2, 58, 63), more(3, 79, 83)]
    client = FakeClient([*requests, ready])
    path = run((video, clip, timeline, library), tmp_path, client)
    assert json.loads(path.read_text()).get("contract_version") == 2
    assert len(client.payloads) == 4


def test_director_rejects_wrong_clip_association(scene: Any, tmp_path: Path) -> None:
    def wrong(payload: Any) -> Any:
        response = ready(payload)
        response["plan"]["clip_id"] = "other-clip"
        return response
    path = run(scene, tmp_path, FakeClient([wrong]))
    assert "clip_id" in json.loads(path.read_text())["reason"]


def test_director_disallows_overlay_over_known_content(scene: Any, tmp_path: Path) -> None:
    def obscured(payload: Any) -> Any:
        response = ready(payload)
        response["plan"]["shots"][0]["title"] = dict(text="Local headline", start=0, end=1,
            destination=dict(x=.1, y=.4, width=.8, height=.1), z_index=2)
        return response
    path = run(scene, tmp_path, FakeClient([obscured]))
    assert json.loads(path.read_text())["status"] == "review_required"


def identity_scene(scene: Any, *, local_confident: bool) -> Any:
    video, clip, timeline, library = scene
    from clipperx.tests.test_visual_contracts import make_face_track, make_webcam_track
    face = make_face_track(first_seen=0, last_seen=90)
    webcam = make_webcam_track(source_ranges=[dict(start=0, end=90)])
    spans = [dict(s.model_dump(), **{
        "state": "full_camera", "streamer_track_id": "face_main" if local_confident else None,
        "identity_status": "confident" if local_confident else "unavailable", "webcam_track_id": None,
    }) for s in timeline.spans]
    timeline = SourceVisualTimeline.model_validate(dict(timeline.model_dump(), spans=spans,
        face_tracks=[face.model_dump()], webcam_tracks=[webcam.model_dump()]))
    return video, clip, timeline, library


def test_director_accepts_confident_local_streamer_track(scene: Any, tmp_path: Path) -> None:
    def camera(payload: Any) -> Any:
        response = ready(payload)
        for shot in response["plan"]["shots"]:
            shot["primary_source"] = "streamer"
            shot["layers"][0].update(kind="streamer_camera", source_ref="face_main")
        return response
    path = run(identity_scene(scene, local_confident=True), tmp_path, FakeClient([camera]))
    assert json.loads(path.read_text()).get("contract_version") == 2


def test_director_known_global_face_cannot_override_unavailable_local_identity(scene: Any,
                                                                            tmp_path: Path) -> None:
    def camera(payload: Any) -> Any:
        response = ready(payload)
        response["plan"]["shots"][0]["layers"][0].update(kind="streamer_camera", source_ref="face_main")
        return response
    path = run(identity_scene(scene, local_confident=False), tmp_path, FakeClient([camera]))
    assert json.loads(path.read_text())["status"] == "review_required"


def test_director_full_source_can_preserve_unavailable_identity(scene: Any, tmp_path: Path) -> None:
    path = run(identity_scene(scene, local_confident=False), tmp_path, FakeClient([ready]))
    assert json.loads(path.read_text()).get("contract_version") == 2


def test_director_payload_uses_only_approved_top_level_fields(scene: Any, tmp_path: Path) -> None:
    client = FakeClient([ready])
    run(scene, tmp_path, client)
    assert set(client.payloads[0]) == {"clip", "local_transcript", "source_spans", "face_tracks",
        "webcam_tracks", "audio_windows", "safe_zones", "style_cards", "frame_manifest", "creator_preferences"}


def test_director_uses_approved_platform_safe_zones(scene: Any, tmp_path: Path) -> None:
    def below_title_zone(payload: Any) -> Any:
        response = ready(payload)
        response["plan"]["shots"][0]["title"] = dict(text="Local headline", start=0, end=1,
            destination=dict(x=.1, y=.26, width=.8, height=.06), z_index=3)
        return response
    client = FakeClient([below_title_zone])
    path = run(scene, tmp_path, client)
    assert json.loads(path.read_text())["status"] == "review_required"
    assert client.payloads[0]["safe_zones"]["profile"] == "common_vertical"


def test_director_accepts_host_brand_title_font_and_safe_subtitles(scene: Any, tmp_path: Path) -> None:
    def overlays(payload: Any) -> Any:
        response = ready(payload)
        for shot in response["plan"]["shots"]:
            shot["title"] = dict(text="Local headline", font_asset_id="headline_impact", start=0, end=1,
                destination=dict(x=.1, y=.1, width=.8, height=.1), z_index=3)
            shot["brand"] = dict(asset_id="kick_brand", start=0, end=1,
                destination=dict(x=.4, y=.25, width=.2, height=.05), z_index=4)
            shot["subtitles"] = dict(destination=dict(x=.2, y=.7, width=.6, height=.08))
        return response
    path = run(scene, tmp_path, FakeClient([overlays]))
    assert json.loads(path.read_text()).get("contract_version") == 2


def test_style_retrieval_full_camera_regions_do_not_mean_external_content(scene: Any,
                                                                        monkeypatch: Any) -> None:
    from clipperx.style_retrieval import RetrievalResult, StyleMatch
    from clipperx.tests.test_style_retrieval import card
    from clipperx.visual_director import _select_styles
    chosen = card("full", scene_family="full_camera", content_regions=[])
    queries = []
    def retrieve(query: Any, *args: Any, **kwargs: Any) -> Any:
        queries.append(query)
        return RetrievalResult((StyleMatch(chosen, 1, 1, ()),), "sparse")
    monkeypatch.setattr("clipperx.visual_director.retrieve_style_cards", retrieve)
    _, clip, timeline, library = identity_scene(scene, local_confident=True)
    cards, coverage = _select_styles(clip, timeline, library, VisualDirectorLimits())
    assert cards[0].card_id == "full"
    assert all(not q.preserve_content for q in queries)
    assert all(q.scene_family.value == "full_camera" for q in queries)
    assert coverage


@pytest.mark.parametrize("state", [
    SourceState.CONTENT_WITH_CAMERA, SourceState.CONTENT_ONLY,
    SourceState.BROWSER_TEXT, SourceState.REACTED_VIDEO,
])
def test_style_bridge_keeps_declared_content_without_detected_regions(
    scene: tuple[Path, ClipPlan, SourceVisualTimeline, StyleLibrary],
    monkeypatch: pytest.MonkeyPatch, state: SourceState,
) -> None:
    from clipperx.tests.test_style_retrieval import CAMERA, FakeEncoder, card
    from clipperx.tests.test_visual_contracts import make_webcam_track
    from clipperx.visual_director import _select_styles

    _, clip, timeline, library = scene
    with_camera = state == SourceState.CONTENT_WITH_CAMERA
    camera_regions = [CAMERA] if with_camera else []
    compatible = card("external-content", scene_family=(
        "unknown" if state != SourceState.CONTENT_ONLY else "content_only"),
        camera_regions=camera_regions)
    full_camera = card("full-camera-only", scene_family="full_camera", content_regions=[])

    class MemoryLibrary(StyleLibrary):
        def list_cards(self) -> list[StyleCard]:
            return [full_camera, compatible]

    data = timeline.model_dump()
    data["spans"] = [dict(s.model_dump(), state=state, important_regions=[], text_regions=[],
        webcam_track_id="webcam_main" if with_camera else None) for s in timeline.spans]
    data["webcam_tracks"] = [make_webcam_track(face_track_ids=[],
        source_ranges=[dict(start=0, end=100)]
        ).model_dump()] if with_camera else []
    timeline = SourceVisualTimeline.model_validate(data)
    monkeypatch.setattr("clipperx.style_retrieval.LocalStyleEncoder", FakeEncoder)
    selected, coverage = _select_styles(
        clip, timeline, MemoryLibrary(library.root), VisualDirectorLimits())
    assert [c.card_id for c in selected] == [compatible.card_id]
    assert all(row["coverage"] == "sparse" for row in coverage)
    assert all(row["scene_family"] == (
        "content_only" if state == SourceState.CONTENT_ONLY else "unknown") for row in coverage)
    assert all(any(r["card_id"] == full_camera.card_id
                   and "content:preservation_not_supported" in r["reasons"]
                   for r in row["rejections"]) for row in coverage)


def test_style_bridge_full_camera_face_regions_pass_real_compatibility(
    scene: tuple[Path, ClipPlan, SourceVisualTimeline, StyleLibrary],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clipperx.tests.test_style_retrieval import FakeEncoder, card
    from clipperx.visual_director import _select_styles

    _, clip, timeline, library = scene
    chosen = card("talking-head", scene_family="full_camera", content_regions=[])

    class MemoryLibrary(StyleLibrary):
        def list_cards(self) -> list[StyleCard]:
            return [chosen]

    data = timeline.model_dump()
    data["spans"] = [dict(s.model_dump(), state="full_camera") for s in timeline.spans]
    timeline = SourceVisualTimeline.model_validate(data)
    assert all(span.important_regions for span in timeline.spans)
    monkeypatch.setattr("clipperx.style_retrieval.LocalStyleEncoder", FakeEncoder)
    selected, coverage = _select_styles(
        clip, timeline, MemoryLibrary(library.root), VisualDirectorLimits())
    assert [c.card_id for c in selected] == [chosen.card_id]
    assert all(row["coverage"] == "sparse" for row in coverage)


def test_style_retrieval_considers_each_local_state_under_global_cap(scene: Any,
                                                                  monkeypatch: Any) -> None:
    from clipperx.style_retrieval import RetrievalResult, StyleMatch
    from clipperx.tests.test_style_retrieval import card
    from clipperx.visual_contracts import SourceVisualSpan
    from clipperx.visual_director import _select_styles
    _, clip, timeline, library = scene
    spans = [SourceVisualSpan.model_validate(dict(s.model_dump(), state=state))
             for s, state in zip(timeline.spans, ["full_camera", "content_only"])]
    timeline = SourceVisualTimeline.model_validate(dict(timeline.model_dump(), spans=[s.model_dump() for s in spans]))
    queries = []
    def retrieve(query: Any, *args: Any, **kwargs: Any) -> Any:
        queries.append(query)
        family = query.scene_family.value
        cards = [card(f"{family}-{i}", scene_family=family) for i in range(3)]
        return RetrievalResult(tuple(StyleMatch(c, 1, 1, ()) for c in cards), "sufficient")
    monkeypatch.setattr("clipperx.visual_director.retrieve_style_cards", retrieve)
    selected, _ = _select_styles(clip, timeline, library, VisualDirectorLimits())
    assert {q.scene_family.value for q in queries} == {"full_camera", "content_only"}
    assert {c.scene_family.value for c in selected} == {"full_camera", "content_only"}
    assert len(selected) <= 3


def test_director_manifest_caps_five_source_plus_three_style_views(tmp_path: Path) -> None:
    from PIL import Image
    from clipperx.tests.test_style_retrieval import card
    from clipperx.visual_director import _manifest
    library = StyleLibrary(tmp_path / "library")
    library.root.mkdir()
    image = library.root / "view.jpg"
    Image.new("RGB", (32, 32)).save(image)
    cards = [card(str(i), keyframes=[dict(timestamp=0, path="view.jpg"),
                                    dict(timestamp=1, path="view.jpg")]) for i in range(4)]
    bundles = [{"frames": [manifest_frame(i,image) for i in range(12)]}]
    manifest, paths = _manifest(bundles, cards, library, VisualDirectorLimits())
    assert len(paths) == 8
    assert sum(item["kind"] == "source" for item in manifest) == 5
    assert sum(item["kind"] == "style" for item in manifest) == 3
    assert all(item["approval"] == "unreviewed" for item in manifest if item["kind"] == "style")


def test_director_rejects_requested_window_outside_local_expansion(scene: Any, tmp_path: Path) -> None:
    path = run(scene, tmp_path, FakeClient([more(1, 0, 4)]))
    assert json.loads(path.read_text())["status"] == "review_required"
    assert not list((path.parent / "evidence").glob("followup*.json"))


def test_director_rechecks_source_authority_at_publication(scene: Any, tmp_path: Path,
                                                        monkeypatch: Any) -> None:
    import clipperx.visual_director as module
    validate = module._validate_final
    def drift(*args: Any, **kwargs: Any) -> Any:
        value = validate(*args, **kwargs)
        scene[0].touch()
        return value
    monkeypatch.setattr(module, "_validate_final", drift)
    path = run(scene, tmp_path, FakeClient([ready]))
    assert json.loads(path.read_text())["status"] == "review_required"
    assert not (path.parent / "visual_plan.json").exists()


def test_director_asset_registry_checks_asset_kind(scene: Any, tmp_path: Path) -> None:
    def wrong_kind(payload: Any) -> Any:
        response = ready(payload)
        response["plan"]["shots"][0]["title"] = dict(text="Local headline", font_asset_id="kick_brand",
            start=0, end=1, destination=dict(x=.1, y=.1, width=.8, height=.1), z_index=3)
        return response
    path = run(scene, tmp_path, FakeClient([wrong_kind]))
    assert json.loads(path.read_text())["status"] == "review_required"


def test_director_fractional_editorial_boundaries_use_only_inside_frames(scene: Any,
                                                                      tmp_path: Path) -> None:
    video, clip, timeline, library = scene
    data = clip.model_dump()
    data["segments"][0]["source"] = {"start": 60.1, "end": 63.9}
    clip = ClipPlan.model_validate(data)
    path = run((video, clip, timeline, library), tmp_path, FakeClient([ready]))
    assert json.loads(path.read_text()).get("contract_version") == 2
    bundle = json.loads((path.parent / "evidence" / "initial.json").read_text())
    assert all(60.1 <= frame["at"] < 63.9 for frame in bundle["frames"])


def test_director_preserves_known_local_face_even_when_not_streamer(scene: Any,
                                                                 tmp_path: Path) -> None:
    from clipperx.tests.test_visual_contracts import make_face_track
    video, clip, timeline, library = scene
    data = timeline.model_dump()
    for span in data["spans"]:
        span["important_regions"] = []
    data["face_tracks"] = [make_face_track(first_seen=60, last_seen=64,
        positions=[dict(x=.05, y=.05, width=.2, height=.2)]).model_dump()]
    timeline = SourceVisualTimeline.model_validate(data)
    def cut_face(payload: Any) -> Any:
        response = ready(payload)
        response["plan"]["shots"][0]["layers"][0]["source_crop"] = dict(x=.4, y=0, width=.6, height=1)
        return response
    path = run((video, clip, timeline, library), tmp_path, FakeClient([cut_face]))
    assert json.loads(path.read_text())["status"] == "review_required"


def test_director_rejects_crop_motion_crossing_brand_between_keyframes(scene: Any,
                                                                    tmp_path: Path) -> None:
    video, clip, timeline, library = scene
    data = timeline.model_dump()
    for span in data["spans"]:
        span["important_regions"] = [dict(x=.45, y=.4, width=.01, height=.03)]
    timeline = SourceVisualTimeline.model_validate(data)
    def motion(payload: Any) -> Any:
        response = ready(payload)
        shot = response["plan"]["shots"][0]
        crops = [dict(x=x, y=0, width=.31640625, height=1) for x in (.2, .4)]
        shot["layers"][0].update(fill="cover", source_crop=crops[0],
            crop_track=[dict(at=0, crop=crops[0]), dict(at=2, crop=crops[1])])
        shot["brand"] = dict(asset_id="kick_brand", start=0, end=2,
            destination=dict(x=.49, y=.4, width=.02, height=.03), z_index=3)
        return response
    path = run((video, clip, timeline, library), tmp_path, FakeClient([motion]))
    assert json.loads(path.read_text())["status"] == "review_required"


def test_director_rejects_probe_decode_dimension_disagreement(scene: Any, tmp_path: Path,
                                                            monkeypatch: Any) -> None:
    from clipperx.tests.test_evidence import install_observer
    import clipperx.evidence as e
    from clipperx.render_clock import SourceTimeOrigin,StreamClock
    origin,_=install_observer(monkeypatch,scene[0])
    altered=SourceTimeOrigin(origin.path,origin.size,origin.mtime_ns,origin.fingerprint,
        origin.origin,origin.duration,(StreamClock(0,"video",origin.streams[0].time_base,
        origin.streams[0].start,origin.streams[0].end,108,192),origin.streams[1]),origin.decoder_id,{})
    monkeypatch.setattr(e,"probe_source",lambda *a:altered)
    client = FakeClient([ready])
    path = run(scene, tmp_path, client)
    assert json.loads(path.read_text())["status"] == "review_required"
    assert not client.payloads


def cropped_identity_scene(scene: Any) -> Any:
    video, clip, timeline, library = identity_scene(scene, local_confident=True)
    data = timeline.model_dump()
    for span in data["spans"]:
        span["important_regions"] = []
    data["face_tracks"][0]["positions"] = [dict(x=.05, y=.05, width=.1, height=.1)]
    return video, clip, SourceVisualTimeline.model_validate(data), library


def streamer_crop_response(payload: Any, mode: str) -> Any:
    response = ready(payload)
    for shot in response["plan"]["shots"]:
        layer = dict(shot["layers"][0], layer_id="camera", kind="streamer_camera",
            source_ref="face_main", source_crop=dict(x=0, y=0, width=.2, height=.2),
            destination=dict(x=.75, y=.75, width=.2, height=.2), z_index=1)
        if mode == "wrong":
            layer["source_crop"] = dict(x=.7, y=.7, width=.2, height=.2)
        elif mode == "cover":
            layer.update(source_crop=dict(x=0, y=0, width=1, height=1), fill="cover")
        elif mode == "motion":
            layer["crop_track"] = [dict(at=0, crop=layer["source_crop"]),
                dict(at=shot["output"]["end"] - shot["output"]["start"],
                     crop=dict(x=.7, y=.7, width=.2, height=.2))]
        shot["layers"].append(layer)
    return response


@pytest.mark.parametrize("mode", ["wrong", "cover", "motion"])
def test_director_rejects_false_streamer_crop_despite_preserving_source(
    scene: Any, tmp_path: Path, mode: str,
) -> None:
    path = run(cropped_identity_scene(scene), tmp_path,
        FakeClient([lambda payload: streamer_crop_response(payload, mode)]))
    receipt = json.loads(path.read_text())
    assert receipt.get("status") == "review_required"
    assert "Streamer layer" in receipt["reason"]


def test_director_accepts_legitimate_streamer_closeup(scene: Any, tmp_path: Path) -> None:
    path = run(cropped_identity_scene(scene), tmp_path,
        FakeClient([lambda payload: streamer_crop_response(payload, "valid")]))
    assert json.loads(path.read_text()).get("contract_version") == 2


def test_final_validation_binds_each_streamer_layer_to_visible_track_geometry(scene: Any) -> None:
    from clipperx.visual_contracts import VisualDirectorResponse
    from clipperx.visual_director import _fingerprints, _required_shots, _validate_final
    _, clip, timeline, library = cropped_identity_scene(scene)
    fingerprints = _fingerprints(clip, timeline, library.fingerprint)
    payload = {"clip": {"clip_id": clip.clip_id, "host_context": {
        "fingerprints": fingerprints, "required_shots": _required_shots(clip, timeline)}}}
    response = VisualDirectorResponse.model_validate(streamer_crop_response(payload, "wrong"))
    with pytest.raises(ValueError, match="Streamer layer"):
        _validate_final(response, clip, timeline, fingerprints, [], [], 192, 108)


def requested_windows_response(windows: list[tuple[float, float]]) -> Any:
    response = more(1)
    response["evidence_requests"] = [
        dict(more(i + 1, start, end)["evidence_requests"][0])
        for i, (start, end) in enumerate(windows)]
    return response


def test_director_next_call_has_actual_views_from_both_requested_windows(scene: Any,
                                                                       tmp_path: Path) -> None:
    video, clip, timeline, library = scene
    data = timeline.model_dump()
    data["spans"] = [dict(data["spans"][0], span_id=f"window-{i}",
        source=dict(start=start, end=end), evidence_ids=["e-0" if start < 62 else "e-1"])
        for i, (start, end) in enumerate([(0, 62), (62, 66), (66, 67), (67, 68),
                                         (68, 69), (69, 100)])]
    timeline = SourceVisualTimeline.model_validate(data)
    client = FakeClient([requested_windows_response([(59, 61), (65, 74)]), ready])
    path = run((video, clip, timeline, library), tmp_path, client)
    assert json.loads(path.read_text()).get("contract_version") == 2
    sources = [frame for frame in client.payloads[1]["frame_manifest"] if frame["kind"] == "source"]
    times = {frame["at"] for frame in sources}
    assert len(times) == len(sources) <= 5
    for start, end in [(59, 61), (65, 74)]:
        assert len([at for at in times if start <= at < end]) >= 2
    assert all(Path(frame["path"]) in client.images[1] for frame in sources)
    assert len(client.payloads[1]["clip"]["host_context"]["evidence_bundles"]) == 3


def test_director_reviews_when_requested_temporal_views_cannot_fit(scene: Any,
                                                                tmp_path: Path) -> None:
    client = FakeClient([requested_windows_response([(56, 59), (63, 66), (68, 71)]), ready])
    path = run(scene, tmp_path, client)
    receipt = json.loads(path.read_text())
    assert receipt.get("status") == "review_required"
    assert "image budget" in receipt["reason"]
    assert len(client.payloads) == 1
    assert not (path.parent / "visual_plan.json").exists()


@pytest.mark.parametrize("mode", ["valid", "face_only", "missing_local"])
def test_director_webcam_assertion_requires_its_own_local_geometry(
    scene: Any, tmp_path: Path, mode: str,
) -> None:
    video, clip, timeline, library = cropped_identity_scene(scene)
    data = timeline.model_dump()
    for span in data["spans"]:
        span["webcam_track_id"] = "webcam_main"
    data["webcam_tracks"][0]["observations"] = [dict(at=at,
        rect=dict(x=0, y=0, width=.3, height=.3), confidence=1, evidence_id=f"camera-{at}")
        for at in ([1] if mode == "missing_local" else [61, 63])]
    timeline = SourceVisualTimeline.model_validate(data)
    def camera(payload: Any) -> Any:
        response = streamer_crop_response(payload, "valid")
        for shot in response["plan"]["shots"]:
            shot["layers"][-1]["source_ref"] = "webcam_main"
            if mode == "valid":
                shot["layers"][-1]["source_crop"] = dict(x=0, y=0, width=.3, height=.3)
        return response
    path = run((video, clip, timeline, library), tmp_path, FakeClient([camera]))
    result = json.loads(path.read_text())
    if mode == "valid":
        assert result.get("contract_version") == 2
    else:
        assert result.get("status") == "review_required"
        assert "Streamer layer" in result["reason"]


def test_manifest_spreads_dense_requested_windows_under_five_view_cap(tmp_path: Path) -> None:
    from PIL import Image
    from clipperx.visual_director import _manifest
    image = tmp_path / "view.jpg"
    Image.new("RGB", (16, 16)).save(image)
    bundles = [dict(frames=[manifest_frame(at,image) for at in range(start, start + 5)],
        request=dict(source=dict(start=start, end=start + 5), reason="camera_boundary"))
        for start in (60, 120)]
    manifest, images = _manifest(bundles, [], StyleLibrary(tmp_path / "library"), VisualDirectorLimits())
    times = [frame["at"] for frame in manifest]
    assert {60, 64, 120, 124}.issubset(times)
    assert len(set(times)) == len(images) == 5
    assert all(frame["sha256"] for frame in manifest)


def test_director_reviews_temporal_request_with_only_one_decoded_instant(scene: Any,
                                                                      tmp_path: Path) -> None:
    client = FakeClient([more(1, 59, 59.09), ready])
    path = run(scene, tmp_path, client)
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "review_required"
    assert "distinct temporal" in receipt["reason"]
    assert len(client.payloads) == 1


def test_director_request_schema_binds_all_host_fingerprints(scene: Any, tmp_path: Path) -> None:
    from typing import cast

    from pydantic import ValidationError

    from clipperx.structured_client import _strict_json_schema
    from clipperx.visual_contracts import VisualDirectorResponse
    from clipperx.visual_director import _response_model_for_fingerprints

    base_schema = VisualDirectorResponse.model_json_schema()
    client = FakeClient([more(1), ready])
    path = run(scene, tmp_path, client)
    assert path.name == "visual_plan.json"
    assert len(client.output_models) == 2
    assert client.output_models[0] is client.output_models[1]
    response_model = cast(type[VisualDirectorResponse], client.output_models[0])
    schema = _strict_json_schema(response_model)

    expected = client.payloads[0]["clip"]["host_context"]["fingerprints"]
    names = (
        "source_fingerprint",
        "composition_fingerprint",
        "source_timeline_fingerprint",
        "identity_fingerprint",
        "style_library_fingerprint",
    )
    plan_schema = next(
        definition for definition in schema["$defs"].values()
        if isinstance(definition, dict)
        and isinstance(definition.get("properties"), dict)
        and set(names).issubset(definition["properties"])
    )
    assert set(names).issubset(plan_schema["required"])
    assert {name: plan_schema["properties"][name]["const"] for name in names} == {
        name: expected[name] for name in names
    }
    assert VisualDirectorResponse.model_json_schema() == base_schema
    assert response_model.model_validate(ready(client.payloads[0])).status == "ready"
    assert response_model.model_validate(
        {"status": "review_required", "reasoning": "bounded review"}
    ).status == "review_required"
    assert response_model.model_validate(more(2)).status == "need_more_evidence"

    mutable_fingerprints = dict(expected)
    isolated_model = _response_model_for_fingerprints(mutable_fingerprints)
    mutable_fingerprints[names[0]] = "changed-after-binding"
    isolated_schema = _strict_json_schema(isolated_model)
    isolated_plan = next(
        definition for definition in isolated_schema["$defs"].values()
        if isinstance(definition, dict)
        and isinstance(definition.get("properties"), dict)
        and set(names).issubset(definition["properties"])
    )
    assert isolated_plan["properties"][names[0]]["const"] == expected[names[0]]

    for name in names:
        tampered = ready(client.payloads[0])
        tampered["plan"][name] = "wrong-" + name
        assert VisualDirectorResponse.model_validate(tampered).plan is not None
        with pytest.raises(ValidationError):
            response_model.model_validate(tampered)


def test_director_final_fingerprint_gate_rejects_client_ignoring_request_model(
    scene: Any, tmp_path: Path,
) -> None:
    from typing import get_args

    from clipperx.visual_contracts import VisualDirectorResponse

    class NonconformingClient(FakeClient):
        def generate(self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
                     images: list[Path] | None = None, temperature: float = 0.2) -> StructuredResult[ModelT]:
            payload = json.loads(user_prompt)
            self.payloads.append(payload)
            self.images.append(images or [])
            self.output_models.append(output_model)
            response = ready(payload)
            response["plan"]["source_timeline_fingerprint"] = "wrong-timeline-fingerprint"
            parsed = VisualDirectorResponse.model_validate(response)
            plan_model = next(
                model for model in get_args(output_model.model_fields["plan"].annotation)
                if isinstance(model, type) and issubclass(model, BaseModel)
            )
            assert parsed.plan is not None
            unvalidated_plan = plan_model.model_construct(
                **{name: getattr(parsed.plan, name) for name in plan_model.model_fields}
            )
            unvalidated_response = output_model.model_construct(
                status=parsed.status,
                reasoning=parsed.reasoning,
                plan=unvalidated_plan,
                evidence_requests=parsed.evidence_requests,
                style_card_ids=parsed.style_card_ids,
            )
            return StructuredResult(unvalidated_response, {"total_tokens": 10}, {})

    client = NonconformingClient([ready])
    path = run(scene, tmp_path, client)
    receipt = json.loads(path.read_text())
    saved_response = json.loads((path.parent / "call-01" / "response.json").read_text())
    assert path.name == "receipt.json"
    assert receipt["status"] == "review_required"
    assert not (path.parent / "visual_plan.json").exists()
    assert saved_response["plan"]["source_timeline_fingerprint"] == "wrong-timeline-fingerprint"
