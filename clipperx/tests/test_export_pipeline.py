from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import threading
from typing import Any

import pytest

from clipperx import visual_director
from clipperx.render_assets import RenderAssetRegistry
from clipperx.render_contracts import RenderProfile
from clipperx.tests.test_visual_planner_integration import (
    Inputs, inputs, ready, review, visual_run,
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def planning_manifest(case: Inputs, monkeypatch: pytest.MonkeyPatch,
                      statuses: tuple[str, ...] = ("ready", "ready"), *,
                      clip_ids: tuple[str, ...] | None = None) -> Path:
    original = case.composition.clips[0]
    case.composition.clips = [original.model_copy(
        update={"clip_id": clip_ids[index] if clip_ids is not None else
                original.clip_id if index == 0 else f"clip-{index}"}, deep=True)
        for index in range(len(statuses))]
    case.plan.write_text(case.composition.model_dump_json(), encoding="utf-8")

    def director(video: Path, clip: Any, timeline: Any, **kwargs: Any) -> Path:
        index = next(i for i, item in enumerate(case.composition.clips)
                     if item.clip_id == clip.clip_id)
        root = case.root / f"director-{index}"
        return (ready(root, clip, timeline, case.library) if statuses[index] == "ready"
                else review(root, clip, timeline))

    with monkeypatch.context() as patch:
        patch.setattr(visual_director, "run_visual_director", director)
        result = visual_run(case)
    manifest = read(result.visual_plan_manifest)
    for entry, status in zip(manifest["clips"], statuses):
        if status == "pending":
            entry.update(status="pending", visual_plan=None, receipt=None,
                         reason="awaiting visual evidence")
            entry.pop("provenance", None)
            entry.pop("provenance_sha256", None)
            entry.pop("host_context", None)
    result.visual_plan_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    return result.visual_plan_manifest


class Renderer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, Path, dict[str, Any]]] = []

    def __call__(self, source: Path, plan: Path, output: Path, **kwargs: Any) -> list[Path]:
        self.calls.append((source, plan, output, kwargs))
        output.mkdir(parents=True, exist_ok=True)
        target = output / "clip.mp4"
        target.write_bytes(b"external renderer bytes: " + str(len(self.calls)).encode())
        return [target]


def export(case: Inputs, manifest: Path, renderer: Any, **kwargs: Any) -> Any:
    from clipperx.export_pipeline import export_visual_manifest
    assets = kwargs.pop("assets", RenderAssetRegistry({}, {}, {}))
    return export_visual_manifest(case.video, manifest, case.root / "exports",
                                  assets=assets, render=renderer,
                                  **kwargs)


def test_two_ready_clips_render_without_claiming_quality(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch)
    renderer = Renderer()
    result = export(inputs, manifest, renderer)
    assert len(result.clips) == len(renderer.calls) == 2
    assert isinstance(result.clips, tuple)
    assert read(result.summary)["status"] == "rendered_unreviewed"
    state = read(result.manifest)
    assert [entry["status"] for entry in state["clips"]] == ["rendered_unreviewed"] * 2
    assert "qa_passed" not in json.dumps(state)
    for (source, plan, output, kwargs), clip in zip(renderer.calls, result.clips):
        assert source == inputs.video.resolve()
        assert plan.name == "visual_plan.json"
        assert kwargs["transcript_path"] is None
        assert clip.is_relative_to(output) and output.is_relative_to(inputs.root / "exports")
    with pytest.raises(FrozenInstanceError):
        result.clips = ()


def test_mixed_manifest_preserves_review_and_pending_reasons(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready", "review_required", "pending"))
    renderer = Renderer()
    result = export(inputs, manifest, renderer)
    state = read(result.manifest)
    assert len(renderer.calls) == len(result.clips) == 1
    assert read(result.summary)["status"] == "partial"
    assert [e["status"] for e in state["clips"]] == [
        "rendered_unreviewed", "review_required", "pending"]
    assert state["clips"][1]["reason"] == "needs evidence"
    assert state["clips"][2]["reason"] == "awaiting visual evidence"


def test_partial_failure_retains_output_and_resume_retries_only_failure(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch)
    renderer = Renderer()
    diagnostic: list[Path] = []

    class NativeFailure(RuntimeError):
        pass

    def flaky(source: Path, plan: Path, output: Path, **kwargs: Any) -> list[Path]:
        if plan.parent.name == "director-1" and not diagnostic:
            output.mkdir(parents=True, exist_ok=True)
            path = output / "diagnostic.json"
            path.write_text('{}')
            diagnostic.append(path)
            error = NativeFailure("SECRET must not enter export state")
            error.diagnostic_path = path
            raise error
        return renderer(source, plan, output, **kwargs)

    first = export(inputs, manifest, flaky)
    state = read(first.manifest)
    assert len(first.clips) == 1 and first.clips[0].is_file()
    assert state["clips"][1]["status"] == "render_failed"
    attempt = state["clips"][1]["attempts"][0]
    assert attempt["error_type"] == "NativeFailure"
    assert attempt["diagnostic_path"] == str(diagnostic[0])
    assert "SECRET" not in first.manifest.read_text()
    assert read(first.summary)["status"] == "partial"
    second = export(inputs, manifest, flaky)
    assert first.manifest == second.manifest
    assert len(second.clips) == 2 and second.clips[0] == first.clips[0]
    assert len(renderer.calls) == 2
    assert len(read(second.manifest)["clips"][1]["attempts"]) == 2
    assert read(second.summary)["status"] == "rendered_unreviewed"


@pytest.mark.parametrize("mutation", ["bytes", "missing"])
def test_changed_output_is_never_reused_and_attempts_are_preserved(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch)
    renderer = Renderer()
    first = export(inputs, manifest, renderer)
    old = first.clips[0]
    if mutation == "bytes":
        old.write_bytes(b"tampered")
    else:
        old.unlink()
    second = export(inputs, manifest, renderer)
    assert len(renderer.calls) == 3
    assert second.clips[0] != old and second.clips[1] == first.clips[1]
    assert len(read(second.manifest)["clips"][0]["attempts"]) == 2
    if mutation == "bytes":
        assert old.read_bytes() == b"tampered"


@pytest.mark.parametrize("changed", ["profile", "transcript", "asset"])
def test_input_identity_change_creates_a_new_run(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch)
    renderer = Renderer()
    asset = inputs.root / "registered.asset"
    asset.write_bytes(b"v1")
    kwargs = dict(transcript_path=inputs.transcript,
                  assets=RenderAssetRegistry({"brand": asset}, {}, {}))
    first = export(inputs, manifest, renderer, **kwargs)
    if changed == "profile":
        kwargs["profile"] = RenderProfile(video_encoder="libx264")
    elif changed == "transcript":
        inputs.transcript.write_text('[]')
    else:
        asset.write_bytes(b"v2")
    second = export(inputs, manifest, renderer, **kwargs)
    assert len(renderer.calls) == 4 and first.manifest != second.manifest
    assert all(p.is_file() for p in (*first.clips, *second.clips))


def test_valid_plan_change_creates_new_run(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch)
    renderer = Renderer()
    first = export(inputs, manifest, renderer)
    value = read(manifest)
    entry = value["clips"][0]
    plan = Path(entry["visual_plan"])
    payload = read(plan)
    payload["shots"][0]["reason"] = "new validated reason"
    visual_director._write(plan, payload)
    provenance = plan.parent / "provenance.json"
    companion = read(provenance)
    companion["plan"] = dict(visual_director._file_record(plan),
                             fingerprint=visual_director._hash(payload))
    visual_director._write(provenance, companion)
    receipt_path = plan.parent / "receipt.json"
    receipt = read(receipt_path)
    receipt.update(plan_sha256=visual_director.file_sha256(plan),
                   companion_sha256=visual_director.file_sha256(provenance))
    visual_director._write(receipt_path, receipt)
    entry["provenance_sha256"] = receipt["companion_sha256"]
    manifest.write_text(json.dumps(value))
    second = export(inputs, manifest, renderer)
    assert first.manifest != second.manifest and len(renderer.calls) == 4
    assert all(p.is_file() for p in first.clips)


@pytest.mark.parametrize("restore_stat", [False, True])
def test_source_changes_cannot_reuse_stale_exports(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch, restore_stat: bool,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch)
    renderer = Renderer()
    first = export(inputs, manifest, renderer)
    metadata = inputs.video.stat()
    inputs.video.write_bytes(b"x" * metadata.st_size)
    if restore_stat:
        os.utime(inputs.video, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        second = export(inputs, manifest, renderer)
        assert second.manifest != first.manifest and len(renderer.calls) == 4
    else:
        with pytest.raises(ValueError):
            export(inputs, manifest, renderer)
        assert len(renderer.calls) == 2
    assert all(p.is_file() for p in first.clips)


@pytest.mark.parametrize("defect", [
    "missing_plan", "plan_bytes", "forged_receipt", "receipt_path", "provenance_path",
    "provenance_hash", "host_context", "clip_id", "source_fingerprint",
    "composition_fingerprint", "source_timeline_sha256", "style_library_fingerprint",
    "review_as_ready", "duplicate_clip",
])
def test_invalid_ready_associations_reject_before_any_render(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch, defect: str,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch,
                                 ("ready", "review_required" if defect == "review_as_ready" else "ready"))
    value = read(manifest)
    entry = value["clips"][1]
    if defect == "missing_plan":
        Path(entry["visual_plan"]).unlink()
    elif defect == "plan_bytes":
        Path(entry["visual_plan"]).write_text('{}')
    elif defect == "forged_receipt":
        Path(entry["receipt"]).write_text('{"status":"ready"}')
    elif defect in {"receipt_path", "provenance_path"}:
        entry["receipt" if defect == "receipt_path" else "provenance"] = value["clips"][0][
            "receipt" if defect == "receipt_path" else "provenance"]
    elif defect == "provenance_hash":
        entry["provenance_sha256"] = "0" * 64
    elif defect == "host_context":
        entry["host_context"] = {}
    elif defect == "clip_id":
        entry["clip_id"] = "another"
    elif defect == "review_as_ready":
        entry.update(status="ready", visual_plan=entry["receipt"])
    elif defect == "duplicate_clip":
        value["clips"].append(value["clips"][0])
    else:
        value[defect] = "0" * 64
    manifest.write_text(json.dumps(value))
    renderer = Renderer()
    with pytest.raises((ValueError, OSError, visual_director.DirectorPublicationError)):
        export(inputs, manifest, renderer)
    assert renderer.calls == []


def test_renderer_cannot_return_a_path_outside_its_attempt(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    outside = inputs.root / "do-not-touch.mp4"
    outside.write_bytes(b"preserved")
    result = export(inputs, manifest, lambda *a, **k: [outside])
    assert result.clips == () and read(result.summary)["status"] == "error"
    assert read(result.manifest)["clips"][0]["status"] == "render_failed"
    assert outside.read_bytes() == b"preserved"


def test_duplicate_invocations_serialize_and_reuse_identical_outputs(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    manifest = planning_manifest(inputs, monkeypatch)
    renderer = Renderer()
    started, release = threading.Event(), threading.Event()

    def slow(*args: Any, **kwargs: Any) -> list[Path]:
        started.set()
        assert release.wait(10)
        return renderer(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(export, inputs, manifest, slow)
        assert started.wait(10)
        second = pool.submit(export, inputs, manifest, slow)
        release.set()
        a, b = first.result(timeout=20), second.result(timeout=20)
    assert a == b and len(renderer.calls) == 2
    assert len({call[2] for call in renderer.calls}) == 2


def test_cancellation_persists_progress_then_propagates_and_can_resume(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready", "ready", "ready"))
    renderer = Renderer()

    def cancelled(source: Path, plan: Path, output: Path, **kwargs: Any) -> list[Path]:
        if plan.parent.name == "director-1":
            raise KeyboardInterrupt("cancel")
        return renderer(source, plan, output, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        export(inputs, manifest, cancelled)
    saved = next((inputs.root / "exports").glob("export_runs/*/export_manifest.json"))
    state = read(saved)
    assert [e["status"] for e in state["clips"]] == [
        "rendered_unreviewed", "render_failed", "pending"]
    assert state["clips"][1]["attempts"][0]["error_type"] == "KeyboardInterrupt"
    result = export(inputs, manifest, renderer)
    assert result.manifest == saved and len(result.clips) == len(renderer.calls) == 3


@pytest.mark.parametrize("statuses, expected", [
    ((), "empty"), (("review_required",), "review_required"), (("pending",), "review_required"),
])
def test_no_ready_clip_never_calls_renderer(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch, statuses: tuple[str, ...], expected: str,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch, statuses)
    renderer = Renderer()
    result = export(inputs, manifest, renderer)
    assert result.clips == () and renderer.calls == []
    assert read(result.summary)["status"] == expected


def test_oversized_json_rejected_before_render(inputs: Inputs) -> None:
    manifest = inputs.root / "too-large.json"
    manifest.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    renderer = Renderer()
    with pytest.raises(ValueError):
        export(inputs, manifest, renderer)
    assert renderer.calls == []


def test_later_renderer_cannot_cause_tampered_earlier_output_to_be_returned(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch)
    renderer = Renderer()
    first_output: list[Path] = []

    def interfering(*args: Any, **kwargs: Any) -> list[Path]:
        paths = renderer(*args, **kwargs)
        if first_output:
            first_output[0].write_bytes(b"changed during later export")
        else:
            first_output.extend(paths)
        return paths

    result = export(inputs, manifest, interfering)
    assert first_output[0] not in result.clips
    assert len(result.clips) == 1
    assert read(result.manifest)["clips"][0]["status"] == "pending"
    assert read(result.summary)["status"] == "partial"


def test_middle_failure_continues_independent_ready_clips(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready", "ready", "ready"))
    renderer = Renderer()

    def middle_failure(source: Path, plan: Path, output: Path, **kwargs: Any) -> list[Path]:
        if plan.parent.name == "director-1":
            raise RuntimeError("external failure")
        return renderer(source, plan, output, **kwargs)

    result = export(inputs, manifest, middle_failure)
    assert len(result.clips) == len(renderer.calls) == 2
    assert [e["status"] for e in read(result.manifest)["clips"]] == [
        "rendered_unreviewed", "render_failed", "rendered_unreviewed"]


def test_default_renderer_is_the_visual_renderer(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clipperx import export_pipeline
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    renderer = Renderer()
    monkeypatch.setattr(export_pipeline, "render_visual_composition_plan", renderer)
    result = export_pipeline.export_visual_manifest(inputs.video, manifest, inputs.root / "exports",
        assets=RenderAssetRegistry({}, {}, {}))
    assert len(result.clips) == len(renderer.calls) == 1


def test_model_ids_with_same_slug_receive_distinct_contained_host_names(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both normalize to the same old safe_slug; neither may become a generated path.
    manifest = planning_manifest(inputs, monkeypatch,
        clip_ids=("../../unsafe/clip", "..\\..\\unsafe\\clip"))
    value = read(manifest)
    assert "../" in value["clips"][0]["clip_id"]
    result = export(inputs, manifest, Renderer())
    state = read(result.manifest)
    assert len({e["host_id"] for e in state["clips"]}) == 2
    for entry, output in zip(state["clips"], result.clips):
        assert entry["clip_id"] not in entry["host_id"]
        assert output.is_relative_to(result.manifest.parent / entry["host_id"])


@pytest.mark.parametrize("reference", ["visual_plan", "receipt", "provenance"])
def test_missing_ready_reference_is_a_validation_error(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch, reference: str,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    value = read(manifest)
    value["clips"][0].pop(reference)
    manifest.write_text(json.dumps(value))
    renderer = Renderer()
    with pytest.raises(ValueError):
        export(inputs, manifest, renderer)
    assert renderer.calls == []


def _rebind_terminal(manifest: Path, *, transcript: Path | None = None,
                     fixed_transcript: bool = False, subtitles: bool = False,
                     sfx: dict[str, Any] | None = None, creator: bool = False,
                     only: int | None = None) -> None:
    data = read(manifest)
    if fixed_transcript and transcript is not None:
        data.update(transcript_path=str(transcript.resolve()),
                    transcript_sha256=visual_director.file_sha256(transcript))
    for index, entry in enumerate(data["clips"]):
        if only is not None and index != only:
            continue
        root = Path(entry["receipt"]).parent
        authority = read(root / "authority.json")
        if transcript is not None:
            authority["host_context"]["text"] = {
                "status": "unverified_legacy", **visual_director._file_record(transcript)}
        if sfx is not None:
            authority["host_context"]["sfx"] = sfx
        if creator:
            from clipperx.visual_prompts import creator_overlay_requirements
            authority["limits"]["require_creator_overlays"] = True
            authority["host_context"]["creator_overlays"] = creator_overlay_requirements(True)
        visual_director._write(root / "authority.json", authority)
        plan = read(root / "visual_plan.json")
        if subtitles:
            plan["shots"][0]["subtitles"] = {
                "destination": {"x": 0.1, "y": 0.7, "width": 0.8, "height": 0.2}}
        if creator:
            plan["shots"][0]["brand"] = {"asset_id": "kick_brand", "start": 0, "end": 2,
                "destination": {"x": .35, "y": .25, "width": .3, "height": .05}, "z_index": 4}
            plan["shots"][0]["title"] = {"text": "Hook", "font_asset_id": "headline_impact",
                "start": 0, "end": 1, "destination": {"x": .1, "y": .1, "width": .8, "height": .1},
                "z_index": 3}
        if sfx is not None:
            plan["shots"][0]["audio"] = {"cues": [{"asset_id": "whoosh_01",
                "start": 0.1, "end": 0.63, "gain_db": -18, "duck_source_db": 0}]}
        visual_director._write(root / "visual_plan.json", plan)
        provenance = read(root / "provenance.json")
        provenance["host_context"] = authority["host_context"]
        provenance["authority"] = visual_director._file_record(root / "authority.json")
        provenance["plan"] = dict(visual_director._file_record(root / "visual_plan.json"),
                                   fingerprint=visual_director._hash(plan))
        visual_director._write(root / "provenance.json", provenance)
        receipt = read(root / "receipt.json")
        receipt.update(host_context=authority["host_context"],
            plan_sha256=visual_director.file_sha256(root / "visual_plan.json"),
            companion_sha256=visual_director.file_sha256(root / "provenance.json"))
        visual_director._write(root / "receipt.json", receipt)
        entry.update(host_context=authority["host_context"],
                     provenance_sha256=receipt["companion_sha256"])
        visual_director._load_terminal(root)
    manifest.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("choice", ["none", "other", "wrong_manifest_hash"])
def test_fixed_ready_transcript_rejects_mismatch_before_render(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch, choice: str,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    _rebind_terminal(manifest, transcript=inputs.transcript, fixed_transcript=True)
    other = inputs.root / "other.json"
    other.write_text('[{"start":0,"end":2,"text":"unrelated","words":[]}]')
    selected = {"none": None, "other": other, "wrong_manifest_hash": inputs.transcript}[choice]
    if choice == "wrong_manifest_hash":
        data = read(manifest); data["transcript_sha256"] = "0" * 64
        manifest.write_text(json.dumps(data))
    renderer = Renderer()
    with pytest.raises(ValueError):
        export(inputs, manifest, renderer, transcript_path=selected)
    assert not renderer.calls


def test_fixed_ready_transcript_accepts_matching_file(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    _rebind_terminal(manifest, transcript=inputs.transcript, fixed_transcript=True)
    renderer = Renderer()
    result = export(inputs, manifest, renderer, transcript_path=inputs.transcript)
    assert len(result.clips) == 1 and renderer.calls[0][3]["transcript_path"] == inputs.transcript.resolve()


def test_fixed_transcript_requires_every_ready_match_before_any_render(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready", "ready"))
    _rebind_terminal(manifest, transcript=inputs.transcript, fixed_transcript=True)
    other = inputs.root / "other.json"
    other.write_text('[]')
    _rebind_terminal(manifest, transcript=other, only=1)
    renderer = Renderer()
    with pytest.raises(ValueError):
        export(inputs, manifest, renderer, transcript_path=inputs.transcript)
    assert not renderer.calls


@pytest.mark.parametrize("choice", ["none", "other"])
def test_legacy_subtitles_require_captured_transcript(inputs: Inputs, monkeypatch: pytest.MonkeyPatch, choice: str) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    _rebind_terminal(manifest, transcript=inputs.transcript, subtitles=True)
    other = inputs.root / "other.json"
    other.write_text('[]')
    renderer = Renderer()
    with pytest.raises(ValueError):
        export(inputs, manifest, renderer, transcript_path=other if choice == "other" else None)
    assert not renderer.calls


def test_legacy_subtitles_accept_captured_transcript(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    _rebind_terminal(manifest, transcript=inputs.transcript, subtitles=True)
    renderer = Renderer()
    assert len(export(inputs, manifest, renderer, transcript_path=inputs.transcript).clips) == 1


def test_creator_silence_still_requires_captured_transcript(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs.transcript.write_text('[]')
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    _rebind_terminal(manifest, transcript=inputs.transcript, creator=True)
    renderer = Renderer()
    with pytest.raises(ValueError):
        export(inputs, manifest, renderer)
    assert not renderer.calls
    assert len(export(inputs, manifest, renderer, transcript_path=inputs.transcript).clips) == 1


def _sfx_case(inputs: Inputs, manifest: Path) -> tuple[Path, Path]:
    import hashlib
    import wave
    from clipperx.sfx_authority import capture_sfx_authority
    root = inputs.root / "sfx"; root.mkdir()
    approved, alternate = root / "approved.wav", root / "other.wav"
    for path, sample in ((approved, b"\0\1"), (alternate, b"\0\2")):
        with wave.open(str(path), "wb") as out:
            out.setparams((1, 2, 48000, 25440, "NONE", "not compressed"))
            out.writeframes(sample * 25440)
    catalog = root / "catalog.json"
    catalog.write_text(json.dumps({"version": 1, "assets": [{"asset_id": "whoosh_01",
        "family": "whoosh", "path": approved.name, "sha256": hashlib.sha256(approved.read_bytes()).hexdigest(),
        "sample_rate": 48000, "channels": 1, "frames": 25440, "transient_sample": 7896,
        "default_gain_db": -18, "approval": {"status": "approved", "reference": "review"}, "provenance": {}}]}))
    _rebind_terminal(manifest, sfx=capture_sfx_authority(catalog))
    return approved, alternate


def test_ready_sfx_rejects_different_bytes_before_render(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    approved, alternate = _sfx_case(inputs, manifest)
    renderer = Renderer()
    with pytest.raises(ValueError):
        export(inputs, manifest, renderer, assets=RenderAssetRegistry({}, {}, {"whoosh_01": alternate}))
    assert not renderer.calls and approved.read_bytes() != alternate.read_bytes()


def test_ready_sfx_accepts_identical_copy_and_detects_drift(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    approved, alternate = _sfx_case(inputs, manifest)
    alternate.write_bytes(approved.read_bytes())
    renderer = Renderer()
    registry = RenderAssetRegistry({}, {}, {"whoosh_01": alternate})
    first = export(inputs, manifest, renderer, assets=registry)
    assert len(first.clips) == 1
    alternate.write_bytes(b"changed")
    with pytest.raises(ValueError):
        export(inputs, manifest, renderer, assets=registry)
    assert len(renderer.calls) == 1 and first.clips[0].is_file()


def test_sfx_change_during_render_rejects_and_preserves_attempt(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = planning_manifest(inputs, monkeypatch, ("ready",))
    approved, alternate = _sfx_case(inputs, manifest)
    alternate.write_bytes(approved.read_bytes())
    renderer = Renderer()
    def mutate(*args: Any, **kwargs: Any) -> list[Path]:
        outputs = renderer(*args, **kwargs)
        alternate.write_bytes(b"changed during render")
        return outputs
    with pytest.raises(ValueError):
        export(inputs, manifest, mutate, assets=RenderAssetRegistry({}, {}, {"whoosh_01": alternate}))
    saved = next((inputs.root / "exports").glob("export_runs/*/export_manifest.json"))
    state = read(saved)
    assert state["clips"][0]["status"] == "render_failed"
    assert renderer.calls[0][2].is_dir()


@pytest.mark.parametrize("statuses, expected", [(("pending",), "error"), (("ready", "pending"), "partial")])
def test_upstream_error_has_own_state_and_summary(inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
                                                  statuses: tuple[str, ...], expected: str) -> None:
    manifest = planning_manifest(inputs, monkeypatch, statuses)
    data = read(manifest); data["clips"][-1].update(status="error", reason="planning crashed")
    manifest.write_text(json.dumps(data))
    result = export(inputs, manifest, Renderer())
    state = read(result.manifest)
    assert state["clips"][-1]["status"] == "error"
    assert state["clips"][-1]["reason"] == "planning crashed"
    assert read(result.summary)["status"] == expected
    assert read(result.summary)["error_clips"] == 1
