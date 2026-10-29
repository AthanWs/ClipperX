"""Approved SFX identity, conservative cues and captured director authority."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import wave

import pytest

from clipperx.tests.test_visual_director import FakeClient, ready, run, scene as director_scene
from clipperx.visual_contracts import VisualCompositionPlan, VisualDirectorResponse

scene = director_scene


def _module():
    assert importlib.util.find_spec("clipperx.sfx_authority") is not None, "SFX authority API missing"
    return importlib.import_module("clipperx.sfx_authority")


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    root = tmp_path / "host-sfx"
    root.mkdir()
    audio = root / "whoosh.wav"
    with wave.open(str(audio), "wb") as output:
        output.setparams((2, 2, 48000, 25440, "NONE", "not compressed"))
        output.writeframes(b"\0\1" * 25440 * 2)
    entry = dict(asset_id="whoosh_01", family="whoosh", path=audio.name,
                 sha256=hashlib.sha256(audio.read_bytes()).hexdigest(), sample_rate=48000,
                 channels=2, frames=25440, transient_sample=7896, default_gain_db=-18,
                 approval=dict(status="approved", reference="host-review-001"), provenance={})
    path = root / "catalog.json"
    document = dict(version=1, assets=[entry])
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, audio, document


def _host(catalog, monkeypatch):
    path, audio, document = catalog
    module = _module()
    monkeypatch.setattr(module, "default_sfx_catalog_path", lambda: path)
    return module, path, audio, document


def _mono_host(catalog, monkeypatch):
    module, path, audio, document = _host(catalog, monkeypatch)
    with wave.open(str(audio), "wb") as output:
        output.setparams((1, 2, 48000, 25440, "NONE", "not compressed"))
        output.writeframes(b"\0\1" * 25440)
    document["assets"][0].update(channels=1, sha256=hashlib.sha256(audio.read_bytes()).hexdigest())
    path.write_text(json.dumps(document), encoding="utf-8")
    return module, path, audio, document


def _plan(starts=(0.1,)):
    from clipperx.tests.test_visual_contracts import make_shot
    shots = [make_shot(shot_id="host", source={"start": 0, "end": 12},
                       output={"start": 0, "end": 12}, reason="The source scene transition lands on its payoff",
                       evidence_ids=["source-event"], audio=dict(cues=[dict(
                           asset_id="whoosh_01", start=start, end=start + .53,
                           gain_db=-18, duck_source_db=0) for start in starts]))]
    return VisualCompositionPlan.model_validate(dict(contract_version=2, clip_id="clip",
        source_fingerprint="s", composition_fingerprint="c", source_timeline_fingerprint="t",
        identity_fingerprint="i", style_library_fingerprint="l", shots=shots))


def test_capture_offers_same_approved_set_as_default_registry(catalog, monkeypatch):
    module, path, audio, document = _host(catalog, monkeypatch)
    captured = module.capture_sfx_authority()
    assert captured["version"] == 1 and captured["catalog"]["path"] == str(path.resolve())
    assert captured["catalog"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [row["asset_id"] for row in module.sfx_offers(captured, source_channels=2)] == ["whoosh_01"]
    from clipperx.render_assets import default_render_assets
    assert default_render_assets().sfx == {"whoosh_01": audio.resolve()}
    document["assets"][0]["approval"] = dict(status="pending", reference="")
    path.write_text(json.dumps(document), encoding="utf-8")
    assert module.sfx_offers(module.capture_sfx_authority(), source_channels=2) == []
    assert default_render_assets().sfx == {}


@pytest.mark.parametrize("change", ["revoked", "bytes", "removed", "metadata"])
def test_dependencies_detect_catalog_changes_and_revocation(catalog, change, monkeypatch):
    module, path, audio, document = _host(catalog, monkeypatch)
    captured = module.capture_sfx_authority()
    if change == "revoked":
        document["assets"][0]["approval"] = dict(status="pending", reference="")
        path.write_text(json.dumps(document), encoding="utf-8")
    elif change == "bytes":
        audio.write_bytes(audio.read_bytes()[:-2] + b"\0\0")
    elif change == "removed":
        path.unlink()
    else:
        path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        module.validate_sfx_authority(captured)
    module.validate_sfx_authority(captured, check_files=False)


def test_absent_catalog_is_no_sfx_and_invalid_existing_catalog_is_not_silently_enabled(tmp_path, monkeypatch):
    module = _module()
    path = tmp_path / "missing.json"
    monkeypatch.setattr(module, "default_sfx_catalog_path", lambda: path)
    assert module.capture_sfx_authority() is None
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        module.capture_sfx_authority()


def test_valid_exact_cues_allow_two_separated_by_five_seconds(catalog, monkeypatch):
    module, *_ = _host(catalog, monkeypatch)
    module.validate_sfx_cues(_plan((.1, 5.1)), module.capture_sfx_authority(), source_channels=2)


@pytest.mark.parametrize(("stored_default", "effective"), [(-6, -18), (-24, -24)])
def test_soft_policy_caps_louder_defaults_and_retains_quieter_defaults(catalog, monkeypatch, stored_default, effective):
    module, path, _, document = _host(catalog, monkeypatch)
    document["assets"][0]["default_gain_db"] = stored_default
    path.write_text(json.dumps(document), encoding="utf-8")
    captured = module.capture_sfx_authority()
    assert captured["assets"][0]["default_gain_db"] == stored_default
    assert module.sfx_offers(captured, source_channels=2)[0]["default_gain_db"] == effective
    plan = _plan()
    plan.shots[0].audio.cues[0].gain_db = effective
    module.validate_sfx_cues(plan, captured, source_channels=2)
    plan.shots[0].audio.cues[0].gain_db = effective + 1
    with pytest.raises(ValueError):
        module.validate_sfx_cues(plan, captured, source_channels=2)


@pytest.mark.parametrize("defect", ["legacy", "unknown", "loud", "duck", "short", "long", "close", "count", "sound_only"])
def test_conservative_cue_guards_fail_closed(catalog, defect, monkeypatch):
    module, *_ = _host(catalog, monkeypatch)
    captured = module.capture_sfx_authority()
    plan = _plan((.1, 5.1, 10.1) if defect == "count" else (.1, 4.9) if defect == "close" else (.1,))
    cue = plan.shots[0].audio.cues[0]
    if defect == "legacy":
        captured = None
    elif defect == "unknown":
        cue.asset_id = "pending_or_unknown"
    elif defect == "loud":
        cue.gain_db = -17
    elif defect == "duck":
        cue.duck_source_db = -1
    elif defect in ("short", "long"):
        cue.end += -.01 if defect == "short" else .01
    elif defect == "sound_only":
        plan.shots[0].reason = "whoosh SFX"
    with pytest.raises(ValueError):
        module.validate_sfx_cues(plan, captured, source_channels=2)


def _ready_with_cue(payload):
    response = ready(payload)
    response["plan"]["shots"][0]["reason"] = "The source scene transition lands on the payoff"
    response["plan"]["shots"][0]["audio"] = dict(cues=[dict(
        asset_id="whoosh_01", start=.1, end=.63, gain_db=-18, duck_source_db=0)])
    return response


def test_live_authority_receipt_and_provenance_preserve_optional_sfx(catalog, scene, tmp_path, monkeypatch):
    module, *_ = _mono_host(catalog, monkeypatch)
    client = FakeClient([_ready_with_cue])
    result = run(scene, tmp_path, client)
    assert result.name == "visual_plan.json"
    receipt = json.loads((result.parent / "receipt.json").read_text())
    companion = json.loads((result.parent / "provenance.json").read_text())
    authority = json.loads((result.parent / "authority.json").read_text())
    assert receipt["host_context"]["sfx"] == companion["host_context"]["sfx"] == authority["host_context"]["sfx"]
    offered = client.payloads[0]["clip"]["host_context"]["authorized_sfx"]
    assert [row["asset_id"] for row in offered] == ["whoosh_01"]
    module.validate_sfx_authority(authority["host_context"]["sfx"])


def test_dependency_drift_after_client_becomes_review_without_cue_publication(catalog, scene, tmp_path, monkeypatch):
    _, path, _, document = _mono_host(catalog, monkeypatch)
    def revoke(payload):
        response = _ready_with_cue(payload)
        document["assets"][0]["approval"] = dict(status="pending", reference="")
        path.write_text(json.dumps(document), encoding="utf-8")
        return response
    result = run(scene, tmp_path, FakeClient([revoke]))
    assert result.name == "receipt.json"
    assert json.loads(result.read_text())["status"] == "review_required"
    assert not (result.parent / "visual_plan.json").exists()


def test_recovery_rechecks_approved_catalog_revocation(catalog, scene, tmp_path, monkeypatch):
    _, path, _, document = _mono_host(catalog, monkeypatch)
    result = run(scene, tmp_path, FakeClient([_ready_with_cue]))
    assert result.name == "visual_plan.json"
    from clipperx.visual_director import DirectorPublicationError, recover_visual_director_publication
    document["assets"][0]["approval"] = dict(status="pending", reference="")
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DirectorPublicationError):
        recover_visual_director_publication(scene[0], result.parent, jobs_root=tmp_path / "jobs")


def test_legacy_final_context_does_not_gain_new_default_sfx_permission(catalog, scene, monkeypatch):
    _host(catalog, monkeypatch)
    from clipperx.visual_director import _fingerprints, _required_shots, _validate_final
    _, clip, timeline, library = scene
    fingerprints = _fingerprints(clip, timeline, library.fingerprint)
    payload = dict(clip=dict(clip_id=clip.clip_id, host_context=dict(
        fingerprints=fingerprints, required_shots=_required_shots(clip, timeline))))
    response = VisualDirectorResponse.model_validate(_ready_with_cue(payload))
    with pytest.raises(ValueError):
        _validate_final(response, clip, timeline, fingerprints, [], [], 192, 108)


@pytest.mark.parametrize("channels", [None, 1, 6])
def test_stereo_asset_is_not_offered_or_authorized_for_incompatible_source(catalog, monkeypatch, channels):
    module, *_ = _host(catalog, monkeypatch)
    captured = module.capture_sfx_authority()
    assert module.sfx_offers(captured, source_channels=channels) == []
    with pytest.raises(ValueError):
        module.validate_sfx_cues(_plan(), captured, source_channels=channels)


def test_mono_live_source_keeps_no_cues_valid_and_does_not_offer_stereo(catalog, scene, tmp_path, monkeypatch):
    _host(catalog, monkeypatch)
    client = FakeClient([ready])
    result = run(scene, tmp_path, client)
    assert result.name == "visual_plan.json"
    assert client.payloads[0]["clip"]["host_context"]["authorized_sfx"] == []
    assert all(not shot["audio"]["cues"] for shot in json.loads(result.read_text())["shots"])


def test_first_audio_stream_controls_eligibility():
    module = _module()
    assert module.source_audio_channels({"streams": []}) is None
    assert module.source_audio_channels({"streams": [{"kind": "video"},
        {"kind": "audio", "channels": 1}, {"kind": "audio", "channels": 2}]}) == 1


def test_legacy_ready_recovery_keeps_no_sfx_and_rejects_forged_cues(catalog, scene, tmp_path, monkeypatch):
    import clipperx.visual_director as director
    from clipperx.tests.test_visual_director import reseal_terminal
    module, *_ = _host(catalog, monkeypatch)
    monkeypatch.setattr(module, "default_sfx_catalog_path", lambda: tmp_path / "absent.json")
    result = run(scene, tmp_path, FakeClient([ready]))
    assert result.name == "visual_plan.json"
    root = result.parent
    authority = json.loads((root / "authority.json").read_text())
    assert "sfx" not in authority["host_context"]
    assert director.recover_visual_director_publication(scene[0], root, jobs_root=tmp_path / "jobs") == result
    plan = json.loads(result.read_text())
    plan["shots"][0]["audio"]["cues"] = [dict(asset_id="whoosh_01", start=.1, end=.63, gain_db=-18, duck_source_db=0)]
    director._write(result, plan)
    companion = json.loads((root / "provenance.json").read_text())
    receipt = json.loads((root / "receipt.json").read_text())
    companion["plan"] = {**director._file_record(result), "fingerprint": director._hash(plan)}
    receipt["plan_sha256"] = companion["plan"]["sha256"]
    reseal_terminal(root, receipt, companion)
    with pytest.raises(director.DirectorPublicationError):
        director.recover_visual_director_publication(scene[0], root, jobs_root=tmp_path / "jobs")


def test_product_catalog_has_only_exact_human_approved_b():
    from clipperx.sfx_catalog import load_sfx_catalog
    root = Path(__file__).resolve().parents[2]
    catalog = load_sfx_catalog(root / "assets/sfx/catalog.json")
    assert set(catalog.approved_asset_paths()) == {"whoosh_sa3_softgate_01"}
    asset = catalog.assets[0]
    assert asset.sha256 == "595ee7906f6a25822a7fb2267201dfd16c887327db58fd6043b66dc76f112097"
    assert asset.approval.reference == "user_reply:call_0431f6ef8b7143c1a90a682015ca3b72:0"
    assert asset.provenance["approval_quote"] == "Sí, usarla suavemente"
    pending = load_sfx_catalog(root / "data/sfx_candidates/sa3_cpu_probe_01/pending-catalog-v1.json")
    assert pending.approved_asset_paths() == {}
