"""Default public CLI cannot bypass the approved director SFX authority."""
import importlib
import importlib.util
import json

import pytest
from typer.testing import CliRunner

from clipperx.tests.test_sfx_authority import _plan, _mono_host, _ready_with_cue
from clipperx.tests.test_visual_director import FakeClient, run

pytest_plugins = ["clipperx.tests.test_sfx_authority"]


def _gate():
    assert importlib.util.find_spec("clipperx.render_sfx_gate") is not None
    return importlib.import_module("clipperx.render_sfx_gate").validate_default_sfx_render_plan


def test_default_cli_rejects_unreceipted_sfx_before_render(tmp_path, monkeypatch):
    from clipperx import cli, plan_render
    source = tmp_path / "source.mp4"
    source.write_bytes(b"Not decoded; admission must stop first")
    plan = tmp_path / "manual.json"
    manual = _plan()
    manual.shots[0].audio.cues[0].asset_id = "whoosh_sa3_softgate_01"
    manual.shots[0].audio.cues[0].gain_db = 0
    manual.shots[0].audio.cues[0].duck_source_db = -6
    plan.write_text(manual.model_dump_json(), encoding="utf-8")
    calls = []
    monkeypatch.setattr(plan_render, "render_visual_composition_plan",
        lambda *args, **kwargs: calls.append(args) or [tmp_path / "out.mp4"])
    result = CliRunner().invoke(cli.app, ["render-visual-plan", str(source), str(plan),
        "--output-dir", str(tmp_path / "exports")])
    assert result.exit_code != 0 and calls == []
    assert "READY" in result.output
    assert not (tmp_path / "exports").exists()


def test_no_cue_plan_keeps_manual_host_route(tmp_path):
    plan = _plan()
    plan.shots[0].audio.cues = []
    path = tmp_path / "manual.json"
    path.write_text(plan.model_dump_json(), encoding="utf-8")
    _gate()(tmp_path / "not-read-source.mp4", path)


def test_current_ready_authority_allows_only_its_exact_plan_and_source(catalog, scene, tmp_path, monkeypatch):
    _mono_host(catalog, monkeypatch)
    result = run(scene, tmp_path, FakeClient([_ready_with_cue]))
    assert result.name == "visual_plan.json"
    gate = _gate()
    gate(scene[0], result)
    other = tmp_path / "different-source.mp4"
    other.write_bytes(b"Not the bound source")
    with pytest.raises(ValueError):
        gate(other, result)
    copied = tmp_path / "copied.json"
    copied.write_bytes(result.read_bytes())
    with pytest.raises(ValueError):
        gate(scene[0], copied)


@pytest.mark.parametrize("change", ["revoked", "louder"])
def test_existing_ready_cannot_bypass_revocation_or_edited_plan(catalog, scene, tmp_path, monkeypatch, change):
    _, catalog_path, _, document = _mono_host(catalog, monkeypatch)
    result = run(scene, tmp_path, FakeClient([_ready_with_cue]))
    assert result.name == "visual_plan.json"
    if change == "revoked":
        document["assets"][0]["approval"] = {"status": "pending", "reference": ""}
        catalog_path.write_text(json.dumps(document), encoding="utf-8")
    else:
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["shots"][0]["audio"]["cues"][0]["gain_db"] = 0
        result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        _gate()(scene[0], result)


def test_gate_rejects_implicit_version_before_any_render(tmp_path):
    plan = _plan().model_dump(mode="json")
    plan.pop("contract_version")
    path = tmp_path / "implicit.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        _gate()(tmp_path / "source.mp4", path)
