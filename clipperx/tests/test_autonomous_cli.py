"""Explicit CLI admission and truthful automatic-flow handoff."""
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from clipperx import cli
from clipperx import autonomous_pipeline as workflow

runner = CliRunner()


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"Not decoded at this CLI boundary")
    return source, tmp_path / "exports"


def test_auto_edit_confirmation_precedes_client_and_processing(inputs, monkeypatch):
    source, output = inputs
    monkeypatch.setattr(cli, "_automatic_client", lambda *a, **k: pytest.fail("credential/client boundary reached"), raising=False)
    monkeypatch.setattr(workflow, "run_autonomous_pipeline", lambda *a, **k: pytest.fail("processing started"))
    result = runner.invoke(cli.app, ["auto-edit", str(source), "--output-dir", str(output)])
    assert result.exit_code != 0
    assert "confirm-inference" in result.output
    assert not output.exists()


@pytest.mark.parametrize("status", ["rendered_unreviewed", "partial", "budget_exhausted"])
def test_auto_edit_runs_coverage_creator_mode_and_reports_real_state(inputs, monkeypatch, status):
    source, output = inputs
    observed = {}
    client = object()
    def confirmed(model, context, confirmation):
        assert confirmation is True
        observed.update(model=model, context=context)
        return client
    def run(video, destination, **kwargs):
        observed.update(video=video, destination=destination, **kwargs)
        destination.mkdir()
        summary = destination / "summary.json"
        summary.write_text(json.dumps(dict(status=status, full_qa_passed=False)), encoding="utf-8")
        return SimpleNamespace(summary_path=summary, state_path=destination / "state.json", clips=())
    monkeypatch.setattr(cli, "_automatic_client", confirmed, raising=False)
    monkeypatch.setattr(workflow, "run_autonomous_pipeline", run)
    result = runner.invoke(cli.app, ["auto-edit", str(source), "--output-dir", str(output),
        "--confirm-inference", "--encoder", "libx264"])
    assert result.exit_code == (0 if status == "rendered_unreviewed" else 2), result.output
    assert observed["client"] is client and observed["model"] == "gpt-6-luna(low)"
    assert observed["editorial_limits"].target_clips is None
    assert observed["visual_limits"].require_creator_overlays is True
    assert observed["profile"].video_encoder == "libx264"
    assert status in result.output and "summary.json" in result.output


def test_auto_edit_validates_encoder_without_constructing_client(inputs, monkeypatch):
    source, output = inputs
    monkeypatch.setattr(cli, "_automatic_client", lambda *a, **k: pytest.fail("client constructed"), raising=False)
    result = runner.invoke(cli.app, ["auto-edit", str(source), "--output-dir", str(output),
        "--confirm-inference", "--encoder", "invalid"])
    assert result.exit_code != 0 and "encoder" in result.output


def test_existing_local_proxy_config_can_supply_selected_model_without_persisting_key(tmp_path, monkeypatch):
    from clipperx.config import settings
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"host": "127.0.0.1", "port": 8317,
                                  "api-keys": ["test-only-local-key"]}), encoding="utf-8")
    before = config.read_bytes()
    monkeypatch.setattr(settings, "llm_provider", "disabled")
    monkeypatch.setattr(cli, "_proxy_config_path", lambda: config, raising=False)
    client = cli._automatic_client("gpt-6-luna(low)", 65536, True)
    assert client.config.base_url == "http://127.0.0.1:8317/v1"
    assert client.config.model == "gpt-6-luna(low)"
    assert client.config.api_key == "test-only-local-key"
    assert config.read_bytes() == before


def test_local_proxy_configuration_is_never_read_without_confirmation(monkeypatch):
    monkeypatch.setattr(cli, "_proxy_config_path", lambda: pytest.fail("proxy configuration read"), raising=False)
    with pytest.raises(Exception, match="confirm-inference"):
        cli._automatic_client("gpt-6-luna(low)", 65536, False)


def test_malformed_proxy_config_has_safe_error_without_credentials(tmp_path, monkeypatch):
    from clipperx.config import settings
    config = tmp_path / "config.yaml"
    config.write_text("api-keys: [ PLANTED_SECRET_MUST_NOT_APPEAR", encoding="utf-8")
    monkeypatch.setattr(settings, "llm_provider", "disabled")
    monkeypatch.setattr(cli, "_proxy_config_path", lambda: config, raising=False)
    import typer
    with pytest.raises(typer.BadParameter) as raised:
        cli._automatic_client("gpt-6-luna(low)", 65536, True)
    assert "PLANTED_SECRET" not in str(raised.value)
