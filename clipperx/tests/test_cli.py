from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

from typer.testing import CliRunner

import clipperx
from clipperx import cli
from clipperx.config import settings
from clipperx.model_assets import ModelAssetError
import pytest
from typing import Any


runner = CliRunner()


@pytest.mark.parametrize("arguments", [[], ["missing.mp4", "bad.json", "--output-dir", "out"],
    ["missing.mp4", "bad.json", "--output-dir", "out", "--transcript", "old.json", "--preset", "default"]])
def test_render_plan_deprecation_does_not_open_or_sniff(monkeypatch: pytest.MonkeyPatch,
        arguments: list[str]) -> None:
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("deprecated route read a plan"))
    result = runner.invoke(cli.app, ["render-plan", *arguments])
    assert result.exit_code != 0
    assert "render-visual-plan" in result.output and "render-legacy-plan" in result.output


def test_visual_command_passes_explicit_registry_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import clipperx.plan_render as module
    from clipperx.render_assets import default_render_assets
    from clipperx.tests.test_overlay_ass import plan_with_two_subtitle_regions
    plan = tmp_path / "plan.json"
    plan.write_text(plan_with_two_subtitle_regions().model_dump_json(), encoding="utf-8")
    seen: dict[str, Any] = {}
    def execute(*args: object, **kwargs: object) -> list[Path]:
        seen.update(kwargs)
        return [Path("clip.mp4")]
    monkeypatch.setattr(module, "render_visual_composition_plan", execute)
    result = runner.invoke(cli.app, ["render-visual-plan", "source.mp4", str(plan), "--output-dir", "out", "--encoder", "libx264"])
    assert result.exit_code == 0, result.output
    assert seen["assets"].sfx == default_render_assets().sfx
    assert seen["profile"].video_encoder == "libx264"


def test_setup_face_models_uses_configured_download_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_ensure(model_dir: Path, allow_download: bool) -> dict[str, Path]:
        observed["model_dir"] = Path(model_dir)
        observed["allow_download"] = allow_download
        return {
            "yunet": Path(model_dir) / "face_detection_yunet_2023mar.onnx",
            "sface": Path(model_dir) / "face_recognition_sface_2021dec.onnx",
        }

    monkeypatch.setattr(settings, "face_model_dir", "data/models/face_identity")
    monkeypatch.setattr(settings, "allow_model_downloads", True)
    monkeypatch.setattr("clipperx.model_assets.ensure_face_models", fake_ensure)

    result = runner.invoke(cli.app, ["setup-face-models"])

    project_root = Path(cli.__file__).resolve().parent.parent
    assert result.exit_code == 0, result.output
    assert observed == {
        "model_dir": project_root / "data/models/face_identity",
        "allow_download": True,
    }
    assert "YuNet" in result.output
    assert "SFace" in result.output


def test_setup_face_models_converts_asset_error_to_actionable_cli_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise ModelAssetError("Activa ALLOW_MODEL_DOWNLOADS=true")

    monkeypatch.setattr("clipperx.model_assets.ensure_face_models", fail)

    result = runner.invoke(cli.app, ["setup-face-models"])

    assert result.exit_code != 0
    assert "ALLOW_MODEL_DOWNLOADS=true" in result.output


def test_doctor_never_calls_face_model_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("doctor must remain strictly offline")

    monkeypatch.setattr("clipperx.model_assets.ensure_face_models", forbidden)
    monkeypatch.setattr(
        "clipperx.diagnostics.collect_diagnostics",
        lambda: {"ok": True, "checks": []},
    )

    result = runner.invoke(cli.app, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    assert '"ok": true' in result.output


def test_setup_face_models_reports_mkdir_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import clipperx.model_assets as assets

    target = tmp_path / "uncreatable"

    def fail_mkdir(*args: object, **kwargs: object) -> None:
        raise OSError("filesystem denied")

    monkeypatch.setattr(settings, "face_model_dir", str(target))
    monkeypatch.setattr(assets.Path, "mkdir", fail_mkdir)

    result = runner.invoke(cli.app, ["setup-face-models"])

    assert result.exit_code == 1
    assert "filesystem" in result.output.lower()
    assert "filesystem denied" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("face_crop", "expected"),
    [
        ("13,-4,640,360", (13, -4, 640, 360)),
        ("13,-4,640", None),
        ("13,-4,640,360,2", None),
    ],
)
def test_legacy_run_preserves_face_crop_width_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    face_crop: str,
    expected: tuple[int, int, int, int] | None,
) -> None:
    from clipperx.editing import LayoutConfig

    calls: list[dict[str, object]] = []

    def execute(**kwargs: object) -> list[Path]:
        calls.append(kwargs)
        return [Path("clip.mp4")]

    pipeline = ModuleType("clipperx.pipeline")
    monkeypatch.setattr(pipeline, "full_pipeline", execute, raising=False)
    monkeypatch.setitem(sys.modules, "clipperx.pipeline", pipeline)
    monkeypatch.setattr(clipperx, "pipeline", pipeline, raising=False)
    monkeypatch.setattr(cli, "configure_logging", lambda **kwargs: None)
    source = tmp_path / "source.mp4"
    source.write_text("disposable CLI path fixture", encoding="utf-8")

    result = runner.invoke(cli.app, ["run", str(source), "--face-crop", face_crop])

    if expected is None:
        assert result.exit_code != 0
        assert "El formato debe ser x,y,w,h" in result.output
        assert calls == []
    else:
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        layout = calls[0]["layout_override"]
        assert isinstance(layout, LayoutConfig)
        assert layout.face_crop == expected
        assert calls[0]["video_path"] == source
