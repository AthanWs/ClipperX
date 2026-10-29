from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn

import pytest

from clipperx import diagnostics
from clipperx.diagnostics import RenderDiagnosticCheck, RenderDiagnosticsReport


def _check(report: RenderDiagnosticsReport, name: str) -> RenderDiagnosticCheck:
    return next(item for item in report["checks"] if item["name"] == name)


def _legacy_check(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in report["checks"] if item["name"] == name)


def test_installed_distribution_names_preserve_missing_empty_and_normalized_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMetadata:
        def __init__(self, name: str | None) -> None:
            self.name = name

        def get(self, key: str) -> str | None:
            assert key == "Name"
            return self.name

        def __contains__(self, key: str) -> bool:
            assert key == "Name"
            return self.name is not None

        def __getitem__(self, key: str) -> str:
            assert key == "Name"
            assert self.name is not None
            return self.name

    class FakeDistribution:
        def __init__(self, name: str | None) -> None:
            self._metadata = FakeMetadata(name)
            self.metadata_reads = 0

        @property
        def metadata(self) -> FakeMetadata:
            self.metadata_reads += 1
            return self._metadata

    distributions = [
        FakeDistribution(None),
        FakeDistribution(""),
        FakeDistribution("OpenCV_Contrib_Python"),
        FakeDistribution("NumPy"),
    ]
    monkeypatch.setattr(diagnostics.metadata, "distributions", lambda: distributions)

    assert diagnostics._installed_distribution_names() == {"opencv-contrib-python", "numpy"}
    assert [distribution.metadata_reads for distribution in distributions] == [1, 1, 1, 1]


def test_diagnostics_detects_conflicting_opencv_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_modules: list[str] = []
    doctor_calls: list[list[str]] = []

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        imported_modules.append(name)
        return ModuleType(name)

    def fake_doctor(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert args[1:] == ["-m", "clipperx.diarization_worker", "doctor"]
        assert kwargs["timeout"] == 60
        doctor_calls.append(args)
        return subprocess.CompletedProcess(args, 0, "fixture doctor", "")

    fake_torch = ModuleType("torch")
    fake_torch.__dict__.update(
        __version__="fixture", cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setattr(
        diagnostics,
        "_installed_distribution_names",
        lambda: {"opencv-python", "opencv-contrib-python"},
    )
    monkeypatch.setattr(diagnostics, "_face_model_diagnostics", lambda _: [])
    monkeypatch.setattr(diagnostics, "_command_version", lambda _: (True, "fixture binary"))
    monkeypatch.setattr(
        diagnostics, "collect_render_diagnostics", lambda: {"ok": True, "checks": []}
    )
    monkeypatch.setattr(diagnostics.importlib, "import_module", fake_import)
    monkeypatch.setattr(diagnostics.subprocess, "run", fake_doctor)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    report = diagnostics.collect_diagnostics()

    check = _legacy_check(report, "opencv distribution")
    assert check["ok"] is False
    assert "opencv-python" in str(check["detail"])
    assert "opencv-contrib-python" in str(check["detail"])
    assert imported_modules == [
        "cv2", "faster_whisper", "numpy", "pyannote.audio", "soundfile",
        "textual", "torch", "whisperx",
    ]
    assert len(doctor_calls) <= 1


def test_face_model_diagnostics_never_call_ensure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("diagnostics must remain strictly offline")

    import clipperx.model_assets as assets

    monkeypatch.setattr(assets, "ensure_face_models", forbidden)
    monkeypatch.setattr(assets, "_download_to", forbidden)
    monkeypatch.setattr(assets, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    checks = diagnostics._face_model_diagnostics(tmp_path)

    assert {check["name"] for check in checks} >= {
        "FaceDetectorYN_create",
        "FaceRecognizerSF_create",
        "face model yunet",
        "face model sface",
    }


def test_face_model_diagnostics_survives_file_race_after_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import clipperx.model_assets as assets

    for asset in assets.FACE_MODELS.values():
        (tmp_path / asset.filename).write_bytes(b"present-during-validation")

    def disappear_during_load(asset: object, path: Path) -> None:
        path.unlink()
        raise assets.ModelAssetError("file disappeared")

    monkeypatch.setattr(assets, "validate_model_file", lambda path, asset: True)
    monkeypatch.setattr(assets, "_load_model", disappear_during_load)

    checks = diagnostics._face_model_diagnostics(tmp_path)
    model_checks = [
        check
        for check in checks
        if check["name"] in {"face model yunet", "face model sface"}
    ]

    assert len(model_checks) == 2
    assert all(check["ok"] is False for check in model_checks)
    assert all("file disappeared" in check["detail"] for check in model_checks)
    assert all("sha256=no disponible" in check["detail"] for check in model_checks)
    assert all("bytes=no disponible" in check["detail"] for check in model_checks)


def test_face_model_diagnostics_reports_missing_manifest_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import clipperx.model_assets as assets

    for asset in assets.FACE_MODELS.values():
        (tmp_path / asset.filename).write_bytes(b"model")

    def forbidden_write(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("diagnostics must never create a baseline manifest")

    monkeypatch.setattr(assets, "validate_model_file", lambda path, asset: True)
    monkeypatch.setattr(assets, "_load_model", lambda asset, path: None)
    monkeypatch.setattr(assets, "_write_manifest_atomically", forbidden_write)

    checks = diagnostics._face_model_diagnostics(tmp_path)
    manifest_check = next(
        check for check in checks if check["name"] == "face model manifest"
    )

    assert manifest_check["ok"] is False
    assert "ausente" in manifest_check["detail"]
    assert not (tmp_path / assets.MANIFEST_FILENAME).exists()


def test_config_module_import_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("config import must not create runtime directories")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    config_path = Path(diagnostics.__file__).with_name("config.py")
    runpy.run_path(str(config_path), run_name="clipperx_config_read_only_probe")


def test_render_diagnostics_are_read_only_and_report_binary_not_hardware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess
    from clipperx import visual_compiler
    from clipperx.render_assets import RenderAssetRegistry

    commands: list[list[str]] = []
    monkeypatch.setattr(visual_compiler.shutil, "which", lambda command: command)

    def inspect(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        assert args[1:] in (["-version"], ["-hide_banner", "-encoders"])
        output = (" V....D h264_nvenc NVIDIA encoder\n V....D libx264 CPU\n"
                  if "-encoders" in args else f"{args[0]} version HOST-fixture")
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(visual_compiler.subprocess, "run", inspect)
    registry = RenderAssetRegistry({"kick_brand": tmp_path / "missing.png"},
        {"headline_impact": tmp_path / "impact.ttf", "subtitle_bold": tmp_path / "arialbd.ttf"}, {})
    report = diagnostics.collect_render_diagnostics(assets=registry)
    assert _check(report, "ffmpeg")["detail"] == "ffmpeg version HOST-fixture"
    assert _check(report, "h264_nvenc")["ok"] is True
    assert _check(report, "libx264")["ok"] is True
    gpu = _check(report, "FFmpeg GPU")
    assert "not exposed" in str(gpu["detail"])
    assert "hardware encoding untested" in str(gpu["detail"])
    assert _check(report, "render brand kick_brand")["ok"] is False
    assert str(tmp_path / "impact.ttf") in str(
        _check(report, "render font headline_impact")["detail"])
    assert _check(report, "render font subtitle_bold")["ok"] is False
    assert not list(tmp_path.iterdir())
    assert len(commands) == 3


def test_render_diagnostics_survive_unavailable_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clipperx import visual_compiler
    from clipperx.render_assets import RenderAssetRegistry

    monkeypatch.setattr(visual_compiler.shutil, "which", lambda command: None)
    report = diagnostics.collect_render_diagnostics(assets=RenderAssetRegistry({}, {}, {}))
    assert report["ok"] is False
    assert _check(report, "h264_nvenc")["ok"] is False
    assert "not found" in str(_check(report, "ffprobe")["detail"])
