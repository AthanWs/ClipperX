"""Runtime diagnostics that never contact an LLM provider."""

from __future__ import annotations

import importlib
from importlib import metadata
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

from .render_assets import RenderAssetRegistry, default_render_assets, validate_asset_loadability
from .visual_compiler import RenderPreflightError, available_encoders, binary_version


def _installed_distribution_names() -> set[str]:
    """Return normalized distribution names visible in this Python runtime."""

    names: set[str] = set()
    for distribution in metadata.distributions():
        distribution_metadata = distribution.metadata
        name = distribution_metadata["Name"] if "Name" in distribution_metadata else None
        if name:
            names.add(name.lower().replace("_", "-"))
    return names


def _face_model_diagnostics(model_dir: Path) -> list[dict[str, Any]]:
    """Inspect face assets only; never call the downloader or an LLM."""

    from .model_assets import (
        FACE_MODELS,
        MANIFEST_FILENAME,
        _load_model,
        load_face_manifest,
        sha256_file,
        validate_face_manifest,
        validate_model_file,
    )

    checks: list[dict[str, Any]] = []
    try:
        import cv2

        detector_api = callable(getattr(cv2, "FaceDetectorYN_create", None))
        recognizer_api = callable(getattr(cv2, "FaceRecognizerSF_create", None))
        checks.append(
            {
                "name": "FaceDetectorYN_create",
                "ok": detector_api,
                "detail": "API disponible" if detector_api else "API no disponible",
                "required": True,
            }
        )
        checks.append(
            {
                "name": "FaceRecognizerSF_create",
                "ok": recognizer_api,
                "detail": "API disponible" if recognizer_api else "API no disponible",

                "required": True,
            }
        )
    except Exception as exc:
        checks.extend(
            {
                "name": name,
                "ok": False,
                "detail": f"cv2 no disponible: {exc}",
                "required": True,
            }
            for name in ("FaceDetectorYN_create", "FaceRecognizerSF_create")
        )


    manifest_path = model_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        checks.append(
            {
                "name": "face model manifest",
                "ok": False,
                "detail": f"manifest ausente: {manifest_path}",
                "required": True,
            }
        )
    else:
        try:
            manifest = load_face_manifest(manifest_path)
            mismatches = validate_face_manifest(model_dir, manifest)
        except Exception as exc:
            checks.append(
                {
                    "name": "face model manifest",
                    "ok": False,
                    "detail": f"manifest malformado o ilegible: {exc}",
                    "required": True,
                }
            )
        else:
            checks.append(
                {
                    "name": "face model manifest",
                    "ok": not mismatches,
                    "detail": (
                        "manifest coincide"
                        if not mismatches
                        else "manifest no coincide: " + "; ".join(mismatches)
                    ),
                    "required": True,
                }
            )
    for name, asset in FACE_MODELS.items():
        path = model_dir / asset.filename
        try:
            is_valid = validate_model_file(path, asset)
        except Exception as exc:
            is_valid = False
            validation_detail = f"validación fallida: {exc}"
        else:
            validation_detail = f"ausente o inválido: {path}"
        if not is_valid:
            checks.append(
                {
                    "name": f"face model {name}",
                    "ok": False,
                    "detail": validation_detail,
                    "required": True,
                }
            )
            continue

        try:
            _load_model(asset, path)
        except Exception as exc:
            load_ok = False
            load_detail = f"carga fallida: {exc}"
        else:
            load_ok = True
            load_detail = "carga correcta"
        try:
            digest = sha256_file(path)
        except Exception:
            digest = "no disponible"
        try:
            byte_count: int | str = path.stat().st_size
        except Exception:
            byte_count = "no disponible"
        metadata_ok = digest != "no disponible" and byte_count != "no disponible"
        detail = f"sha256={digest}, bytes={byte_count}, {load_detail}"
        checks.append(
            {
                "name": f"face model {name}",
                "ok": load_ok and metadata_ok,
                "detail": detail,
                "required": True,
            }
        )
    return checks


def _command_version(command: str) -> tuple[bool, str]:
    executable = shutil.which(command)
    if not executable:
        return False, "no encontrado en PATH"
    result = subprocess.run(
        [executable, "-version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    first_line = (
        (result.stdout or result.stderr).splitlines()[0]
        if result.returncode == 0
        else "error"
    )
    return result.returncode == 0, first_line


class RenderDiagnosticCheck(TypedDict):
    name: str
    ok: bool
    detail: str
    required: bool


class RenderDiagnosticsReport(TypedDict):
    ok: bool
    checks: list[RenderDiagnosticCheck]


def collect_render_diagnostics(
    *, assets: RenderAssetRegistry | None = None
) -> RenderDiagnosticsReport:
    """Read-only render dependency checks, without models, settings or encoding.

    Encoder listings identify binary capabilities, not a working GPU. Version
    and encoder-list queries do not expose a physical GPU name; do not infer one
    from the NVIDIA encoder's label or initialize hardware to discover it.
    """
    registry = assets if assets is not None else default_render_assets()
    checks: list[RenderDiagnosticCheck] = []
    for command in ("ffmpeg", "ffprobe"):
        try:
            version = binary_version(command)
        except RenderPreflightError as exc:
            checks.append({"name": command, "ok": False, "detail": str(exc), "required": True})
        else:
            checks.append({"name": command, "ok": True, "detail": version, "required": True})
    try:
        detected = available_encoders()
        detail = f"binary encoders={sorted(detected)}; hardware encoding untested"
    except RenderPreflightError as exc:
        detected = set()
        detail = str(exc)
    for encoder in ("h264_nvenc", "libx264"):
        checks.append({"name": encoder, "ok": encoder in detected,
                       "detail": detail, "required": False})
    checks.append({"name": "FFmpeg GPU", "ok": False, "required": False,
                   "detail": "GPU name not exposed by version/encoder listing; "
                             "hardware encoding untested"})
    for kind, table in (("brand", registry.brand_images), ("font", registry.fonts),
                        ("sfx", registry.sfx)):
        for asset_id, path in table.items():
            try:
                # SFX decoding/probing remains the compiler's preflight gate.
                if kind == "brand":
                    validate_asset_loadability("brand", path)
                elif kind == "font":
                    validate_asset_loadability("font", path)
                else:
                    validate_asset_loadability("sfx", path)
            except (OSError, ValueError) as exc:
                ok, asset_detail = False, str(exc)
            else:
                ok = True
                asset_detail = f"loadable: {path}" if kind != "sfx" else f"present: {path}; probe pending"
            checks.append({"name": f"render {kind} {asset_id}", "ok": ok,
                           "detail": asset_detail, "required": True})
    return {"ok": all(check["ok"] for check in checks if check["required"]), "checks": checks}


def collect_diagnostics() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "required": required})

    add("python", sys.version_info >= (3, 10), sys.version.split()[0])
    installed = _installed_distribution_names()
    opencv_conflict = {"opencv-python", "opencv-contrib-python"}.issubset(installed)
    add(
        "opencv distribution",
        not opencv_conflict,
        (
            "instalaciones conflictivas: opencv-python y opencv-contrib-python; "
            "conserva solo opencv-contrib-python"
            if opencv_conflict
            else "runtime OpenCV no ambiguo"
        ),
    )


    for command in ("ffmpeg", "ffprobe"):
        ok, detail = _command_version(command)
        add(command, ok, detail)

    checks.extend(dict(check) for check in collect_render_diagnostics()["checks"]
                  if check["name"] not in {"ffmpeg", "ffprobe"})

    required_modules = (
        "cv2",
        "faster_whisper",
        "numpy",
        "pyannote.audio",
        "soundfile",
        "textual",
        "torch",
        "whisperx",
    )
    for module in required_modules:
        try:
            importlib.import_module(module)
            add(module, True, "import correcto")
        except Exception as exc:
            add(module, False, str(exc))

    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        detail = f"torch={torch.__version__}, cuda={torch.version.cuda}, available={cuda_ok}"
        if cuda_ok:
            detail += f", gpu={torch.cuda.get_device_name(0)}"
        add("cuda", cuda_ok, detail)
    except Exception as exc:
        add("cuda", False, str(exc))

    project_root = Path(__file__).resolve().parent.parent
    diarization_python = project_root / ".venv-diarization" / "Scripts" / "python.exe"
    if diarization_python.exists():
        result = subprocess.run(
            [str(diarization_python), "-m", "clipperx.diarization_worker", "doctor"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        add(
            "community-1 runtime",
            result.returncode == 0,
            (result.stdout or result.stderr).strip(),
            required=False,
        )
    else:
        add(
            "community-1 runtime",
            False,
            "opcional: ejecuta scripts\\setup_diarization_env.ps1",
            required=False,
        )

    from .config import settings

    face_model_dir = Path(settings.face_model_dir)
    if not face_model_dir.is_absolute():
        face_model_dir = project_root / face_model_dir
    checks.extend(_face_model_diagnostics(face_model_dir))

    return {
        "ok": all(check["ok"] for check in checks if check["required"]),
        "checks": checks,
    }
