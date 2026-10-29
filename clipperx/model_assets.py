"""Local, reproducible assets required for face identity analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import TracebackType
from typing import BinaryIO, Callable, Mapping, Protocol, cast
from urllib.request import urlopen


class ModelAssetError(RuntimeError):
    """Raised when face model assets are absent, corrupt, or unusable."""


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """An externally published ONNX model required by the local runtime."""

    filename: str
    url: str
    minimum_bytes: int


FACE_MODELS: Mapping[str, ModelAsset] = {
    "yunet": ModelAsset(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        minimum_bytes=200_000,
    ),
    "sface": ModelAsset(
        filename="face_recognition_sface_2021dec.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        minimum_bytes=30_000_000,
    ),
}

MANIFEST_FILENAME = "face_models.manifest.json"
_TRANSACTION_PREFIX = ".face_models_txn_"
_STAGING_PREFIX = ".face_models_staging_"
_JOURNAL_FILENAME = "transaction.json"

_LOCK_FILENAME = ".face_models.lock"
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class _PosixFileLock(Protocol):
    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, file_descriptor: int, operation: int) -> None: ...


class _InterProcessLock:
    """Non-blocking OS lock released automatically when the process exits."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: BinaryIO | None = None

    def __enter__(self) -> "_InterProcessLock":
        stream = self.path.open("a+b")
        self._stream = stream
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl = cast(_PosixFileLock, import_module("fcntl"))

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            self._stream = None
            raise ModelAssetError(
                "otro proceso está preparando los modelos faciales. Espera a que termine "
                "setup-face-models y vuelve a intentarlo."
            ) from exc
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl = cast(_PosixFileLock, import_module("fcntl"))

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
            self._stream = None


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model_file(path: Path, asset: ModelAsset) -> bool:
    """Return whether a candidate has the expected ONNX extension and size."""

    try:
        if not path.is_file() or path.name not in {asset.filename, f"{asset.filename}.part"}:
            return False
        if path.stat().st_size < asset.minimum_bytes:
            return False
        with path.open("rb") as stream:
            return not stream.read(1024).lstrip().startswith(_LFS_POINTER_PREFIX)
    except OSError:
        return False


def _download_to(asset: ModelAsset, destination: Path) -> None:
    """Download an asset to its caller-provided staging path."""

    try:
        with urlopen(asset.url, timeout=120) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except OSError as exc:
        raise ModelAssetError(f"No se pudo descargar {asset.filename}: {exc}") from exc


def _load_model(asset: ModelAsset, path: Path) -> None:
    """Use the target OpenCV API to prove an ONNX model can be instantiated."""

    try:
        import cv2

        if asset.filename == FACE_MODELS["yunet"].filename:
            factory = cast(Callable[[str, str, tuple[int, int]], object], getattr(cv2, "FaceDetectorYN_create"))
            model = factory(str(path), "", (320, 320))
        elif asset.filename == FACE_MODELS["sface"].filename:
            recognizer_factory = cast(Callable[[str, str], object], getattr(cv2, "FaceRecognizerSF_create"))
            model = recognizer_factory(str(path), "")
        else:
            raise ModelAssetError(f"Activo facial no reconocido: {asset.filename}")
    except ModelAssetError:
        raise
    except Exception as exc:
        raise ModelAssetError(f"OpenCV no pudo cargar {asset.filename}: {exc}") from exc
    if model is None:
        raise ModelAssetError(f"OpenCV devolvió un modelo vacío para {asset.filename}")


def _manifest_payload(models: Mapping[str, Path]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "models": {
            name: {
                "filename": FACE_MODELS[name].filename,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in models.items()
        },
    }


def load_face_manifest(path: Path) -> dict[str, object]:
    """Load and strictly validate a schema-1 face-model manifest."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelAssetError(f"Manifest facial ilegible o malformado: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ModelAssetError(f"Manifest facial malformado: schema_version debe ser 1: {path}")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != set(FACE_MODELS):
        raise ModelAssetError(f"Manifest facial malformado: modelos esperados ausentes: {path}")
    for name, asset in FACE_MODELS.items():
        entry = models.get(name)
        if not isinstance(entry, dict):
            raise ModelAssetError(f"Manifest facial malformado: entrada {name}: {path}")
        filename = entry.get("filename")
        byte_count = entry.get("bytes")
        digest = entry.get("sha256")
        if filename != asset.filename:
            raise ModelAssetError(
                f"Manifest facial malformado: filename de {name} no coincide: {path}"
            )
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ModelAssetError(
                f"Manifest facial malformado: bytes de {name} no es válido: {path}"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ModelAssetError(
                f"Manifest facial malformado: sha256 de {name} no es válido: {path}"
            )
    return payload


def validate_face_manifest(model_dir: Path, manifest: Mapping[str, object]) -> list[str]:
    """Return filename, size, and digest mismatches without writing anything."""

    mismatches: list[str] = []
    models = manifest.get("models")
    if not isinstance(models, dict):
        return ["models ausente"]
    for name, asset in FACE_MODELS.items():
        entry = models.get(name)
        if not isinstance(entry, dict):
            mismatches.append(f"{name}: entrada ausente")
            continue
        path = model_dir / asset.filename
        try:
            byte_count = path.stat().st_size
            digest = sha256_file(path)
        except OSError as exc:
            mismatches.append(f"{name}: archivo no disponible: {exc}")
            continue
        if entry.get("filename") != asset.filename:
            mismatches.append(f"{name}: filename no coincide")
        if entry.get("bytes") != byte_count:
            mismatches.append(f"{name}: bytes no coincide")
        if entry.get("sha256") != digest:
            mismatches.append(f"{name}: sha256 no coincide")
    return mismatches

def _stage_manifest(directory: Path, models: Mapping[str, Path]) -> Path:
    staged = directory / f".{MANIFEST_FILENAME}.part"
    staged.write_text(
        json.dumps(_manifest_payload(models), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return staged


def _write_manifest_atomically(model_dir: Path, models: Mapping[str, Path]) -> None:
    os.replace(_stage_manifest(model_dir, models), model_dir / MANIFEST_FILENAME)


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _matches_recorded_file(path: Path, entry: Mapping[str, object]) -> bool:
    try:
        return (
            path.is_file()
            and path.stat().st_size == entry["bytes"]
            and sha256_file(path) == entry["sha256"]
        )
    except (KeyError, OSError):
        return False


def _create_transaction(model_dir: Path, targets: list[Path]) -> tuple[Path, dict[str, object]]:
    transaction_dir = Path(tempfile.mkdtemp(prefix=_TRANSACTION_PREFIX, dir=model_dir))
    entries: list[dict[str, object]] = []
    for target in targets:
        exists = target.is_file()
        entry: dict[str, object] = {
            "target": target.name,
            "original_exists": exists,
            "backup": target.name,
        }
        if exists:
            entry["bytes"] = target.stat().st_size
            entry["sha256"] = sha256_file(target)
        entries.append(entry)

    journal: dict[str, object] = {
        "schema_version": 1,
        "state": "preparing",
        "targets": entries,
    }
    journal_path = transaction_dir / _JOURNAL_FILENAME
    _write_json_atomically(journal_path, journal)
    for entry in entries:
        if entry["original_exists"]:
            target = model_dir / str(entry["target"])
            backup = transaction_dir / str(entry["backup"])
            shutil.copy2(target, backup)
            if not _matches_recorded_file(backup, entry):
                raise ModelAssetError(f"El backup transaccional no coincide: {backup}")
    journal["state"] = "prepared"
    _write_json_atomically(journal_path, journal)
    return transaction_dir, journal


def _load_transaction_journal(transaction_dir: Path) -> dict[str, object]:
    journal_path = transaction_dir / _JOURNAL_FILENAME
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelAssetError(
            f"Journal de recuperación ilegible; evidencia preservada en {transaction_dir}: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("state") not in {"preparing", "prepared", "committed"}
        or not isinstance(payload.get("targets"), list)
    ):
        raise ModelAssetError(
            f"Journal de recuperación inválido; evidencia preservada en {transaction_dir}"
        )
    targets = payload["targets"]
    expected_targets = {
        *(asset.filename for asset in FACE_MODELS.values()),
        MANIFEST_FILENAME,
    }
    seen_targets: set[str] = set()
    transaction_root = transaction_dir.resolve()

    def invalid(detail: str) -> ModelAssetError:
        return ModelAssetError(
            f"Journal de recuperación inválido; evidencia preservada en "
            f"{transaction_dir}: {detail}"
        )

    for raw_entry in targets:
        if not isinstance(raw_entry, dict):
            raise invalid("entrada de target no es un objeto")
        target = raw_entry.get("target")
        backup = raw_entry.get("backup")
        original_exists = raw_entry.get("original_exists")
        if (
            not isinstance(target, str)
            or target not in expected_targets
            or Path(target).name != target
            or backup != target
            or target in seen_targets
            or type(original_exists) is not bool
        ):
            raise invalid(f"entrada de target insegura o duplicada: {raw_entry!r}")
        backup_path = (transaction_dir / target).resolve()
        if backup_path.parent != transaction_root:
            raise invalid(f"backup fuera de la transacción: {target}")

        if original_exists:
            byte_count = raw_entry.get("bytes")
            digest = raw_entry.get("sha256")
            if (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise invalid(f"metadata original inválida: {target}")
        elif "bytes" in raw_entry or "sha256" in raw_entry:
            raise invalid(f"metadata inesperada para target originalmente ausente: {target}")
        seen_targets.add(target)

    if seen_targets != expected_targets:
        missing = sorted(expected_targets - seen_targets)
        raise invalid(f"targets incompletos; faltan: {missing}")
    return payload


def _restore_transaction(
    model_dir: Path, transaction_dir: Path, journal: Mapping[str, object]
) -> None:
    entries = journal["targets"]
    assert isinstance(entries, list)
    restore_dir = transaction_dir / ".restore"
    restore_dir.mkdir(exist_ok=True)
    prepared: dict[str, Path] = {}
    failures: list[str] = []

    # Validate every source before mutating any target. Backups are copied,
    # never consumed, so a failed restore remains recoverable.
    for raw_entry in entries:
        assert isinstance(raw_entry, dict)
        target_name = str(raw_entry["target"])
        if raw_entry["original_exists"] is not True:
            continue
        target = model_dir / target_name
        backup = transaction_dir / target_name
        if _matches_recorded_file(target, raw_entry):
            continue
        if not _matches_recorded_file(backup, raw_entry):
            failures.append(f"{target_name}: backup ausente o no coincide")
            continue
        candidate = restore_dir / target_name
        try:
            shutil.copy2(backup, candidate)
        except OSError as exc:
            failures.append(f"{target_name}: no se pudo copiar backup: {exc}")
            continue
        if not _matches_recorded_file(candidate, raw_entry):
            failures.append(f"{target_name}: copia de backup no coincide")
            continue
        prepared[target_name] = candidate

    if failures:
        raise ModelAssetError("; ".join(failures))

    for raw_entry in entries:
        assert isinstance(raw_entry, dict)
        target_name = str(raw_entry["target"])
        target = model_dir / target_name
        try:
            if raw_entry["original_exists"] is True:
                if _matches_recorded_file(target, raw_entry):
                    continue
                prepared_candidate = prepared.get(target_name)
                if prepared_candidate is None:
                    raise OSError(f"copia restaurable ausente: {target_name}")
                os.replace(prepared_candidate, target)
                if not _matches_recorded_file(target, raw_entry):
                    raise OSError(f"restauración no coincide: {target_name}")
            elif target.exists():
                target.unlink()
        except OSError as exc:
            failures.append(f"{target_name}: {exc}")
    if failures:
        raise ModelAssetError("; ".join(failures))
    shutil.rmtree(restore_dir, ignore_errors=True)


def _recover_interrupted_transactions(model_dir: Path) -> None:
    for transaction_dir in sorted(model_dir.glob(f"{_TRANSACTION_PREFIX}*")):
        if not transaction_dir.is_dir():
            continue
        journal = _load_transaction_journal(transaction_dir)
        state = journal["state"]
        if state == "prepared":
            try:
                _restore_transaction(model_dir, transaction_dir, journal)
            except ModelAssetError as exc:
                raise ModelAssetError(
                    f"No se pudo recuperar la transacción; evidencia preservada en "
                    f"{transaction_dir}: {exc}"
                ) from exc
        if state in {"preparing", "prepared", "committed"}:
            shutil.rmtree(transaction_dir)


def _cleanup_abandoned_staging(model_dir: Path) -> None:
    for staging_dir in model_dir.glob(f"{_STAGING_PREFIX}*"):
        if staging_dir.is_dir():
            shutil.rmtree(staging_dir)


def _promote_bundle_atomically(
    staged: Mapping[str, Path], staged_manifest: Path, model_dir: Path
) -> dict[str, Path]:
    """Promote the model pair and manifest as one recoverable transaction."""

    finals = {name: model_dir / asset.filename for name, asset in FACE_MODELS.items()}
    promotions = [(staged[name], finals[name]) for name in FACE_MODELS]
    promotions.append((staged_manifest, model_dir / MANIFEST_FILENAME))
    transaction_dir, journal = _create_transaction(
        model_dir, [target for _, target in promotions]
    )
    try:
        for source, target in promotions:
            os.replace(source, target)
        journal["state"] = "committed"
        _write_json_atomically(transaction_dir / _JOURNAL_FILENAME, journal)
    except Exception as promotion_error:
        try:
            _restore_transaction(model_dir, transaction_dir, journal)
        except Exception as rollback_error:
            raise ModelAssetError(
                f"Falló la promoción y también el rollback; recuperación preservada en "
                f"{transaction_dir}: {rollback_error}"
            ) from promotion_error
        shutil.rmtree(transaction_dir)
        raise ModelAssetError(
            f"Falló la promoción; se restauró el bundle anterior: {promotion_error}"
        ) from promotion_error
    shutil.rmtree(transaction_dir, ignore_errors=True)
    return finals


def _ensure_face_models_unlocked(directory: Path, allow_download: bool) -> dict[str, Path]:
    _recover_interrupted_transactions(directory)
    _cleanup_abandoned_staging(directory)
    finals = {name: directory / asset.filename for name, asset in FACE_MODELS.items()}
    valid = {name: validate_model_file(path, FACE_MODELS[name]) for name, path in finals.items()}

    if all(valid.values()):
        manifest_path = directory / MANIFEST_FILENAME
        manifest_ok = True
        if not manifest_path.exists():
            if not allow_download:
                raise ModelAssetError(
                    "El manifest facial está ausente; no se confiará en modelos sin "
                    "procedencia verificada. Ejecuta setup con ALLOW_MODEL_DOWNLOADS=true."
                )
            manifest_ok = False
        else:
            try:
                manifest = load_face_manifest(manifest_path)
                mismatches = validate_face_manifest(directory, manifest)
            except ModelAssetError:
                if not allow_download:
                    raise
                manifest_ok = False
            else:
                if mismatches:
                    if not allow_download:
                        raise ModelAssetError(
                            "El manifest facial no coincide con los archivos: "
                            + "; ".join(mismatches)
                        )
                    manifest_ok = False
        if manifest_ok:
            for name, path in finals.items():
                _load_model(FACE_MODELS[name], path)
        if manifest_ok:
            return finals

    missing = ", ".join(name for name, is_valid in valid.items() if not is_valid)
    if not allow_download:
        raise ModelAssetError(
            f"Faltan o son inválidos los modelos faciales: {missing}. "
            "Ejecuta el setup explícito con ALLOW_MODEL_DOWNLOADS=true."
        )

    staging_dir = Path(tempfile.mkdtemp(prefix=".face_models_staging_", dir=directory))
    try:
        staged = {
            name: staging_dir / f"{asset.filename}.part"
            for name, asset in FACE_MODELS.items()
        }
        for name, asset in FACE_MODELS.items():
            _download_to(asset, staged[name])
            if not validate_model_file(staged[name], asset):
                raise ModelAssetError(f"La descarga de {asset.filename} no superó la validación")

        ready = {name: staging_dir / asset.filename for name, asset in FACE_MODELS.items()}
        for name in FACE_MODELS:
            os.replace(staged[name], ready[name])
            _load_model(FACE_MODELS[name], ready[name])
        staged_manifest = _stage_manifest(staging_dir, ready)
        promoted = _promote_bundle_atomically(ready, staged_manifest, directory)
        return promoted
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def ensure_face_models(model_dir: Path | str, allow_download: bool) -> dict[str, Path]:
    """Validate both face assets and optionally obtain them as one transaction."""

    directory = Path(model_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with _InterProcessLock(directory / _LOCK_FILENAME):
            return _ensure_face_models_unlocked(directory, allow_download)
    except ModelAssetError:
        raise
    except OSError as exc:
        raise ModelAssetError(
            f"Error de filesystem al preparar los modelos faciales en {directory}: {exc}"
        ) from exc
