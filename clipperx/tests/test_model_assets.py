from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest

from clipperx.model_assets import (
    FACE_MODELS,
    ModelAssetError,
    ModelAsset,
    ensure_face_models,
    validate_model_file,
)


def test_face_models_use_named_onnx_files() -> None:
    assert FACE_MODELS["yunet"].filename == "face_detection_yunet_2023mar.onnx"
    assert FACE_MODELS["sface"].filename == "face_recognition_sface_2021dec.onnx"
    assert FACE_MODELS["yunet"].url == (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    assert FACE_MODELS["yunet"].minimum_bytes == 200_000
    assert FACE_MODELS["sface"].url == (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    )
    assert FACE_MODELS["sface"].minimum_bytes == 30_000_000

def test_equivalent_model_asset_uses_stable_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2
    import clipperx.model_assets as assets

    expected = FACE_MODELS["yunet"]
    equivalent = ModelAsset(
        filename=expected.filename,
        url=expected.url,
        minimum_bytes=expected.minimum_bytes,
    )
    observed: list[Path] = []
    sentinel = object()

    def record_factory(path: str, config: str, size: tuple[int, int]) -> object:
        observed.append(Path(path))
        return sentinel

    monkeypatch.setattr(cv2, "FaceDetectorYN_create", record_factory)

    assets._load_model(equivalent, tmp_path / equivalent.filename)
    assert observed == [tmp_path / equivalent.filename]


def test_lfs_pointer_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / FACE_MODELS["yunet"].filename
    model.write_text("version https://git-lfs.github.com/spec/v1", encoding="ascii")
    assert validate_model_file(model, FACE_MODELS["yunet"]) is False


def test_wrong_extension_and_small_file_are_rejected(tmp_path: Path) -> None:
    wrong_extension = tmp_path / "face_detection_yunet_2023mar.bin"
    wrong_extension.write_bytes(b"x" * FACE_MODELS["yunet"].minimum_bytes)
    tiny_onnx = tmp_path / FACE_MODELS["yunet"].filename
    tiny_onnx.write_bytes(b"x")

    assert validate_model_file(wrong_extension, FACE_MODELS["yunet"]) is False
    assert validate_model_file(tiny_onnx, FACE_MODELS["yunet"]) is False


def test_disabled_download_has_actionable_error_and_never_calls_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    import clipperx.model_assets as assets

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("disabled setup must not touch the network")

    monkeypatch.setattr(assets, "_download_to", forbidden)
    monkeypatch.setattr(assets, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    with pytest.raises(ModelAssetError, match="ALLOW_MODEL_DOWNLOADS=true"):
        ensure_face_models(tmp_path, allow_download=False)


def test_manifest_is_written_only_after_both_models_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clipperx.model_assets as assets

    def fake_download(asset: ModelAsset, destination: Path) -> None:
        destination.write_bytes(b"x" * asset.minimum_bytes)

    def fake_validate(path: Path, asset: ModelAsset) -> bool:
        return path.exists() and path.stat().st_size >= asset.minimum_bytes

    def fake_load(asset: ModelAsset, path: Path) -> None:
        if asset.filename == FACE_MODELS["sface"].filename:
            raise ModelAssetError("SFace no carga")

    monkeypatch.setattr(assets, "_download_to", fake_download)
    monkeypatch.setattr(assets, "validate_model_file", fake_validate)
    monkeypatch.setattr(assets, "_load_model", fake_load)

    with pytest.raises(ModelAssetError, match="SFace no carga"):
        ensure_face_models(tmp_path, allow_download=True)

    assert not any(tmp_path.glob("*.onnx"))
    assert not (tmp_path / "face_models.manifest.json").exists()


def test_manifest_contains_hashes_after_both_models_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clipperx.model_assets as assets

    def fake_download(asset: ModelAsset, destination: Path) -> None:
        destination.write_bytes(b"x" * asset.minimum_bytes)

    monkeypatch.setattr(assets, "_download_to", fake_download)
    monkeypatch.setattr(assets, "validate_model_file", lambda path, asset: path.exists())
    monkeypatch.setattr(assets, "_load_model", lambda asset, path: None)

    result = ensure_face_models(tmp_path, allow_download=True)
    manifest = json.loads((tmp_path / "face_models.manifest.json").read_text(encoding="utf-8"))

    assert set(result) == {"yunet", "sface"}
    assert set(manifest["models"]) == {"yunet", "sface"}
    assert all(len(details["sha256"]) == 64 for details in manifest["models"].values())


def test_manifest_promotion_failure_restores_previous_pair_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clipperx.model_assets as assets

    previous = {
        "yunet": b"old-yunet",
        "sface": b"old-sface",
        "manifest": b'{"previous": true}\n',
    }
    for name, asset in FACE_MODELS.items():
        (tmp_path / asset.filename).write_bytes(previous[name])
    manifest_path = tmp_path / "face_models.manifest.json"
    manifest_path.write_bytes(previous["manifest"])

    def fake_validate(path: Path, asset: ModelAsset) -> bool:
        return path.parent != tmp_path and path.exists()

    def fake_download(asset: ModelAsset, destination: Path) -> None:
        destination.write_bytes(f"new-{asset.filename}".encode("ascii"))

    real_replace = assets.os.replace
    manifest_failure_injected = False

    def fail_manifest_promotion_once(source: Path, destination: Path) -> None:
        nonlocal manifest_failure_injected
        destination_path = Path(destination)
        if destination_path == manifest_path and not manifest_failure_injected:
            manifest_failure_injected = True
            raise OSError("manifest promotion failed")
        real_replace(source, destination)

    monkeypatch.setattr(assets, "validate_model_file", fake_validate)
    monkeypatch.setattr(assets, "_download_to", fake_download)
    monkeypatch.setattr(assets, "_load_model", lambda asset, path: None)
    monkeypatch.setattr(assets.os, "replace", fail_manifest_promotion_once)

    with pytest.raises(ModelAssetError, match="manifest promotion failed"):
        ensure_face_models(tmp_path, allow_download=True)

    assert (tmp_path / FACE_MODELS["yunet"].filename).read_bytes() == previous["yunet"]
    assert (tmp_path / FACE_MODELS["sface"].filename).read_bytes() == previous["sface"]
    assert manifest_path.read_bytes() == previous["manifest"]


def test_every_download_uses_only_part_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clipperx.model_assets as assets

    destinations: list[Path] = []
    loaded: list[Path] = []

    def capture_download(asset: ModelAsset, destination: Path) -> None:
        destinations.append(destination)
        destination.write_bytes(b"staged")

    def fake_validate(path: Path, asset: ModelAsset) -> bool:
        return path.name.endswith(".part") and path.exists()

    monkeypatch.setattr(assets, "_download_to", capture_download)
    monkeypatch.setattr(assets, "validate_model_file", fake_validate)
    monkeypatch.setattr(assets, "_load_model", lambda asset, path: loaded.append(path))
    monkeypatch.setattr(
        assets,
        "_stage_manifest",
        lambda directory, models: directory / ".manifest.part",
    )
    monkeypatch.setattr(
        assets, "_promote_bundle_atomically", lambda staged, manifest, directory: staged
    )

    ensure_face_models(tmp_path, allow_download=True)

    assert len(destinations) == len(FACE_MODELS) == 2
    assert all(destination.name.endswith(".part") for destination in destinations)
    assert {path.name for path in loaded} == {
        asset.filename for asset in FACE_MODELS.values()
    }


def test_failed_rollback_preserves_durable_transaction_for_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clipperx.model_assets as assets

    old_contents = {"yunet": b"old-yunet", "sface": b"old-sface"}
    for name, asset in FACE_MODELS.items():
        (tmp_path / asset.filename).write_bytes(old_contents[name])
    old_manifest = b'{"schema_version": 1, "models": {}}\n'
    manifest_path = tmp_path / assets.MANIFEST_FILENAME
    manifest_path.write_bytes(old_manifest)

    def fake_validate(path: Path, asset: ModelAsset) -> bool:
        return path.parent != tmp_path and path.exists()

    def fake_download(asset: ModelAsset, destination: Path) -> None:
        destination.write_bytes(f"new-{asset.filename}".encode("ascii"))

    real_replace = assets.os.replace
    promotion_failed = False
    rollback_failed = False

    def fail_promotion_and_one_restore(source: Path, destination: Path) -> None:
        nonlocal promotion_failed, rollback_failed
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == manifest_path and not promotion_failed:
            promotion_failed = True
            raise OSError("manifest promotion failed")
        if (
            source_path.parent.name == ".restore"
            and source_path.name == FACE_MODELS["sface"].filename
            and destination_path == tmp_path / FACE_MODELS["sface"].filename
            and not rollback_failed
        ):
            rollback_failed = True
            raise OSError("rollback restore failed")
        real_replace(source, destination)

    monkeypatch.setattr(assets, "validate_model_file", fake_validate)
    monkeypatch.setattr(assets, "_download_to", fake_download)
    monkeypatch.setattr(assets, "_load_model", lambda asset, path: None)
    monkeypatch.setattr(assets.os, "replace", fail_promotion_and_one_restore)

    with pytest.raises(ModelAssetError, match="preservada en") as error:
        ensure_face_models(tmp_path, allow_download=True)

    transaction_dirs = list(tmp_path.glob(".face_models_txn_*"))
    assert len(transaction_dirs) == 1
    assert str(transaction_dirs[0]) in str(error.value)
    assert (transaction_dirs[0] / "transaction.json").is_file()
    assert (transaction_dirs[0] / FACE_MODELS["sface"].filename).is_file()


def test_next_ensure_recovers_prepared_transaction_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clipperx.model_assets as assets

    finals = {name: tmp_path / asset.filename for name, asset in FACE_MODELS.items()}
    old_contents = {"yunet": b"old-yunet", "sface": b"old-sface"}
    for name, path in finals.items():
        path.write_bytes(old_contents[name])
    manifest_path = tmp_path / assets.MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(assets._manifest_payload(finals)),
        encoding="utf-8",
    )
    old_manifest = manifest_path.read_bytes()

    transaction_dir, _ = assets._create_transaction(
        tmp_path,
        [finals["yunet"], finals["sface"], manifest_path],
    )
    for path in finals.values():
        path.write_bytes(b"partial-new")
    manifest_path.write_bytes(b"partial-new-manifest")
    abandoned = tmp_path / ".face_models_staging_abandoned"
    abandoned.mkdir()
    (abandoned / "partial.onnx.part").write_bytes(b"partial")

    monkeypatch.setattr(assets, "validate_model_file", lambda path, asset: True)
    monkeypatch.setattr(assets, "_load_model", lambda asset, path: None)

    ensure_face_models(tmp_path, allow_download=False)

    assert finals["yunet"].read_bytes() == old_contents["yunet"]
    assert finals["sface"].read_bytes() == old_contents["sface"]
    assert manifest_path.read_bytes() == old_manifest
    assert not transaction_dir.exists()
    assert not abandoned.exists()


def test_interprocess_lock_rejects_concurrent_setup(
    tmp_path: Path,
) -> None:
    import clipperx.model_assets as assets

    lock_path = tmp_path / ".face_models.lock"
    with assets._InterProcessLock(lock_path):
        with pytest.raises(ModelAssetError, match="otro proceso.*setup-face-models"):
            ensure_face_models(tmp_path, allow_download=False)

    assert lock_path.is_file()


def test_existing_manifest_mismatch_is_rejected_without_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clipperx.model_assets as assets

    finals = {name: tmp_path / asset.filename for name, asset in FACE_MODELS.items()}
    for name, path in finals.items():
        path.write_bytes(f"original-{name}".encode("ascii"))
    manifest_path = tmp_path / assets.MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(assets._manifest_payload(finals)),
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_bytes()
    finals["yunet"].write_bytes(b"tampered-yunet")

    def forbidden_rewrite(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("an existing manifest must never be silently rewritten")

    monkeypatch.setattr(assets, "validate_model_file", lambda path, asset: True)
    monkeypatch.setattr(assets, "_load_model", lambda asset, path: None)
    monkeypatch.setattr(assets, "_write_manifest_atomically", forbidden_rewrite)

    with pytest.raises(ModelAssetError, match="manifest.*no coincide"):
        ensure_face_models(tmp_path, allow_download=False)

    assert manifest_path.read_bytes() == original_manifest


def test_recovery_rejects_incomplete_journal_without_deleting_evidence(
    tmp_path: Path,
) -> None:
    import clipperx.model_assets as assets

    transaction_dir = tmp_path / ".face_models_txn_invalid"
    transaction_dir.mkdir()
    (transaction_dir / "transaction.json").write_text(
        json.dumps({"schema_version": 1, "state": "prepared", "targets": []}),
        encoding="utf-8",
    )

    with pytest.raises(ModelAssetError, match="Journal.*inválido"):
        assets._recover_interrupted_transactions(tmp_path)

    assert transaction_dir.is_dir()


def test_recovery_rejects_traversal_backup_and_preserves_external_file(
    tmp_path: Path,
) -> None:
    import clipperx.model_assets as assets

    finals = {name: tmp_path / asset.filename for name, asset in FACE_MODELS.items()}
    for name, path in finals.items():
        path.write_bytes(f"old-{name}".encode("ascii"))
    manifest_path = tmp_path / assets.MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(assets._manifest_payload(finals)),
        encoding="utf-8",
    )
    transaction_dir, _ = assets._create_transaction(
        tmp_path,
        [finals["yunet"], finals["sface"], manifest_path],
    )
    journal_path = transaction_dir / "transaction.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    targets = journal["targets"]
    assert isinstance(targets, list)
    targets[0]["backup"] = "../outside.onnx"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"sensitive")
    finals["yunet"].write_bytes(b"partial-new")

    with pytest.raises(ModelAssetError, match="Journal.*inválido"):
        assets._recover_interrupted_transactions(tmp_path)

    assert outside.read_bytes() == b"sensitive"
    assert transaction_dir.is_dir()


def test_recovery_validates_backup_before_mutating_targets(
    tmp_path: Path,
) -> None:
    import clipperx.model_assets as assets

    finals = {name: tmp_path / asset.filename for name, asset in FACE_MODELS.items()}
    for name, path in finals.items():
        path.write_bytes(f"old-{name}".encode("ascii"))
    manifest_path = tmp_path / assets.MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(assets._manifest_payload(finals)),
        encoding="utf-8",
    )
    transaction_dir, _ = assets._create_transaction(
        tmp_path,
        [finals["yunet"], finals["sface"], manifest_path],
    )
    corrupted_backup = transaction_dir / FACE_MODELS["yunet"].filename
    corrupted_backup.write_bytes(b"corrupted-backup")
    for path in finals.values():
        path.write_bytes(b"partial-new")
    manifest_path.write_bytes(b"partial-manifest")

    with pytest.raises(ModelAssetError, match="backup.*no coincide"):
        assets._recover_interrupted_transactions(tmp_path)

    assert corrupted_backup.read_bytes() == b"corrupted-backup"
    assert finals["yunet"].read_bytes() == b"partial-new"
    assert transaction_dir.is_dir()


def test_interprocess_lock_blocks_a_second_python_process(tmp_path: Path) -> None:
    import subprocess
    import sys
    import time
    import clipperx.model_assets as assets

    lock_path = tmp_path / ".face_models.lock"
    ready_path = tmp_path / "child.ready"
    release_path = tmp_path / "child.release"
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from clipperx.model_assets import _InterProcessLock\n"
        "with _InterProcessLock(Path(sys.argv[1])):\n"
        "    Path(sys.argv[2]).write_text('ready', encoding='ascii')\n"
        "    while not Path(sys.argv[3]).exists():\n"
        "        time.sleep(0.02)\n"
    )
    project_root = Path(assets.__file__).resolve().parent.parent
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(lock_path),
            str(ready_path),
            str(release_path),
        ],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        while not ready_path.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready_path.exists(), process.stderr.read() if process.stderr else ""
        with pytest.raises(ModelAssetError, match="otro proceso.*setup-face-models"):
            ensure_face_models(tmp_path, allow_download=False)
    finally:
        release_path.write_text("release", encoding="ascii")
        process.wait(timeout=10)
    assert process.returncode == 0, process.stderr.read() if process.stderr else ""


def test_existing_models_without_manifest_are_rejected_without_download_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clipperx.model_assets as assets

    for name, asset in FACE_MODELS.items():
        (tmp_path / asset.filename).write_bytes(f"untrusted-{name}".encode("ascii"))

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("missing manifest must not be silently baselined")

    monkeypatch.setattr(assets, "validate_model_file", lambda path, asset: True)
    monkeypatch.setattr(assets, "_load_model", forbidden)
    monkeypatch.setattr(assets, "_write_manifest_atomically", forbidden)

    with pytest.raises(ModelAssetError, match="manifest.*ausente"):
        ensure_face_models(tmp_path, allow_download=False)

    assert not (tmp_path / assets.MANIFEST_FILENAME).exists()


def test_existing_models_without_manifest_are_reinstalled_when_download_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clipperx.model_assets as assets

    finals = {name: tmp_path / asset.filename for name, asset in FACE_MODELS.items()}
    for name, path in finals.items():
        path.write_bytes(f"untrusted-{name}".encode("ascii"))
    downloaded: list[str] = []

    def fake_download(asset: ModelAsset, destination: Path) -> None:
        downloaded.append(asset.filename)
        destination.write_bytes(f"official-{asset.filename}".encode("ascii"))

    def fake_validate(path: Path, asset: ModelAsset) -> bool:
        return path.exists()

    monkeypatch.setattr(assets, "validate_model_file", fake_validate)
    monkeypatch.setattr(assets, "_download_to", fake_download)
    monkeypatch.setattr(assets, "_load_model", lambda asset, path: None)

    result = ensure_face_models(tmp_path, allow_download=True)

    assert downloaded == [
        FACE_MODELS["yunet"].filename,
        FACE_MODELS["sface"].filename,
    ]
    assert set(result) == {"yunet", "sface"}
    assert (tmp_path / assets.MANIFEST_FILENAME).is_file()
