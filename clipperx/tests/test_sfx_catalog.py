"""Real WAV fixtures for the explicit host-controlled SFX catalog boundary."""
from dataclasses import FrozenInstanceError
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import struct
import wave

import pytest


def _loader():
    assert importlib.util.find_spec("clipperx.sfx_catalog") is not None, "SFX catalog API is missing"
    return importlib.import_module("clipperx.sfx_catalog").load_sfx_catalog


def _wav(path: Path, *, frames: int = 96, rate: int = 48000, channels: int = 2) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((channels, 2, rate, frames, "NONE", "not compressed"))
        output.writeframes(struct.pack("<h", 1000) * frames * channels)
    return path.read_bytes()


def _fixture(tmp_path: Path):
    audio = tmp_path / "cues/pop.wav"
    raw = _wav(audio)
    entry = {
        "asset_id": "pop_01", "family": "pop", "path": "cues/pop.wav",
        "sha256": hashlib.sha256(raw).hexdigest(), "sample_rate": 48000, "channels": 2,
        "frames": 96, "transient_sample": 10, "default_gain_db": -18,
        "approval": {"status": "pending", "reference": ""},
        "provenance": {"generation": {"seeds": [101]}},
    }
    return tmp_path / "catalog.json", audio, {"version": 1, "assets": [entry]}


def _write(path: Path, document: dict):
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return _loader()(path)


def test_pending_catalog_is_verified_but_not_offered_and_is_immutable(tmp_path: Path) -> None:
    path, audio, document = _fixture(tmp_path)
    catalog = _write(path, document)
    assert catalog.approved_asset_paths() == {}
    assert isinstance(catalog.assets, tuple)
    asset = catalog.assets[0]
    assert asset.path == audio.resolve()
    assert asset.frames == 96 and asset.approval.status == "pending"
    assert len(catalog.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        asset.frames = 1
    with pytest.raises(TypeError):
        asset.provenance["new"] = 1
    assert asset.provenance["generation"]["seeds"] == (101,)
    with pytest.raises(TypeError):
        asset.provenance["generation"]["new"] = 1


def test_only_host_approved_assets_are_offered_and_map_is_detached(tmp_path: Path) -> None:
    path, audio, document = _fixture(tmp_path)
    document["assets"][0]["approval"] = {"status": "approved", "reference": "host-review-001"}
    catalog = _write(path, document)
    offered = catalog.approved_asset_paths()
    assert offered == {"pop_01": audio.resolve()}
    offered.clear()
    assert catalog.approved_asset_paths() == {"pop_01": audio.resolve()}


def test_approved_file_changes_fail_closed_when_paths_are_requested(tmp_path: Path) -> None:
    path, audio, document = _fixture(tmp_path)
    document["assets"][0]["approval"] = {"status": "approved", "reference": "host-review-001"}
    catalog = _write(path, document)
    audio.write_bytes(audio.read_bytes()[:-2] + b"\0\0")
    with pytest.raises(ValueError):
        catalog.approved_asset_paths()


def test_fingerprint_tracks_approval_and_verified_bytes_but_not_json_whitespace(tmp_path: Path) -> None:
    path, audio, document = _fixture(tmp_path)
    pending = _write(path, document).fingerprint
    path.write_text(json.dumps(document, indent=4), encoding="utf-8")
    assert _loader()(path).fingerprint == pending
    document["assets"][0]["approval"] = {"status": "approved", "reference": "host-review-001"}
    approved = _write(path, document).fingerprint
    assert approved != pending
    audio.write_bytes(audio.read_bytes()[:-2] + b"\0\0")
    document["assets"][0]["sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
    assert _write(path, document).fingerprint != approved


@pytest.mark.parametrize(("field", "value"), [
    ("asset_id", "../pop"), ("asset_id", "Pop"), ("asset_id", "póp"),
    ("family", "pop/hit"), ("sha256", "a" * 64), ("sha256", "z" * 64),
    ("sample_rate", 44100), ("sample_rate", True), ("channels", 3),
    ("frames", 95), ("frames", True), ("transient_sample", -1), ("transient_sample", 96),
    ("default_gain_db", -37), ("default_gain_db", -5), ("default_gain_db", True),
    ("default_gain_db", float("nan")), ("default_gain_db", 10**1000), ("provenance", []),
    ("approval", {"status": "generated", "reference": "model"}),
    ("approval", {"status": "approved", "reference": "   "}),
    ("approval", {"status": "approved", "reference": None}),
])
def test_rejects_invalid_or_unverified_entry_fields(tmp_path: Path, field: str, value) -> None:
    path, _, document = _fixture(tmp_path)
    document["assets"][0][field] = value
    with pytest.raises(ValueError):
        _write(path, document)


@pytest.mark.parametrize("value", [
    "../outside.wav", "/outside.wav", "C:/outside.wav", "C:outside.wav",
    "\\\\host\\share\\outside.wav", "cues\\pop.wav", "cues/../cues/pop.wav",
    "cues//pop.wav", "cues/pop.wav:stream", "./cues/pop.wav", "",
])
def test_rejects_noncontained_or_ambiguous_wav_paths(tmp_path: Path, value: str) -> None:
    path, _, document = _fixture(tmp_path)
    document["assets"][0]["path"] = value
    with pytest.raises(ValueError):
        _write(path, document)


def test_rejects_truncated_wav_even_when_hash_is_updated(tmp_path: Path) -> None:
    path, audio, document = _fixture(tmp_path)
    audio.write_bytes(audio.read_bytes()[:-2])
    document["assets"][0]["sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        _write(path, document)


@pytest.mark.parametrize("kind", ["duration", "size", "not_wav", "empty"])
def test_rejects_oversized_long_or_invalid_audio(tmp_path: Path, kind: str) -> None:
    path, audio, document = _fixture(tmp_path)
    entry = document["assets"][0]
    if kind == "duration":
        _wav(audio, frames=48000 * 12 + 1)
        entry["frames"] = 48000 * 12 + 1
    elif kind == "size":
        audio.write_bytes(audio.read_bytes() + b"\0" * (4 * 1024**2))
    elif kind == "empty":
        _wav(audio, frames=0)
        entry["frames"] = 0
    else:
        audio.write_bytes(b"not a WAV")
    entry["sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        _write(path, document)


@pytest.mark.parametrize("kind", ["version", "unknown", "duplicate_id", "count", "json_size", "duplicate_key"])
def test_rejects_invalid_catalog_structure(tmp_path: Path, kind: str) -> None:
    path, _, document = _fixture(tmp_path)
    if kind == "version":
        document["version"] = True
    elif kind == "unknown":
        document["automatic_approval"] = True
    elif kind == "duplicate_id":
        document["assets"].append(dict(document["assets"][0]))
    elif kind == "count":
        document["assets"] = [dict(document["assets"][0], asset_id=f"asset_{i}") for i in range(33)]
    elif kind == "json_size":
        document["assets"][0]["provenance"]["padding"] = "x" * 1024**2
    else:
        path.write_text('{"version":1,"version":1,"assets":[]}', encoding="utf-8")
        with pytest.raises(ValueError):
            _loader()(path)
        return
    with pytest.raises(ValueError):
        _write(path, document)


@pytest.mark.parametrize("directory", [False, True])
def test_rejects_real_file_or_directory_symlinks(tmp_path: Path, directory: bool) -> None:
    path, audio, document = _fixture(tmp_path)
    link = tmp_path / ("alias" if directory else "alias.wav")
    try:
        link.symlink_to(audio.parent if directory else audio, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"OS cannot create symlink fixture: {exc}")
    document["assets"][0]["path"] = "alias/pop.wav" if directory else "alias.wav"
    with pytest.raises(ValueError):
        _write(path, document)


def test_accepts_empty_and_mono_catalogs(tmp_path: Path) -> None:
    path, audio, document = _fixture(tmp_path)
    raw = _wav(audio, channels=1)
    document["assets"][0].update(channels=1, sha256=hashlib.sha256(raw).hexdigest())
    assert _write(path, document).assets[0].channels == 1
    assert _write(path, {"version": 1, "assets": []}).assets == ()
