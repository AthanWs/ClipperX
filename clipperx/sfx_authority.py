"""Explicit approved-SFX snapshots, dependency checks and conservative cue gates."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING

from .sfx_catalog import _read_regular, load_sfx_catalog

if TYPE_CHECKING:
    from .visual_contracts import VisualCompositionPlan

_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ROW_FIELDS = {"asset_id", "family", "path", "sha256", "bytes", "sample_rate", "channels",
               "frames", "transient_sample", "default_gain_db", "approval_reference"}


def default_sfx_catalog_path() -> Path:
    from .local_paths import paths
    return paths.base_dir / "assets/sfx/catalog.json"


def capture_sfx_authority(path: Path | None = None) -> dict[str, Any] | None:
    """Missing catalog grants nothing; invalid/revoked bytes are never enabled."""
    path = Path(path) if path is not None else default_sfx_catalog_path()
    if not path.exists():
        return None
    raw = _read_regular(path.absolute(), 1024**2)
    catalog = load_sfx_catalog(path)
    approved = catalog.approved_asset_paths()
    rows = []
    for asset in catalog.assets:
        if asset.asset_id not in approved:
            continue
        data = _read_regular(asset.path, 4 * 1024**2)
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise ValueError("Approved SFX changed during capture")
        rows.append(dict(asset_id=asset.asset_id, family=asset.family, path=str(asset.path),
            sha256=asset.sha256, bytes=len(data), sample_rate=48000, channels=asset.channels,
            frames=asset.frames, transient_sample=asset.transient_sample,
            default_gain_db=asset.default_gain_db, approval_reference=asset.approval.reference))
    if _read_regular(path.absolute(), 1024**2) != raw:
        raise ValueError("SFX catalog changed during capture")
    return {"version": 1, "catalog": {"path": str(path.resolve()), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "fingerprint": catalog.fingerprint},
            "assets": sorted(rows, key=lambda row: row["asset_id"])}


def _object(value: Any, fields: set[str]) -> None:
    if type(value) is not dict or set(value) != fields:
        raise ValueError("Invalid captured SFX authority fields")


def _hash(value: Any) -> None:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ValueError("Invalid captured SFX SHA-256")


def _absolute(value: Any) -> Path:
    if (type(value) is not str or not value or len(value) > 4000 or "\0" in value
            or not Path(value).is_absolute() or str(Path(value)) != value
            or ".." in Path(value).parts):
        raise ValueError("Invalid captured SFX absolute path")
    return Path(value)


def validate_sfx_authority(value: Any, *, check_files: bool = True) -> None:
    """Static REVIEW checks retain history; live/READY checks detect revocation."""
    _object(value, {"version", "catalog", "assets"})
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unsupported captured SFX authority version")
    catalog = value["catalog"]
    _object(catalog, {"path", "bytes", "sha256", "fingerprint"})
    root = _absolute(catalog["path"]).parent
    if type(catalog["bytes"]) is not int or not 0 < catalog["bytes"] <= 1024**2:
        raise ValueError("Invalid captured SFX catalog byte count")
    _hash(catalog["sha256"])
    _hash(catalog["fingerprint"])
    if type(value["assets"]) is not list or len(value["assets"]) > 32:
        raise ValueError("Invalid captured SFX asset count")
    ids = []
    for row in value["assets"]:
        _object(row, _ROW_FIELDS)
        for field in ("asset_id", "family"):
            if type(row[field]) is not str or not _ID.fullmatch(row[field]):
                raise ValueError("Invalid captured SFX asset identifier")
        ids.append(row["asset_id"])
        if not _absolute(row["path"]).is_relative_to(root):
            raise ValueError("Captured SFX asset escapes catalog")
        _hash(row["sha256"])
        for name in ("bytes", "sample_rate", "channels", "frames", "transient_sample"):
            if type(row[name]) is not int:
                raise ValueError("Invalid captured SFX integer metadata")
        if (not 0 < row["bytes"] <= 4 * 1024**2 or row["sample_rate"] != 48000
                or row["channels"] not in (1, 2) or not 0 < row["frames"] <= 576000
                or not 0 <= row["transient_sample"] < row["frames"]):
            raise ValueError("Invalid captured SFX format/duration")
        gain = row["default_gain_db"]
        if type(gain) not in (int, float) or not -36 <= gain <= -6 or not math.isfinite(gain):
            raise ValueError("Invalid captured SFX gain")
        if type(row["approval_reference"]) is not str or not row["approval_reference"].strip():
            raise ValueError("Captured SFX lacks host approval reference")
    if ids != sorted(set(ids)):
        raise ValueError("Captured SFX IDs must be unique and sorted")
    if check_files and capture_sfx_authority(Path(catalog["path"])) != value:
        raise ValueError("SFX catalog/approval/bytes changed or were revoked")


def source_audio_channels(origin: dict[str, Any]) -> int | None:
    """The renderer uses the first audio stream; do not invent channel conversion."""
    for stream in origin.get("streams", []):
        if stream.get("kind") == "audio":
            channels = stream.get("channels")
            return channels if type(channels) is int and channels > 0 else None
    return None


def sfx_offers(value: dict[str, Any] | None, *, source_channels: int | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    validate_sfx_authority(value, check_files=False)
    return [dict(asset_id=row["asset_id"], family=row["family"],
                 duration_seconds=row["frames"] / 48000,
                 transient_offset_seconds=row["transient_sample"] / 48000,
                 default_gain_db=min(-18, row["default_gain_db"]), channels=row["channels"])
            for row in value["assets"] if row["channels"] == source_channels]


def approved_sfx_paths() -> dict[str, Path]:
    captured = capture_sfx_authority()
    return {row["asset_id"]: Path(row["path"]) for row in captured["assets"]} if captured else {}


def validate_sfx_cues(plan: VisualCompositionPlan, authority: dict[str, Any] | None, *,
                      source_channels: int | None, check_files: bool = True) -> None:
    cues = [(shot, cue) for shot in plan.shots for cue in shot.audio.cues]
    if authority is not None:
        validate_sfx_authority(authority, check_files=check_files)
    if not cues:
        return
    if authority is None:
        raise ValueError("No SFX cue assets are authorized for this legacy host context")
    available = {row["asset_id"]: row for row in authority["assets"]
                 if row["channels"] == source_channels}
    if len(cues) > 2:
        raise ValueError("SFX limit is two cues per clip")
    starts = []
    for shot, cue in cues:
        asset = available.get(cue.asset_id)
        if asset is None:
            raise ValueError("Unknown, unoffered or channel-incompatible SFX asset")
        if (not -36 <= cue.gain_db <= min(-18, asset["default_gain_db"]) or cue.duck_source_db != 0
                or not math.isclose((cue.end - cue.start) * 48000, asset["frames"],
                                    rel_tol=0, abs_tol=1e-7)
                or not 0 <= cue.start < cue.end <= shot.output.duration):
            raise ValueError("SFX must retain exact asset duration, safe gain and original source audio")
        # This rejects a sound label standing alone; semantic grounding still
        # depends on the existing shot reason/local evidence, not a generated fact.
        words = re.findall(r"[a-záéíóúñ]+", shot.reason.lower())
        if (not shot.evidence_ids or not words
                or set(words) <= {"sfx", "whoosh", "sound", "sonido", "audio", "effect", "efecto"}):
            raise ValueError("SFX requires an existing source-grounded shot reason and evidence")
        starts.append(shot.output.start + cue.start)
    if any(b - a < 5 - 1e-9 for a, b in zip(sorted(starts), sorted(starts)[1:])):
        raise ValueError("SFX cues must start at least five seconds apart")
