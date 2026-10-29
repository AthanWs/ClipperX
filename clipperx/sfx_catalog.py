"""Version-1 local host SFX catalogs; loading never grants editorial approval.

JSON: {"version": 1, "assets": [...]}. Each asset's ``path`` is a relative WAV
path under the catalog directory. Provenance is descriptive JSON, not approval.
``transient_sample`` is a per-channel frame index, not an interleaved offset.
The fingerprint identifies canonical metadata backed by verified WAV bytes.
Returned paths are rechecked when offered; consumers must retain/recheck files
when rendering, since a path is not a permanent handle to immutable bytes.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from types import MappingProxyType
from typing import Any
import wave

_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")  # Same host-ID contract as render_assets.
_SHA = re.compile(r"^[0-9a-f]{64}$")
_JSON_CAP = 1024**2
_WAV_CAP = 4 * 1024**2
_FIELDS = {"asset_id", "family", "path", "sha256", "sample_rate", "channels", "frames",
           "transient_sample", "default_gain_db", "approval", "provenance"}


@dataclass(frozen=True, slots=True)
class SfxApproval:
    status: str
    reference: str


@dataclass(frozen=True, slots=True)
class SfxAsset:
    asset_id: str
    family: str
    relative_path: str
    path: Path
    sha256: str
    sample_rate: int
    channels: int
    frames: int
    transient_sample: int
    default_gain_db: float
    approval: SfxApproval
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SfxCatalog:
    assets: tuple[SfxAsset, ...]
    fingerprint: str
    _root: Path

    def approved_asset_paths(self) -> dict[str, Path]:
        """Offer only explicit host approvals, failing closed if files changed."""
        return {asset.asset_id: _verify_wav(self._root, asset) for asset in self.assets
                if asset.approval.status == "approved" and asset.approval.reference.strip()}


def _fields(value: object, names: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("SFX catalog object has missing or unknown fields")
    return value


def _no_links(path: Path) -> None:
    for part in (*reversed(path.parents), path):
        info = part.lstat()
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("SFX catalog paths cannot contain symlinks or reparse points")


def _read_regular(path: Path, cap: int) -> bytes:
    _no_links(path)
    with path.open("rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > cap:
            raise ValueError("SFX catalog file is not regular or exceeds its byte cap")
        data = source.read(cap + 1)
        after = os.fstat(source.fileno())
    _no_links(path)
    named = path.stat()
    if (len(data) > cap or len(data) != info.st_size
            or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino)):
        raise ValueError("SFX catalog file changed while being verified")
    return data


def _relative_wav(root: Path, value: object) -> Path:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in ("", ".", "..") for part in value.split("/"))
            or PurePosixPath(value).suffix.lower() != ".wav"):
        raise ValueError("SFX path must be an unambiguous relative WAV path")
    path = root.joinpath(*value.split("/"))
    _no_links(path)
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("SFX WAV path escapes the catalog directory")
    return path


def _verify_wav(root: Path, asset: SfxAsset) -> Path:
    try:
        path = _relative_wav(root, asset.relative_path)
        data = _read_regular(path, _WAV_CAP)
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise ValueError("SFX WAV SHA-256 does not match verified catalog metadata")
        with wave.open(BytesIO(data), "rb") as audio:
            if (audio.getcomptype() != "NONE" or audio.getframerate() != 48000
                    or audio.getnchannels() != asset.channels or audio.getnframes() != asset.frames):
                raise ValueError("SFX WAV metadata does not match catalog frames/rate/channels")
            expected = asset.frames * asset.channels * audio.getsampwidth()
            if len(audio.readframes(asset.frames + 1)) != expected:
                raise ValueError("SFX WAV sample data is truncated")
        return path
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"SFX WAV is not safely readable: {exc}") from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _asset(root: Path, value: object) -> SfxAsset:
    row = _fields(value, _FIELDS)
    for name in ("asset_id", "family"):
        if not isinstance(row[name], str) or not _ID.fullmatch(row[name]):
            raise ValueError("SFX asset ID and family require ASCII host identifiers")
    if not isinstance(row["sha256"], str) or not _SHA.fullmatch(row["sha256"]):
        raise ValueError("SFX asset requires a lowercase SHA-256")
    for name in ("sample_rate", "channels", "frames", "transient_sample"):
        if type(row[name]) is not int:
            raise ValueError(f"SFX {name} must be an integer")
    if (row["sample_rate"] != 48000 or row["channels"] not in (1, 2)
            or not 0 < row["frames"] <= 48000 * 12
            or not 0 <= row["transient_sample"] < row["frames"]):
        raise ValueError("SFX WAV rate/channels/duration/transient outside catalog limits")
    gain = row["default_gain_db"]
    if type(gain) not in (int, float) or not -36 <= gain <= -6 or not math.isfinite(gain):
        raise ValueError("SFX default gain must be finite and between -36 and -6 dB")
    approval = _fields(row["approval"], {"status", "reference"})
    if (approval["status"] not in ("pending", "approved")
            or not isinstance(approval["reference"], str)
            or (approval["status"] == "approved" and not approval["reference"].strip())):
        raise ValueError("Approved SFX requires an explicit nonempty host approval reference")
    if not isinstance(row["provenance"], dict):
        raise ValueError("SFX provenance must be a JSON object")
    path = _relative_wav(root, row["path"])
    asset = SfxAsset(row["asset_id"], row["family"], row["path"], path, row["sha256"],
                     row["sample_rate"], row["channels"], row["frames"], row["transient_sample"],
                     float(gain), SfxApproval(approval["status"], approval["reference"]),
                     _freeze(row["provenance"]))
    _verify_wav(root, asset)
    return asset


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("SFX JSON cannot contain duplicate keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"SFX JSON cannot contain {value}")


def load_sfx_catalog(path: Path | str) -> SfxCatalog:
    """Read a bounded local catalog and verify every pending/approved WAV."""
    try:
        catalog_path = Path(os.path.abspath(path))
        data = _read_regular(catalog_path, _JSON_CAP)
        document = _fields(json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                                     parse_constant=_reject_constant), {"version", "assets"})
        if type(document["version"]) is not int or document["version"] != 1:
            raise ValueError("Unsupported SFX catalog version")
        rows = document["assets"]
        if not isinstance(rows, list) or len(rows) > 32:
            raise ValueError("SFX catalog requires at most 32 assets")
        root = catalog_path.parent.resolve(strict=True)
        assets = tuple(_asset(root, row) for row in rows)
        if len({asset.asset_id for asset in assets}) != len(assets):
            raise ValueError("SFX catalog asset IDs must be unique")
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False).encode("utf-8")
        return SfxCatalog(assets, hashlib.sha256(canonical).hexdigest(), root)
    except (OSError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"SFX catalog is not safely readable: {exc}") from exc
