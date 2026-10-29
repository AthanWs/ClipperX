"""Private original-byte staging; measurement and renderer choice are separate gates."""

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
import hashlib
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import BinaryIO, Iterator, Literal, Mapping, NoReturn, cast, TYPE_CHECKING, TypedDict

from .render_assets import RenderAssetRegistry
if TYPE_CHECKING:
    from ._retained_files import RetainedBinary, RetainedFiles


class _FileOptions(TypedDict, total=False):
    _files: "RetainedFiles"

_IDS = ("headline_impact", "subtitle_bold")
_PINS = {
    "headline_impact": "d8bbdb72e3f930157824b2e59677d3a912d4512c14a948e6656a2e3fb0c935ad",
    "subtitle_bold": "e8f4e3baf6cc35fed6fcce3a540e8b39e8f6cda1d22a28f2ec8f526fef7a43f5",
}
_MAX = 16 * 1024 * 1024
_CHUNK = 64 * 1024
_FontID = Literal["headline_impact", "subtitle_bold"]
_Authority = Literal["approved-originals-v1", "test-fixture"]


def _integer(value: object, *, positive: bool = False) -> None:
    if type(value) is not int or not (1 if positive else 0) <= value < 10 ** 32:
        raise ValueError("Invalid bounded stamp scalar")


def _hash(value: object) -> str:
    if type(value) is not str or len(value) != 64 or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("Expected a 64-hex SHA-256")
    return value.lower()


def _path(value: object) -> Path:
    if not isinstance(value, (Path, str)) or not 1 <= len(str(value)) <= 2048:
        raise ValueError("Invalid bounded path text")
    path = Path(value)
    if len(path.parts) > 32 or not path.is_absolute():
        raise ValueError("Path must be absolute with at most 32 components")
    if os.name == "nt" and (not re.fullmatch(r"[A-Za-z]:", path.drive) or path.root != "\\"):
        raise ValueError("Network/device paths are forbidden")
    for part in path.parts[1:]:
        if (part in (".", "..") or any(ord(char) < 32 for char in part)
                or any(char in part for char in ':<>"|?*')
                or part.endswith((".", " "))
                or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part, re.I)):
            raise ValueError("Special/device/alternate-stream path component")
    return path


def _bounded_items(value: object, *, max_entries: int) -> Iterator[tuple[object, object]]:
    # A proxy can wrap a lying/custom mapping. Its exact built-in dict_items
    # view supplies the actual backing count, without invoking key iterators.
    if type(value) not in (dict, MappingProxyType):
        raise ValueError("Expected an ordinary finite mapping")
    items = cast(Mapping[object, object], value).items()
    if type(items) is not type({}.items()) or len(items) > max_entries:
        raise ValueError(f"Expected finite dict_items with at most {max_entries} entries")
    entry: tuple[object, object]
    for consumed, entry in enumerate(items):
        if consumed >= max_entries:
            raise ValueError(f"Finite mapping consumed more than {max_entries} entries")
        yield entry


def _table(value: object) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, path in _bounded_items(value, max_entries=64):
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
            raise ValueError("Invalid bounded asset ID")
        if not isinstance(path, Path) or len(str(path)) > 2048:
            raise ValueError("Invalid registry path value")
        result[key] = path
    return result


def _registry(value: RenderAssetRegistry) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    if type(value) is not RenderAssetRegistry:
        raise ValueError("Expected RenderAssetRegistry")
    return _table(value.brand_images), _table(value.fonts), _table(value.sfx)


def _policy(expected: Mapping[str, str], authority: _Authority) -> dict[str, str]:
    if type(authority) is not str or authority not in ("approved-originals-v1", "test-fixture"):
        raise ValueError("Unknown identity authority")
    supplied: dict[str, str] = {}
    for key, digest in _bounded_items(expected, max_entries=2):
        if type(key) is not str or key not in _IDS:
            raise ValueError("Expected exactly the two required font hashes")
        supplied[key] = _hash(digest)
    if set(supplied) != set(_IDS):
        raise ValueError("Expected exactly the two required font hashes")
    result = {key: supplied[key] for key in _IDS}
    if authority == "approved-originals-v1" and result != _PINS:
        raise ValueError("Approved authority requires both historical original pins")
    return result


def _key(path: Path) -> tuple[str, ...]:
    return tuple(os.path.normcase(part) for part in path.parts)


def _within(path: Path, parent: Path) -> bool:
    child_parts, parent_parts = _key(path), _key(parent)
    return child_parts[:len(parent_parts)] == parent_parts


@dataclass(frozen=True, slots=True)
class FontFileStamp:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    link_count: int

    def __post_init__(self) -> None:
        for field in fields(self):
            _integer(getattr(self, field.name), positive=field.name in
                     ("device", "inode", "size_bytes", "link_count"))
        if self.size_bytes > _MAX:
            raise ValueError("Font exceeds 16 MiB")


@dataclass(frozen=True, slots=True)
class StagedFont:
    asset_id: _FontID
    original_path: Path
    expected_original_sha256: str
    original_sha256: str
    original_stamp: FontFileStamp
    staged_path: Path
    staged_sha256: str
    staged_stamp: FontFileStamp
    face_index: Literal[0] = 0

    def __post_init__(self) -> None:
        if (type(self.asset_id) is not str or self.asset_id not in _IDS
                or type(self.face_index) is not int or self.face_index != 0):
            raise ValueError("Invalid required font ID/face index")
        for name in ("original_path", "staged_path"):
            object.__setattr__(self, name, _path(getattr(self, name)))
        for name in ("expected_original_sha256", "original_sha256", "staged_sha256"):
            object.__setattr__(self, name, _hash(getattr(self, name)))
        for stamp in (self.original_stamp, self.staged_stamp):
            if type(stamp) is not FontFileStamp:
                raise ValueError("Invalid font file stamp")
            stamp.__post_init__()
        if (self.expected_original_sha256 != self.original_sha256
                or self.original_sha256 != self.staged_sha256
                or self.original_stamp.size_bytes != self.staged_stamp.size_bytes
                or self.staged_stamp.link_count != 1
                or _key(self.original_path) == _key(self.staged_path)
                or _object_id(self.original_stamp) == _object_id(self.staged_stamp)):
            raise ValueError("Inconsistent original/staged evidence")

    @property
    def measurement_path(self) -> Path:
        return self.staged_path

    @property
    def measurement_sha256(self) -> str:
        return self.staged_sha256

    @property
    def measurement_face_index(self) -> Literal[0]:
        return self.face_index


@dataclass(frozen=True, slots=True)
class FontStage:
    stage_dir: Path
    stage_directory_id: tuple[int, int]
    identity_authority: _Authority
    fonts: tuple[StagedFont, StagedFont]
    registry: RenderAssetRegistry
    selected_file_identity_status: Literal["unverified"] = "unverified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_dir", _path(self.stage_dir))
        if type(self.stage_directory_id) not in (tuple, list) or len(self.stage_directory_id) != 2:
            raise ValueError("Invalid stage directory object ID")
        for scalar in self.stage_directory_id:
            _integer(scalar, positive=True)
        object.__setattr__(self, "stage_directory_id", tuple(self.stage_directory_id))
        if type(self.fonts) not in (tuple, list) or len(self.fonts) != 2:
            raise ValueError("Expected exactly two staged fonts")
        object.__setattr__(self, "fonts", tuple(self.fonts))
        if (type(self.selected_file_identity_status) is not str
                or self.selected_file_identity_status != "unverified"):
            raise ValueError("Renderer-selected identity must remain unverified")
        for font in self.fonts:
            if type(font) is not StagedFont:
                raise ValueError("Invalid staged font record")
            # Validate copies; construction must not mutate supplied frozen records.
            if replace(font) != font:
                raise ValueError("Noncanonical staged font record")
        if tuple(font.asset_id for font in self.fonts) != _IDS:
            raise ValueError("Wrong required font order")
        _policy({font.asset_id: font.expected_original_sha256 for font in self.fonts},
                self.identity_authority)
        brands, fonts, sounds = _registry(self.registry)
        required = {font.asset_id: font.staged_path for font in self.fonts}
        if fonts != required:
            raise ValueError("Registry does not match manifest paths")
        for font in self.fonts:
            if (_key(font.staged_path.parent) != _key(self.stage_dir)
                    or font.staged_path.name != f"{font.asset_id}-{font.staged_sha256}.ttf"
                    or _within(self.stage_dir, font.original_path.parent)
                    or _within(font.original_path, self.stage_dir)):
                raise ValueError("Invalid staged containment/source parent")
        _distinct([(font.original_path, font.original_stamp) for font in self.fonts]
                  + [(font.staged_path, font.staged_stamp) for font in self.fonts])
        object.__setattr__(self, "registry", RenderAssetRegistry(brands, fonts, sounds))


def _stamp(value: os.stat_result) -> FontFileStamp:
    # Windows pathname ctime is creation time through Python 3.12. Some 3.12
    # descriptor stats expose metadata-change time there; birthtime gives the same
    # Windows creation-time observation on both paths. Older runtimes use ctime.
    ctime = getattr(value, "st_birthtime_ns", value.st_ctime_ns) if os.name == "nt" else value.st_ctime_ns
    return FontFileStamp(value.st_dev, value.st_ino, value.st_size,
                         value.st_mtime_ns, ctime, value.st_nlink)


def _object_id(value: FontFileStamp) -> tuple[int, int]:
    return value.device, value.inode


def _distinct(files: list[tuple[Path, FontFileStamp]]) -> None:
    if len({_key(path) for path, _ in files}) != len(files) or len(
            {_object_id(stamp) for _, stamp in files}) != len(files):
        raise ValueError("Font path/object alias")


def _checked_path(value: Path, *, directory: bool = False,
                  _files: "RetainedFiles | None" = None) -> tuple[Path, os.stat_result]:
    path = _path(value)
    if _files is not None:
        for declared in (*reversed(path.parents), path):
            _files._register_checked(declared)
    current = Path(path.anchor)
    observed = current.lstat() if _files is None else cast(os.stat_result, _files.path_action("lstat", current))
    for index, part in enumerate(path.parts):
        if index:
            current /= part
            observed = current.lstat() if _files is None else cast(os.stat_result, _files.path_action("lstat", current))
        if (stat.S_ISLNK(observed.st_mode)
                or getattr(observed, "st_file_attributes", 0) & 0x400):
            raise ValueError("Symlink/reparse path component")
        want_directory = index < len(path.parts) - 1 or directory
        if not (stat.S_ISDIR(observed.st_mode) if want_directory else stat.S_ISREG(observed.st_mode)):
            raise ValueError("Nonregular font or nondirectory component")
    resolved = _path(path.resolve(strict=True) if _files is None else _files.resolve_path(path))
    if _key(path) != _key(resolved):
        raise ValueError("Path resolution changed checked components")
    return resolved, observed


def _directory_id(path: Path, *, _files: "RetainedFiles | None" = None) -> tuple[int, int]:
    _, observed = _checked_path(path, directory=True, **({} if _files is None else {"_files": _files}))
    for scalar in (observed.st_dev, observed.st_ino):
        _integer(scalar, positive=True)
    return observed.st_dev, observed.st_ino


@contextmanager
def _owned_file(path: Path, mode: str, *, _files: "RetainedFiles | None" = None
                ) -> Iterator["BinaryIO | RetainedBinary"]:
    file = (cast(BinaryIO, path.open(mode)) if _files is None else
            _files.open_file(path, cast(Literal["rb", "xb"], mode)).__enter__())
    try:
        yield file
    except BaseException as primary:
        try:
            file.close()
        except BaseException as cleanup:
            detail = f"close also failed; resource retirement uncertain: {cleanup}"
            primary.args = (*primary.args, detail)
            if isinstance(primary, OSError) and primary.errno is not None:
                primary.strerror = f"{primary.strerror}; {detail}"
        raise
    else:
        try:
            file.close()
        except (OSError, ValueError) as exc:
            _context_error(exc, "close; resource retirement uncertain", "owned-file", path)


def _read_checked(path: Path, *, max_bytes: int, _files: "RetainedFiles | None" = None) -> tuple[bytes, FontFileStamp]:
    if type(max_bytes) is not int or not 1 <= max_bytes <= _MAX:
        raise ValueError("Read cap must be a positive integer <=16 MiB")
    options: _FileOptions = {} if _files is None else {"_files": _files}
    path, observed = _checked_path(path, **options)
    before = _stamp(observed)
    raw_path_ctime = observed.st_ctime_ns
    _integer(raw_path_ctime)
    if before.size_bytes > max_bytes:
        raise ValueError("Font stat size exceeds read cap")
    with _owned_file(path, "rb", **options) as file:
        opened_stat = os.fstat(file.fileno()) if _files is None else cast("RetainedBinary", file).fstat()
        opened = _stamp(opened_stat)
        _integer(opened_stat.st_ctime_ns)
        if opened != before:
            raise ValueError("Font changed before read")
        data = bytearray()
        while len(data) <= max_bytes:
            chunk = file.read(min(_CHUNK, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after_stat = os.fstat(file.fileno()) if _files is None else cast("RetainedBinary", file).fstat()
        after = _stamp(after_stat)
        _integer(after_stat.st_ctime_ns)
        if opened_stat.st_ctime_ns != after_stat.st_ctime_ns:
            raise ValueError("Descriptor ctime changed during read")
    final_stat = path.lstat() if _files is None else cast(os.stat_result, _files.path_action("lstat", path))
    _integer(final_stat.st_ctime_ns)
    if (len(data) > max_bytes or len(data) != before.size_bytes
            or opened != after or after != _stamp(final_stat)
            or raw_path_ctime != final_stat.st_ctime_ns):
        raise ValueError("Changed/excessive font or stat/read count disagreement")
    return bytes(data), after


def _copy_exclusive(path: Path, data: bytes, *, _files: "RetainedFiles | None" = None) -> None:
    with _owned_file(path, "xb", **({} if _files is None else {"_files": _files})) as file:
        for offset in range(0, len(data), _CHUNK):
            chunk = data[offset:offset + _CHUNK]
            if (cast(BinaryIO, file).write(chunk) if _files is None else
                    cast("RetainedBinary", file).write(memoryview(chunk))) != len(chunk):
                raise OSError("Incomplete font write")
        file.flush()
        if _files is None:
            os.fsync(file.fileno())
        else:
            cast("RetainedBinary", file).fsync()


def stage_original_fonts(registry: RenderAssetRegistry, *, stage_dir: Path,
                         _files: "RetainedFiles | None" = None) -> FontStage:
    """Copy the two historically pinned originals into a fresh host-owned child.

    The host must ensure private local ownership and retain files for consumers.
    Byte integrity does not qualify font style, glyphs, or renderer selection.
    """
    return _stage_fonts(registry, stage_dir=stage_dir, expected_sha256=_PINS,
                        identity_authority="approved-originals-v1", **({} if _files is None else {"_files": _files}))


def _stage_fonts(registry: RenderAssetRegistry, *, stage_dir: Path,
                 expected_sha256: Mapping[str, str], identity_authority: _Authority,
                 _files: "RetainedFiles | None" = None) -> FontStage:
    phase, asset_id = "preflight", "required-fonts"
    try:
        stage_dir = _path(stage_dir)
        options: _FileOptions = {} if _files is None else {"_files": _files}
        expected = _policy(expected_sha256, identity_authority)
        brands, registered, sounds = _registry(registry)
        if not all(key in registered for key in _IDS):
            raise ValueError("Missing required font ID")
        parent, _ = _checked_path(stage_dir.parent, directory=True, **options)
        stage_dir = parent / stage_dir.name
        if _files is not None:
            _files.register(stage_dir, role="directory", owned=True, max_bytes=0)
        source_paths: dict[str, Path] = {}
        source_stamps: dict[str, FontFileStamp] = {}
        outputs: dict[str, Path] = {}
        for asset_id in _IDS:
            path, observed = _checked_path(registered[asset_id], **options)
            source_paths[asset_id] = path
            source_stamps[asset_id] = _stamp(observed)
            if _within(stage_dir, path.parent) or _within(path, stage_dir):
                raise ValueError("Destination overlaps original parent")
            outputs[asset_id] = _path(stage_dir / f"{asset_id}-{expected[asset_id]}.ttf")
            if _files is not None:
                _files.register(outputs[asset_id], role="font", owned=True, max_bytes=_MAX)
        _distinct(list((source_paths[key], source_stamps[key]) for key in _IDS))
        try:
            stage_dir.lstat() if _files is None else _files.path_action("lstat", stage_dir)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("Destination already exists")
        originals: dict[str, tuple[bytes, FontFileStamp, str]] = {}
        for asset_id in _IDS:
            data, stamp = _read_checked(source_paths[asset_id], max_bytes=_MAX, **options)
            digest = hashlib.sha256(data).hexdigest()
            if stamp != source_stamps[asset_id] or digest != expected[asset_id]:
                raise ValueError("Original identity/stamp mismatch")
            originals[asset_id] = (data, stamp, digest)
        phase, asset_id = "mkdir", "required-fonts"
        stage_dir.mkdir(exist_ok=False) if _files is None else _files.path_action("mkdir", stage_dir)
        directory_id = _directory_id(stage_dir, **options)
        phase = "copy"
        for asset_id in _IDS:
            _copy_exclusive(outputs[asset_id], originals[asset_id][0], **options)
        phase = "readback"
        if _directory_id(stage_dir, **options) != directory_id:
            raise ValueError("Stage directory identity changed")
        fonts: list[StagedFont] = []
        for asset_id in _IDS:
            copied, stamp = _read_checked(outputs[asset_id], max_bytes=_MAX, **options)
            digest = hashlib.sha256(copied).hexdigest()
            original_data, original_stamp, original_digest = originals[asset_id]
            if len(copied) != len(original_data) or digest != original_digest:
                raise ValueError("Staged byte identity mismatch")
            fonts.append(StagedFont(cast(_FontID, asset_id), source_paths[asset_id],
                                    expected[asset_id], original_digest, original_stamp,
                                    outputs[asset_id], digest, stamp))
            del copied
        _distinct([(source_paths[key], originals[key][1]) for key in _IDS]
                  + [(font.staged_path, font.staged_stamp) for font in fonts])
        phase = "source-recheck"
        for asset_id in _IDS:
            final, final_stamp = _read_checked(source_paths[asset_id], max_bytes=_MAX, **options)
            if (final_stamp != originals[asset_id][1]
                    or hashlib.sha256(final).hexdigest() != originals[asset_id][2]):
                raise ValueError("Original changed after copy")
            del final
        phase, asset_id = "directory-recheck", "required-fonts"
        if _directory_id(stage_dir, **options) != directory_id:
            raise ValueError("Stage directory identity changed")
        staged_registry = RenderAssetRegistry(brands, {font.asset_id: font.staged_path for font in fonts},
                                              sounds)
        return FontStage(stage_dir, directory_id, identity_authority,
                         (fonts[0], fonts[1]), staged_registry)
    except (OSError, ValueError) as exc:
        _context_error(exc, phase, asset_id, stage_dir)


def verify_font_stage(stage: FontStage, *, _files: "RetainedFiles | None" = None) -> None:
    """Check recorded source/stage bytes and observations without repairing them."""
    asset_id = "required-fonts"
    try:
        if type(stage) is not FontStage:
            raise ValueError("Expected FontStage")
        # Reconstruct bounded evidence to validate bypassed frozen records, without
        # changing the caller's manifest or replacing its usable registry.
        validated = replace(stage)
        if (validated != stage or type(stage.fonts) is not tuple
                or type(stage.stage_directory_id) is not tuple
                or any(type(table) is not MappingProxyType for table in
                       (stage.registry.brand_images, stage.registry.fonts, stage.registry.sfx))):
            raise ValueError("Noncanonical/mutable stage record")
        options: _FileOptions = {} if _files is None else {"_files": _files}
        if _directory_id(stage.stage_dir, **options) != stage.stage_directory_id:
            raise ValueError("Stage directory identity changed")
        observations: list[tuple[Path, FontFileStamp]] = []
        for font in stage.fonts:
            asset_id = font.asset_id
            for path, digest, stamp in ((font.original_path, font.original_sha256, font.original_stamp),
                                        (font.staged_path, font.staged_sha256, font.staged_stamp)):
                data, observed = _read_checked(path, max_bytes=_MAX, **options)
                if observed != stamp or hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError("Font integrity mismatch")
                observations.append((path, observed))
                del data
        _distinct(observations)
        if _directory_id(stage.stage_dir, **options) != stage.stage_directory_id:
            raise ValueError("Stage directory identity changed")
    except (OSError, ValueError) as exc:
        _context_error(exc, "verify", asset_id, getattr(stage, "stage_dir", "invalid-stage"))


def _context_error(exc: OSError | ValueError, phase: str, asset_id: str, stage_dir: object) -> NoReturn:
    message = f"{phase}/{asset_id}/stage_dir={stage_dir}: {exc}"
    if isinstance(exc, OSError):
        raise OSError(exc.errno, message) from exc
    raise ValueError(message) from exc
