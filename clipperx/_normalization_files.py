"""Four-path driver-owned file custody for one normalization operation.

This is cooperative HOST custody, not adversarial filesystem or image attestation.
Construction and snapshots are inert; every blocking action is retained and checked.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path, PosixPath, WindowsPath
from typing import TYPE_CHECKING, Protocol

from ._source_host_ownership import Clock, HostedObservationControl, SourceHostJob, Stage
from ._retained_files import _RetainedActions
from .render_runtime import RenderRuntime

if TYPE_CHECKING:
    from ._normalization_transport import PreparedNormalization

FileStamp = tuple[int, int, int]
_SLOTS = ("ffmpeg", "ffprobe", "enabled", "normalized")
_OPERATIONS = ("open", "read", "write", "flush", "stat", "close", "delete")
_PHASES = ("reserved", "in_flight", "returned", "closed", "unresolved")
_RUNTIME_CAP = 268435456
_PCM_CAP = 68736000
_MAX_FILE_BYTES = 268435456


class BinaryFile(Protocol):
    def read(self, size: int) -> bytes: ...
    def write(self, data: memoryview) -> int: ...
    def flush(self) -> None: ...
    def fileno(self) -> int: ...
    def close(self) -> None: ...


def _stamp(value: FileStamp) -> FileStamp:
    if (type(value) is not tuple or len(value) != 3
            or any(type(v) is not int or not 0 <= v <= 2**128 - 1 for v in value)):
        raise ValueError("bounded fstat identity required")
    return value


@dataclass(frozen=True, slots=True)
class FileEvidence:
    slot: str
    stamp: FileStamp
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _stamp(self.stamp)
        if (type(self.slot) is not str or self.slot not in _SLOTS
                or type(self.bytes) is not int or not 0 <= self.bytes <= _MAX_FILE_BYTES
                or type(self.sha256) is not str or len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("invalid file evidence")


@dataclass(frozen=True, slots=True)
class FileActionStatus:
    generation: int
    slot: str
    operation: str
    phase: str
    in_flight: bool
    uncertain: bool
    known_bytes: int
    stamp: FileStamp | None

    def __post_init__(self) -> None:
        if (type(self.generation) is not int or not 1 <= self.generation <= 16
                or type(self.slot) is not str or self.slot not in _SLOTS
                or type(self.operation) is not str or self.operation not in _OPERATIONS
                or type(self.phase) is not str or self.phase not in _PHASES
                or type(self.in_flight) is not bool or type(self.uncertain) is not bool
                or type(self.known_bytes) is not int or not 0 <= self.known_bytes <= _MAX_FILE_BYTES + 1):
            raise ValueError("invalid file action status")
        if self.stamp is not None:
            _stamp(self.stamp)


@dataclass(frozen=True, slots=True)
class NormalizationFileStatus:
    rows: tuple[FileActionStatus, ...]
    enabled_present: bool
    normalized_present: bool
    output_transferred: bool
    cleanup_required: bool
    files_clear: bool

    def __post_init__(self) -> None:
        if (type(self.rows) is not tuple or len(self.rows) > 16
                or any(type(r) is not FileActionStatus for r in self.rows)
                or any(type(b) is not bool for b in (self.enabled_present, self.normalized_present,
                    self.output_transferred, self.cleanup_required, self.files_clear))):
            raise ValueError("invalid normalization file status")


def _open_binary(path: Path, mode: str) -> BinaryFile:
    return path.open(mode, buffering=0)


def _file_stamp(stream: BinaryFile) -> FileStamp:
    value = os.fstat(stream.fileno())
    return value.st_dev, value.st_ino, value.st_size


def _unlink_exact(path: Path) -> None:
    path.unlink()


class _Entry:
    def __init__(self, generation: int, slot: str, path: Path, mode: str) -> None:
        self.row = FileActionStatus(generation, slot, "open", "reserved", False, False, 0, None)
        self.path, self.mode = path, mode
        self.stream: BinaryFile | None = None
        self.closed = False


class _Reader:
    def __init__(self, files: "NormalizationFiles", path: Path) -> None:
        self.files, self.path = files, path
        self.entry: _Entry | None = None

    def __enter__(self) -> "_Reader":
        if self.entry is not None:
            raise ValueError("file context replay")
        self.entry = self.files._acquire(self.path, "rb")
        self.files._stat_entry(self.entry)
        return self

    def read(self, size: int) -> bytes:
        if self.entry is None:
            raise ValueError("file reader not entered")
        return self.files._read(self.entry, size)

    def __exit__(self, *args: object) -> None:
        if self.entry is None:
            raise ValueError("file reader not entered")
        # Preserve a primary parser/hash/IO failure if close also fails.
        if args[0] is not None:
            try:
                self.files._close(self.entry, cleanup=True)
            except Exception:
                pass
        else:
            try:
                self.files._stat_entry(self.entry)
            except Exception:
                try:
                    self.files._close(self.entry, cleanup=True)
                except Exception:
                    pass
                raise
            else:
                self.files._close(self.entry)


class NormalizationFiles(_RetainedActions):
    _file_stage: Stage = "normalizer"

    def __init__(self, job: SourceHostJob, runtime: RenderRuntime, scratch: Path, *, clock: Clock,
                 _open: Callable[[Path, str], BinaryFile] = _open_binary,
                 _stat: Callable[[BinaryFile], FileStamp] = _file_stamp,
                 _unlink: Callable[[Path], None] = _unlink_exact) -> None:
        if type(job) is not SourceHostJob or type(runtime) is not RenderRuntime or clock is not job._clock:
            raise ValueError("same job, runtime and clock required")
        if (type(scratch) not in (WindowsPath, PosixPath)
                or not hasattr(runtime, "ffmpeg") or not hasattr(runtime, "ffprobe")
                or any(type(p) not in (WindowsPath, PosixPath) for p in (runtime.ffmpeg, runtime.ffprobe))):
            raise ValueError("concrete runtime and scratch paths required")
        paths = (runtime.ffmpeg, runtime.ffprobe, scratch / "enabled.f32", scratch / "normalized.f32")
        for path in (*paths, scratch):
            if (not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts
                    or len(str(path).encode("utf-8")) > 32768 or "\0" in str(path)):
                raise ValueError("bounded absolute private paths required")
        names = tuple(str(p).replace("\\", "/").casefold() for p in paths)
        directory = str(scratch).replace("\\", "/").casefold().rstrip("/")
        if len(set(names)) != 4 or any(n == directory or n.startswith(directory + "/") for n in names[:2]):
            raise ValueError("runtime and private scratch paths overlap")
        self._job, self._runtime = job, runtime
        self._paths = dict(zip(_SLOTS, paths, strict=True))
        self._open, self._stat, self._unlink = _open, _stat, _unlink
        self._entries: list[_Entry] = []
        self._present = {"enabled": False, "normalized": False}
        self._identities: dict[str, FileStamp] = {}
        self._active = False
        self._used = False
        self._result: PreparedNormalization | None = None
        self._transferred = self._consumer_live = self._released = False
        self._input_consumers: tuple[HostedObservationControl, ...] = ()

    def _acquire(self, path: Path, mode: str) -> _Entry:
        self._check()
        slot = next((s for s, p in self._paths.items() if p == path), None)
        if slot is None or mode not in ("rb", "xb") or (mode == "xb" and slot not in self._present):
            raise ValueError("file path/mode outside four slots")
        if len(self._entries) >= 16 or any(not e.closed for e in self._entries):
            raise ValueError("file generation or concurrent descriptor limit")
        if mode == "xb" and self._present[slot]:
            raise ValueError("scratch file creation replay")
        if mode == "rb" and slot in self._present and not self._present[slot]:
            raise ValueError("scratch file not owned")
        entry = _Entry(len(self._entries) + 1, slot, path, mode)
        self._entries.append(entry)

        def acquired(value: object) -> None:
            from typing import cast
            entry.stream = cast(BinaryFile, value)
            if mode == "xb":
                self._present[slot] = True

        self._call(entry, "open", lambda: self._open(path, mode), commit=acquired)
        return entry

    def _stat_entry(self, entry: _Entry) -> FileStamp:
        stream = entry.stream
        assert stream is not None

        def observed(value: object) -> None:
            from typing import cast
            stamp = _stamp(cast(FileStamp, value))
            old = self._identities.get(entry.row.slot)
            if old is not None and (stamp[:2] != old[:2] or (entry.mode != "xb" and stamp[2] != old[2])):
                raise ValueError("file stamp changed")
            cap = _RUNTIME_CAP if entry.row.slot in ("ffmpeg", "ffprobe") else _PCM_CAP
            if stamp[2] > cap:
                raise ValueError("file size cap")
            entry.row = replace(entry.row, stamp=stamp)
            self._identities[entry.row.slot] = stamp

        self._call(entry, "stat", lambda: self._stat(stream), commit=observed)
        assert entry.row.stamp is not None
        return entry.row.stamp

    def _read(self, entry: _Entry, size: int) -> bytes:
        if type(size) is not int or not 1 <= size <= 1024 * 1024:
            raise ValueError("bounded read required")
        cap = _RUNTIME_CAP if entry.row.slot in ("ffmpeg", "ffprobe") else _PCM_CAP
        request = min(size, cap - entry.row.known_bytes + 1)
        stream = entry.stream
        assert stream is not None

        def counted(value: object) -> None:
            if type(value) is not bytes or len(value) > request:
                raise ValueError("unknown file read transfer")
            entry.row = replace(entry.row, known_bytes=entry.row.known_bytes + len(value))
            if entry.row.known_bytes > cap:
                raise ValueError("file read overflow")

        result = self._call(entry, "read", lambda: stream.read(request), commit=counted)
        assert type(result) is bytes
        return result

    def _close(self, entry: _Entry, *, cleanup: bool = False) -> None:
        if entry.closed:
            return
        assert entry.stream is not None
        self._call(entry, "close", entry.stream.close,
                   commit=lambda _: setattr(entry, "closed", True), cleanup=cleanup)

    def open_binary(self, path: Path) -> _Reader:
        if path not in (self._paths["ffmpeg"], self._paths["ffprobe"]):
            raise ValueError("runtime reader outside paired paths")
        return _Reader(self, path)

    def _begin(self, job: SourceHostJob, runtime: RenderRuntime) -> None:
        self._check()
        if self._used or job is not self._job or runtime is not self._runtime:
            raise ValueError("single-use exact file context required")
        self._used = True

    def _write(self, slot: str, data: bytes) -> None:
        entry = self._acquire(self._paths[slot], "xb")
        self._stat_entry(entry)
        stream = entry.stream
        assert stream is not None
        offset = 0
        while offset < len(data):
            view = memoryview(data)[offset:offset + 65536]

            def counted(value: object) -> None:
                if type(value) is not int or not 1 <= value <= len(view):
                    raise ValueError("unknown file write transfer")
                entry.row = replace(entry.row, known_bytes=entry.row.known_bytes + value)

            result = self._call(entry, "write", lambda: stream.write(view), commit=counted)
            assert type(result) is int
            offset += result
        self._call(entry, "flush", stream.flush)
        stamp = self._stat_entry(entry)
        if stamp[2] != len(data):
            raise ValueError("written file size mismatch")
        self._close(entry)

    def _verify(self, slot: str, size: int, digest: str) -> FileEvidence:
        reader = _Reader(self, self._paths[slot])
        sha = hashlib.sha256()
        remaining = size
        with reader:
            while remaining:
                chunk = reader.read(min(65536, remaining))
                if not chunk:
                    raise ValueError("short owned PCM file")
                remaining -= len(chunk)
                sha.update(chunk)
            if reader.read(1) != b"":
                raise ValueError("extra owned PCM file bytes")
        if sha.hexdigest() != digest:
            raise ValueError("owned PCM readback hash mismatch")
        assert reader.entry is not None and reader.entry.row.stamp is not None
        return FileEvidence(slot, reader.entry.row.stamp, size, digest)

    def _claim_enabled_input(self, control: HostedObservationControl) -> None:
        self._check()
        if (type(control) is not HostedObservationControl or control._job is not self._job
                or control._kind != "normalization" or not self._present["enabled"]
                or len(self._input_consumers) >= 2 or any(c is control for c in self._input_consumers)):
            raise ValueError("at most two exact normalization input consumers required")
        control._check()
        # Each normalization control admits only one child. Retaining its scope
        # before _launch covers late acquisition and every child/native row.
        self._input_consumers += (control,)

    def _input_consumers_retired(self) -> bool:
        for control in self._input_consumers:
            status = control.ownership_snapshot()
            if (status.phase not in ("retired_ok", "stopped_clean")
                    or not status.window_resources_zero or status.scope_open or status.consumer_active
                    or status.payload_retained or status.lease_outstanding or status.cleanup_required):
                return False
        return True

    def _delete(self, slot: str, *, cleanup: bool = False) -> None:
        if not self._present[slot]:
            return
        if slot == "enabled" and not self._input_consumers_retired():
            raise ValueError("enabled input still has an unresolved normalization consumer")
        entries = [e for e in self._entries if e.row.slot == slot]
        if any(not e.closed or e.row.uncertain for e in entries):
            raise ValueError("file descriptor ownership unresolved")
        entry = entries[-1]
        self._call(entry, "delete", lambda: self._unlink(entry.path),
                   commit=lambda _: self._present.__setitem__(slot, False), cleanup=cleanup)

    def _seal(self, result: "PreparedNormalization") -> None:
        self._check()
        if self._result is not None or not self._present["normalized"]:
            raise ValueError("normalization result seal replay")
        self._result = result

    def _result_check(self, result: "PreparedNormalization") -> None:
        self._check()
        if result is not self._result or self._released or not self._present["normalized"]:
            raise ValueError("exact live returned normalization result required")

    def output_path(self, result: "PreparedNormalization") -> Path:
        self._result_check(result)
        if self._consumer_live:
            raise ValueError("output handoff replay")
        self._consumer_live = self._transferred = True
        return self._paths["normalized"]

    def release_output(self, result: "PreparedNormalization") -> None:
        self._result_check(result)
        if not self._consumer_live:
            raise ValueError("output consumer handoff required")
        self._consumer_live = False
        self._released = True
        self._delete("normalized")

    def snapshot(self) -> NormalizationFileStatus:
        rows = tuple(e.row for e in self._entries)
        unresolved = any(not e.closed or e.row.uncertain or e.row.in_flight for e in self._entries)
        owned = self._present["enabled"] or self._present["normalized"]
        return NormalizationFileStatus(rows, self._present["enabled"], self._present["normalized"],
            self._transferred, unresolved or owned, not unresolved and not owned)

    def retire(self) -> NormalizationFileStatus:
        try:
            self._cleanup_check()
        except Exception:
            return self.snapshot()
        for entry in self._entries:
            if entry.stream is not None and not entry.closed and not entry.row.uncertain and not entry.row.in_flight:
                try:
                    self._close(entry, cleanup=True)
                except Exception:
                    pass
        for slot in ("enabled", "normalized"):
            if slot == "normalized" and self._consumer_live:
                continue
            try:
                self._delete(slot, cleanup=True)
            except Exception:
                pass
        return self.snapshot()
