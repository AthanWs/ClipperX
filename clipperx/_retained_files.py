"""Finite, registered-driver file custody; no workers or path discovery."""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
import os
import shutil
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Literal, Protocol, cast, ContextManager

from ._source_host_ownership import Clock, SourceHostJob, Stage


class BinaryFile(Protocol):
    def read(self, size: int) -> bytes: ...
    def write(self, data: memoryview) -> int: ...
    def flush(self) -> None: ...
    def fileno(self) -> int: ...
    def close(self) -> None: ...


class _ActionEntry(Protocol):
    row: Any
    closed: bool


class _RetainedActions:
    """The shared reserve/commit/check order, independent of adapter limits."""

    _job: SourceHostJob
    _active: bool
    _file_stage: Stage = "driver"

    def _check(self) -> None:
        self._job.check()
        control = self._job._control
        if control is not None and control._scope:
            control._check()

    def _cleanup_check(self) -> None:
        if self._job._current_worker() is not self._job._driver:
            raise ValueError("file cleanup belongs to original driver")
        if self._job._failure is None:
            self._check()
        elif self._job._now() >= self._job._failure.cleanup_deadline_ns:
            raise ValueError("original cleanup deadline expired")

    def _call(self, entry: _ActionEntry, operation: str, action: Callable[[], object], *,
              commit: Callable[[object], None] | None = None, cleanup: bool = False) -> object:
        check = self._cleanup_check if cleanup else self._check
        check()
        if self._active or entry.row.uncertain or entry.row.in_flight:
            raise ValueError("one certain file action required")
        context = (self._job._driver_cleanup_io() if cleanup and operation in ("close", "delete")
                   else nullcontext())
        with context:
            return self._call_admitted(entry, operation, action, commit=commit, check=check)

    def _call_admitted(self, entry: _ActionEntry, operation: str, action: Callable[[], object], *,
                       commit: Callable[[object], None] | None, check: Callable[[], None]) -> object:
        self._active = True
        entry.row = replace(entry.row, operation=operation, phase="in_flight", in_flight=True)
        try:
            try:
                result = action()
                if commit is not None:
                    commit(result)
                entry.row = replace(entry.row, phase="closed" if entry.closed else "returned", in_flight=False)
            except Exception:
                entry.row = replace(entry.row, phase="unresolved", in_flight=False, uncertain=True)
                if self._job._failure is None:
                    self._job.fail("io", self._file_stage)
                raise
            check()
            return result
        finally:
            self._active = False


_ROLES = ("original", "runtime", "checked", "directory", "raw", "pcm", "asset", "font",
          "ass", "graph", "sidecar", "candidate", "temporary")
_READONLY = ("original", "runtime", "checked")
_MAX_BYTES = 2**48


def _bounded(value: int, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("bounded plain integer required")


def _path(path: Path) -> None:
    if (type(path) not in (WindowsPath, PosixPath) or not path.is_absolute()
            or ".." in path.parts or not 1 <= len(str(path)) <= 2048 or "\0" in str(path)):
        raise ValueError("bounded concrete absolute path required")


def _open(path: Path, mode: str) -> BinaryFile:
    return path.open(mode, buffering=0)


def _stat(file: BinaryFile) -> os.stat_result:
    return os.fstat(file.fileno())


def _path_action(operation: str, path: Path, destination: Path | None) -> object:
    if operation == "resolve":
        return path.resolve(strict=True)
    if operation == "stat":
        return path.stat()
    if operation == "lstat":
        return path.lstat()
    if operation == "disk_usage":
        return shutil.disk_usage(path)
    if operation == "mkdir":
        path.mkdir(exist_ok=False)
        return None
    if operation == "replace" and destination is not None:
        return path.replace(destination)
    if operation == "unlink":
        path.unlink()
        return None
    if operation == "rmdir":
        path.rmdir()
        return None
    raise ValueError("unknown path action")


@dataclass(frozen=True, slots=True)
class RetainedActionStatus:
    generation: int
    path: str
    operation: str
    phase: str
    in_flight: bool = False
    uncertain: bool = False
    known_bytes: int = 0
    destination: str | None = None

    def __post_init__(self) -> None:
        _bounded(self.generation, 1, 262144)
        for path in (self.path, self.destination):
            if path is not None and (type(path) is not str or not 1 <= len(path) <= 2048 or "\0" in path):
                raise ValueError("bounded detached path required")
        if self.operation not in ("open", "read", "write", "flush", "fileno", "fstat", "fsync", "close",
                "resolve", "stat", "lstat", "mkdir", "replace", "delete", "disk_usage"):
            raise ValueError("finite file operation required")
        if self.phase not in ("reserved", "in_flight", "returned", "closed", "unresolved"):
            raise ValueError("finite file phase required")
        if type(self.in_flight) is not bool or type(self.uncertain) is not bool:
            raise ValueError("plain file status flags required")
        _bounded(self.known_bytes, 0, _MAX_BYTES + 1)


@dataclass(frozen=True, slots=True)
class RetainedPathStatus:
    path: str
    role: str
    owned: bool
    present: bool
    settled_generations: int
    known_bytes: int
    max_bytes: int
    size_bytes: int | None
    retained: bool

    def __post_init__(self) -> None:
        if type(self.path) is not str or not 1 <= len(self.path) <= 2048 or self.role not in _ROLES:
            raise ValueError("bounded registered path metadata required")
        if any(type(flag) is not bool for flag in (self.owned, self.present, self.retained)):
            raise ValueError("plain registered path flags required")
        _bounded(self.settled_generations, 0, 262144)
        _bounded(self.known_bytes, 0, 262144 * (_MAX_BYTES + 1))
        _bounded(self.max_bytes, 0, _MAX_BYTES)
        if self.size_bytes is not None:
            _bounded(self.size_bytes, 0, 2**63 - 1)


@dataclass(frozen=True, slots=True)
class RetainedFilesStatus:
    rows: tuple[RetainedActionStatus, ...]
    paths: tuple[RetainedPathStatus, ...]
    consumers: tuple[int, ...]
    generations: int
    cleanup_required: bool
    files_clear: bool

    def __post_init__(self) -> None:
        for values, cap, cls in ((self.rows, 16, RetainedActionStatus), (self.paths, 32768, RetainedPathStatus)):
            if type(values) is not tuple or len(values) > cap or any(type(value) is not cls for value in values):
                raise ValueError("bounded frozen file records required")
        if type(self.consumers) is not tuple or len(self.consumers) > 16:
            raise ValueError("bounded consumer tuple required")
        for token in self.consumers:
            _bounded(token, 1, 262144)
        _bounded(self.generations, 0, 262144)
        if type(self.cleanup_required) is not bool or type(self.files_clear) is not bool:
            raise ValueError("plain file cleanup flags required")


@dataclass(slots=True)
class _Registration:
    role: str
    owned: bool
    cap: int
    present: bool = False
    settled: int = 0
    known: int = 0
    size: int | None = None
    retained: bool = False


class _Entry:
    def __init__(self, generation: int, path: Path, mode: str) -> None:
        self.path, self.mode = path, mode
        self.row = RetainedActionStatus(generation, str(path), "open", "reserved")
        self.stream: BinaryFile | None = None
        self.closed = False


class RetainedBinary:
    def __init__(self, files: "RetainedFiles", path: Path, mode: str) -> None:
        self._files, self._path, self._mode = files, path, mode
        self._entry: _Entry | None = None

    def __enter__(self) -> "RetainedBinary":
        if self._entry is not None:
            raise ValueError("file context replay")
        self._entry = self._files._acquire(self._path, self._mode)
        self.fstat()
        return self

    def _live(self) -> tuple[_Entry, BinaryFile]:
        entry = self._entry
        if entry is None or entry.stream is None or entry.closed:
            raise ValueError("live retained file required")
        return entry, entry.stream

    def read(self, size: int) -> bytes:
        _bounded(size, 1, 1024**2)
        entry, stream = self._live()
        if entry.mode != "rb":
            raise ValueError("read mode required")
        cap = self._files._paths[entry.path].cap
        request = min(size, cap - entry.row.known_bytes + 1)
        def commit(value: object) -> None:
            if type(value) is not bytes or len(value) > request:
                raise ValueError("unknown file read transfer")
            entry.row = replace(entry.row, known_bytes=entry.row.known_bytes + len(value))
            if entry.row.known_bytes > cap:
                raise ValueError("file read cap")
        return cast(bytes, self._files._call(entry, "read", lambda: stream.read(request), commit=commit))

    def write(self, data: memoryview) -> int:
        entry, stream = self._live()
        if not isinstance(data, (bytes, memoryview)) or not 1 <= len(data) <= 65536 or entry.mode != "xb":
            raise ValueError("bounded write in exclusive mode required")
        if entry.row.known_bytes + len(data) > self._files._paths[entry.path].cap:
            raise ValueError("file write cap")
        def commit(value: object) -> None:
            if type(value) is not int or not 1 <= value <= len(data):
                raise ValueError("unknown file write transfer")
            entry.row = replace(entry.row, known_bytes=entry.row.known_bytes + value)
        return cast(int, self._files._call(entry, "write", lambda: stream.write(memoryview(data)), commit=commit))

    def flush(self) -> None:
        entry, stream = self._live()
        self._files._call(entry, "flush", stream.flush)

    def fileno(self) -> int:
        entry, stream = self._live()
        return cast(int, self._files._call(entry, "fileno", stream.fileno))

    def fstat(self) -> os.stat_result:
        entry, stream = self._live()
        def commit(value: object) -> None:
            size = cast(os.stat_result, value).st_size
            _bounded(size, 0, 2**63 - 1)
            registered = self._files._paths[entry.path]
            registered.size = size
            _bounded(size, 0, registered.cap)
        return cast(os.stat_result, self._files._call(entry, "fstat",
            lambda: self._files._stat(stream), commit=commit))

    def fsync(self) -> None:
        entry, stream = self._live()
        self._files._call(entry, "fsync", lambda: self._files._fsync(stream.fileno()))

    def close(self) -> None:
        if self._entry is not None:
            self._files._close(self._entry)

    def __exit__(self, *args: object) -> None:
        if self._entry is None:
            raise ValueError("file context not entered")
        if args[0] is not None:
            try:
                self._files._close(self._entry, cleanup=True)
            except Exception:
                pass
        else:
            self.close()


class RetainedFiles(_RetainedActions):
    def __init__(self, job: SourceHostJob, *, root: Path, clock: Clock, max_paths: int = 32768,
                 max_live: int = 16, max_generations: int = 262144,
                 _open: Callable[[Path, str], BinaryFile] = _open,
                 _stat: Callable[[BinaryFile], os.stat_result] = _stat,
                 _path_action: Callable[[str, Path, Path | None], object] = _path_action,
                 _fsync: Callable[[int], None] = os.fsync) -> None:
        if type(job) is not SourceHostJob or clock is not job._clock:
            raise ValueError("same job and clock required")
        _path(root)
        _bounded(max_paths, 1, 32768)
        _bounded(max_live, 1, 16)
        _bounded(max_generations, 1, 262144)
        self._job, self.root = job, root
        self._max_paths, self._max_live, self._max_generations = max_paths, max_live, max_generations
        self._open, self._stat, self._path_action, self._fsync = _open, _stat, _path_action, _fsync
        self._paths: dict[Path, _Registration] = {}
        self._entries: list[_Entry] = []
        self._consumers: dict[int, tuple[tuple[Path, ...], tuple[Path, ...]]] = {}
        self._generation = self._next_token = 0
        self._active = False
        self._runtime_candidates: set[Path] = set()

    def register(self, path: Path, *, role: str, owned: bool, max_bytes: int) -> None:
        _path(path)
        _bounded(max_bytes, 0, _MAX_BYTES)
        if type(role) is not str or role not in _ROLES or type(owned) is not bool:
            raise ValueError("finite file role required")
        if owned and (role in _READONLY or path == self.root or self.root not in path.parents):
            raise ValueError("only exact generated descendants may be owned")
        old = self._paths.get(path)
        if old is not None:
            if (old.role, old.owned, old.cap) != (role, owned, max_bytes):
                raise ValueError("registration cannot change")
            return
        if len(self._paths) >= self._max_paths:
            raise ValueError("registered path cap")
        self._paths[path] = _Registration(role, owned, max_bytes)

    def _registered(self, path: Path) -> _Registration:
        _path(path)
        if path not in self._paths:
            raise ValueError("unregistered path")
        return self._paths[path]

    def _register_checked(self, path: Path) -> None:
        if path not in self._paths:
            self.register(path, role="checked", owned=False, max_bytes=_MAX_BYTES)

    def _compact(self) -> None:
        keep = []
        for entry in self._entries:
            if entry.closed and not entry.row.uncertain and not entry.row.in_flight:
                registered = self._paths[entry.path]
                registered.settled += 1
                registered.known += entry.row.known_bytes
            else:
                keep.append(entry)
        self._entries = keep

    def _reserve(self, path: Path, mode: str) -> _Entry:
        self._registered(path)
        self._compact()
        if len(self._entries) >= self._max_live or self._generation >= self._max_generations:
            raise ValueError("retained action generation/live cap")
        self._generation += 1
        entry = _Entry(self._generation, path, mode)
        self._entries.append(entry)
        return entry

    def _busy(self, path: Path) -> bool:
        return (self._paths[path].retained or any((e.path == path or e.row.destination == str(path))
                    and (not e.closed or e.row.uncertain or e.row.in_flight)
                    for e in self._entries)
                or any(path in (*inputs, *outputs) for inputs, outputs in self._consumers.values()))

    def _acquire(self, path: Path, mode: str) -> _Entry:
        self._check()
        registered = self._registered(path)
        if mode not in ("rb", "xb") or self._busy(path):
            raise ValueError("certain unclaimed path and binary mode required")
        if mode == "xb" and (not registered.owned or registered.present or registered.role == "directory"):
            raise ValueError("exclusive generated leaf required")
        entry = self._reserve(path, mode)
        def commit(value: object) -> None:
            entry.stream = cast(BinaryFile, value)
            if mode == "xb":
                registered.present = True
        self._call(entry, "open", lambda: self._open(path, mode), commit=commit)
        return entry

    def _close(self, entry: _Entry, *, cleanup: bool = False) -> None:
        if entry.closed:
            return
        assert entry.stream is not None
        self._call(entry, "close", entry.stream.close,
            commit=lambda _: setattr(entry, "closed", True), cleanup=cleanup)

    def open_file(self, path: Path, mode: Literal["rb", "xb"]) -> RetainedBinary:
        self._registered(path)
        return RetainedBinary(self, path, mode)

    def open_binary(self, path: Path) -> RetainedBinary:
        return self.open_file(path, "rb")

    def stage_deadline(self, deadline_ns: int) -> ContextManager[None]:
        return self._job.driver_io_deadline(deadline_ns)

    def resolve_path(self, path: Path) -> Path:
        return cast(Path, self.path_action("resolve", path))

    def resolve_runtime_path(self, path: Path) -> Path:
        """Register only the existing selector's two actual candidates and aliases."""
        _path(path)
        if path not in self._runtime_candidates and len(self._runtime_candidates) >= 2:
            raise ValueError("paired runtime candidate cap")
        self.register(path, role="runtime", owned=False, max_bytes=268435456)
        self._runtime_candidates.add(path)
        return self.resolve_path(path)

    def stat_path(self, path: Path) -> os.stat_result:
        return cast(os.stat_result, self.path_action("stat", path))

    def path_action(self, operation: str, path: Path, *, destination: Path | None = None,
                    _cleanup: bool = False) -> object:
        registered = self._registered(path)
        if operation not in ("resolve", "stat", "lstat", "mkdir", "replace", "delete", "disk_usage"):
            raise ValueError("finite path action required")
        if operation == "disk_usage" and registered.role not in ("checked", "directory"):
            raise ValueError("disk usage requires registered directory/ancestor")
        if operation in ("mkdir", "replace", "delete"):
            if not registered.owned or self._busy(path):
                raise ValueError("unclaimed owned path required")
        if operation == "mkdir" and registered.role != "directory":
            raise ValueError("registered generated directory required")
        target = None
        if operation == "replace":
            if destination is None:
                raise ValueError("registered destination required")
            target = self._registered(destination)
            if not target.owned or target.present or self._busy(destination) or target.role == "directory":
                raise ValueError("absent unclaimed owned destination required")
            try:
                self.path_action("lstat", destination)
            except FileNotFoundError:
                pass
            else:
                raise ValueError("replacement destination already exists")
        elif destination is not None:
            raise ValueError("unexpected destination")
        entry = self._reserve(path, operation)
        if destination is not None:
            entry.row = replace(entry.row, destination=str(destination))
        missing = False
        actual = ("rmdir" if registered.role == "directory" else "unlink") if operation == "delete" else operation
        def action() -> object:
            nonlocal missing
            try:
                return self._path_action(actual, path, destination)
            except FileNotFoundError:
                if operation not in ("stat", "lstat"):
                    raise
                missing = True
                return None
        def commit(value: object) -> None:
            entry.closed = True
            if operation == "mkdir":
                registered.present = True
            elif operation in ("delete", "replace"):
                registered.present = False
                if target is not None:
                    target.present = True
            elif operation in ("stat", "lstat") and not missing:
                size = cast(os.stat_result, value).st_size
                _bounded(size, 0, 2**63 - 1)
                registered.size = size
                _bounded(size, 0, registered.cap if registered.role != "directory" else _MAX_BYTES)
            elif operation == "resolve":
                _path(cast(Path, value))
                if value != path:
                    if registered.role != "runtime" or path not in self._runtime_candidates:
                        raise ValueError("registered path resolution changed")
                    self.register(cast(Path, value), role="runtime", owned=False, max_bytes=268435456)
            elif operation == "disk_usage":
                if not isinstance(value, tuple) or len(value) != 3:
                    raise ValueError("bounded disk usage triple required")
                for scalar in value:
                    _bounded(scalar, 0, 2**63 - 1)
                if value[1] > value[0] or value[2] > value[0]:
                    raise ValueError("inconsistent disk usage facts")
        result = self._call(entry, operation, action, commit=commit, cleanup=_cleanup)
        if missing:
            raise FileNotFoundError(path)
        return result

    def acquire_consumer(self, paths: tuple[Path, ...], *, outputs: tuple[Path, ...] = ()) -> int:
        self._check()
        if type(paths) is not tuple or type(outputs) is not tuple or len(paths) + len(outputs) > self._max_paths:
            raise ValueError("bounded exact path tuples required")
        if len(set((*paths, *outputs))) != len(paths) + len(outputs):
            raise ValueError("distinct producer/consumer paths required")
        if len(self._consumers) >= self._max_live or self._next_token >= self._max_generations:
            raise ValueError("consumer token cap")
        for path in (*paths, *outputs):
            registered = self._registered(path)
            if self._busy(path):
                raise ValueError("path custody unresolved")
            if path in outputs:
                if not registered.owned or registered.present or registered.role == "directory":
                    raise ValueError("new owned output required")
                try:
                    self.path_action("lstat", path)
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError("external output already exists")
        self._next_token += 1
        self._consumers[self._next_token] = (paths, outputs)
        # A failed/late launch can have created the leaf before returning.
        for path in outputs:
            self._paths[path].present = True
        return self._next_token

    def release_consumer(self, token: int, *, retired: bool) -> None:
        _bounded(token, 1, self._max_generations)
        if type(retired) is not bool or token not in self._consumers:
            raise ValueError("live exact consumer token required")
        if not retired:
            return
        self._cleanup_check()
        _, outputs = self._consumers[token]
        for path in outputs:
            try:
                self.path_action("stat", path, _cleanup=True)
            except FileNotFoundError:
                self._paths[path].present = False
        del self._consumers[token]

    def snapshot(self) -> RetainedFilesStatus:
        unresolved = any(not e.closed or e.row.uncertain or e.row.in_flight for e in self._entries)
        preserved = self._retained_ancestors()
        present = any(p.owned and p.present and not p.retained and path not in preserved
                      for path, p in self._paths.items())
        dirty = unresolved or present or bool(self._consumers)
        return RetainedFilesStatus(tuple(e.row for e in self._entries), tuple(
            RetainedPathStatus(str(path), p.role, p.owned, p.present, p.settled, p.known, p.cap, p.size, p.retained)
            for path, p in self._paths.items()), tuple(self._consumers), self._generation, dirty, not dirty)

    def _retained_ancestors(self) -> set[Path]:
        return {parent for path, p in self._paths.items() if p.retained
                for parent in path.parents if parent in self._paths}

    def retire(self, *, retain: tuple[Path, ...] = ()) -> RetainedFilesStatus:
        if type(retain) is not tuple or len(retain) > self._max_paths or len(set(retain)) != len(retain):
            raise ValueError("bounded exact output retention tuple required")
        for path in retain:
            registered = self._registered(path)
            if (not registered.owned or not registered.present or registered.retained
                    or registered.role not in ("candidate", "sidecar", "graph", "ass", "asset", "font")
                    or self._busy(path)):
                raise ValueError("only certainly settled owned output leaves may transfer")
        try:
            self._cleanup_check()
        except Exception:
            return self.snapshot()
        for path in retain:
            self._paths[path].retained = True
        for entry in tuple(self._entries):
            if entry.stream is not None and not entry.closed and not entry.row.uncertain and not entry.row.in_flight:
                try:
                    self._close(entry, cleanup=True)
                except Exception:
                    pass
        preserved = self._retained_ancestors()
        for path, registered in sorted(self._paths.items(), key=lambda p: len(p[0].parts), reverse=True):
            if registered.owned and registered.present and path not in preserved and not self._busy(path):
                try:
                    self.path_action("delete", path, _cleanup=True)
                except Exception:
                    pass
        return self.snapshot()
