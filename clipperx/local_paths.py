"""Provider-free local paths, source identities, and atomic file writes."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Iterator, Sequence, Protocol


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return slug[:48] or "video"


class SourceIdentityFiles(Protocol):
    def resolve_path(self, path: Path) -> Path: ...
    def stat_path(self, path: Path) -> os.stat_result: ...


def source_fingerprint(source: Path, *, _files: SourceIdentityFiles | None = None) -> str:
    resolved = Path(source).resolve() if _files is None else _files.resolve_path(Path(source))
    stat = resolved.stat() if _files is None else _files.stat_path(resolved)
    payload = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(slots=True)
class Paths:
    """Filesystem-only project paths; never reads environment or .env files."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)
    transcripts_dir: Path = field(init=False)
    clips_dir: Path = field(init=False)
    subtitles_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    downloads_dir: Path = field(init=False)
    broll_dir: Path = field(init=False)
    jobs_dir: Path = field(init=False)
    timelines_dir: Path = field(init=False)
    profiles_dir: Path = field(init=False)
    face_model_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.base_dir / "data"
        self.audio_dir = self.data_dir / "processed" / "audio"
        self.transcripts_dir = self.data_dir / "processed" / "transcripts"
        self.clips_dir = self.data_dir / "processed" / "clips"
        self.subtitles_dir = self.clips_dir / "subtitles"
        self.frames_dir = self.data_dir / "processed" / "frames"
        self.logs_dir = self.base_dir / "logs"
        self.downloads_dir = self.data_dir / "downloads"
        self.broll_dir = self.data_dir / "processed" / "broll"
        self.jobs_dir = self.data_dir / "processed" / "jobs"
        self.timelines_dir = self.data_dir / "processed" / "timelines"
        self.profiles_dir = self.data_dir / "profiles"
        self.face_model_dir = self.data_dir / "models" / "face_identity"

    def ensure(self) -> "Paths":
        for path in self.directories():
            path.mkdir(parents=True, exist_ok=True)
        return self

    def directories(self) -> Sequence[Path]:
        return (
            self.data_dir,
            self.audio_dir,
            self.transcripts_dir,
            self.clips_dir,
            self.subtitles_dir,
            self.frames_dir,
            self.logs_dir,
            self.downloads_dir,
            self.broll_dir,
            self.jobs_dir,
            self.timelines_dir,
            self.profiles_dir,
            self.face_model_dir,
        )


paths = Paths()

_atomic_write_locks_guard = threading.Lock()
_atomic_write_locks: dict[str, threading.RLock] = {}


def _atomic_write_lock(target: Path) -> threading.RLock:
    key = str(target.resolve())
    with _atomic_write_locks_guard:
        lock = _atomic_write_locks.get(key)
        if lock is None:
            lock = _atomic_write_locks[key] = threading.RLock()
        return lock


def _replace_with_windows_retry(temporary: Path, target: Path) -> None:
    for attempt in range(6):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01 * (2**attempt))


def source_job_root(source: Path, *, jobs_root: Path | None = None) -> Path:
    resolved = Path(source).resolve()
    return (jobs_root or paths.jobs_dir) / (
        f"{safe_slug(resolved.stem)}-{source_fingerprint(resolved)}"
    )


def global_streamer_identity_path() -> Path:
    return paths.profiles_dir / "streamer_identity.json"


def atomic_write_text(
    target: Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write text using a unique sibling temporary and one atomic replace."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        with _atomic_write_lock(target):
            _replace_with_windows_retry(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _batch_state_paths(targets: Sequence[Path]) -> tuple[Path, Path]:
    parent_strings = [str(target.parent.resolve()) for target in targets]
    try:
        common_parent = Path(os.path.commonpath(parent_strings))
    except ValueError:
        # Windows volumes have no common ancestor. Readers and writers must
        # still share one deterministic lock/journal for this target set.
        common_parent = Path(min(parent_strings))
    transaction_dir = common_parent / ".clipperx-transactions"
    transaction_dir.mkdir(parents=True, exist_ok=True)
    identity = "\n".join(sorted(str(target.resolve()) for target in targets))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return (
        transaction_dir / f"{digest}.lock",
        transaction_dir / f"{digest}.journal.json",
    )


@contextmanager
def _interprocess_file_lock(lock_path: Path, *, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Hold one byte of a lock file across processes on Windows or POSIX."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
        os.fsync(handle.fileno())
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    try:
        while not acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl = importlib.import_module("fcntl")

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for batch lock: {lock_path}") from exc
                time.sleep(0.025)
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl = importlib.import_module("fcntl")

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _restore_batch_from_journal(
    journal_path: Path,
    targets: Sequence[Path],
    *,
    encoding: str,
) -> None:
    if not journal_path.is_file():
        return
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid batch recovery journal: {journal_path}") from exc
    entries = payload.get("entries") if isinstance(payload, dict) else None
    expected_paths = {str(target.resolve()) for target in targets}
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or not isinstance(entries, list)
        or len(entries) != len(expected_paths)
    ):
        raise RuntimeError(f"invalid batch recovery journal: {journal_path}")
    previous_by_path: dict[str, tuple[bool, str | None]] = {}
    # Journal order belongs to the writer; readers may request the same set
    # in another order. Validate every entry before restoring any target.
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("existed"), bool):
            raise RuntimeError(f"invalid batch recovery journal: {journal_path}")
        entry_path = entry.get("path")
        if (
            not isinstance(entry_path, str)
            or entry_path not in expected_paths
            or entry_path in previous_by_path
        ):
            raise RuntimeError(f"invalid batch recovery journal: {journal_path}")
        previous_text = entry.get("previous_text")
        if entry["existed"] and not isinstance(previous_text, str):
            raise RuntimeError(f"invalid batch recovery journal: {journal_path}")
        previous_by_path[entry_path] = (
            entry["existed"], previous_text if entry["existed"] else None,
        )
    if set(previous_by_path) != expected_paths:
        raise RuntimeError(f"invalid batch recovery journal: {journal_path}")
    for target in targets:
        existed, previous_text = previous_by_path[str(target.resolve())]
        if existed:
            current = target.read_text(encoding=encoding) if target.is_file() else None
            if current != previous_text:
                if previous_text is None:
                    raise RuntimeError(f"invalid batch recovery journal: {journal_path}")
                atomic_write_text(target, previous_text, encoding=encoding)
        elif target.exists():
            target.unlink()
    journal_path.unlink(missing_ok=True)


def _batch_guard(targets: Sequence[Path]) -> tuple[ExitStack, Path]:
    resolved_keys = [str(target.resolve()) for target in targets]
    ordered_locks = sorted(
        ((_atomic_write_lock(target), key) for target, key in zip(targets, resolved_keys)),
        key=lambda item: item[1],
    )
    lock_path, journal_path = _batch_state_paths(targets)
    stack = ExitStack()
    try:
        for lock, _ in ordered_locks:
            stack.enter_context(lock)
        stack.enter_context(_interprocess_file_lock(lock_path))
    except BaseException:
        stack.close()
        raise
    return stack, journal_path


def atomic_read_text_batch(
    targets: Sequence[Path],
    *,
    encoding: str = "utf-8",
) -> tuple[str | None, ...]:
    """Read a batch consistently, recovering any interrupted publication first."""

    normalized = tuple(Path(target) for target in targets)
    if not normalized:
        raise ValueError("targets must not be empty")
    resolved_keys = [str(target.resolve()) for target in normalized]
    if len(set(resolved_keys)) != len(resolved_keys):
        raise ValueError("targets must be unique")
    stack, journal_path = _batch_guard(normalized)
    with stack:
        _restore_batch_from_journal(journal_path, normalized, encoding=encoding)
        return tuple(
            target.read_text(encoding=encoding) if target.is_file() else None
            for target in normalized
        )


def atomic_write_text_batch(
    targets: Sequence[Path],
    text: str,
    *,
    encoding: str = "utf-8",
) -> tuple[Path, ...]:
    """Publish several files with interprocess locking and crash recovery."""

    normalized = tuple(Path(target) for target in targets)
    if not normalized:
        raise ValueError("targets must not be empty")
    resolved_keys = [str(target.resolve()) for target in normalized]
    if len(set(resolved_keys)) != len(resolved_keys):
        raise ValueError("targets must be unique")
    for target in normalized:
        target.parent.mkdir(parents=True, exist_ok=True)

    stack, journal_path = _batch_guard(normalized)
    with stack:
        _restore_batch_from_journal(journal_path, normalized, encoding=encoding)
        previous = {
            target: target.read_text(encoding=encoding) if target.is_file() else None
            for target in normalized
        }
        journal = {
            "version": 1,
            "entries": [
                {
                    "path": str(target.resolve()),
                    "existed": previous[target] is not None,
                    "previous_text": previous[target],
                }
                for target in normalized
            ],
        }
        atomic_write_text(
            journal_path,
            json.dumps(journal, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        try:
            for target in normalized:
                atomic_write_text(target, text, encoding=encoding)
        except Exception as exc:
            try:
                _restore_batch_from_journal(
                    journal_path,
                    normalized,
                    encoding=encoding,
                )
            except Exception as rollback_exc:
                raise RuntimeError(
                    "batch write failed and rollback was incomplete: "
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                ) from exc
            raise
        journal_path.unlink(missing_ok=True)
    return normalized
