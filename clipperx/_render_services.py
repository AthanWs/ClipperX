"""One serial owned render child, bounded capture or streaming to retained files."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import sys

from ._retained_files import RetainedBinary, RetainedFiles, _bounded
from ._source_host_ownership import ChildLease, Clock, HostedObservationControl, RenderKind, SourceHostJob

_TIMEOUTS = {"probe": 45, "spatial": 900, "finalmix": 120, "encoder": 900}
_RAW_CAP = 8294400 * 10740
_EVIDENCE_CAP = 256 * 1024**2


@dataclass(frozen=True, slots=True)
class ToolResult:
    argv: tuple[str, ...]
    kind: RenderKind
    exit_code: int
    control_id: int
    generation: int
    natural_eof: bool
    retired: bool
    stdout_bytes: int
    stdout_sha256: str
    stdout: bytes | None
    stderr_bytes: int
    stderr_sha256: str
    stderr_tail: bytes
    stderr_truncated: bool

    def __post_init__(self) -> None:
        if (type(self.argv) is not tuple or not 0 < len(self.argv) <= 128
                or any(type(arg) is not str or "\0" in arg or len(arg) > 65536 for arg in self.argv)
                or sum(len(arg.encode("utf-8")) for arg in self.argv) > 65536):
            raise ValueError("bounded detached argv required")
        if type(self.kind) is not str or self.kind not in _TIMEOUTS:
            raise ValueError("finite render result kind required")
        _bounded(self.exit_code, 0, 2**32 - 1)
        _bounded(self.control_id, 1, 2**63 - 1)
        _bounded(self.generation, 1, 2**63 - 1)
        _bounded(self.stdout_bytes, 0, _RAW_CAP)
        _bounded(self.stderr_bytes, 0, 16 * 1024**2)
        for digest in (self.stdout_sha256, self.stderr_sha256):
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("bounded render SHA256 required")
        if self.stdout is not None and (type(self.stdout) is not bytes or len(self.stdout) > 1_000_000
                or len(self.stdout) != self.stdout_bytes or self.kind != "probe"):
            raise ValueError("bounded probe capture required")
        if type(self.stderr_tail) is not bytes or len(self.stderr_tail) > 65536:
            raise ValueError("bounded stderr tail required")
        if any(type(flag) is not bool for flag in (self.natural_eof, self.retired, self.stderr_truncated)):
            raise ValueError("plain render result flags required")


class OwnedRenderTools:
    def __init__(self, job: SourceHostJob, files: RetainedFiles, *, clock: Clock) -> None:
        if (type(job) is not SourceHostJob or type(files) is not RetainedFiles
                or files._job is not job or clock is not job._clock):
            raise ValueError("same exact job, files and clock required")
        self._job, self._files = job, files
        self._evidence_bytes = 0

    def _settle_consumer(self, control: HostedObservationControl, token: int,
                         kind: RenderKind, lease: ChildLease | None) -> bool:
        """Keep this driver's exact claim until certain retirement or its existing end."""
        job = self._job
        end: int | None = None
        advance = True
        while True:
            now, current = job._now(), job._current_worker()
            with job._lock:
                if (current is not job._driver or not job._registered or job._done
                        or control._job is not job or job._control is not control):
                    return False
                ends = ([job._failure.cleanup_deadline_ns] if job._failure else [])
                if control._cleanup_deadline is not None:
                    ends.append(control._cleanup_deadline)
                if not ends:
                    return False
                end = min(ends) if end is None else min(end, *ends)
                if now >= end:
                    return False
                status = control.ownership_snapshot()
                if status.observation_id != control._id or status.kind != kind:
                    return False
                if lease is not None and (lease.observation_id != control._id or lease.kind != kind
                        or not status.resources or not status.diagnostics
                        or any(row.generation != lease.generation for row in status.resources)
                        or any(d.generation != lease.generation or d.kind != kind for d in status.diagnostics)):
                    return False
                retired = (status.phase in ("retired_ok", "stopped_clean") and status.window_resources_zero
                    and not status.scope_open and not status.consumer_active and not status.payload_retained
                    and not status.lease_outstanding and not status.cleanup_required)
                if not retired and not advance:
                    job._cleanup_condition.wait(min(.01, (end - now) / 1e9))
                    advance = True
                    continue
            if retired:
                # An exception ends this attempt; never replay potentially uncertain output stats.
                self._files.release_consumer(token, retired=True)
                return True
            control._cleanup_child()
            advance = False

    def run(self, argv: tuple[str, ...], *, kind: RenderKind, timeout: int,
            max_stdout_bytes: int, max_stderr_bytes: int,
            expected_stdout_bytes: int | None = None, sink: RetainedBinary | None = None,
            inputs: tuple[Path, ...] = (), outputs: tuple[Path, ...] = ()) -> ToolResult:
        if type(kind) is not str or kind not in _TIMEOUTS:
            raise ValueError("explicit render kind required")
        _bounded(timeout, 1, _TIMEOUTS[kind])
        _bounded(max_stdout_bytes, 1, 1_000_000 if kind == "probe" else _RAW_CAP)
        _bounded(max_stderr_bytes, 1, 16 * 1024**2)
        if expected_stdout_bytes is not None:
            _bounded(expected_stdout_bytes, 0, max_stdout_bytes)
        if kind in ("spatial", "finalmix") and expected_stdout_bytes is None:
            raise ValueError("exact stage stdout count required")
        if kind == "encoder" and (expected_stdout_bytes != 0 or max_stdout_bytes != 1):
            raise ValueError("encoder requires zero stdout and one EOF reservation")
        if sink is not None and (type(sink) is not RetainedBinary or sink._files is not self._files):
            raise ValueError("same retained sink required")
        if self._evidence_bytes + max_stderr_bytes > _EVIDENCE_CAP:
            raise ValueError("aggregate render evidence cap")
        if (type(argv) is not tuple or not 0 < len(argv) <= 128
                or any(type(arg) is not str or "\0" in arg or len(arg) > 65536 for arg in argv)
                or sum(len(arg.encode("utf-8")) for arg in argv) > 65536):
            raise ValueError("bounded canonical argv required")
        if type(inputs) is not tuple or type(outputs) is not tuple:
            raise ValueError("exact immutable render path tuples required")
        argv_bytes = sum(len(arg.encode("utf-8")) for arg in argv)
        capture_budget = max_stdout_bytes if kind == "probe" and sink is None else 0
        if self._evidence_bytes + max_stderr_bytes + capture_budget + argv_bytes > _EVIDENCE_CAP:
            raise ValueError("aggregate render evidence cap")
        # Runtime/source/probe path operands are already registered by their host.
        # Graph-internal dependencies still require the explicit inputs tuple.
        inputs = tuple(dict.fromkeys((*inputs, *(path for path in self._files._paths
            if str(path) in argv and path not in outputs))))
        self._job.check()
        control = self._job.new_render(kind,
            deadline_ns=min(self._job._deadline, self._job._now() + timeout * 10**9))
        token: int | None = None
        lease: ChildLease | None = None
        release_attempted = False
        success = False
        log_hash = hashlib.sha256()
        log_size = 0
        capture = bytearray() if kind == "probe" and sink is None else None
        accepted, output_hash = 0, hashlib.sha256()
        try:
            control._enter(kind, max_stderr_bytes, max_stdout_bytes, timeout * 10**9, self._job._grace)
            token = self._files.acquire_consumer(inputs, outputs=outputs)
            lease = control._launch(kind, argv, timeout * 10**9)

            def read_log() -> None:
                nonlocal log_size
                while True:
                    line = control._readline_stderr(lease, 16385)
                    if not line:
                        return
                    log_hash.update(line)
                    log_size += len(line)

            control._start_metadata(lease, read_log)
            limit = max_stdout_bytes if expected_stdout_bytes is None else expected_stdout_bytes
            natural_eof = False
            while accepted < limit:
                raw = control._read_stdout(lease, min(65536, limit - accepted))
                if not raw:
                    natural_eof = True
                    break
                if sink is not None:
                    view = memoryview(raw)
                    try:
                        offset = 0
                        while offset < len(view):
                            count = sink.write(view[offset:])
                            output_hash.update(view[offset:offset + count])
                            accepted += count
                            offset += count
                    finally:
                        view.release()
                else:
                    accepted += len(raw)
                    output_hash.update(raw)
                    if capture is not None:
                        capture.extend(raw)
            if expected_stdout_bytes is not None and accepted != expected_stdout_bytes:
                raise ValueError("short render stdout")
            if not natural_eof:
                if control._read_stdout(lease, 1):
                    raise ValueError("extra render stdout")
                natural_eof = True
            outcome = control._finish_child(lease, natural_eof)
            diagnostic = control.ownership_snapshot().diagnostics[-1]
            if diagnostic.stdout_bytes != accepted or diagnostic.stdout_sha256 != output_hash.hexdigest():
                raise ValueError("render accepted count/hash association mismatch")
            control._leave(True)
            status = control.ownership_snapshot()
            if status.phase != "retired_ok" or not status.window_resources_zero:
                raise ValueError("render control not certainly retired")
            release_attempted = True
            self._files.release_consumer(token, retired=True)
            token = None
            success = True
            return ToolResult(argv, kind, outcome.exit_code, lease.observation_id, lease.generation,
                outcome.natural_eof, True, accepted, output_hash.hexdigest(),
                bytes(capture) if capture is not None else None, log_size, log_hash.hexdigest(),
                diagnostic.stderr_tail, diagnostic.truncated)
        finally:
            original_error = sys.exc_info()[1]
            self._evidence_bytes += log_size + (len(capture) if capture is not None else 0) + argv_bytes
            try:
                current = self._job._current_worker()
                with self._job._lock:
                    leave = (not success and current is self._job._driver and self._job._registered
                        and not self._job._done and control._job is self._job and self._job._control is control
                        and control._phase not in ("retired_ok", "stopped_clean"))
                if leave:
                    control._leave(False)
                if token is not None and not release_attempted and self._settle_consumer(control, token, kind, lease):
                    token = None
            except Exception:
                if original_error is None:
                    raise

    def __call__(self, argv: tuple[str, ...], *, timeout: int,
                 max_stdout_bytes: int, max_stderr_bytes: int) -> subprocess.CompletedProcess[bytes]:
        result = self.run(argv, kind="probe", timeout=timeout,
            max_stdout_bytes=max_stdout_bytes, max_stderr_bytes=max_stderr_bytes)
        # stderr is explicitly only the bounded tail; ToolResult carries count/hash/truncation.
        return subprocess.CompletedProcess(argv, result.exit_code, result.stdout, result.stderr_tail)
