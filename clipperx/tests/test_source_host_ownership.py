"""Scripted ownership checks. No native operations or real worker execution."""

from __future__ import annotations

import ctypes
import subprocess
import threading
import io
import os
import queue
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("unexpected real external boundary")


def guard_source_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    original = os.stat
    def guarded(path: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(path, (str, bytes, os.PathLike)):
            text = os.fsdecode(path).replace("\\", "/").casefold()
            if text == "host" or text.startswith("o:/host/"):
                forbidden()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "stat", guarded)


@pytest.fixture(autouse=True)
def closed_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    guard_source_stat(monkeypatch)
    monkeypatch.setattr(ctypes, "WinDLL", forbidden, raising=False)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)
    monkeypatch.setattr(threading.Thread, "join", forbidden)
    monkeypatch.setattr(threading.Timer, "start", forbidden)
    monkeypatch.setattr(threading.Event, "wait", forbidden)
    monkeypatch.setattr(threading.Condition, "wait", forbidden)
    monkeypatch.setattr(queue.Queue, "get", forbidden)
    monkeypatch.setattr(queue.Queue, "put", forbidden)
    monkeypatch.setattr(time, "monotonic_ns", forbidden)
    monkeypatch.setattr(time, "monotonic", forbidden)
    import clipperx.source_decode as source
    monkeypatch.setattr(source, "source_fingerprint", forbidden)
    monkeypatch.setattr(source, "_verify_pins", forbidden)
    monkeypatch.setattr(source, "probe_source", forbidden)


def test_inert_defaults_and_bounds() -> None:
    from clipperx._source_host_ownership import SourceHostJob, HostDecodeDiagnostic, ChildLease

    class InertWorker:
        daemon = False
        ident = None
        start = forbidden
        join = forbidden
        is_alive = forbidden

    job = SourceHostJob(InertWorker(), deadline_ns=1, clock=forbidden,
        io_factory=forbidden, worker_factory=forbidden, event_factory=forbidden,
        current_worker=forbidden, process_factory=forbidden)
    status = job.snapshot()
    assert status.phase == "reserved" and status.own0 is False
    for deadline in (True, 0, 2**63):
        with pytest.raises(ValueError):
            SourceHostJob(InertWorker(), deadline_ns=deadline)
    for grace in (True, 0, 5_000_000_001):
        with pytest.raises(ValueError):
            SourceHostJob(InertWorker(), deadline_ns=1, cleanup_grace_ns=grace)
    with pytest.raises(ValueError):
        ChildLease(True, 1, "audio_f32")
    with pytest.raises(ValueError):
        HostDecodeDiagnostic(1, "audio_f32", 0, "A" * 64, 0, b"", False, False, False, None)
    h = admitted()
    h.now = 9
    with pytest.raises(ValueError):
        h.job.check()
    assert h.job.snapshot().failure.code == "overflow"
    overflow = admitted()
    overflow.now = 2**63 - 2
    with pytest.raises(ValueError):
        overflow.job.fail("validation", "validate")
    assert overflow.job.snapshot().failure is None


class FakeEvent:
    def __init__(self) -> None:
        self.value = False

    def set(self) -> None:
        self.value = True

    def clear(self) -> None:
        self.value = False

    def is_set(self) -> bool:
        return self.value

    def wait(self, timeout: float) -> bool:
        if not self.value:
            raise AssertionError("unscripted event wait")
        return self.value


class ScriptWorker:
    daemon = False
    ident: int | None = None

    def __init__(self, host: Any, name: str, target: Callable[[], None] | None = None) -> None:
        self.host, self.name, self.target = host, name, target
        self.alive = False
        self.started = self.joined = 0
        self.start_action: Callable[[], None] | None = None

    def start(self) -> None:
        self.started += 1
        if self.start_action:
            self.start_action()
        self.alive = True
        self.ident = self.started
        if self.name == "source-host-watchdog":
            self.host.events[1].set()
        elif self.target is not None:
            prior, self.host.current = self.host.current, self
            try:
                self.target()
            finally:
                self.alive = False
                self.host.current = prior

    def join(self, timeout: float) -> None:
        assert self.host.current is not self
        self.joined += 1

    def is_alive(self) -> bool:
        return self.alive


class ScriptProcess:
    def __init__(self, payload: bytes = b"abcd", log: bytes = b"tag\n", *, pid: int = 71) -> None:
        from clipperx.tests.test_windows_pipe_io import FakeHandle
        self.pid = pid
        self.returncode: int | None = None
        self.stdout, self.stderr = io.BytesIO(payload), io.BytesIO(log)
        self._handle = FakeHandle(pid + 1000)
        self.actions: list[str] = []

    def poll(self) -> int | None:
        self.actions.append("poll")
        return self.returncode

    def wait(self, timeout: float) -> int:
        self.actions.append("wait")
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.actions.append("terminate")


def scripted_host() -> Any:
    from clipperx._windows_pipe_io import WindowsSyncIo
    from clipperx._source_host_ownership import SourceHostJob
    from clipperx.tests.test_windows_pipe_io import FakeKernel
    host = SimpleNamespace(now=10, current=None, events=[], workers=[], children=[], native_calls=0)
    host.kernel = FakeKernel()
    duplicate = host.kernel.duplicate
    def unique(*args: object) -> int:
        host.kernel.output += 1
        return duplicate(*args)
    host.kernel.DuplicateHandle.action = unique
    def current_worker() -> object:
        assert not host.job._lock._is_owned(), "external identity callback under state lock"
        return host.current
    def native_factory() -> WindowsSyncIo:
        return WindowsSyncIo(kernel32=host.kernel, last_error=host.kernel.last_error)
    def event() -> FakeEvent:
        result = FakeEvent()
        host.events.append(result)
        return result
    def worker(target: Callable[[], None], name: str) -> ScriptWorker:
        result = ScriptWorker(host, name, target)
        host.workers.append(result)
        return result
    def process(argv: tuple[str, ...]) -> ScriptProcess:
        result = ScriptProcess(pid=71 + len(host.children))
        host.children.append(result)
        return result
    host.driver = ScriptWorker(host, "driver")
    host.job = SourceHostJob(host.driver, deadline_ns=1_000_000_000, cleanup_grace_ns=100,
        clock=lambda: host.now, io_factory=native_factory, worker_factory=worker,
        event_factory=event, current_worker=current_worker, process_factory=process)
    host.io_factory, host.process_factory = native_factory, process
    return host


def admitted() -> Any:
    h = scripted_host()
    h.job.start_driver()
    h.current = h.driver
    h.job.register_current_driver()
    return h


def entered(h: Any, kind: str = "materialized_f32") -> Any:
    control = h.job.new_observation(kind, deadline_ns=10000)
    control._enter(kind, 4096, 32, None, 100)
    return control


def finish_job(h: Any) -> Any:
    if not h.job.snapshot().driver_done:
        h.job.driver_done()
    h.driver.alive = False
    h.current = None
    h.workers[0].alive = False
    status = h.job.service_cleanup()
    assert len(h.kernel.events) <= 256
    return status


def test_cleanup_token_identity_deadline_and_custody() -> None:
    h = admitted()
    h.job.fail("validation", "driver")
    context = h.job._driver_cleanup_io()
    with context:
        assert h.job._cleanup_token is context and context.active
        with pytest.raises(ValueError):
            h.job._driver_cleanup_io().__enter__()
        h.job._cancel(h.job._handle_resource)
        assert not h.kernel.CancelSynchronousIo.calls
        h.now = 110
        h.job._cancel(h.job._handle_resource)
        assert len(h.kernel.CancelSynchronousIo.calls) == 1
    assert h.job._cleanup_token is None
    with pytest.raises(ValueError):
        context.__enter__()
    with pytest.raises(ValueError):
        h.job._driver_cleanup_io().__enter__()
    assert finish_job(h).own0


@pytest.mark.parametrize("requested", [True, False])
def test_cleanup_pending_drains_exact_prior_cancel_claim(requested: bool) -> None:
    h = admitted()
    h.job.fail("validation", "driver")
    handle = h.job._handle_resource
    claim = h.job._claim(handle)
    waits = []
    def wait(timeout: float) -> bool:
        waits.append(timeout)
        token = h.job._cleanup_token
        assert token is not None and not token.active
        # Faithful condition seam: release state lock while prior external call returns.
        h.job._lock.release()
        try:
            h.job._cancel(handle)
            assert not h.kernel.CancelSynchronousIo.calls
            h.kernel.CancelSynchronousIo.action = lambda _: 1 if requested else 0
            h.kernel.error = 1168
            h.job._perform(handle, claim, lambda: handle.io.request_cancel(handle.handle),
                           stage="retire", code="native", commit=lambda _: handle.update(phase="live"))
        finally:
            h.job._lock.acquire()
        return True
    h.job._cleanup_condition.wait = wait
    with h.job._driver_cleanup_io():
        assert waits and waits[0] <= .01
        assert len(h.kernel.CancelSynchronousIo.calls) == 1
    assert not handle.row.in_flight and h.job._cleanup_token is None
    assert finish_job(h).own0


@pytest.mark.parametrize("case", ["foreign", "unknown", "drain_expired", "shorten", "leak"])
def test_cleanup_admission_preserves_unknown_and_original_grace(case: str) -> None:
    h = admitted()
    failure = h.job.fail("validation", "driver")
    handle = h.job._handle_resource
    if case == "foreign":
        h.current = None
    elif case == "unknown":
        handle.update(uncertain=True)
    elif case == "drain_expired":
        claim = h.job._claim(handle)
        def wait(timeout: float) -> bool:
            h.now = failure.cleanup_deadline_ns
            return False
        h.job._cleanup_condition.wait = wait
    if case in ("foreign", "unknown", "drain_expired"):
        with pytest.raises(ValueError):
            h.job._driver_cleanup_io().__enter__()
        assert h.job._cleanup_token is None
        if case == "drain_expired":
            assert handle.row.in_flight
            h.job._release_claim(handle, claim)
        elif case == "unknown":
            assert handle.row.uncertain
            return  # Unknown custody intentionally remains unresolved in this fake job.
        h.current = h.driver
    else:
        context = h.job._driver_cleanup_io()
        context.__enter__()
        if case == "shorten":
            h.job.request_stop("cancelled", deadline_ns=10)
            h.job._cancel(handle)
            assert len(h.kernel.CancelSynchronousIo.calls) == 1
        else:
            h.job.driver_done()
            assert h.job.snapshot().failure.last_secondary == "protocol"
            assert not h.job.snapshot().own0
        context.__exit__(None, None, None)
    assert h.job.snapshot().failure.first_ns == failure.first_ns
    assert h.job.snapshot().failure.cleanup_deadline_ns <= failure.cleanup_deadline_ns
    assert finish_job(h).own0


def test_watchdog_services_expiry_after_protected_pass() -> None:
    h = admitted()
    h.job.fail("validation", "driver")
    with h.job._driver_cleanup_io():
        clock_calls = 0
        def clock() -> int:
            nonlocal clock_calls
            assert not h.job._lock._is_owned()
            clock_calls += 1
            # service sample and cancel gate sample precede final clock check.
            return 10 if clock_calls < 3 else 110
        h.job._clock = clock
        h.current = h.workers[0]
        try:
            h.workers[0].target()
            assert len(h.kernel.CancelSynchronousIo.calls) == 1
        finally:
            h.current = h.driver
    assert finish_job(h).own0


@pytest.mark.parametrize("case", ["normal", "prestop", "factory", "start_uncertain", "late_stop", "inflight", "unexpected_exit",
    "watch_entered_raise", "watch_no_entry_raise", "watch_late_entry"])
def test_owner_start_states(case: str) -> None:
    h = scripted_host()
    if case in ("watch_entered_raise", "watch_no_entry_raise", "watch_late_entry"):
        factory = h.job._workers
        def make(target: Callable[[], None], name: str) -> ScriptWorker:
            worker = factory(target, name)
            def stop_wait(timeout: float) -> bool:
                row = h.job.snapshot().resources[-1]
                assert row.in_flight and not h.job.snapshot().own0
                h.job._watch_exit = True
                return False
            h.events[0].wait = stop_wait
            def raised_start() -> None:
                worker.started += 1
                worker.alive = True
                prior, h.current = h.current, worker
                try:
                    if case == "watch_entered_raise":
                        target()
                finally:
                    worker.alive = False
                    h.current = prior
                raise OSError("start returned uncertain after possible entry")
            worker.start = raised_start
            return worker
        h.job._workers = make
        with pytest.raises(ValueError):
            h.job.start_driver()
        row = h.job.snapshot().resources[-1]
        assert row.start == "uncertain" and row.uncertain and not row.in_flight
        status = h.job.service_cleanup()
        assert status.own0 is (case == "watch_entered_raise")
        if case == "watch_late_entry":
            h.current = h.workers[0]
            h.workers[0].alive = True
            h.workers[0].target()
            h.workers[0].alive = False
            h.current = None
            assert h.job.service_cleanup().own0
            assert h.job.snapshot().failure == status.failure
        elif case == "watch_no_entry_raise":
            assert h.workers[0].joined == 0
        else:
            assert h.workers[0].joined == 1
        assert h.driver.started == 0
        return
    if case == "prestop":
        assert h.job.request_stop("cancelled").own0
        with pytest.raises(ValueError):
            h.job.start_driver()
        assert h.driver.started == 0
        return
    if case == "factory":
        h.job._workers = forbidden
        with pytest.raises(ValueError):
            h.job.start_driver()
        assert h.driver.started == 0
        return
    if case in ("start_uncertain", "late_stop", "inflight"):
        def during() -> None:
            if case == "start_uncertain":
                raise RuntimeError("uncertain")
            status = h.job.request_stop("cancelled")
            assert status.resources[0].in_flight and not status.own0
        h.driver.start_action = during
    if case == "normal":
        assert h.job.start_driver().phase == "running"
        with pytest.raises(ValueError):
            h.job.start_driver()
    elif case == "unexpected_exit":
        h.job.start_driver()
        h.driver.alive = False
        status = h.job.service_cleanup()
        assert status.failure.code == "protocol"
    else:
        with pytest.raises(ValueError):
            h.job.start_driver()
        assert not h.job.snapshot().own0
        if case == "start_uncertain":
            assert h.job.snapshot().resources[0].uncertain
            h.driver.alive = False
            assert not h.job.service_cleanup().own0
        else:
            h.current = h.driver
            with pytest.raises(ValueError):
                h.job.register_current_driver()
            h.job.driver_done()
            assert h.job.snapshot().failure.code == "cancelled"


@pytest.mark.parametrize("case", ["normal", "registration_failed", "registration_uncertain", "clock_order"])
def test_driver_registration_and_job_stop(case: str) -> None:
    if case == "clock_order":
        h = admitted()
        original_lock = h.job._lock
        pending = []
        samples = iter((10, 11))
        seen = []
        class DeferredClock(Exception):
            pass
        class OrderedClock:
            busy = False
            def __enter__(self) -> None:
                if self.busy:
                    raise DeferredClock()
                self.busy = True
            def __exit__(self, *args: Any) -> None:
                self.busy = False
        class PublishBoundary:
            armed = True
            def __enter__(self) -> None:
                if self.armed:
                    self.armed = False
                    try:
                        h.job._now()
                    except DeferredClock:
                        pending.append(lambda: h.job._now())
                original_lock.__enter__()
            def __exit__(self, *args: Any) -> None:
                original_lock.__exit__(*args)
            def _is_owned(self) -> bool:
                return original_lock._is_owned()
        def sample() -> int:
            assert not h.job._lock._is_owned()
            value = next(samples)
            seen.append(value)
            return value
        h.job._clock = sample
        h.job._clock_lock = OrderedClock()
        h.job._lock = PublishBoundary()
        assert h.job._now() == 10
        assert len(pending) == 1 and pending.pop()() == 11
        assert seen == [10, 11] and h.job.snapshot().failure is None
        h.job._lock = original_lock
        h.now = 11
        h.job._clock = lambda: h.now
        assert finish_job(h).own0
        return
    if case != "normal":
        h = scripted_host()
        h.job.start_driver()
        h.current = h.driver
        if case == "registration_failed":
            h.kernel.DuplicateHandle.action = lambda *args: 0
        else:
            h.kernel.DuplicateHandle.action = lambda *args: 1
        with pytest.raises(ValueError):
            h.job.register_current_driver()
        assert h.job.snapshot().failure.winerror == (5 if case == "registration_failed" else None)
        assert finish_job(h).own0 is (case == "registration_failed")
        return
    h = admitted()
    assert h.kernel.events.count("duplicate") == 1
    assert h.job.snapshot().driver_admitted
    with pytest.raises(ValueError):
        h.job.register_current_driver()
    h.job.request_stop("cancelled")
    h.current = None
    h.job.service_cleanup()
    assert "cancel" in h.kernel.events
    h.current = h.driver
    h.job.driver_done()
    assert not h.job.snapshot().own0
    h.driver.alive = False
    h.current = None
    h.workers[0].alive = False
    assert h.job.service_cleanup().own0
    assert h.driver.joined == 1


@pytest.mark.parametrize("case", ["normal", "wrong_kind", "foreign_lease", "foreign_driver", "time_overflow"])
def test_control_single_use(case: str) -> None:
    h = admitted()
    if case == "foreign_driver":
        h.current = None
        with pytest.raises(ValueError):
            h.job.new_observation("materialized_f32", deadline_ns=10000)
        h.current = h.driver
        assert finish_job(h).own0
        return
    if case == "wrong_kind":
        c = h.job.new_observation("materialized_f32", deadline_ns=10000)
        with pytest.raises(ValueError):
            c._enter("streamed_s16", 4096, 32, None, 100)
        h.job.request_stop("cancelled")
        assert finish_job(h).own0
        return
    c = entered(h)
    if case == "time_overflow":
        h.now = 2**63 - 2
        h.job._deadline = c._deadline = 2**63 - 1
        with pytest.raises(ValueError):
            c._launch("audio_f32", ("ffmpeg",), 1000)
        assert c._child_count == 0 and c.ownership_snapshot().resources == ()
        return
    if case == "foreign_lease":
        from clipperx._source_host_ownership import ChildLease
        child = c._launch("audio_f32", ("ffmpeg",), 1000)
        with pytest.raises(ValueError):
            c._read_stdout(ChildLease(child.observation_id, child.generation, child.kind), 4)
        c._leave(False)
        assert finish_job(h).own0
        return
    with pytest.raises(ValueError):
        h.job.new_observation("streamed_s16", deadline_ns=9000)
    c._leave(True)
    assert c.ownership_snapshot().phase == "retired_ok"
    assert not h.job.snapshot().own0
    with pytest.raises(ValueError):
        c._enter("materialized_f32", 4096, 32, None, 100)
    second = entered(h, "streamed_s16")
    assert second._id != c._id
    second._leave(True)
    assert finish_job(h).own0


@pytest.mark.parametrize("case", ["normal", "late_launch", "launch_error", "late_factory", "reader_start_error", "metadata_tail",
    "launch_claim_gap", "metadata_claim_gap", "metadata_factory_gap", "finish_claim_gap", "claimed_metadata_stop"])
def test_child_launch_and_reader_start(case: str) -> None:
    h = admitted()
    c = entered(h)
    if case in ("launch_claim_gap", "metadata_claim_gap", "metadata_factory_gap", "finish_claim_gap"):
        calls = [0]
        check = c._check
        after_detach = []
        if case != "launch_claim_gap":
            child = c._launch("audio_f32", ("ffmpeg",), 1000)
        if case == "finish_claim_gap":
            def parse_all() -> None:
                c._readline_stderr(child, 16385)
                c._readline_stderr(child, 16385)
            c._start_metadata(child, parse_all)
            c._read_stdout(child, 4)
            c._read_stdout(child, 1)
            process = h.children[0]
            wait = process.wait
            def checked_wait(timeout: float) -> int:
                if process._handle.closed:
                    after_detach.append(True)
                    forbidden()
                return wait(timeout)
            process.wait = checked_wait
        def stop_in_gap() -> None:
            check()
            calls[0] += 1
            target = 2 if case == "metadata_claim_gap" else 1
            if calls[0] == target:
                h.job.request_stop("cancelled")
                h.current = None
                h.job.service_cleanup()
                h.current = h.driver
        c._check = stop_in_gap
        with pytest.raises(ValueError):
            if case == "launch_claim_gap":
                c._launch("audio_f32", ("ffmpeg",), 1000)
            elif case in ("metadata_claim_gap", "metadata_factory_gap"):
                c._start_metadata(child, forbidden)
            else:
                c._finish_child(child, True)
        if case == "launch_claim_gap":
            assert h.children == [], "no process factory after stop before claim"
        elif case == "metadata_claim_gap":
            assert h.workers[1].started == 0, "no metadata start after stop before claim"
            assert h.kernel.events.count("duplicate") == 1
        elif case == "metadata_factory_gap":
            assert len(h.workers) == 1 and h.kernel.events.count("duplicate") == 1
        else:
            assert not after_detach, "no stale Popen wait after detach"
        c._check = check
        c._leave(False)
        assert finish_job(h).own0
        return
    if case in ("late_launch", "launch_error"):
        process = h.job._processes
        def launch(argv: tuple[str, ...]) -> Any:
            status = c.ownership_snapshot()
            assert status.resources[0].in_flight
            if case == "launch_error":
                raise OSError("partial launch cannot be disproved")
            result = process(argv)
            h.job.request_stop("cancelled")
            return result
        h.job._processes = launch
        with pytest.raises(ValueError):
            c._launch("audio_f32", ("ffmpeg",), 1000)
        c._leave(False)
        assert finish_job(h).own0 is (case == "late_launch")
        return
    child = c._launch("audio_f32", ("ffmpeg", "-f", "f32le"), 1000)
    if case == "claimed_metadata_stop":
        factory = h.job._workers
        def create_claimed(target: Callable[[], None], name: str) -> ScriptWorker:
            worker = factory(target, name)
            def stop_after_claim() -> None:
                assert c.ownership_snapshot().resources[4].in_flight
                h.job.request_stop("cancelled")
                h.current = None
                h.job.service_cleanup()
                h.current = h.driver
                assert c._child is not None
            worker.start_action = stop_after_claim
            return worker
        h.job._workers = create_claimed
        with pytest.raises(ValueError):
            c._start_metadata(child, forbidden)
        assert h.workers[1].started == 1
        c._leave(False)
        assert finish_job(h).own0
        return
    if case in ("late_factory", "reader_start_error"):
        factory = h.job._workers
        def make(target: Callable[[], None], name: str) -> ScriptWorker:
            worker = factory(target, name)
            if case == "late_factory":
                h.job.request_stop("cancelled")
            else:
                worker.start_action = lambda: (_ for _ in ()).throw(OSError("start uncertainty"))
            return worker
        h.job._workers = make
        with pytest.raises(ValueError):
            c._start_metadata(child, forbidden)
        c._leave(False)
        assert finish_job(h).own0 is (case == "late_factory")
        return
    def parser() -> None:
        status = c.ownership_snapshot()
        assert any(r.role == "metadata_handle" and r.phase == "live" for r in status.resources)
        assert c._readline_stderr(child, 16385) == b"tag\n"
        assert c._readline_stderr(child, 16385) == b""
    c._start_metadata(child, parser)
    if case == "metadata_tail":
        c._child.reader_done = False
        h.workers[1].alive = True
        joined = h.workers[1].join
        def complete(timeout: float) -> None:
            joined(timeout)
            h.workers[1].alive = False
            c._child.reader_done = True
        h.workers[1].join = complete
    assert c._read_stdout(child, 4) == b"abcd"
    assert c._read_stdout(child, 1) == b""
    assert c._finish_child(child, True).natural_eof
    c._leave(True)
    assert len(h.workers) == 2 and len(h.children) == 1
    assert finish_job(h).own0


@pytest.mark.parametrize("case", ["read", "readinto", "line", "extra", "stop_partial", "no_pending", "exact_cap", "extra_at_cap", "transfer_error", "partial_readinto"])
def test_cancel_and_partial_io(case: str) -> None:
    h = admitted()
    c = entered(h)
    if case in ("exact_cap", "extra_at_cap"):
        c._max_decoded = 4
        c._max_log = 4
    if case == "extra_at_cap":
        h.job._processes = lambda argv: ScriptProcess(b"abcdx")
    child = c._launch("audio_f32", ("ffmpeg",), 1000)
    def read_log() -> None:
        c._readline_stderr(child, 16385)
        c._readline_stderr(child, 16385)
    c._start_metadata(child, read_log)
    if case == "partial_readinto":
        class Partial(io.BytesIO):
            def readinto(self, buffer: Any) -> int:
                h.job.request_stop("cancelled")
                return super().readinto(buffer[:2])
        h.children[0].stdout = Partial(b"abcd")
        with pytest.raises(ValueError):
            c._readinto_stdout(child, memoryview(bytearray(4)))
        diagnostic = c.ownership_snapshot().diagnostics[0]
        import hashlib
        assert diagnostic.stdout_bytes == 2 and diagnostic.stdout_sha256 == hashlib.sha256(b"ab").hexdigest()
        c._leave(False)
        assert finish_job(h).own0
        return
    if case == "transfer_error":
        class Broken(io.BytesIO):
            def read(self, size: int | None = -1) -> bytes:
                raise OSError("unknown partial transfer")
        h.children[0].stdout = Broken()
        with pytest.raises(ValueError):
            c._read_stdout(child, 4)
        assert c.ownership_snapshot().diagnostics[0].transfer_uncertain
        c._leave(False)
        assert finish_job(h).own0
        return
    if case == "no_pending":
        h.kernel.CancelSynchronousIo.action = lambda handle: 0
        h.kernel.error = 1168
        h.job.request_stop("cancelled")
        h.current = None
        assert not h.job.service_cleanup().own0
        assert h.job.snapshot().failure.code == "cancelled"
        return
    if case == "readinto":
        raw = bytearray(4)
        assert c._readinto_stdout(child, memoryview(raw)) == 4
        assert raw == b"abcd"
    elif case == "line":
        assert c.ownership_snapshot().diagnostics[0].log_bytes == 4
    else:
        assert c._read_stdout(child, 4) == b"abcd"
    if case == "extra":
        assert c._read_stdout(child, 1) == b""
    if case == "exact_cap":
        assert c._read_stdout(child, 1) == b""
        c._finish_child(child, True)
        c._leave(True)
        assert finish_job(h).own0
        return
    if case == "extra_at_cap":
        with pytest.raises(ValueError):
            c._read_stdout(child, 1)
        assert c.ownership_snapshot().diagnostics[0].stdout_bytes == 5
    if case == "stop_partial":
        h.job.request_stop("cancelled")
        with pytest.raises(ValueError):
            c._read_stdout(child, 1)
        assert c.ownership_snapshot().diagnostics[0].stdout_bytes == 4
    c._leave(False)
    assert finish_job(h).own0


@pytest.mark.parametrize("case", ["normal", "close_uncertain", "detach_uncertain", "read_inflight", "metadata_survives", "wait_timeout", "foreign_wait_claim", "joined_claim", "diagnostic_handoff"])
def test_retirement_and_uncertainty(case: str) -> None:
    h = admitted()
    if case == "diagnostic_handoff":
        import sys
        c = entered(h, "streamed_s16")
        def consume(kind: str) -> Any:
            lease = c._launch(kind, ("ffmpeg",), 1000)
            def log_all() -> None:
                c._readline_stderr(lease, 16385)
                c._readline_stderr(lease, 16385)
            c._start_metadata(lease, log_all)
            assert c._read_stdout(lease, 4) == b"abcd"
            assert c._read_stdout(lease, 1) == b""
            return lease
        video = consume("video")
        c._finish_child(video, True)
        assert len(c.ownership_snapshot().diagnostics) == 1
        audio = consume("audio_s16")
        current_audio = c._child
        before = c.ownership_snapshot()
        assert len(before.diagnostics) == 2 and before.child_deadline_ns is not None
        assert not before.window_resources_zero
        assert type(c._diagnostics) is list
        original_lock = h.job._lock
        actor = ["retire"]
        pending: list[bool] = []
        snapshots = []
        errors = []
        attempts: list[bool] = []
        class DeferredSnapshot(Exception):
            pass
        def competing_snapshot() -> None:
            prior, actor[0] = actor[0], "snapshot"
            try:
                snapshots.append((c.ownership_snapshot(), h.job.snapshot()))
            except DeferredSnapshot:
                pending.append(True)
            except ValueError as exc:
                errors.append(str(exc))
            finally:
                actor[0] = prior
        class PublishLock:
            owner: str | None = None
            depth = 0
            def __enter__(self) -> None:
                if self.owner is not None and self.owner != actor[0]:
                    raise DeferredSnapshot()
                original_lock.__enter__()
                self.owner = actor[0]
                self.depth += 1
            def __exit__(self, *args: Any) -> None:
                original_lock.__exit__(*args)
                self.depth -= 1
                if self.depth == 0:
                    self.owner = None
                    if pending:
                        pending.pop()
                        competing_snapshot()
            def _is_owned(self) -> bool:
                return original_lock._is_owned()
        target_code = c._cleanup_child_claimed.__func__.__code__
        def switch_at_handoff(frame: Any, event: str, arg: Any) -> Any:
            if (event == "line" and frame.f_code is target_code and not attempts
                    and len(c._diagnostics) == 2 and c._child is current_audio):
                attempts.append(True)
                competing_snapshot()
            return switch_at_handoff
        h.job._lock = PublishLock()
        previous_trace = sys.gettrace()
        try:
            sys.settrace(switch_at_handoff)
            c._finish_child(audio, True)
        finally:
            sys.settrace(previous_trace)
            h.job._lock = original_lock
        assert len(attempts) == 1, "script must reach the second-child publication boundary"
        assert not errors, errors
        assert len(snapshots) == 1 and not pending
        observed, job = snapshots[0]
        assert len(observed.diagnostics) == 2
        assert len({d.generation for d in observed.diagnostics}) == 2
        assert tuple(d.kind for d in observed.diagnostics) == ("video", "audio_s16")
        assert observed.phase == "active" and observed.scope_open
        assert observed.child_deadline_ns is None and job.current == observed
        assert observed.window_resources_zero and not observed.cleanup_required
        assert not job.own0 and job.failure is None
        c._leave(True)
        assert c.ownership_snapshot().phase == "retired_ok"
        assert finish_job(h).own0
        return
    c = entered(h)
    child = c._launch("audio_f32", ("ffmpeg",), 1000)
    def log() -> None:
        c._readline_stderr(child, 16385)
        c._readline_stderr(child, 16385)
    c._start_metadata(child, log)
    if case == "joined_claim":
        worker, handle = c._child.rows[4:6]
        claim = h.job._claim(worker)
        # A returned join can publish its result while its claim is still owned.
        worker.update(phase="joined", start="joined", alive=False)
        before = h.kernel.events.count("close")
        h.current = None
        assert not h.job._retire_worker(worker, handle)
        assert h.kernel.events.count("close") == before
        h.job._release_claim(worker, claim)
        h.current = h.driver
        c._leave(False)
        assert finish_job(h).own0
        return
    if case == "foreign_wait_claim":
        from dataclasses import replace
        active = c._child
        proc = active.rows[1]
        original_lock = h.job._lock
        armed = [True]
        class ClaimAtBoundary:
            def __enter__(self) -> None:
                original_lock.__enter__()
                if armed[0] and active.terminated and not proc.row.in_flight:
                    armed[0] = False
                    proc.row = replace(proc.row, in_flight=True)
            def __exit__(self, *args: Any) -> None:
                original_lock.__exit__(*args)
            def _is_owned(self) -> bool:
                return original_lock._is_owned()
        h.job._lock = ClaimAtBoundary()
        c._cleanup_deadline = 110
        c._leave(False)
        assert proc.row.in_flight, "cleanup must not clear a different action's claim"
        h.job._lock = original_lock
        proc.update(in_flight=False)
        status = finish_job(h)
        assert status.own0 and status.current.phase == "stopped_clean"
        return
    if case == "wait_timeout":
        c._read_stdout(child, 4)
        c._read_stdout(child, 1)
        wait = h.children[0].wait
        calls = [0]
        def timeout_once(timeout: float) -> int:
            calls[0] += 1
            if calls[0] == 1:
                raise subprocess.TimeoutExpired("script", timeout)
            return wait(timeout)
        h.children[0].wait = timeout_once
        with pytest.raises(ValueError):
            c._finish_child(child, True)
    if case == "metadata_survives":
        h.workers[1].alive = True
        h.job.request_stop("cancelled")
        h.job.driver_done()
        h.driver.alive = False
        h.current = None
        h.job.service_cleanup()
        assert not h.job._watch_exit, "metadata survivor still requires watchdog cancellation"
        assert not h.children[0].stdout.closed and not h.children[0].stderr.closed
        h.workers[1].alive = False
        h.current = h.driver
    if case == "close_uncertain":
        h.kernel.CloseHandle.action = lambda handle: 0
    if case == "detach_uncertain":
        h.children[0]._handle.mode = "raises"
    if case == "read_inflight":
        class StoppedRead(io.BytesIO):
            def read(self, size: int | None = -1) -> bytes:
                h.job.request_stop("cancelled")
                h.current = None
                s = h.job.service_cleanup()
                assert not h.children[0].stdout.closed and not s.own0
                h.current = h.driver
                return super().read(size)
        h.children[0].stdout = StoppedRead(b"abcd")
        with pytest.raises(ValueError):
            c._read_stdout(child, 4)
    c._leave(False)
    status = finish_job(h)
    if case in ("close_uncertain", "detach_uncertain"):
        assert not status.own0
        calls = h.kernel.events.count("close")
        h.job.service_cleanup()
        assert h.kernel.events.count("close") == calls
    else:
        assert status.own0
    assert h.children[0].actions.index("terminate") < h.children[0].actions.index("wait")


@pytest.mark.parametrize("case", ["primary", "job_equal", "child_equal", "retained", "secondary_overflow",
    "watch_control", "watch_child", "watch_wake", "publish_wake", "service_control", "retired_gap",
    "retired_stop", "retired_stop_bound", "open_zero_stop"])
def test_deadline_and_primary_failure(case: str) -> None:
    h = admitted()
    if case in ("retired_stop", "retired_stop_bound", "open_zero_stop"):
        open_scope = case == "open_zero_stop"
        c = entered(h, "streamed_s16" if open_scope else "materialized_f32")
        c._retain_cleanup_deadline(50)
        lease = c._launch("video" if open_scope else "audio_f32", ("ffmpeg",), 1000)
        def read_all_log() -> None:
            c._readline_stderr(lease, 16385)
            c._readline_stderr(lease, 16385)
        c._start_metadata(lease, read_all_log)
        assert c._read_stdout(lease, 4) == b"abcd"
        assert c._read_stdout(lease, 1) == b""
        assert c._finish_child(lease, True).natural_eof
        if not open_scope:
            c._leave(True)
        prior = c.ownership_snapshot()
        assert prior.window_resources_zero and not prior.cleanup_required
        assert prior.scope_open is open_scope
        assert prior.phase == ("active" if open_scope else "retired_ok")
        assert not h.job.snapshot().own0 and h.job.snapshot().failure is None
        h.now = 60
        h.job.check()
        bound = 140 if case == "retired_stop_bound" else None
        failure = h.job.request_stop("cancelled", deadline_ns=bound).failure
        expected = 50 if open_scope else 140 if bound is not None else 160
        assert failure.cleanup_deadline_ns == expected
        assert failure.code == "cancelled" and failure.first_ns == 60
        h.now = 70
        h.job.fail("io", "driver")
        assert h.job.snapshot().failure.cleanup_deadline_ns == expected
        h.job.request_stop("deadline", deadline_ns=expected + 1)
        assert h.job.snapshot().failure.cleanup_deadline_ns == expected
        h.job.request_stop("cancelled", deadline_ns=expected - 1)
        final = h.job.snapshot().failure
        assert final.cleanup_deadline_ns == expected - 1
        assert (final.code, final.stage, final.first_ns) == (failure.code, failure.stage, failure.first_ns)
        if open_scope:
            with pytest.raises(ValueError):
                c._launch("audio_s16", ("ffmpeg",), 1000)
            assert len(h.children) == 1
            c._leave(False)
        assert finish_job(h).own0
        return
    if case == "publish_wake":
        c = h.job.new_observation("materialized_f32", deadline_ns=10000)
        assert h.events[0].is_set(), "new control must wake the watchdog"
        h.events[0].clear()
        c._enter("materialized_f32", 4096, 32, None, 100)
        assert h.events[0].is_set(), "entry budget publication must wake the watchdog"
        h.events[0].clear()
        c._launch("audio_f32", ("ffmpeg",), 1000)
        assert h.events[0].is_set(), "new child deadline must wake the watchdog"
        c._leave(False)
        assert finish_job(h).own0
        return
    if case in ("watch_control", "watch_child", "watch_wake", "service_control", "retired_gap"):
        c = entered(h)
        if case == "retired_gap":
            c._leave(True)
            h.now = 10000
            assert h.job.service_cleanup().failure is None
            assert finish_job(h).own0
            return
        deadline = 10000
        if case in ("watch_child", "watch_wake"):
            child = c._launch("audio_f32", ("ffmpeg",), 1000)
            c._start_metadata(child, lambda: None)
            deadline = 1010
        h.now = deadline - 10 if case == "watch_wake" else deadline
        waits = []
        def one_wait(timeout: float) -> bool:
            waits.append(timeout)
            if case == "watch_wake" and len(waits) == 1:
                assert timeout == 10 / 10**9
                h.now = deadline
                return False
            h.job._watch_exit = True
            return False
        h.events[0].wait = one_wait
        h.current = None if case == "service_control" else h.workers[0]
        if case == "service_control":
            h.job.service_cleanup()
        else:
            h.workers[0].target()
        status = h.job.snapshot()
        assert status.failure is not None and status.failure.code == "deadline"
        assert status.failure.first_ns == deadline
        assert h.kernel.events.count("cancel") >= 1
        assert len(waits) <= (2 if case == "watch_wake" else 1)
        assert all(0 <= wait <= .01 for wait in waits)
        h.current = h.driver
        c._leave(False)
        assert finish_job(h).own0
        return
    if case == "job_equal":
        h.now = 1_000_000_000
        with pytest.raises(ValueError):
            h.job.check()
        assert h.job.snapshot().failure.code == "deadline"
        assert finish_job(h).own0
        return
    c = entered(h)
    child = c._launch("audio_f32", ("ffmpeg",), 1000)
    c._start_metadata(child, lambda: c._readline_stderr(child, 16385))
    if case == "child_equal":
        h.now = 1010
        with pytest.raises(ValueError):
            c._read_stdout(child, 4)
        assert c.ownership_snapshot().diagnostics[0].stdout_bytes == 0
        c._leave(False)
        assert finish_job(h).own0
        return
    if case == "retained":
        c._retain_cleanup_deadline(50)
        c._fail("validation", "validate")
        h.now = 20
        c.request_stop("cancelled", deadline_ns=90)
        assert h.job.snapshot().failure.cleanup_deadline_ns == 50
        c._leave(False)
        assert finish_job(h).own0
        return
    c._read_stdout(child, 4)
    c._fail("validation", "validate")
    if case == "secondary_overflow":
        from dataclasses import replace
        h.job._failure = replace(h.job._failure, secondary_count=2**63 - 1)
        c._fail("cleanup", "retire")
        assert h.job.snapshot().failure.secondary_count == 2**63 - 1
        assert h.job.snapshot().failure.last_secondary == "overflow"
    primary = h.job.snapshot().failure
    assert primary.cleanup_deadline_ns == 110
    h.now = 20
    c._fail("cleanup", "retire")
    assert h.job.snapshot().failure.first_ns == primary.first_ns
    assert h.job.snapshot().failure.cleanup_deadline_ns == 110
    assert c.ownership_snapshot().diagnostics[0].stdout_bytes == 4
    h.now = 10000
    with pytest.raises(ValueError):
        c._check()
    assert h.job.snapshot().failure.code == "validation"
    c._leave(False)
    assert finish_job(h).own0
