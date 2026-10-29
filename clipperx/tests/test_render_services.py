"""Serial render adapter integration using the real owner and fake IO only."""

import hashlib
import io
import queue
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from clipperx.source_decode import probe_source as REAL_PROBE_SOURCE

from clipperx.tests.test_retained_files import memory as memory, ROOT
from clipperx.tests.test_source_host_ownership import (
    ScriptProcess, closed_boundaries as closed_boundaries, finish_job,
)

_QUEUE_GET, _QUEUE_PUT = queue.Queue.get, queue.Queue.put


def tools_for(c: Any, payload: bytes, log: bytes = b"tag\n") -> Any:
    from clipperx._render_services import OwnedRenderTools
    def factory(argv: tuple[str, ...]) -> ScriptProcess:
        process = ScriptProcess(payload, log)
        c.h.children.append(process)
        return process
    c.h.job._processes = factory
    return OwnedRenderTools(c.h.job, c.files, clock=c.h.job._clock)


def scripted_cleanup_wait(c: Any, step: Any) -> list[float]:
    """Finite test scheduler; callbacks run with the actual condition unlocked."""
    condition = c.h.job._cleanup_condition
    waits: list[float] = []
    def wait(timeout: float) -> bool:
        assert condition._is_owned()
        assert 0 < timeout <= .01 and len(waits) < 10
        assert c.h.job._cleanup_token is None
        end = c.h.job._failure.cleanup_deadline_ns
        if c.h.job._control._cleanup_deadline is not None:
            end = min(end, c.h.job._control._cleanup_deadline)
        assert timeout <= (end - c.h.now) / 1e9
        waits.append(timeout)
        saved = condition._release_save()
        try:
            assert not c.h.job._lock._is_owned()
            control = c.h.job._control
            assert control._retirement_lock.acquire(blocking=False)
            control._retirement_lock.release()
            assert c.h.job._cleanup_lock.acquire(blocking=False)
            c.h.job._cleanup_lock.release()
            step(len(waits))
        finally:
            condition._acquire_restore(saved)
        return False
    condition.wait = wait
    return waits


def delayed_render(c: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Real owner and file custody, with only process/worker scheduling faked."""
    from clipperx._render_services import OwnedRenderTools
    h, path = c.h, ROOT / "delayed.bgr"
    def clock() -> int:
        assert not h.job._lock._is_owned(), "clock callback under state lock"
        return h.now
    h.job._clock = clock
    c.files.register(path, role="raw", owned=True, max_bytes=8)
    with c.files.open_file(path, "xb") as stream:
        stream.write(memoryview(b"abcd"))
    state = SimpleNamespace(path=path, reaped=False, metadata=None, original=None,
        failure=None, releases=[], leave_calls=0)
    raise_failure = h.job._raise
    def capture_failure() -> Any:
        try:
            raise_failure()
        except ValueError as error:
            if state.original is None:
                state.original = error
            raise
    monkeypatch.setattr(h.job, "_raise", capture_failure)
    workers = h.job._workers
    def worker(target: Any, name: str) -> Any:
        result = workers(target, name)
        assert name == "source-host-metadata"
        state.metadata = result
        def start() -> None:
            result.started += 1
            result.ident = result.started
            result.alive = True
        result.start = start
        return result
    h.job._workers = worker
    class CancelPipe(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            h.job.request_stop("cancelled")
            state.failure = h.job._failure
            raise OSError(22, "synthetic operation aborted")
    child = ScriptProcess(b"", b"")
    child.stdout = CancelPipe()
    def process(argv: tuple[str, ...]) -> ScriptProcess:
        assert c.files._busy(path)
        h.children.append(child)
        return child
    def wait(timeout: float) -> int:
        assert timeout == 0 and not h.job._lock._is_owned()
        child.actions.append("wait")
        if not state.reaped:
            raise subprocess.TimeoutExpired("fake", timeout)
        child.returncode = 1
        return 1
    monkeypatch.setattr(child, "wait", wait)
    h.job._processes = process
    from clipperx._source_host_ownership import HostedObservationControl
    leave = HostedObservationControl._leave
    def counted_leave(control: Any, success: bool) -> None:
        if control._job is h.job:
            state.leave_calls += 1
        leave(control, success)
    monkeypatch.setattr(HostedObservationControl, "_leave", counted_leave)
    release = c.files.release_consumer
    def releasing(token: int, *, retired: bool) -> None:
        assert not h.job._lock._is_owned()
        assert h.current is h.driver and h.job._registered and not h.job._done
        if retired:
            status = h.job._control.ownership_snapshot()
            assert status.phase == "stopped_clean" and status.window_resources_zero
            assert not status.scope_open and not status.consumer_active
            assert not status.payload_retained and not status.lease_outstanding and not status.cleanup_required
        state.releases.append(retired)
        release(token, retired=retired)
    monkeypatch.setattr(c.files, "release_consumer", releasing)
    state.tools = OwnedRenderTools(h.job, c.files, clock=clock)
    state.child = child
    return state


def run_cancelled(c: Any, state: Any) -> None:
    with pytest.raises(ValueError) as raised:
        state.tools.run(("fake",), kind="spatial", timeout=900, max_stdout_bytes=4,
            max_stderr_bytes=100, expected_stdout_bytes=4, inputs=(state.path,))
    assert raised.value is state.original
    failure = c.h.job._failure
    assert (failure.code, failure.first_ns, failure.cleanup_deadline_ns) == (
        state.failure.code, state.failure.first_ns, state.failure.cleanup_deadline_ns)
    assert state.leave_calls == 1


def finish_metadata(c: Any, state: Any) -> None:
    prior, c.h.current = c.h.current, state.metadata
    try:
        state.metadata.target()
    finally:
        c.h.current = prior
        state.metadata.alive = False


def test_delayed_process_and_metadata_release_before_driver_done(memory: Any,
        monkeypatch: pytest.MonkeyPatch) -> None:
    c = memory
    state = delayed_render(c, monkeypatch)
    def step(index: int) -> None:
        assert c.files._busy(state.path) and state.path in c.fs.data
        assert c.h.job._control.ownership_snapshot().cleanup_required
        assert not c.files.retire().files_clear
        c.h.now += 10
        if index == 1:
            assert state.child.actions.count("wait") >= 2
            state.reaped = True
        else:
            assert index == 2 and state.child.returncode == 1 and state.metadata.alive
            finish_metadata(c, state)
    waits = scripted_cleanup_wait(c, step)
    run_cancelled(c, state)
    assert not c.files.snapshot().consumers, "delayed retired child must release original input claim"
    assert len(waits) == 2 and state.releases == [True]
    status = c.h.job._control.ownership_snapshot()
    assert status.diagnostics[-1].transfer_uncertain
    assert all(not row.uncertain and not row.in_flight for row in status.resources)
    assert len(c.h.children) == 1 and len(c.h.workers) == 2
    assert c.files.retire().files_clear and state.path not in c.fs.data
    assert finish_job(c.h).own0


@pytest.mark.parametrize("fault", ["process", "metadata", "equality", "shortened"])
def test_delayed_retirement_keeps_claim_at_existing_end(memory: Any,
        monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    c = memory
    state = delayed_render(c, monkeypatch)
    original_end = c.h.now + c.h.job._grace
    def step(index: int) -> None:
        assert c.files._busy(state.path) and state.path in c.fs.data
        control = c.h.job._control
        if fault == "shortened":
            control._retain_cleanup_deadline(c.h.now + 1)
            state.failure = c.h.job._failure
        if fault in ("process", "metadata") and index == 1:
            c.h.now += 10
            if fault == "metadata":
                state.reaped = True
            return
        assert index <= 2
        if fault in ("equality", "shortened"):
            state.reaped = True
            finish_metadata(c, state)
            # Publish genuine terminal proof, but only at the authoritative end.
            control._cleanup_child()
            assert control.ownership_snapshot().phase == "stopped_clean"
        c.h.now = c.h.job._failure.cleanup_deadline_ns
    waits = scripted_cleanup_wait(c, step)
    run_cancelled(c, state)
    assert len(waits) in (1, 2) and state.releases == []
    assert c.h.job._failure.cleanup_deadline_ns <= original_end
    assert c.files.snapshot().consumers and not c.files.retire().files_clear
    assert state.path in c.fs.data and ("unlink", state.path) not in c.fs.events
    finish_job(c.h)


def test_release_checks_deadline_again_after_terminal_observation(memory: Any,
        monkeypatch: pytest.MonkeyPatch) -> None:
    c = memory
    state = delayed_render(c, monkeypatch)
    def step(index: int) -> None:
        assert index == 1
        c.h.now += 10
        state.reaped = True
        finish_metadata(c, state)
    scripted_cleanup_wait(c, step)
    release = c.files.release_consumer
    def expired(token: int, *, retired: bool) -> None:
        c.h.now = c.h.job._failure.cleanup_deadline_ns
        release(token, retired=retired)
    monkeypatch.setattr(c.files, "release_consumer", expired)
    run_cancelled(c, state)
    assert state.releases == [True]
    assert c.files.snapshot().consumers and not c.files.retire().files_clear
    assert state.path in c.fs.data
    assert finish_job(c.h).own0


@pytest.mark.parametrize("foreign", [False, True])
def test_clean_other_control_cannot_release_captured_consumer(memory: Any,
        monkeypatch: pytest.MonkeyPatch, foreign: bool) -> None:
    from clipperx.tests.test_source_host_ownership import admitted
    c = memory
    other_job = admitted().job if foreign else c.h.job
    clean = other_job.new_render("probe", deadline_ns=10000)
    clean._enter("probe", 100, 4, None, 100)
    clean._leave(True)
    assert clean.ownership_snapshot().phase == "retired_ok"
    state = delayed_render(c, monkeypatch)
    captured: list[Any] = []
    def step(index: int) -> None:
        assert index == 1
        captured.append(c.h.job._control)
        c.h.job._control = clean
        c.h.now += 10
    waits = scripted_cleanup_wait(c, step)
    run_cancelled(c, state)
    assert len(waits) == 1 and state.releases == []
    assert captured[0].ownership_snapshot().cleanup_required
    assert c.files.snapshot().consumers and state.path in c.fs.data
    # Restore only the test's identity fault so the real owner still reports its unresolved child.
    c.h.job._control = captured[0]
    assert not c.files.retire().files_clear
    assert not finish_job(c.h).own0


def test_wrong_terminal_diagnostic_generation_retains_claim(memory: Any,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from clipperx._source_host_ownership import HostedObservationControl
    c = memory
    state = delayed_render(c, monkeypatch)
    cleanup = HostedObservationControl._cleanup_child
    def wrong(control: Any) -> None:
        cleanup(control)
        if control._job is c.h.job and control._child is None and control._diagnostics:
            diagnostic = control._diagnostics[-1]
            control._diagnostics[-1] = replace(diagnostic, generation=diagnostic.generation + 1)
    monkeypatch.setattr(HostedObservationControl, "_cleanup_child", wrong)
    def step(index: int) -> None:
        assert index == 1
        c.h.now += 10
        state.reaped = True
        finish_metadata(c, state)
    waits = scripted_cleanup_wait(c, step)
    run_cancelled(c, state)
    assert len(waits) == 1 and state.releases == []
    assert c.files.snapshot().consumers and not c.files.retire().files_clear
    assert state.path in c.fs.data and finish_job(c.h).own0


@pytest.mark.parametrize("field", ["generation", "observation_id", "kind"])
def test_wrong_returned_lease_keeps_input_claim(memory: Any,
        monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    from dataclasses import replace
    from clipperx._source_host_ownership import HostedObservationControl
    c, path = memory, ROOT / "wrong-lease.bgr"
    c.files.register(path, role="raw", owned=True, max_bytes=8)
    with c.files.open_file(path, "xb") as stream:
        stream.write(memoryview(b"abcd"))
    launch = HostedObservationControl._launch
    def wrong(control: Any, *args: Any) -> Any:
        lease = launch(control, *args)
        if field == "kind":
            return replace(lease, kind="encoder")
        if field == "generation":
            return replace(lease, generation=lease.generation + 1)
        return replace(lease, observation_id=lease.observation_id + 1)
    monkeypatch.setattr(HostedObservationControl, "_launch", wrong)
    tools = tools_for(c, b"abcd", b"")
    with pytest.raises(ValueError):
        tools.run(("fake",), kind="spatial", timeout=900, max_stdout_bytes=4,
            max_stderr_bytes=100, expected_stdout_bytes=4, inputs=(path,))
    assert c.h.job._control.ownership_snapshot().window_resources_zero
    assert c.files.snapshot().consumers and not c.files.retire().files_clear
    assert path in c.fs.data and finish_job(c.h).own0


@pytest.mark.parametrize("fault", ["pipe", "process_handle"])
def test_uncertain_close_is_never_replayed_or_released(memory: Any,
        monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    c = memory
    state = delayed_render(c, monkeypatch)
    close_calls: list[str] = []
    if fault == "pipe":
        def close() -> None:
            close_calls.append("pipe")
            raise OSError("unknown pipe close")
        monkeypatch.setattr(state.child.stdout, "close", close)
    else:
        close_handle = c.h.kernel.CloseHandle.action
        def close_native(handle: int) -> Any:
            if handle == state.child.pid + 1000:
                close_calls.append("process_handle")
                raise OSError("unknown process handle close")
            return close_handle(handle)
        c.h.kernel.CloseHandle.action = close_native
    def step(index: int) -> None:
        if index == 1:
            c.h.now += 10
            state.reaped = True
            finish_metadata(c, state)
        else:
            assert index == 2 and close_calls == [fault]
            c.h.now = c.h.job._failure.cleanup_deadline_ns
    waits = scripted_cleanup_wait(c, step)
    run_cancelled(c, state)
    assert len(waits) == 2 and close_calls == [fault] and state.releases == []
    assert any(row.uncertain for row in c.h.job._control.ownership_snapshot().resources)
    assert c.files.snapshot().consumers and not c.files.retire().files_clear
    assert state.path in c.fs.data and not finish_job(c.h).own0
    assert close_calls == [fault]


@pytest.mark.parametrize("fault", ["leave", "advance", "release"])
def test_cleanup_exception_preserves_first_unwinding_error(memory: Any,
        monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    from clipperx._source_host_ownership import HostedObservationControl
    c = memory
    state = delayed_render(c, monkeypatch)
    if fault == "leave":
        leave = HostedObservationControl._leave
        def broken_leave(control: Any, success: bool) -> None:
            leave(control, success)
            raise RuntimeError("cleanup leave failed")
        monkeypatch.setattr(HostedObservationControl, "_leave", broken_leave)
    elif fault == "advance":
        cleanup = HostedObservationControl._cleanup_child
        def broken_advance(control: Any) -> None:
            if state.leave_calls:
                cleanup(control)
                if state.child.actions.count("wait") >= 2:
                    raise RuntimeError("cleanup advance failed")
        monkeypatch.setattr(HostedObservationControl, "_cleanup_child", broken_advance)
    else:
        def step(index: int) -> None:
            assert index == 1
            c.h.now += 10
            state.reaped = True
            finish_metadata(c, state)
        scripted_cleanup_wait(c, step)
        def broken_release(token: int, *, retired: bool) -> None:
            state.releases.append(retired)
            raise OSError("uncertain output stat")
        monkeypatch.setattr(c.files, "release_consumer", broken_release)
    run_cancelled(c, state)
    assert state.releases == ([True] if fault == "release" else [])
    assert state.tools._evidence_bytes == len("fake")
    assert c.files.snapshot().consumers and not c.files.retire().files_clear
    if fault == "advance":
        monkeypatch.setattr(HostedObservationControl, "_cleanup_child", cleanup)
    finish_job(c.h)


@pytest.mark.parametrize("fault", ["authority", "driver", "unregistered", "done", "control_job"])
def test_settlement_without_original_authority_keeps_claim(memory: Any, fault: str) -> None:
    from clipperx._render_services import OwnedRenderTools
    from clipperx.tests.test_source_host_ownership import admitted
    c, path = memory, ROOT / "authority.bgr"
    c.files.register(path, role="raw", owned=True, max_bytes=8)
    with c.files.open_file(path, "xb") as stream:
        stream.write(memoryview(b"abcd"))
    token = c.files.acquire_consumer((path,))
    control = c.h.job.new_render("probe", deadline_ns=10000)
    control._enter("probe", 100, 4, None, 100)
    control._leave(True)
    assert control.ownership_snapshot().phase == "retired_ok"
    if fault != "authority":
        c.h.job.fail("validation", "driver")
    if fault == "driver":
        c.h.current = None
    elif fault == "unregistered":
        c.h.job._registered = False
    elif fault == "done":
        c.h.job.driver_done()
    elif fault == "control_job":
        control._job = admitted().job
    tools = OwnedRenderTools(c.h.job, c.files, clock=c.h.job._clock)
    assert not tools._settle_consumer(control, token, "probe", None)
    assert c.files.snapshot().consumers == (token,) and path in c.fs.data
    c.h.current, c.h.job._registered, control._job = c.h.driver, True, c.h.job
    assert finish_job(c.h).own0


def test_success_path_release_stat_exception_is_not_replayed(memory: Any,
        monkeypatch: pytest.MonkeyPatch) -> None:
    c, output = memory, ROOT / "release-error.mp4"
    c.files.register(output, role="candidate", owned=True, max_bytes=8)
    tools = tools_for(c, b"", b"")
    calls: list[int] = []
    release = c.files.release_consumer
    def releasing(token: int, *, retired: bool) -> None:
        calls.append(token)
        release(token, retired=retired)
    monkeypatch.setattr(c.files, "release_consumer", releasing)
    error = OSError("unknown output stat")
    def fail_stat(operation: str, path: Path) -> None:
        if operation == "stat" and path == output:
            raise error
    c.fs.hook = fail_stat
    with pytest.raises(OSError) as raised:
        tools.run(("fake",), kind="encoder", timeout=900, max_stdout_bytes=1,
            max_stderr_bytes=100, expected_stdout_bytes=0, outputs=(output,))
    assert raised.value is error and calls == [1]
    assert c.files.snapshot().consumers and not c.files.retire().files_clear
    assert c.h.job._control.ownership_snapshot().window_resources_zero
    assert finish_job(c.h).own0


@pytest.mark.parametrize("kind", ["probe", "spatial", "finalmix", "encoder"])
def test_distinct_serial_render_admission(memory: Any, kind: str) -> None:
    c = memory
    with pytest.raises(ValueError):
        c.h.job.new_observation(kind, deadline_ns=10000)
    control = c.h.job.new_render(kind, deadline_ns=10000)
    control._enter(kind, 1024, 1 if kind == "encoder" else 8, None, 100)
    with pytest.raises(ValueError):
        c.h.job.new_normalization(deadline_ns=10000)
    with pytest.raises(ValueError):
        control._launch("audio_f32", ("fake",), 100)
    control._leave(True)
    assert control.ownership_snapshot().kind == kind
    assert control.ownership_snapshot().transferred_pcm_bytes == 0
    assert finish_job(c.h).own0


def test_probe_captures_actual_bounded_output(memory: Any) -> None:
    c = memory
    tools = tools_for(c, b"abcd")
    result = tools.run(("fake",), kind="probe", timeout=15,
        max_stdout_bytes=4, max_stderr_bytes=100)
    assert result.stdout == b"abcd" and result.stdout_bytes == 4
    assert result.stdout_sha256 == hashlib.sha256(b"abcd").hexdigest()
    assert result.natural_eof and result.retired and result.exit_code == 0
    assert result.stderr_sha256 == hashlib.sha256(b"tag\n").hexdigest()
    assert c.files.snapshot().files_clear
    assert finish_job(c.h).own0


def test_sink_streams_known_short_writes_without_capture(memory: Any) -> None:
    c, path = memory, ROOT / "shot.raw"
    c.files.register(path, role="raw", owned=True, max_bytes=8)
    c.fs.write_chunk = 2
    tools = tools_for(c, b"abcd")
    with c.files.open_file(path, "xb") as sink:
        result = tools.run(("fake",), kind="spatial", timeout=900,
            max_stdout_bytes=4, max_stderr_bytes=100, expected_stdout_bytes=4, sink=sink)
    assert result.stdout is None and result.stdout_bytes == 4
    assert c.fs.data[path] == b"abcd"
    assert c.h.job.snapshot().current.transferred_pcm_bytes == 0
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


@pytest.mark.parametrize("payload", [b"abc", b"abcde"])
def test_exact_count_rejects_short_and_extra(memory: Any, payload: bytes) -> None:
    c = memory
    tools = tools_for(c, payload)
    with pytest.raises(ValueError):
        tools.run(("fake",), kind="finalmix", timeout=120,
            max_stdout_bytes=4, max_stderr_bytes=100, expected_stdout_bytes=4)
    assert c.h.job.snapshot().failure is not None
    assert finish_job(c.h).own0


def test_encoder_zero_stdout_and_output_claim_before_launch(memory: Any) -> None:
    c, path = memory, ROOT / "candidate.mp4"
    c.files.register(path, role="candidate", owned=True, max_bytes=32)
    tools = tools_for(c, b"")
    factory = c.h.job._processes
    def process(argv: tuple[str, ...]) -> ScriptProcess:
        assert c.files.snapshot().consumers
        c.fs.data[path] = b"mp4"
        return factory(argv)
    c.h.job._processes = process
    result = tools.run(("fake", str(path)), kind="encoder", timeout=900,
        max_stdout_bytes=1, max_stderr_bytes=100, expected_stdout_bytes=0, outputs=(path,))
    assert result.stdout_bytes == 0 and result.retired
    assert not c.files.snapshot().consumers
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


def test_resolver_owned_hashes_and_source_identity(memory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx import render_runtime as rr
    from clipperx.local_paths import source_fingerprint
    c = memory
    pair = (Path("O:/HOST/ffmpeg.exe"), Path("O:/HOST/ffprobe.exe"))
    monkeypatch.setattr(rr, "_runtime_paths", lambda _, **kwargs: pair)
    for path, raw in zip(pair, (b"ffmpeg", b"ffprobe")):
        c.fs.data[path] = raw
        c.files.register(path, role="runtime", owned=False, max_bytes=100)
    runtime = rr.resolve_render_runtime(rr.RenderRequirements(), _files=c.files)
    assert runtime.ffmpeg_sha256 == hashlib.sha256(b"ffmpeg").hexdigest().upper()
    source = Path("O:/HOST/source.mp4")
    c.fs.data[source] = b"source"
    c.files.register(source, role="original", owned=False, max_bytes=100)
    expected = hashlib.sha256(f"{source}|6|100".encode()).hexdigest()[:16]
    assert source_fingerprint(source, _files=c.files) == expected
    assert not any(op == "open" and path == source for op, path in c.fs.events)
    assert c.files.snapshot().files_clear
    assert finish_job(c.h).own0


def test_runtime_qualifier_uses_owned_runner(memory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx import render_runtime as rr
    c = memory
    runtime = rr.RenderRuntime(Path("O:/HOST/ffmpeg.exe"), Path("O:/HOST/ffprobe.exe"),
        "system", "A" * 64, "B" * 64, None, None, "-filter_complex_script", None)
    tools = tools_for(c, b"ffmpeg version test\n")
    factory = c.h.job._processes
    def process(argv: tuple[str, ...]) -> ScriptProcess:
        if "-filters" in argv:
            child = ScriptProcess(b" ... null video\n", b"")
            c.h.children.append(child)
            return child
        return factory(argv)
    c.h.job._processes = process
    result = rr.qualify_render_runtime(runtime, rr.RenderRequirements(required_filters=("null",)),
        _files=c.files, _runner=tools)
    assert result.filters == ("null",) and len(c.h.children) == 3
    assert finish_job(c.h).own0


def test_source_probe_real_origin_owned_metadata_and_hashes(memory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from clipperx import source_decode as sd, render_runtime as rr
    from clipperx.local_paths import source_fingerprint
    c = memory
    paths = (Path("O:/HOST/ffmpeg.exe"), Path("O:/HOST/ffprobe.exe"))
    hashes = tuple(hashlib.sha256(raw).hexdigest().upper() for raw in (b"a", b"b"))
    for path, raw in zip(paths, (b"a", b"b")):
        c.fs.data[path] = raw
        c.files.register(path, role="runtime", owned=False, max_bytes=100)
    for module in (rr, sd):
        monkeypatch.setattr(module, "FFMPEG_SHA256", hashes[0])
        monkeypatch.setattr(module, "FFPROBE_SHA256", hashes[1])
    monkeypatch.setattr(sd, "_verify_pins", rr._verify_pins)
    monkeypatch.setattr(sd, "source_fingerprint", source_fingerprint)
    source = Path("O:/HOST/source.mp4")
    c.fs.data[source] = b"media"
    c.files.register(source, role="original", owned=False, max_bytes=100)
    runtime = rr.RenderRuntime(paths[0], paths[1], rr.NATIVE_RUNTIME, hashes[0], hashes[1],
        hashes[0], hashes[1], "-/filter_complex", None)
    metadata = {"format": {"start_time": "0", "duration": "1"}, "streams": [
        {"index": 0, "codec_type": "video", "time_base": "1/60", "start_pts": 0,
         "duration_ts": 60, "width": 2, "height": 2},
        {"index": 1, "codec_type": "audio", "time_base": "1/48000", "start_pts": 0,
         "duration_ts": 48000, "sample_rate": "48000", "channels": 1, "sample_fmt": "flt"}]}
    tools = tools_for(c, json.dumps(metadata).encode(), b"")
    origin = REAL_PROBE_SOURCE(source, runtime, _files=c.files, _runner=tools)
    assert origin.fingerprint == source_fingerprint(source, _files=c.files)
    assert origin.streams[0].width == 2 and origin.streams[1].sample_format == "flt"
    assert not any(op == "open" and path == source for op, path in c.fs.events)
    assert finish_job(c.h).own0


@pytest.mark.parametrize("fault", ["", "geometry", "hdr", "assembly"])
def test_real_preflight_uses_fresh_assembly_without_old_builder(memory: Any,
        monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    import json
    from dataclasses import replace
    from clipperx import visual_compiler as vc, render_runtime as rr
    from clipperx.local_paths import source_fingerprint
    from clipperx.render_assets import RenderAssetRegistry
    from clipperx.render_contracts import RenderProfile, EncoderName
    from clipperx.tests.test_visual_compiler import pure_plan, pure_overlays
    from clipperx.tests.test_source_host_ownership import forbidden
    c, source = memory, Path("O:/HOST/source.mp4")
    c.fs.data[source] = b"media"
    c.files.register(source, role="original", owned=False, max_bytes=100)
    plan = pure_plan()
    plan.source_fingerprint = source_fingerprint(source, _files=c.files)
    assembly = vc.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    if fault == "assembly":
        assembly = replace(assembly, source_size=(4, 4))
    monkeypatch.setattr(vc, "compile_visual_layers", forbidden)
    runtime = rr.RenderRuntime(Path("O:/HOST/ffmpeg.exe"), Path("O:/HOST/ffprobe.exe"),
        "system", "A" * 64, "B" * 64, None, None, "-filter_complex_script", None)
    tools = tools_for(c, b"")
    metadata = {"format": {"start_time": "0", "duration": "1"}, "streams": [
        {"codec_type": "video", "width": 3 if fault == "geometry" else 2, "height": 2,
         "start_time": "0", "duration": "1", "color_transfer": "smpte2084" if fault == "hdr" else "bt709"},
        {"codec_type": "audio", "start_time": "0", "duration": "1"}]}
    def process(argv: tuple[str, ...]) -> ScriptProcess:
        payload = (b"ffmpeg test\n" if "-version" in argv else b" V..... libx264 video\n"
                   if "-encoders" in argv else json.dumps(metadata).encode())
        child = ScriptProcess(payload, b"")
        c.h.children.append(child)
        return child
    c.h.job._processes = process
    def invoke() -> Any:
        return vc.preflight_visual_plan(plan, source=source, assets=RenderAssetRegistry({}, {}, {}),
            profile=RenderProfile(video_encoder=EncoderName.LIBX264), runtime=runtime,
            _runtime_files=c.files, _tool_runner=tools, _assembly=assembly)
    if fault:
        with pytest.raises(ValueError):
            invoke()
    else:
        assert invoke() is None
    assert finish_job(c.h).own0


def test_probe_adapter_claims_registered_argv_inputs(memory: Any) -> None:
    c, source = memory, Path("O:/HOST/source.mp4")
    c.fs.data[source] = b"source"
    c.files.register(source, role="original", owned=False, max_bytes=100)
    tools = tools_for(c, b"ok", b"")
    factory = c.h.job._processes
    def process(argv: tuple[str, ...]) -> ScriptProcess:
        assert c.files._busy(source)
        return factory(argv)
    c.h.job._processes = process
    assert tools(("fake", "-i", str(source)), timeout=15,
        max_stdout_bytes=100, max_stderr_bytes=100).stdout == b"ok"
    assert not c.files._busy(source)
    assert finish_job(c.h).own0


@pytest.mark.parametrize("kind,changed", [("float", False), ("float", True), ("stream", False), ("stream", True)])
def test_real_source_observers_forward_all_pin_rechecks(memory: Any,
        monkeypatch: pytest.MonkeyPatch, kind: str, changed: bool) -> None:
    from array import array
    from fractions import Fraction as F
    import zlib
    from clipperx import source_decode as sd, render_runtime as rr
    from clipperx.local_paths import source_fingerprint
    from clipperx.render_clock import StreamClock, SourceTimeOrigin, StreamDecodeLimits
    from clipperx.tests.test_source_decode import video_line
    c, source = memory, Path("O:/HOST/source.mp4")
    c.fs.data[source] = b"media"
    c.files.register(source, role="original", owned=False, max_bytes=100)
    pair = (Path("O:/HOST/ffmpeg.exe"), Path("O:/HOST/ffprobe.exe"))
    hashes = tuple(hashlib.sha256(raw).hexdigest().upper() for raw in (b"a", b"b"))
    for path, raw in zip(pair, (b"a", b"b")):
        c.fs.data[path] = raw
        c.files.register(path, role="runtime", owned=False, max_bytes=100)
    for module in (rr, sd):
        monkeypatch.setattr(module, "FFMPEG_SHA256", hashes[0])
        monkeypatch.setattr(module, "FFPROBE_SHA256", hashes[1])
    monkeypatch.setattr(sd, "_verify_pins", rr._verify_pins)
    monkeypatch.setattr(sd, "source_fingerprint", source_fingerprint)
    monkeypatch.setattr(sd.time, "monotonic_ns", lambda: c.h.now)
    monkeypatch.setattr(queue.Queue, "get", lambda self, *a, **k: _QUEUE_GET(self, False))
    monkeypatch.setattr(queue.Queue, "put", lambda self, item, *a, **k: _QUEUE_PUT(self, item, False))
    runtime = rr.RenderRuntime(pair[0], pair[1], rr.NATIVE_RUNTIME, hashes[0], hashes[1],
        hashes[0], hashes[1], "-/filter_complex", None)
    origin = SourceTimeOrigin(str(source), 5, 100, source_fingerprint(source, _files=c.files), F(0), F(1),
        (StreamClock(0, "video", F(1, 48000), F(0), F(1), width=1, height=1),
         StreamClock(1, "audio", F(1, 48000), F(0), F(1), sample_rate=48000, channels=1, sample_format="flt")),
         sd._runtime_identity(runtime), {})
    pcm = array("f", [.125] * 4).tobytes()
    def process(argv: tuple[str, ...]) -> ScriptProcess:
        if "-vf" in argv:
            raw = b"".join(bytes([n]) * 3 for n in range(3))
            log = "[showinfo@cxv @ 000] config in time_base: 1/48000\n"
            log += "\n".join(video_line(n, n*2, bytes([n])*3, size="1x1") for n in range(3)) + "\n"
        else:
            raw = pcm if kind == "float" else bytes(10)
            samples, fmt = (4, "flt") if kind == "float" else (5, "s16")
            checksum = zlib.adler32(raw, 0)
            log = (f"[ashowinfo@cxain @ 000] n:0 pts:0 fmt:flt channels:1 rate:48000 nb_samples:{samples} checksum:{checksum:08X}\n"
                f"[ashowinfo@cxaout @ 000] n:0 pts:0 fmt:{fmt} channels:1 rate:48000 nb_samples:{samples} checksum:{checksum:08X}\n")
        child = ScriptProcess(raw, log.encode(), pid=80 + len(c.h.children))
        c.h.children.append(child)
        wait = child.wait
        def reaped(timeout: float) -> int:
            if changed and "-vf" not in argv:
                c.fs.data[pair[0]] = b"changed"
            return wait(timeout)
        setattr(child, "wait", reaped)
        return child
    c.h.job._processes = process
    control = c.h.job.new_observation("materialized_f32" if kind == "float" else "streamed_s16", deadline_ns=10000)
    def consume() -> None:
        if kind == "float":
            limits = sd.FloatAudioDecodeLimits(preroll=F(0), postroll=F(0), max_window=F(1),
                max_samples=8, max_decoded_bytes=32, max_retained_bytes=32, max_block_samples=8)
            result = sd.observe_float_audio_window(runtime, origin, sd.Window(F(0), F(1, 12000)),
                audio_policy="render48k-f32-mono-v1", limits=limits, _host=control, _runtime_files=c.files)
            assert result.pcm == pcm
        else:
            limits_s = StreamDecodeLimits(preroll=F(0), postroll=F(0), max_window=F(1), timeout=1,
                max_frames=3, max_samples=8, max_pixels=1, max_decoded_bytes=48, max_log_bytes=4096,
                window_timeout=2, cleanup_grace=1, max_requests=2, max_deliveries=2,
                io_chunk_bytes=16, max_live_payload_bytes=128, max_selected_unique_bytes=48,
                max_request_emission_bytes=48, max_finalization_bytes=100000)
            with sd.stream_observed_window(runtime, origin, sd.Window(F(0), F(5, 48000)),
                    (F(1, 48000), F(3, 48000)), audio_policy="render48k-mono", limits=limits_s,
                    _host=control, _runtime_files=c.files) as stream:
                for index in range(2):
                    selection = stream.next_selection()
                    assert selection is not None
                    del selection
                    stream.acknowledge(index)
                assert stream.next_selection() is None
                stream.finish()
    if changed:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            consume()
        assert control.ownership_snapshot().failure is not None
    else:
        consume()
        assert control.ownership_snapshot().phase == "retired_ok"
        assert len([1 for op, path in c.fs.events if op == "open" and path == pair[0]]) == (2 if kind == "float" else 4)
    assert c.files.snapshot().files_clear
    assert finish_job(c.h).own0


def test_partial_services_fail_before_legacy_io(memory: Any) -> None:
    from clipperx import render_runtime as rr, visual_compiler as vc
    c = memory
    runtime = rr.RenderRuntime(Path("O:/HOST/ffmpeg.exe"), Path("O:/HOST/ffprobe.exe"),
        "system", "A" * 64, "B" * 64, None, None, "-filter_complex_script", None)
    with pytest.raises(ValueError, match="paired"):
        rr.qualify_render_runtime(runtime, rr.RenderRequirements(), _files=c.files)
    with pytest.raises(ValueError, match="paired"):
        vc.binary_version("ffmpeg", runtime=runtime, _runtime_files=c.files)
    assert c.h.children == []
    assert finish_job(c.h).own0


def test_none_runtime_call_shapes_remain_unchanged(memory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    from clipperx import render_runtime as rr, visual_compiler as vc
    c = memory
    pair = (Path("O:/HOST/ffmpeg.exe"), Path("O:/HOST/ffprobe.exe"))
    monkeypatch.setattr(rr, "_runtime_paths", lambda requirements: pair)
    hashes = []
    def hash_file(path: Path) -> str:
        hashes.append(path)
        return "A" * 64
    monkeypatch.setattr(rr, "_hash", hash_file)
    runtime = rr.resolve_render_runtime(rr.RenderRequirements())
    assert hashes == list(pair)
    calls = []
    def run(argv: Any, **kwargs: Any) -> Any:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0,
            "version\n" if kwargs.get("text") else b"version\n", "" if kwargs.get("text") else b"")
    monkeypatch.setattr(subprocess, "run", run)
    assert rr._run(runtime, "ffmpeg", ("-version",)).stdout == b"version\n"
    assert vc.binary_version("ffmpeg", runtime=runtime) == "version"
    assert calls == [([str(pair[0]), "-version"], {"capture_output": True, "timeout": 15, "check": False}),
        ([str(pair[0]), "-version"], {"capture_output": True, "text": True, "timeout": 15, "check": False})]
    assert finish_job(c.h).own0


@pytest.mark.parametrize("fault", ["longline", "logcap", "nonzero", "late_launch", "uncertain_launch", "encoder_bytes"])
def test_tool_failure_retirement_and_output_claim(memory: Any, fault: str) -> None:
    c, output = memory, ROOT / "candidate.mp4"
    c.files.register(output, role="candidate", owned=True, max_bytes=100)
    log = b"x" * 16385 if fault == "longline" else b"0123456789\n" if fault == "logcap" else b"ok\n"
    tools = tools_for(c, b"x" if fault == "encoder_bytes" else b"", log)
    factory = c.h.job._processes
    def process(argv: tuple[str, ...]) -> ScriptProcess:
        assert c.files._busy(output)
        c.fs.data[output] = b"partial"
        child = factory(argv)
        if fault == "nonzero":
            def failed_exit(_: float) -> int:
                child.returncode = 7
                return 7
            child.wait = failed_exit
        if fault == "late_launch":
            c.h.now = c.h.job._deadline
        if fault == "uncertain_launch":
            raise OSError("launch may have created child")
        return child
    c.h.job._processes = process
    if fault == "uncertain_launch":
        def expire(index: int) -> None:
            assert index == 1
            c.h.now = c.h.job._failure.cleanup_deadline_ns
        scripted_cleanup_wait(c, expire)
    with pytest.raises(ValueError):
        tools.run(("fake", str(output)), kind="encoder", timeout=900, max_stdout_bytes=1,
            max_stderr_bytes=5 if fault == "logcap" else 20000, expected_stdout_bytes=0, outputs=(output,))
    status = c.h.job.snapshot()
    assert status.failure is not None
    if fault == "uncertain_launch":
        assert c.files.snapshot().consumers and not c.files.retire().files_clear
        assert not finish_job(c.h).own0
    else:
        assert not c.files.snapshot().consumers
        assert c.files.retire().files_clear
        assert finish_job(c.h).own0


def test_log_tail_is_honestly_truncated(memory: Any) -> None:
    c = memory
    log = b"line\n" * 15000
    tools = tools_for(c, b"ok", log)
    result = tools.run(("fake",), kind="probe", timeout=15, max_stdout_bytes=10, max_stderr_bytes=100000)
    assert result.stderr_bytes == len(log) and result.stderr_tail == log[-65536:]
    assert result.stderr_truncated and result.stderr_sha256 == hashlib.sha256(log).hexdigest()
    assert finish_job(c.h).own0


def test_real_native_path_selector_and_pins(memory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx import render_runtime as rr
    from clipperx.tests.test_source_host_ownership import forbidden
    c = memory
    monkeypatch.setattr(rr, "Paths", forbidden)
    requirements = rr.RenderRequirements(rr.NATIVE_RUNTIME)
    pair = rr._runtime_paths(requirements)
    assert pair == tuple(rr._local_paths.base_dir / "data/tools/ffmpeg/9.0.2/bin" / name
                         for name in ("ffmpeg.exe", "ffprobe.exe"))
    for path, raw in zip(pair, (b"ffmpeg", b"ffprobe")):
        c.files.register(path, role="runtime", owned=False, max_bytes=100)
        c.fs.data[path] = raw
    monkeypatch.setattr(rr, "FFMPEG_SHA256", hashlib.sha256(b"ffmpeg").hexdigest().upper())
    monkeypatch.setattr(rr, "FFPROBE_SHA256", hashlib.sha256(b"ffprobe").hexdigest().upper())
    runtime = rr.resolve_render_runtime(requirements, _files=c.files)
    assert runtime.ffmpeg == pair[0] and runtime.expected_ffmpeg_sha256 == runtime.ffmpeg_sha256
    assert len([1 for op, path in c.fs.events if op == "open"]) == 4
    c.fs.data[pair[0]] = b"changed"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        rr._verify_pins(runtime, _files=c.files)
    assert finish_job(c.h).own0


def test_tool_result_metadata_is_bounded_and_frozen(memory: Any) -> None:
    from dataclasses import replace
    c = memory
    result = tools_for(c, b"ok", b"").run(("fake",), kind="probe", timeout=15,
        max_stdout_bytes=2, max_stderr_bytes=10)
    with pytest.raises(ValueError):
        replace(result, stderr_tail=b"x" * 65537)
    with pytest.raises(ValueError):
        replace(result, stdout_bytes=True)
    assert finish_job(c.h).own0


def test_system_selector_registers_candidates_before_owned_resolution(memory: Any,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx import render_runtime as rr
    c = memory
    selected = {name: Path("O:/HOST/bin") / (name + ".exe") for name in ("ffmpeg", "ffprobe")}
    aliases = {path: Path("O:/HOST/real") / path.name for path in selected.values()}
    calls = []
    def which(name: str) -> str:
        calls.append(name)
        return str(selected[name])
    monkeypatch.setattr(rr.shutil, "which", which)
    action = c.files._path_action
    def resolve(op: str, path: Path, destination: Path | None) -> Any:
        assert path in c.files._paths, "path traversal before pure registration"
        return aliases[path] if op == "resolve" else action(op, path, destination)
    c.files._path_action = resolve
    for path in aliases.values():
        c.fs.data[path] = b"binary"
    runtime = rr.resolve_render_runtime(rr.RenderRequirements(), _files=c.files)
    assert calls == ["ffmpeg", "ffprobe"]
    assert runtime.ffmpeg == aliases[selected["ffmpeg"]]
    assert len(c.files._runtime_candidates) == 2
    with pytest.raises(ValueError, match="candidate cap"):
        c.files.resolve_runtime_path(Path("O:/HOST/third.exe"))
    assert c.files.snapshot().files_clear
    assert finish_job(c.h).own0


def test_probe_capture_counts_toward_aggregate_evidence(memory: Any,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx import _render_services as services
    c = memory
    monkeypatch.setattr(services, "_EVIDENCE_CAP", 30)
    tools = tools_for(c, b"abcd", b"log\n")
    tools.run(("fake",), kind="probe", timeout=15, max_stdout_bytes=8, max_stderr_bytes=8)
    with pytest.raises(ValueError, match="aggregate"):
        tools.run(("fake",), kind="probe", timeout=15, max_stdout_bytes=8, max_stderr_bytes=8)
    assert len(c.h.children) == 1
    assert finish_job(c.h).own0
