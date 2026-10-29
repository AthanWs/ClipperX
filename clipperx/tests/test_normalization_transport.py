"""Owned normalization integration; all external work is scripted before entry."""

from array import array
from dataclasses import replace
from fractions import Fraction
import hashlib
from pathlib import Path
import io
import os
import subprocess
from typing import Any, cast

import pytest

from clipperx.tests.test_source_host_ownership import (
    ScriptProcess, admitted, closed_boundaries as closed_boundaries, finish_job, forbidden,
)
from clipperx.tests.test_normalization_runner import OBSERVATION


class MemoryFile:
    def __init__(self, fs: Any, path: Path, mode: str) -> None:
        self.fs, self.path, self.mode, self.position = fs, path, mode, 0
        self.closed = False

    def read(self, size: int) -> bytes:
        self.fs.event("read", self)
        raw = bytes(self.fs.data[self.path][self.position:self.position + size])
        self.position += len(raw)
        return raw

    def write(self, data: memoryview) -> int:
        self.fs.event("write", self)
        size = min(len(data), self.fs.write_chunk)
        self.fs.data[self.path].extend(data[:size])
        return size

    def flush(self) -> None:
        self.fs.event("flush", self)

    def close(self) -> None:
        self.fs.event("close", self)
        self.closed = True

    def fileno(self) -> int:
        return 100


class MemoryFiles:
    def __init__(self, runtime: Any) -> None:
        self.data = {runtime.ffmpeg: bytearray(b"tiny ffmpeg"),
                     runtime.ffprobe: bytearray(b"tiny ffprobe")}
        self.events: list[tuple[str, Path]] = []
        self.handles: list[MemoryFile] = []
        self.hook: Any = None
        self.write_chunk = 65536

    def event(self, operation: str, handle: MemoryFile) -> None:
        self.events.append((operation, handle.path))
        if self.hook is not None:
            self.hook(operation, handle)

    def open(self, path: Path, mode: str) -> MemoryFile:
        if mode == "xb":
            if path in self.data:
                raise FileExistsError(path)
            self.data[path] = bytearray()
        elif path not in self.data:
            raise FileNotFoundError(path)
        handle = MemoryFile(self, path, mode)
        self.handles.append(handle)
        self.event("open", handle)
        return handle

    def stat(self, handle: MemoryFile) -> tuple[int, int, int]:
        self.event("stat", handle)
        return (1, {"ffmpeg.exe": 1, "ffprobe.exe": 2,
                    "enabled.f32": 3, "normalized.f32": 4}[handle.path.name],
                len(self.data[handle.path]))

    def unlink(self, path: Path) -> None:
        self.events.append(("delete", path))
        del self.data[path]


@pytest.fixture
def case(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    request.getfixturevalue("closed_boundaries")
    from types import SimpleNamespace
    from clipperx import audio_mix as audio, render_runtime as rr
    from clipperx.render_runtime import RenderRuntime
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "unlink", forbidden)
    monkeypatch.setattr(os, "fstat", forbidden)
    h = admitted()
    fmt = audio.AudioFormat(48000, "mono")
    pcm = array("f", [.125]).tobytes() * 19200
    schedule = audio.AudioSchedule("owned-source", (audio.AudioSpan(
        Fraction(10), Fraction(52, 5), Fraction(0), Fraction(2, 5),
        0, 19200, True, 0., False),), ())
    binding = audio.SourceClockBinding("owned-source", "owned-decoder", hashlib.sha256(pcm).hexdigest())
    runtime = RenderRuntime(Path("O:/HOST/ffmpeg.exe"), Path("O:/HOST/ffprobe.exe"),
        rr.NATIVE_RUNTIME, hashlib.sha256(b"tiny ffmpeg").hexdigest().upper(),
        hashlib.sha256(b"tiny ffprobe").hexdigest().upper(), None, None, "-/filter_complex", None)
    monkeypatch.setattr(rr, "FFMPEG_SHA256", runtime.ffmpeg_sha256)
    monkeypatch.setattr(rr, "FFPROBE_SHA256", runtime.ffprobe_sha256)
    fs = MemoryFiles(runtime)
    c = SimpleNamespace(h=h, audio=audio, schedule=schedule, pcm=pcm, binding=binding,
                        fmt=fmt, runtime=runtime, fs=fs, output=pcm, log=OBSERVATION,
                        first_output=b"", calls=[], process_hook=None)

    def process(argv: tuple[str, ...]) -> ScriptProcess:
        c.calls.append(argv)
        child = ScriptProcess(c.first_output if len(c.calls) == 1 else c.output,
                              c.log, pid=71 + len(h.children))
        h.children.append(child)
        if c.process_hook is not None:
            c.process_hook(child)
        return child

    h.job._processes = process
    return c


def files_for(c: Any) -> Any:
    from clipperx._normalization_files import NormalizationFiles
    return NormalizationFiles(c.h.job, c.runtime, Path("O:/HOST/scratch"), clock=c.h.job._clock,
                              _open=c.fs.open, _stat=c.fs.stat, _unlink=c.fs.unlink)


@pytest.mark.parametrize("operation", ["close", "delete"])
def test_normalization_cleanup_uses_shared_cancel_admission(case: Any, operation: str) -> None:
    c, files = case, files_for(case)
    entry = files._acquire(files._paths["enabled"], "xb")
    if operation == "delete":
        files._close(entry)
    c.h.job.fail("validation", "driver")
    before = len(c.h.kernel.CancelSynchronousIo.calls)
    def race(op: str, _: Any) -> None:
        if op == operation:
            c.h.job._cancel(c.h.job._handle_resource)
            if len(c.h.kernel.CancelSynchronousIo.calls) != before:
                raise OSError(22, "synthetic 995")
    c.fs.hook = race
    original = c.fs.unlink
    def unlink(path: Path) -> None:
        race("delete", path)
        original(path)
    files._unlink = unlink
    assert files.retire().files_clear
    assert len(c.h.kernel.CancelSynchronousIo.calls) == before
    assert finish_job(c.h).own0


def prepare(c: Any, files: Any) -> Any:
    from clipperx._normalization_transport import prepare_normalized_audio
    return prepare_normalized_audio(c.schedule, c.pcm, binding=c.binding,
        audio_format=c.fmt, runtime=c.runtime, job=c.h.job, files=files)


def test_normalization_admission(case: Any) -> None:
    h = case.h
    with pytest.raises(ValueError):
        h.job.new_observation("normalization", deadline_ns=10000)
    control = h.job.new_normalization(deadline_ns=10000)
    control._enter("normalization", 65536, 1, 1000, 100)
    assert control.ownership_snapshot().kind == "normalization"
    with pytest.raises(ValueError):
        control._launch("audio_f32", ("fake",), 100)
    control._note_payload(False, False)
    control._leave(True)
    assert control.ownership_snapshot().phase == "retired_ok"
    assert finish_job(h).own0


def test_files_constructor_is_inert(case: Any) -> None:
    f = files_for(case)
    assert case.fs.events == []
    assert f.snapshot().files_clear
    assert f.snapshot().rows == ()


@pytest.mark.parametrize("layout,mode", [("mono", "linear"), ("stereo", "dynamic")])
def test_owned_two_pass_file_bridge(case: Any, layout: str, mode: str) -> None:
    c = case
    if layout == "stereo":
        c.fmt = c.audio.AudioFormat(48000, layout)
        c.pcm *= 2
        c.output = c.pcm
        c.binding = replace(c.binding, selected_pcm_sha256=hashlib.sha256(c.pcm).hexdigest())
    if mode == "dynamic":
        c.log = c.log.replace(b'"5.0"', b'"15.0"').replace(b'"linear"', b'"dynamic"')
    f = files_for(c)
    result = prepare(c, f)
    assert result.normalized.pcm == c.pcm
    assert result.receipt.mode == mode
    assert len(c.calls) == 2
    assert tuple(p.pass_kind for p in result.passes) == ("measure", "process")
    assert len({p.control_id for p in result.passes}) == 2
    assert all(p.input_file.sha256 == c.binding.selected_pcm_sha256 for p in result.passes)
    for evidence, argv in zip(result.passes, c.calls, strict=True):
        assert evidence.actual_argv == argv
        assert argv[argv.index("-i") + 1] == str(Path("O:/HOST/scratch/enabled.f32"))
        assert evidence.canonical_pipe_argv[evidence.canonical_pipe_argv.index("-i") + 1] == "pipe:0"
        assert evidence.natural_eof and evidence.exit_code == 0
    assert result.passes[1].stdout_sha256 == result.normalized_file.sha256
    assert len(f.snapshot().rows) == 16
    assert not f.snapshot().enabled_present
    assert f.snapshot().normalized_present and not f.snapshot().files_clear
    assert f.output_path(result) == Path("O:/HOST/scratch/normalized.f32")
    assert f.snapshot().output_transferred
    f.retire()
    assert f.snapshot().normalized_present
    f.release_output(result)
    assert f.snapshot().files_clear
    with pytest.raises(ValueError):
        f.release_output(result)
    assert finish_job(c.h).own0


@pytest.mark.parametrize("failure", ["measure_stdout", "short", "extra", "partial", "nan", "json", "duplicate", "mode", "nonfinite_stats", "log_line"])
def test_bounded_streams_and_engine_errors(case: Any, failure: str) -> None:
    c = case
    if failure == "measure_stdout":
        c.first_output = b"x"
    elif failure == "short":
        c.output = c.output[:-4]
    elif failure == "extra":
        c.output += b"xxxx"
    elif failure == "partial":
        c.output = c.output[:-1]
    elif failure == "nan":
        c.output = array("f", [float("nan")]).tobytes() + c.output[4:]
    elif failure == "json":
        c.log = b"{}"
    elif failure == "duplicate":
        c.log *= 2
    elif failure == "mode":
        c.log = c.log.replace(b'"linear"', b'"dynamic"')
    elif failure == "nonfinite_stats":
        c.log = c.log.replace(b'"-20.0"', b'"nan"')
    else:
        c.log = b"x" * 16385
    f = files_for(c)
    with pytest.raises(ValueError):
        prepare(c, f)
    assert len(c.calls) <= 2
    assert not f.snapshot().output_transferred
    assert not f.snapshot().normalized_present
    f.retire()
    assert finish_job(c.h).own0


@pytest.mark.parametrize("failure", ["pin", "pair", "input_hash", "input_count", "late_open", "close", "stamp"])
def test_runtime_and_file_custody(case: Any, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    c = case
    if failure == "pin":
        from clipperx import render_runtime as rr
        monkeypatch.setattr(rr, "FFMPEG_SHA256", "A" * 64)
    elif failure == "pair":
        c.runtime = replace(c.runtime, ffprobe_sha256="B" * 64)
    elif failure == "input_hash":
        c.binding = replace(c.binding, selected_pcm_sha256="a" * 64)
    elif failure == "input_count":
        c.pcm = c.pcm[:-4]
        c.binding = replace(c.binding, selected_pcm_sha256=hashlib.sha256(c.pcm).hexdigest())
    else:
        fired = [False]
        def hook(op: str, handle: MemoryFile) -> None:
            if fired[0]:
                return
            if failure == "late_open" and op == "open":
                fired[0] = True
                c.h.now = c.h.job._deadline
            elif failure == "close" and op == "close":
                fired[0] = True
                raise OSError("unknown close")
            elif failure == "stamp" and op == "read" and handle.path.name == "enabled.f32":
                fired[0] = True
                c.fs.data[handle.path].extend(b"x")
        c.fs.hook = hook
    f = files_for(c)
    with pytest.raises((ValueError, OSError)):
        prepare(c, f)
    assert not c.calls
    f.retire()
    if failure == "close":
        assert f.snapshot().cleanup_required
        assert sum(op == "close" for op, _ in c.fs.events) == 1
    assert not f.snapshot().output_transferred
    finish_job(c.h)


def test_result_identity_and_lifetimes(case: Any) -> None:
    c = case
    c.fs.write_chunk = 997
    f = files_for(c)
    result = prepare(c, f)
    with pytest.raises(ValueError):
        f.output_path(replace(result))
    with pytest.raises(ValueError):
        f.release_output(replace(result))
    assert not f.snapshot().output_transferred
    assert f.output_path(result) == Path("O:/HOST/scratch/normalized.f32")
    f.release_output(result)
    with pytest.raises(ValueError):
        prepare(c, f)
    finish_job(c.h)


def test_empty_schedule_uses_no_files_or_child(case: Any) -> None:
    c = case
    c.schedule = c.audio.AudioSchedule("owned-source", (replace(c.schedule.spans[0], normalize=False),), ())
    c.pcm, c.binding = b"", None
    f = files_for(c)
    assert prepare(c, f) is None
    assert c.fs.events == [] and c.calls == []
    assert finish_job(c.h).own0


def test_legacy_hash_and_audio_defaults(case: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager
    from clipperx import render_runtime as rr
    c = case
    calls: list[Path] = []
    @contextmanager
    def default_open(path: Path, mode: str) -> Any:
        assert mode == "rb"
        calls.append(path)
        yield io.BytesIO(c.fs.data[path])
    monkeypatch.setattr(Path, "open", default_open)
    assert rr._hash(c.runtime.ffmpeg) == c.runtime.ffmpeg_sha256
    rr._verify_pins(c.runtime)
    c.audio._validate_pcm(c.schedule, c.pcm, c.binding, c.fmt, c.runtime)
    assert calls == [c.runtime.ffmpeg, c.runtime.ffmpeg, c.runtime.ffprobe,
        c.runtime.ffmpeg, c.runtime.ffprobe, c.runtime.ffmpeg, c.runtime.ffprobe]
    finish_job(c.h)


@pytest.mark.parametrize("event", ["hash_deadline", "between_passes", "stop_read", "cleanup_expired"])
def test_deadline_retirement_and_publication(case: Any, event: str) -> None:
    c = case
    fired = [False]
    def hook(operation: str, handle: MemoryFile) -> None:
        if fired[0]:
            return
        if event == "hash_deadline" and operation == "read" and handle.path == c.runtime.ffmpeg:
            fired[0] = True
            c.h.now = c.h.job._deadline
        elif event == "between_passes" and operation == "close" and handle.path.name == "enabled.f32" and len(c.calls) == 1:
            fired[0] = True
            c.h.now = c.h.job._deadline
        elif event == "stop_read" and operation == "read" and handle.path.name == "enabled.f32":
            fired[0] = True
            c.h.job.request_stop("cancelled")
        elif event == "cleanup_expired" and operation == "open":
            fired[0] = True
            c.h.job.request_stop("cancelled")
            c.h.now += 100
    c.fs.hook = hook
    f = files_for(c)
    with pytest.raises(ValueError):
        prepare(c, f)
    assert fired[0] and len(c.calls) <= (1 if event == "between_passes" else 0)
    assert not f.snapshot().output_transferred
    if event == "cleanup_expired":
        assert f.snapshot().cleanup_required and not f.snapshot().files_clear
        before = list(c.fs.events)
        f.retire()
        assert c.fs.events == before
        assert finish_job(c.h).own0
        assert not f.snapshot().files_clear
    else:
        f.retire()
        finish_job(c.h)


@pytest.mark.parametrize("event", ["unknown_write", "unknown_read", "flush", "late_close", "delete"])
def test_partial_late_and_uncertain_file_actions(case: Any, event: str) -> None:
    c = case
    fired = [False]
    operation = {"unknown_write": "write", "unknown_read": "read", "flush": "flush", "late_close": "close"}.get(event)
    def hook(op: str, handle: MemoryFile) -> None:
        if not fired[0] and op == operation:
            fired[0] = True
            if event == "late_close":
                c.h.now = c.h.job._deadline
            else:
                raise OSError("synthetic uncertain transfer")
    c.fs.hook = hook
    f = files_for(c)
    if event == "delete":
        def unlink(path: Path) -> None:
            fired[0] = True
            raise OSError("synthetic uncertain delete")
        f._unlink = unlink
    with pytest.raises((ValueError, OSError)):
        prepare(c, f)
    assert fired[0]
    status = f.snapshot()
    if event != "late_close":
        assert status.cleanup_required and any(r.uncertain for r in status.rows)
    before = sum(op == operation for op, _ in c.fs.events)
    f.retire()
    assert sum(op == operation for op, _ in c.fs.events) == before
    finish_job(c.h)


@pytest.mark.parametrize("change", ["foreign_path", "overlap", "relative", "huge_integer", "unknown_record", "generation", "runtime_cap"])
def test_guards_caps_and_inert_rejections(case: Any, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    from clipperx._normalization_files import NormalizationFiles
    from clipperx import _normalization_files as nf
    c = case
    if change in ("overlap", "relative"):
        runtime = replace(c.runtime, ffmpeg=Path("O:/HOST/scratch/enabled.f32") if change == "overlap" else Path("ffmpeg.exe"))
        with pytest.raises(ValueError):
            NormalizationFiles(c.h.job, runtime, Path("O:/HOST/scratch"), clock=c.h.job._clock,
                               _open=forbidden, _stat=forbidden, _unlink=forbidden)
    else:
        f = files_for(c)
        if change == "foreign_path":
            with pytest.raises(ValueError):
                f.open_binary(Path("O:/HOST/other"))
        elif change == "generation":
            from clipperx import render_runtime as rr
            for _ in range(16):
                rr._hash(c.runtime.ffmpeg, _files=f)
            before = list(c.fs.events)
            with pytest.raises(ValueError):
                rr._hash(c.runtime.ffmpeg, _files=f)
            assert c.fs.events == before
        elif change == "runtime_cap":
            monkeypatch.setattr(nf, "_RUNTIME_CAP", 4)
            with pytest.raises(ValueError):
                prepare(c, f)
            assert ("open", c.runtime.ffmpeg) in c.fs.events
            assert not any(op == "read" and p == c.runtime.ffmpeg for op, p in c.fs.events)
        else:
            c.schedule = c.audio.AudioSchedule("owned-source", (replace(c.schedule.spans[0],
                first=2**4097) if change == "huge_integer" else object(),), ())
            with pytest.raises(ValueError):
                prepare(c, f)
            assert c.fs.events == []
        f.retire()
    finish_job(c.h)


def test_open_read_runtime_hash_order_and_no_fallback(case: Any) -> None:
    c = case
    f = files_for(c)
    result = prepare(c, f)
    paths = [p for op, p in c.fs.events if op == "open" and p in (c.runtime.ffmpeg, c.runtime.ffprobe)]
    assert paths == [c.runtime.ffmpeg, c.runtime.ffprobe] * 4
    assert result.receipt.runtime_identity == c.audio._runtime_identity(c.runtime)
    assert len(c.h.job.snapshot().resources) == 3
    assert len(c.h.job.snapshot().current.resources) == 6
    assert c.h.job.snapshot().current.diagnostics[0].kind == "normalizer"
    f.output_path(result)
    f.release_output(result)
    finish_job(c.h)


def test_prepared_tail_uses_actual_retained_result_without_loudnorm(case: Any) -> None:
    from clipperx.tests.test_prepared_audio_tail import plan
    c = case
    p = plan(modes=(True,), duration=.4)
    c.schedule = c.audio.schedule_audio(p)
    c.binding = replace(c.binding, source_fingerprint=p.source_fingerprint)
    f = files_for(c)
    result = prepare(c, f)
    # The actual integrated result is admissible to the existing pure tail.
    compiled = c.audio.compile_prepared_audio(p, audio_format=c.fmt,
        schedule_identity=result.schedule_identity, normalized_input_label="0:a",
        binding=result.binding, runtime=c.runtime, receipt=result.receipt)
    assert "loudnorm" not in compiled.filter_complex
    f.output_path(result)
    f.release_output(result)
    finish_job(c.h)


def test_normalizer_child_timeout_is_capped(case: Any) -> None:
    c = case
    control = c.h.job.new_normalization(deadline_ns=c.h.job._deadline)
    control._enter("normalization", 4096, 32, None, 100)
    with pytest.raises(ValueError):
        control._launch("normalizer", ("fake",), 45_000_000_001)
    assert c.calls == []
    control._note_payload(False, False)
    control._leave(True)
    finish_job(c.h)


def test_constructor_rejects_unknown_scratch_before_traversal(case: Any) -> None:
    from clipperx._normalization_files import NormalizationFiles
    class Unknown:
        def __truediv__(self, value: object) -> Any:
            raise AssertionError("unknown scratch traversed")
    c = case
    with pytest.raises(ValueError):
        NormalizationFiles(c.h.job, c.runtime, cast(Path, Unknown()), clock=c.h.job._clock,
                           _open=forbidden, _stat=forbidden, _unlink=forbidden)
    finish_job(c.h)


def test_primary_file_error_survives_cleanup_error(case: Any) -> None:
    from clipperx import render_runtime as rr
    c = case
    f = files_for(c)
    count = [0]
    primary = OSError("specific primary stat failure")
    def stat(handle: MemoryFile) -> tuple[int, int, int]:
        count[0] += 1
        if count[0] == 2:
            raise primary
        return c.fs.stat(handle)
    f._stat = stat
    with pytest.raises(rr.RenderRuntimeError) as caught:
        rr._hash(c.runtime.ffmpeg, _files=f)
    # _hash maps the primary OSError, with its exact identity as cause.
    assert caught.value.__cause__ is primary
    assert f.snapshot().cleanup_required
    finish_job(c.h)


@pytest.mark.parametrize("event", ["stdout_cancel", "late_wait", "close_uncertain", "log_uncertain", "nonzero"])
def test_actual_bridge_child_failure_retirement(case: Any, event: str) -> None:
    c = case
    def process_hook(child: ScriptProcess) -> None:
        if event == "stdout_cancel":
            class Stopped(io.BytesIO):
                def read(self, size: int | None = -1) -> bytes:
                    c.h.job.request_stop("cancelled")
                    return super().read(size)
            child.stdout = Stopped()
        elif event == "late_wait":
            def wait(timeout: float) -> int:
                child.actions.append("wait")
                child.returncode = 0
                c.h.now = c.h.job._deadline
                return 0
            setattr(child, "wait", wait)
        elif event == "close_uncertain":
            class Uncertain(io.BytesIO):
                def close(self) -> None:
                    child.actions.append("uncertain_close")
                    raise OSError("uncertain child pipe close")
            child.stdout = Uncertain()
        elif event == "log_uncertain":
            class UnknownLog(io.BytesIO):
                def readline(self, size: int | None = -1) -> bytes:
                    raise OSError("uncertain child log read")
            child.stderr = UnknownLog()
        else:
            def nonzero(timeout: float) -> int:
                child.actions.append("wait")
                child.returncode = 7
                return 7
            setattr(child, "wait", nonzero)
    c.process_hook = process_hook
    f = files_for(c)
    with pytest.raises(ValueError):
        prepare(c, f)
    assert len(c.calls) == 1
    assert not f.snapshot().normalized_present and not f.snapshot().output_transferred
    child = c.h.children[0]
    if event in ("stdout_cancel", "log_uncertain"):
        assert child.actions.index("terminate") < child.actions.index("wait")
    if event == "close_uncertain":
        assert child.actions.count("uncertain_close") == 1
    status = finish_job(c.h)
    if event == "close_uncertain":
        assert not status.own0
    else:
        assert status.own0
    if event == "log_uncertain":
        assert status.current.diagnostics[0].transfer_uncertain


def test_measurement_stdout_reads_exactly_one_terminal_byte(case: Any) -> None:
    c = case
    reads: list[int | None] = []
    def process_hook(child: ScriptProcess) -> None:
        class Counted(io.BytesIO):
            def read(self, size: int | None = -1) -> bytes:
                reads.append(size)
                return super().read(size)
        child.stdout = Counted(b"x")
    c.process_hook = process_hook
    f = files_for(c)
    with pytest.raises(ValueError):
        prepare(c, f)
    assert reads == [1]
    assert c.h.job.snapshot().current.diagnostics[0].stdout_bytes == 1
    finish_job(c.h)


def test_total_log_cap_counts_overflow_sentinel(case: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx import _normalization_transport as transport
    c = case
    monkeypatch.setattr(transport, "_LOG_CAP", 4096)
    c.log = b"line\n" * 820
    f = files_for(c)
    with pytest.raises(ValueError):
        prepare(c, f)
    diagnostic = c.h.job.snapshot().current.diagnostics[0]
    assert diagnostic.log_bytes == 4097 and diagnostic.transfer_uncertain
    assert len(c.calls) == 1
    finish_job(c.h)


@pytest.mark.parametrize("after_failure", ["certain_retirement", "grace_expired"])
def test_enabled_input_survives_unretired_normalizer(case: Any, after_failure: str) -> None:
    c = case
    may_reap = [False]
    def process_hook(child: ScriptProcess) -> None:
        class CancelledInput(io.BytesIO):
            def read(self, size: int | None = -1) -> bytes:
                c.h.job.request_stop("cancelled")
                return super().read(size)
        child.stdout = CancelledInput()
        def wait(timeout: float) -> int:
            child.actions.append("wait")
            assert timeout == 0
            if not may_reap[0]:
                raise subprocess.TimeoutExpired("scripted-normalizer", timeout)
            child.returncode = 0
            return 0
        setattr(child, "wait", wait)
    c.process_hook = process_hook
    f = files_for(c)
    with pytest.raises(ValueError):
        prepare(c, f)
    control = c.h.job._control
    assert control is not None and control.ownership_snapshot().phase == "cleanup_required"
    deadline = c.h.job.snapshot().failure.cleanup_deadline_ns
    enabled = Path("O:/HOST/scratch/enabled.f32")
    assert ("delete", enabled) not in c.fs.events
    assert enabled in c.fs.data
    assert f.snapshot().enabled_present and not f.snapshot().files_clear
    assert len(f.snapshot().rows) == 7
    if after_failure == "certain_retirement":
        may_reap[0] = True
        control._cleanup_child()
        assert control.ownership_snapshot().phase == "stopped_clean"
        f.retire()
        assert ("delete", enabled) in c.fs.events
        assert enabled not in c.fs.data
        assert f.snapshot().files_clear
        assert c.h.job.snapshot().failure.cleanup_deadline_ns == deadline
        assert finish_job(c.h).own0
    else:
        c.h.now = deadline
        file_events, native_events = list(c.fs.events), list(c.h.kernel.events)
        f.retire()
        assert c.fs.events == file_events and c.h.kernel.events == native_events
        assert c.h.job.snapshot().failure.cleanup_deadline_ns == deadline
        assert f.snapshot().enabled_present and not f.snapshot().files_clear
        finish_job(c.h)
