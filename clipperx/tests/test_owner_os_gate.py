"""Pure source gates: no real owner, worker, child, native or host execution."""
from __future__ import annotations

import builtins
import ctypes
import dataclasses
import functools
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest


ROOT = pathlib.Path(__file__).parents[2]
GATE = ROOT / ".superpowers/sdd/host/owner_os_gate.py"
FIXTURE = ROOT / ".superpowers/sdd/host/owner_byte_fixture.py"
REAL_OPEN = builtins.open


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("real external effect forbidden")


@contextmanager
def effects_guard() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as guard:
        for obj, names in (
            (threading.Thread, ("__init__", "start", "join")),
            (threading, ("Event", "Timer")),
            (threading.Condition, ("wait", "wait_for")),
            (subprocess, ("Popen", "run", "call", "check_output")),
            (time, ("monotonic_ns", "monotonic", "time", "time_ns", "sleep")),
            (pathlib.Path, ("stat", "resolve", "read_bytes", "open", "exists", "mkdir")),
            (builtins, ("open", "input")),
            (io, ("FileIO",)),
            (os, ("stat", "open", "read", "write", "system", "mkdir")),
        ):
            for name in names:
                guard.setattr(obj, name, forbidden)
        guard.setattr(ctypes, "WinDLL", forbidden, raising=False)
        yield


def pure(function: Any) -> Any:
    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        with effects_guard():
            return function(*args, **kwargs)
    return guarded


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Any:
    # Only the two owned source files may be read; no production input access.
    def owned_bytes(path: pathlib.Path) -> bytes:
        assert path in (GATE, FIXTURE)
        try:
            with REAL_OPEN(path, "rb") as stream:
                data = stream.read(262145)
        except FileNotFoundError:
            return b""
        assert len(data) <= 262144
        return data

    data = owned_bytes(GATE)
    fixture_data = owned_bytes(FIXTURE)
    with effects_guard():
        module: Any = types.ModuleType("owner_gate_pure")
        module.__file__ = str(GATE)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(data, str(GATE), "exec"), module.__dict__)
        fixture = types.ModuleType("owner_fixture_pure")
        monkeypatch.setitem(sys.modules, fixture.__name__, fixture)
        exec(compile(fixture_data, str(FIXTURE), "exec"), fixture.__dict__)
        module.fixture_test = fixture
        return module


def manifest_dict(g: Any) -> dict[str, Any]:
    workspace = "O:/test/ClipperX"
    artifacts = [dict(role=k, path=workspace + "/" + v, bytes=1, sha256="a" * 64)
                 for k, v in g.ARTIFACT_PATHS.items()]
    package = next(a for a in artifacts if a["role"] == "package_init")
    package.update(bytes=408, sha256="58f1735ac519dadb81613312b1fde303bafd814e67c02f96e7a7d329d5cc1631")
    exe = dict(role="python", path="C:/frozen/python.exe", bytes=1, sha256="b" * 64)
    fixture = next(a["path"] for a in artifacts if a["role"] == "fixture")
    cases = []
    for cid, codes in g.CASE_CODES.items():
        fixtures = []
        for code in codes:
            out_n, out_sha, err_n, err_sha = g.ORACLE[code]
            fixtures.append(dict(code=code, argv=[exe["path"], "-I", "-B", fixture, code],
                                 stdout_bytes=out_n, stdout_sha256=out_sha,
                                 stderr_bytes=err_n, stderr_sha256=err_sha))
        cases.append(dict(id=cid, fixtures=fixtures))
    return dict(version=1, kind="clipperx-owner-os45-once",
                run_id="12345678-1234-4234-8234-123456789abc", workspace=workspace,
                host=dict(executable=exe, version="3.12.0", bitness=64, windows_version="frozen"),
                artifacts=artifacts, cases=cases, limits=dict(g.LIMITS))


def parse(g: Any, value: dict[str, Any] | None = None) -> Any:
    value = manifest_dict(g) if value is None else value
    evidence = value["workspace"] + "/.superpowers/sdd/qualification/owner-os45/" + value["run_id"]
    return g.parse_manifest(json.dumps(value).encode(), "O:/test/manifest.json", evidence)


@pure
def test_gate_inert_import_and_cli(gate: Any) -> None:
    assert callable(getattr(gate, "parse_manifest", None))
    before = tuple(k for k in sys.modules if k == "clipperx" or k.startswith("clipperx."))
    assert gate.main(["--help"]) == 0
    assert gate.main([]) == 2
    assert tuple(k for k in sys.modules if k == "clipperx" or k.startswith("clipperx.")) == before
    assert gate.fixture_test.main([]) == 2
    assert gate.main(["--manifest", "relative", "--evidence", "relative"]) == 2
    assert gate.main(["--force"]) == 2
    # Existing raw objects only, with single writes; no descriptor is constructed.
    class Raw:
        closed = False
        def __init__(self) -> None:
            self.blocks: list[bytes] = []
            self.reads: list[int] = []
        def write(self, data: bytes) -> int:
            self.blocks.append(data)
            return len(data)
        def read(self, size: int) -> bytes:
            self.reads.append(size)
            return b""
    out, err, inp = Raw(), Raw(), Raw()
    with pytest.MonkeyPatch.context() as local:
        local.setattr(io, "FileIO", Raw)
        for name, raw in (("stdout", out), ("stderr", err), ("stdin", inp)):
            local.setattr(sys, name, types.SimpleNamespace(buffer=types.SimpleNamespace(raw=raw)))
        assert gate.fixture_test.main(["emit-c2s"]) == 0
        assert out.blocks == [b"S" * 32] and err.blocks == [b"READY C2S\n", b"DONE C2S\n"]
        assert gate.fixture_test.main(["hold-c5"]) == 0 and inp.reads == [1]
        local.setattr(Raw, "write", lambda self, data: 0)
        assert gate.fixture_test.main(["emit-c2s"]) == 1


@pytest.mark.parametrize("fault", ("valid", "duplicate", "depth", "bool", "null", "argv", "path", "limit",
                                  "float", "nan", "bom", "utf8", "size", "number", "key", "package",
                                  "guard_alias", "uuid", "hash", "roles"))
@pure
def test_gate_manifest_finite(gate: Any, fault: str) -> None:
    assert hasattr(gate, "ARTIFACT_PATHS"), "manifest implementation missing"
    value = manifest_dict(gate)
    if fault == "valid":
        assert len(parse(gate, value).artifacts) == 15
        return
    data: bytes | None = None
    if fault == "duplicate":
        data = b'{"version":1,"version":1}'
    elif fault == "depth":
        data = b"[" * 9 + b"0" + b"]" * 9
    elif fault == "bool":
        value["version"] = True
    elif fault == "null":
        value["artifacts"][-1]["sha256"] = None
    elif fault == "argv":
        value["cases"][0]["fixtures"][0]["argv"][1] = "-B"
    elif fault == "path":
        value["workspace"] = "O:/test/../other"
    elif fault == "limit":
        value["limits"]["children"] = 4
    elif fault == "float":
        data = b'{"version":1.0}'
    elif fault == "nan":
        data = b'{"version":NaN}'
    elif fault == "bom":
        data = b"\xef\xbb\xbf{}"
    elif fault == "utf8":
        data = b"\xff"
    elif fault == "size":
        data = b" " * 32769
    elif fault == "number":
        data = b'{"version":99999999999999999999}'
    elif fault == "key":
        value["extra"] = 1
    elif fault == "package":
        value["artifacts"][-1]["bytes"] = 0
    elif fault == "guard_alias":
        with pytest.raises(ValueError):
            gate.parse_manifest(json.dumps(value).encode(), value["workspace"] +
                                "/.superpowers/sdd/qualification/owner-os45.run-once",
                                value["workspace"] + "/.superpowers/sdd/qualification/owner-os45/" + value["run_id"])
        return
    elif fault == "uuid":
        value["run_id"] = "12345678-1234-1234-8234-123456789abc"
    elif fault == "hash":
        value["artifacts"][0]["sha256"] = "A" * 64
    elif fault == "roles":
        value["artifacts"][0]["role"] = "windows_io"
    with pytest.raises(ValueError):
        if data is None:
            parse(gate, value)
        else:
            gate.parse_manifest(data, "O:/test/m.json", "O:/test/e")


@pure
def test_gate_exclusive_reservation(gate: Any) -> None:
    value = manifest_dict(gate)
    tiny = b"pass\n"
    config = parse(gate, value)
    config = dataclasses.replace(config, artifacts=tuple(
        gate.Artifact(a.role, a.path, len(tiny), hashlib.sha256(tiny).hexdigest()) for a in config.artifacts))
    actions: list[str] = []
    env = types.SimpleNamespace(
        file=gate.SerialFile(), now_ns=lambda: 1, inspect_host=lambda: config.host,
        resolve=lambda p: p, read_bytes=lambda p, cap: tiny,
        hash_file=lambda p, cap: (len(tiny), hashlib.sha256(tiny).hexdigest()),
        write_exclusive=lambda p, data: actions.append(p), mkdir_exclusive=lambda p: actions.append(p))
    state = gate.preflight_and_reserve(config, b"{}", env)
    assert state.reserved and actions[0] == state.guard
    assert len(actions) == 4 and len(state.owned_bytes) == 3
    registry: dict[str, Any] = {}
    compiled: list[bytes] = []
    job_type, io_type = type("Job", (), {}), type("Io", (), {})

    def compiler(data: bytes, path: str, mode: str, **kwargs: Any) -> bytes:
        assert kwargs == dict(flags=0, dont_inherit=True, optimize=0)
        compiled.append(data)
        return data

    def execute(data: bytes, namespace: dict[str, Any]) -> None:
        assert registry[namespace["__name__"]].__dict__ is namespace
        assert namespace["__loader__"] is None and namespace["__cached__"] is None
        if namespace["__name__"] == "clipperx":
            assert namespace["__path__"] == []
        elif namespace["__name__"].endswith("_windows_pipe_io"):
            namespace["WindowsSyncIo"] = io_type
        else:
            namespace["SourceHostJob"] = job_type
    assert gate._load_owned_three(state, registry, compiler, execute) == (job_type, io_type)
    assert compiled == [tiny] * 3
    with pytest.raises(ValueError, match="ambient"):
        gate._load_owned_three(state, registry, compiler, execute)
    env.write_exclusive = forbidden
    with pytest.raises(AssertionError):
        gate.preflight_and_reserve(config, b"{}", env)
    assert not env.state.reserved
    env.write_exclusive = lambda p, data: actions.append(p)
    env.mkdir_exclusive = forbidden
    with pytest.raises(AssertionError):
        gate.preflight_and_reserve(config, b"{}", env)
    assert env.state.reserved  # Partial reservation consumes the guard permanently.


class FakeThread:
    def __init__(self, target: Any, name: str, daemon: bool) -> None:
        self.target, self.name, self.daemon = target, name, daemon
        self.ident: int | None = None
        self.calls: list[str] = []
        self.alive = False

    def start(self) -> None:
        self.calls.append("start")
        self.ident = 1
        self.alive = True

    def run(self) -> None:
        self.target()
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.calls.append("join")

    def is_alive(self) -> bool:
        return self.alive


@pure
def test_gate_real_thread_identity_contract(gate: Any) -> None:
    assert hasattr(gate, "ThreadLedger"), "thread recorder missing"
    ledger = gate.ThreadLedger()
    cls = gate._recording_thread_type(FakeThread, ledger)
    assert issubclass(cls, FakeThread)
    assert issubclass(gate._recording_thread_type(threading.Thread, ledger), threading.Thread)
    worker = cls(target=lambda: None, name="driver", role="driver")
    assert ledger.rows[0].worker is worker
    worker.start()
    worker.run()
    assert not ledger.zero
    worker.join(0)
    assert ledger.zero and worker.calls == ["start", "join"]
    ledger.rows[0].start_inflight = True
    assert not ledger.zero


class FakeIo:
    calls: list[tuple[str, Any]] = []

    def duplicate_current_thread(self) -> int:
        self.calls.append(("duplicate", None))
        return 41

    def request_cancel(self, handle: int) -> str:
        self.calls.append(("cancel", handle))
        return "requested"

    def is_signaled(self, handle: int) -> bool:
        self.calls.append(("signal", handle))
        return True

    def close_handle(self, handle: int) -> None:
        self.calls.append(("close", handle))

    def detach_reaped_process(self, process: Any) -> int:
        self.calls.append(("detach", process))
        assert self.is_signaled(73)
        return 73


@pure
def test_gate_native_trace_capacity(gate: Any) -> None:
    assert hasattr(gate, "NativeTrace"), "native recorder missing"
    FakeIo.calls = []
    trace = gate.NativeTrace(lambda: 7)
    thread = object()
    custody = gate.Custody(trace)
    cls = gate._recording_io_type(FakeIo, trace, custody, lambda: thread)
    native = cls()
    assert issubclass(cls, FakeIo)
    assert native.duplicate_current_thread() == 41
    assert native.request_cancel(41) == "requested"
    assert native.is_signaled(41) is True
    native.close_handle(41)
    assert trace.rows[0].closed
    assert native.duplicate_current_thread() == 41
    assert trace.rows[1].generation != trace.rows[0].generation
    trace.rows.extend([trace.rows[-1]] * (32 - len(trace.rows)))
    count = len(FakeIo.calls)
    with pytest.raises(ValueError):
        native.duplicate_current_thread()
    assert len(FakeIo.calls) == count and trace.failed
    # A timestamp/recording failure after acquisition must not hide the handle.
    broken = gate.NativeTrace(forbidden)
    native2 = gate._recording_io_type(FakeIo, broken, gate.Custody(broken), lambda: thread)()
    assert native2.duplicate_current_thread() == 41
    assert broken.failed and broken.rows[0].handle == 41
    context, _ = scripted_case(gate, "C1")
    class NativeError(ValueError):
        operation = "cancel_sync"
        winerror = 5
        handle = 41
        ownership_uncertain = False
    outcomes = iter(("requested", "no_pending", NativeError(), "requested"))
    class SequenceIo(FakeIo):
        def request_cancel(self, handle: int) -> str:
            outcome = next(outcomes)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
    recording = gate._recording_io_type(SequenceIo, context.trace, context.custody,
                                         lambda: context.driver)()
    assert recording.duplicate_current_thread() == 41
    assert recording.request_cancel(41) == "requested"
    assert recording.request_cancel(41) == "no_pending"
    with pytest.raises(NativeError):
        recording.request_cancel(41)
    assert recording.request_cancel(41) == "requested"
    recording.is_signaled(41)
    recording.close_handle(41)
    context.snapshot = context.job.snapshot()
    row = json.loads(gate._case_evidence(context))["native"][0]
    methods = row["methods"]
    cancel = methods["cancel"]
    assert set(methods) == {"duplicate", "cancel", "signal", "detach", "close"}
    assert (cancel["called"], cancel["returned"], cancel["errors"]) == (4, 3, 1)
    assert cancel["first"]["result"] == cancel["last"]["result"] == "requested"
    assert cancel["first_error"]["operation"] == "cancel_sync"
    assert cancel["last_error"]["winerror"] == 5 and row["had_error"]
    assert methods["close"]["last"]["state"] == "returned"
    assert methods["close"]["last"]["result"] is None
    assert methods["detach"]["first"]["state"] == "not_called"
    original_error = NativeError()
    class ErrorIo(FakeIo):
        def request_cancel(self, handle: int) -> str:
            raise original_error
    failing_trace = gate.NativeTrace(lambda: 1)
    faulty = gate._recording_io_type(ErrorIo, failing_trace, gate.Custody(failing_trace), lambda: object())()
    faulty.duplicate_current_thread()
    with pytest.MonkeyPatch.context() as local:
        local.setattr(gate.NativeErrorFacts, "capture", forbidden)
        with pytest.raises(NativeError) as caught:
            faulty.request_cancel(41)
    assert caught.value is original_error
    assert failing_trace.failed and failing_trace.rows[0].handle == 41
    clock_trace = gate.NativeTrace(lambda: 1)
    clock_io = gate._recording_io_type(ErrorIo, clock_trace, gate.Custody(clock_trace), lambda: object())()
    clock_io.duplicate_current_thread()
    clock_trace.now = forbidden
    with pytest.raises(NativeError):
        clock_io.request_cancel(41)
    clock_trace.now = lambda: 9
    with pytest.raises(NativeError):
        clock_io.request_cancel(41)
    cell = clock_trace.rows[0].methods["cancel"]
    assert cell.first_error.observed_ns is None and cell.last_error.observed_ns == 9
    clock_trace.now = forbidden
    with pytest.raises(NativeError):
        clock_io.request_cancel(41)
    assert cell.last_error.observed_ns is None


@pure
def test_gate_custody_transfer(gate: Any) -> None:
    assert hasattr(gate, "Custody"), "process custody missing"
    trace = gate.NativeTrace(lambda: 1)
    custody = gate.Custody(trace)
    row = custody.reserve("C5", "hold-c5", True)
    pipe = types.SimpleNamespace(closed=False, close=lambda: None)
    process = types.SimpleNamespace(pid=9, stdout=pipe, stderr=pipe, stdin=pipe)
    custody.created(row, process)
    custody.offer(row)
    assert not custody.zero
    with pytest.raises(ValueError):
        custody.process_action(row, "poll")
    row.native.handle, row.native.closed, row.native.detached = 73, True, True
    pipe.closed = True
    custody.confirm_disposal(row, True)
    assert custody.zero
    class RetiredProcess:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError("retired process must not be probed again")
    row.process = RetiredProcess()
    custody.confirm_disposal(row, True)
    assert custody.zero and row.source_disposed
    row.borrowed = True
    assert not custody.zero
    for problem in ("stdout_absent", "stderr_absent", "stdout_open", "stderr_open", "native_open", "job_nonzero"):
        bounded = gate.Custody(gate.NativeTrace(lambda: 1))
        pending = bounded.reserve("C5", "hold-c5", True)
        stdout = types.SimpleNamespace(closed=problem != "stdout_open")
        stderr = types.SimpleNamespace(closed=problem != "stderr_open")
        process = types.SimpleNamespace(pid=1, stdout=None if problem == "stdout_absent" else stdout,
                                        stderr=None if problem == "stderr_absent" else stderr,
                                        stdin=types.SimpleNamespace(closed=True))
        bounded.created(pending, process)
        bounded.offer(pending)
        pending.native.detached = True
        pending.native.closed = problem != "native_open"
        if problem.endswith("_absent"):
            with pytest.raises(AttributeError):
                bounded.confirm_disposal(pending, True)
        else:
            bounded.confirm_disposal(pending, problem != "job_nonzero")
        assert not bounded.zero and not pending.source_disposed and pending.phase == "offered_source"


@pure
def test_gate_c1_c2_flow(gate: Any) -> None:
    assert hasattr(gate, "PipeBudget"), "case accounting missing"
    budget = gate.PipeBudget()
    fixture = parse(gate).cases[0].fixtures[0]
    budget.reserve_fixture(fixture)
    assert budget.attempted == 4113
    budget.charge(1024)
    assert budget.attempted == 5137
    with pytest.raises(ValueError):
        budget.charge(65537)
    assert gate._carried_bound(9, 100, 200) == 100
    # Independent expected oracle agrees with fixture recipes, without sharing imports.
    for code, expected in gate.ORACLE.items():
        out, err = gate.fixture_test.fixture_bytes(code)
        assert (len(out), hashlib.sha256(out).hexdigest(), len(err), hashlib.sha256(err).hexdigest()) == expected
    run_positive_scripts(gate)


@pytest.mark.parametrize("mode", ("requested", "no_pending", "early_peer", "fallback",
                                  "stopped_before_claim", "expired_before_claim", "unadmitted",
                                  "creation_race", "watchdog_only", "unwind_before_result", "unwind_equal",
                                  "call_no_pending", "call_error", "wrong_driver", "wrong_generation",
                                  "unarmed_call", "returned_before_claim", "released_claim", "terminated_claim",
                                  "dead_peer", "stale_read", "stale_generation", "history_only", "retired_devnull"))
@pure
def test_gate_c3_c4_stop(gate: Any, mode: str) -> None:
    if mode in ("unwind_before_result", "unwind_equal", "call_no_pending", "call_error", "wrong_driver",
                "wrong_generation", "unarmed_call", "returned_before_claim", "released_claim", "terminated_claim",
                "dead_peer", "stale_read", "stale_generation", "history_only"):
        check_c4_cancel_schedule(gate, mode)
        return
    assert hasattr(gate, "_c4_proof"), "stop proof decision missing"
    pair = gate.CancelPair(call=1, generation=1, read_id=1, valid=True, completed=True, same_call=True,
                           claim_ns=4, result="requested", result_observed_ns=6, read_return_ns=5,
                           driver_bound=True, armed_read=True, peer_held=True)
    facts: dict[str, Any] = dict(ready_ns=2, arm_ns=3, pair=pair, returned_ns=5,
                                 joined=True, peer_alive=True, released=False, terminated=False)
    assert gate._c4_proof(**facts)
    if mode == "no_pending":
        facts["pair"] = None
    elif mode == "early_peer":
        facts["peer_alive"] = False
    elif mode == "fallback":
        facts["terminated"] = True
    else:
        facts["joined"] = False
    assert not gate._c4_proof(**facts)
    if mode == "retired_devnull":
        context, state = scripted_case(gate, "C4")
        context.starting = True
        context.snapshot = context.job.start_driver()
        context.starting = False
        # Execute the actual controller step: native-shaped peer is retired here
        # before its row reaches source-disposal confirmation in that SAME step.
        gate._controller_step(context)
        assert context.peer.process.stderr is None and context.peer.process.stdin is not None
        assert context.peer.phase == "retired" and not context.peer.source_disposed
        assert context.reason is None and gate._combined_zero(state, context)
        before = tuple(tuple(row.counts.items()) for row in context.trace.rows)
        gate._controller_step(context)
        assert tuple(tuple(row.counts.items()) for row in context.trace.rows) == before
        assert not context.peer.source_disposed and context.reason is None
    elif mode in ("stopped_before_claim", "expired_before_claim", "unadmitted", "creation_race", "watchdog_only"):
        check_c4_boundaries(gate, mode)
    elif mode == "requested":
        for cid in ("C3", "C4"):
            context, state = scripted_case(gate, cid)
            gate.drive_case(context.spec, state, context.env)
            assert context.reason is None
            assert gate._combined_zero(state, context)
        assert context.job_zero_before_peer_release
    else:
        context, state = scripted_case(gate, "C4", mode)
        gate.drive_case(context.spec, state, context.env)
        assert context.reason == "cancel_proof"
        assert gate._combined_zero(state, context)
        assert context.peer_terminated == (mode == "fallback")


@pytest.mark.parametrize("mode", ("clean", "wrong_admit", "same_tick", "generic_unwind"))
@pure
def test_gate_c5_late_handoff(gate: Any, mode: str) -> None:
    assert hasattr(gate, "_late_offer"), "late factory handoff missing"
    trace = gate.NativeTrace(lambda: 1)
    custody = gate.Custody(trace)
    row = custody.reserve("C5", "hold-c5", True)
    process = types.SimpleNamespace(pid=1)
    custody.created(row, process)
    event = types.SimpleNamespace(set=lambda: None, wait=forbidden, is_set=lambda: False)
    errors: list[str] = []
    result = gate._late_offer(custody, row, event, event, lambda: 1, 5,
                              lambda: errors.append("recording"))
    assert result is process and row.phase == "offered_source" and errors == ["recording"]
    context, state = scripted_case(gate, "C5", mode)
    if mode == "same_tick":
        context.env.now_ns = lambda: 100000001
        context.trace.now = context.env.now_ns
    gate.drive_case(context.spec, state, context.env)
    if mode in ("wrong_admit", "generic_unwind"):
        assert context.reason == "protocol"
        assert context.job.failure.code == "cancelled" and gate._combined_zero(state, context)
        if mode == "wrong_admit":
            assert context.custody.rows[0].lease is not None
        return
    assert context.reason is None and gate._combined_zero(state, context)
    assert context.created.is_set() and context.release.is_set()
    assert context.observed == () and context.control.metadata_calls == 0
    assert (context.late_created and context.late_stop_confirmed and context.late_released
            and context.late_returned and context.late_unwound)
    assert context.custody.rows[0].lease is None
    assert context.failure_identity == (context.job.failure.code, context.job.failure.stage, context.job.failure.first_ns)
    assert context.failure_deadline == context.job.failure.cleanup_deadline_ns


@pytest.mark.parametrize("scenario", ("pass", "no_pending", "evidence", "operator_stop", "survivor"))
@pure
def test_gate_terminal_caps(gate: Any, scenario: str) -> None:
    assert hasattr(gate, "_combined_zero"), "terminal retention missing"
    state = types.SimpleNamespace(file=gate.SerialFile())
    context = types.SimpleNamespace(
        threads=gate.ThreadLedger(), custody=gate.Custody(gate.NativeTrace(lambda: 0)),
        trace=gate.NativeTrace(lambda: 0), snapshot=types.SimpleNamespace(own0=True, resources=(), current=None))
    assert gate._combined_zero(state, context)
    state.file.uncertain = True
    assert not gate._combined_zero(state, context)
    budget = gate.PipeBudget(attempted=8388608)
    with pytest.raises(ValueError):
        budget.charge(1)
    assert gate._evidence_charge(1048575, 1, 65536) == 1048576
    with pytest.raises(ValueError):
        gate._evidence_charge(1048576, 1, 65536)
    retained = gate.SerialFile()
    raw = types.SimpleNamespace(closed=False, close=forbidden)
    with pytest.raises(AssertionError):
        retained.perform(lambda: raw, lambda f: 1)
    assert retained.raw is raw and retained.uncertain and not retained.zero
    with pytest.raises(ValueError):
        retained.perform(forbidden, forbidden)
    certain = gate.SerialFile()
    def absent() -> Any:
        raise FileNotFoundError("tiny fake absence")
    with pytest.raises(FileNotFoundError):
        certain.perform(absent, forbidden)
    assert certain.zero
    # Exercise the real outer campaign; loader/effects alone are replaced.
    template, campaign = scripted_case(gate, "C1")
    env = template.env
    campaign.current = None
    env.thread_type = type(template.driver).__bases__[0]
    writes: list[str] = []
    env.write_exclusive = lambda p, data: writes.append(p)
    env.inline_start = True
    env.scenario = scenario
    if scenario == "operator_stop":
        env.stop_exists = lambda path: True
    elif scenario == "evidence":
        def failed_write(path: str, data: bytes) -> None:
            campaign.file.uncertain = True
            raise OSError("scripted evidence uncertainty")
        env.write_exclusive = failed_write

    def fake_job(driver: Any, **kwargs: Any) -> Any:
        assert set(kwargs) == {"deadline_ns", "cleanup_grace_ns", "io_factory", "worker_factory", "process_factory"}
        context = campaign.current
        assert driver is context.driver
        env.bound_worker = kwargs["worker_factory"]
        env.bound_process = kwargs["process_factory"]
        scripted_case(gate, context.spec.id, override=context)
        return context.job

    with pytest.MonkeyPatch.context() as local:
        local.setattr(gate, "_load_owned_three", lambda state: (fake_job, template.io_type.__bases__[0]))
        result = gate.run_campaign(campaign, env)
    if scenario == "pass":
        assert result.verdict == "PASS" and result.completed_cases == ("C1", "C2", "C3", "C4", "C5")
        assert len(writes) == 6
    elif scenario == "no_pending":
        assert result.verdict == "INCONCLUSIVE" and result.completed_cases == ("C1", "C2", "C3")
        assert result.reason == "cancel_proof" and result.failed_case == "C4"
    elif scenario == "operator_stop":
        assert result.reason == "operator_stop" and result.completed_cases == () and env.transitions == 0
    else:
        assert result.verdict == "CLEANUP_REQUIRED" and result.failed_case == "C1"
        assert not result.combined_zero and campaign.current is not None
    assert env.transitions <= 32


def run_positive_scripts(gate: Any) -> None:
    for cid in ("C1", "C2"):
        context, state = scripted_case(gate, cid)
        gate.drive_case(context.spec, state, context.env)
        assert context.reason is None and gate._combined_zero(state, context)
        assert len(context.observed) == len(context.spec.fixtures)
        assert all(row.process.stdin is None and row.process.stderr is not None for row in context.custody.rows)
        if cid == "C2":
            assert context.job.control.bounds == [context.bound_ns]
            assert context.bound_sample_ns < context.retired_ns[0] < context.bound_ns
            assert context.job.control.order.index("finish-1") < context.job.control.order.index("launch-2")
            exported = json.loads(gate._case_evidence(context))
            metadata = [r for r in exported["native"] if r["worker"] is not None and r["worker"]["role"] == "metadata"]
            assert len(metadata) == 2
            first, second = (r["worker"] for r in metadata)
            assert first["identity"] != second["identity"]
            assert first["lease"]["generation"] == 1 and second["lease"]["generation"] == 2
            assert first["times"]["joined"] <= second["times"]["start_claim"]
            assert all(r["worker"]["times"]["signal"] is not None and
                       r["worker"]["times"]["close"] is not None for r in metadata)
            processes = [r for r in exported["native"] if r["kind"] == "process"]
            for native, launch in zip(processes, exported["custody"], strict=True):
                assert native["launch"]["ordinal"] == launch["ordinal"]
                assert native["generation"] == launch["native_generation"]
                assert native["launch"]["lease"]["generation"] == launch["source_generation"]


class ScriptFailure(Exception):
    def __init__(self, failure: Any = None) -> None:
        self.failure = failure


HostedObservationFailure = ScriptFailure


def check_c4_cancel_schedule(gate: Any, mode: str) -> None:
    # Obtain the REAL production observer/wrapper through run_campaign, stopping
    # at fake start_driver. No duplicate test observer or actual worker runs.
    template, state = scripted_case(gate, "C4")
    state.current = None
    state.config = dataclasses.replace(state.config, cases=(template.spec,))
    env = template.env
    env.thread_type = FakeThread
    env.tick = 1
    env.now_ns = lambda: env.tick
    captured: list[Any] = []
    calls: list[int] = []
    class Captured(Exception):
        pass
    class NativeError(ValueError):
        operation, winerror, handle, ownership_uncertain = "cancel_sync", 5, 41, False
    error = NativeError()
    class RaceIo(FakeIo):
        def request_cancel(self, handle: int) -> str:
            calls.append(handle)
            ctx = captured[0]
            if mode == "history_only" and len(calls) == 1:
                return "requested"
            env.tick = 20 if mode == "unwind_equal" else 30
            if mode == "stale_read":
                ctx.c4_read_id += 1
            if mode == "stale_generation":
                ctx.driver.record.native_generation += 1
            gate._note_c4_read_return(ctx)
            ctx.peer.borrowed = ctx.peer.read_inflight = False
            env.tick = 20 if mode == "unwind_equal" else 40
            if mode == "call_error":
                raise error
            return "no_pending" if mode in ("call_no_pending", "history_only") else "requested"
    def fake_job(driver: Any, **kwargs: Any) -> Any:
        ctx = state.current
        captured.append(ctx)
        return types.SimpleNamespace(start_driver=lambda: stop(), snapshot=template.job.snapshot)
    def stop() -> None:
        raise Captured()
    with pytest.MonkeyPatch.context() as local:
        local.setattr(gate, "_load_owned_three", lambda state: (fake_job, RaceIo))
        with pytest.raises(Captured):
            gate.run_campaign(state, env)
    ctx = captured[0]
    ctx.snapshot = ctx.job.snapshot()
    ctx.ready_ns = ctx.arm_ns = 10
    ctx.driver_armed = True
    ctx.c4_read_id = 1
    ctx.peer = ctx.custody.reserve("C4", "hold-c4", True)
    peer = ScriptProcess("hold-c4", env, b"READY\n", b"", 9, stderr_pipe=False)
    ctx.custody.created(ctx.peer, peer)
    ctx.peer.phase = "harness_owned"
    ctx.peer.borrowed = ctx.peer.read_inflight = True
    env.current = ctx.driver if mode != "wrong_driver" else object()
    recording = ctx.io_type()
    handle = recording.duplicate_current_thread()
    row = ctx.trace.rows[-1]
    if mode == "wrong_generation":
        ctx.driver.record.native_generation += 1
    if mode == "unarmed_call":
        ctx.driver_armed = False
        ctx.arm_ns = None
    elif mode == "returned_before_claim":
        ctx.driver_return_ns = 15
    elif mode == "released_claim":
        ctx.peer_released = True
    elif mode == "terminated_claim":
        ctx.peer_terminated = True
    elif mode == "dead_peer":
        peer.returncode = 0
    elif mode == "history_only":
        ctx.arm_ns = None
        assert recording.request_cancel(handle) == "requested"
        ctx.arm_ns = 10
    env.tick = 20
    if mode == "call_error":
        with pytest.raises(NativeError) as caught:
            recording.request_cancel(handle)
        assert caught.value is error
    else:
        result = recording.request_cancel(handle)
        assert result == ("no_pending" if mode in ("call_no_pending", "history_only") else "requested")
    recording.is_signaled(handle)
    recording.close_handle(handle)
    driver = ctx.driver.record
    driver.start_returned = driver.entered = driver.returned = driver.joined = True
    ctx.peer_alive = peer.returncode is None
    proved = gate._c4_cancel_proof(ctx)
    positive = mode in ("unwind_before_result", "unwind_equal")
    assert proved is positive
    if positive:
        exported = json.loads(gate._case_evidence(ctx))
        native = next(r for r in exported["native"] if r["generation"] == row.generation)
        pair = native["cancel_pair"]
        assert pair["call"] == 1 and pair["generation"] == row.generation and pair["read_id"] == 1
        assert pair["claim_ns"] == 20 and pair["result"] == "requested" and pair["same_call"]
        assert pair["driver_bound"] and pair["armed_read"] and pair["peer_held"]
        assert exported["c4_read_id"] == pair["read_id"]
        assert pair["read_return_ns"] == (20 if mode == "unwind_equal" else 30)
        assert pair["result_observed_ns"] == (20 if mode == "unwind_equal" else 40)
        assert len(calls) == 1


def check_c4_boundaries(gate: Any, mode: str) -> None:
    context, _ = scripted_case(gate, "C4")
    context.prior_retired = True
    snapshot = context.job.snapshot()
    snapshot.own0 = False
    calls: list[Any] = []
    original = context.env.popen
    def creation(argv: tuple[str, ...], stdin: bool, stderr: bool) -> Any:
        calls.append(context.peer)
        assert context.peer.phase == "constructing"
        if mode == "creation_race":
            context.job.request_stop("cancelled", deadline_ns=context.deadline - 1)
            snapshot.failure = context.job.failure
            snapshot.driver_admitted = False
        return original(argv, stdin, stderr)
    context.env.popen = creation
    if mode == "stopped_before_claim":
        context.job.request_stop("cancelled", deadline_ns=context.deadline - 1)
        snapshot = context.job.snapshot()
    elif mode == "expired_before_claim":
        context.env.now = context.deadline + 1
    elif mode == "unadmitted":
        snapshot.driver_admitted = False
    elif mode == "watchdog_only":
        driver = context.threads.rows[0]
        driver.start_returned = driver.entered = driver.returned = driver.joined = True
        native = context.trace.reserve("thread", context.driver)
        native.handle, native.signaled, native.closed = 101, True, True
        row = context.custody.reserve("C4", "hold-c4", True)
        context.peer = row
        context.custody.created(row, original(context.spec.fixtures[1].argv, True, False))
        row.phase = "harness_owned"
        context.job.request_stop("cancelled", deadline_ns=5000000001)
        snapshot = context.job.snapshot()
        snapshot.driver_done = True
        snapshot.resources = tuple(types.SimpleNamespace(role=role, phase=phase, start=start,
                                                        in_flight=False, uncertain=False)
                                   for role, phase, start in (("driver_thread", "joined", "joined"),
                                                              ("driver_handle", "closed", None),
                                                              ("watchdog_thread", "live", "returned")))
        context.stop_ns, context.cleanup_deadline = 1, 5000000001
        context.env.now = 4000000002
        def terminate() -> None:
            calls.append("terminate")
        row.process.terminate = terminate
    context.job.service_cleanup = lambda: snapshot
    context.job.snapshot = lambda: snapshot
    stop = context.job.request_stop
    def request_stop(code: str, deadline_ns: int | None = None) -> Any:
        stop(code, deadline_ns=deadline_ns)
        snapshot.failure = context.job.failure
        snapshot.driver_admitted = False
        return snapshot
    context.job.request_stop = request_stop
    gate._controller_step(context)
    if mode == "creation_race":
        assert len(calls) == 1 and context.peer.process is not None
        assert context.peer.phase == "harness_owned" and not context.handoff.is_set()
        assert context.cleanup_deadline == context.deadline - 1
        assert not context.custody.zero
    else:
        assert calls == []


class ScriptEvent:
    def __init__(self, env: Any) -> None:
        self.env, self.flag = env, False

    def set(self) -> None:
        self.flag = True

    def clear(self) -> None:
        self.flag = False

    def is_set(self) -> bool:
        return self.flag

    def wait(self, timeout: float) -> bool:
        assert 0 <= timeout <= 0.01
        self.env.pump()
        return self.flag


class ScriptPipe:
    def __init__(self, data: bytes, env: Any, hold: bool = False, stdin: bool = False) -> None:
        self.data, self.env, self.hold, self.stdin = data, env, hold, stdin
        self.closed = False
        self.process: Any = None

    def read(self, size: int) -> bytes:
        assert not self.closed and 0 < size <= 65536
        if not self.data and self.hold:
            if self.env.scenario == "early_peer" and self.process is not None:
                self.process.returncode = 0
            self.env.pump()
            if self.env.scenario == "fallback":
                self.env.now += 4000000001
                self.env.pump()
            raise OSError(995, "scripted cancellation")
        size = min(size, 11)  # Exercise real parent short-read handling.
        result, self.data = self.data[:size], self.data[size:]
        return result

    def readline(self, limit: int) -> bytes:
        end = self.data.find(b"\n") + 1
        if end:
            result, self.data = self.data[:end], self.data[end:]
            return result
        return self.read(limit)

    def readinto(self, view: memoryview) -> int:
        data = self.read(len(view))
        view[:len(data)] = data
        return len(data)

    def close(self) -> None:
        assert not self.closed
        self.closed = True
        if self.stdin and self.process is not None:
            self.process.returncode = 0


class ScriptProcess:
    def __init__(self, code: str, env: Any, stdout: bytes, stderr: bytes, pid: int, *,
                 stdin_pipe: bool = True, stderr_pipe: bool = True) -> None:
        self.code, self.env, self.pid = code, env, pid
        self.returncode: int | None = None if code.startswith("hold-") else 0
        self.stdout = ScriptPipe(stdout, env, code.startswith("hold-"))
        self.stderr: Any = ScriptPipe(stderr, env, code == "hold-c3") if stderr_pipe else None
        self.stdin: Any = ScriptPipe(b"", env, stdin=True) if stdin_pipe else None
        if self.stdin is not None:
            self.stdin.process = self
        self.stdout.process = self
        self.reaped = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        assert self.returncode is not None
        self.reaped = True
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 1


def scripted_case(gate: Any, cid: str, scenario: str = "requested", override: Any = None) -> tuple[Any, Any]:
    config = parse(gate)
    state = gate.RunState(config, b"{}", 200000000000, gate.SerialFile(), reserved=True)
    env = (types.SimpleNamespace(now=100000000, handle=100, current=None, transitions=0, scenario=scenario)
           if override is None else override.env)

    def now() -> int:
        env.now += 1
        return int(env.now)

    env.now_ns = now
    env.event = lambda: ScriptEvent(env)
    env.current_thread = lambda: env.current
    env.stop_exists = lambda path: False
    trace = gate.NativeTrace(now) if override is None else override.trace
    spec = next(c for c in config.cases if c.id == cid)
    context = gate.CaseRun(spec, state, env, trace, now() + 15000000000) if override is None else override
    context.owner_failure_type = ScriptFailure
    if override is not None:
        state = override.state
    state.current = context
    if override is None:
        gate._bind_c4_cancel(context)

    class Worker(FakeThread):
        def run(self) -> None:
            previous, env.current = env.current, self
            env.transitions += 1
            assert env.transitions <= 32
            try:
                super().run()
            finally:
                env.current = previous

    class Native(FakeIo):
        def duplicate_current_thread(self) -> int:
            env.handle += 1
            return int(env.handle)

        def detach_reaped_process(self, process: Any) -> int:
            assert process.reaped
            handle = 200 + process.pid
            assert self.is_signaled(handle)
            return int(handle)

        def request_cancel(self, handle: int) -> str:
            return "no_pending" if env.scenario == "no_pending" else "requested"

    thread_type = gate._recording_thread_type(Worker, context.threads) if override is None else type(context.driver)
    if override is None:
        context.io_type = gate._recording_io_type(Native, trace, context.custody, env.current_thread)
        context.driver = thread_type(target=lambda: gate._driver(context), name="driver", role="driver")

    def popen(argv: tuple[str, ...], stdin: bool, stderr: bool) -> ScriptProcess:
        code = argv[-1]
        # Independent tiny fake transport. Only C1's prescribed 4096-byte recipe is large.
        out, err = gate.fixture_test.fixture_bytes(code)
        return ScriptProcess(code, env, out, err, len(context.custody.rows),
                             stdin_pipe=stdin, stderr_pipe=stderr)
    env.popen = popen

    class Control:
        def __init__(self, job: Any) -> None:
            self.job = job
            self.phase, self.scope_open = "new", False
            self.generation = 0
            self.bounds: list[int] = []
            self.order: list[str] = []
            self.rows: list[Any] = []
            self.metadata_calls = 0
            self.child: Any = None
            self.reader: Any = None
            self.reader_native: Any = None
            self.reader_handle: int | None = None

        def _enter(self, *args: Any) -> None:
            self.scope_open, self.phase = True, "active"

        def _launch(self, role: str, argv: tuple[str, ...], timeout: int) -> Any:
            self.generation += 1
            self.order.append("launch-" + str(self.generation))
            factory = getattr(env, "bound_process", context.factory)
            self.child = factory(argv)
            if self.job.failure is not None and scenario != "wrong_admit":
                if scenario == "generic_unwind":
                    raise ValueError("not an owner unwind")
                raise ScriptFailure(self.job.failure)
            lease = types.SimpleNamespace(generation=self.generation, kind=role, observation_id=1)
            self.rows = [types.SimpleNamespace(role="process", pid=self.child.pid, generation=self.generation)]
            return lease

        def ownership_snapshot(self) -> Any:
            return types.SimpleNamespace(resources=(), phase=self.phase, observation_id=1,
                                         kind="streamed_s16" if cid == "C2" else "materialized_f32",
                                         diagnostics=(), scope_open=self.scope_open,
                                         window_resources_zero=not self.scope_open) if not self.scope_open else types.SimpleNamespace(
                                             resources=tuple(self.rows), phase=self.phase, scope_open=True)

        def _start_metadata(self, lease: Any, reader: Any) -> None:
            self.metadata_calls += 1
            def target() -> None:
                self.reader_native = context.io_type()
                self.reader_handle = self.reader_native.duplicate_current_thread()
                try:
                    reader()
                except (ScriptFailure, OSError):
                    pass
            if hasattr(env, "bound_worker"):
                self.reader = env.bound_worker(target, "metadata")
            else:
                self.reader = thread_type(target=target, name="metadata", role="metadata", lease=lease)
            self.reader.start()

        def _readline_stderr(self, lease: Any, limit: int) -> bytes:
            if self.job.failure is not None:
                raise ScriptFailure(self.job.failure)
            return self.child.stderr.readline(limit)

        def _readinto_stdout(self, lease: Any, view: memoryview) -> int:
            return int(self.child.stdout.readinto(view))

        def _read_stdout(self, lease: Any, size: int) -> bytes:
            if self.child.code == "hold-c3" and not self.child.stdout.data:
                self.reader.run()
                raise ScriptFailure(self.job.failure)
            return self.child.stdout.read(size)

        def _retain_cleanup_deadline(self, deadline: int) -> None:
            self.bounds.append(deadline)

        def cleanup(self) -> None:
            if self.reader is not None and self.reader.record.entered:
                self.reader.join(0)
                if self.reader_handle is not None:
                    self.reader_native.is_signaled(self.reader_handle)
                    self.reader_native.close_handle(self.reader_handle)
                    self.reader_handle = None
            self.job.clean_processes()

        def _finish_child(self, lease: Any, natural_eof: bool) -> Any:
            self.reader.run()
            self.cleanup()
            self.order.append("finish-" + str(self.generation))
            return types.SimpleNamespace(exit_code=0, natural_eof=True)

        def _leave(self, success: bool) -> None:
            self.scope_open, self.phase = False, "retired_ok" if success else "stopped_clean"

    class Job:
        def __init__(self) -> None:
            self.failure: Any = None
            self.control: Any = None
            self.done = False
            self.running = False
            self.own0 = False
            self.driver_handle: int | None = None
            self.native = context.io_type()
            self.watchdog: Any = None

        def start_driver(self) -> Any:
            if hasattr(env, "bound_worker"):
                self.watchdog = env.bound_worker(lambda: None, "watchdog")
            else:
                self.watchdog = thread_type(target=lambda: None, name="watchdog", role="watchdog")
            self.watchdog.start()
            context.driver.start()
            if getattr(env, "inline_start", False):
                self.running = True
                context.driver.run()
            return self.snapshot()

        def register_current_driver(self) -> None:
            assert env.current is context.driver
            self.driver_handle = self.native.duplicate_current_thread()

        def check(self) -> None:
            if self.failure:
                raise ScriptFailure(self.failure)

        def fail(self, code: str, stage: str) -> None:
            self.request_stop(code)

        def request_stop(self, code: str, deadline_ns: int | None = None) -> Any:
            if self.failure is None:
                self.failure = types.SimpleNamespace(code=code, stage="driver", first_ns=now(),
                                                     cleanup_deadline_ns=deadline_ns or now() + 5000000000,
                                                     winerror=None, secondary_count=0, last_secondary=None)
            if self.driver_handle is not None:
                self.native.request_cancel(self.driver_handle)
            return self.snapshot()

        def new_observation(self, kind: str, deadline_ns: int) -> Any:
            self.control = Control(self)
            return self.control

        def driver_done(self) -> None:
            self.done = True

        def snapshot(self) -> Any:
            return types.SimpleNamespace(own0=self.own0, failure=self.failure, phase="retired" if self.own0 else "running",
                                         driver_id=1, driver_admitted=self.failure is None, driver_done=self.done, resources=(),
                                         current=self.control.ownership_snapshot() if self.control is not None else None)

        def clean_processes(self) -> None:
            for row in context.custody.rows:
                if row.phase in ("offered_source", "source_confirmed") and not row.native.closed:
                    if row.process.returncode is None:
                        row.process.terminate()
                    row.process.wait(0)
                    handle = self.native.detach_reaped_process(row.process)
                    self.native.close_handle(handle)
                    row.process.stdout.close()
                    row.process.stderr.close()

        def service_cleanup(self) -> Any:
            if not self.running and not self.done:
                self.running = True
                context.driver.run()
            if env.scenario == "survivor" and self.done:
                env.now += 20000000001
                return self.snapshot()
            if self.done and context.driver.record.returned and not self.own0:
                if self.control is not None:
                    self.control.cleanup()
                self.clean_processes()
                context.driver.join(0)
                self.watchdog.run()
                self.watchdog.join(0)
                if self.driver_handle is not None:
                    self.native.is_signaled(self.driver_handle)
                    self.native.close_handle(self.driver_handle)
                self.own0 = True
            return self.snapshot()

    context.job = Job()
    env.pump = lambda: gate._controller_step(context)
    return context, state
