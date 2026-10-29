"""Explicit Windows ownership for a supplied source driver and its observations.

This module owns resources, never decode policy, parsing, receipts or payloads.
Construction and snapshots are passive. OS actions are claimed before invocation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import hashlib
import subprocess
import threading
import time
from typing import Literal, NoReturn, Protocol, TypeAlias, cast

from ._windows_pipe_io import WindowsSyncIo, WindowsIoError

ObservationKind: TypeAlias = Literal["streamed_s16", "materialized_f32"]
RenderKind: TypeAlias = Literal["probe", "spatial", "finalmix", "encoder"]
ControlKind: TypeAlias = Literal["streamed_s16", "materialized_f32", "normalization", "probe", "spatial", "finalmix", "encoder"]
ChildKind: TypeAlias = Literal["video", "audio_s16", "audio_f32", "normalizer", "probe", "spatial", "finalmix", "encoder"]
StopCode: TypeAlias = Literal["cancelled", "deadline"]
FailureCode: TypeAlias = Literal["cancelled", "deadline", "start", "launch", "io", "metadata",
    "validation", "identity", "cleanup", "native", "protocol", "overflow"]
Stage: TypeAlias = Literal["driver", "entry", "launch", "video", "audio_s16", "audio_f32", "normalizer", "probe", "spatial", "finalmix", "encoder", "validate", "retire"]
JobPhase: TypeAlias = Literal["reserved", "starting", "running", "draining", "stopping", "retired", "cleanup_required"]
ObservationPhase: TypeAlias = Literal["new", "entering", "active", "validating", "retiring", "retired_ok", "stopped_clean", "cleanup_required"]
StartPhase: TypeAlias = Literal["reserved", "in_flight", "returned", "uncertain", "never_started", "joined"]
ResourceRole: TypeAlias = Literal["driver_thread", "driver_handle", "watchdog_thread", "launch",
    "process", "stdout", "stderr", "metadata_thread", "metadata_handle"]
ResourcePhase: TypeAlias = Literal["reserved", "registering", "live", "armed", "in_call", "quiescent",
    "exited", "joined", "reaped", "detached", "closed", "never_created", "unresolved"]


class Worker(Protocol):
    daemon: bool
    @property
    def ident(self) -> int | None: ...
    def start(self) -> None: ...
    def join(self, timeout: float) -> None: ...
    def is_alive(self) -> bool: ...


class WakeEvent(Protocol):
    def set(self) -> None: ...
    def clear(self) -> None: ...
    def wait(self, timeout: float) -> bool: ...
    def is_set(self) -> bool: ...


class BinaryPipe(Protocol):
    @property
    def closed(self) -> bool: ...
    def read(self, size: int) -> bytes: ...
    def readinto(self, buffer: memoryview) -> int | None: ...
    def readline(self, limit: int) -> bytes: ...
    def close(self) -> None: ...


class Process(Protocol):
    @property
    def pid(self) -> int: ...
    @property
    def returncode(self) -> int | None: ...
    @property
    def stdout(self) -> BinaryPipe: ...
    @property
    def stderr(self) -> BinaryPipe: ...
    @property
    def _handle(self) -> object: ...
    def poll(self) -> int | None: ...
    def wait(self, timeout: float) -> int: ...
    def terminate(self) -> None: ...


Clock: TypeAlias = Callable[[], int]
IoFactory: TypeAlias = Callable[[], WindowsSyncIo]
WorkerFactory: TypeAlias = Callable[[Callable[[], None], str], Worker]
EventFactory: TypeAlias = Callable[[], WakeEvent]
CurrentWorker: TypeAlias = Callable[[], object]
ProcessFactory: TypeAlias = Callable[[tuple[str, ...]], Process]
_MAX = 2**63 - 1
_FAILURES = ("cancelled", "deadline", "start", "launch", "io", "metadata", "validation", "identity",
    "cleanup", "native", "protocol", "overflow")
_RENDERS = ("probe", "spatial", "finalmix", "encoder")
_CONTROLS = ("streamed_s16", "materialized_f32", "normalization", *_RENDERS)
_CHILDREN = ("video", "audio_s16", "audio_f32", "normalizer", *_RENDERS)
_STAGES = ("driver", "entry", "launch", *_CHILDREN, "validate", "retire")


def _int(value: int, minimum: int = 0, maximum: int = _MAX) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("bounded plain integer required")
    return value


def _choice(value: str, values: tuple[str, ...]) -> None:
    if type(value) is not str or value not in values:
        raise ValueError("unknown ownership state")


def _bool(value: bool) -> None:
    if type(value) is not bool:
        raise ValueError("plain boolean required")


@dataclass(frozen=True, slots=True)
class HostResourceStatus:
    resource_id: int
    role: ResourceRole
    generation: int
    phase: ResourcePhase
    start: StartPhase | None
    pid: int | None
    alive: bool | None
    in_flight: bool
    uncertain: bool

    def __post_init__(self) -> None:
        _int(self.resource_id, 1)
        _int(self.generation)
        _choice(self.role, ("driver_thread", "driver_handle", "watchdog_thread", "launch", "process",
            "stdout", "stderr", "metadata_thread", "metadata_handle"))
        _choice(self.phase, ("reserved", "registering", "live", "armed", "in_call", "quiescent", "exited",
            "joined", "reaped", "detached", "closed", "never_created", "unresolved"))
        if self.start is not None:
            _choice(self.start, ("reserved", "in_flight", "returned", "uncertain", "never_started", "joined"))
        if self.pid is not None:
            _int(self.pid, 1, 2**32 - 1)
        if self.alive is not None:
            _bool(self.alive)
        _bool(self.in_flight)
        _bool(self.uncertain)


@dataclass(frozen=True, slots=True)
class HostDecodeDiagnostic:
    generation: int
    kind: ChildKind
    stdout_bytes: int
    stdout_sha256: str
    log_bytes: int
    stderr_tail: bytes
    truncated: bool
    transfer_uncertain: bool
    natural_eof: bool
    exit_code: int | None

    def __post_init__(self) -> None:
        _int(self.generation, 1)
        _choice(self.kind, _CHILDREN)
        _int(self.stdout_bytes)
        _int(self.log_bytes)
        if (type(self.stdout_sha256) is not str or len(self.stdout_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.stdout_sha256)):
            raise ValueError("bounded SHA256 required")
        if type(self.stderr_tail) is not bytes or len(self.stderr_tail) > 65536:
            raise ValueError("bounded detached tail required")
        for flag in (self.truncated, self.transfer_uncertain, self.natural_eof):
            _bool(flag)
        if self.exit_code is not None:
            _int(self.exit_code, 0, 2**32 - 1)


@dataclass(frozen=True, slots=True)
class HostFailure:
    code: FailureCode
    stage: Stage
    first_ns: int
    cleanup_deadline_ns: int
    winerror: int | None
    secondary_count: int
    last_secondary: FailureCode | None

    def __post_init__(self) -> None:
        _choice(self.code, _FAILURES)
        _choice(self.stage, _STAGES)
        _int(self.first_ns)
        _int(self.cleanup_deadline_ns, 1)
        _int(self.secondary_count)
        if self.winerror is not None:
            _int(self.winerror, 0, 2**32 - 1)
        if self.last_secondary is not None:
            _choice(self.last_secondary, _FAILURES)


def _records(values: tuple[object, ...], cls: type, cap: int) -> None:
    if type(values) is not tuple or len(values) > cap or any(type(v) is not cls for v in values):
        raise ValueError("bounded detached status tuple required")


@dataclass(frozen=True, slots=True)
class ObservationStatus:
    observation_id: int
    kind: ControlKind
    phase: ObservationPhase
    deadline_ns: int
    child_deadline_ns: int | None
    scope_open: bool
    consumer_active: bool
    borrowed_driver_id: int
    resources: tuple[HostResourceStatus, ...]
    diagnostics: tuple[HostDecodeDiagnostic, ...]
    failure: HostFailure | None
    payload_retained: bool
    lease_outstanding: bool
    transferred_pcm_bytes: int
    window_resources_zero: bool
    cleanup_required: bool

    def __post_init__(self) -> None:
        _int(self.observation_id, 1)
        _choice(self.kind, _CONTROLS)
        _choice(self.phase, ("new", "entering", "active", "validating", "retiring", "retired_ok", "stopped_clean", "cleanup_required"))
        _int(self.deadline_ns, 1)
        if self.child_deadline_ns is not None:
            _int(self.child_deadline_ns, 1)
        _int(self.borrowed_driver_id, 1)
        _records(self.resources, HostResourceStatus, 6)
        _records(self.diagnostics, HostDecodeDiagnostic, 2)
        if self.failure is not None and type(self.failure) is not HostFailure:
            raise ValueError("frozen failure required")
        for flag in (self.scope_open, self.consumer_active, self.payload_retained,
                self.lease_outstanding, self.window_resources_zero, self.cleanup_required):
            _bool(flag)
        _int(self.transferred_pcm_bytes)


@dataclass(frozen=True, slots=True)
class SourceJobStatus:
    phase: JobPhase
    driver_id: int
    driver_admitted: bool
    driver_done: bool
    resources: tuple[HostResourceStatus, ...]
    current: ObservationStatus | None
    failure: HostFailure | None
    own0: bool

    def __post_init__(self) -> None:
        _choice(self.phase, ("reserved", "starting", "running", "draining", "stopping", "retired", "cleanup_required"))
        _int(self.driver_id, 1)
        _records(self.resources, HostResourceStatus, 3)
        for flag in (self.driver_admitted, self.driver_done, self.own0):
            _bool(flag)
        if self.current is not None and type(self.current) is not ObservationStatus:
            raise ValueError("frozen observation required")
        if self.failure is not None and type(self.failure) is not HostFailure:
            raise ValueError("frozen failure required")


@dataclass(frozen=True, slots=True)
class ChildLease:
    observation_id: int
    generation: int
    kind: ChildKind

    def __post_init__(self) -> None:
        _int(self.observation_id, 1)
        _int(self.generation, 1)
        _choice(self.kind, _CHILDREN)


@dataclass(frozen=True, slots=True)
class ChildOutcome:
    exit_code: int
    finished_ns: int
    natural_eof: bool

    def __post_init__(self) -> None:
        _int(self.exit_code, 0, 2**32 - 1)
        _int(self.finished_ns)
        _bool(self.natural_eof)


class HostedObservationFailure(ValueError):
    def __init__(self, failure: HostFailure, status: SourceJobStatus,
                 diagnostics: tuple[HostDecodeDiagnostic, ...]) -> None:
        self.failure, self.status, self.diagnostics = failure, status, diagnostics
        super().__init__(f"hosted source {failure.code} at {failure.stage}")


def _thread_worker(target: Callable[[], None], name: str) -> Worker:
    return threading.Thread(target=target, name=name, daemon=False)


def _popen_source(argv: tuple[str, ...]) -> Process:
    return cast(Process, subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0, text=False, shell=False, close_fds=True))


class SourceHostJob:
    def __init__(self, driver: Worker, *, deadline_ns: int, cleanup_grace_ns: int = 5_000_000_000,
                 clock: Clock = time.monotonic_ns, io_factory: IoFactory = WindowsSyncIo,
                 worker_factory: WorkerFactory = _thread_worker, event_factory: EventFactory = threading.Event,
                 current_worker: CurrentWorker = threading.current_thread,
                 process_factory: ProcessFactory = _popen_source) -> None:
        _int(deadline_ns, 1)
        _int(cleanup_grace_ns, 1, 5_000_000_000)
        if driver.daemon is not False or driver.ident is not None:
            raise ValueError("exclusive unstarted non-daemon driver required")
        self._driver = driver
        self._deadline, self._grace = deadline_ns, cleanup_grace_ns
        self._driver_io_deadline: int | None = None
        self._clock, self._io_factory, self._workers = clock, io_factory, worker_factory
        self._events, self._current_worker, self._processes = event_factory, current_worker, process_factory
        self._lock = threading.RLock()
        self._cleanup_condition = threading.Condition(self._lock)
        self._cleanup_token: _DriverCleanupIo | None = None
        self._clock_lock = threading.Lock()
        self._phase: JobPhase = "reserved"
        self._failure: HostFailure | None = None
        self._admitted = self._done = False
        self._rows = [HostResourceStatus(1, "driver_thread", 0, "reserved", "reserved", None, None, False, False)]
        self._control: HostedObservationControl | None = None
        self._next_id = 2
        self._next_observation = 1
        self._next_generation = 1
        self._next_action = 1
        self._last_now: int | None = None
        self._registered = False
        self._start_claimed = False
        self._wake: WakeEvent | None = None
        self._ready: WakeEvent | None = None
        self._watch_exit = False
        self._cleanup_lock = threading.Lock()
        self._driver_resource = _Resource(self._rows[0], self, worker=driver)
        self._handle_resource: _Resource | None = None
        self._watch_resource: _Resource | None = None
        self._job_resources = [self._driver_resource]

    def _new_resource(self, role: ResourceRole, generation: int = 0,
                      start: StartPhase | None = None) -> _Resource:
        with self._lock:
            _int(self._next_id, 1)
            result = _Resource(HostResourceStatus(self._next_id, role, generation, "reserved", start,
                None, None, False, False), self)
            self._next_id = _int(self._next_id + 1, 1)
            return result

    def _now(self) -> int:
        # Sampling and publication have one order, with no callback under state lock.
        with self._clock_lock:
            value = _int(self._clock())
            with self._lock:
                if self._last_now is not None and value < self._last_now:
                    self._record("overflow", "driver", self._last_now)
                    self._raise()
                self._last_now = value
            return value

    def _record(self, code: FailureCode, stage: Stage, now: int,
                winerror: int | None = None, deadline_ns: int | None = None) -> HostFailure:
        _choice(code, _FAILURES)
        _choice(stage, _STAGES)
        if winerror is not None:
            _int(winerror, 0, 2**32 - 1)
        with self._lock:
            # retired_ok certifies scope/consumer/aliases and native resources clear.
            normal = (self._control._cleanup_deadline if self._control
                and self._control._phase != "retired_ok" else None)
            if self._failure is None:
                end = _int(now + self._grace, 1)
                if normal is not None:
                    end = min(end, normal)
                if deadline_ns is not None:
                    end = min(end, deadline_ns)
                self._failure = HostFailure(code, stage, now, end, winerror, 0, None)
            else:
                f = self._failure
                end = min(f.cleanup_deadline_ns, deadline_ns) if deadline_ns is not None else f.cleanup_deadline_ns
                count = min(_MAX, f.secondary_count + 1)
                secondary: FailureCode = "overflow" if f.secondary_count == _MAX else code
                self._failure = replace(f, cleanup_deadline_ns=end, secondary_count=count, last_secondary=secondary)
            self._admitted = False
            self._phase = "stopping"
            return self._failure

    def fail(self, code: FailureCode, stage: Stage) -> HostFailure:
        return self._record(code, stage, self._now())

    def _raise(self) -> NoReturn:
        status = self.snapshot()
        assert status.failure is not None
        raise HostedObservationFailure(status.failure, status,
            status.current.diagnostics if status.current else ())

    def _admit_locked(self, now: int, current: object, control: HostedObservationControl | None,
                      child: _Child | None, *, metadata: bool = False) -> None:
        if self._failure:
            self._raise()
        deadline, stage = self._effective_deadline()
        if now >= deadline:
            self._record("deadline", stage, now)
            self._raise()
        if self._done or not self._admitted:
            raise ValueError("work admission revoked")
        if control is not None:
            if (self._control is not control or not control._scope
                    or control._phase not in ("entering", "active", "validating")):
                raise ValueError("obsolete or terminal observation")
            if child is not None and control._child is not child:
                raise ValueError("obsolete child generation")
        worker = child.rows[4].worker if metadata and child else self._driver
        if current is not worker:
            raise ValueError("wrong worker for action admission")

    def _claim(self, resource: _Resource, *, now: int | None = None, current: object = None,
               control: HostedObservationControl | None = None, child: _Child | None = None,
               work: bool = False, job_start: bool = False, start_call: bool = False,
               recover: bool = False, phases: tuple[ResourcePhase, ...] | None = None
               ) -> tuple[int, ResourcePhase]:
        with self._lock:
            if control is not None and (self._control is not control or child is None
                    or control._child is not child or resource not in child.rows):
                raise ValueError("obsolete action resource")
            if work:
                assert now is not None
                self._admit_locked(now, current, control, child, metadata=resource.row.role == "stderr")
            if job_start:
                assert now is not None
                if self._failure:
                    self._raise()
                if now >= self._deadline:
                    self._record("deadline", "driver", now)
                    self._raise()
                if self._phase != "starting":
                    raise ValueError("job start no longer admitted")
            if (resource.row.phase in ("closed", "never_created", "joined")
                    or (resource.row.phase == "detached" and phases != ("detached",))
                    or (phases is not None and resource.row.phase not in phases)):
                raise ValueError("terminal or invalid resource phase")
            if resource.row.in_flight or (resource.row.uncertain and not (recover and resource.entered)):
                raise ValueError("resource action already owned or uncertain")
            ident = self._next_action
            self._next_action = _int(ident + 1, 1)
            previous_phase = resource.row.phase
            resource.action_id = ident
            resource.update(in_flight=True, phase="in_call",
                start="in_flight" if start_call else None)
            if start_call and resource is self._driver_resource:
                self._start_claimed = True
            return ident, previous_phase

    def _release_claim(self, resource: _Resource, claim: tuple[int, ResourcePhase],
                       phase: ResourcePhase | None = None) -> None:
        with self._lock:
            if resource.action_id != claim[0] or not resource.row.in_flight:
                raise ValueError("action claim mismatch")
            resource.update(in_flight=False, phase=phase if phase is not None else
                claim[1] if resource.row.phase == "in_call" else resource.row.phase)
            resource.action_id = None
            self._cleanup_condition.notify_all()

    def _perform(self, resource: _Resource, claim: tuple[int, ResourcePhase], action: Callable[[], object], *,
                 stage: Stage, code: FailureCode, commit: Callable[[object], None] | None = None,
                 routine_timeout: bool = False) -> object:
        try:
            result = action()
            with self._lock:
                if resource.action_id != claim[0] or not resource.row.in_flight:
                    raise ValueError("returned action claim mismatch")
                if commit is not None:
                    commit(result)
            return result
        except Exception as exc:
            if routine_timeout and isinstance(exc, subprocess.TimeoutExpired):
                raise
            winerror = None
            uncertain = True
            phase: ResourcePhase = "unresolved"
            if type(exc) is WindowsIoError:
                winerror = exc.winerror
                if exc.operation == "duplicate_current_thread":
                    # DuplicateHandle returning FALSE is qualified no-acquisition.
                    if winerror is not None:
                        uncertain, phase = False, "never_created"
                elif not exc.ownership_uncertain:
                    uncertain, phase = False, "live"
            elif isinstance(exc, subprocess.TimeoutExpired) and resource.row.role == "process":
                uncertain, phase = False, "live"
            with self._lock:
                resource.update(uncertain=uncertain, phase=phase,
                    start="uncertain" if resource.row.start == "in_flight" else None)
            self._record(code, stage, self._now(), winerror)
            raise
        finally:
            self._release_claim(resource, claim)

    def _action(self, resource: _Resource, action: Callable[[], object], *,
                stage: Stage = "retire", code: FailureCode = "native",
                control: HostedObservationControl | None = None, child: _Child | None = None,
                work: bool = False, job_start: bool = False, start_call: bool = False,
                recover: bool = False, phases: tuple[ResourcePhase, ...] | None = None,
                commit: Callable[[object], None] | None = None) -> object:
        now = self._now() if work or job_start else None
        current = self._current_worker() if work else None
        claim = self._claim(resource, now=now, current=current, control=control, child=child,
            work=work, job_start=job_start, start_call=start_call, recover=recover, phases=phases)
        return self._perform(resource, claim, action, stage=stage, code=code, commit=commit)

    def start_driver(self) -> SourceJobStatus:
        with self._lock:
            if self._phase != "reserved" or self._start_claimed:
                raise ValueError("driver start is single-use")
            self._phase = "starting"
        if self._now() >= self._deadline:
            self.request_stop("deadline")
            self._raise()
        try:
            # The watchdog reservation covers both its factory and start call.
            watch = self._new_resource("watchdog_thread", start="reserved")
            with self._lock:
                self._watch_resource = watch
                self._job_resources.append(watch)
            def create_watch() -> object:
                self._wake, self._ready = self._events(), self._events()
                watch.worker = self._workers(self._watchdog, "source-host-watchdog")
                if watch.worker.daemon is not False or watch.worker.ident is not None:
                    raise ValueError("unstarted non-daemon watchdog required")
                return None
            self._action(watch, create_watch, stage="driver", code="start", job_start=True)
            if self._failure:
                watch.update(phase="never_created", start="never_started")
                self._raise()
            self._action(watch, lambda: cast(Worker, watch.worker).start(), stage="driver", code="start",
                job_start=True, start_call=True, commit=lambda _: watch.update(start="returned", phase="live"))
            assert self._ready is not None
            while not self._ready.is_set():
                if self._failure or self._now() >= self._deadline:
                    self.request_stop("deadline")
                    self._raise()
                self._ready.wait(min(.01, (self._deadline - self._now()) / 1e9))
            self._action(self._driver_resource, self._driver.start, stage="driver", code="start",
                job_start=True, start_call=True,
                commit=lambda _: self._driver_resource.update(start="returned", phase="live"))
            with self._lock:
                if self._failure:
                    self._raise()
                self._phase = "draining" if self._done else "running"
            return self.snapshot()
        except HostedObservationFailure:
            raise
        except Exception:
            with self._lock:
                if not self._start_claimed:
                    self._driver_resource.update(phase="never_created", start="never_started", alive=False)
                else:
                    self._driver_resource.update(phase="unresolved", start="uncertain", uncertain=True)
            if self._failure is None:
                self.fail("start", "driver")
            self._raise()
        finally:
            with self._lock:
                if self._failure and not self._start_claimed:
                    self._driver_resource.update(phase="never_created", start="never_started", alive=False)
                    pending_watch = self._watch_resource
                    if (pending_watch and pending_watch.row.start == "reserved"
                            and not pending_watch.row.in_flight and not pending_watch.row.uncertain):
                        pending_watch.update(phase="never_created", start="never_started", alive=False)

    def register_current_driver(self) -> None:
        current = self._current_worker()
        with self._lock:
            if current is not self._driver or not self._start_claimed or self._registered:
                raise ValueError("driver registration identity/start/replay")
            self._registered = True
            self._driver_resource.entered = True
            handle = self._new_resource("driver_handle")
            self._handle_resource = handle
            self._job_resources.insert(1, handle)
        def duplicate() -> object:
            handle.io = self._io_factory()
            handle.handle = handle.io.duplicate_current_thread()
            return None
        try:
            self._action(handle, duplicate, stage="driver", commit=lambda _: handle.update(phase="live"))
            with self._lock:
                self._admitted = not self._failure and not self._done
            self.check()
        except HostedObservationFailure:
            raise
        except Exception:
            self._raise()

    def check(self) -> None:
        now = self._now()
        current = self._current_worker()
        with self._lock:
            deadline, stage = ((self._deadline, "driver") if self._driver_io_deadline is None
                               else self._effective_deadline())
            if now >= deadline and self._failure is None:
                self._record("deadline", cast(Stage, stage), now)
            if self._failure:
                self._raise()
            if not self._admitted or self._done or current is not self._driver:
                raise ValueError("unadmitted source driver")

    def driver_done(self) -> None:
        if self._current_worker() is not self._driver:
            raise ValueError("foreign driver completion")
        now = self._now()
        with self._lock:
            if self._done or not self._start_claimed:
                raise ValueError("driver completion replay/start")
            self._done, self._admitted = True, False
            if (not self._registered or (self._control and self._control._scope)
                    or self._driver_io_deadline is not None or self._cleanup_token is not None):
                self._record("protocol", "driver", now)
            if self._failure is None:
                self._phase = "draining"
        if self._wake:
            self._wake.set()

    def new_observation(self, kind: ObservationKind, *, deadline_ns: int) -> HostedObservationControl:
        _choice(kind, ("streamed_s16", "materialized_f32"))
        return self._new_control(kind, deadline_ns)

    def driver_io_deadline(self, deadline_ns: int) -> _DriverIoDeadline:
        return _DriverIoDeadline(self, deadline_ns)

    def _driver_cleanup_io(self) -> _DriverCleanupIo:
        return _DriverCleanupIo(self)

    def new_normalization(self, *, deadline_ns: int) -> HostedObservationControl:
        return self._new_control("normalization", deadline_ns)

    def new_render(self, kind: RenderKind, *, deadline_ns: int) -> HostedObservationControl:
        _choice(kind, _RENDERS)
        return self._new_control(kind, deadline_ns)

    def _new_control(self, kind: ControlKind, deadline_ns: int) -> HostedObservationControl:
        _int(deadline_ns, 1)
        self.check()
        now, current = self._now(), self._current_worker()
        with self._lock:
            self._admit_locked(now, current, None, None)
            if self._control and self._control._phase != "retired_ok":
                raise ValueError("previous observation still owned")
            ident = _int(self._next_observation, 1)
            self._next_observation = _int(ident + 1, 1)
            control = HostedObservationControl(self, ident, kind, min(self._deadline, deadline_ns))
            self._control = control
        if self._wake:
            self._wake.set()
        return control

    def request_stop(self, code: StopCode, *, deadline_ns: int | None = None) -> SourceJobStatus:
        _choice(code, ("cancelled", "deadline"))
        if deadline_ns is not None:
            _int(deadline_ns, 1)
        self._record(code, "driver", self._now(), deadline_ns=deadline_ns)
        with self._lock:
            if not self._start_claimed and self._phase == "stopping" and self._watch_resource is None:
                self._driver_resource.update(phase="never_created", start="never_started", alive=False)
            if self._control and self._control._phase == "new":
                self._control._phase = "stopped_clean"
        if self._wake:
            self._wake.set()
        return self.snapshot()

    def _watchdog(self) -> None:
        current = self._current_worker()
        with self._lock:
            watch = self._watch_resource
            if watch is None or current is not watch.worker:
                raise ValueError("watchdog entry identity")
            watch.entered = True
            if watch.row.start not in ("in_flight", "returned", "uncertain"):
                watch.update(phase="unresolved", start="uncertain", uncertain=True)
                unexpected = True
            else:
                unexpected = False
        if unexpected:
            self.fail("protocol", "driver")
        assert self._ready is not None and self._wake is not None
        self._ready.set()
        while not self._watch_exit:
            self.service_cleanup()
            if self._watch_exit:
                break
            now = self._now()
            with self._lock:
                deadline, _ = self._effective_deadline()
                remaining = (self._failure.cleanup_deadline_ns if self._failure else deadline) - now
                expired = self._failure is not None and remaining <= 0
            if expired:
                # Clock may cross grace after a protected service pass.
                self.service_cleanup()
                break
            self._wake.wait(min(.01, max(0, remaining) / 1e9))
            self._wake.clear()

    def _effective_deadline(self) -> tuple[int, Stage]:
        """Called under state lock; retired windows cannot shorten job gaps."""
        deadline = self._deadline
        stage: Stage = "driver"
        if self._driver_io_deadline is not None:
            deadline = min(deadline, self._driver_io_deadline)
        control = self._control
        if control and control._phase not in ("retired_ok", "stopped_clean"):
            if control._deadline < deadline:
                deadline, stage = control._deadline, "entry"
            child = control._child
            if child and child.deadline < deadline:
                deadline, stage = child.deadline, child.lease.kind
        return deadline, stage

    def _cancel(self, resource: _Resource | None) -> None:
        now = self._now()
        try:
            with self._lock:
                if (resource is None or resource.handle is None or resource.row.in_flight
                        or resource.row.uncertain or resource.row.phase in ("closed", "never_created")):
                    return
                if (resource is self._handle_resource and self._cleanup_token is not None
                        and (self._failure is None or now < self._failure.cleanup_deadline_ns)):
                    return
                claim = self._claim(resource)
            self._perform(resource, claim,
                lambda: cast(WindowsSyncIo, resource.io).request_cancel(cast(int, resource.handle)),
                stage="retire", code="native",
                commit=lambda _: resource.update(phase="live"))
        except Exception:
            pass

    def _retire_worker(self, resource: _Resource, handle: _Resource | None) -> bool:
        if resource.row.in_flight:
            return False
        if resource.row.phase in ("joined", "never_created"):
            return self._close_thread_handle(handle)
        if resource.worker is None or resource.row.in_flight:
            return False
        if resource.row.start not in ("returned", "uncertain", "joined"):
            return False
        # A raised start with no later registration is not no-creation evidence.
        if resource.row.uncertain and not resource.entered:
            return False
        if resource.worker is self._current_worker():
            return False
        try:
            alive = resource.worker.is_alive()
            if type(alive) is not bool:
                raise ValueError("unknown liveness")
            resource.update(alive=alive)
            if alive:
                return False
            handle_zero = handle is None or handle.row.phase in ("never_created", "closed")
            if not handle_zero and handle and (handle.row.in_flight or handle.row.uncertain or handle.handle is None):
                return False
            if not handle_zero and handle and not self._action(handle,
                    lambda: cast(WindowsSyncIo, handle.io).is_signaled(cast(int, handle.handle))):
                return False
            self._action(resource, lambda: cast(Worker, resource.worker).join(0), recover=True,
                commit=lambda _: resource.update(phase="joined", start="joined", alive=False, uncertain=False))
            return self._close_thread_handle(handle)
        except Exception:
            return False

    def _close_thread_handle(self, resource: _Resource | None) -> bool:
        if resource is None:
            return True
        if resource.row.in_flight:
            return False
        if resource.row.phase in ("closed", "never_created"):
            return True
        if resource.row.in_flight or resource.row.uncertain or resource.handle is None:
            return False
        try:
            def closed(_: object) -> None:
                resource.update(phase="closed", alive=False)
                resource.handle = None
            self._action(resource, lambda: cast(WindowsSyncIo, resource.io).close_handle(cast(int, resource.handle)),
                commit=closed)
            return True
        except Exception:
            return False

    def service_cleanup(self) -> SourceJobStatus:
        if not self._cleanup_lock.acquire(blocking=False):
            return self.snapshot()
        try:
            now = self._now()
            with self._lock:
                deadline, stage = self._effective_deadline()
                if self._failure is None and now >= deadline:
                    self._record("deadline", stage, now)
            if self._failure:
                self._cancel(self._handle_resource)
                if self._control:
                    self._control._cleanup_child()
            if self._done or self._failure or self._driver_resource.row.start == "returned":
                driver_zero = self._retire_worker(self._driver_resource, self._handle_resource)
                if driver_zero:
                    if not self._done and self._driver_resource.row.start != "never_started":
                        self._record("protocol", "driver", self._now())
                    windows_zero = self._control is None or self._control.ownership_snapshot().window_resources_zero
                    if windows_zero:
                        self._watch_exit = True
                        if self._wake:
                            self._wake.set()
                        if self._watch_resource:
                            self._retire_worker(self._watch_resource, None)
            status = self.snapshot()
            if not status.own0 and self._failure:
                with self._lock:
                    self._phase = "cleanup_required"
            elif status.own0:
                with self._lock:
                    self._phase = "retired"
            return self.snapshot()
        finally:
            self._cleanup_lock.release()

    def snapshot(self) -> SourceJobStatus:
        with self._lock:
            current = self._control.ownership_snapshot() if self._control else None
            own0 = all(r.row.phase in ("joined", "closed", "never_created")
                and not r.row.in_flight and not r.row.uncertain for r in self._job_resources)
            own0 = own0 and self._cleanup_token is None
            own0 = own0 and (current is None or (not current.scope_open and not current.consumer_active
                and not current.payload_retained and not current.lease_outstanding and current.window_resources_zero))
            return SourceJobStatus(self._phase, 1, self._admitted, self._done,
                tuple(r.row for r in self._job_resources), current, self._failure, own0)


class _Resource:
    def __init__(self, row: HostResourceStatus, job: SourceHostJob, *, worker: Worker | None = None) -> None:
        self.row, self.worker = row, worker
        self._job = job
        self.handle: int | None = None
        self.io: WindowsSyncIo | None = None
        self.entered = False
        self.action_id: int | None = None

    def update(self, *, phase: ResourcePhase | None = None, start: StartPhase | None = None,
               pid: int | None = None, alive: bool | None = None,
               in_flight: bool | None = None, uncertain: bool | None = None) -> None:
        with self._job._lock:
            old = self.row
            self.row = replace(old, phase=old.phase if phase is None else phase,
                start=old.start if start is None else start, pid=old.pid if pid is None else pid,
                alive=old.alive if alive is None else alive,
                in_flight=old.in_flight if in_flight is None else in_flight,
                uncertain=old.uncertain if uncertain is None else uncertain)


class _Child:
    def __init__(self, lease: ChildLease, rows: list[_Resource], deadline: int) -> None:
        self.lease, self.rows, self.deadline = lease, rows, deadline
        self.process: Process | None = None
        self.stdout_bytes = self.log_bytes = 0
        self.sha = hashlib.sha256()
        self.tail = bytearray()
        self.stdout_eof = self.stderr_eof = self.reader_done = False
        self.natural = self.transfer_uncertain = False
        self.exit_code: int | None = None
        self.finished_ns: int | None = None
        self.terminated = self.reaped = self.detached = False
        self.process_handle: int | None = None

    def diagnostic(self) -> HostDecodeDiagnostic:
        return HostDecodeDiagnostic(self.lease.generation, self.lease.kind, self.stdout_bytes,
            self.sha.hexdigest(), self.log_bytes, bytes(self.tail), self.log_bytes > 65536,
            self.transfer_uncertain, self.natural, self.exit_code)


class _DriverCleanupIo:
    """One original-driver action; pending admission drains earlier handle claims."""

    def __init__(self, job: SourceHostJob) -> None:
        self._job, self._used, self.active = job, False, False

    def __enter__(self) -> None:
        job = self._job
        now, current = job._now(), job._current_worker()
        with job._lock:
            if self._used or job._cleanup_token is not None:
                raise ValueError("single-use nonnested cleanup action required")
            self._used = True
            if current is not job._driver or not job._registered or job._done:
                raise ValueError("cleanup belongs to live registered driver")
            if job._failure is None:
                job._admit_locked(now, current, None, None)
            elif now >= job._failure.cleanup_deadline_ns:
                raise ValueError("original cleanup deadline expired")
            job._cleanup_token = self
        try:
            while True:
                now = job._now()
                with job._lock:
                    handle = job._handle_resource
                    if job._cleanup_token is not self:
                        raise ValueError("cleanup token identity lost")
                    deadline = (job._failure.cleanup_deadline_ns if job._failure else
                                job._effective_deadline()[0])
                    if now >= deadline:
                        raise ValueError("original cleanup deadline expired")
                    if (handle is None or handle.handle is None or handle.row.uncertain
                            or handle.row.phase not in ("live", "in_call")):
                        raise ValueError("certain live driver handle required")
                    if not handle.row.in_flight:
                        self.active = True
                        return
                    job._cleanup_condition.wait(min(.01, (deadline-now)/1e9))
        except BaseException:
            self._clear()
            raise

    def _clear(self) -> None:
        with self._job._lock:
            if self._job._cleanup_token is not self:
                raise ValueError("cleanup token identity mismatch")
            self._job._cleanup_token = None
            self.active = False
            self._job._cleanup_condition.notify_all()

    def __exit__(self, *args: object) -> None:
        if not self.active or self._job._current_worker() is not self._job._driver:
            raise ValueError("cleanup exit requires original driver and active token")
        self._clear()


class _DriverIoDeadline:
    """One explicit driver IO scope, visible to the existing watchdog."""

    def __init__(self, job: SourceHostJob, deadline_ns: int) -> None:
        _int(deadline_ns, 1)
        self._job, self._deadline, self._used = job, deadline_ns, False
        self._entered = False

    def __enter__(self) -> None:
        job = self._job
        job.check()
        with job._lock:
            if self._used or job._driver_io_deadline is not None:
                raise ValueError("single-use nonnested driver IO scope required")
            self._used = True
            job._driver_io_deadline = min(job._deadline, self._deadline)
            self._entered = True
        if job._wake:
            job._wake.set()
        try:
            job.check()
        except BaseException:
            self._clear()
            raise

    def _clear(self) -> None:
        with self._job._lock:
            self._job._driver_io_deadline = None
            self._entered = False
        if self._job._wake:
            self._job._wake.set()

    def __exit__(self, *args: object) -> None:
        if not self._entered:
            raise ValueError("IO scope exit replay or missing entry")
        if self._job._current_worker() is not self._job._driver:
            raise ValueError("IO scope belongs to registered driver")
        try:
            if args[0] is None:
                self._job.check()
        finally:
            self._clear()


class HostedObservationControl:
    def __init__(self, job: SourceHostJob, ident: int, kind: ControlKind, deadline: int) -> None:
        self._job, self._id, self._kind, self._deadline = job, ident, kind, deadline
        self._phase: ObservationPhase = "new"
        self._scope = self._consumer = self._payload = self._lease = False
        self._transferred = 0
        self._rows: list[_Resource] = []
        self._child: _Child | None = None
        self._diagnostics: list[HostDecodeDiagnostic] = []
        self._child_count = 0
        self._max_log = self._max_decoded = 0
        self._grace = job._grace
        self._cleanup_deadline: int | None = None
        self._retirement_lock = threading.Lock()

    def _valid(self) -> None:
        if self._job._control is not self:
            raise ValueError("foreign or superseded observation")

    def _check(self) -> None:
        self._valid()
        now = self._job._now()
        if now >= min(self._deadline, self._child.deadline if self._child else self._deadline):
            if self._job._failure is None:
                self._job._record("deadline", "entry", now)
        if self._job._failure:
            self._job._raise()
        current = self._job._current_worker()
        if current is not self._job._driver and not (self._child and current is self._child.rows[4].worker):
            raise ValueError("foreign observation worker")
        if not self._job._admitted or not self._scope:
            raise ValueError("observation work not admitted")

    def _enter(self, kind: ControlKind, max_log_bytes: int, max_decoded_bytes: int,
               window_timeout_ns: int | None, cleanup_grace_ns: int) -> None:
        self._valid()
        _choice(kind, _CONTROLS)
        if self._phase != "new" or kind != self._kind:
            raise ValueError("single-use exact observation kind required")
        _int(max_log_bytes, 1, 65536 if kind == "normalization" else
             16 * 1024**2 if kind == "materialized_f32" or kind in _RENDERS else 2 * 1024**2)
        _int(max_decoded_bytes, 1, 68736000 if kind in ("normalization", "finalmix") else
             1_000_000 if kind == "probe" else 1 if kind == "encoder" else
             8294400 * 10740 if kind == "spatial" else _MAX)
        _int(cleanup_grace_ns, 1, 5_000_000_000)
        now, current = self._job._now(), self._job._current_worker()
        deadline = self._deadline
        if window_timeout_ns is not None:
            _int(window_timeout_ns, 1)
            deadline = min(deadline, _int(now + window_timeout_ns, 1))
        with self._job._lock:
            self._valid()
            if self._phase != "new":
                raise ValueError("observation already entered or retired")
            self._job._admit_locked(now, current, None, None)
            self._scope = self._consumer = True
            self._phase = "entering"
            self._max_log, self._max_decoded = max_log_bytes, max_decoded_bytes
            self._grace = min(cleanup_grace_ns, self._job._grace)
            self._job._grace = min(self._job._grace, self._grace)
            self._deadline = deadline
        if self._job._wake:
            self._job._wake.set()
        self._check()
        self._phase = "active"

    def request_stop(self, code: StopCode, *, deadline_ns: int | None = None) -> ObservationStatus:
        self._valid()
        self._job.request_stop(code, deadline_ns=deadline_ns)
        return self.ownership_snapshot()

    def _fail(self, code: FailureCode, stage: Stage, winerror: int | None = None) -> None:
        self._job._record(code, stage, self._job._now(), winerror)

    def _retain_cleanup_deadline(self, deadline_ns: int) -> None:
        _int(deadline_ns, 1)
        with self._job._lock:
            self._cleanup_deadline = min(self._cleanup_deadline, deadline_ns) if self._cleanup_deadline else deadline_ns
            if self._job._failure:
                self._cleanup_deadline = min(self._cleanup_deadline, self._job._failure.cleanup_deadline_ns)
                self._job._failure = replace(self._job._failure,
                    cleanup_deadline_ns=self._cleanup_deadline)

    def _note_payload(self, retained: bool, lease: bool, transferred_bytes: int = 0) -> None:
        _bool(retained)
        _bool(lease)
        _int(transferred_bytes, 0, self._max_decoded)
        with self._job._lock:
            self._payload, self._lease, self._transferred = retained, lease, transferred_bytes

    def _launch(self, kind: ChildKind, argv: tuple[str, ...], timeout_ns: int) -> ChildLease:
        self._check()
        _choice(kind, _CHILDREN)
        expected = ((self._kind,) if self._kind in _RENDERS else
                    ("normalizer",) if self._kind == "normalization" else
                    ("audio_f32",) if self._kind == "materialized_f32" else ("video", "audio_s16"))
        if kind not in expected or self._child is not None or self._child_count >= len(expected):
            raise ValueError("child kind/count/concurrency")
        if kind != expected[self._child_count]:
            raise ValueError("child order")
        _int(timeout_ns, 1, 45_000_000_000 if self._kind in ("normalization", "probe") else
             120_000_000_000 if self._kind == "finalmix" else
             900_000_000_000 if self._kind in ("spatial", "encoder") else _MAX)
        if type(argv) is not tuple or not 0 < len(argv) <= 128:
            raise ValueError("bounded canonical argv tuple required")
        total = 0
        for arg in argv:
            if type(arg) is not str or len(arg) > 65536 or "\0" in arg:
                raise ValueError("bounded canonical argv string required")
            total += len(arg.encode("utf-8"))
            if total > 65536:
                raise ValueError("argv transport cap")
        now, current = self._job._now(), self._job._current_worker()
        child_deadline = min(self._deadline, _int(now + timeout_ns, 1))
        with self._job._lock:
            self._job._admit_locked(now, current, self, None)
            if self._child is not None or self._child_count >= len(expected):
                raise ValueError("child reservation already used")
            _int(self._job._next_id + 6, 1)
            generation = self._job._next_generation
            self._job._next_generation = _int(generation + 1, 1)
            self._child_count += 1
            lease = ChildLease(self._id, generation, kind)
            roles: tuple[ResourceRole, ...] = ("launch", "process", "stdout", "stderr", "metadata_thread", "metadata_handle")
            self._rows = [self._job._new_resource(role, generation,
                "reserved" if role == "metadata_thread" else None) for role in roles]
            child = _Child(lease, self._rows, child_deadline)
            self._child = child
            claim = self._job._claim(child.rows[0], now=now, current=current,
                control=self, child=child, work=True)
        try:
            def launch() -> object:
                if self._job._wake:
                    self._job._wake.set()
                child.process = self._job._processes(argv)
                pid = _int(child.process.pid, 1, 2**32 - 1)
                with self._job._lock:
                    for row in child.rows[:4]:
                        row.update(phase="live", pid=pid)
                    child.rows[0].update(phase="quiescent")
                return None
            self._job._perform(child.rows[0], claim, launch, stage="launch", code="launch")
            self._check()
            return lease
        except HostedObservationFailure:
            raise
        except Exception:
            if self._job._failure is None:
                self._fail("launch", "launch")
            self._job._raise()
            raise AssertionError("unreachable")

    def _get(self, lease: ChildLease) -> _Child:
        self._valid()
        if self._child is None or lease is not self._child.lease:
            raise ValueError("foreign/replayed child lease")
        return self._child

    def _start_metadata(self, child: ChildLease, reader: Callable[[], None]) -> None:
        active = self._get(child)
        self._check()
        worker, handle = active.rows[4:6]
        if worker.row.start != "reserved":
            raise ValueError("metadata start is single-use")
        def target() -> None:
            try:
                current = self._job._current_worker()
                with self._job._lock:
                    if current is not worker.worker or self._child is not active:
                        raise ValueError("metadata entry identity/generation")
                    worker.entered = True
                def duplicate() -> object:
                    handle.io = self._job._io_factory()
                    handle.handle = handle.io.duplicate_current_thread()
                    return None
                self._job._action(handle, duplicate, stage=child.kind, control=self, child=active,
                    commit=lambda _: handle.update(phase="live"))
                self._check()
                reader()
            except HostedObservationFailure:
                pass
            except Exception:
                self._fail("metadata", child.kind)
            finally:
                active.reader_done = True
        def create() -> object:
            worker.worker = self._job._workers(target, "source-host-metadata")
            if worker.worker.daemon is not False or worker.worker.ident is not None:
                raise ValueError("unstarted non-daemon metadata worker required")
            return None
        try:
            self._job._action(worker, create, stage=child.kind, code="start", control=self, child=active, work=True)
            self._check()
            self._job._action(worker, lambda: cast(Worker, worker.worker).start(), stage=child.kind, code="start",
                control=self, child=active, work=True, start_call=True,
                commit=lambda _: worker.update(phase="live", start="returned"))
            self._check()
        except HostedObservationFailure:
            raise
        except Exception:
            if self._job._failure is None:
                self._fail("start", child.kind)
            self._job._raise()

    def _read_call(self, lease: ChildLease, role: int, action: Callable[[BinaryPipe], object],
                   count: int, into: memoryview | None = None) -> object:
        active = self._get(lease)
        self._check()
        current = self._job._current_worker()
        if (role == 2 and current is not self._job._driver) or (role == 3 and (
                current is not active.rows[4].worker or active.rows[5].handle is None)):
            raise ValueError("pipe reader identity/registration")
        resource = active.rows[role]
        _int(count, 1, 16385 if role == 3 else self._max_decoded + 1)
        if role == 2 and active.stdout_bytes + count > self._max_decoded + 1:
            raise ValueError("decoded byte cap before read")
        if role == 3 and count > min(16385, max(1, self._max_log - active.log_bytes + 1)):
            count = min(16385, max(1, self._max_log - active.log_bytes + 1))
        assert active.process is not None
        pipe = active.process.stdout if role == 2 else active.process.stderr
        claim = self._job._claim(resource, now=self._job._now(), current=current,
            control=self, child=active, work=True, phases=("live", "quiescent"))
        try:
            self._check()
            result = action(pipe)
            if into is not None:
                if type(result) is not int or not 0 <= result <= len(into):
                    active.transfer_uncertain = True
                    raise ValueError("unknown pipe transfer")
                raw = bytes(into[:result])
            else:
                if type(result) is not bytes or len(result) > count:
                    active.transfer_uncertain = True
                    raise ValueError("unknown pipe transfer")
                raw = result
            with self._job._lock:
                if role == 2:
                    active.stdout_bytes = _int(active.stdout_bytes + len(raw))
                    active.sha.update(raw)
                    active.stdout_eof = not raw
                else:
                    active.log_bytes = _int(active.log_bytes + len(raw))
                    active.tail.extend(raw)
                    del active.tail[:-65536]
                    active.stderr_eof = not raw
            if role == 2 and active.stdout_bytes > self._max_decoded:
                self._fail("protocol", lease.kind)
                self._job._raise()
            if role == 3 and (active.log_bytes > self._max_log or len(raw) > 16384):
                raise ValueError("metadata log cap")
            self._check()
            return result
        except HostedObservationFailure:
            raise
        except Exception:
            active.transfer_uncertain = True
            self._fail("io", lease.kind)
            self._job._raise()
            raise AssertionError("unreachable")
        finally:
            self._job._release_claim(resource, claim, "quiescent")

    def _read_stdout(self, child: ChildLease, size: int) -> bytes:
        return cast(bytes, self._read_call(child, 2, lambda pipe: pipe.read(size), size))

    def _readinto_stdout(self, child: ChildLease, buffer: memoryview) -> int | None:
        return cast(int, self._read_call(child, 2, lambda pipe: pipe.readinto(buffer), len(buffer), buffer))

    def _readline_stderr(self, child: ChildLease, limit: int) -> bytes:
        active = self._get(child)
        bounded = min(limit, 16385, max(1, self._max_log - active.log_bytes + 1))
        return cast(bytes, self._read_call(child, 3, lambda pipe: pipe.readline(bounded), bounded))

    def _finish_child(self, child: ChildLease, natural_eof: bool) -> ChildOutcome:
        active = self._get(child)
        _bool(natural_eof)
        self._check()
        worker = active.rows[4]
        if natural_eof and not active.reader_done and worker.worker is not None:
            try:
                self._job._action(worker, lambda: cast(Worker, worker.worker).join(
                    max(0, active.deadline - self._job._now()) / 1e9), stage=child.kind, code="cleanup",
                    control=self, child=active, work=True)
            except Exception:
                self._job._raise()
            self._check()
        if natural_eof and not (active.stdout_eof and active.stderr_eof and active.reader_done):
            self._fail("protocol", child.kind)
            self._job._raise()
        assert active.process is not None
        try:
            def reaped(result: object) -> None:
                active.exit_code = _int(cast(int, result), 0, 2**32 - 1)
                active.reaped = True
                active.rows[1].update(phase="reaped", alive=False)
            self._job._action(active.rows[1],
                lambda: cast(Process, active.process).wait(max(0, active.deadline - self._job._now()) / 1e9),
                stage=child.kind, code="cleanup", control=self, child=active, work=True,
                phases=("live",), commit=reaped)
            self._check()
            if active.exit_code != 0:
                self._fail("validation", child.kind)
                self._job._raise()
            finished = self._job._now()
            current = self._job._current_worker()
            with self._job._lock:
                self._job._admit_locked(finished, current, self, active)
                active.natural = natural_eof
                active.finished_ns = finished
            outcome = ChildOutcome(active.exit_code, active.finished_ns, natural_eof)
            self._cleanup_child()
            if self._child is not None:
                self._fail("cleanup", "retire")
                self._job._raise()
            self._check()
            return outcome
        except HostedObservationFailure:
            raise
        except Exception:
            if self._job._failure is None:
                self._fail("cleanup", "retire")
            self._job._raise()
            raise AssertionError("unreachable")

    def _cleanup_child(self) -> None:
        if not self._retirement_lock.acquire(blocking=False):
            return
        try:
            self._cleanup_child_claimed()
        finally:
            self._retirement_lock.release()

    def _cleanup_child_claimed(self) -> None:
        active = self._child
        if active is None:
            return
        if self._cleanup_deadline is None:
            self._retain_cleanup_deadline(_int(self._job._now() + self._grace, 1))
        launch, proc, stdout, stderr, worker, handle = active.rows
        if launch.row.in_flight or launch.row.uncertain or active.process is None:
            return
        if self._job._failure or not active.natural:
            self._job._cancel(handle)
            if not active.terminated and not active.reaped and not proc.row.in_flight and not proc.row.uncertain:
                try:
                    self._job._action(proc, active.process.terminate, code="cleanup", control=self, child=active,
                        phases=("live",), commit=lambda _: setattr(active, "terminated", True))
                except Exception:
                    return
        if not active.reaped and not proc.row.in_flight and not proc.row.uncertain:
            try:
                claim = self._job._claim(proc, control=self, child=active, phases=("live",))
                def reaped(result: object) -> None:
                    active.exit_code = _int(cast(int, result), 0, 2**32 - 1)
                    active.reaped = True
                    proc.update(phase="reaped", alive=False)
                self._job._perform(proc, claim, lambda: cast(Process, active.process).wait(0),
                    stage="retire", code="cleanup", commit=reaped, routine_timeout=True)
            except subprocess.TimeoutExpired:
                return
            except Exception:
                return
        if worker.row.start == "reserved" and not worker.row.in_flight:
            worker.update(phase="never_created", start="never_started", alive=False)
            handle.update(phase="never_created", alive=False)
        if not self._job._retire_worker(worker, handle):
            return
        if stdout.row.in_flight or stderr.row.in_flight:
            return
        for resource, pipe in ((stdout, active.process.stdout), (stderr, active.process.stderr)):
            if resource.row.phase == "closed":
                continue
            if resource.row.uncertain:
                return
            try:
                def close() -> object:
                    pipe.close()
                    if pipe.closed is not True:
                        raise ValueError("pipe close unresolved")
                    return None
                self._job._action(resource, close, code="cleanup", control=self, child=active,
                    commit=lambda _: resource.update(phase="closed", alive=False))
            except Exception:
                return
        if active.reaped and not active.detached and not proc.row.uncertain and not proc.row.in_flight:
            try:
                def detach() -> object:
                    if proc.io is None:
                        proc.io = self._job._io_factory()
                    active.process_handle = proc.io.detach_reaped_process(active.process)
                    return None
                def detached(_: object) -> None:
                    active.detached = True
                    proc.update(phase="detached")
                self._job._action(proc, detach, code="cleanup", control=self, child=active,
                    phases=("reaped",), commit=detached)
            except Exception:
                return
        if active.detached and proc.row.phase != "closed" and not proc.row.uncertain:
            try:
                self._job._action(proc, lambda: cast(WindowsSyncIo, proc.io).close_handle(
                    cast(int, active.process_handle)), code="cleanup", control=self, child=active,
                    phases=("detached",), commit=lambda _: proc.update(phase="closed", alive=False))
            except Exception:
                return
        with self._job._lock:
            if (self._job._control is not self or self._child is not active or self._rows is not active.rows
                    or active.lease.observation_id != self._id
                    or any(row.row.generation != active.lease.generation for row in active.rows)):
                raise ValueError("obsolete child retirement publication")
            if (proc.row.phase == "closed" and all(not row.row.in_flight and not row.row.uncertain
                    for row in active.rows) and all(row.row.phase in ("closed", "joined", "never_created")
                    for row in active.rows[1:])):
                launch.update(phase="closed", alive=False)
                self._diagnostics.append(active.diagnostic())
                self._child = None
                if not self._scope and not self._consumer and not self._payload and not self._lease:
                    self._phase = "stopped_clean" if self._job._failure else "retired_ok"

    def _leave(self, success: bool) -> None:
        self._valid()
        _bool(success)
        if self._phase in ("retired_ok", "stopped_clean"):
            raise ValueError("observation exit replay")
        if not success and self._job._failure is None:
            self._fail("validation", "validate")
        self._phase = "retiring"
        self._consumer = False
        self._cleanup_child()
        self._scope = False
        status = self.ownership_snapshot()
        if status.window_resources_zero and not self._payload and not self._lease:
            self._phase = "retired_ok" if success and not self._job._failure else "stopped_clean"
        else:
            self._phase = "cleanup_required"
            if self._job._failure is None:
                self._fail("cleanup", "retire")

    def ownership_snapshot(self) -> ObservationStatus:
        with self._job._lock:
            diagnostics = tuple(self._diagnostics) + ((self._child.diagnostic(),) if self._child else ())
            zero = all(r.row.phase in ("closed", "joined", "never_created")
                and not r.row.in_flight and not r.row.uncertain for r in self._rows)
            return ObservationStatus(self._id, self._kind, self._phase, self._deadline,
                self._child.deadline if self._child else None, self._scope, self._consumer, 1,
                tuple(r.row for r in self._rows), diagnostics, self._job._failure, self._payload,
                self._lease, self._transferred, zero, not zero or self._payload or self._lease)
