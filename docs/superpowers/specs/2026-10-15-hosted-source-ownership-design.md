# Hosted source ownership — adopted Task6b5 design

2026-10-15. ADOPTED technical design; exact Task45 activation governs implementation.
Root FULL-read final17935B contract and preserved37700B background proposal, SHA
5DD3AF8A927BBF7B3E59CDCAD7FA3059A7F780F880638520B62CAAAA06733AE8.
Replaces proposals, not39/41/42. No Task44 WIP or execution/delegation.

## Scope

Four paths: NEW clipperx/_source_host_ownership.py and
clipperx/tests/test_source_host_ownership.py; EXISTING
clipperx/source_decode.py and clipperx/tests/test_source_decode.py. No clock,
adapter41, receipt/policy/limit, audio normalizer, font or transport changes.
Both39stream/42float share resource ownership, no second data/task engine.

Approved42/41 source facts/identities remain in prior proposal, not native proof.

OWNER-START of supplied unstarted driver avoids external start-ticket ambiguity.
HOST transfers exclusive start/join authority; job accepts no task/callback.
Target first registers, finally driver_done. No extra driver.

ONE non-daemon job watchdog covers gaps. Prep: driver/watchdog/metadata=3 threads,
1child. Later executor7 already includes driver; plus watcher/metadata=9. No
hidden worker/hosted Timer. _host=None preserves defaults.

## Types and APIs

NEW Python>=3.10/stdlib types. Constructors: validation only, no external calls.

```text
Clock = Callable[[], int]
IoFactory = Callable[[], WindowsSyncIo]
WorkerFactory = Callable[[Callable[[], None], str], Worker]
EventFactory = Callable[[], WakeEvent]
CurrentWorker = Callable[[], object]
ProcessFactory = Callable[[tuple[str, ...]], Process]

Worker protocol:
  daemon: bool; ident: int|None
  start() -> None; join(timeout: float) -> None; is_alive() -> bool
WakeEvent protocol:
  set() -> None; clear() -> None; wait(timeout: float) -> bool
BinaryPipe protocol:
  closed: bool; read(size: int) -> bytes
  readinto(buffer: memoryview) -> int|None; readline(limit: int) -> bytes
  close() -> None
Process protocol:
  pid: int; returncode: int|None; stdout: BinaryPipe; stderr: BinaryPipe
  _handle: object
  poll() -> int|None; wait(timeout: float) -> int; terminate() -> None

SourceHostJob(driver: Worker, *, deadline_ns: int,
  cleanup_grace_ns: int = 5_000_000_000,
  clock: Clock = time.monotonic_ns,
  io_factory: IoFactory = WindowsSyncIo,
  worker_factory: WorkerFactory = _thread_worker,
  event_factory: EventFactory = threading.Event,
  current_worker: CurrentWorker = threading.current_thread,
  process_factory: ProcessFactory = _popen_source)
job.start_driver() -> SourceJobStatus
job.register_current_driver() -> None
job.check() -> None
job.driver_done() -> None
job.new_observation(kind: ObservationKind, *, deadline_ns: int)
  -> HostedObservationControl
job.request_stop(code: StopCode, *, deadline_ns: int|None = None)
  -> SourceJobStatus
job.fail(code: FailureCode, stage: Stage) -> HostFailure
job.snapshot() -> SourceJobStatus
job.service_cleanup() -> SourceJobStatus

control.request_stop(code: StopCode, *, deadline_ns: int|None = None)
  -> ObservationStatus
control.ownership_snapshot() -> ObservationStatus
```

_thread_worker returns unstarted Thread(target=target,name=name,daemon=False),
internal watchdog/metadata only. _popen_source returns private Popen(list(argv),
stdin=DEVNULL,stdout=PIPE,stderr=PIPE,bufsize=0,text=False,shell=False,close_fds=True).
Trusted test seams, no backend selectors; defaults are stdlib/current41.

Plain int times/counters0..2^63-1, no bool/subclass; deadlines>0, grace1..5e9ns,
stream grace<=existing limit. Reject overflow/backwards clock; now<deadline.
Reject known-started/daemon drivers; exclusive transfer is HOST invariant,
never inferred from ident/is_alive. No public raw handle.

Fixed string-enum values (no aliases/extensions):

```text
ObservationKind = streamed_s16 | materialized_f32
ChildKind = video | audio_s16 | audio_f32
StopCode = cancelled | deadline
FailureCode = cancelled | deadline | start | launch | io | metadata |
  validation | identity | cleanup | native | protocol | overflow
Stage = driver | entry | launch | video | audio_s16 | audio_f32 | validate | retire
JobPhase = reserved | starting | running | draining | stopping | retired |
  cleanup_required
ObservationPhase = new | entering | active | validating | retiring |
  retired_ok | stopped_clean | cleanup_required
StartPhase = reserved | in_flight | returned | uncertain | never_started | joined
ResourceRole = driver_thread | driver_handle | watchdog_thread | launch |
  process | stdout | stderr | metadata_thread | metadata_handle
ResourcePhase = reserved | registering | live | armed | in_call | quiescent | exited |
  joined | reaped | detached | closed | never_created | unresolved
```

Frozen/slotted records; all fields required, None only where typed:

```text
HostResourceStatus(resource_id:int, role:ResourceRole, generation:int,
  phase:ResourcePhase, start:StartPhase|None, pid:int|None,
  alive:bool|None, in_flight:bool, uncertain:bool)
HostDecodeDiagnostic(generation:int, kind:ChildKind, stdout_bytes:int,
  stdout_sha256:str, log_bytes:int, stderr_tail:bytes, truncated:bool,
  transfer_uncertain:bool, natural_eof:bool, exit_code:int|None)
HostFailure(code:FailureCode, stage:Stage, first_ns:int,
  cleanup_deadline_ns:int, winerror:int|None,
  secondary_count:int, last_secondary:FailureCode|None)
ObservationStatus(observation_id:int, kind:ObservationKind,
  phase:ObservationPhase, deadline_ns:int, child_deadline_ns:int|None,
  scope_open:bool, consumer_active:bool, borrowed_driver_id:int,
  resources:tuple[HostResourceStatus,...],
  diagnostics:tuple[HostDecodeDiagnostic,...], failure:HostFailure|None,
  payload_retained:bool, lease_outstanding:bool, transferred_pcm_bytes:int,
  window_resources_zero:bool, cleanup_required:bool)
SourceJobStatus(phase:JobPhase, driver_id:int, driver_admitted:bool,
  driver_done:bool, resources:tuple[HostResourceStatus,...],
  current:ObservationStatus|None, failure:HostFailure|None, own0:bool)
ChildLease(observation_id:int, generation:int, kind:ChildKind)
ChildOutcome(exit_code:int, finished_ns:int, natural_eof:bool)
```

IDs/generations never reused. <=9 resource rows: job3, observation6. ONE active
control/child, max2 sequential stream children or1 float. Retain last detached
status only; diagnostics<=2, tail<=65536, lowercase64hex hash. PID>0 DWORD;
exit/winerror optional DWORD. Unknown alive=None; False never proves start/join.
secondary_count saturates2^63-1 with last_secondary=overflow, primary preserved;
other overflow fails before update. No objects/rawhandles/payloads/tracebacks/
mutable mappings in records; frozen consistency is not execution evidence.

Preserve queue16/line16384/tail65536, stream2MB/float<=16MB log caps. Capture only
counts/tail, no full-log spool/parser. Existing payload/phase/metadata caps govern.

## Driver lifecycle

Reserved driver makes own0=False until explicit retirement; no __del__ cleanup.
Single-use start_driver claims watchdog factory/start and driver.start BEFORE
each invocation, separately tracked, no lock across calls. Wait watchdog-ready
before driver start, <=10ms waits within job deadline. Failure before invoking
driver.start proves never_started; exception FROM start leaves uncertain despite
identNone/aliveFalse. No retry/zero without later entry/exit/join or qualified
no-creation evidence. An unresolved start failure remains cleanup_required.

Pre-claim stop prohibits starts, records never_created, permits own0 without OS.
Post-claim stop retains in_flight. Late entry/registration ALWAYS attaches, no
admission/revival. Even worker exit before start returns cannot clear in-flight;
retire handle only after call returned AND actual exit/join.

Registration requires current_worker identity equals supplied driver, claimed
start, no earlier registration. Record native-call-in-flight, duplicate real
handle through41 outside lock; attach even after stop, then deny work. Failure
forbids work; uncertainty stays. No IDs/OpenThread/per-window duplicate. Unexpected
exit without registration/done fails protocol. HOST target catches/reports via
job.fail, driver-unwinds generators, finally driver_done; job runs no callbacks.

job.check guards before/after filesystem/validation between controls; no unrelated
driver I/O. request_stop works even without current observation: sticky failure,
revoke admission, wake watchdog, return snapshot without OS wait. Control stop
delegates; delivered terminal data not revoked. Foreign/replayed controls fail.

Watchdog repeats driver/metadata CancelSynchronousIo nominal<=10ms until signal+
join or original grace (<=501/5s), independent of requested/no_pending. Covers
filesystem I/O with NO observation. Check-to-call race remains; repeated sweeps
catch delayed entry subject to scheduling/native progress. Controller
service_cleanup advances SAME serialized actions/zero waits, no new worker.

driver_done once revokes work; scopes unwound or explicitly unresolved. It is NOT
exit proof. Observe actual driver exit, join(0),41 signal, stop cancel claimants,
close once. Failure uses original deadline. Watchdog exits after resources retire;
controller joins it. No self-join/daemon/timeout-as-exit. Failed registration needs
certain no-acquisition+join, else unresolved. Watchdog has no pipe-I/O handle;
its native calls tracked in-flight, never claimed universally cancellable.

Window zero covers observation6 resources, excludes job driver/handle/watchdog.
retired_ok also requires scope closed, consumer inactive, aliases/lease released.
Job own0 requires scopes retired, starts resolved, workers joined/never_started,
handles closed/never_created, no action/uncertainty. Retain failed job. Returned
float PCM transfers to caller storage separately; no global-memory-zero claim.

## Shared child lifecycle

Public signatures gain only `_host`:

```text
stream_observed_window(runtime:RenderRuntime, origin:SourceTimeOrigin,
  window:Window, requests:Sequence[Fraction], *, audio_policy:str,
  limits:StreamDecodeLimits=StreamDecodeLimits(),
  _host:HostedObservationControl|None=None) -> Iterator[ObservedWindowStream]
observe_float_audio_window(runtime:RenderRuntime, origin:SourceTimeOrigin,
  window:Window, *, audio_policy:str, limits:FloatAudioDecodeLimits,
  _host:HostedObservationControl|None=None) -> DecodedFloatAudioWindow
```

new_observation: admitted driver only, single-use/exact-kind/no concurrency/replay.
Attach before source checks; try/finally before workers/Popen. Pre-entry stop
retires unclaimed control; claimed/never-yielded entry must unwind. Only
retired_ok/quiescent claims permit next; failure forbids work. Deadline=min(job,
supplied,entry+stream window_timeout); child=min(operation,launch_start+timeout).

Forward `_host:HostedObservationControl|None=None` through _StreamOwner OR float
_audio_window -> payload_stream -> stream_payloads. Reject _owner+_host together;
S16 uses already-hosted owner, float never _StreamOwner.pairs. Preserve all current
audio-contract/clock/metadata-budget/phase keywords and42 charge-before-queue/read.

Private hooks; ChildLease carries identity only:

```text
_enter(kind:ObservationKind, max_log_bytes:int, max_decoded_bytes:int,
  window_timeout_ns:int|None, cleanup_grace_ns:int) -> None
_check() -> None
_launch(kind:ChildKind, argv:tuple[str,...], timeout_ns:int) -> ChildLease
_start_metadata(child:ChildLease, reader:Callable[[],None]) -> None
_read_stdout(child:ChildLease, size:int) -> bytes
_readinto_stdout(child:ChildLease, buffer:memoryview) -> int|None
_readline_stderr(child:ChildLease, limit:int) -> bytes
_finish_child(child:ChildLease, natural_eof:bool) -> ChildOutcome
_note_payload(retained:bool, lease:bool, transferred_bytes:int=0) -> None
_retain_cleanup_deadline(deadline_ns:int) -> None
_fail(code:FailureCode, stage:Stage, winerror:int|None=None) -> None
_leave(success:bool) -> None
```

_enter gets actual stream window_timeout/grace; float window_timeout=None.
It tightens job/control limits, never widens; other passed caps are actual limits.

_start_metadata uses existing parser only; _launch existing canonical argv<=128
guarded strings. Hook owns sole resource lifecycle, source retains parsing/queue/
selector/validation. _StreamOwner logical owner, no duplicate resource authority.
Default bytes/commands/exceptions preserved.

Reserve launch BEFORE process_factory. Late result always adopted for cleanup;
constructor exception leaves unknown partial creation unless cleanup established.
Reserve metadata factory/start likewise; duplicate/register real handle BEFORE
readline/admission. Late handle stays owned, failure permits no reads. No reader
replacement until retirement.

Read hooks mark ARMED/in-flight, recheck stop outside lock, call FileIO, capture
known count/hash BEFORE full-block validation, clear after return, recheck stop.
Cover read/readinto/one-byte EOF/readline; original parser/short-read semantics
stay. No ReadFile/data engine. Unknown transfer fails without retry. Producer
aliases release before next allocation; consumer views before ack. No early close.

_finish_child requires real raw/metadata EOF and exit0 for natural_eof=True;
failure never becomes EOF. On failure terminate/cancel peers BEFORE waits.
One claimant serializes process/handle operations, no lock across OS/join/close.
Driver alone unwinds generators; watchdog cancels/retires resources only.

Actual41: duplicate_current_thread()->int; request_cancel(handle)->requested/
no_pending; is_signaled(handle)->bool/wait0; detach_reaped_process(process)->int;
close_handle(handle)->None. Separate registration adapters, serialize later use.
Metadata signal+join and cancelled action quiescence precede close. Reap, prohibit
Popen use, Detach verifies signal/identity/closed=True, then close once. Uncertain
detach retains Popen/raw handle; uncertain close never retries recycled integer.
FileIO owns fd; close only after reads clear/reentry forbidden, no native borrowed
handle close/cross-close unblock. Stalled native action/worker stays owned.

Video retires BEFORE mandatory incidental S16; next=None/real S16/source/pins/seal
remain. Float receipt/PCM/fingerprint stay inside context through _leave; return
only after retirement/final check. No skipped audio/receipt change/fake cleanup.
Observed finite f32 is not source-input finiteness/native fidelity/prepared audio;
never relabel it or incidental S16. All-SOLID still needs audio.

## Deadlines and failures

ONE first failure: cleanup=min(first_ns+grace,caller stop deadline,current earlier
stream cleanup). Later calls only tighten; primary/code/time stable, secondaries
bounded. Equality fails before work/ack/return. Job deadline covers gaps; float
operation deadline is new hosting constraint, not receipt field.

Preserve39 FIRST-cleanup deadline, even normal video retirement, through S16.
Clean next window excludes previous normal deadline; failure makes job grace
sticky. Defaults unchanged. Failed qualification needs separate revision, no
silent fresh audio grace or limit increase.

HostedObservationFailure(ValueError) holds frozen failure:HostFailure,
status:SourceJobStatus, diagnostics:tuple[HostDecodeDiagnostic,...]. Owner stores
no exceptions/tracebacks/str(exc), only bounded DecodeFailure facts. Primary
EOF/checksum/receipt/source error survives cleanup errors; count/hash/tail persist
through post-EOF checks/exit. Driver releases provisional PCM/result/generators;
caller copies/tracebacks remain outside producer storage guarantees.

service_cleanup uses zero waits/checked native actions, watchdog remaining grace.
Expiry retains exact survivors, allows zero-wait progress only, no fresh grace/
uncertain-close retry/work. No universal latency/return/launch interruption or
native/RSS/queue proof. Unknown/in-flight/uncertain ownership blocks publication.

## Pure test manifest

Fake EVERY clock/Worker/Event/Popen/FileIO/current-worker/native boundary BEFORE
entry; actual41 fake DLL, no whole positive validators faked. No real Thread/
Timer.start/wait/import child. BGR<=48B, PCM<=32B, calls<=16B, <=32 events/case.
Reuse real39/42 tiny receipts/byte validators in scoped integration nodes.

New test_source_host_ownership.py nodes:

1. test_inert_defaults_and_bounds: no constructor calls; enums/scalars/caps,
   wrong driver/kind, overflow/backwards clock.
2. test_owner_start_states: pre-stop/factory failure; blocked/raising/late start;
   identNone/aliveFalse not zero; late registration/early exit/no retry.
3. test_driver_registration_and_job_stop:41 duplicate/admission; cancellation
   before register/between controls/filesystem; done not exit; signal+join.
4. test_control_single_use: float->stream one driver; replay/concurrency/foreign
   kind reject; pre-entry stop/never-yielded launch retained.
5. test_child_launch_and_reader_start: reserve BEFORE Popen/start; late outcome,
   metadata nativehandle before read, no extra worker.
6. test_cancel_and_partial_io: all three read APIs/EOF; no_pending/late entry;
   requested/normal/abort/error; partial count/hash/tail, sticky stop.
7. test_retirement_and_uncertainty: terminate-before-wait; no cross-close;
   actual41 reap/Detach/Close errors, signal+join/no selfjoin/retry, both zeros.
8. test_deadline_and_primary_failure: equality/retained39/single grace;
   late validation+cleanup error; diagnostics preserved/exact survivors/no PCM.

Existing test_source_decode.py adds ONLY scoped nodes:

9. test_hosted_float_real_path: real42 one/two finite blocks/phase early rejection/
   wrapper/receipt/late failure; retire-before-return/default compatibility.
10. test_hosted_stream_real_path: real39 selection/ack/None/S16/loader, aliases,
    video->audio retire/borrowed failure, unchanged seal/default, no promotion.

Unexecuted. Review one owner/defaults/actual41. Prior OS manifest UNEXECUTED,
no expansion. Native races/ABI/latency/positive completion remain runtime gates,
no universal bounds. Editor/publication gates remain. FINAL_IDLE: document own0.


## Controller adoption clarifications — Task45 activation still pending

Root FULL-read final17935B contract SHA
FE1C4C0FACB1AD3DBB9379A658525C14DA97104106AA4DFFDF5B3A6997B7BD82
and prior37700B proposal. This adopted copy makes the exact types/records/APIs/
owner-start lifecycle/hooks/bounds/default preservation above binding despite
its retained historical proposal/unexecuted wording. No source writer is
activated until44independentclosure, exact45brief and ORIGINAL45freeze. Earlier
SourceStreamControl/external-start proposals are superseded background, not APIs.
No user style/vision change, actual OS/media/model campaign or new authority.

Dependency direction: source_decode -> _source_host_ownership -> _windows_pipe_io
and stdlib. The owner MUST NOT import source_decode/render_clock/audio_selection/
fonts/renderer/models/config, nor duplicate parser/selector/PCM/receipt engines.
Existing logical streamed owner still owns selection/leases, new helper owns
HOSTED child/pipe/thread/native lifecycle exclusively. _host=None stays unchanged.
No second active supervisor/transport/file publisher is hidden in this unit.

Use the exact SEVEN frozen/slotted records, HostedObservationFailure and explicit
protocol/type aliases above. Fixed string values may be Literal aliases or
str+Enum compatible withPython3.10, no StrEnum requirement. Record tuples/byte
strings detach under bounded exact-type guards; no MappingProxy backing/count
assumptions, unknown iterator or generic serializer. Cap before traversal/copy;
fields have the stated finite scalar/tuple limits. ChildLease is metadata only,
not permission for a different job/control to claim resources. Status snapshots
contain no Popen/Thread/native handle/generator/frame/PCM/exception objects.

Add is_set()->bool to WakeEvent protocol (the actual stdlib Event and required
fake can provide it). Default factories remain inert until start/registration/
launch at the named boundary. Locks/queues used for state/storage are permitted;
newtests must fake every potentially blocking wait/start. No native call or join
while state lock is held. A claimed action keeps ownership until its return,
including after worker exit or deadline. Snapshot never executes blocking native
calls to manufacture current truth.

SourceHostJob constructor has exclusive authority over the supplied genuinely
unstarted driver. HOST must never call driver.start/join independently after
transfer. Owner-start is the ONLY invocation; HOST defines that driver's target
before transfer and retains job for servicing/cleanup. Target first registers,
executes explicit source work, catches unclassified errors via job.fail, and
finally calls driver_done. A HostedObservationFailure already recorded need not
be double-counted as a fresh primary. The job executes no target callback itself.
Reserved own0=False is conservative even before acquisition; a pre-claim stop
can retire never-invoked starts under exclusive transfer. No ident/alive inference
can retire a claimed/uncertain Thread.start. Actual signal+join/action quiescence
precedes close. Partial/late start uncertainty cannot be discarded for convenience.

Job watchdog covers deadline/stop outside controls, including future job-owned
filesystem preparation. Job-level request_stop is mandatory and must not require
an invented active audio window. new_observation is driver-only after admission;
a control is single-use and bound to exact job/kind. Windowresourceszero excludes
live job driver/handle/watchdog; jobown0 includes all. No extra driver/watchdog
is charged to an imagined future owner. Metadata workers remain per-child and
retire before next child. At most1observerchild,2sequentialstreamchildren,9live
resource rows (3job+6observation),2diagnostics,64KiB tail/child.

Pure exact argument/type/limit guards may precede attaching a control only if
no source/identity/hash/filesystem/native/worker/process work is reached and the
HOST's failure path retires the unused reserved control. Valid hosted entry
attaches before all those operations, and its try/finally covers never-yielded
contexts/start failure/post-EOF receipt/source checks. Private hook organization
may refine the two-stage entry/tightening implementation, without a new public
mode, duplicated lifecycle or losing reserved-entry ownership. Preserve existing
Window class and @contextmanager Generator annotations, publicdefaults and all
private float contract/metadata-phase arguments; only optional _host is added.

Root focused current source confirmed BOTH metadata queues are maxsize16 and
public float/stream entries use actual Window. Child maxlog/maxdecoded/timeout
must be bounded against the actual passed39/42 limits BEFORE work, not only the
broad record's63bit range. Use actual stream2MB/float16MB logging, line16384 and
existing payload/frame/sample limits; no field/receipt/policy ceiling change.
The explicit job deadline is a HOST budget; later production/harness budgets are
separate, not a licence for unbounded window/child work. No new implicit retries.

Canonical argv is already produced by existing builders. Owner only validates
its bounded tuple/plain string transport (<=128args, total UTF8<=64KiB before
Popen; no shell/modeltext/inference), not another command/filter builder. This
transport cap leaves legitimate current builders untouched; exhausted cap fails
before launch. Default process uses stdinDEVNULL/stdoutPIPE/stderrPIPE/bufsize0/
textFalse/shellFalse/close_fdsTrue as stated; no process per metadata tag/request.

Preserve39's earlier FIRST cleanup deadline through incidental S16; do not fix
that separately recorded qualification limit silently. A clean NEXT observation
has its own normal window deadline, but any job failure's first cleanup grace
never resets. Requested/no_pending cancellation is never completion; false
liveness or a done callback is not exit; no crossclose/generatorclose race,
uncertain native close retry, daemon abandonment or own0/publication fiction.
A delayed platform operation may outlive grace: retain exact owner/action state,
block next work and leave actual runtime guarantees unclaimed. No universal
hard-return/launch-interruption bound is added by fake tests.

Hosted failure preserves first code/time/stage, known stdout count/hash and
bounded logtail through validation/EOF/cleanup; no exception/traceback/str(exc)
history stored in owner. Raising a failure may chain caller-visible original
cause; that is not an owner history/cache. Raw status values still get bounded
plain-type validation before inclusion. No generic logging subsystem/diskspool.

This unit's ordinary gate is PURE/FAKE only. Eight exact new ownership-test names
and TWO new source integration names above, tiny actual39/42 constructors/bytes,
actual41adapter with fakeDLL. Every Popen/Thread/Timer/Eventwait/clock/native and
HOST file/pin/stat boundary is fake BEFORE entry. Existing queue/lock storage is
allowed; real worker starts/waits are not. Do not run old streamed-owner tests
that use actual threads or misleading fake-child tests using sys.executable.
Only exact separately enumerated old PURE source nodes may be added in45brief.
No whole source/host/native/media/font/GPU/model suite or old proof guard.

Separate OS-only manifest from earlier proposal remains UNEXECUTED and will need
its own exact guarded adoption after task approval. No testfile automatically
launches it. Source signal/phase/RSS/performance, preparedaudio, spatial/encoder/
metrics/CLI/finalA-V/textvisual/model/human gates remain later. This source owner
seam alone does not deliver an MP4 or prove Windows ABI/cancellation.

Test-budget precision: <=16B applies to RAW payload read/readinto chunks, not
the required textual showinfo/ashowinfo lines. Synthetic logtotal<=4096B/child
and actual linecap16384 still apply. <=32events/case means explicit scripted
scheduler transitions; cap actual fake native calls<=256/case too. A cancellation
counter boundary can be seeded near its limit, not tested with501real/fake loop
iterations. This avoids a contradictory fake manifest while keeping tiny data,
bounded finite transitions and no actual worker/native operations.

Job own0 concerns ONLY resources owned by SourceHostJob. Filesystem preparation
in its driver can be cancelled through the registered handle, but this helper
is not a general broker of every caller-opened FD. Future host must ALSO retain
and gate on font/preparation file-close uncertainty; sourcejobown0 alone cannot
erase another helper's uncertain cleanup or authorize a new campaign.

Protocol-boundary precision: preserve existing one-byte terminal stdout EOF
probe. Reserve at mostONE such extra byte perchild outside accepted decoded
payload; zero bytes proves EOF, an actual byte is unassociated failure and is
counted honestly in diagnostics, never accepted data or a cap increase. Stderr
reads likewise use bounded remaining-log/line overflow detection (+1 sentinel),
then fail before metadata admission/extra retention if exhausted. A stream
exactly at its byte/log cap must still be able to prove naturalEOF. No retry or
unlimited probe loop. Cover exactcap positive and extra-byte failure with tiny
fakes; actual source limits/receipt meanings stay unchanged.

Stream.finish remains callable INSIDE its context as before: child/metadata
resources may be zero while the control's scope is still open. Its returned
SourceStreamFinalization is metadata, not a native retirement/publication seal.
Control retired_ok additionally requires context exit/lease and aliases clear;
float returns only after full context retirement. Do not create an impossible
finish/exit dependency or certify jobzero from the stream's metadata.

Handle-generation precision: IDs/generations are never reused, but Windows raw
handle INTEGERS can legitimately be recycled after a CERTAIN close. Keep live/
inflight/uncertain generations distinct; do not require global historical integer
uniqueness or grow a permanent used-handle set. Never close/retry an uncertain
old integer as a newly admitted handle. Simultaneously live private native
resources must not be conflated; fake fixtures need distinct live handles.

