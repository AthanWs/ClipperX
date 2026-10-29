# Task47: bounded owner OS harness SOURCE design

Adopted by Root after Task45 fix3 and Task46 independent source approval.
This adopts implementation and PURE source verification of THREE new files,
NOT execution of the Windows/native campaign. No manifest/run guard/run directory
is created by this adoption. User vision and Phase3 global constraints unchanged.

## Binding documents and precedence

Read these COMPLETE. Their unchanged bytes are incorporated as exact contracts:

- `.superpowers/sdd/phase3-owner-os-harness-contract-proposal.md`,26108B,
  SHA256 D1241C4D79F40BB91223657ACCC3E8669D269E2152225349DB13304E4CE26D6D,
  INCLUDING its6148B controller reconciliation.
- `.superpowers/sdd/phase3-owner-os-manifest-proposal.md`,15929B,
  SHA256 06B4F2B9A29630CF1088D7F6C0E759966EEA671B4732099EF4BFCCE9F72321FE.
- `.superpowers/sdd/phase3-owner-os-controller-notes.md`,11329B,
  SHA256 895B8CF86F536581FA778A8A12438068FAE41E1DCA84957AC76CB57FFC8A7306.
- `docs/superpowers/specs/2026-10-15-hosted-source-ownership-design.md`,28215B,
  SHA256 D8D46CF9891150CE9CAD8ECE6A4BEC9B3B461C781356899540766655DD476C58.
- `.superpowers/sdd/phase3-global-constraints.md`.

THIS design overrides stale/pending status, proposal-only/doc-writer restrictions
and the explicit clauses below. Then the contract reconciliation overrides the
original contract; controller final clarifications override earlier proposal
wording. Five-case proposal supplies case intent, not an additional campaign.
Historical45I5/46pending statements are superseded by completed source reviews.
No open menu of implementation choices or new public owner API is adopted.

## Exact scope and independent gates

Only NEW:

1. `.superpowers/sdd/host/owner_os_gate.py`
2. `.superpowers/sdd/host/owner_byte_fixture.py`
3. `clipperx/tests/test_owner_os_gate.py`

Actual45/41 remain black boxes; source_decode, originals, configs, dependencies,
CLI/editor/normalizer, font/assets/model/API/network/media are outside this unit.
One fresh writer; independent frozen-diff Spec+Quality review before source
completion. Then Root separately freezes ACTUAL host/artifact/argv/byte manifest
and adopts ONE C1..C5 native campaign. Source writer MUST NOT run it or a fixture,
actual worker, Popen, WinDLL, native call or OS-only probe, including for debugging.
No campaign from pytest collection/import/help. No Git init/repair/worktree/commit.

Source import is inert: constants/record/function/class definitions only. No path
read/probe, owner import, native constructor, worker/Event construction, process,
clock or CLI parse until guarded main/functions. Importing stdlib is permitted;
constructing an actual RecordingIo subclass occurs only after reserved loading.
No top-level skip/env gate. CLI exactly required absolute --manifest/--evidence
and help, no force/retry/case/alternate backend. Guarded fixture takes ONE code.

## Final approved inputs / actual hook binding

These are immutable READ inputs, not writer-owned files:

| Input | Bytes | SHA256 |
| --- | ---: | --- |
| clipperx/_source_host_ownership.py | 60133 | 6A846C12B050024501D60FA793BFC38FF26C48EC17C4FC78B667648C96BEA879 |
| clipperx/source_decode.py | 92887 | CDB3B713C403E1F7F2173D09F5E6DE44817A46622DA9DD446C311D7EC14A5151 |
| clipperx/tests/test_source_host_ownership.py | 39646 | 035D3F56517BFD4E85D0A73D8C2A4309AC3542B3507F3E16EAB84C0420F1CD38 |
| clipperx/tests/test_source_decode.py | 85791 | 323153FFF9245B2300DD257D5A5015917687437B5D7D99189694BDABD7111594 |
| clipperx/_windows_pipe_io.py | 8272 | 9FADCADC482953CEF0946F98398890B1008DEBE8A32524CDBD2DFDF63422E72D |
| clipperx/__init__.py | 408 | 58F1735AC519DADB81613312B1FDE303BAFD814E67C02F96E7A7D329D5CC1631 |
| .superpowers/sdd/phase3-task-6b5-report.md | 44331 | 43C1786C6681C748DBDA2DB95B07F28C1C3DA6B7D363BF50D45882B571A31408 |
| .superpowers/sdd/phase3-task-6b5-review.md | 39009 | 51FD24A9C57E50BC29A5CB8F16F65DCBF929947B5062A6CFF8A213C75C997348 |

Root read exact signature/status sections after writer READY. Constructor:
SourceHostJob(driver,*,deadline_ns,cleanup_grace_ns=5e9,clock=monotonic_ns,
io_factory=WindowsSyncIo,worker_factory=_thread_worker,event_factory=Event,
current_worker=current_thread,process_factory=_popen_source).
Production harness leaves clock/current_worker/event_factory at actual defaults;
inject ONLY real recording worker/io/process factories. Actual methods:
start_driver(),register_current_driver(),check(),driver_done(),
new_observation(kind,*,deadline_ns),request_stop(code,*,deadline_ns=None),
service_cleanup(),snapshot(),fail(code,stage).

Control _enter(kind,max_log_bytes,max_decoded_bytes,window_timeout_ns,
cleanup_grace_ns); _launch(kind,argv,timeout_ns)->ChildLease;
_start_metadata(child,reader); _read_stdout(child,size);
_readinto_stdout(child,memoryview); _readline_stderr(child,limit);
_finish_child(child,natural_eof)->ChildOutcome; _retain_cleanup_deadline(ns);
_leave(success); ownership_snapshot(). Use ACTUAL returned leases and public
frozen status fields. No owner private resource traversal or invented getter.
Kinds streamed_s16/materialized_f32; child video/audio_s16/audio_f32 are fixture
labels ONLY. ProcessFactory receives frozen argv. WorkerFactory receives target.
Failure code/stage choices are actual owner literals, no new status vocabulary.

Task45 clean retired_ok excludes old normal deadline for a new first job failure;
native window zero while SCOPE OPEN does not. Sticky existing failure can only
tighten. Preserve these policies: no harness grace extension or owner changes.

## Finite manifest correction: exact15 artifact roles

Use the contract's finite records/keysets/caps/parser/path/UUID rules, with
Artifact.sha256:str for EVERY artifact. package_init EXISTS and must match408B
hash above. Null/ABSENT is rejected, not an alternative mode.

Replace ambiguous `design` role with TWO exact roles: owner_design,harness_design.
The exact15 roles and fixed workspace-relative paths are:

| Role | Path |
| --- | --- |
| owner | clipperx/_source_host_ownership.py |
| source_decode | clipperx/source_decode.py |
| owner_tests | clipperx/tests/test_source_host_ownership.py |
| source_tests | clipperx/tests/test_source_decode.py |
| windows_io | clipperx/_windows_pipe_io.py |
| owner_design | docs/superpowers/specs/2026-10-15-hosted-source-ownership-design.md |
| harness_design | docs/superpowers/specs/2026-10-15-owner-os-harness-design.md |
| owner_report | .superpowers/sdd/phase3-task-6b5-report.md |
| owner_review | .superpowers/sdd/phase3-task-6b5-review.md |
| harness | .superpowers/sdd/host/owner_os_gate.py |
| fixture | .superpowers/sdd/host/owner_byte_fixture.py |
| harness_tests | clipperx/tests/test_owner_os_gate.py |
| harness_report | .superpowers/sdd/phase3-task-6b7-report.md |
| harness_review | .superpowers/sdd/phase3-task-6b7-review.md |
| package_init | clipperx/__init__.py |

Host executable is separately exact role python. Its bytes/hash/version/bitness/
Windows build are captured by Root AFTER source review; source tests fake them.
No guessed future values/example production manifest or new file during source
work. Keep UTF8<=32768B/depth8, exact finite types/no bool-int/floats/NaN/BOM/
duplicates/unknown keys; per-artifact64MiB/aggregate128MiB; canonical absolute
Windows paths<=1024/no UNC/device/ADS/NUL/dot-dot/aliases. Paths bind workspace,
fixed guard and exclusive run directory; no run-id bypass. Filesystem checks
are bounded protocol, not race-proof or hostile-filesystem sandbox guarantees.

Fixed limits (do not add a native-call quota): case_active_ns15000000000,
cleanup_ns5000000000,campaign_ns200000000000,children3,workers3,
io_bytes8388608,raw_call_bytes65536,stderr_bytes131072,line_bytes16384,
evidence_bytes1048576,handle_generations32,service_ns10000000,
stop_check_ns100000000. One cleanup budget tightened by all earlier bounds.
No hard native-return guarantee or new grace/component/case/retry.

## Owned source loading, not ambient imports

Apply complete fixed-three-module reconciliation verbatim: preflight bounded-read
package_init/windows_io/owner once, retain verified immutable source bytes, then
AFTER exclusive guard/evidence reservation compile/exec in fixed canonical order.
Reject ANY ambient clipperx or clipperx.* sys.modules entry; do not purge/reload.
ModuleType+ModuleSpec(loader=None), __cached__/__loader__ None, package search
paths EMPTY; register BEFORE exec, bind completed children. Compile verified bytes
with flags0,dont_inherit=True,optimize0; no reopened path/PYC/finder/sys.path/
generic import engine. Return SAME actual SourceHostJob/WindowsSyncIo objects;
actual RecordingIo subclasses that same41. No get_version/distribution call.
CPython/stdlib/startup trusted, not a sandbox. Other source/tests/reports are
identity pins ONLY; never import source_decode or fixture to obtain an oracle.
Both future foreground and fixture argv use exact frozen exe with -I -B.
Pure loader tests use private fake bounded registry/compiler/executor effects
on tiny inert bytes, never actual45source execution or ambient module purging.

## Fixture writes and attempted I/O accounting

Use existing raw binary FileIO stdio, no text/buffered writes, new fd, detach,
msvcrt or alternate backend. Missing/unsupported raw stream fails nonzero.
Each nonempty READY/body/DONE literal block has ONE raw write; require plain int
exactly len(block), otherwise fail without retry. Hold input is ONE raw read(1).
No child timer/thread/sleep/import of controller/editor. Parent supports short
reads; literal emit/hold modes and exact digests are the controller notes table.

Controller constants are independent literal/oracle values, never imported from
fixture. Before launch precharge known fixture output requested sizes PLUS max
one-byte hold input. This is conservative reservation, NOT measured child kernel
call counts. Before EACH parent read/hook/direct read charge REQUESTED size,
including READY/log/EOF probes. Aggregate<=8MiB/case/raw<=64KiB; actual observed
count/SHA separate. No hidden buffered retries/raw dumps/unbounded communicate.
Artifact hashing is outside fixture-pipe accounting, with its own bounded reads.

## Recorder, custody and cases: final binding

Use complete proposal machinery, no second process engine. Real Thread subclass
preserves default current_thread identity; Source45 alone starts/joins driver.
Record factory injection honestly, not literal-default-factory qualification.
Reserve rows BEFORE start/acquisition; metadata binding uses actual ChildLease.
RecordingIo delegates each actual41 method ONCE unchanged outcome/error, retains
raw result BEFORE logging/formatting, coalesces32 generation records. Unknown
capacity prevents acquiring; certain retired raw integers may recycle into NEW
generations, uncertain ones cannot. No extra APIs/OpenThread/private traversal.

Reserve Popen launch row BEFORE construction. After offered_source commit harness
NEVER polls/waits/terminates/reaps/Detaches/closes source process/stdout/stderr,
even when acknowledgment unknown. Return actual successful Popen despite late
stop/recording failure. Harness separately owns C3/C5 stdin and C4 external peer
plus endpoints/borrow flags. No dropped late result, double owner, active-pipe
cross-close or Popen operation after Detach. Exact disposal evidence or retain.
C5 ownerown0 alone is insufficient; actual SAME Popen detach/close+FileIO closed
can confirm late disposal without ChildLease; missing proof remains unresolved.

C1 normal4096B exact EOF/reap/reader+driver signal/join/close/watchdog retirement.
C2 ONE streamed_s16 control, SAME driver, video64B then audio_s16 32B, first child
fully retired BEFORE next. At first EOF BEFORE hook/retirement sample t_bound;
D_bound=min(t_bound+5e9,case_deadline,campaign_deadline). Pass exact D_bound ONCE
to _retain_cleanup_deadline before first _finish_child; carry through second child
and _leave. Record timely completions BEFORE D_bound, label HARNESS_CARRIED_BOUND,
not actual private cleanup-start/deadline getter or OS expiry-enforcement proof.

C3 both exact READY outputs/arm markers precede stop. Arm is not pending proof;
actual calls unwind/sticky failure/retirement required, then harness stdin close.
C4 prior tiny control retired_ok/scopeclosed before direct between-control read
of separately harness-owned peer stdout. REQUIRED actual requested cancel on
registered driver generation AFTER READY/arm, peer still alive and unreleased/
unterminated. Capture actual return/error, not presumed995. Actual driver signal/
join BEFORE normal stdin release, then peer EOF exit0/reap/41Detach/Close/stdout
close after borrower return. Record jobown0 with harnesszeroFalse then combined0.
Missing requested/early release/death is INCONCLUSIVE and stops later cases.
At min(stop+4s,original_cleanup_deadline-1s), unresolved driver permits FAILED
proof fallback termination of ONLY that harness peer, not a PASS/cross-close or
new grace. C3/C5 source-owned processes receive no harness fallback termination.
C5 real child CREATED on existing driver, factory release held. Controller stop
and frozen deadline BEFORE release. Factory always offers actual Popen late for
owner cleanup-only, NO metadata/data/new grace. Release Event also on controller
failure/STOP. This is delayed return of already-created process, NOT interruption
of stuck CreateProcess. No sixth case, sleep-to-expiry or automatic replay.

## Main serial file custody, evidence and terminal retention

Preflight/evidence file operations are also owned: max ONE active raw file plus
action marker, retained reference and uncertainty on unknown operation/failed
close. No uncertain close retry/drop or false combinedzero from jobown0. This is
small serial custody, not generic file broker/worker. It exists early enough to
retain preflight failures too. File/evidence failure blocks subsequent cases.

All evidence INCLUDING guard/STOP/two manifest copies<=1MiB, per-case128KiB,
summary64KiB; tails<=1024 with explicit truncation, no silent completion claim.
No per-sweep lists/logger worker. Coherent owner snapshots and separate timestamped
harness facts are not fabricated atomic cross-ledger snapshots. Combinedzero
requires jobown0+harnessretired+no unconfirmed offers/inflight/file uncertainty.
No global handles-zero claim.

STOP checked nominal<=100ms/service<=10ms with no input worker, all waits remaining
original budget. Stopfirst unexpected failure/unknown/inconclusive/survivor/record
or evidence failure. PASS only with actual required proof and combinedzero.
Retain guard/evidence on every outcome. main0 PASS/2 invalid preflight/1 other
CLEAN failures; CLEANUP_REQUIRED retains RunState/resources and foreground session
for scoped recovery, no forced exit/abandonment/detached service/visible console.
No guarantee cleanup/session returns by200s if native/FS action stalls. No global
census/kill/delete/install/service/database/SoraTV/foreign workspace action.

## Source TDD and acceptance

Use EXACT10 ordinary PURE node names from reconciled contract; parameterization
allowed, zero skips/xfail/env flags. Guards BEFORE RED cover actual Thread init/
start/join, Event/Condition/queue waits, timers, real clock, Popen/run, WinDLL/
native, host/path/input/fixture and any fallback. Testfile may bounded-read ONLY
its own code and tiny local test fixtures/import metadata as explicitly guarded;
no production inputs/probes. Dependency/import fixtures restored; no actual owner
source exec in source tests. Fake effects, real parser/ledger/decisions/orchestration
and delegation wrappers, not a mock-result-only PASS. Finite<=32 scripted steps/
variant, payload<=64B except literal4096B recipe, seed near caps/no huge loops.

RED inert/manifest/reservation then smallest implementation; RED recorder/custody,
then fixed cases/caps/retention. After LAST edit: exact NEW testfile ONLY, scoped
3path Ruff/mypy/py_compile, text-only3.10 grammar. No45/46/old/native/media/proof
repeats, package suite or suppressions/config tweaks/new interpreter/dependency.
Report genuine command/output/exits and any intermediate issues honestly.

Root freezes3ABSENT markers before dispatch; never refresh. Writer changes ONLY
3newpaths and writes phase3-task-6b7-report.md, ideally<=18000B. READY includes
sizes/SHA, immutable input/marker preservation, TDD/final gate evidence, honest
limits and FINAL_IDLE owncommands/children/helpers/workers/handles/sessions0.
If adopted hooks cannot support requirements, report precise blocker/options;
do not silently add owner API, alternate loader/native backend or weakened proof.
No actual Windows/native/latency/PCM/A-V/rendered-font/MP4/editor/model/publication
claim follows from PURE source gates. Separate campaign activation remains later.
