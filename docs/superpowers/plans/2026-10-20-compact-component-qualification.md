# Compact component qualification implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the candidate-only component qualifier without duplicating Root's historical build audit or reducing any of the24 real media operations.

**Architecture:** The externally pinned v2 admission binds one fixed Root acceptance and exact actual files. Parse only admission/acceptance/candidate; hash the finite raw record inventory without traversing it. Reuse existing pure fixture/oracle helpers and the original owner/files/tools, then integrate the24-operation driver in a second reviewed unit.

**Tech Stack:** Python3.12.6, existing ClipperX source-host ownership/retained-file/render services, pytest fake process/clock boundaries; original native FFmpeg candidate only in a later separately admitted campaign.

## Global Constraints

- Task116 accepted design12508B/SHA256960CACE54A3C9B02ED7A331E6C2EE169AE277F0CF87293D675F70133C9293CEC and complete review12562B/SHA2563E84A3A1FE7CDA1FA46A165AB24F30C55B8C59502317446466A37CD039A0B2D8 govern compact admission. Task106 sections1,3,4,5,6 and Task112 lifecycle clarification remain normative.
- All24 Task106 section4 operations remain in order;24 attempts/one live child, stop on failure. No listing substitutes for execution.
- All limits remain:1200s from first admission=1185work+5cleanup+10settlement; original per-child deadlines,256MiB owned,32MiB captures/sidecars,1572864 stderr,256paths/4096generations/16live,1GiB free beyond remaining reservation,16384guard/262144operation/2097152terminal and every original stdout/stderr/graph/ASS/list/direct-file cap.
- Use the same SourceHostJob/RetainedFiles/OwnedRenderTools owner, clock and injected factory, immutable CandidatePair and pure candidate-only origin. No helper/build-producer import before byte/origin custody, selector override, second owner, alternate runner or OS descendant/disk-quota guarantee.
- Source-only units own exactly scripts/qualify_production_runtime.py and clipperx/tests/test_production_runtime_qualification.py plus their assigned report. No app/helper/build/lock/receipt/asset/media/provider mutation or actual native process. All existing native failures/snapshots/history remain in place.
- No Git repository exists; do not initialize/commit/push. Filesystem snapshots/full diffs are review identities. No finalfile fake suite before integration; focused fake RED/GREEN per unit, whole new fake file once after integration, broad app/global/native gates later.
- Actual runtime acceptance is absent. A different recovered build root/identity requires a finite exact-path design amendment, never permissive discovery or a fallback. This does not block fake-only admission implementation for the accepted fixed contract.
- Root subsequently accepted Task120's exact recovery mapping: O8=O:/Escritorio/ClipperX/.superpowers/sdd/upstream-motion/production-build-v8-nvenc, build_id=ffmpeg-9.0.2-clipperx-production-v8-nvenc. Task116 runtime_root/publication/pair/inspection/closure/source_notice/maps/acceptance paths map O7 to O8 exactly; publisher_source=O8/recovery.py, publisher_lock=O8/recovery-lock.json, publisher_review=.superpowers/sdd/task-120-source-review.md. Add accepted Task120 design12443B/3C50B97C24C22EF67C17EE05E826CA248E6FD16D9130DD7B433B47CDB3A04B1D and review10251B/51435A3253D5B47F20FA33DA34C5A5712D0A035DD5A50DC129EF6A8AF89B925C to required pinned lineage dependencies. This supersedes only the old fixed target, not any schema/rawrole/media/cap/owner/quality contract. See task-120-root-design-acceptance.md. Actual source/publication/acceptance remains absent; no runtime fallback.

## File responsibilities

scripts/qualify_production_runtime.py remains the bounded controller: stdlib-only import/default entry; compact validation/custody; pure candidate fixtures/oracles; original-owner campaign/terminal orchestration. Remove superseded transitive historical verification, cost tables, parser exceptions and active-only tests; original bytes are frozen under task114/head2 and task117/base. Do not add files/frameworks or duplicate operation table/fixture code.

clipperx/tests/test_production_runtime_qualification.py contains detached tiny synthetic inputs, real controller checks, original owner services with injected fake boundaries and native-escape traps. Preserve fixture/oracle/owner coverage, replace only superseded history-input tests. Final tests never invent actual Root acceptance/native identities.

### Task1: compact admission and immutable operands (Task117)

**Modify:** the two source files above. **Report:** .superpowers/sdd/task-117-report.md.

**Consumes:** accepted Task116 exact schemas/paths/roles and current Pin, canonical_json, CandidatePair, DetachedIO and pure fixture/oracle helpers. **Produces:** validate_inputs(cli_pin, *, io=None, clock=None) returning an immutable ValidatedInputs value; guard_bytes(inputs, cli_pin); existing admit/main remain fail-closed until Task2. Preserve original pure helper signatures unless the new input contract requires a named parameter change, which the report must list for Task2.

- [ ] Freeze the exact existing source before edits; Root owns base snapshots. Read Task116 fully and Task106's retained clauses; do not read old transitive receipts/inventories. Keep Pin type/caps; reduce JSON parsing to default bounded small-document grammar.
- [ ] Add genuine failing tests using a tiny synthetic acceptance/candidate/files fixture. Required first contract test has this shape (fixture creates only detached fake bytes, no real pins):

```python
def test_compact_admission_requires_external_identity(compact_case):
    case = compact_case
    bad = Pin(case.cli_pin.path, "0" * 64, case.cli_pin.bytes)
    with pytest.raises(ValueError, match="pin|identity|hash"):
        validate_inputs(bad, io=case.io, clock=case.clock)
    assert case.io.created == []
    assert case.native_attempts == []
```

Run only the new contract selection with approved .venv-whisperx/Scripts/python.exe -B -m pytest clipperx/tests/test_production_runtime_qualification.py -q -k compact_admission, using a unique Task117 basetemp. Record actual RED and why expected before implementation.
- [ ] Implement exact v2/admission/acceptance/candidate projection, eight raw roles, ten notice order, source/assets/limit/Pin custody, fixed destinations, path/file-identity aliases and absent disjoint namespace validation from Task116. Raw bounded hashes must stream one closed handle/chunks; do not retain maps/executables in a giant cache. No referenced metadata traversal or inherited-lock equality. Immutable operands retain exact admitted bytes/Pins and the first monotonic_ns campaign origin. The entry point contract is:

```python
def admit(cli_pin, *, io=None, clock=None):
    inputs = validate_inputs(cli_pin, io=io, clock=clock)
    raise ValueError("NOT_ADMITTED: component driver not integrated")
```

No mkdir/owner/process from this increment. Guard serialization validates exact guard-v2 ten keys and every field projection before any later mkdir, including serialization/oversize failure. Validate named code/source origins before importing local helpers; tests are pinned, not imported.
- [ ] Test all Task116 section6A negatives: v1/missing/extra/duplicates/nonfinite/bool, hashes/sizes/caps, wrong role/root/pair/notice, aliases/reparse/overlap/conflict, import origin/helper pin drift, limits, oversized small documents and prechecked guard. Assert opaque fake maps/inspection containing non-JSON bytes hash successfully without parse/traversal. Preserve default CLI NOT_RUN/exit0/zero setup I/O and native-escape traps. The immutable result must not expose mutable admitted authority.
- [ ] Run affected focused selections once final; retain existing pure fixture/oracle/owner tests that were not changed. Report exact RED/GREEN commands/output, final hashes, removed superseded symbols, source dependencies and exported Task2 interfaces. Root freezes full two-file diff; independent task review must approve SPEC and QUALITY before Task2. No actual native campaign or whole-file suite yet.

### Task2: original24-operation driver and settlement (Task119)

**Modify:** the same two files only. **Report:** .superpowers/sdd/task-119-report.md.

**Consumes:** reviewed Task117 ValidatedInputs/validate_inputs/guard_bytes and preserved operation_argv/fixture_bytes/oracles/OwnedAttemptSequence/publish_terminal plus original same-owner helper APIs. **Produces:** admit(...).execute() through original SourceHostJob driver; default behavior remains zero-I/O.

- [ ] Read Task117's exact report/interfaces and accepted diff; freeze base. Add one failing fake complete-run test before orchestration. It must assert exactly24 original argv, single-live-child, original owner/files/tools identity and final detached terminal sequencing, not mock a ready-made pass. The decisive assertions are:

```python
assert observed_argv == operation_argv(pair, origin=origin, brand=brand, impact=impact)
assert len(observed_argv) == 24
assert terminal_start_observation.own0 is True
assert terminal_start_files.files_clear is True
assert result["runtime_integrated"] is False
assert result["production_accepted"] is False
```

- [ ] Wire exact guard precheck/exclusive namespace, same original owner/RetainedFiles/OwnedRenderTools, staged assets/fixtures/graphs/ASS/outputs, original24 numeric operations and operation-specific oracles from Task106. Before every child and settlement rehash pair/publication/admission/acceptance via original RetainedFiles; handles closed before child consumer. Do not regenerate an origin from global runtime or reobserve user footage.
- [ ] Charge before injected process factory; failed spawn consumes attempt, block25th/concurrency/unplanned argv, enforce exact deadline/reservations/stdout/artifact/stderr/generation/control/EOF/exit association. Validate whole ramp/PCM/float/tag/mix/alpha/raster/Vulkan/full CPU+NVENC delivery/probe/AAC/natural-EOF oracles, not only file existence or list output.
- [ ] Keep driver finally retirement exactly:

```python
try:
    run_operations()
finally:
    files.retire(retain=keep)
    job.driver_done()
```

Here run_operations/keep/job/files are the campaign's original driver-local values, not a new owner. Controller services original cleanup until actual own0/files_clear; then publish_terminal uses detached bounded exclusive I/O with original reserved bytes/paths/deadline. Short/failed/overrun/cancel/uncertain final write cannot pass. Persisted terminal remains provisional until post-return check and later Root observation.
- [ ] Add focused fake failures for between-child pin drift, skipped/swapped argv, blocked25th/failed spawn/concurrency, remaining-deadline clipping, reservations/free-space/output growth/truncation, corrupt each oracle, no natural EOF, exit mismatch, reused generation/control, cleanup uncertainty and each terminal failure. Native process factory escape must always fail test. Record RED/GREEN at changed boundaries.
- [ ] Run complete fake file exactly once after final integration: approved python -B -m pytest clipperx/tests/test_production_runtime_qualification.py -q --basetemp=.superpowers/sdd/task-119-pytest-final. If actual further edits follow a failure, rerun only affected selection and truthfully record full-run result. Root packages both Task2 diff and final whole-unit diff for independent SPEC/QUALITY and broad source review. No actual campaign before publication+Root acceptance+separate admission. No broad app/global suite here.

## Controller checklist

- [x] Task116 accepted; historical incomplete verifier preserved/superseded. Task120 exact recovery mapping independently accepted; actual runtime acceptance remains absent.
- [x] Task117 reviewed compact source, fake-only evidence; two Important findings fixed with genuine RED/GREEN and finite independent approval. No campaign authority.
- [ ] Task119 reviewed integrated source and one complete fake-file run.
- [ ] Final whole-unit review/constraints resolved; actual publication remains separate.

Self-review: section1-4 Task116 covered by Task1; section5 and6B by Task2; section6A by Task1. No historical re-audit graph, new owner/file, cap change or actual acceptance invented. No unspecified new helper signature is delegated to a later task: Task117 report binds its exact immutable interface before Task119 dispatch.
