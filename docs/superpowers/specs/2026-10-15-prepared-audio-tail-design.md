# Prepared-audio graph tail — adopted design

2026-10-15. **PROPOSAL NOT ADOPTED.** Only this document; no implementation,
Task45 review, tests/native/media/Python/delegation. Brainstorming/writing-plans
guidance stays within the brief.

## Decision and evidence

Extract ONE private normalized-restoration/gain-through-final-count tail in
`clipperx/audio_mix.py`; preserve original-PTS compile_audio; add an explicit
pure prepared-input entry. No new materializer, runner, transport or broad receipt.
Alternatives: unchanged compile_audio double-trims selected PCM; copied effects
duplicate semantics; rewritten source times change schedule identity.

Complete reads: tail brief;59143B executor proposal; controller notes;Task5
brief/report;global constraints;actual authority
`docs/superpowers/specs/2026-10-14-audio-compilation-design.md`.
Suggested2026-07-26 path absent. Older proposals are not implemented APIs.

Vertical1080x1920, mixed track,179s remain. No voice isolation, new knobs/cues,
300s/1GiB budget. Task42/43 float observation/interval selection stay upstream.

Text anchors in clipperx/audio_mix.py unless specified:

- 42-60: `quantize_time(seconds: float | Fraction) -> int`,
  `_seconds(Fraction) -> str`, `_number_text(float) -> str`; absolute rational
  Qa at48000 and precise authored decimals.
- 64-81: `AudioFormat(sample_rate: int,layout: str)`; prepared48k mono/stereo,
  .channels1/2, not original-media qualification.
- 85-105: `AudioSpan(source_start,source_end,start,end: Fraction,first,last: int,
  normalize: bool,gain_db: float,limiter: bool)`; `ScheduledCue(asset_id: str,
  start,end: Fraction,first,last: int,gain_db,duck_db: float)`.
- 109-162: `AudioSchedule(source_fingerprint: str,spans: Sequence[AudioSpan],
  cues: Sequence[ScheduledCue])` detaches tuples; total_samples,enabled_samples,
  identity properties. Identity=sha256(repr(self).encode()).
  `schedule_audio(plan: VisualCompositionPlan) -> AudioSchedule` revalidates,
  calls OutputTimeline; preserves spans,Qa,zero-sample rejection and coalescing.
- 165-187: duck_db(schedule,time) and _duck_expression(schedule): cue smoothstep/
  current-min law. 191-208: `SourceClockBinding(source_fingerprint: str,
  decoder_identity: str,selected_pcm_sha256: str)`; hash means COMPLETE ENABLED
  selected PCM. `LoudnessStats(input_i,input_lra,input_tp,input_thresh,
  target_offset: float)`.
- 244-274: _runtime_identity(runtime) returns9 strings.
  `NormalizationReceipt(schedule_identity: str,binding: SourceClockBinding,
  runtime_identity: tuple[str,...],audio_format: AudioFormat,samples: int,
  stats: LoudnessStats,mode: Mode,mode_reason: str,
  scope: str="mixed_track_normalization")`.
  _validate_receipt(schedule,receipt,binding,fmt,runtime) -> None checks matching
  source/schedule/binding/runtime/format/count and recomputed mode/reason.
- 277-290: normalization_filter(receipt) -> str; measured loudnorm +48k restore.
  verify_normalization_observation(receipt,observed_mode,samples): mode/count.
- 319-337,349-370: HOST `measure_normalization(schedule,pcm: bytes,*,binding,
  audio_format,runtime) -> NormalizationReceipt`; `process_normalized_pcm(
  schedule,pcm: bytes,*,receipt,binding,audio_format,runtime) -> NormalizedPCM`.
  Never call from pure compilation. _run_pcm304-316 uses subprocess.run.
- 341-346: `NormalizedPCM(pcm: bytes,observed_mode: str,samples: int,
  upstream_observation: LoudnessStats,output_loudness: tuple[float,float,float])`
  lacks request/receipt/runtime association.
- 374-381: `CompiledAudio(schedule: AudioSchedule,filter_complex: str,
  output_label: str,cue_inputs: tuple[tuple[str,Path,str],...],
  requirements: RenderRequirements,receipt: NormalizationReceipt | None,
  normalization_scope: str="mixed_track_normalization")`.

Legacy signature384-389 stays unchanged:

```python
compile_audio(plan: VisualCompositionPlan, *, audio_format: AudioFormat,
    assets: RenderAssetRegistry | None = None,
    cue_formats: Mapping[str, AudioFormat] | None = None,
    binding: SourceClockBinding | None = None,
    runtime: RenderRuntime | None = None,
    receipt: NormalizationReceipt | None = None,
    input_label: str = "0:a", cue_input_start: int = 1) -> CompiledAudio
```

Extraction anchors: validation397-405; split/concat409-418; original extraction
420-426; selected concat/slicing428-435; enabled concat/loudnorm436-439;
restoration440-447; gain449-452; cues453-475; limiter476-489; final490-492.

Dependency risk: audio_selection.py15-18 imports audio_mix; never reverse it.
FloatAudioSelectionPlan81-104,FloatAudioSelectionMap116-146 and
SelectedFloatAudioGroup150-154 (mapping,map_sha256,pcm,pcm_sha256) bind selection
and ONE group's bytes, not whole normalized material. Keep them in HOST provenance.

render_runtime.py36-50 canonicalizes RenderRequirements; combine_render_requirements
71-75 preserves native/Vulkan. visual_compiler.py1391-1408 compile_plan_audio
resolves visual plan,checks native runtime/device,calls legacy,combines. Keep it;
future HOST uses same checks/helpers. render_assets.py27-48 resolves ID to Path;
loadability68 is HOST-only.

## New pure public entry

No new record:

```python
def compile_prepared_audio(
    plan: VisualCompositionPlan, *,
    schedule_identity: str,
    audio_format: AudioFormat,
    selected_input_label: str | None = None,
    normalized_input_label: str | None = None,
    assets: RenderAssetRegistry | None = None,
    cue_formats: Mapping[str, AudioFormat] | None = None,
    binding: SourceClockBinding | None = None,
    runtime: RenderRuntime | None = None,
    receipt: NormalizationReceipt | None = None,
    cue_input_start: int = 1,
) -> CompiledAudio:
    ...
```

Required lowercase64-hex schedule_identity is existing schedule.identity attached
to HOST request, compared with fresh schedule_audio(plan), including disabled
authority. It detects stale declarations, not authenticated provenance. Existing
receipt remains measurement-only; no PreparedAudio/ProcessedBus API assumed.

New labels: canonical external N:a only,N0..32767,plain str<=7; distinct buses/
cues, no internal-label collisions. cue_input_start plain int1..32767,last used
cue<=32767. Nonzero allocation allowed; legacy grammar unchanged. No JSON routing.

HOST inputs: complete finite packed f32le48k matching mono/stereo audio_format,
one frame per channel/index, sample0 at PTS0. No source conversion/inference.
Let T=total_samples,E=enabled_samples,D=T-E:

| Case | Required graph inputs | Example with registered cue |
| --- | --- | --- |
| E=T | normalized exactlyE frames; selected MUST be None | normalized0:a; cues from1; no selected input |
| 0<E<T | selected exactlyT frames + normalized exactlyE frames | selected0:a,normalized1:a; cues from2 |
| E=0 | selected exactlyT frames; normalized MUST be None | selected0:a; cues from1; no normalization artifact |

Mixed selected is full output-ordered selected material, not compact disabled
material. Split ONLY for disabled spans, which trim absolute span.first/last;
no unused enabled branches. All-enabled must not open/consume redundant selected
input. Normalized bus contains all enabled portions in order; restore offsets
accumulate integer enabled lengths. Cue-free plans allocate no cue streams.

E>0 requires receipt,binding,runtime,normalized label; bounded checks precede
existing _validate_receipt. E=0 requires receipt=None,binding=None,normalized=None;
optional runtime supplies declared context only. Selected label iff D>0.
Retain measurement_required/identity_mismatch/mode_mismatch as applicable;
malformed labels/shapes/limits use descriptive ValueError. No silent fallback.

## One shared private seam

```python
def _audio_split(graph: list[str], label: str, count: int,
                 prefix: str) -> list[str]: ...
def _audio_concat(graph: list[str], labels: list[str], output: str) -> None: ...
def _append_audio_tail(
    graph: list[str], filters: set[str], schedule: AudioSchedule, *,
    bypass_labels: tuple[str | None, ...],
    normalized_label: str | None,
    audio_format: AudioFormat,
    assets: RenderAssetRegistry | None,
    cue_formats: Mapping[str, AudioFormat] | None,
    cue_input_start: int,
) -> tuple[tuple[str, Path, str], ...]: ...
```

Move409-418 unchanged, including count1 anull. Tail appends440-447 and449-490,
returns cue_inputs; caller creates CompiledAudio. Preconditions: one bypass slot
per span; disabled slot is complete zero-based local label, enabled slot None;
normalized label present iff E>0.

Legacy preserves397-439/order/labels; loudnorm requirement remains in prefix;
pass raw{i} only for disabled slots plus normalized if enabled. Require exact
legacy graph/result parity, including omitted AudioMixPlan true/true defaults
(`visual_contracts.py:365-369`), not test fixture false/false defaults.
Do not apply prepared constraints to legacy/shared-tail callers.

Prepared prefix splits selected by DISABLED SPAN COUNT, emits
`atrim=start_sample=span.first:end_sample=span.last,asetpts=N/48000/TB[raw{i}]`.
All-enabled emits no selected prefix. Shared restoration splits normalized by
enabled-span count and trims cumulative enabled offsets. No normbus,
normalization_filter, aresample, original-time atrim/PTS-source subtraction,
source apad or async healing. One tail, not copied effects blocks.

Preserve gain exactly once AFTER restoration; exact decimal formatting;
cue-only val(ch)/c=same smoothstep duck, attack=min5ms,d/4,release=min10ms,d/4,
strongest CURRENT negative dB; registered cue splitting/gain/padding/sample delay;
amix duration=first,weights1,normalize0,dropout_transition0; maximal limiter runs
independent of normalization/gain/visual splits. Exact constant remains:
`alimiter=limit=0.944:level=0:latency=1:level_in=1:level_out=1:attack=5:release=50:asc=0`.
Final trim exactlyT and PTS=N/48000/TB; no final-mix normalization.

Legacy requirements unchanged. Prepared conservative base:
atrim,asetpts,asplit,concat,volume,anull; add actual cue/limiter filters.
No aformat/aresample/loudnorm source-leg requirement; format/layout are HOST
preconditions. Output_label stays audio_out; HOST allocates it once and checks
global collisions. If tail materializes verified final-mix PCM first, final
encoder maps that input and must not execute these effects again.

## New-entry bounds

Reuse `filtergraph.py:26-29,32-143` _guard_known_input/_utf8_size:
8,000,000 generated bytes,32768 metadata entries,4096-bit numeric ceiling.
Task43 AudioSelectionLimits39-48 already has4096 shots/cues,2,000,000 plan bytes
and8,592,000 selected frames. Match these new-entry caps locally; avoid the
audio_selection import cycle. No PCM/RSS/owner budgets. Validation order:

1. Exact VisualCompositionPlan; plain list/tuple shots0<len<=4096; exact known
   shot/audio types and bounded cue lists/tuples totaling<=4096 before children.
   _guard_known_input(plan,text_limit=2_000_000,entry_limit=32768) rejects
   unknown shapes/cycles BEFORE schedule_audio strict recursion. Current179s
   schema and absolute Qa remain authoritative.
2. Plain label/index/hash types and lengths before regex/conversion; label<=7,
   canonical `(?:0|[1-9][0-9]{0,4}):a`, index<=32767; lowercase hash64.
   Exact AudioFormat/int48000/str mono|stereo before channels arithmetic.
3. Exact receipt/binding/stats/runtime records; bounded fields before equality,
   repr or _validate_receipt. Source/decoder strings nonempty, enabled hash64;
   runtime_identity tuple exactly9 plain strings; sample count plain
   int1..8,592,000; stats finite plain int/float excluding bool, existing ranges;
   mode linear/dynamic and scope mixed_track_normalization. Runtime Path/scalars
   checked against existing annotations, never stat/hash files. Use known-record
   guard with explicit record_types and aggregate2,000,000 text budget.
4. cue_formats only plain dict/MappingProxyType<=4096, safe asset IDs and exact
   AudioFormat. Exact RenderAssetRegistry; inspect bounded sfx dict/proxy with
   safe IDs/Path values. No arbitrary Mapping callbacks before shape/size checks.
   Detach needed entries after bounds. Registered path/matching format required;
   actual loadability is HOST.
5. Fresh schedule_audio, T<=8,592,000; compare schedule_identity, E/D input
   presence, receipt, cue collisions. No supplied schedule traversal or duplicate
   selector; declared seconds retained separately from sample indices.
6. Before splitting/emission reserve
   `65536 + 32768*S + 16384*C + 4*B <= 8_000_000`;
   S=scheduled spans,C=cue occurrences,B=bounded aggregate input text bytes.
   Conservative coefficients cover decimal expansions/repeated cue expressions;
   implementation must verify them against exact fragment templates.
   Prepared-only bounded list.append charges UTF8 fragments/separators before
   retention; check total before join. Same helper gets this list; legacy gets
   ordinary list. Never build an unbounded graph then check final size.

Reserve rejects excessive NEW requests explicitly, without truncation or changing
legacy acceptance. No general graph/cache/schema.

## Pure versus HOST boundary

Pure checks establish bounded declarations, schedule/receipt consistency, mode,
label usage, counts/formats/paths and deterministic graph. They cannot authenticate
N:a bytes. Self-consistent fabricated receipts pass; new digest/flag cannot fix
that. No new receipt/materializer API is needed.

BEFORE execution HOST associates actual Task42 observations/Task43 group maps
and bytes with immutable request, verifies coverage/order/counts/hashes,
assembles all and only enabled frames once, preserves enabled-input binding
hash meaning, runs existing pass one/pass two once on same bytes/runtime, binds
NormalizedPCM to that request/receipt, verifies actual mode/count/4*C*E length/
finiteness/normalized hash, and binds input labels to those verified opened/
consumed buses and registered cue bytes. All-enabled still needs upstream
selection evidence but no redundant selected input.

Helpers check input/runtime hashes and observed mode/count, not later file binding.
Owner/runner/request-result-file association remains separate. Publication needs
actual finalT PCM count/hash/layout/EOF, delivered A/V/AAC/loudness/peak gates and
retirement. Strings prove no fidelity/phase/latency/true-peak/AAC-delay/sync.
Targets stay upstream I=-14,LRA11,TP=-1.5; .944 is not final true-peak compliance.

## Pure TDD (not run)

Future modify audio_mix.py; create clipperx/tests/test_prepared_audio_tail.py.
Use literal plan,Path-only fake runtime,finite synthetic receipt. No old fixtures
that write host.bin or resolve FFmpeg.

Before initial RED guard Popen/run,os.system/spawn before lazy production import;
deny runtime resolution/qualification, source observers, normalization _run_pcm,
real worker/native entrypoints and asset/media/font loadability/open/stat on
fixture paths. Permit only interpreter/pytest code reads for import, not arbitrary
media IO. First node verifies fail-closed sentinels; no actual child/native load.

At most12 new nodes, finite tiny cases:

1. `test_tail_guard_blocks_host_work`: run/Popen/resolver/worker/asset sentinels.
2. `test_legacy_default_graph_parity`: omitted AudioMixPlan true/true,1s receipt;
   reviewed literal old graph/result/requirements,one loudnorm/limiter.
3. `test_legacy_nondefault_graph_parity`: old literal3-span late mixed cue/gain/
   limiter graph/result; neither parity test uses same-helper expected values.
4. `test_prepared_offset_has_no_second_trim_or_loudnorm`: source10..11/20..21,
   output2s; sample0..96000; no original trim/subtraction,aresample,loudnorm.
5. `test_prepared_bus_usage`:3 cases; account every input/output,no dangling labels.
6. `test_prepared_restore_offsets_and_gain_once`: true/false/true1s; normalized
   0..48000/48000..96000,bypass48000..96000; precise gains once AFTER restore.
7. `test_prepared_binding_mismatch_is_pure`:7 wrong declarations: schedule,
   source,enabled hash versus receipt,format,runtime,count,mode; fail before graph.
8. `test_prepared_input_presence_and_collision`: missing/redundant buses,shared/
   cue label,bool/negative/excess index,noncanonical/injected/oversized label.
9. `test_prepared_cue_duck_and_limiter_tail_parity`:2 overlapping cues/repeated
   asset; exact gain/suffix (canonicalize external normalized label only),numeric
   duck outside/plateau,maximal limiter runs.
10. `test_prepared_requirements_counts_and_layout`: mono/stereo;T/output/receipt/
    cues,no loudnorm/aresample,combine retains native visual need.
11. `test_prepared_absolute_clock_and_subdivision`: fractional/visual-split/
    repeated-source cases; schedule invariance where applicable,absolute Qa.
12. `test_prepared_guards_precede_traversal_and_generation`: tiny type/hash/cycle/
    stats/mapping cases; low patched caps test boundaries; sentinels prove order.

Optional safe existing node ONLY:
`clipperx/tests/test_audio_mix.py::test_boolean_rate_and_nonfinite_stats_are_rejected`.
No runtime/IO fixtures. No whole-file/broad -k gate: runtime126-130 and tests167-342
include real FFmpeg; other pure-looking plan tests also write host.bin.

After Root adopts implementation brief:
1. Scoped Ruff/mypy/compile baseline before production edits; actual imported debt,
   not historical seven-error assumption. No ignores/config/unrelated repairs.
2. Guarded RED for missing seam/behavior; legacy parity already passes.
3. Mechanical extraction then bounded adapter; no runner/normalization edits.
4. New pure file + optional safe node; scoped Ruff/mypy/compile on2 files;
   actual results once, no real/native/media/full compiler campaign.
5. Root independent review; pure completion is not HOST acceptance.

## Open risks and disposition

No adopted-spec contradiction found. Reserve needs template-bound review;
global labels/actual bytes remain HOST duties. No new product choice or policy
relaxation. Only proposal changed.

**PROPOSAL NOT ADOPTED. FINAL_IDLE own0.**

## Controller reconciliation (supersedes two proposals above)

Original17979B prefix SHA256:
30FE8398A201C9D8938FA1E20917EFA241017B400A541589DB1CAF0732B72441.
Still NOT ADOPTED; audio policy/legacy acceptance unchanged. No Task44/45 reads.

### Mapping bounds

Do NOT trust len(mappingproxy): its backing dict subclass can lie. For each
new-entry cue_formats/sfx mapping accept exact dict or exact MappingProxyType
only. Exact dict uses dict.items(mapping); proxy calls mapping.items() ONCE.
Before iteration require type(view) is type({}.items()), then len(view)<=4096.
This exact builtin dict_items size reads its actual backing dict entry count,
not an overridden mapping length. Reject lists/generators/iterator-like views
before traversal. Do not use tuple/list/dict(view) as the initial validation.

Iterate this checked builtin view with a consumed counter, at most size+1 next
attempts: before retaining each item reject consumed>=initial_size or4096.
Require exact two-item tuple, safe plain-str asset ID, expected value shape;
reject duplicate IDs. Check consumed==initial_size and len(view)==initial_size
after traversal; mutation/iteration errors fail. Charge bounded scalar/path text
BEFORE inserting into a fresh builtin dict. All subsequent lookup/traversal uses
that snapshot, never the original mapping's get/resolve/items again. For sfx,
pass an internal RenderAssetRegistry with only validated sfx snapshot and empty
brand/font maps to the shared tail; no original registry callback remains.
Validate formats by existing fields and paths as exact concrete Path platform
types, not arbitrary subclasses. No IO or new audio-to-font import/cycle.

This bounds consumption, NOT Python callbacks or concurrent mutation. Proxy items
may invoke its backing subclass callback; trusted in-process HOST construction
is assumed. It cannot authenticate that view or sandbox callback execution.
HOST must not mutate during compilation. No execution proof follows.

### Replace the arbitrary reserve with checked fragment construction

Withdraw unproved65536+32768*S+16384*C+4*B. Retain8,000,000 UTF8 graph-byte
ceiling, input/collection caps and legacy acceptance.
Source anchors: _seconds50-55, _number_text58-60, _duck_expression177-187,
split/concat409-418, prepared/restoration/effect templates432-490.

Use a tiny audio-local fragment accumulator in place of graph:list[str] in the
proposed private signatures. It exposes append_parts(parts: tuple[str,...])->None
and finish()->str. Legacy uses limit=None; prepared uses8,000,000. This is one
shared template implementation, not a second effect builder. append_parts checks
bounded part count/types and sums UTF8 sizes plus one separator if nonfirst,
against remaining capacity BEFORE joining or retaining the fragment. finish
joins only already-charged fragments with semicolons. Every interpolated value
must first be a bounded token; no caller constructs a large f-string before
append_parts. Legacy keeps exactly the same characters/order/default behavior.

Before formatting: input label<=7, internal ordinal/count<=4096
(mix inputs<=4097), sample indices<=8592000. Enumerated prefixes/fixed literals
are constant. Prepared numeric serialization requires exact Fraction numerator/
denominator bits<=4096 before calling unchanged _seconds. Derived d/4 and gains'
Fraction(str(value)) receive the same check; arithmetic itself is on previously
bounded finite operands. Because 2^4096<10^1234, nonzero ratio exponent lies
between-1234 and1233. Decimal precision400 at50-55 can add at most400 significant
digits (carry at most one exponent position); fixed-format output including
sign/leading zeros is below2048 characters. Zero is trivial. Thus one numeric
token is bounded BEFORE format; check result<=2048 as an invariant. No extra
rounding or formatter change. Validated finite plan-derived values fit this
bound. Any new-entry violation fails, never silently changes a value.

Split/concat: first validate count and every bounded label, calculate exact bytes
as sum(len(label)+2) for bracketed labels plus fixed header/filter/output tokens.
Check remaining graph capacity before creating wrapped-label parts or joining.
At most4096 span/limiter/cue branches; lists have bounded counts, and actual
formatted label bytes have been precharged. count1 still emits anull. Gain,
trim,delay,limiter and final fragments use constant-count bounded token tuples;
append_parts charges before construction. Cue amix likewise validates/counts
its<=4096 cue labels plus ducked before building joined inputs.

Ducking needs a preconstruction check, not merely bounded append afterwards.
Refine existing _duck_expression with optional keyword max_bytes:int|None=None;
prepared passes remaining fragment capacity minus [gained]/[ducked] bytes and
separator, legacy passes None. Share the SAME numeric law/literal templates.
For each cue, first render bounded a,b,attack,release,duck-db tokens. Compute
len(x) by summing the literal/token parts in181-182 without joining. At183 x
occurs THREE times: exact value length is 3*len(x)+len(db)+the literal-parts sum.
Before constructing x/value, require projected COMPLETE duck bytes to fit:
outer-wrapper bytes +1 for initial '0' + sum(6+len(value)) over retained/current
cues. Six is len('min(')+len(',')+len(')'). No arbitrary expansion coefficient.
Then retain value and discard x/temporary parts.

Do not repeatedly copy the nested reduction at185-186. Produce identical text
once from bounded flat parts: C copies of 'min(', then '0', then for each value
in original order ',',value,')', enclosed by the exact187 wrapper. This is
min(min(0,v0),v1) for two cues, preserving legacy bytes. List length is
4*C+3 including wrapper parts, bounded by16400; value sum and final joined duck
expression were charged before allocation. There is no unbounded intermediate:
one bounded x/value transient, retained values<=remaining graph budget, joined
expression<=that budget, final graph<=8M. Copies are explicitly bounded, not
an RSS/runtime guarantee. Bound fragment count by existing32768 metadata ceiling
for prepared only; reject BEFORE retaining an additional fragment.

Node12 adds lying-length proxy,generator-view rejection,view mutation/count
mismatch,low-cap split/concat/duck exact-fit/one-over with prejoin sentinels.
Node9 checks exact duck parity. No extra nodes, execution or source changes.

**Reconciliation remains PROPOSAL NOT ADOPTED. FINAL_IDLE own0.**

## Controller adoption — Task6b6 /46

Root FULL-read original17979B plus6466B reconciliation (24445B proposal
6C35D3B0038451DFF7702FBE1C2C804E75F88D9AF4666882E9D7FDCECE81C39F),
verified originalprefix and inspected exact current format/duck/compile/guard/
registry/receipt/runtime interfaces. This derived document ADOPTS the pure
prepared-tail contract with the reconciliation and clarifications here.
Retained PROPOSAL/NOTADOPTED text is historical background, superseded HERE.
Only exact46 activation may start source edits; no native gate is adopted.

Task45 independently source-approved, all46approvedpaths exact. Native harness
contract can continue docs-only in parallel; scheduling the independent pure
audio seam first does not substitute for native Windows qualification. ONE code
writer remains. Product/style/cue/source/normalization/publication policy unchanged.

Only TWO code/test paths: existing clipperx/audio_mix.py and NEW
clipperx/tests/test_prepared_audio_tail.py. No new receipt or runner/materializer/
source/clock/schema/CLI/dependency/native/renderer/backend changes.

The exact compile_prepared_audio signature above is binding, CompiledAudio stays
unchanged. Its receipt is measurement-consistency metadata, NOT proof normalized
bytes or final publication. Required schedule_identity is a bounded lowercase
64hex declaration compared with actual schedule_audio(plan); caller binding
selected_pcm_sha256 still means the ENABLED input bus, not the full selected bus.
No actual bytes are read by pure compilation. HOST owns byte/result/input binding.

Shared private graph accumulator is named _AudioGraph, with append_parts and
finish behavior in reconciliation; _audio_split/_audio_concat/_append_audio_tail
use it instead of the historical proposed list signature. _duck_expression gains
ONLY optional max_bytes=None; exact previous expressions remain byte-identical.
No copied restoration/effect engine. Legacy prefix/requirements/signature/defaults
and existing accepted ordinary labels remain unchanged.

ALL new record/type/token/index/count/byte constraints apply ONLY to prepared
entry/its non-None budget. Legacy None-budget helper path must NOT inherit
prepared4096/7character/32767/8M/32768 limits or require schedule_identity.
Prepared labels canonicalN:a N0..32767, distinct bus/cue labels; cue_input_start
plain1..32767 with used indices checked before generation; output audio_out.
SFX ID grammar exactly existing ^[a-z][a-z0-9_]{0,63}$, plain str before regex.
Only registered explicitly named cues, no default impact or asset selection.
Original caller registry is not mutated; bounded validated sfx snapshot only.

Input guard2,000,000 means AGGREGATE KNOWN INPUT TEXT bytes (plan/records/tables)
plus32768 known metadata entries/4096numeric bits, not a claim every model_dump
or complete JSON object/RSS is2M. Task43's separate full-plan JSON limit remains
unchanged and HOST still verifies its actual selected plan/receipt. Do not claim
a fullJSON/RSS bound from the text guard. Counts<=4096shots/totalcues,
T<=8592000frames/ch are new-entry bounds only; product179s remains exact.

Use existing filtergraph _guard_known_input/_utf8_size; actual dependency chain
already audio_mix -> render_assets -> filtergraph, and filtergraph/output_timeline
do not import audio_mix/audio_selection. Do not reverse-import audio_selection.
Explicit concrete Path types, builtin dict/items-view snapshots and consumed
counter precede registry construction; do not trust proxy len or lookup again
through original mapping. Trusted proxy.items callback is NOT a Python sandbox.
Existing runtime annotations/finite known metadata are checked without probes,
qualification/filehash/loadability. Existing _validate_receipt and mode selector
remain real; no self-consistent value becomes authenticated evidence.

Graph8,000,000UTF8byte and32768fragment bounds charge separators/fragments before
retention and joins. Reconciliation's2048numeric-token derivation and exact split/
concat/duck preconstruction accounting replace the withdrawn magic reserve.
No build-full-expression-then-check. No numeric rounding/duck law changes;
flat nested min rendering must match all legacy literal expected bytes.

Exactly12 new pure test node names in proposal, finite parametrization, fail-only
host/IO/native/worker boundaries before RED; no native test on import. Newtest
fixtures build literal plans/Path-only runtime/finite real receipts without
old plan(tmp_path) helper that writes host.bin. Root checked old safe node
test_boolean_rate_and_nonfinite_stats_are_rejected: lines91-98 pure/no fixture;
it is the ONLY existing audio_mix node allowed as a separate regression gate.

No unresolved binding contradiction/manual human choice. Native execution,
normalizer runner extension, selected/enabled/normalized file custody, finalmix/
AAC/A-V/fonts/metrics/CLI/model/human tests remain separate. This pure entry
does not produce an MP4. Activation requires ORIGINAL46 and exact implementation
brief; no Git init/repair/worktree/commit/push (invalid repository).

