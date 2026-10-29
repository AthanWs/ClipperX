# Pure48k sample selection — adopted Task6b3 design

ADOPTED technical design,2026-10-15; activated exact Task43 brief governs source work.
Task42 Approved; phase3-task-6b2-review.md final fix1 append closes I1.

## Authority and actual interfaces

Read COMPLETE controller notes/global constraints/adopted42 spec including
clarifications. Verified live hashes BEFORE source reads:

- render_clock.py: 272B7A7C34D4848F8F4C9F6A3BE594400BA8ACAF116EB95FE3694D713265B2CE
- source_decode.py: 020B8ECE0E00C617A32EC4E726E7CF0F1D51D4B6228392ECD503B28AED1A4F84

Focused anchors: render_clock.py:43-104,728-801,870-916,980-1060,1077-1106,
1258-1298 (clocks/guards/receipts/JSON); source_decode.py:116-138,777-827
(limits/wrapper/bytes); audio_mix.py:42-162,190-199 (schedule/Qa/binding);
output_timeline.py:20-50,128-145,172-245 (mapping). Recipes:
test_render_clock.py:1066-1094,test_audio_mix.py:21-37,
test_visual_contracts.py:156-182.

AudioFormat(sample_rate,layout) derives channels; AudioSchedule(source_fingerprint,
spans,cues) hashes repr(self). SourceClockBinding(source_fingerprint,decoder_identity,
selected_pcm_sha256) cannot bind maps/operations; do not create it for partial buses.

## Recommended boundary and ownership

Own new clipperx/audio_selection.py and clipperx/tests/test_audio_selection.py
only. No existing source/test changes. Responsibility: bounded occurrence plan,
receipt-based metadata map, and actual byte application for ONE group.

One-way imports: audio_selection -> audio_mix/output_timeline/render_clock/
source_decode/visual_contracts/existing strict-model boundary; no reverse import.
Use schedule_audio, Qa,42 wrapper/receipts and39 guards. Import stays inert;
no second scheduler/parser/decoder/resampler/normalizer or new owner API.

Recommend plan + per-group map + byte application; caller can release each
observation/result before the next. Final assets, complete-bus assembly/hash,
normalization, audio_mix tail and host publication/execution are outside this unit.

Alternatives: whole-bus apply simplifies calls but retains observations plus bus;
combined map/copy hides the independently reviewable map; per-sample rows create
millions of records. Prefer per-group application with <=2 constant-offset runs.

## Exact proposed API and records

All names below are NEW PROPOSED. Records are frozen/slotted/keyword-only;
tuple elements have exact declared types; int excludes bool; Fraction is plain
and bounded. Each public entry revalidates bypassed objects. No persisted loader.

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class AudioSelectionLimits:
    max_shots: int = 4096
    max_cues: int = 4096
    max_groups: int = 4096
    max_runs: int = 8192
    max_plan_bytes: int = 2_000_000
    max_metadata_bytes: int = 2_000_000
    max_observed_bytes: int = 72_000_000
    max_selected_samples: int = 8_592_000
    max_selected_bytes: int = 68_736_000

@dataclass(frozen=True, slots=True, kw_only=True)
class AudioSelectionSpan:
    shot_index: int
    shot_id: str
    schedule_span_index: int
    group_index: int
    source_start: Fraction
    source_end: Fraction
    start: Fraction
    end: Fraction
    first: int
    last: int

@dataclass(frozen=True, slots=True, kw_only=True)
class AudioSelectionGroup:
    index: int
    shot_first: int
    shot_last: int                 # exclusive
    schedule_first: int
    schedule_last: int             # exclusive
    source_start: Fraction
    source_end: Fraction
    start: Fraction
    end: Fraction
    first: int
    last: int

@dataclass(frozen=True, slots=True, kw_only=True)
class FloatAudioSelectionPlan:
    policy: str
    origin_id: str
    origin_decoder_id: str
    runtime_json: str
    schedule: AudioSchedule
    schedule_id: str
    audio_format: AudioFormat
    spans: tuple[AudioSelectionSpan, ...]
    groups: tuple[AudioSelectionGroup, ...]
    total_samples: int
    limits: AudioSelectionLimits

@dataclass(frozen=True, slots=True, kw_only=True)
class AudioSelectionRun:
    first: int                    # absolute output sample
    last: int                     # exclusive
    source_first: int             # frame relative to g
    residual: Fraction            # constant exact rho

@dataclass(frozen=True, slots=True, kw_only=True)
class FloatAudioSelectionMap:
    policy: str
    selection_sha256: str
    schedule_id: str
    origin_id: str
    origin_decoder_id: str
    runtime_sha256: str
    operation_id: str
    receipt_sha256: str
    observed_pcm_sha256: str
    group: AudioSelectionGroup
    sample_grid_origin: Fraction
    observed_samples: int
    observed_bytes: int
    channels: int
    runs: tuple[AudioSelectionRun, ...]
    reused_first_interval: bool
    selected_samples: int
    selected_bytes: int

@dataclass(frozen=True, slots=True, kw_only=True)
class SelectedFloatAudioGroup:
    mapping: FloatAudioSelectionMap
    map_sha256: str
    pcm: bytes
    pcm_sha256: str

def plan_float_audio_selection(plan: VisualCompositionPlan, *,
    origin: SourceTimeOrigin, runtime: Mapping[str, object],
    limits: AudioSelectionLimits) -> FloatAudioSelectionPlan: ...

def map_float_audio_group(plan: VisualCompositionPlan, group_index: int,
    receipt: SourceFloatAudioDecodeReceipt, *, origin: SourceTimeOrigin,
    runtime: Mapping[str, object], decode_limits: FloatAudioDecodeLimits,
    limits: AudioSelectionLimits) -> FloatAudioSelectionMap: ...

def apply_float_audio_group(plan: VisualCompositionPlan,
    mapping: FloatAudioSelectionMap, observed: DecodedFloatAudioWindow, *,
    origin: SourceTimeOrigin, runtime: Mapping[str, object],
    decode_limits: FloatAudioDecodeLimits,
    limits: AudioSelectionLimits) -> SelectedFloatAudioGroup: ...
```

Plan/map expose record()->dict[str, object] and sha256=digest(canonical(record())).
Fresh detached record keys exactly match fields; nested records likewise.
AudioFormat projects sample_rate/layout; schedule projects actual
source_fingerprint/spans/cues and their existing fields. Fractions use rational(),
finite float gains/duck use str(value), tuples become lists. No embedded self-hash.
schedule_id is actual schedule.identity, distinct from new hashes. Policy is
"float48k;half-open-interval;group-v1". No complete-bus PCM hash is returned.
runtime_json=canonical(expected runtime); runtime_sha256=digest(runtime_json).
selection_sha256=rebuilt_plan.sha256; map_sha256=mapping.sha256; PCM hashes digest
actual bytes. Receipt fields/digest come only from the fresh validated42 receipt.

Plan remains explicit at map/apply: rebuild the bounded plan with schedule_audio;
apply recomputes the receipt map and compares EVERY field before selected PCM
work. This trades repeated bounded metadata work for no stored full-plan copy or
trust in frozen object/hash provenance. Plan result can discover decode windows.

Use existing AudioPreparationError with new fixed codes SELECTION_INVALID,
SELECTION_LIMIT, SELECTION_ASSOCIATION, SELECTION_UNAVAILABLE, SELECTION_PCM;
normalize malformed-field failures, never fallback or swallow programmer errors.

## Mapping, group policy and exact residuals

After bounded preflight, strictly detach/revalidate plan as schedule_audio does,
then call schedule_audio. Use OutputTimeline.from_shots validation, not video60
quantization. Preserve each shot/order/exact Fraction(str(decimal)) endpoints/Qa
bounds. Associate each shot with one schedule span and group. Visual subdivisions
can merge in schedule; flag changes split it. Retain original rows and all cues.

Group maximal consecutive shots with EXACT source AND output adjacency,
regardless of normalize/gain/limiter/cues. Existing schedule spans cannot straddle
these groups. Keep exclusive shot/schedule ordinal ranges. Repeats/reversals are
separate ordered occurrences, never deduplicated by source/PCM. Original Qa
slices partition each group's [Qa(start),Qa(end)).

Require global first0, next.first==previous.last, positive shot/span/group counts,
final last==schedule.total_samples<=8,592,000, and authored end<=179s. Preserve
endpoints; no snapping/rebase/discards. OutputTimeline's <=1e-9 representation
tolerance remains: no blanket exact-duration restriction. Non-exact adjacency
separates groups; a discrepancy violating Qa partition/group requests rejects.

For one group [s,e)->[a,b), O=origin.origin, R=48000, actual observed g,
F=Qa(a), L=Qa(b), use the controller's adopted formula verbatim:

```text
x_n = max(a, n/R)
tau_n = O + s + x_n - a
j_n = floor(R * (tau_n - g))                      F <= n < L
rho_n = (g + j_n/R - O - s + a) - n/R
```

One42 receipt per group must request EXACT [s,e], channel-matched float policy,
covered status and trusted derived margins. No stitching restarted observations.
Require contiguous blocks/actual PCM, 0<=j_n<observed_samples, a<=x_n<b,
s<=tau_n-O<e, and g+j_n/R<=tau_n<g+(j_n+1)/R. Right EOF equality rejects; left
equality is legal. Reject unavailable even if selected indices exist. No guessed PTS.

Containment applies to maximal GROUP endpoints. No internal shot/flag re-clamp or
restart: a leading Qa processing slice can precede its internal exact boundary.
Keep its exact declaration and Qa slice separately. Later processing applies
gains/normalization/limiter/cues to the already selected continuous group bus.

Compact proof: D=R*(O+s-a-g), k=floor(D). For n>=F+1, n/R>a, so j_n=n+k.
j_F is F+k or F+k+1. First case: one [F,L) run, no duplicate. Otherwise emit
[F,F+1) plus nonempty [F+1,L); with >=2 outputs j_F==j_(F+1), exactly one legal
reuse. A singleton cannot reuse. Coalesce equal offsets, prohibit other skips/
repeats/runs and set reused_first_interval precisely. At most two runs/group.

Each run has j_n=source_first+n-first and constant rho=
g+(source_first-first)/R-O-s+a in (-1/R,1/(2R)]. Monotonicity proves interval/
source/output containment from both endpoints, including singleton. Record exact
residuals, not a drift verdict. No waveform/phase/AAC proof, padding, interpolation
or seek retry follows from this mathematical bound.

## Bounds and validation before work

Limits are defaults AND ceilings; only tighten positive plain ints.
179*48000=8,592,000 frames/channel, stereo selected bytes<=68,736,000.
Observed bytes<=min(new cap, trusted42 decoded/retained caps,4*C*trusted42 samples).
No42 increase:180s with margins,9M samples,72MB,16384 blocks/stage,65536/block,8MB
envelope. These independent ceilings do not extend179s.

Plan/map each have2MB COMPLETE projected metadata including schedule/cues,
associations/strings/scalar widths/hash placeholders. Input plan has2MB.
groups<=shots; total derived runs<=2*groups<=max_runs (no cross-call ledger).
Charge records/separators BEFORE append; no retained map collection.

Before dump/strict recursion: bounded traversal of known exact visual model
types/plain containers, depth<=10, <=65536 nodes and2MB ledger. Reject extras,
cycles/unknown objects/iterators; bound container length before traversal. Use
_stream_guard for strings; charge finite float str values and keys/container
overhead before conversion. _stream_int precedes str/Fraction. This protects,
not replaces, existing strict schema validation. Then detach/validate/schedule;
guard all projections before identities/hash.

Reuse39 _stream_int/_stream_fraction/_stream_rational/_stream_guard/_stream_keys,
_bounded_json, canonical/digest/rational. Guard operands BEFORE arithmetic and
retained results AFTER: ints/Fraction numerator/denominator<=4096 bits. Check
operand-bit upper bounds against16384-bit temporary ceiling before products/
sums/floor; work stepwise and reject retained results over4096. Counts/frame
widths obey smaller domain caps before multiplication. No approximation/parser clone.

Guard exact trusted decode_limits fields before record()/validate_float_audio_limits;
plain dict runtime before real float-runtime validation/hash. Use _float_origin
(real39/StreamClock guards) before origin identity, with tuple/known-slot shape
checks first. origin.fingerprint==schedule.source_fingerprint. Existing
source_decoder_identity stays distinct from42 float operation_id.
Plan requires all source endpoints inside origin.duration and the42 supported
audio source:44100/48000,1/2 channels,flt/fltp. AudioFormat is48000 with preserved
mono/stereo layout; group source lengths and selected counts obey caps independently.

Receipt: require exact type; bound ALL six stored strings BEFORE record/to_json
(which otherwise parse/hash unguarded JSON). _bounded_json parses both stored JSON
strings under trusted42 caps. Schema-check, reconstruct real
SourceFloatAudioDecodeReceipt(origin,payload,runtime=expected_runtime,
limits=trusted_decode_limits.record()), compare all six fields and use fresh
receipt/public sha256/record only. validate_origin alone uses self-payload limits,
so cannot enforce tighter caller limits. No invented payload/validate_pcm methods.

The exact records bind origin/legacy decoder/full runtime/float operation/receipt/
observed PCM/schedule/policy/occurrence/runs/counts. Plan hash includes original
shots, schedule/cues and limits; map hash includes residuals/reuse. This is
consistency, not authenticated execution. Stat-derived path/size/mtime fingerprint
is not full source-content authentication. HOST labels/rehashed receipts cannot
prove execution; future host rechecks files and owns linear observation/use.

Apply preflights exact observed.pcm bytes/length, receipt strings, map types/counts/
widths and all selected capacity BEFORE PCM hash/copy. Reject subclasses, mutable
bypassed fields, bool counts, malformed tuples/oversized metadata. Recompute map
from current inputs/fresh receipt; compare all fields. Reserve complete result/map
metadata with64-character hash placeholders and selected capacity before PCM validation.

Reconstruct DecodedFloatAudioWindow(fresh_receipt,observed.pcm) to invoke REAL
block SHA256/Adler32(seed0)/aggregate/final-offset/finite-word validators, regardless
of prior wrapper construction. Then copy whole interleaved frames with stride4*C
according to runs into exactly selected_bytes; legal reuse copies one frame twice.
Use bounded memoryviews/one bytearray then immutable bytes; release views on all
paths. Hash actual selected bytes; return no mutable alias. Test input detachment.

These cap payload/serialized metadata, not RSS. Caller owns observed PCM;
selection owns new PCM. Bytearray->bytes can double selected payload plus input/
metadata/Python overhead. One179s group can approach bus size. Caller retention,
hard cancellation and streaming42 ownership remain outside this seam.

## Complete tiny positive recipe (text only; do not execute now)

Future pure make_plan/make_shot recipe; never call file-writing test_audio_mix.plan,
host_plan or runtime fixtures. All validators stay real. No execution now.

```python
from fractions import Fraction as F
import zlib
from clipperx.render_clock import (StreamClock, SourceTimeOrigin,
    SourceFloatAudioDecodeReceipt, source_decoder_identity,
    float_audio_decode_argv, digest)
from clipperx.source_decode import FloatAudioDecodeLimits, DecodedFloatAudioWindow
from clipperx.tests.test_visual_contracts import make_plan, make_shot
from clipperx.visual_contracts import VisualCompositionPlan
# These imports exist only AFTER a separately authorized implementation:
from clipperx.audio_selection import (AudioSelectionLimits,
    plan_float_audio_selection, map_float_audio_group, apply_float_audio_group)

def tiny_case(channels):
    runtime = dict(runtime_id="HOST", ffmpeg="O:/HOST/ffmpeg.exe",
        ffprobe="O:/HOST/ffprobe.exe", ffmpeg_sha256="A"*64, ffprobe_sha256="B"*64)
    origin = SourceTimeOrigin("O:/HOST/float.nut", 16, 1, "0123456789abcdef",
        F(0), F(1), (
            StreamClock(0,"video",F(1,60),F(0),F(1),width=1,height=1),
            StreamClock(1,"audio",F(1,48000),F(0),F(1),sample_rate=48000,
                        channels=channels,sample_format="flt")),
        source_decoder_identity(runtime), {})
    dl = FloatAudioDecodeLimits(preroll=F(0),postroll=F(0),max_window=F(1),
        timeout=1,max_blocks=2,max_samples=8,max_block_samples=8,
        max_decoded_bytes=32,max_retained_bytes=32,max_log_bytes=4096,
        max_metadata_bytes=8192,io_chunk_bytes=16)
    pcm = (bytes.fromhex("00000000 0000803f 000080bf 00008035") if channels==1
        else bytes.fromhex(
            "00000000 00000080 0000803f 01000000 "
            "000080bf 0000c03f 00008035 000000b5"))
    policy = "render48k-f32-mono-v1" if channels==1 else "render48k-f32-stereo-v1"
    block = dict(n=0,pts=0,time_base="1/48000",rate=48000,channels=channels,
        samples=4,format="flt",checksum=f"{zlib.adler32(pcm,0):08X}",stage="cxain")
    window = ["0/1","1/12500"]             # 80 microseconds
    audio = dict(policy=policy,input=[block],output=[dict(block,stage="cxaout",
        bytes=len(pcm),sha256=digest(pcm))],samples=4,bytes=len(pcm),
        pcm_sha256=digest(pcm),sample_grid_origin="0/1",source_phase="0/1",
        coverage=[dict(window=window,status="covered")])
    payload = dict(windows=[window],limits=dict(dl.record(),decoder_start="0/1",
        decoder_end="1/12500"),audio=[audio],argv=[float_audio_decode_argv(
            runtime["ffmpeg"],origin,F(0),F(1,12500),policy)])
    receipt = SourceFloatAudioDecodeReceipt(origin,payload,runtime=runtime,
                                            limits=dl.record())
    backing = bytearray(pcm)
    observed = DecodedFloatAudioWindow(receipt,backing)
    backing[:] = bytes(len(backing))
    base = make_plan(source_fingerprint=origin.fingerprint, shots=[make_shot(
        source={"start":0.0,"end":0.00008},
        output={"start":0.0,"end":0.00008})]).model_dump()
    plan = VisualCompositionPlan.model_validate(base)
    limits = AudioSelectionLimits()
    selection = plan_float_audio_selection(plan,origin=origin,runtime=runtime,limits=limits)
    mapping = map_float_audio_group(plan,0,receipt,origin=origin,runtime=runtime,
                                    decode_limits=dl,limits=limits)
    selected = apply_float_audio_group(plan,mapping,observed,origin=origin,
                       runtime=runtime,decode_limits=dl,limits=limits)
    assert selection.total_samples == 4
    assert mapping.selected_samples == 4 and mapping.selected_bytes == 16*channels
    assert [(r.first,r.last,r.source_first,r.residual) for r in mapping.runs] == [(0,4,0,F(0))]
    assert not mapping.reused_first_interval
    assert selected.pcm == pcm == observed.pcm
    # Independent finite small oracle, not production mapper calling itself:
    for n in range(4):
        request = F(n,48000)
        assert F(0) <= request < F(1,12500)
        assert F(n,48000) <= request < F(n+1,48000)
        assert selected.pcm[n*4*channels:(n+1)*4*channels] == pcm[n*4*channels:(n+1)*4*channels]
    return selected

# Future tests: tiny_case(1), tiny_case(2). Neither invocation occurs here.
```

Four synthetic frames cover1/12000; request ends1/12500, respecting actual42
coverage/upper-edge allowance. Stereo includes negative0/subnormal/1.5. Input
checksum is synthetic metadata, not native proof.

## Smallest future verification manifest and TDD order

Own only the two new files. TDD after activation: failing real42 recipe/group/
fractional tests, minimum implementation, then hostile/budget cases. Record real
RED/GREEN; none executed now. Exact new test names/manifest:

1. test_selection_real42_mono_stereo_bytes: literal bytes/hashes, buffer detachment,
   real42 roundtrip and tighter trusted limits.
2. test_selection_groups_preserve_occurrences_and_processing_slices: visual/flag/
   cue splits preserve groups; repeats/discontinuities separate; original exact
   spans/cues/counts/schedule associations stay unchanged; no internal clamp.
3. test_selection_fractional_first_interval_runs: independent tiny Fraction/floor
   oracle for requests,j,bytes,containment,rho; no production helper as oracle.
   Include singleton/no-duplicate/reuse/negativeD/nonzeroO,g/half ties/late source.
   Example a=.000026,s=.000002,O=g=0: F=1,D=-1.152,k=-2,j_1=j_2=0;
   rho_1=.000024-1/48000,rho_2=.000024-2/48000. Use valid preceding occurrence
   and covered42 receipt. Same a,s=.000016 gives j_1=0,j_2=1: no duplicate.
4. test_selection_missing_coverage_and_eof: unavailable, absent predecessor,
   exact rightEOF, gaps/overlaps/missing bytes, group containment/Qa failures;
   retain tolerated declarations or reject violated invariants, never repair.
5. test_selection_association_and_hostile_records: wrong identity/policy/operation/
   receipt/schedule/map/order; forged/bypassed/subclass/mutated objects; real
   block/checksum/aggregate/offset/NaN/Inf failures despite rehashed metadata.
6. test_selection_limits_precede_payload_work: boundary/over caps, strict limits,
   huge ints/Fractions/exponent text/duplicate JSON/depth/cycles/extras/unknown
   objects. Tiny independent sentinels prove rejection BEFORE hash/copy/allocation;
   no near72MB fixtures and no swallowed sentinel exception counted as success.

FUTURE exact gate: new test file + full existing known-pure test_render_clock.py;
scoped ruff/mypy for new files. Scheduling regressions use new pure fixtures.
No test_audio_mix whole-file/not-host selector: its plan fixture writes host.bin
and its runtime fixtures/other nodes run host work. No source_decode/streamed39/
compiler/pipe suites. No historical test edits/permissive validators. Zero actual
children/threads/timers/models/media/native calls. Host sentinels only fail.

Later gates: native48k/44.1k phase/startup/EOF/waveform/event/count, normalization,
complete bus/finalAAC+AV (<=one60fps frame; AAC delay<.04s), Windows ownership/RSS.
Preserve179s, original source times, mixed-track semantics and parent gates.

## Self-review disposition

No authority conflict found. Root confirmed group-only containment and exact
per-shot Qa association. No hidden source/owner change. Adoption remains required.


## Controller adoption clarifications — governing Task43

Root FULL-read21999B proposal SHA B3C338236142A3234F7B4860B5FA0E79DAD57D078ACA9F9D9E26A8847FDB0447
and focused actual42 receipt/wrapper, OutputTimeline, visual-contract and pure
fixture interfaces. The original proposal remains immutable background. All
NEW API/record/policy/limits/mapping/byte requirements above are now binding
for this TWO-new-path unit despite retained proposal/future wording in that
historical body. This section supersedes its not-adopted disposition. No
strong owner/native/media/model or next-stage implementation is authorized.
User's autonomous-tool permission preserves vision; this is internal technical
selection, not a new editorial/style choice requiring manual work.

Use the exact THREE public functions/SEVEN frozen records above. Plan and map
record()/sha256 are the only required public projection helpers; nested scalar
projections can remain private. Do not invent record serializers/loaders for
every type or a generic schema framework. Structurally guard projections before
hashing, including bypassed models/records. Known strict-model traversal is a
small bound-before-dump guard, not a copy of Pydantic business validation.
Rebuild from the actual accepted VisualCompositionPlan at map/apply; the existing
schedule_audio remains the sole scheduling authority. Return original immutable
schedule with its existing identity, not a separately authored schedule model.

selection_sha256 is scoped to the AUDIO projection defined by fields above:
original shot IDs/order/source+output endpoints/Qa/schedule associations, actual
schedule/cues, origin/runtime/format/policy/limits. It is not the complete visual
plan hash and does not assert equivalent geometry/creative decisions. Preserve
all exact declaration rows; do not add a new whole-plan JSON field only to bind
irrelevant visual geometry. Future HOST separately retains the complete plan.

max_runs bounds the conservative planned run capacity: sum(1 for a singleton
sample group else2), checked before any map creation; actual maps still contain
only canonical1or2 runs. This is an explicit conservative tightening constraint,
not a claimed cross-call ledger or a reason to create empty/duplicate runs.
For a singleton with the clamped first request, always store ONE run and
reused_first_interval=False even if its offset differs from the hypothetical
next sample. Verify the actual first/last formula requests, not a nonexistent
successor. Tolerated source/output mismatch remains declared; never round it
away or impose blanket exact-duration equality. Internal shot/audio-flag slices
never clamp/restart group sampling.

The bounds-before-work contract means bounded input operands and prospective
intermediate bit-width checks BEFORE arithmetic, then real39 checks on retained
results. It does not claim no arithmetic occurs in validators. Guard plain
Fraction numerator/denominator and exact model/container types before strict
recursion, origin.identity, receipt.record/to_json/sha256 or payload hashing.
Trusted42 decode caps are separate from2MB selection metadata. Preflight all six
stored receipt strings, then reuse real42 constructor with caller's tighter
limits. Do not duplicate42 JSON/conversion/PCM validation. Apply must use fresh
real DecodedFloatAudioWindow so bypassed wrappers cannot skip actual byte checks.

Fixed new error codes are exactly the proposal's SELECTION_INVALID,
SELECTION_LIMIT, SELECTION_ASSOCIATION, SELECTION_UNAVAILABLE, SELECTION_PCM.
Catch only named expected malformed-input failures at the corresponding boundary;
never catch AssertionError or BaseException, and never let a swallowed sentinel
exception stand in for correct pre-work rejection. No optional fallback/output
repair or limits increase. No input-finiteness/native execution claim.

Pure verification scope is EXACT new six test names/full test_audio_selection.py
plus full known-pure test_render_clock.py; scoped TWO-path Ruff/mypy/compile.
Use make_plan/make_shot only as inert validated fixture builders or duplicate the
small literal recipe locally. No file-writing plan/host/runtime fixture, whole
Task5/source/compiler tests, Thread/Timer/Popen/native/OS/probe/font/image/model/
network/old guard. Enforce tiny data and fail-only external I/O sentinels. All
parent179s/mixed-track/finalAAC+AV/model/human gates remain unchanged.

