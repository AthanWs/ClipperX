# Global raster allocation and exact overlay intervals — Task6a2

Status: ADOPTED technical refinement after6a1-B independent review36 Approved,
I1/I2/R1 closed. Final B package234984B SHA256
9108F4C1C2163D38037578557F6C94B1799A4361DE99BF3B0AE9454A8D60FD4A.
Root read the full read-only6a2 proposal plus addenda10–12, finalized the smaller
sequential boundary and self-reviewed exact APIs/scope/compatibility/counterexamples.
This adopts6a2-A FOUR mapper/overlay paths first; source edits require its exact
activated brief and ORIGINAL37 snapshot.6a2-B needs A's approved handoff and its
own exact brief/snapshot. No6a2 source/test execution preceded this adoption.
The automatic vertical-shorts objective, editorial source ranges, styles, model
gate and final A/V acceptance targets remain unchanged.

## One time mapper and explicit sampling rule

Use the existing OutputTimeline only. Interpret each finite validated plan/ASR
float operand once as Fraction(str(value)); do not rationalize an already-added
float and claim its original operands were retained. Fractions remain exact over
the validated values, not over JSON/ASR information lost before validation.
Existing1e-9 supported duration/continuity tolerance remains, without snapping.

For each declared occurrence `[o0,o1)` ↔ `[s0,s1)`:

```text
Qv(t)=floor(60*t+1/2)
K0=Qv(o0); K1=Qv(o1); N=K1-K0
for K in[K0,K1):
    n=K-K0
    representative=max(K/60,o0)
    requested_source=s0+representative-o0
```

The maximum is the DEFINED representative of the allocated leading raster cell,
not a decoder seek clamp or a change of declared endpoints. Record the two rule
names declared_start_for_leading_raster_cell and
inverse_declared_mapper_at_global_pts. There is no user/model rule selector.
Require exact s0<=request<s1, including representation-tolerance cases; fail
rather than repair an unsupported request. Repeated/out-of-order source ranges
retain occurrence order and unique shot IDs. Native crop always uses n, not K,
source PTS or request-s0.2seconds keeps boundary120 and rendered local0..119.

Positive N=0 shots fail with identity/original boundaries. Require integer raster
continuity between adjacent occurrences. Expose exact start/end/duration deltas
between raster and declared spans; do not sum independently rounded durations.
Forty adjacent exact.025s occurrences total60frames (2,1 repeated), not80.
Source audio retains Task5's original ranges and independent absolute sample grid;
do not shift/pad it to compensate for raster phase.

The alternative s0+n/60 is rejected: output start.025, source start2.007 and a
persistent simultaneous flash/click at source offset1/600 produce7/300s (1.4
frames) onset delay. The selected global representative chooses the known flash
at globalK2, delay1/150s. Represented persistent transitions have less than one
sampling-frame lag; the leading cell may show start content at most half a frame
early. This is not a promise to preserve every subframe flash or proof of encoded
A/V. Named independent marker/AAC/A-V acceptance gates remain mandatory.

## 6a2-A exact mapper/overlay producer seam

Own FOUR existing files only: output_timeline.py/test_output_timeline.py and
overlay_ass.py/test_overlay_ass.py. No graph, clock/decoder, runtime/assets,
audio/Director, model schema, public-render or legacy v1 edit. Fresh filesystem
snapshot/review gate before6a2-B. Exact brief will define focused TDD commands.

Add these frozen/slotted pure records on output_timeline.py:

```text
RasterSpan:
  occurrence_index:int, shot_id:str
  source_start/end, output_start/end:Fraction
  k_start/k_end:int
  properties frame_count,start_delta,end_delta,duration_delta
VideoSampleRequest:
  global_k:int, occurrence_index:int, shot_id:str, local_n:int
  requested_source:Fraction
  request_rule:the two controlled names above
```

Extend existing OutputTimeline with:

```text
raster_spans()->tuple[RasterSpan,...]
video_request(global_k:int)->VideoSampleRequest
raster_window(start:Fraction,end:Fraction,*,shot_id:str)->tuple[int,int]
overlay_output_interval(shot_id:str,local_start:float,local_end:float)
    ->tuple[Fraction,Fraction]
```

Reject bools as indices, unknown occurrences, nonfinite/negative coordinates,
empty/reversed/out-of-span intervals, out-of-output K and positive intervals
collapsing to zero raster cells. raster_window uses GLOBAL exact boundaries and
checks containment in that occurrence's declared and raster spans. The overlay
method adds original exact output-start and local operands once; never compute
Qv(output_start)+Qv(local). All constructors detach caller containers, no I/O.
Keep render_clock.exact's rejection of binary floats unchanged; convert plan
operands explicitly at this one mapping boundary.

Add optional immutable exact_interval:tuple[Fraction,Fraction]|None, defaultNone,
to existing MappedWord, MappedEvent and OverlayEvent. Old direct canonical-only
constructors remain possible; the new exact assembly will reject missing exact
subtitle intervals, never recover them from compatibility floats. Mapper-produced
words/events always have exact intervals and float start/end projections of those
same values. Use ONE rational clipping/affine calculation inside existing remap,
not a second transcript traversal or competing remapper. Retain public float
mapping APIs, text/word ordering, punctuation and occurrence semantics.

Existing caption production carries exact intervals through word packing, max
ends, active-boundary splitting and active-membership tests. It must not discard
precision into a float set/list between these steps. Subtitle OverlayEvent's
compatibility start/end are projections of its exact interval. Measured text,
font selection/layout/geometry/style, collision rules and ASS escaping remain
the existing mechanisms, not reimplemented temporal companions.
The existing distinct-caption collision predicate must compare exact intervals
when present, with its existing float behavior retained for canonical-only old
constructors. Otherwise an exact sub-ULP overlap can disappear at the float edge.
This preserves the SAME half-open collision rule, not a second collision system.

For titles, preserve the current compatibility float expression/fields and
canonical serializer; additionally obtain exact_interval from the shared
overlay_output_interval and carry it. Brand compatibility objects stay unchanged;
6a2-B derives exact execution timing from original plan operands through that
same method. No new field in model-controlled contracts.

Rational subtitle arithmetic may change last float bits, including a canonical
ASS centisecond tie. This is an explicit correction for newly generated v2 data,
not preservation of erroneous float arithmetic. Original declared inputs and all
historical artifacts remain untouched; v1 subtitle mapping is not modified.

Required independent edge cases include both:

```text
.15+.075=.225 ->Qv14, not float-added.22499999999999998 ->13
.15+(.075-1e-18)=9/40-1/10^18 ->Qv13,
even if its public float projection is.225 ->Qv14
```

Also cover clipped words, active gaps/overlap membership, repeated/out-of-order
occurrences, missing versus known-silent transcript, immutable nested intervals,
all Qv tie residues, the candidateL counterexample and N1 negative-phase shots.
Pure/font-measurement/file unit checks are not decoded-libass/DirectWrite or real
font-file qualification. Do not implicitly rerun that existing decoded campaign.

## 6a2-B independent primitives/assembly boundary

Exact adopted follow-on after A review37 Approved:
docs/superpowers/specs/2026-10-15-ordered-visual-assembly-design.md,
.superpowers/sdd/phase3-task-6a2-B-brief.md and frozen ORIGINAL38. Sparse source
transport skips all-SOLID only; output K remains unchanged. That exact design
refines PNG execution to strict interior E brackets via the same hook and returns
finite local_transport shot streams after global painting. No host topology,
runtime success or source/render gate is implied by that pure continuation.

After A's approved handoff, own filtergraph.py/test_filtergraph.py and
visual_compiler.py/test_visual_compiler.py under a separate exact brief/snapshot.
Reuse current geometry, crop-track expressions, preservation, fills and native
policy. Source/camera split the SAME scheduled observed pixels, with independent
spatial primitives; no original-file reseek. The old compile_shot_layers temporal
trim/fps/final-fps wrapper remains explicitly component/legacy-clock only and is
not called by new assembly. No new FPS repair or audio concat/padding.

Use global ascending paint order(z_index,tie_slot,declaration_index): source/
camera/solid0,title1,brand2,subtitle3. Subtitle keeps the existing max+1 convention
because it has no z field. Derive exact raster event windows through the mapper.
Execution ASS uses C(K)=floor(100*K/60)/100 through SAME serialize_overlay_events;
canonical declared events/document remain separately available. This produces
exact active index sets Kstart<=j<Kend; no second serializer/escaping or text in
shell syntax. PNG uses the existing pure brand hook with the exact B design's
derived global execution brackets E; original declared/raster windows stay intact.

Pure assembly structurally joins freshly prepared OverlayCompilation to the
resolved plan and outputs finite per-shot fragments, compiler-assigned ASS keys/
documents/AssApply steps and brand resource labels/required asset IDs. It performs
no asset/font reads/measurement and returns no misleading runnable ASS graph
with unresolved filenames.6b binds trusted host resources/paths explicitly.
Standalone OverlayCompilation cannot prove transcript completeness/known silence
or registry/font identity;6b must produce it freshly from its same selected
inputs, not infer those facts from serialization.

Actual completed-batch binding validates host policy/current supplied origin,
receipt rows/outcomes and exact immutable BGR bytes against each mapper request.
Its metadata alone may be decoded_observed; schedule/geometry stays
compiler_resolved. Unknown terminal frame duration still fails. No new receipt,
streaming route or EOF policy is adopted in6a2.

## General execution remains a separate gate

The existing materialized64MB ceiling counts ALL selected frame bytes+PCM, not
peak memory. Do not adopt the proposal's64batch/128child bridge as general clip
support: it cannot safely cover ordinary60s1080p/4K clips. A separate reviewed
streaming prerequisite must share the observer/selector, preserve old receipt
semantics and distinguish provisional delivery/sealed finalization, aggregate
bytes/peak-live buffers and late failure. Its exact numeric policy remains outside
this design's source execution authority.

The finite one-inlet split/shot-trim/video-concat graph is execution-unqualified;
it can queue future branches. Retain reusable per-shot fragments;6b must choose
and qualify a bounded topology, queues/GPU/pipe/cancellation/color interpretation
and one final encode. A documented queue-abort option is not bounded success.
Sequential raw spatial transforms would be a different expressly reviewed host
topology, not an automatic fallback. No performance/RSS/full-VOD/font/encoded-A-V/
model-semantic/human acceptance claim follows from pure6a2 tests.

Existing global constraints and source-clock authority remain binding. No
installation/download, original rewrite, API/model campaign, Git repair/init/
worktree/commit/push or new user-visible control. Root adopts each exact brief
only after its predecessor's independent gate, then continues the existing plan.
