# Source-clock authority and observed decode — Phase3 integration amendment

Status: adopted technical design, preserving the approved automatic vertical-
shorts objective, original narrative ranges, style, model gate and quality targets.
This precedes full renderer assembly. No new model, layout, RenderProfile or UI
control is introduced. Controller read the complete executor proposal and actual
R/S addendum; this compact authority selects the implementation boundary below.

## Evidence and invariant

Current user VODs are at R:/Streaming/(2025-Adelante)/VOD's and
S:/Streaming/streams. Read-only metadata proof is retained in
.superpowers/sdd/full-vod-locations-20261015.md. Actual video starts differ from
container/audio origin by15ms and59ms. This is not evidence of an A/V defect.
Nominal/average FPS does not prove frame timing or CFR/VFR. Current OpenCV index
sampling returns pixels but labels them with requested time, without decoded PTS.
That historical v7 evidence must not be retrospectively declared clock-qualified.

Current source seconds mean file-relative presentation time when qualified:
`raw_time = format.start_time + source_time`. Use ONE common O, retaining each
stream's offset, not independent first-decoded-frame/sample rebasing. Preserve
all original declared source/output values; no narrative snapping, retiming,
silence repair, default frame padding or hidden timestamp healing.

Renderer fidelity and upstream evidence compatibility are separate. A technical
render can use the declared current convention with upstream_clock_status
unverified_legacy; it cannot thereby become automatically accepted. Automatic
semantic acceptance needs matching receipts for the evidence actually consumed.
Missing receipts do not trigger full-source reanalysis from render. Conflicting
or stale receipts fail explicitly. No new source-clock UI knob is needed.

## Unit6a0: pure clock records and bounded observed decoding

Own render_clock.py/source_decode.py, their two tests and the existing paired
render_runtime.py/test_render_runtime.py. Do not yet edit the analyzer, old cache,
ASR/audio mixing, visual compiler, executor, CLI, profiles or source media.

Add the smallest detached immutable records needed for:

- SourceTimeOrigin: source stat/fingerprint identity and its actual method,
  exact container origin/duration, stream indices/timebases/start/end bounds,
  audio format, relevant color metadata, binary/policy identity and canonical hash.
- SourceDecodeReceipt: requested bounded windows, decoder-only margins, actual
  frame/audio observations/outcomes and bytes/count/hash association, declared
  conversion policy, origin identity, limits and scope=window.
- Observed frame payload: immutable BGR24 bytes/dimensions plus actual integer
  PTS/rational timebase and established half-open presentation interval. An array
  returned to analysis must not mutate the retained pixel identity. Requested
  time, observed time and coverage are distinct, including unavailable outcomes.

Do not introduce a second rich SourceClockBinding: Task5 already owns its small
source/decoder/enabled-PCM binding. Later derive its decoder identity and exact
enabled PCM hash from these host facts. JSON receipts use deterministic finite
values/rational representations and strict identity validation, never pickle.
Current source_fingerprint hashes path/size/mtime, NOT full video content. Do not
read/hash an entire14GB VOD to construct that existing identity.

Use the explicitly supplied PAIRED pinned FFmpeg9.0.2 runtime for observed
decoding, not an availability-selected alternate decoder. The same decode must
provide pixels and their PTS: bounded raw BGR output with tagged showinfo and
ordered count/checksum association is the selected route. Validate one-to-one
association, including no muxer-driven frame duplication/drop. Never attach an
unrelated ffprobe frame list to OpenCV pixels or derive PTS from requested time,
nominal/average FPS, packet DTS or frame index. Observe before fps/setpts.

Batch nearby requests into bounded windows, no process-per-request/frame policy.
Stream raw payload and observation metadata with bounded buffering; do not
capture an entire decoded window's full-resolution RGB in memory. Associate
integer pts with the observed filter timebase, not rounded pts_time strings.
Retain selected payloads only. Missing/duplicate/nonmonotonic metadata, checksum/
count/byte disagreement, unsupported dynamic dimensions or exhausted limits fail
explicitly and cannot produce a qualified receipt.

Select the frame whose half-open presentation interval covers a requested time.
Use next observed PTS or independently established positive frame duration at
EOF. Missing predecessor coverage cannot be replaced by a later first frame.
Requests before video start yield unavailable, never a known visual sample at an
earlier time. Missing lookahead/unknown terminal duration remains unavailable
or fails explicitly; do not infer it from average FPS. Do not infer a black frame
or silence from an absence of decoded evidence.

Audio observations use the same O/bounded decoder mapping and record actual
input/output sample PTS, count, format/conversion and PCM identity. Support named
analysis16k-mono and render48k-mono/stereo preparation policies only as qualified;
the former never impersonates Task5's prepared bus. Original44.1/48k conversion
needs a real phase/count test, not an implicit source-rate assumption. No audio
normalization, gain, cue or limiter logic is duplicated here. No async healing
or source apad. Sample-grid/phase representation is recorded, not an invented
source-range change. Unsupported conversion combinations fail explicitly.

Keep finite host limits explicit and tested: decode-window/preroll/postroll,
frame/sample/pixel/retained-byte limits and process timeout. Initial proof windows
are2s; no decoder may silently expand into hours after a failed seek. Resource
limits are technical host policy, not new user/director controls. A bounded
failure is evidence of failure, not permission for an unbounded retry.

## Pinned CPU capability, not a different binary selector

Amend the current conflation between the pinned bundle identity and Vulkan use:
RenderRequirements may request the SAME pinned runtime with needs_vulkan=false.
Vulkan still requires the pinned bundle; system+Vulkan remains invalid. Combine
requirements by pinned-runtime identity independently of the union Vulkan flag.
Resolution/pin checks stay in render_runtime; no second selector or PATH change.
Pinned CPU resolution has device_index=None and no device arguments. CPU
qualification checks its actual required tools/filters and bounded CPU behavior;
it does not execute or claim a Vulkan probe. Native motion still needs/qualifies
Vulkan exactly as before, with no availability-based CPU/encoder fallback.

This additive capability lets analysis use the already-qualified executable
without unnecessary GPU initialization and keeps Python>=3.10 compatibility.
PyAV18 was considered locally but requiresPython>=3.11; no PyAV dependency or
Python-minimum change is adopted. No install/global environment/driver changes.

## Unit6a0 verification and handoff

TDD pure origin/raw mapping/offset bounds, unknown metadata, caller-container
isolation and receipt corruption; pinned CPU resolution/requirements without
Vulkan; tagged frame/audio sequence/byte/checksum failures and boundedness.
Then small HOST fixtures with known nonzero origin,15/59ms video offset, VFR
spacing, decoder reorder, coverage edges and both audio preparation policies.
Verify selected pixel identities against independent known-fixture frame
references and exact observed PTS, not a self-consistent log parser alone.
No original asset/font/media changes or neural/API/model inference.

Tiny CPU-encoded HOST input fixtures (<=8s,<=320x180) are allowed to establish
reorder/VFR/offset ground truth. These are not real-video transcoding, production
encoder qualification, delivered AAC synchronization or edited MP4 acceptance.

After synthetic controls pass, ONE bounded observation per actual source:

- R:/Streaming/(2025-Adelante)/VOD's/HOLA VOLVÍ.mp4,10800..10802s;
- S:/Streaming/streams/EMPEZAMOS UN JUEGO MUY INCREÍBLE The Last Of Us 1.mp4,
  10800..10802s.

Each child window has a finite timeout no longer than45s and explicit decoder
margins/frame/byte limits; no repeat loop or full decode. Reprobe identity before
use. Retain request/argv, observations/receipts and small diagnostic artifacts in
a new host proof directory. These prove scoped observation only, NOT waveform
A/V correlation, analyzer semantic correctness, full3h throughput or acceptance
of an old model plan. Original v7 timelines/frames/manifests remain untouched.

Run only the new clock/decode tests and covering changed-runtime pure tests,
scoped Ruff/mypy/compile; do NOT repeat Task3 GPU/minification/1080 controls,
Task4 font/logo renders, Task5 audio normalization suites or the full repository.
Independent spec+quality review closes6a0 before analyzer/renderer integration.

Post-review technical follow-up decision,2026-10-15: both ORIGINAL R/S attempts
failed checksum association and retained no original stderr; their real cause
remains unconfirmed. The synthetic CR defect and strict receipt/diagnostic fixes
received independent Spec Compliance/Task quality Approved at package33 SHA
A4458C664294208D86464D9C4B72F6A1901DBB942685FD81415ECF189B4AE00A. Root separately
activated .superpowers/sdd/phase3-task-6a0-post-review-observation.md: at most ONE
additional observation per same source/window under the SAME limits, NEW exclusive
evidence, no code changes or retry loop, STOP on the first new failure. This does
not rewrite original failed evidence or authorize full-VOD/model/edited-A/V work.

## Remaining sequential units (not implemented by6a0)

2026-10-15 prerequisite amendment:6a0-audio-seam precedes6a1, under the adopted
docs/superpowers/specs/2026-10-15-audio-only-observation-design.md and its focused
brief. It adds a shared audio-only route so ordinary10s feature windows neither
exceed combined8s policy nor decode the full source video. The separate audio-only
300s ceiling does NOT widen combined/video limits. Four existing paths only;
no analyzer or original-media work.6a0 code and separately scoped real R/S follow-up
are complete; original failures remain historical failures, not erased.

6a1 integrates observed decoding into new immutable analyzer generations and host
companion evidence association, preserving v7 bytes and legacy PCM/transcripts.
Its6a1-A exact adopted integration is
docs/superpowers/specs/2026-10-15-observed-visual-index-design.md with brief
.superpowers/sdd/phase3-task-6a1-A-brief.md; TEN paths and separate review35.
6a1-B then handles actual Director image/manifest association; no model/schema
changes are implied by A. The old focused proposal remains background only.
Its adopted six-path integration is now
docs/superpowers/specs/2026-10-15-director-image-provenance-design.md with exact
.superpowers/sdd/phase3-task-6a1-B-brief.md and ORIGINAL36. Extraction and consumers
have separate TDD steps but one review gate because old followup rewrites and
fixtures must migrate atomically with the new immutable bundle contract.
Qualified cache lookup requires matching current receipts; absence is unverified,
not a fabricated upgrade. Keep requested vs observed coverage and unknown states.
No identity/editorial threshold change or full eight-source rebuild is implied.

2026-10-15 adopted6a2 refinement after B review36 Approved:
docs/superpowers/specs/2026-10-15-global-raster-and-exact-overlay-design.md.
Execute independently reviewed6a2-A (FOUR mapper/overlay paths; exact Fraction
intervals must survive subtitle production), then6a2-B (FOUR graph/compiler
paths from A handoff). Exact6a2-A brief is
.superpowers/sdd/phase3-task-6a2-A-brief.md, ORIGINAL37; no source edit before
activation/snapshot. This refines internal timing, not editorial ranges/styles,
model gates or final acceptance; streaming/6b remain separately scoped.
After A review37 Approved, exact6a2-B continuation is adopted at
docs/superpowers/specs/2026-10-15-ordered-visual-assembly-design.md, FOUR graph/
compiler paths, activated phase3-task-6a2-B-brief.md and ORIGINAL38. It returns
finite local-transport shot templates after global paint, sparse source-slot
obligations, typed resources and COMPLETE-window actual-byte associations. No
whole-clip executor, source/host-preflight repair or streaming policy is hidden
in that unit. Raster mapping/native keys/final A/V/model gates remain unchanged.

After6a2-B review38 Approved, streaming prerequisite6a3 is adopted at
docs/superpowers/specs/2026-10-15-streamed-source-window-design.md, exact brief
.superpowers/sdd/phase3-task-6a3-brief.md and ORIGINAL39. FOUR clock/decoder paths
only: separate finalization kind, pull/one borrowed selection, shared selector/
numeric engine, independently owned deadlines/cleanup and explicit live versus
aggregate accounting. Old materialized/audio receipts and limits stay unchanged.
Pure/fake tests only; no new actual OS/media/model campaign. A separately scoped
TWO-path pure compiler adapter follows, then6b writer/executor qualification.
No encoded/output/throughput proof follows from the observer's accounted limits.

After39 independent approval,6a4 purestreambinding is adopted at
docs/superpowers/specs/2026-10-15-stream-binding-design.md with exact
.superpowers/sdd/phase3-task-6a4-brief.md and ORIGINAL40. TWO compiler paths,
captured actualimmutableassembly for sourceprefix/Arequest revalidation, real
39 constructors/loader/chainhelpers, one borrowedframe result and finalmetadata.
No source39change, secondmapper/owner, actualstream/encoder/IO or productiongate.
HOSTlinearuse/publication and6b qualification remain outside the purecursor.

After40 independent approval,6b1 native-call prerequisite is separately adopted
at docs/superpowers/specs/2026-10-15-windows-sync-io-adapter-design.md with exact
.superpowers/sdd/phase3-task-6b1-brief.md and ORIGINAL41 absence markers. TWOnew
adapter/test paths, fake-native ABI/errors/handle-transfer gate only, no source
observer change/actualWindows/media or supervisor/encoder implementation. The
later HOST remains sole lifecycle owner; actual cancellation needs its own gate.

After41 independent approval,6b2 float audio observation is separately adopted
at docs/superpowers/specs/2026-10-15-float-audio-observer-design.md, exact
.superpowers/sdd/phase3-task-6b2-brief.md and ORIGINAL42. FOUR existing clock/
decoder paths, NEW versioned packed-float operation/receipt/limits with shared
old engine and unchanged s16/combined/streamed meanings. Pure/fake gate only;
no prepared-map/normalization/Windows owner or actualmedia campaign. Current
origin decoder identity stays byte-identical, newoperation identity is separate.

After42 independent approval,6b3 pure float sample selection is separately adopted
at docs/superpowers/specs/2026-10-15-float-audio-selection-design.md, exact
.superpowers/sdd/phase3-task-6b3-brief.md and ORIGINAL43 absence markers. TWOnew
selection/test paths only; existing Task5 schedule/Qa and42 real receipt/PCM
validators reused. Compact group maps plus actual per-group byte selection,
no per-sample metadata or new decoder/normalizer/owner/file materializer. Pure
scope does not qualify native phase/fidelity/finalAAC+AV or Windows execution.

After43 independent approval,6b4 original-byte font staging is separately adopted
at docs/superpowers/specs/2026-10-15-original-font-stage-design.md, exact
.superpowers/sdd/phase3-task-6b4-brief.md and ORIGINAL44 absence markers. TWOnew
fontstage/test paths only, real synthetic temp-file IO and fixed approved-original
vs fixture authority; no real font bytes/load/render or host wiring. Human-approved
originals plus renderedvalidation/selectedfileidentityunverified remainbinding.

After44 independent approval,6b5 hosted source ownership is separately adopted
at docs/superpowers/specs/2026-10-15-hosted-source-ownership-design.md, exact
.superpowers/sdd/phase3-task-6b5-brief.md and ORIGINAL45. FOURpaths: newowner/test
plus existing source_decode/test, one shared opt-in HOST lifecycle for39stream
and42float, actual41adapter with fakeDLL tests. Original no-host defaults/receipt
meanings/limits remain. No actual Windows/source/transport/encoder campaign;
owner-start driver/jobwatchdog control is not finalrenderer or wholejobown0.

6a2 uses shared origin and global absolute Qv(t)=floor(60*t+1/2) allocations,
predecessor/EOF-qualified sampling, unretimed local native n0/frame120, independent
source-layer primitives, global source/text/PNG z-order and existing ASS event
projections. Forty .025s shots total60frames, not80. Preserve declared boundaries
and expose raster deltas; zero-frame positive shots fail. No duplicate geometry,
second mapper, second fps repair, audio/video longest-stream padding or new effect.

6b stages exact original font bytes before measurement and validates rendered
visual fidelity, leaving selected-file identity explicitly unverified under
the human-approved amendment
docs/superpowers/specs/2026-10-15-font-staging-and-visual-validation-design.md.
Family/provider messages never become byte-selection proof.6b prepares bounded audio with Task5,
executes ONE final video encode, isolates legacy imports and writes truthful
sidecars/metrics. Compiler-resolved geometry is not decoded-observed pixel data.
Original public render signatures stay unchanged. Explicit encoder, no fallback.

Their concrete briefs/ownership/snapshots are finalized from each preceding
handoff before dispatch. Original Task6 routing/deliverable requirements and
Task7 synthetic/three-real-smoke gates remain binding. Neither technical QA nor
HOST receipts are model acceptance/human approval. Both Luna trials remain
none_passed; no third model/API campaign under this checkpoint.
