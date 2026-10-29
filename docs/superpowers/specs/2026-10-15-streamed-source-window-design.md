# Bounded streamed source windows — observer prerequisite

Status: ADOPTED technical continuation after Task6a2-B independent review38
Spec compliant / Task quality Approved. User's automatic vertical-shorts vision,
authored edits/style and original ranges are unchanged. Horizontal long edits,
chat ingestion, posting, model acceptance and editorial acceptance remain outside
this unit. No new UI/model control, toolchain, decoder or encoder selection.

Root read COMPLETE51217B streaming proposal (including exact§11), checked the
current clock/decoder interfaces and pure/fake test boundaries. Proposal SHA256
is D01A7F417BAE273B7AD639530BE6F2B68ECCC84E64EA73AA542405B864C147E9;
its prior23524B prefix was verified unchanged. This document governs adoption;
the proposal is background, not extra implementation authority. The exact unit
brief and ORIGINAL39 must exist before implementation.

## Scope and approach

ONE integrated observer/finalization gate owns FOUR existing paths:

- clipperx/render_clock.py and clipperx/tests/test_render_clock.py;
- clipperx/source_decode.py and clipperx/tests/test_source_decode.py.

No new source split/dependency or compiler/mapper/filtergraph/motion/overlay/
asset/audio-mix/runtime/Director/analyzer/schema/CLI/public-render source edit.
No actual media/probe/qualification/font/image/GPU/model/API/install campaign.
Tests use small supplied bytes and fake clock/child/I/O boundaries. No previously
guarded proof or full old decoder/media suite. A later TWO-path pure compiler
adapter and6b writer/executor/publication are separate gates, not implemented now.

Recommend a context-managed pull interface with ONE outstanding selection,
explicit acknowledgment and independent observer resource ownership. Reject an
arbitrary synchronous callback: killing the decoder does not interrupt caller
code or an encoder write. Do not add detached callback threads. Process/IPC
isolation is a larger alternative requiring a new scope and host qualification,
not a fallback. The pull contract bounds the observer's own resources;6b must
separately enforce its writer deadline and release discipline.

Reuse ONE covering-frame selection core and the existing pinned CPU FFmpeg pair,
raw BGR/PTS/checksum path, origin, margins and audio conversions. No FPS-derived
timing, retries, seek expansion, video-only path or positive EOF inference.

## Pure records and limits in render_clock.py

Keep dependency one-way source_decode→render_clock. The latter never imports
Window, VideoTag, ObservedFrame or source_decode. Define these new public seams:

```python
STREAM_DECODE_POLICY = 'bgr24-showinfo;pcm-s16-ashowinfo;copyts;no-healing;v1;selected-stream;v1'

# frozen/slotted, keyword-only, all fields listed below
StreamDecodeLimits(**fields)
limits.operation_record() -> dict[str, object]
limits.delivery_record() -> dict[str, object]
StreamDecodeLimits.from_records(operation: Mapping[str, object],
                               delivery: Mapping[str, object]) -> StreamDecodeLimits

SourceStreamFinalization(origin: SourceTimeOrigin, session_id: str,
    payload: Mapping[str, object], *,
    limits: StreamDecodeLimits = StreamDecodeLimits())
finalization.to_json() -> str
finalization.record() -> dict[str, object]
SourceStreamFinalization.from_json(text: str, *, origin: SourceTimeOrigin,
    policy: str, limits: StreamDecodeLimits,
    expected_session_id: str) -> SourceStreamFinalization

def video_decode_argv(executable: str, origin: SourceTimeOrigin,
                     start: Fraction, end: Fraction) -> list[str]: ...
```

StreamDecodeLimits has EXACTLY these fields. Defaults are ceilings except
max_live_selections, which is exactly1. HOST may only tighten. No max_retained_bytes
or max_windows alias and no streaming boolean that bypasses an old validator.

| Field | Type | Default / upper bound | Lower bound |
| --- | --- | --- | --- |
| preroll | Fraction | 1/2 | 0 |
| postroll | Fraction | 1/4 | 0 |
| max_window | Fraction | 8 | strictly positive |
| timeout | int seconds | 45 | 1 |
| max_frames | int | 1200 | 1 |
| max_samples | int | 480000 | 1 |
| max_pixels | int | 8847360 | 1 |
| max_decoded_bytes | int | 2000000000 per child | 1 |
| max_log_bytes | int | 2000000 per child | 1 |
| window_timeout | int seconds | 90 | 1 |
| cleanup_grace | int seconds | 5 | 1 |
| max_requests | int | 1200 | 1 |
| max_deliveries | int | 1200 | 1 |
| max_live_selections | int | exactly1 | exactly1 |
| io_chunk_bytes | int | 65536 | 1 |
| max_live_payload_bytes | int | 96000000 | 1 |
| max_selected_unique_bytes | int | 2000000000 | 1 |
| max_request_emission_bytes | int | 8000000000 | 1 |
| max_finalization_bytes | int | 8000000 | 1 |

operation_record contains exactly the first NINE fields; delivery_record the
remaining TEN. Fractions serialize through existing rational. from_records
requires both exact key sets and the real constructor. No coupled default
silently expands when another field is tightened. Requests also fit max_frames.
New margins only tighten their defaults; old DecodeLimits still allows its
existing1s margins unchanged.

Strict scalar contract: constructor integer fields are plain int, not bool or
subclasses; time fields are plain Fraction, detached/normalized. Requests are a
plain nonempty list/tuple of Fraction, nondecreasing, inside the original
half-open window and capped BEFORE copying. Duplicates remain distinct ordinals.
Canonical UUIDv4 strings use lowercase hyphenated RFC4122 form; IDs/hashes have
their exact lengths/formats. Raw PTS are signed plain integers, indexes/counts
nonnegative, dimensions/rates positive and checksums unsigned32-bit.

Serialized rationals must equal rational(exact(text)), reduced with positive
denominator. Cap token length before int/Fraction conversion; numeric components
have at most4096bits. Reject floats/NaN/Infinity, booleans except required True
completion flags, subclasses, duplicate/unknown keys, unexpected shapes/depth
and oversized metadata before parsing/cloning/canonicalization/hashing. Bound
all strings/paths/argv/rational tokens in chunked UTF8; reserve known ASCII-JSON
expansion before serialization and verify actual whole envelope bytes. Use narrow
known-schema guards, not a general validation framework or silent truncation.

Keep current metadata queue16, stderr tail65536bytes and line16384bytes as fixed
operation constants. Argv arrays have at most128 nonempty plain strings each.
Invalid inputs fail as ValueError-family errors, including safe handling of
integer/float representability, before child work. No I/O in pure constructors.

SourceStreamFinalization is frozen/slotted and stores only
origin_id,decoder_id,session_id,policy,payload_json. It exposes sha256, window
as a Fraction pair, requests as a Fraction tuple and limits as the validated
presented StreamDecodeLimits. record() returns a detached complete envelope.
Constructor derives origin/decoder IDs, fixes the new policy and revalidates
the actual known origin/scalars and all payload facts against HOST limits.
Loader bounds before parsing, rejects duplicate/invalid JSON, checks hash,
constant HOST-expected policy, current supplied origin and expected session,
then uses the same real constructor. No validate=False/fallback/promotion.

## Exact finalization schema

Envelope exact keys:

```text
version=1, kind='streamed_video_window', scope='window',
session_id, origin_id, decoder_id, policy=STREAM_DECODE_POLICY, payload, sha256
```

sha256 is digest(canonical(envelope without sha256)). Existing canonical/digest/
exact/rational functions remain the only serializers/numeric representation.
Payload exact keys:

```text
window, requests, operation_limits, delivery_limits, decoder_start, decoder_end,
video, audio, outcomes, argv, deliveries, accounting, completion
```

- window is a two-canonical-rational source interval0<=start<end within origin;
  requests are canonical source-time rationals inside it in order. Decoder bounds
  equal the existing origin-clipped margins and fit max_window. CLI microsecond
  serialization does not change declared coordinates.
- video is the bounded ordered OLD observation-row shape, exact keys
  n,pts,time_base,width,height,format,checksum,bytes,sha256,coverage_end. Dense n,
  bgr24, eight-uppercase-hex checksum, lowercase SHA256, positive canonical TB,
  bytes=3*width*height, common dimensions/TB, increasing raw PTS. Each nonterminal
  coverage_end equals next actual raw PTS; terminal end is None. Reuse the real
  origin/coverage rules, never infer final duration.
- audio is ONE object of OLD shape: policy,input,output,samples,bytes,pcm_sha256,
  sample_grid_origin,source_phase,coverage. Input rows exact keys
  n,pts,time_base,rate,channels,samples,format,checksum,stage; output rows add
  bytes,sha256. Stages cxain/cxaout, output s16. One coverage row has window,status;
  window equals this declared window and covered/unavailable follows the SAME
  audio count/format/conversion/phase/origin validator. No engine copy.
- outcomes are exactly requested,status,frame_n, one per request in order.
  Requested values equal requests; status covered only; frame_n identifies a
  NONterminal row that actually covers raw O+request. This new success subset
  requires all requests covered; old materialized unavailable outcomes remain.
- argv is exactly two arrays, video then audio. Both executable strings match;
  full canonical arguments equal shared pure producer builders using the supplied
  origin/bounds/policy. Loading never executes argv or proves executable bytes.
- deliveries are exactly index,request_ordinals,frame_n,successor_frame_n,
  chain_sha256. Dense indices; nonempty increasing ordinal groups concatenate
  exactly0..len(requests)-1. Selected frame_n increases per delivery, agrees with
  outcomes and is offered once; successor_frame_n=frame_n+1 exists. A single
  frame can serve several request ordinals without copying it.

Accounting exact keys, all plain nonnegative integers:

```text
decoded_video_frames, decoded_video_bytes, decoded_audio_output_bytes,
selected_unique_frames, selected_unique_bytes,
offered_deliveries, offered_payload_bytes,
acknowledged_deliveries, acknowledged_requests, acknowledged_request_bytes,
reserved_payload_capacity_bytes, peak_live_payload_bytes_accounted,
metadata_bytes, video_log_bytes, audio_log_bytes
```

Recompute all row/count/selected/delivery sums. Acknowledged deliveries equal
offered; acknowledged requests equal request count; acknowledged_request_bytes
sums selected frame bytes PER REQUEST including repeats. Offered bytes count
one payload per delivery. They are NOT measured encoder writes. Video and audio
decoded totals are separately limited, as in the existing operation, not their sum.

Let F=3*source_width*source_height; A=max_samples*output_channels*2. Reserve exactly
max(3*F,4*A) before allocating/starting children and require it fits max_live_payload_bytes.
Store that value as reserved_payload_capacity_bytes. Peak is the producer's
live-payload ledger, positive, at least max(F,audio.bytes), no larger than the
reservation. Loader checks arithmetic/bounds, not real lifetimes or RSS. Logs
are positive and no larger than their per-child cap. metadata_bytes equals UTF8
bytes of canonical(payload with accounting/completion removed); whole envelope
must also fit max_finalization_bytes. No self-referential byte count.

Completion exact keys:

```text
children=[
 {kind:'video',exit_code:0,started_ns,finished_ns,cleanup_complete:true},
 {kind:'audio',exit_code:0,started_ns,finished_ns,cleanup_complete:true}],
source_fingerprint_after, pins_verified_before_children:true,
all_requests_acknowledged:true, cleanup_complete:true, elapsed_ns
```

Nanoseconds are nonnegative relative offsets from session entry. Require
start<=finish, video finish<=audio start, finishes<=elapsed_ns. Root's explicit
deadline-boundary refinement is STRICT: each finish-start<timeout*1e9 and
elapsed_ns<window_timeout*1e9. At-or-after an absolute deadline is failure, never
successful sealing. This fixes equality/race ambiguity without an epsilon.
Fingerprint-after equals supplied origin.fingerprint; boolean flags must be actual
True. These are consistent producer facts, not proof a pure fixture ran a child.

## Canonical delivery chain, not a new raw-frame identity

H0=digest(canonical(header)), where header exact keys are:

```text
chain='selected-stream-v1', session_id, origin_id, decoder_id,
policy=STREAM_DECODE_POLICY, window, requests,
operation_limits, delivery_limits, decoder_start, decoder_end, audio_policy
```

Values use finalization's canonical representations; audio_policy is captured
before work. For delivery i, D is its first FOUR fields (not chain_sha256), V
the selected FULL CLOSED row, U the successor row with ONLY coverage_end removed:

```python
next_hash = digest(canonical({'previous': previous_hash, 'delivery': D,
                              'selected': V, 'successor': U}))
```

Validate/hash both row identities before yielding. Provisional successor end is
None; final data may close that end but changes no other successor field. V must
equal its final row including end; V.end equals successor PTS*TB. Loader recomputes
each link from final rows and all request/outcome/delivery associations. No
quadratic prefix copying, raw pixels in JSON or authenticity claim.

## Pull API, borrowed selection and lifecycle in source_decode.py

```python
@contextmanager
def stream_observed_window(runtime: RenderRuntime, origin: SourceTimeOrigin,
    window: Window, requests: Sequence[Fraction], *, audio_policy: str,
    limits: StreamDecodeLimits = StreamDecodeLimits()
) -> Iterator[ObservedWindowStream]: ...

class ObservedWindowStream:
    def next_selection(self) -> ProvisionalSelection | None: ...
    def acknowledge(self, delivery_index: int) -> None: ...
    def finish(self) -> SourceStreamFinalization: ...
    def close(self) -> None: ...

ProvisionalSelection(session_id: str, delivery_index: int,
    request_ordinals: Sequence[int], frame: ObservedFrame,
    selected_row: Mapping[str, object], successor_row: Mapping[str, object], *,
    previous_chain_sha256: str)
selection.selected_record() -> dict[str, object]
selection.successor_record() -> dict[str, object]
selection.delivery_record() -> dict[str, object]
```

Signatures denote exact interfaces, not stubs to add. Runtime entry accepts ONE
actual Window and creates one fresh HOST UUIDv4 per entered context. Caller cannot
reuse/supply session identity there; pure constructors/loaders accept explicit
canonical fixture IDs. No global registry or second SourceClockBinding.

ProvisionalSelection is frozen/slotted, storing session_id,delivery_index,
request_ordinals tuple,frame,selected_row_json,successor_row_json,
delivery_chain_sha256. Require real ObservedFrame/VideoTag with strict plain
integer/Fraction/immutable-bytes fields, actual length/SHA/seed0 Adler, full
tag/row correspondence, consecutive successor n, same dimensions/TB and positive
successor-established end. Canonical detached row JSON only; never serialize
frame bytes. Constructor checks association, not actual possession/checking of
successor pixels; the observer MUST check those bytes before offering its row.
Owner checks captured header/session/previous hash/ordinal continuity. There is
NO completed receipt_hash or qualified-window flag on provisional selections.

Session exposes read-only session_id,origin_id,decoder_id,policy,limits,window,
requests,deadline_ns,state,cleanup_required. Private owner alone holds lock,
process handles/PIDs, reader/timer, current lease, mutable ledger and cancellation.
No absolute clock values/handles are serialized. Single consumer, one active
child, no raw-selection prefetch queue. next while a lease is unacknowledged fails;
ack matches exactly that outstanding index once. It means consumer acceptance of
provisional bytes, NOT encoder consumption/rendering. No success artifact on break.

After all selections, drain/check the remaining bounded video and exit status;
do not stop the child merely because the last request was covered. next returns
None only after video is complete. finish requires None and all acknowledgments,
then the SAME incidental-audio path, actual PCM validation, current-source/pin
verification, deadline/resource/cleanup checks, and final sealing. No PCM retained
in finalization; no normalization/features/Task5-prepared-audio promotion.

Lifecycle OPEN→VIDEO_COMPLETE→FINALIZED, or unfinished state→FAILED/CLOSED.
Terminal data cannot resume/retry/reopen. close is idempotent after known complete
cleanup; failed cleanup is explicit, never successful own0 or a detached worker.
No direct mapper/K/shot/crop/slot knowledge in this observer. Later pure adapter
will use B's single request/frame helper and A's mapper separately.

## Shared selector and exact buffer ownership

Factor ONE internal incremental selector; old select_frames collects its events
and retains its original aggregate-selected checks and accepted request order,
including unsorted requests. New stream requires monotone requests, offers one
event and releases it before advancing. Both predecessor AND successor must
pass strict count/PTS/TB/dimensions/bytes/seed0 checksum/prefix limits BEFORE
coverage is offered. No last-frame success without a successor.

Use fixed F-byte raw read storage, bounded readinto memoryview slices, explicit
short-read loop, freeze once and release mutable storage/views before yielding.
Do not rely on growing-bytearray capacity or allocate a full second read chunk.
At read/freeze the observer owns predecessor F + fixed buffer F + frozen F=3F;
at a borrowed selected lease it owns selected predecessor and successor=2F.
Queue16 is metadata only. Caller-owned copies/pipe/OS/FFmpeg/GPU buffers are not
magically included. Cap known storage reservations before allocations.

Ack cannot erase suspended generator locals. On resume, drop yielded selected
frame/old-predecessor aliases and temporary containers FIRST, promote the checked
successor to sole predecessor, THEN call next raw input. Otherwise a fourth frame
could overlap the next freeze. Consumer must release selection/views before next
pull: retain delivery_index, consume under6b's deadline, delete references, ack,
then pull. On failure release/close instead. Python cannot enforce external
consumer retention;6b must budget its own storage. No RSS claim from this ledger.

Before audio, release all video payloads and complete video cleanup. Streaming
uses a fixed-capacity PCM accumulator, sharing the existing _audio_window
conversion/observation engine without changing its materialized/audio-only
defaults. Conservative audio payload reservation4*A covers known raw/freeze/
accumulator/final bytes. Verify aggregate and each ordered OUTPUT block's actual
SHA/seed0 Adler through memoryview slices, exact final offset; never hash missing
input bodies. Discard PCM before finalization. Metadata retains audio identities
and honest covered/unavailable status, not a prepared audio bus.

## Deadlines and cleanup ownership

One absolute monotonic window deadline starts at context entry. Child deadline
is min(child_start+timeout,window_deadline); no pause/extension during leases.
Check at-or-after expiry before new work and sealing. A private lifetime owner
OUTSIDE suspended generators independently marks failure, signals waits and kills
its own active FFmpeg on expiry; it never invokes caller code. It owns bounded
reap/join/handle cleanup, coordinated idempotently with generator-finally/close;
never generator.close concurrently with executing next. Watchdog cannot join itself.

Cleanup grace is ONE absolute budget across the cleanup attempt, not5s anew for
each wait/handle. Subsequent close does not restart grace. Incomplete cleanup
raises ValueError-family structured failure with cleanup_required and exact
still-owned PID/thread/handle evidence; retain ownership for host handling, emit
no finalization and never report own0 while survivors exist. No global orphan
registry, blanket exception swallowing or success after a failed cleanup.

Synchronous source/pin/filesystem calls count toward time; this does not make
arbitrary OS reads/caller writes interruptible. Guarantee is independent
termination/cleanup initiation for OWN children plus no late progress/sealing,
subject to real-host qualification.6b must use a separately cancellable/supervised
writer with deadline no later than the exposed deadline, release leases, close
in finally and qualify stalled-write cleanup. This API cannot interrupt arbitrary
consumer Python/OS code. Failed host qualification is explicit, not a fallback.

## Compatibility and verification boundary

Old observe_windows/observe_audio_window/select_frames signatures, valid output
bytes/status/order, canonical operation lists, SourceDecodeReceipt and
SourceAudioDecodeReceipt formats/hashes and aggregate retained inequalities stay
unchanged. New finalization is incompatible with both old loaders, and its loader
rejects old envelopes. No v1 promotion/rewrite or materialized64MB reinterpretation.
New stream aggregates may exceed old64MB under their own caps; live payload is
separately capped96MB. They are accounting guarantees, not general performance.

Narrowly factor common real video/origin/outcome/audio numeric validators. New
data may use a transient common VIEW with windows=[window],audio=[audio] and
operation fields; do not forge an old receipt with raised retention or add a
boolean to skip old validation. Each explicit wrapper enforces its own schema
and resource semantics. Add pure video_decode_argv next to existing command
builders and make OLD/NEW producer use its identical current spelling; no new
command engine or decoder/backend selection.

The video's exact existing suffix after decoder_input_argv is:

```python
['-map', f'0:{video.index}', '-an', '-sn', '-dn',
 '-vf', f'trim=end={decoder_seconds(origin.to_raw(end))},format=bgr24,showinfo@cxv',
 '-fps_mode', 'passthrough', '-enc_time_base', 'filter', '-threads', '1',
 '-c:v', 'rawvideo', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
```

Select the existing selected video, not an inferred frame rate or new backend.
Pure constructors/loaders compare metadata and operation consistency; they do
not possess raw payloads. Coherently fabricated/rehashed metadata is not
authenticated observation. Actual BGR/PCM integrity belongs to the real producer
and provisional-frame constructor while bytes are held, before final sealing.

Use exact allowlisted pure/fake test nodes and FOUR scoped Ruff/mypy/compile
after a recorded pre-edit baseline. Exclude old real Python-pipe, HOST media,
decoder/font/native/Director/A/fullrepo/guarded campaigns. Test actual shared
selector/validator, independent tiny BGR/PCM, bytes/counters/lease ordering,
late failure, fake watchdog and cleanup ownership. Lower caps instead of large
allocations. Pure synthetic completion is not actual observation. New actual
pinned-runtime/OS streaming control requires a separate bounded gate.

Official Python3.10 io/subprocess docs fetched via Context7 support unbuffered
short reads and readinto on supplied storage, not zero-allocation/Windows cleanup.
Actual3.10 compatibility, OS cancellation, downstream queues/topology/count/color,
exact font/asset identity, encoded A/V, full-VOD performance, AI/editorial/human
acceptance remain later gates. Existing ALL-plan HDR rejection is unchanged.
