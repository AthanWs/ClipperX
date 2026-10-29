# Pure compiled-source stream binding — Task6a4

Status: ADOPTED after observer review39 and assembly review38 Approved. This is
ONE pure TWO-path integration gate, not a writer, observer, encoder or6b host.
User's automatic vertical-shorts/style/ranges vision remains unchanged. The
human-approved original-font/visual-validation amendment remains separate.

Root read COMPLETE27750B proposal and approved39/38 reports/reviews, then checked
actual B helper, A timeline/value records and approved39 helper signatures.
Proposal SHA989D5F1A952F4B3DD3305550628509EA7D29F0FF4C9433127EC6B20349BF3692.
This design governs over its proposals. Root chooses a captured ACTUAL immutable
CompiledVisualAssembly, rather than timeline-only context, to recheck all source
consumer prefix sums as well as A requests. Share immutable metadata; do not copy
graphs/pixels or pretend a coherent substituted assembly can be authenticated.

## Scope and exact APIs

Only clipperx/visual_compiler.py and clipperx/tests/test_visual_compiler.py.
No upstream clock/decoder/A/filtergraph/overlay/assets/audio/runtime/Director/
schema/renderer/CLI source/test changes, new file split or dependency. No I/O,
source stat/probe, runtime selection/qualification, process/thread/handle, font/
image/media/GPU/model/API/network/install operation. No new mapping/selector,
raw-frame identity, SourceClockBinding, registry or publication behavior.

```python
def begin_source_stream_binding(assembly: CompiledVisualAssembly, *,
    slot_start: int, slot_end: int, origin: SourceTimeOrigin, session_id: str,
    window: tuple[Fraction, Fraction],
    expected_audio_policy: Literal['analysis16k-mono','render48k-mono','render48k-stereo'],
    limits: StreamDecodeLimits = StreamDecodeLimits()) -> SourceStreamBindingCursor: ...

def bind_source_stream_selection(cursor: SourceStreamBindingCursor, *,
    selection: ProvisionalSelection
) -> tuple[SourceStreamBindingCursor, ProvisionalSourceDelivery]: ...

def finalize_source_stream_binding(cursor: SourceStreamBindingCursor, *,
    finalization: SourceStreamFinalization) -> FinalizedSourceBinding: ...
```

Invalid input raises the existing RenderPreflightError/ValueError family before
expensive work. Python>=3.10. These signatures define interfaces, not stubs.
Functions never invoke a stream's next/ack/finish/close, callback or writer.

## Frozen/slotted immutable records

```text
SourceStreamBindingContext:
  assembly:CompiledVisualAssembly, origin:SourceTimeOrigin
  source_size:(int,int), expected_source_fingerprint:str
  occurrence:RasterSpan, transport:SourceTransportSpan
  slot_start:int, slot_end:int, session_id:str, policy:str
  expected_audio_policy:str, window:(Fraction,Fraction)
  requests:tuple[VideoSampleRequest,...], limits:StreamDecodeLimits
  chain_header_json:str

SourceStreamBindingCursor:
  context:SourceStreamBindingContext
  next_delivery_index:int, next_request_ordinal:int
  last_selected_frame_n:int|None, chain_sha256:str
  selected_unique_bytes:int, request_emission_bytes:int

ProvisionalScheduledSlot:
  transport_slot:int, request:VideoSampleRequest, requested_raw:Fraction

ProvisionalSourceDelivery:
  session_id:str, delivery_index:int, delivery_chain_sha256:str
  frame_n:int, frame:ObservedFrame, slots:tuple[ProvisionalScheduledSlot,...]

FinalizedScheduledSlot:
  transport_slot:int, request:VideoSampleRequest, requested_raw:Fraction, frame_n:int

FinalizedSourceBinding:
  slot_start:int, slot_end:int, origin_id:str, decoder_id:str, session_id:str
  policy:str, finalization_sha256:str, final_chain_sha256:str
  slots:tuple[FinalizedScheduledSlot,...]
  selected_unique_bytes:int, request_emission_bytes:int
  producer_audio_policy:str, producer_pcm_sha256:str, producer_pcm_bytes:int
  producer_audio_coverage:str
```

No caller-mutable list/dict/plan in context/cursor/results. Actual compiler
assemblies contain immutable tuples/records/paths/JSON; require that known shape
before sharing their reference. Reject mutable malformed containers rather than
retaining them or a generic deepcopy framework. Existing frozen records may be
shared after checks. Revalidate on every public transition; frozen Python objects
aren't unforgeable capabilities. A coherent whole-context replacement cannot be
authenticated by these pure functions; HOST retains actual begin context.

Only provisional delivery borrows ONE actual frame; all its slots share that
reference. Context/cursor/final result retain NO frames/BGR/PCM, even through
hidden containers. No receipt_hash, empty/predicted seal or qualified-window flag
in provisional records. Final result names the NEW finalization digest, never an
old materialized-receipt hash. Producer PCM fields describe final metadata only;
the pure adapter does not rehash absent audio or claim Task5 prepared audio.

## Begin and repeated context validation

Bound known record types/counts/strings/numbers/cycles/container shapes before
clone, identity formatting, serialization or hash. Validate actual origin fields
and rerun their real constructors; no live source/pin verification. Match origin
fingerprint/dimensions to assembly and source resource. Source resource must be
source_pixels/bgr24 with consuming-frame total. All-SOLID has no binding call.

Reconstruct the actual A OutputTimeline from real TimelineSpan values and call
its raster_spans. Check strict numeric types/no bool/finite representability
BEFORE constructors/math can leak OverflowError/TypeError. Compare every shot
span, original intervals, LOCAL_TRANSPORT1/60,n_start0,count and source layer
kinds. Consuming iff an authored source/camera primitive exists, not visibility.
Verify all transport prefix sums skipping only all-SOLID; resource count agrees.
A nonempty strict-integer slice fits ONE consuming occurrence and request caps.

This is the SAME structural association B's COMPLETE binder already performs.
A narrow shared helper extraction inside the TWO owned paths is allowed and
preferred over copying its substantial validation block. Preserve all valid old
outputs/byte references/receipt/PCM/limits and old public signatures. Keep
materialized and streaming resource/schema checks in their own wrappers, not a
streaming=True bypass or changed old64MB retention.

For each captured slot use the validated transport correspondence
K=transport.k_start+slot-transport.slot_start, then ACTUAL timeline.video_request(K).
Never use slot as K or copy A's inverse/leading-cell formula. Store those real
requests including local_n120 for a later batch. Repeated/out-of-order occurrences
keep distinct global K/slot identity. Original source occurrence/window contains
all exact requests; no seek/EOF/source-time repair.

On bind/finalize, recheck context against captured assembly and regenerate ALL
captured requests through A, comparing complete typed VideoSampleRequest facts,
occurrence/transport/window/source identity and header. A matching bool/int is
not a type check. This catches inconsistent mutated metadata, not coherent
replacement or use of an old valid snapshot.

## Reuse approved39 boundary/chain helpers

Current actual pure signatures, verified after review39:

```python
# source_decode, pure; real Window value construction performs no I/O
_decoder_interval(origin, window: Window,
    limits: DecodeLimits | AudioDecodeLimits | StreamDecodeLimits) -> tuple[Fraction,Fraction]
# render_clock, pure
_stream_header(origin: SourceTimeOrigin, session: str,
               payload: Mapping[str,Any]) -> dict[str,Any]
```

Use actual _decoder_interval with Window(*window) and StreamDecodeLimits to
obtain existing clipped margins; do not copy that engine. Use actual
_stream_header with the captured canonical window/requests/operation_limits/
delivery_limits/decoder_start/decoder_end/audio.policy fields, then existing
canonical/digest/rational for H0. No fake finalization/old receipt is made.
Header JSON is detached and bounded. Validate real StreamDecodeLimits and all
HOST ceilings through its constructors/records, not assumed annotations.

## Bind one selection

Require real bounded cursor/context/selection, correct session, dense next
delivery index, contiguous next nonempty ordinal prefix and strictly increasing
selected frame_n. Counts are plain integers. Guard known row/frame metadata and
actual lengths/dimensions before cloning/parsing/hash. Rebuild using REAL
ProvisionalSelection with cursor.chain_sha256, the original immutable frame and
detached selected/successor records. Compare all validated fields and actual
returned delivery hash, not a supplied hash alone. No second chain-link engine.

Call B's ACTUAL existing helper once for the delivery's first scheduled request:

```python
_validate_requested_frame(frame, selected_record, *,
    source_size=checked_source_size, requested_raw=origin.to_raw(request.requested_source))
```

The helper returns None and has no seal. For remaining requests on that SAME
already checked frame, check actual raw half-open interval with origin.to_raw;
do not rehash the same body per repeated request or copy the helper's byte engine.
The approved constructor owns successor-row/closed-end and link validation;
adapter does not possess successor pixels and must not claim it checked them.

Increment delivery/ordinal progress and byte counters under the effective limits
BEFORE constructing result slots. Unique selected bytes once per delivery;
logical emission bytes once per slot including repetitions. Return NEW frozen
cursor and ONE borrowed delivery. Do not accumulate frame/row history in cursor,
copy BGR or serialize frame objects. No acknowledgment or writer success here.

## Finalize with the real new loader

Require all ordinals bound and at least one delivery. Guard known finalization
fields/UTF8/serialization expansion before to_json or record/hash; no fake old
receipt or empty/predicted final hash. Call REAL
SourceStreamFinalization.from_json with captured origin, STREAM_DECODE_POLICY,
captured limits and expected_session_id. Constructor/loader own complete schema,
effective-size/argv/origin/audio/chain/accounting/completion validation.

Join exact captured session/IDs/window/requests/limit records/decoder bounds/
audio policy to validated payload/header. Require actual limits equality with
the entered session, not merely weaker-than-HOST. Compare validated last delivery
chain to cursor.chain_sha256, delivery count/next index, request count/next ordinal,
last selected n and selected_unique_bytes/acknowledged_request_bytes. Do not
reimplement/recompute the loader's chain/numeric/PCM engine. Final rows/outcomes
yield each typed slot's frame_n; A requests still determine K/local_n/raw request.

Return detached final metadata only, with actual finalization.sha256 and producer
PCM identity/count/coverage accurately labeled. No observed-frame objects/pixels,
no source/encoder execution proof and no publication authorization.

## Pure value semantics, not a second owner

A frozen pure cursor cannot invalidate earlier references. Reusing an older
valid cursor/selection or finalizing the same completed value twice is
deterministically possible and must be documented/tested. Throwing an exception
does not mutate the old cursor into FAILED. No registry/consumed flag/stateful
compiler session to fake global single-use. HOST owns latest-cursor linearity,
live observer acknowledgment/closure, output failure disposition and publication.

Actual owner enforces one lease. HOST verifies entered stream's IDs/window/
requests/limits/policy equal its captured context, uses latest cursor, releases
selection/delivery/views before the next pull, acknowledges once only after
acceptance, and discards unpublished render on actual failure. Pure bindings
never interrupt caller code, observe deadlines/children or prove encoder writes.

## Fixed pure-work bounds and verification

Reuse existing B8000000-byte text/metadata,32768-entry and4096bit guards, including
all captured assembly/context/result fields and aggregate strings. No silent
truncation or generic graph/validation framework. Bound before constructor,
parser/hash/clone and check actual generated results. Final envelope cap is
min(captured.max_finalization_bytes,8000000). Source frame bytes bounded before
hash by source dimensions/max_pixels/max_live_payload_bytes. Requests at most
min(max_requests,max_frames,1200); deliveries min(max_deliveries,1200). Enforce
max_selected_unique_bytes and max_request_emission_bytes before updates.
No old retained alias or new UI knob, no RSS/throughput claims.

Guard provisional results through their explicit metadata/slots and actual byte
lengths, not a generic walk/serialization of frame.bgr. For finalization objects,
bound payload_json before to_json can parse/hash it. A narrow shared extraction
of B's existing JSON pre-work accounting is allowed for old/new schema wrappers,
with each wrapper preserving its own shape/numeric semantics. Add lexical depth/
conservative entry reservation before parsing if needed, not a duplicate stream
receipt validator or an upstream source edit. Invalid forged fields must remain
ValueError-family failures; no broad swallow-and-accept handler.

ONE new task review, TWO static paths, exact pure selected test nodes. Do not run
clock/decoder/guarded/native/font/Director/A/fullrepo suites or actual import-
isolation child. Existing test_provider_free_imports_do_not_load_settings_or_models
launches a real15s Python child; exclude explicitly. Existing selected B in-memory
complete-binder/late-local-n/caps/known-shape cases stay useful regressions; no
historical test/helper body rewrite. No fake whole validator on positives.

Positive recipe: real A/B compiled source/SOLID/source plan, slots6..11→K12..17,
two real frame groups for raw5015/5050/5130, six requests,14400 actual zero PCM
fixture identity, real ProvisionalSelection and real SourceStreamFinalization/
loader. Synthetic producer peak60000 within reservation80000 and synthetic child
completion are fixture facts, NOT observed execution. Also later slice120..125
and partial123..125 with real A local_n; all-SOLID rejects binding. Cases cover
foreign/malformed/context/row/body/chain/order/finalization/cap corruption and
honest replayable value semantics. Source39/38 approved evidence is not rerun.

No actual observer, child/timer/worker, file/media/font/encoder/queue runtime
qualification, model/semantic/human acceptance or full-VOD performance follows.
6b owns live origin/pins, writer/supervision/resource materialization/audio/output,
and the user-approved original-font rendered-validation gate.
