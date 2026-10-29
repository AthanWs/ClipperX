# Ordered visual assembly and observed input association — Task6a2-B

Status: ADOPTED technical continuation after6a2-A independent review37 Approved.
Root read COMPLETE22410B concrete proposal, approved A report/review, actual
record/primitive/receipt APIs and the scoped primary FFmpeg documentation noted
in B preflight. This preserves approved vertical-shorts vision/style/ranges and
all model/renderer gates. No B source/test/baseline preceded adoption; edits need
the exact activated brief and ORIGINAL38. No new UI/model knob or encoder choice.

## Scope and exact public seams

FOUR existing paths only: filtergraph.py/test_filtergraph.py and
visual_compiler.py/test_visual_compiler.py. No A, motion, assets, audio, runtime,
clock/decoder, Director/schema, public-render/CLI, original/proof changes. Reuse
existing geometry/crop/fill/preservation/native policy and requirement combiner.
Historical compile_shot_layers/compile_visual_layers remain explicitly old-clock
component APIs; the NEW path never calls them or inherits their trim/fps clauses.

```python
def compile_source_layer(shot: VisualShot, *, layer_index: int,
    raster_span: RasterSpan, source_size: tuple[int, int],
    input_label: str | None) -> SourceLayer: ...

def compile_visual_assembly(plan: VisualCompositionPlan, *,
    source_size: tuple[int, int], overlays: OverlayCompilation
) -> CompiledVisualAssembly: ...

def bind_observed_batch(assembly: CompiledVisualAssembly, *,
    slot_start: int, slot_end: int, origin: SourceTimeOrigin,
    decoded_window: DecodedWindow,
    expected_audio_policy: Literal['analysis16k-mono', 'render48k-mono',
                                   'render48k-stereo'],
    limits: DecodeLimits = DecodeLimits()
) -> BoundSourceBatch: ...
```

Signatures are complete interface definitions, not stubs to add. Invalid/unsupported
input raises the existing RenderPreflightError/ValueError family before any host
operation. No tool/runtime resolution, source/asset/font file access, model,
measurement or decode in these functions. Python>=3.10 remains supported.

## Detached immutable handoff records

All are frozen/slotted, strict scalar types, detached tuples/JSON text. Retain
no caller-mutable plan/layer/list/dict. Existing immutable tracks/rectangles/
prepared assets/observed frames may be referenced; no raw-pixel copies/serialization.

```text
SourceLayer [filtergraph]:
  shot_id,layer_id,declaration_index,kind,z_index,destination:PixelRect
  input_label:str|None,output_label,filter_complex
  crop_track:CompiledCropTrack|None,requirements:RenderRequirements
SourceTransportSpan [visual_compiler]:
  shot_id,k_start,k_end,slot_start,slot_end
SourceInputResource:
  key='source_pixels',pixel_format='bgr24',source_size,frame_count
FilterFragment:
  text,input_labels:tuple[str,...],output_labels:tuple[str,...]
AssApply:
  document_key,input_label,output_label
AssDocumentResource:
  key,document,event_indices:tuple[int,...],windows:tuple[(int,int),...]
BrandInputResource:
  key,graph_label,brand_index,shot_id,required_asset_id
  raster_window:(int,int),placement:PixelRect,asset:PreparedBrandAsset
  declared_interval:(Fraction,Fraction),execution_interval:(Fraction,Fraction)
CompiledShotVisual:
  span:RasterSpan,transport:SourceTransportSpan|None
  source_input_label:str|None,source_layers:tuple[SourceLayer,...]
  steps:tuple[FilterFragment|AssApply,...],output_label
  output_clock='local_transport',output_time_base=Fraction(1,60)
  local_n_start=0,frame_count:int
CompiledVisualAssembly:
  timeline:OutputTimeline,source_size,expected_source_fingerprint
  shots:tuple[CompiledShotVisual,...],source_input:SourceInputResource|None
  ass_documents:tuple[AssDocumentResource,...]
  brand_inputs:tuple[BrandInputResource,...],canonical_ass
  requirements:RenderRequirements,compiler_metadata_json:str
ScheduledFrameBinding:
  transport_slot:int,request:VideoSampleRequest,requested_raw:Fraction
  receipt_hash:str,frame_n:int,frame:ObservedFrame
BoundSourceBatch:
  slot_start,slot_end,origin_id,decoder_id,policy,receipt_hash
  bindings:tuple[ScheduledFrameBinding,...]
  verified_pcm_sha256,verified_pcm_bytes,verified_retained_bytes
```

Metadata includes all original/raster intervals and deltas, primitive IDs/boxes/
tracks, ordered operation/resource references, declared and execution overlay
windows, clock stages, source-slot correspondence and honest provenance labels.
JSON uses existing canonical/rational helpers, never asdict raw frames. Labels/
keys are compiler ordinals or existing safe hashes; model IDs are metadata only.
There is NO runnable whole-clip filter_complex.6b binds source/PNG input labels,
ASS keys to trusted materialized files and any qualified transport topology.

## One mapper, sparse source slots and finite primitive clocks

Resolve once through existing resolve_visual_plan; A's OutputTimeline alone
allocates requests/spans/windows. Source/camera source_ref is provenance, never
a filename/detector instruction. compile_source_layer verifies layer index and
exact shot/span/original-boundary association, then returns ONE destination-sized
spatial primitive. Preserve authored crops/easing/independent components/fills/
blur/preservation; reject SOLID crop tracks. No flattening other layers into it.

Add optional frame_count:int|None=None to existing compile_layer and
compile_motion_layer in the OWNED filtergraph file. None preserves historical
behavior. Positive plain-int N controls generated colors/backgrounds via one
shared finite-color→trim-N→local1/60-clock helper, also reused for the black base.
Do not derive count by rounding duration. Native transforms receive the N-frame
local inlet; static delegates receive N. No new resampler or motion-engine math.

needs_source is true only if an authored source_video/camera primitive exists.
Do not infer visibility/cull occluded sources. Prefix-sum N for consuming shots
only; slots are INPUT ordinals, not timestamps. Source/SOLID/source withN2 each
has slots0,1,2,3→K0,1,4,5. For a slot, K=k_start+slot-slot_start, then call
timeline.video_request(K). Never call it with slot or use slot/60 as output time.
All-SOLID clip has source_input=None, no raw pipe/observation obligation, but
all output K/generated N and audio/source-identity obligations remain.

Each consuming CompiledShotVisual declares exactly N BGR frames on its symbolic
source_input_label for its transport span. Source/camera branch from THAT same
local stream; normalize it to1/60 and frame-count PTS before the split. Native
n is local0..N-1 and never resets at observer batches;2sec boundary remains120.
After spatial operations, canvas/primitives are put on GLOBAL K before painting.
After all painting, the compiled final step resets output TB to1/60 and PTS to
zero-based frame count. Return LOCAL_TRANSPORT metadata above, retaining the
global span: K=span.k_start+n. This supplies either later host topology without
a second mapper. No whole-clip concat, audio padding, source seek or FPS repair.

Combine existing spatial requirements and actual generated/ASS/PNG filters;
add existing decode_requirements only when source_input exists. Requirements
are pure conservative capability declarations, NOT runtime qualification.

## Prepared overlays, stable global z and execution representations

Require actual known record/scalar types and finite fields. Join overlays to the
SAME resolved plan before graph creation; exactly planned titles/brands, no
foreign/extra/duplicate occurrences/events. Missing subtitle events remain valid
for known silence, not proof of transcript completeness. Title compatibility
floats equal existing producer sums, exact_interval equals original operands via
the mapper; text uses existing _clean_text, destination/z and title fixed
white/black/outline2/fitted1..46 policy agree. Subtitle exact intervals are
mandatory/contained through raster_window, floats project them; destination,
font size/colors/outline/bold, max words/lines, text/range/active indices and
existing max+1 z agree. No active indices when emphasis is off. Measure nothing.

Use existing _color/_clean_text, collision function and serialize_overlay_events;
canonical document must equal that serialization exactly. Do not copy formatting,
normalization, wrapping or caption packing. Single-line ASS style scalars reject
comma, CR/LF/NUL/control injection; font family is a bounded nonempty string,
not an exact-file qualification. Check finite reported extents fit destination.
Brands use the existing prepared-asset value constructor and PURE place_brand_asset
to validate occurrence/placement/z/compatibility times against current plan; no
PNG read/retrim. Required asset ID remains an explicit host binding obligation.

OverlayCompilation alone cannot prove transcript/silence completeness or font/
PNG registry identity.6b must freshly produce it from its same selected plan,
transcript and staged registry. Canonical tiny-interval serialization rejection
remains supported-subset failure; do not bypass/merge/drop slices to force success.

Paint ascending(z,tie_slot,declaration_index):source/camera/SOLID0,title1,brand2,
subtitle3. Source index is original layer position; event index is original
canonical order. Subtitle max(source/title/brand)+1 remains. Group maximal adjacent
ASS operations only; source/PNG separates runs. Clone events to adopted
C(K)=floor(100*K/60)/100 for BOTH compatibility and exact execution intervals,
then use SAME serializer. Originals/canonical document never mutate. Record
original event indices/windows. Untrusted text never enters filter syntax.

PNG execution is the explicit root refinement E(0)=0,E(K>0)=(2K-1)/120. ForK>0,
(K-1)/60<E(K)<K/60, so at j/60 the enabled set is EXACTLY Kstart<=j<Kend.
Pass float E projections to SAME compile_brand_layer on a detached timing copy.
Preserve original exact declared interval, exact raster window and E values in
resource metadata. This avoids equality at a raster tick without an epsilon or
assets edit; it does not qualify FFmpeg runtime evaluation. ASS C is unchanged.

AssApply uses resource keys, not filenames. Source/brand graph labels, all keys
and outputs are collision-checked and have complete binding descriptors.6b must
bind them bijectively; no guessed numeric input index or arbitrary replacement.

## Completed observed-window byte association

Nonempty plain-int slot slice must lie inside ONE consuming occurrence. Derive
its exact ordered requests via the mapper. Receipt outcomes must equal those
requests exactly, all covered; no missing/extra/duplicate/unavailable outcome.
Receipt windows are contained in the original source occurrence and cover the
requests; decoder margins remain separate. Unknown terminal duration still fails.

Before parsing/cloning/serialization apply bounded guards below. Require actual
DecodedWindow/SourceDecodeReceipt/ObservedFrame records and immutable bytes;
snapshot frame containers, do not copy large bytearrays/memoryviews into bytes.
Use REAL SourceDecodeReceipt.from_json with supplied origin and HOST DECODE_POLICY,
never receipt's self-asserted policy. Origin fingerprint/dimensions/decoder match
assembly and current supplied origin; caller/host owns live origin verification.
Receipt limits may be no wider than caller DecodeLimits and adopted ceilings.

Selected frame_n values bijectively match actual objects, once per unique frame.
Check strict integer tag fields/no bool, positive Fraction timebase, Fraction
raw_end, full tag/row agreement, exact raw interval, byte length/dimensions,
SHA256 and seed0 Adler32. Requested raw O+s lies inside half-open coverage.
Shared selected frames may serve multiple K/slots without copying/counting twice.
Use(receipt_hash,frame_n)+raw facts, not another raw-frame identity algorithm.

DecodedWindow construction does NOT validate PCM. Match the expected HOST audio
policy (one of existing3); verify actual PCM aggregate bytes/SHA and each ordered
OUTPUT block bytes/SHA/seed0 Adler through bounded memoryview slices, final
offset exactly equals len(pcm). Existing receipt validator retains all sample/
format/PTS/conversion/coverage math; do not copy that engine or hash unavailable
input PCM bodies. Reconcile unique BGR+PCM actual retained bytes. Incidental
coverage may remain unavailable; do not upgrade it, normalize, derive features,
invent another receipt or treat analysis16k as Task5 prepared audio.

Private _validate_requested_frame and _validate_materialized_pcm helpers stay in
visual_compiler. Only COMPLETE wrapper attaches sealed receipt_hash; no provisional
API/empty predicted hash/state machine now. Batch retains BGR refs, not PCM.
Metadata is source_input_observed association, NOT actual rendered visibility.
SOLID-only is not_applicable_no_source_consumer; generated data is compiler_resolved.

## Fixed host pure-work bounds (not RSS or execution budgets)

Primitives4096;events8192;ASS resources4096;brand resources1024;steps32768;
characters/event16384;aggregate INPUT UTF-8 text8,000,000bytes;aggregate generated
fragment/document/metadata8,000,000bytes. Otherwise uncapped nested metadata
sequences share a32768-entry bound. Numeric metadata components have max4096bits
before decimal formatting (normal float-derived rationals fit); no huge scalar
stringification to find its length. These are fixed technical host ceilings, not
model/UI controls. Exceeding a cap rejects, never truncates/omits.

Before clone/model_dump, inspect known plan/overlay/resource shapes only: types,
cycles/unexpected nesting, sequence lengths and ALL textual fields, including
IDs/reasons/fingerprints/font/style/path/canonicalASS/labels/metadata. Count UTF-8
in bounded chunks after scalar-length checks, path components before composing
oversized paths. Reserve conservative known-serializer output expansion BEFORE
geometry/expression/ASS creation; check actual accumulated bytes before retaining
each result. Keep this a narrow known-structure guard, not a generic framework.
Bounds account declared work/bytes, not Python/GPU/OS peak memory or throughput.

Binder additionally caps receipt UTF-8 at8,000,000bytes,requests/video rows1200,
actual unique BGR+PCM retained64,000,000bytes; all other limits no wider than
DecodeLimits defaults/tighter supplied bounds. Hash in place, no full copies.
No earlier64batch/child policy or general long-clip execution support adopted.

## Verification and remaining ownership

Pure TDD uses real A/frame/receipt metadata and tiny authored bytes, not fake
entire validators or copied analyzer/Director fixtures. Test source/SOLID/source
sparse slots and all-SOLID; frame120/localn across batches; exact event activation
all residues; global interleaved z/ties and two ASS runs; resource closure;
mutations/foreign/missing/extra overlay metadata; one escaping path; independent
predecessor/EOF/outcome and BGR/PCM corruption; host limits before expensive
sentinel helpers; immutable/detached results. New entrypoints start no host work.

Only TWO owned test files for final gate, FOUR scoped Ruff/mypy/compile, baseline
first with project interpreter and --explicit-package-bases --follow-imports=silent.
No decoded/native/font/media/decoder/A/Director/fullrepo campaign. The existing
import-isolation Python child is non-media and bounded15s. Fresh independent
spec+quality review38 required. Invalid Git: filesystem snapshots/no commits.

Streaming keeps old materialized receipts intact and needs its own exact policy/
scope.6b owns live origin-aware host preflight (old raw/source comparison must
be repaired under explicit visual_compiler/test ownership), font/asset staging,
safe path/input binding, actual topology/queues/GPU/framesync/count/color/pipe/
cancellation and ONE final encode/publication. Existing ALL-plan HDR rejection
stays. Neither finite templates nor pure active-index proofs establish runtime
success, encoded A/V, full-VOD performance, model semantics or human acceptance.
