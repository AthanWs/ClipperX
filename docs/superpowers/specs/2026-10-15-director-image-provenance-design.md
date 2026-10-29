# Observed Director images and pinned host provenance — Task6a1-B

Status: adopted technical continuation of the approved automatic vertical-shorts
vision.6a1-A is independently COMPLETE (current package35 SHA256
7D776DBE9F2943877FF8A160CDB77E25CE27FDCA69D9142E7353C18D8ED570A0).
Root read the full23749byte architecture proposal/addendum and checked the real
consumer/public seams. This unit changes no editorial/identity threshold, layout
choice, model response schema, prompt module, style, UI setting or ASR behavior.

## One coupled unit, SIX existing paths

Own evidence.py, visual_director.py, planner_engine.py and their corresponding
tests/test_evidence.py, test_visual_director.py and
test_visual_planner_integration.py. Snapshot36 precedes edits; one fresh writer
then independent review36. No analyzer/provenance/decoder/clock/runtime/JobStore/
contracts/prompts/CLI/assets/profile/source changes or new source file split.
Concrete outside-six needs must be reported before editing.

Extraction and host propagation have separate TDD steps but one integration gate:
the old Director followup currently rewrites a collected bundle to add request,
and old Director/integration fixtures decode video without audio. Their callers
and fakes must adapt in the same unit to keep the new bundle immutable. Do not
leave a knowingly incompatible halfway public path. The proposal is background;
this design plus exact brief is authority.

## Provider-free image interfaces

In evidence.py add frozen ObservedEvidenceContext(runtime: RenderRuntime,
origin: SourceTimeOrigin). Its construction performs no tool/model work.
prepare_visual_evidence_context(source_video: Path) returns it after resolving
decode_requirements through resolve_render_runtime, bounded probe_source and ONE
qualify_render_runtime for the pinned CPU pair. No system/Vulkan fallback, face
model, provider, ASR or analyzer. CPU preparation is once per Director run, reused
for initial/followup extraction. Standalone collector callers may prepare it.

Keep collect_visual_evidence's existing Path arguments/return and defaults; add
only optional keyword context: ObservedEvidenceContext | None = None and
request: VisualEvidenceRequest | None = None. Without context it prepares one;
without request it emits initial evidence. Followups pass request at creation,
never modify a published bundle. collect_evidence's legacy editorial route and
its helper/cache are unchanged.

Expose load_visual_evidence_bundle(bundle_path: Path, *, source_video: Path,
timeline: SourceVisualTimeline, windows: Sequence[TimeRange],
context: ObservedEvidenceContext, max_images: int = 5,
request: VisualEvidenceRequest | None = None,
expected_bundle_sha256: str | None = None) -> dict[str, Any].
This performs bounded file/hash/receipt/association checks, returning a detached
parsed record. It does NOT resolve/qualify/probe/decode, construct models, choose
new samples or rebuild evidence. A captured expected file SHA prevents accepting
modified bundles merely by hashing them anew. Host callers retain their expected
windows/request/timeline fingerprint and compare actual files to captured hashes.
Reuse A's public bounded JSON/path/file helpers where applicable, not a second
generic manifest/transaction framework or a dependency on analyzer internals.

## Actual pixel collection and finite work

Keep visual_frame_times' event/boundary/start/end-adjacent priorities and existing
1..5 source request cap. Convert selected floats once through decimal strings to
exact Fractions. Never use FPS, CAP_PROP_POS_MSEC, seek clamping, analyzer previews,
replacement content, fake black pixels or inferred PTS. A selected unavailable
point stays unavailable. Existing associated_face_positions remains untouched.

Use approved observe_windows with analysis16k-mono. Nearby selected points are
grouped with the approved coalesce/Window primitives and a tiny <=5-point budget
partition, not another frame selector or a copied analyzer scheduler. Requested
span <=2s, decoder interval <=3s including .5/.25s margins;45s timeout and all
existing lower hard ceilings remain. Split groups before dispatch if selected raw
BGR plus incidental PCM exceed64,000,000bytes or estimated BGR working copies
exceed192,000,000bytes. Account dimensions, predecessor/lookahead/array copies;
no FPS-based timing or allocation estimate. Process/release each observed result
before another batch. Never retain all raw batches. Estimates are not RSS/NN data.

Each covering ObservedFrame supplies full-size BGR. Encode THAT array once using
cv2.imencode JPEG85, no resize/crop. Store raw BGR SHA, integer PTS/timebase,
source display interval, dimensions and separate lossy JPEG file SHA/bytes/policy.
JPEG bytes are not raw pixels. Observed receipts include incidental audio; validate
the actual immutable result on creation, retain its receipt, then release PCM.
No B WAV, audio features, ASR or independent-audio quality claim is created.

All files belong to an exclusive output_path.parent/output_path.stem directory;
bundle JSON is published LAST at output_path, and is immutable. Paths in its
catalog are relative/confined, reject traversal/drive/backslash/symlink/junction
escape, duplicates, wrong scalar types/nonfinite values, dangling/unassociated
files and mismatched receipt/request/frame associations. Each artifact/JSON has
the existing64,000,000byte ceiling. At most2*max_images receipt/image artifacts
plus one bundle JSON exist (max11files, max704,000,000bytes by these bounds).
Reserve next work and final JSON before observation; check encoded/emitted bytes
BEFORE writes and reconcile actual sizes, including Windows newline translation.
No silent sample reduction or storage cleanup. Prepublication failure retains
uncommitted partial files; no older bundle is overwritten or relabeled.

## Bundle protocol and temporal semantics

Use version1/kind observed_director_images. Exact emitted schemas and types are
checked. Root fields: version,kind,source,source_fingerprint,origin,origin_id,
timeline_fingerprint,windows,max_images,request,selected_requests,outcomes,
observations,frames,catalog. Source is the canonical expected local source;
request is None or the validated followup request. Windows are the original
editorial/request windows, not decoder margins. Preserve their order, including
legitimate repeated editorial ranges. selected_requests retains selection order
and exact rational coordinates; recompute expected selection using the unchanged
visual_frame_times on validation. Each point belongs to its declared windows.

Catalog rows are exactly path/bytes/sha256. Observation rows are exactly
path/bytes/sha256/observation_id, binding relative receipt path, file SHA and its
distinct envelope observation identity. Parse bounded JSON
once and validate SourceDecodeReceipt against current_origin/policy; no unbounded
second read. The expected envelope policy is the HOST constant DECODE_POLICY,
not the envelope's own assertion; incidental audio policy is analysis16k-mono.
Its actual outcomes must bijectively account for selected requests.
No audio-only receipt is substituted for a video observation.

Each unique frame row retains existing evidence_id,at,requested_at,path,width,
height for caller compatibility, plus frame_id, raw_bgr_sha256,raw_pts,time_base,
observed_source_start/end,jpeg and sample_refs. jpeg contains exactly relative_path,
sha256,bytes,quality (integer85),resize (string none),opencv_version (nonempty str).
Its compatibility absolute path must exactly
equal the host-derived confined file; never open a supplied absolute path first.
sample_refs contain requested rational coordinate, original window_indices,
receipt path and frame_n, each checked against the corresponding actual outcome.
One row can represent multiple requests and all applicable repeated windows.
Outcomes bind each selected request/status/receipt/frame_n/frame_id/window_indices;
unavailable has no invented frame. Do not create synthetic face detections.

at/requested_at is the representative PRESENTATION SAMPLE coordinate, usable only
after actual covering observation. Raw PTS and coverage are separate and never
overwritten. A predecessor beginning before a fractional cut is valid when it
covers a requested instant INSIDE that half-open window. Its endpoint is excluded.
Do not confuse this new semantic with legacy bundle versions or raw face times.

frame_id is digest(canonical(record)) with exact fields origin_id,decoder_id,
stream_index,raw_pts,time_base,width,height,raw_bgr_sha256. Use the receipt's
canonical raw timebase and origin identities, not local decoder n, request time
or JPEG SHA. Filename is frame-<full frame_id>.jpg; evidence_id is
<bundle_path.stem>-frame-<first16 frame_id characters>.
The same stream/PTS with conflicting bytes/coverage is an error. Same pixels at
different PTS are distinct views. Same PTS covering multiple requests is ONE view.
Asset filenames/IDs are host-derived from that identity; sample/evidence aliases
remain explicit. Validate created bundle before returning its Path.

## Director and planner host context

Add only optional source_timeline_path: Path | None = None to run_visual_director.
Planner passes its EXACT persisted resolved Path. Never guess JobStore fixed v7,
latest, newest directory or a model-supplied Path. Canonical equality with the
supplied detached timeline object is mandatory. Use A's EXACT Unicode/default-
spaces visual_timeline_fingerprint and load_visual_provenance with current origin.
Current A context is frozen/to_record detached and supplies observed_samples or
unverified_legacy; observed is sample provenance, NOT full/narrative coverage.
An object-only caller stays unverified_legacy/reason object_only_no_path while
its NEW Director images can be observed. Missing/corrupt marked generation is an
error, not reanalysis or legacy fallback. No model may assert this host context.

Replace the Director's unpinned metadata probe with this pinned origin. Preserve
source dimensions/duration/fingerprint checks. Capture selected transcript bytes/
path/SHA once as unverified_legacy or missing, preserve existing text selection/
limits. No shift, ASR rerun, profile repair, human approval or training upgrade.

Before each model call, after its response and before terminal publication:
revalidate source/current origin, pinned timeline+full A companion through its
loader, consumed text, bundles/receipts/JPEGs and style authority against CAPTURED
identities. Fresh metadata/hash resolution is allowed; drift checks never qualify
CPU, construct models, decode or rebuild. A different unrelated latest must not
redirect/invalidate the unchanged pinned generation. Repeated full-generation
hashing cost is unmeasured; do not bypass A validation or promise full-VOD speed.
Never accept changed JPEGs by replacing expected hashes with current hashes.

_manifest deduplicates across initial/followup by frame identity and unions aliases/
sample_refs. Temporal requests camera_boundary/state_transition/crop_motion need
two DISTINCT observed presentation instants with covered refs inside the request,
ordered by raw presentation time; one frame with two refs is not two views.
Choose mandatory earliest/latest eligible identities, then existing event/bundle
priority, maintaining five source plus three style views and existing call limits.
One asset satisfying overlapping requests occupies one image slot. Persist which
images were ACTUALLY sent per call; decoded-but-omitted is not presented evidence.
Keep existing non-image evidence/geometry/protection rules. Image IDs are local
to a shot only via verified covered sample_refs inside that shot, not representative
at alone or mere display-interval overlap. Do not retime face-track associations.
No usable initial image or insufficient requested views => review_required without
a spurious/redundant client call; no arbitrary replacement content.

Propagate captured host status/identities through authority.json, existing
clip.host_context, call images.json, terminal receipt and provenance.json beside
the plan. Keep images.json's existing list shape; source rows include their own
receipt/alias/sample_refs and upstream-context hash, while the full context lives
in authority/companion. Prompt module/model plan schema stay unchanged. Companion
version1/kind visual_plan_provenance binds run UUID, source/origin, pinned upstream
context/path, legacy text status, bundle/image/receipt catalogs, and each call's
actual manifest SHA/presented assets. A ready companion additionally binds both
canonical model-plan fingerprint and plan-file SHA. Review companion records last
validated facts and failed check, never a current validation after drift.

## Terminal commitment, old-stage preservation and explicit recovery

Existing terminal receipt remains publication marker. Use version1/kind
visual_director_receipt,run_id plus existing status/clip/source/calls fields and
companion_path/sha256. Ready includes plan_path/sha256; review includes controlled
reason and NO ready visual_plan.json. Bind exact expected source/timeline/context.
Use stage visual_director:<clip_id>:<actual run root.name UUID>, preserving old
fixed keys/review flags through existing JobStore APIs. Use an existing per-run
stage lock for execution/recovery; no JobStore code or global lock order change.

Prepare known plan/ready-companion/ready-receipt candidates in the unique run.
Success: publish visual_plan.json, provenance.json; revalidate dependencies/exact
bytes; publish receipt.json LAST; then update advisory JobStore. On any failed
rename/replace, inspect only the KNOWN target: exact matching intended bytes and
associations establish that operation materialized; definite FileNotFound is
absence; permission/I/O/unreadable/mismatching state is ambiguous, not rollback.
Even a normally returned replacement must leave expected invariants. No blind
retry/promotion, unknown overwrite or newest-directory scan. Reuse existing
atomic_write_text and narrow postconditions, not a generic transaction framework.

Inspect terminal receipt BEFORE failure handling. A valid materialized READY is
committed even if acknowledgement or JobStore update failed. Never rewrite it as
review or move its plan. A damaged committed association is an integrity error,
not permission to downgrade. A valid REVIEW likewise remains final. A malformed/
unreadable/conflicting terminal receipt requires explicit recovery.

Before terminal commitment, preserve ONLY this invocation's verified final plan
as uncommitted-plan.json and ready companion as
uncommitted-ready-provenance.json; retain candidates. Verify move postconditions,
do not replace conflicting retained bytes. Only after definite absence of final
visual_plan.json and no READY may a failure companion and REVIEW receipt be
published. If preservation/publication is ambiguous or impossible, raise a
recoverable operational error and retain partials; do not invent a review receipt.

Expose DirectorPublicationError carrying phase,run_path,committed_path (or None)
and expected identities. Phases AMBIGUOUS_PUBLICATION,
PRECOMMIT_PRESERVATION_FAILED,COMMITTED_BOOKKEEPING_FAILED are controlled host
values; do not persist provider exception messages. Do not catch this error in
the generic review fallback. recover_visual_director_publication(source_video,
run_path,*,jobs_root=None)->Path validates the exact known run/source namespace,
authority/candidates/terminal associations under the same lock. It can reconcile
an already committed terminal receipt or finish bounded precommit preservation/
review. It never promotes an abandoned candidate to READY, scans or reruns models/
decode/CPU qualification; source metadata/hash checks alone may be refreshed.
Review36 I1 clarification: a final REVIEW generated by source-stat drift is
reconciled in its CAPTURED source/job namespace, not a new namespace derived from
current source stats. Validate the host-supplied source Path, confined known run,
captured fingerprint/job/clip/UUID and existing state/authority/terminal links
under the same run lock, without initializing another job. This does not bypass
current READY dependency validation or assert that failed current inputs became
valid. A coherent zero-call early REVIEW with no captured authority/context is
distinct from a malformed or partially captured post-authority REVIEW.

Planner validates terminal receipt->companion->plan/bundle and upstream identity
against its captured pinned context BEFORE marking ready. A
COMMITTED_BOOKKEEPING_FAILED error with a verified ready Path may be normalized
to that ready artifact with an explicit bookkeeping_error phase on its entry;
do not pretend the committed plan vanished. Other operational/integrity errors
preserve prior successes and remaining pending clips under existing error status.
Keep ready/review_required/error independent of upstream/image/text provenance.
No historical manifest, approval or failed external model run is rewritten.

## TDD and ONE synthetic HOST control

Ordinary tests fake approved resolve/probe/qualification/observe boundaries and
clients, but use actual small BGR->JPEG/files/receipt/hash I/O. Adapt old Director
and planner actual-Director fixtures (MJPG without audio) to source placeholders
plus coherent fake origin/observation and metadata. Preserve existing geometry,
editorial ranges/call/image budgets. No repeated real source-decode matrix.

Cover initial/followup immutable request, raw->JPEG association, actual unavailable,
same-frame aliases versus static pixels at distinct PTS, fractional predecessor,
half-open end/15/59ms, temporal one-view rejection/overlapping budgets/locality,
pinned Path versus unrelated latest, Unicode, legacy/object-only, corrupt marked
upstream and drift before/during/after call. Cover ready/review/partial manifests,
old flags, every pre/postrename/ack ambiguity row and explicit recovery. Resource
tests fail BEFORE fake observations/NN/file writes; no swallowed stale errors.

After fake gates, ONE new local HOST control is authorized under exclusive
.superpowers/sdd/phase3-task6a1-B-proof/, via guarded run-once.py OUTSIDE automatic
pytest collection. This is a deliberate refinement of the proposal's test-node
placement: ordinary regressions cannot accidentally repeat its media campaign.
No new user/product option. The script guard is exclusive and all results persist;
do not retry on failure or run another source without reporting first.

Fixture path host-source.mkv; raw inputs host-bgr.raw and host-audio.s16le.
Exactly10 known64x48 BGR
frames at10fps, FFV1 bgr0, video timestamps (100*n+59)/1000, PCM16 mono48k silence
from0for1.2s (57600 known zero samples). Both raw inputs are finite; explicitly
map them and do not use shortest-stream truncation. Source<=2s. Use the existing
pinned CPU9 pair; root checked current
FFV1 help supports bgr0 (text only, no fixture executed). Independent PTS definition
uses settb=1/1000,setpts=N*100+59, -enc_time_base:v filter and
-fps_mode:v passthrough; no coarse input itsoffset rounding. Mux audio makes the
common raw origin0. Documentation consulted through Context7 official
https://ffmpeg.org/ffmpeg-all.html: filter TB/PTS and encoder/fps modes; mux PTS
still must be asserted, not assumed from flags. No fixtures installed/downloaded.

One production collect_visual_evidence call, one bounded combined observation
window, max_images3; original window[.07,.25), a synthetic important sample at.08
gives priority-selected requests .07,.2,.08. First two chronological requests
.07/.08 cover raw PTS59ms; .2covers159ms. Exactly TWO raw identities/images,
aliases preserved. Assert header/receipt PTS/intervals and compare raw BGR hashes
to independently known arrays; compare JPEG85 bytes/file SHA to independent
cv2.imencode of those arrays. This is a lossy-encoding association oracle, not
raw==JPEG. Capture every child/argv/exit/time, source/current code/model-no-call
guards; each child<=45s. One fixture-creation child, one collector invocation;
normal bounded metadata/CPU qualification and its video+incidental-audio children
are included. No Director/provider/NN/R/S/actual stream/8source/GPU/font/ASR/
delivered-MP4/AAC/encoder qualification or old guarded-runner reuse.

Final focused six-path covering/static checks use .venv-whisperx/Scripts/python.exe.
Record pre-edit Ruff/mypy baseline and exact final delta with existing flags;
no ignores or unrelated debt cleanup. Invalid Git: no init/repair/worktree/commit/
push; intended subject `feat: bind Director images and plans to observed provenance`.
Final report then ORIGINAL36 package/fresh independent spec+quality review.
No third live model/API campaign in this checkpoint. Renderer6a2/6b, Task7,
full-VOD throughput, semantic/model/human acceptance remain separate.
