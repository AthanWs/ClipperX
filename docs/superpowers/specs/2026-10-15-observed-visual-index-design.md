# Observed visual index generations — Task6a1-A

Status: adopted technical integration under the approved automatic vertical-
shorts objective.6a0 and6a0-audio-seam are independently approved. Root read the
complete proposal/final addendum and preflight, and inspected the actual builder,
runner, contracts, feature arithmetic and fingerprint consumers. No new creative
layout, state/identity threshold, model/schema/UI control, dependency or ASR.

## Scope and public compatibility

Own TEN paths: visual_timeline.py, jobs.py, audio_features.py, NEW
visual_provenance.py; their four corresponding test files; and
scripts/validate_source_timeline.py plus tests/test_validation_manifest.py.
No clock/decoder/runtime/audio_mix/visual_contracts/evidence/director/planner/
CLI/legacy timeline.py edits.6a1-B separately qualifies Director images/manifests.

Keep BOTH current public builder signatures and Path return values unchanged.
Add a small internal full-build result seam, `build_source_visual_timeline_result`
with the SAME kwargs/defaults as the full public builder, returning frozen
`VisualTimelineBuildResult(artifact_path: Path, cache_hit: bool,
indexed_samples_this_run: int)`. Public full builder returns its artifact_path.
Exactly0 new samples on a hit; a new generation reports its actual newly analyzed
sample count, not all decoded frames. Preserve Path/custom BuildMetadata runner
compatibility; default runner uses this SAME locked invocation result rather
than guessing from old files, latest-pointer races or historical generation size.

## New immutable namespace

Full builds allocate exclusively at
job.root/visual_observed/generations/<uuid4hex>/; NEW latest.json lives at
job.root/visual_observed/latest.json. Do NOT redefine or overwrite
JobStore.source_visual_timeline_path or any old JSON/frames/probes/PCM/transcript/
profile/model failure receipt. Full build retains existing source stage lock;
only its NEW source_visual_observed:<uuid> JobStore record may change.
Old stages/review_required remain semantically identical.

Bounded inspection allocates at output_dir/visual_observed/<uuid4hex>/, or under
the existing default inspection root's new visual_observed namespace. No
JobStore.for_video, state writes, source stage mutation or shared latest pointer.
Return the generation's source_visual_timeline.json Path. Force always allocates
a new generation; no replacement of any previous generation or old inspection.

Each generation contains source_visual_timeline.json, clock-receipt.json,
individual observed receipt JSONs, derived observation previews and analysis
PCM WAVs. Publish complete.json LAST. It records version/kind/generation ID,
source/origin/config/scope/analyzer identities, timeline fingerprint and file SHA,
clock-receipt SHA, bounded relative file catalog with size/SHA for every artifact,
creation-work counts and canonical self-hash. Latest references only this known
completed generation/marker hash, never arbitrary/model-supplied paths.

Use version1/kind observed_visual_generation for the completion protocol;
clock-receipt version1/kind observed_visual_evidence. Analyzer output version is
source-visual-local-v8-observed; SourceVisualTimeline contract_version stays1.
Companion metadata holds origin/receipt refs, requested/observed pixel/audio
associations, sampling/model/dependency identities and legacy-context hashes.
Validate exact emitted schemas, finite types, required associations and duplicates;
do not build a generic schema/transaction framework.

Internal safety ceilings:50000 total unique scheduled sample points per build,
100000 catalog artifacts,64000000bytes per JSON/artifact and16000000000 total
generation artifact bytes. Check schedule size BEFORE large np.arange/list
allocation, and cumulative dense requests/files before processing. Failure is
explicit, never silent sample reduction. These are host safety policy, not new
user/director controls or a throughput/RSS guarantee.

## Commit boundary and recovery

Under the full-build stage lock: allocate, write/validate all artifacts, atomically
publish complete.json, atomically update NEW latest.json, reconcile NEW JobStore
record. Completion is immutable content commitment; latest selects the current
generation; JobStore is advisory and cannot be globally atomic with both files.

Pre-completion failure leaves no complete marker and old latest untouched.
After complete/before latest retain COMPLETE_UNSELECTED; raise a recoverable
publication error carrying committed Path/hash and phase. After latest/before
state update retain selected data, report COMMITTED_BOOKKEEPING_FAILED with its
Path; do not generic-mark_failed as if artifacts rolled back. No deletion/hidden
rollback. A narrow explicit recovery operation validates a known committed Path
and applicability under the same lock, selects/reconciles without decode/models,
or leaves it historical. Never scan for the newest directory or mutate old flags.

## Current-origin cache and fingerprint contract

Resolve the pinned CPU pair and obtain bounded metadata/stat SourceTimeOrigin.
Resolve checks hashes only. Cache validation may use that metadata probe, but
must not qualify CPU, observe pixels/PCM, construct neural models or rebuild.
Defer qualify_render_runtime/model construction until a real miss.
With allow_model_download=False, use existing READ-ONLY model primitives
load_face_manifest/validate_face_manifest/validate_model_file to resolve cached
paths, then construct the existing backend. Do not invoke asset maintenance,
transaction recovery or abandoned-staging cleanup merely to analyze. Missing/
invalid cached assets fail explicitly. The existing explicit True download flag
may retain its setup behavior, but is never exercised by this unit's verification.
Missing new latest is a miss; old v7 is unverified legacy, not a clock upgrade. Corrupt/stale/
conflicting NEW markers/receipts/scope/config/source/artifacts fail explicitly;
do not silently rebuild over corruption. Rendering/provenance lookup never
initiates reanalysis.

New visual_provenance is provider-free and exposes:

- visual_timeline_fingerprint(timeline): EXACT current planner/director hash:
  SHA256 UTF-8 of json.dumps(timeline.model_dump(mode="json"), ensure_ascii=False,
  sort_keys=True, allow_nan=False), DEFAULT separators including spaces. Do not
  substitute compact/ASCII render_clock.canonical. Accented-text regression.
- load_visual_provenance(timeline_path, *, source_video,
  expected_timeline_fingerprint=None, current_origin=None): immutable host
  VisualProvenanceContext with status observed_samples or unverified_legacy,
  source/timeline fingerprints, scope, generation/complete/clock receipt hashes
  and to_record(). It performs bounded file/hash/association checks, NO tool/model
  resolution or subprocess. New marked generations require supplied current
  origin; missing/incomplete/conflicting companion is an error, not legacy fallback.
  An unmarked legacy artifact returns unverified_legacy, without backfill.

Catalog paths remain relative within the generation; reject traversal, absolute
paths, escaped symlinks/junction resolution, duplicate/conflicting rows and
oversized data. Compare raw receipt hashes using their own clock-envelope identity
domain; keep file SHA and model-canonical timeline fingerprint distinct.

## Observed video analyzer boundary

Replace the default OpenCV/FPS seeking route with approved observe_windows and
actual immutable BGR/PTS intervals. Coalesce requests into resource-aware bounded
batches; analysis uses max2s requested-point span and a <=3s decoder window with
.5/.25s margins. Keep existing lower-level hard caps; budget selected raw BGR by
3*source_width*source_height, reserving combined PCM and finite predecessor/
lookahead/analysis array copies before choosing batch size.
Use a separate192000000byte estimated BGR working-buffer budget for those
in-flight copies, while selected payloads+PCM also fit the decoder's existing
retained-byte cap. This estimate is NOT total measured RSS or model-allocation
coverage; do not reject a source by confusing that cap with all process memory.
Do not retain all DecodedWindow batches. Process/persist each batch then release raw pixels.
Frame timing always comes from observed integer PTS/timebase, never budgeting FPS.

Retain existing max960 INTER_AREA analysis transform and all feature/face/state
arithmetic. Existing _DecodedFrame stores features/96x54 thumbnail/histogram/
faces, not full BGR. Save generation-only JPEG90 previews from the actual analyzed
frame, recording raw BGR SHA, raw PTS/coverage, analysis resize/pixel SHA, preview
file SHA and transform separately. JPEG pixels are NOT claimed identical to the
pre-JPEG analyzer array. No existing image is relabeled as qualified.

Sample.at is requested time ONLY if the associated image interval covers it.
Do not use _decode_timestamp clamping to invent coverage. Track unavailable
attempts separately so dense passes do not retry them as missing records. When
scope contains a known first eligible video time, include and actually observe it.
No fake-zero sample for unavailable requests. Mask proven unavailable prefix/
terminal/internal intervals AFTER ordinary segmentation/finalization using
UNKNOWN/confidence0/identity UNAVAILABLE/no tracks or regions and explicit reason.
Do not convert all unsampled inference gaps to absence. If no frame is observed,
emit unknown-only scope with empty samples/tracks, never a fabricated black image.

## Analysis audio and exact sample features

Use observe_audio_window with analysis16k-mono and its continuous <=300s policy
for complete feature intervals. Group adjacent feature windows only if no feature
must be split; reject an oversize feature, never restart/concatenate resamplers
or average per-chunk RMS/percentiles. Never write/reuse legacy job.audio_path.

Persist one generation PCM WAV, with WAV-file SHA distinct from observed PCM-body
SHA; verify actual raw sample bytes/count and all block/aggregate identities.
No redundant full raw+WAV copy is required. This does not qualify historical ASR.
Transcript/diarization/speaker context hashes remain unverified_legacy (or missing);
no ASR, word shifts, profile repair or human-approval/training-pair upgrade.

One pure map_source_audio_window(receipt, origin, start, end) helper in
visual_provenance validates current receipt and declared coverage, then computes
Qp(t)=floor((t-p)*r+1/2), i=Qp(start), j=Qp(end), count=j-i, with exact Fractions.
p is observed source_phase, r/N observed rate/count. Require unrounded coverage
and0<=i<j<=N. Record requested/included bounds [p+i/r,p+j/r) and both raster deltas.
Do not independently round duration, clamp indices, invent .001s or pad. Floats
enter once via decimal strings. Valid insufficient/empty coverage has an explicit
unavailable outcome; malformed/stale receipt errors propagate, not silent silence.

Return a small immutable AudioSampleSlice with observation identity (existing
receipt envelope SHA), first/count/rate, requested/included bounds and deltas;
do not introduce another SourceClockBinding. Future consumers reuse this mapper.

Add AudioFeatureExtractor.window_samples(first, count, *, observation_id,
source_start, source_end), reading EXACT indices/complete count from immutable
generation WAV. Source bounds are the included grid bounds. Validate positive
integer indices/count, finite exact included times and count/rate duration; short
reads error. Cache binds receipt identity (therefore source/origin/phase/policy/
layout/PCM), indices and included bounds, not rounded milliseconds or PCM SHA alone.
Identical zero PCM at another phase/source cannot alias timestamps. Requested
bounds/deltas stay outside cached numerical values. Keep window(float,float)
legacy behavior/cache intact. Extract ONE existing sample-array calculator;
preserve float32,20ms grouping, percentiles, every threshold and to_dict fields.
Compare complete feature records for identical included arrays. Known visual
samples may have empty audio_features plus explicit unavailable metadata, never
zero-valued measurements invented from missing audio.

## Focused verification and handoff

TDD fake boundaries plus real numeric array/WAV/image I/O for exact features,
phase/sub-ms identity, PNG/JPEG/file-to-receipt associations, unavailable prefixes
15/59ms, unknown-only scopes, per-batch limits, complete/cache/source corruption,
force/concurrency and failures on both commit sides. Verify old artifact hashes
unchanged; bounded inspection leaves JobStore/state bytes unchanged; full build
preserves old stage values/review flags. CLI/enrollment/public Path compatibility
and runner invocation cache/work counts use fakes. No old scenario campaign.

After focused tests pass, ONE bounded real integration smoke is permitted on
data/validation_visual/candidates/01_stable_gameplay_webcam.mp4, source[1,3),
sample_interval1s, default local models/no downloads, NEW exclusive output under
.superpowers/sdd/phase3-task6a1-A-proof/. This is newly changed analyzer/receipt/
WAV integration with existing local face models, NOT LLM/API,8source or full-VOD
campaign. Do not repeat it for polish or run another actual source on failure;
retain failure and report before any new budget. Each decode <=45s, no GPU/
encoding, no original source writes. Recheck old v7/profile/PCM receipt hashes.

Scoped covering tests for TEN owned paths plus existing returned-Path CLI checks;
Ruff/compile/mypy with historical baseline disclosed, no unrelated fixes. Fresh
snapshot35 precedes edits and independent spec+quality review follows.6a1-B,
renderer/MP4/AAC, real model acceptance, full-VOD throughput and human quality
remain separate gates. No third real model/API campaign under this checkpoint.
