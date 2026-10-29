# Native GPU crop rendering and explicit runtime

2026-10-14 America/Bogota. Current Task3 implementation authority. The human
approved better/additional tools preserving the objective and finish; this
amendment replaces the earlier CPU working-raster motion implementation, not
the full v2 geometry/identity/narrative contract. No repeated user choice of
internal filters is needed.

## Decision, evidence and boundaries

Moving crops use the measured FFmpeg9.0.2/libplacebo native crop/placement path.
Shared immutable motion geometry, easing, constrained SG smoothing, strict input
validation, preservation and rational source clocks remain. Static crop/scale
primitives remain as reviewed in Task2. No per-frame Python image rendering.
The slow CPU motion prototype and its known alpha/color defect are archived,
not accepted as an automatic fallback. Do not present replacement as a measured
fix of that prototype or relax its failed assertions to claim it passed.

Read `.superpowers/sdd/phase3-gpu-motion-candidate-design.md` for exact evidence.
Final-policy decoded moving/frozen checks, fractional edges/alpha, color identity,
linear-tagged minification and16-bit independent chroma geometry support this
choice. Representative3frame/null2.516s versusCPU20.835s is an8.28x LIFECYCLE
comparison only;30frames/15.172s is about1.98fps including startup, not realtime.
HDR, encoded subsampled color and long-stream throughput remain unaccepted.

## Pure compiler and graph

Use the existing CompiledCropTrack representation as the only numerical and
expression authority. Add the controlled host frame symbol n (zero-based) to
the serializer. All four crop components vary independently. Segment easing
comes from its departing keyframe. No crop enlargement, discarded movement,
aspect reinterpretation, duration/source-range change or new quality knob.

For source crop(x,y,w,h), local destinationDw,Dh:

```text
s=max(Dw/w,Dh/h) for cover; min(Dw/w,Dh/h) for contained foreground
pw=w*s; ph=h*s; px=(Dw-pw)/2; py=(Dh-ph)/2
libplacebo=w=Dw:h=Dh:crop_x=x:crop_y=y:crop_w=w:crop_h=h:
          pos_x=px:pos_y=py:pos_w=pw:pos_h=ph:reset_sar=1
```

Black is an opaque black native background. Contain-blur splits the same source
into native cover backdrop and transparent contained sharp foreground. Convert
the backdrop to yuv420p and apply the approved boxblur radii/power at FINAL
dimensions, then overlay straight-alpha sharp foreground. No extra source in
letterbox areas and no separate generated CPU pixel mask/supersampling raster.

Image policy is fixed and host-owned: upscaler/downscaler=ewa_lanczos,
disable_linear=0, sigmoid=0, dithering=ordered_fixed, dither_temporal=0,
fps=none, frame_mixer=none, apply_filmgrain=0, apply_dolbyvision=0, deband=0,
peak_detect=0, tonemapping=clip, gamut_mode=clip, contrast_recovery=0; no LUTs,
custom shaders, grading or temporal effects. These settings do not certify HDR.
Preserve known transfer/primaries/matrix/range; never relabel unknown as proved.
Native linear-light resampling need not equal averaging nonlinear code values.
Use LINEAR-tagged numerical averaging oracles separately from tagged identity
and ordinary SDR video tests. Known PQ/HLG inputs must fail actionable preflight
for ALL v2 plans, including static-only basic plans, until separately qualified,
rather than silently tone-map them. This common input guard is the sole
clarification to otherwise unchanged static geometry/runtime behavior. Unknown tags
remain explicitly unknown in qualification/metadata; do not invent BT709 tags.

Common STATIC/DYNAMIC source clock is now tested:
settb=1/60000000 BEFORE exact original trim, setpts=PTS-source_start/TB,
fps=60:start_time=0:round=near, then shot-local trim=end=original_duration before
split. Preserve source/output/shot-local domains and exact declared times.
Two seconds reaches boundary frame120; a real half-open shot contains0..119.
No second temporal resampler inside libplacebo. Preserve all shared clock tests.

Check source crops AND cover-visible crops at keyframes/midpoints/crossings and
scheduled samples after smoothing. Keep two-source-pixel preservation tolerance
and shot/layer/keyframe error identity. Use native edge handling only AFTER plan
validation; successful sampling cannot authorize a lost region or arbitrary crop.

## Runtime APIs and purity

Add provider-free `clipperx/render_runtime.py`. It owns immutable requirements,
host-only executable selection/qualification and bounded inspection, not
creative choices or model settings. No import-time GPU/process/file probing,
PATH mutation, downloading, global state selection or credentials/config import.

Expose frozen/slotted RenderRequirements with runtime_id (system or
ffmpeg-9.0.2-libplacebo), needs_vulkan, and detached immutable required_filters.
Expose pure requirements_for_plan(plan) and combine_render_requirements(...).
Moving tracks require the pinned native runtime; static-only plans retain the
basic system-runtime behavior. This is deterministic from the explicit graph,
NOT availability-based fallback. Add a defaulted requirements field to frozen
CompiledShotLayers, retaining its existing positional fields and JSON-friendly
handoff. Compilation may only return graph/requirements; never initialize GPU.

Expose immutable RenderRuntime (paired executable paths, runtime identity,
observed/pinned hashes, graph-file option and explicit Vulkan device selection)
and resolve_render_runtime(requirements, *, device_index=0). Host callers may
provide an explicit device index; no model/UI field or automatic device retry.
Expose qualify_render_runtime(runtime, requirements) returning immutable actual
version/filter/device/probe evidence. Preserve unknown adapter details as unknown.
The exact safe field spelling can follow repository patterns; document it in
the report, but no mutable returned metadata aliases.

Pinned runtime is ALREADY provisioned by controller at
data/tools/ffmpeg/9.0.2/bin/ffmpeg.exe and ffprobe.exe, with LICENSE/README beside
bin. Exact hashes:

```text
ffmpeg: 589E50B766D251AFDF181DD664D40BD94407E200B019989FD7468C7D118A28D0
ffprobe:02264DFA4F56BAAA31E443ECE8AA16D397F6B008200E7FD184D22B9009E75FD3
```

Resolve against provider-free local_paths base_dir, not an absolute machine path.
Check both pins before native invocation. No installer/network activity from
imports/preflight; missing/mismatched bundle is an actionable error, not system
FFmpeg fallback. Existing C:/ffmpeg and legacy v1 stay untouched.

## Preflight and executor handoff

Extend existing visual_compiler preflight/inspection helpers with optional
explicit runtime parameters while preserving basic no-argument behavior.
Maintain asset/subtitle/unsupported-plan/source-fingerprint checks before tools.
Resolve the plan's runtime, inspect versions/encoders/ffprobe using THAT paired
runtime, validate stream spans/source dimensions, and run pure geometry/graph
validation before any synthetic GPU probe. Native requirements include the
actual filter/option set; qualification must execute a bounded tiny HOST lavfi
Vulkan filter frame, not merely enumerate filters. Fail device/native/format/
timeout/HDR cases explicitly. Never silently switch encoder, GPU device, quality,
toolchain or renderer. Video encoder availability remains separate from actual
encoding acceptance, which belongs later.

The qualified native runtime supplies `-init_hw_device vulkan=vk:<index>` and
`-filter_hw_device vk` BEFORE inputs, and `-/filter_complex` for host-written
graph files. FFmpeg9 removed -filter_complex_script. Pure compilers return text
and immutable requirements; Task6 owns graph-file creation/command execution,
metadata/sidecars and measured encoder tests. Do not hide runtime selection in a
mutable singleton or pass user/model text into shell/filter expressions.

## Ownership and tests

Task3 now owns motion_compiler.py, filtergraph.py and their tests, new
render_runtime.py/test_render_runtime.py, plus visual_compiler.py and its tests.
Snapshot30 has the original four-path base plus pre-integration compiler/test
copies; new runtime files have ABSENT markers. Preserve the shared checkpoint
before replacing CPU backend-specific code/tests. Archive remains in
.superpowers/sdd/checkpoints/phase3-task3-shared-20261014; no history deletion.

TDD: runtime requirements/purity/pin errors/device failures; native independent
crop dimensions/expressions and frozen metadata; actual production-policy
decoded fills/motion/easing/piecewise/keyframe/edge/alpha/minification oracles;
both-axis/chroma coordinate evidence and known/unknown tag handling; shared
fractional30FPS/nonzeroPTS clocks through native output. Use an appropriate
high-bit-depth oracle to distinguish8-bit intensity quantization from geometry,
never widen physical preservation tolerance. Assert alpha0 outside foreground
and source-content independence, not an unexplained relaxed black assertion.
Retire CPU-specific implementation assertions in favor of equivalent-or-stronger
new-backend behavior tests, preserving reported CPU failures in the archive.

Update the known stale visual_compiler test expecting every motion to be rejected:
assert real compiled motion/requirements instead, not test deletion or omission.
No fake GPU test pass via skips on this qualified local host. Use bounded small
synthetic controls; report actual current1080cost once on final implementation.
No old backend loops, original-media rendering, audio/ASS/executor/UI changes,
model calls, new installs or full repository suite in this task. Final focused
motion/filtergraph/runtime/compiler gate, scoped Ruff/mypy/compile, exact report
and independent spec+quality review are mandatory. Whole-repo typing, model
acceptance, real MP4/A/V/QA/human approval remain separate open gates.

## Bounded coverage partition after implementation telemetry

The initial strongly minifying121frame12-case Cartesian diagnostic exhausted
45s per child while emitting81frames; this proves progress/cost, not a geometry
failure or successful complete matrix. Its owned batch was cancelled, preserving
completed timeout records. Production quality, source times, duration and final
1080output profile are unchanged.

Use the complete121frame piecewise/easing/three-fill geometry matrix at200x200
component destinations (no strong minification), plus separately labelled small
ANIMATED strong-minification cases for allthree fills with frozen references,
checker/chroma/alpha checks. The latter do not substitute for the full2sec/frame120
clock proof or pretend to be a121frame minification matrix. Allow ONE complete
121frame strongly minifying cover control up to90s, justified by measured81frame
progress at45s. This changes only the diagnostic wait budget; preserve all
geometry/color/preservation thresholds. Tool waits/polls stay<=45s. If it fails,
report the failure without another timeout loop. Final focused suite must run
all retained tests, without exclusions/skips hiding unqualified paths. Report
the complementary coverage groups and their exact sizes, not a merged claim.
