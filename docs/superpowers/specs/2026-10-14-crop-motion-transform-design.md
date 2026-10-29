# Faithful animated crop transforms

Historical CPU design. Production motion authority is superseded by
2026-10-14-native-gpu-crop-render-design.md after actual1080cost/finish comparison.
The human-approved full geometry objective and shared clock/preservation work
remain; this document retains the measured alternative and its rationale.

2026-10-14 America/Bogota. This is an approved Task3 implementation amendment,
not a change to the automatic vertical-shorts objective. The human explicitly
approved additional/better tools when they preserve the objective and improve
the finish. No further choice of ordinary internal filters is needed.

## Decision and scope

Keep the complete v2 independent x/y/width/height motion and explicit fill
contract. Replace the exclusive zoompan requirement with FFmpeg fixed-raster
affine perspective expressions, selected-crop edge extension, supersampling and
fixed Lanczos reduction. Production remains provider-free, deterministic and
one-encode; Python compiles numbers/graphs, not individual video images.
The existing static crop/scale path and source/output time anchors remain.
No new model field, user-facing control, editor installation or creative effect.

The measured design report is
`.superpowers/sdd/phase3-motion-transform-design.md`. Its small synthetic proofs
are design evidence, not an implemented renderer, human approval of a video or
model-produced plan. Both actual Luna campaigns remain none_passed.

## Alternatives and evidence

- zoompan cannot independently animate both crop dimensions.
- Runtime crop commands succeeded syntactically but the installed FFmpeg
  stopped emitting after an input-size change, even with scale eval=frame.
- The installed libplacebo Vulkan path crashed natively before its first frame;
  it is not a validated backend. No driver/config repair or silent fallback.
- Fixed-raster affine perspective passed108 frozen-transform comparisons and
  27 additional edge-clamped comparisons. Independent source-X error was below
  one pixel. Supersampling reduced a minified checkerboard's measured luma
  standard deviation from about51..58 to about0.5. These are specific measured
  fixtures, not a universal subjective-quality ranking or1080throughput proof.

Filtering on CPU does not change encoder selection. NVENC/libx264 selection
remains explicit; Vulkan failure says nothing about NVENC encode readiness.

## Geometry and clock

One immutable piecewise representation drives both numerical samples and graph
expressions. Interpolate x/y/width/height independently, using the departing
keyframe's easing (linear or3u²−2u³). Keep keyframe frame positions at at*60;
do not silently round original durations. Two seconds ends at frame120. A
half-open two-second shot contains frames0..119; do not lengthen it for the
endpoint test. A121-frame synthetic diagnostic may inspect the boundary.

Normalize the source branch to60FPS before motion. Preserve trim to the exact
source interval and PTS-source_start/TB. Establish a shot-local rational grid;
test fps=60:start_time=0:round=near on fractional cuts and30FPS inputs rather
than assuming it preserves offsets. Actual A/V flash/click proof stays Task7.
The installed perspective counter on is one-based: use on-1. geq N is zero-based.
All branches must receive the same frames without an intervening temporal reset.

Destination geometry retains shared even pixel edges. Moving source geometry
stays floating point. Single-keyframe static samples must truthfully report
the existing static path's quantized source crop. Validate finite values,
source bounds, minimum representability, identity/provenance and fill; never
auto-enlarge a crop or choose a new one.

## Fixed transform and antialiasing

For crop(x,y,w,h) and local destination Dw,Dh, use
s=max(Dw/w,Dh/h) for cover and s=min(Dw/w,Dh/h) for contained foreground.
Set dx=(Dw-w*s)/2,dy=(Dh-h*s)/2. This is uniform source scaling, not a
perspective aesthetic effect or anamorphic stretch.

Choose a fixed integer supersampling factor per layer/fill branch:

- Contained: S=max(1,ceil(max over all keyframes of w/Dw and h/Dh)).
- Cover: visible_width=min(w,h*Dw/Dh); S=max(1,ceil(max_visible_width/Dw)).
  Evaluate keyframes AND segment aspect crossings. Width/height are affine in
  the same monotone interpolation fraction, so their minimum's maximum occurs
  at an endpoint or crossing. The crossing fraction is
  u=(Dw/Dh*h0-w0)/((w1-w0)-Dw/Dh*(h1-h0)), when the denominator is nonzero and
  0<u<1. Include the rectangle interpolated at that fraction, without changing
  its time or creative geometry. A400x100->100x400 track into100x100 requires
  coverS3 at the middle; endpoint-onlyS1 is insufficient. ContainedS4 suffices.

This proven cover-specific bound avoids unnecessary giant work rasters while
retaining the same sample density. The simpler conservative contained bound is
also safe for cover, but must not silently waste unbounded resources. Tests
must prove the optimized bound at crossings for both easings.

Set DsW=S*Dw,DsH=S*Dh; pad the source at0,0 to the fixed work raster
W=max(source_width,DsW),H=max(source_height,DsH), with full-resolution chroma.
Use the transform equations with DsW,DsH (thus ss=S*s,dxs=S*dx,dys=S*dy):

```text
ox=x-dxs/ss+0.5/ss-0.5
oy=y-dys/ss+0.5/ss-0.5
source quad=(ox,oy),(ox+W/ss,oy),(ox,oy+H/ss),(ox+W/ss,oy+H/ss)
perspective=sense=source:eval=frame:interpolation=cubic
fixed crop=DsW:DsH:0:0:exact=1
fixed reduction=scale=Dw:Dh:flags=lanczos       (when S>1)
```

Before perspective, extend the selected crop's edge samples in FFmpeg geq:
qx=clip(X,x,x+w-1),qy=clip(Y,y,y+h-1), copying Y/U/V from qx,qy with bilinear
sampling. This prevents cubic taps from pulling unrelated content across crop
borders. Keep source bounds separate from padded dimensions.

For black/contain_blur, create the sharp foreground's alpha mask at supersampled
pixel centers: inside [dxs,DsW-dxs) by [dys,DsH-dys). Reduce color AND alpha to
fixed final dimensions, retaining extended edge colors to avoid dark fringes.
Overlay it on timed black or the blurred cover backdrop. For contain_blur,
reduce the cover backdrop BEFORE applying the approved boxblur in final pixels.
Do not change blur strength, omit the sharp image or reveal outside-crop content.

## Validation and resource behavior

Before integration acceptance, test both axes/chroma, all crop boundaries,
fractional coordinates, padding larger than the source, aspect crossings,
subpixel pan/zoom, piecewise timing and both easings across all three fills.
Frozen same-transform byte comparisons prove expression/clock fidelity;
independent coordinate ramps/colored markers prove geometry. Do not conflate
them. Supersampled black/blur alpha and minification require decoded tests.

Preserve regions through original crops AND cover-visible crops at keyframes,
midpoints, crossings and scheduled output samples after any allowed smoothing.
Keep the existing two-source-pixel tolerance; fail with shot/layer/keyframe IDs,
never auto-enlarge. Savitzky-Golay remains limited to at least five observations
and corrections below1.5% of source dimensions; preserve intentional moves and
exact endpoints, revalidating smoothed geometry.

Measure a bounded1080x1920 synthetic path with representative multiple layers,
including work-raster sizes and elapsed time. Each diagnostic process is bounded;
record timeout/failure rather than lowering quality or changing the encoder.
Working yuv444p frames cost about3WH bytes and perspective maps about8WH bytes,
before other buffers/branches. Resource failures must be explicit, not a silent
lower-S fallback. No real-time,4K or three-hour throughput claim from small tests.

Long graphs need executor-owned graph-file materialization on Windows; the pure
compiler returns text and never invents mutable output paths. Do not interpolate
user/model text or arbitrary IDs into expressions; use stable safe host labels.

## Implementation boundary

Task3 owns motion_compiler.py, test_motion_compiler.py, filtergraph.py and its
tests. Reuse the prepared snapshot30 baseline. Implement fresh from failing
tests, not by copying the exploratory scripts as production. No codec/ASS/audio/
executor/CLI/UI work, model calls, installs or original-media changes here.
Small synthetic video-only diagnostic renders are authorized for these tests;
no production MP4/product acceptance is inferred. After focused tests, scoped
Ruff/mypy/compile and a complete report, obtain independent spec+quality review.
Global typing debt, model acceptance and later phase gates remain separate.
