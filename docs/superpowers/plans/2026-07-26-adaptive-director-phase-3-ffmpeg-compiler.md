# Adaptive Visual Director Phase 3 FFmpeg Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile every validated `VisualCompositionPlan` v2 into a reproducible 1080x1920 MP4 with exact crops, layers, overlays, subtitles, audio timing, and sidecar evidence.

**Architecture:** Keep creative decisions out of the renderer. A pure compiler converts normalized geometry and source/output mappings into an inspectable FFmpeg render specification. One FFmpeg graph trims each non-contiguous source range, composes each visual shot, remaps subtitles/audio to output time, concatenates shots, and encodes once. The legacy v1 renderer remains available under an explicit command but cannot consume a v2 plan.

**Tech Stack:** Python 3.10+, Pydantic 2, ffmpeg-python, FFmpeg/FFprobe, ASS subtitles, NumPy, pytest.

## Global Constraints

- Requires Phase 1 contracts and a Phase 2 `VisualCompositionPlan` v2.
- The compiler may validate, interpolate, and encode; it may not choose layouts, crops, identities, fill modes, or effects.
- There is no `ADAPTIVE` renderer branch and no default-to-split branch for v2.
- Invalid or unsupported plan values fail before FFmpeg starts.
- All user/model text is escaped through ASS generation; never interpolate raw text into a shell command.
- Preserve source A/V sync across non-contiguous ranges to within one output frame.
- The source has one mixed audio track. Do not claim voice isolation or independent background-music control.
- Ducking applies only when an explicit audio cue is overlaid.
- Centering uses the 1080x1920 canvas center. Platform UI exclusions constrain overlays, not the content axis.
- Encoder selection is explicit and preflighted; do not silently fall back from NVENC to CPU.
- If the workspace is not a Git repository, skip commit commands and record their intended messages.

---

### Task 1: Create a render specification and strict preflight

**Files:**
- Create: `clipperx/render_contracts.py`
- Create: `clipperx/visual_compiler.py`
- Create: `clipperx/render_assets.py`
- Create: `clipperx/tests/test_render_assets.py`
- Create: `clipperx/tests/test_render_contracts.py`
- Create: `clipperx/tests/test_visual_compiler.py`
- Modify: `clipperx/diagnostics.py`

- [ ] **Step 1: Write failing preflight tests**

```python
def test_v2_compiler_rejects_missing_brand_asset(tmp_path) -> None:
    plan = plan_with_brand(asset_id="kick_brand")
    assets = RenderAssetRegistry(brand_images={}, fonts={}, sfx={})
    with pytest.raises(RenderPreflightError, match="brand"):
        preflight_visual_plan(plan, source=tmp_path / "source.mp4", assets=assets)


def test_v2_compiler_never_maps_unknown_layout_to_split() -> None:
    payload = valid_plan_dict()
    payload["shots"][0]["layers"][0]["fill"] = "adaptive"
    with pytest.raises(ValidationError):
        VisualCompositionPlan.model_validate(payload)


def test_encoder_failure_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr("clipperx.visual_compiler.available_encoders", lambda: {"libx264"})
    with pytest.raises(RenderPreflightError, match="h264_nvenc"):
        select_encoder("h264_nvenc")
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_render_contracts.py clipperx/tests/test_visual_compiler.py -q
```

Expected: missing compiler modules.

- [ ] **Step 3: Implement immutable render types**

```python
class EncoderName(str, Enum):
    H264_NVENC = "h264_nvenc"
    LIBX264 = "libx264"


class RenderProfile(BaseModel):
    width: Literal[1080] = 1080
    height: Literal[1920] = 1920
    fps: Literal[60] = 60
    video_encoder: EncoderName = EncoderName.H264_NVENC
    audio_codec: Literal["aac"] = "aac"
    audio_bitrate: Literal["192k"] = "192k"
    audio_rate: Literal[48000] = 48000
    pixel_format: Literal["yuv420p"] = "yuv420p"


class FFmpegRenderSpec(BaseModel):
    source: str
    output: str
    plan_hash: str
    command: list[str]
    filter_complex: str
    expected_duration: float
    profile: RenderProfile
    assets: list[str]
```

Add an explicit registry controlled by the host:

```python
@dataclass(frozen=True, slots=True)
class RenderAssetRegistry:
    brand_images: Mapping[str, Path]
    fonts: Mapping[str, Path]
    sfx: Mapping[str, Path]
```

The default Windows registry resolves `kick_brand` to `assets/branding/KickAthanW.png`, `headline_impact` to `C:/Windows/Fonts/impact.ttf`, and `subtitle_bold` to `C:/Windows/Fonts/arialbd.ttf`. Plans contain asset IDs only. Unknown IDs and arbitrary local paths are rejected before compilation.

`preflight_visual_plan()` checks source/media dimensions, registered assets, fonts, subtitle data, encoder availability, 179-second limit, even pixel dimensions, and that `ffmpeg`/`ffprobe` are callable. It explicitly rejects `dissolve` and any other transition not implemented by the v2 compiler.

- [ ] **Step 4: Implement explicit encoder profiles**

NVENC arguments:

```text
-c:v h264_nvenc -preset p5 -tune hq -rc vbr -cq 19 -b:v 0
```

CPU arguments:

```text
-c:v libx264 -preset medium -crf 18
```

Both use:

```text
-r 60 -pix_fmt yuv420p -movflags +faststart
-c:a aac -b:a 192k -ar 48000
```

The CLI/config must choose one profile. If unavailable, preflight fails with the exact detected encoder set.

- [ ] **Step 5: Add diagnostics**

Report FFmpeg version, `h264_nvenc` availability, `libx264` availability, GPU name when FFmpeg exposes it, registry asset loadability, and Impact/Arial Bold font resolution.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
python -m pytest clipperx/tests/test_render_contracts.py clipperx/tests/test_render_assets.py clipperx/tests/test_visual_compiler.py clipperx/tests/test_diagnostics.py -q
```

Expected: all tests pass without rendering.

- [ ] **Step 7: Commit**

```powershell
git add clipperx/render_contracts.py clipperx/render_assets.py clipperx/visual_compiler.py clipperx/diagnostics.py clipperx/tests/test_render_contracts.py clipperx/tests/test_render_assets.py clipperx/tests/test_visual_compiler.py clipperx/tests/test_diagnostics.py
git commit -m "feat: preflight visual render plans"
```

---

### Task 2: Compile deterministic crop, fill, and layer primitives

**Files:**
- Create: `clipperx/filtergraph.py`
- Create: `clipperx/tests/test_filtergraph.py`
- Modify: `clipperx/visual_compiler.py`

- [ ] **Step 1: Write failing geometry tests**

```python
def test_normalized_rect_maps_to_even_pixels() -> None:
    assert rect_to_pixels(
        NormalizedRect(x=0.1, y=0.2, width=0.5, height=0.4),
        width=1920,
        height=1080,
    ) == PixelRect(x=192, y=216, width=960, height=432)


def test_cover_centers_without_platform_offset() -> None:
    graph = compile_layer(
        source_size=(1920, 1080),
        destination=PixelRect(0, 0, 1080, 960),
        fill=FillMode.COVER,
    )
    assert "(in_w-out_w)/2" in graph
    assert "right_ui" not in graph


def test_black_fill_has_no_blur_filter() -> None:
    graph = compile_layer(
        source_size=(1920, 1080),
        destination=PixelRect(0, 0, 1080, 960),
        fill=FillMode.BLACK,
    )
    assert "color=c=black" in graph
    assert "boxblur" not in graph


def test_contain_blur_has_separate_backdrop_and_foreground() -> None:
    graph = compile_layer(
        source_size=(1920, 1080),
        destination=PixelRect(0, 0, 1080, 960),
        fill=FillMode.CONTAIN_BLUR,
    )
    assert "boxblur" in graph
    assert "force_original_aspect_ratio=decrease" in graph
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_filtergraph.py -q
```

Expected: missing `clipperx.filtergraph`.

- [ ] **Step 3: Define exact fill semantics**

- `cover`: scale up and center-crop to destination;
- `contain_blur`: blurred cover backdrop plus sharp contained foreground;
- `black`: black destination plus sharp contained foreground;
- `solid`: valid only for `LayerKind.SOLID`;
- `inherit`: resolve to the previous shot's same layer fill during preflight; fail if no predecessor exists.

All destination rectangles are clipped to the output canvas only after validation. All source crops are clipped to source bounds only for a 1-pixel rounding difference; larger overflow is an error.

- [ ] **Step 4: Compile z-ordered layers**

For each shot:

1. create a 1080x1920 black base;
2. trim a source video branch to `shot.source`;
3. compile each layer in ascending `z_index`;
4. overlay at exact destination pixels;
5. apply any shot-local title/subtitle ASS stream;
6. normalize SAR to 1 and FPS to 60.

Layer IDs and output labels must be deterministic from `shot_id` and `layer_id`, not list indexes alone.

- [ ] **Step 5: Run primitive tests**

Run:

```powershell
python -m pytest clipperx/tests/test_filtergraph.py clipperx/tests/test_visual_compiler.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/filtergraph.py clipperx/visual_compiler.py clipperx/tests/test_filtergraph.py clipperx/tests/test_visual_compiler.py
git commit -m "feat: compile explicit visual layers"
```

---

### Task 3: Compile crop tracks and stable motion

**Files:**
- Create: `clipperx/motion_compiler.py`
- Create: `clipperx/tests/test_motion_compiler.py`
- Modify: `clipperx/filtergraph.py`
- Modify: `clipperx/tests/test_filtergraph.py`
- Create: `clipperx/render_runtime.py`
- Create: `clipperx/tests/test_render_runtime.py`
- Modify: `clipperx/visual_compiler.py`
- Modify: `clipperx/tests/test_visual_compiler.py`
- Modify: `.gitignore` (controller-owned local renderer-binary exclusion)

- [ ] **Step 1: Write failing motion tests**

```python
def test_single_keyframe_compiles_to_static_crop() -> None:
    result = compile_crop_track(
        [keyframe(0, crop=(0.1, 0.1, 0.4, 0.4))],
        source_size=(1920, 1080),
        fps=60,
    )
    assert result.mode == "static"


def test_motion_expression_reaches_both_endpoints() -> None:
    track = [keyframe(0, crop=(0.0, 0.0, 0.5, 0.5)), keyframe(2, crop=(0.5, 0.5, 0.5, 0.5))]
    compiled = compile_crop_track(track, source_size=(1920, 1080), fps=60)
    assert compiled.sample(0).x == 0
    assert compiled.sample(120).x == 960


def test_subpixel_jitter_is_smoothed_but_camera_move_is_preserved() -> None:
    observations = [rect(100, 100), rect(101, 99), rect(100, 101), rect(400, 100), rect(700, 100)]
    smoothed = smooth_crop_observations(observations, source_size=(1920, 1080))
    assert max(abs(smoothed[index].x - 100) for index in range(3)) <= 2
    assert smoothed[-1].x == 700
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_motion_compiler.py -q
```

Expected: missing module.

- [ ] **Step 3: Implement deterministic interpolation**

Normalize keyframes to output-frame numbers. Apply a Savitzky-Golay filter only when there are at least five observations and the maximum correction is below 1.5% of source width/height. Preserve deliberate moves above that amount.

Static crops use `crop,scale`. Dynamic crops use the human-approved native FFmpeg9.0.2/libplacebo crop/placement design in `docs/superpowers/specs/2026-10-14-native-gpu-crop-render-design.md`, generated from piecewise keyframes with `linear` or cubic ease-in-out. This measured 2026-10-14 amendment supersedes the CPU working-raster implementation, preserving independent crop width/height and the full contract. Keep shared rational source clocks, immutable geometry and preservation; use native EWA Lanczos, explicit fill/alpha and fixed ordered dithering. Pure compilation exposes immutable requirements; explicit pinned runtime and actual bounded Vulkan preflight are separate from encoder selection. No availability-based CPU/toolchain fallback or silent quality reduction. The spec defines exact spatial/clock policy, runtime APIs and decoded tests:

```python
def ease_in_out(t: float) -> float:
    return 3 * t * t - 2 * t * t * t
```

Expression tests must verify finite values and exact first/last geometry. Do not use per-frame Python rendering.

- [ ] **Step 4: Enforce face and important-region preservation**

Before compilation, sample every crop-track keyframe and midpoint. If a `preserve_region` falls outside the crop by more than two source pixels, fail with shot/layer/keyframe IDs. The compiler does not enlarge the crop automatically.

- [ ] **Step 5: Run motion tests**

Run:

```powershell
python -m pytest clipperx/tests/test_motion_compiler.py clipperx/tests/test_filtergraph.py clipperx/tests/test_render_runtime.py clipperx/tests/test_visual_compiler.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/motion_compiler.py clipperx/filtergraph.py clipperx/render_runtime.py clipperx/visual_compiler.py clipperx/tests/test_motion_compiler.py clipperx/tests/test_filtergraph.py clipperx/tests/test_render_runtime.py clipperx/tests/test_visual_compiler.py .gitignore
git commit -m "feat: compile stable crop motion"
```

---

### Task 4: Remap and render per-shot subtitles, title, and brand

**Files:**
- Create: `clipperx/output_timeline.py`
- Create: `clipperx/overlay_ass.py`
- Create: `clipperx/tests/test_output_timeline.py`
- Create: `clipperx/tests/test_overlay_ass.py`
- Modify: `clipperx/render_assets.py` (derived brand PNG preparation in the existing asset boundary)
- Modify: `clipperx/tests/test_render_assets.py`
- Modify: `pyproject.toml` (declare already-installed FontTools for measured-face glyph coverage)

- [ ] **Step 1: Write failing timeline-remapping tests**

```python
def test_non_contiguous_source_words_map_to_contiguous_output() -> None:
    mapper = OutputTimeline.from_shots([
        shot(source=(10, 12), output=(0, 2)),
        shot(source=(20, 22), output=(2, 4)),
    ])
    assert mapper.source_to_output(20.5) == pytest.approx(2.5)


def test_subtitle_position_changes_by_shot() -> None:
    ass = build_overlay_ass(plan_with_two_subtitle_regions(), transcript)
    assert r"\pos(540,1056)" in ass
    assert r"\pos(540,1344)" in ass


def test_default_subtitle_is_chicken_yellow_with_black_outline() -> None:
    style = athan_adaptive_caption()
    assert style.primary_color == "&H001AD5FF"
    assert style.outline_color == "&H00000000"
    assert style.bold is True
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_output_timeline.py clipperx/tests/test_overlay_ass.py -q
```

Expected: missing modules.

- [ ] **Step 3: Implement exact source-to-output mapping**

Map transcript segments and word timestamps through shot ranges. Drop words outside the selected ranges. Split a transcript event when it crosses a shot boundary so each part receives the correct subtitle destination.

Use at most the configured `max_words_per_event` and `max_lines`. Implement punctuation spacing in the v2 module; do not import or modify the legacy subtitle renderer.

- [ ] **Step 4: Generate one escaped ASS file per clip**

The ASS document uses:

- PlayResX 1080 / PlayResY 1920;
- subtitle color `#FFD51A`, bold, black outline;
- `\an5\pos(x,y)` centered inside the shot subtitle destination;
- Impact title centered inside the title destination;
- title size automatically fitted from destination geometry and capped at 46 px, without an additional font-size control;
- escaped braces, backslashes, and line breaks;
- optional active-word emphasis only when requested by `SubtitlePlan`.

Accepted overlapping word times may stay in one coherent caption; emitted
captions must not collide in the planned region when packing splits them.
Preserve text/times/limits/position and explicitly reject an unsatisfiable overlap
before returning ASS; never shorten known word intervals to hide it. Establish
glyph coverage in the measured registered face with FontTools before layout;
missing Unicode cmap, absent characters or .notdef mappings fail explicitly.
Do not invent an unmeasured fallback font. Task6 preserves original font bytes,
stages before measurement and validates rendered fidelity while selected-file
identity remains explicitly unverified under the human-approved2026-10-15
`docs/superpowers/specs/2026-10-15-font-staging-and-visual-validation-design.md` amendment.
Declare `fonttools>=4.63,<5`; the current environment already has
4.63.0, so no install or global configuration change is needed.

`platform_title` and `caption` never enter ASS. The headline overlay uses only `headline.overlay_text`.

- [ ] **Step 5: Compile brand PNG as a visual layer**

Resolve `brand.asset_id` through `RenderAssetRegistry`, trim transparent padding once, preserve aspect ratio, and center it in `brand.destination`. Do not synthesize `kick.com/athanw` with drawtext when `kick_brand` is planned.

- [ ] **Step 6: Run overlay tests and existing subtitle regressions**

Run:

```powershell
python -m pytest clipperx/tests/test_output_timeline.py clipperx/tests/test_overlay_ass.py clipperx/tests/test_subtitles.py clipperx/tests/test_editing.py -q
```

Expected: all tests pass; legacy subtitle templates remain available.

- [ ] **Step 7: Commit**

```powershell
git add clipperx/output_timeline.py clipperx/overlay_ass.py clipperx/render_assets.py clipperx/tests/test_output_timeline.py clipperx/tests/test_overlay_ass.py clipperx/tests/test_render_assets.py clipperx/tests/test_subtitles.py clipperx/tests/test_editing.py pyproject.toml
git commit -m "feat: render per-shot overlays and subtitles"
```

---

### Task 5: Compile mixed-track audio and explicit cues

Technical authority: `docs/superpowers/specs/2026-10-14-audio-compilation-design.md`.
It clarifies measured shared normalization, exact sample-clock ducking and the
Task6 executor boundary, preserving the approved targets and plan decisions.

**Files:**
- Create: `clipperx/audio_mix.py`
- Create: `clipperx/tests/test_audio_mix.py`
- Modify: `clipperx/visual_compiler.py`
- Modify: `clipperx/tests/test_visual_compiler.py`

- [ ] **Step 1: Write failing audio graph tests**

```python
def test_source_audio_uses_same_trim_ranges_as_video() -> None:
    graph = compile_audio(plan_with_ranges([(10, 12), (20, 22)])).filter_complex
    assert "atrim=start=10:end=12" in graph
    assert "atrim=start=20:end=22" in graph


def test_no_cue_means_no_ducking_filter() -> None:
    # Unit fixture disables normalization and uses constant zero source gain.
    graph = compile_audio(plan_without_cues()).filter_complex
    assert "aeval=" not in graph


def test_explicit_cue_is_limited_and_ducks_source() -> None:
    # Host fixture explicitly registers impact; default registry stays cue-free.
    graph = compile_audio(plan_with_cue("impact", start=0.5, end=0.9)).filter_complex
    assert "aeval=" in graph
    assert "alimiter=limit=0.944" in graph
```

Also prove decoded sample behavior: fixed plateau dB, no outside-window
attenuation, smoothstep ramps and strongest-current overlap reduction; strings
alone do not establish ducking or compensated limiter timing.

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_audio_mix.py -q
```

Expected: missing audio compiler.

- [ ] **Step 3: Implement source audio mapping**

For every shot use matching `atrim` and `asetpts`. Concatenate in the same order as video, retaining declared clock offsets. Use one normalization-enabled source bus across visual subdivisions, then restore explicit disabled intervals. When `normalize_dialogue=true`, require a matching EBU R128 measurement before the render and compile explicitly selected measured two-pass `loudnorm` values targeting:

```text
I=-14 LUFS, LRA=11, TP=-1.5 dB
```

Apply exact `source_gain_db` after normalization, before cue duck/mix. Select and record measured linear/dynamic mode before pass two; verify observed mode. Undefined/insufficient/silent measurements have explicit typed preparation outcomes, never fabricated finite stats or a hidden fallback. Preserve targets separately from observed upstream/post-mix values.

Apply `alimiter=limit=0.944:level=0:latency=1` on maximal `limiter=true` runs, with explicit level_in/out1, attack5ms, release50ms and asc0. Visual subdivisions must not reset normalization or limiter state. Verify no auto-boost, compensated first/last samples and EOF length.

Because the source is one mixed track, label metadata as `mixed_track_normalization`, not `voice_normalization`.

- [ ] **Step 4: Implement explicit cue mixing**

For each `AudioCue`, validate the asset and relative shot times, trim/pad it to the cue interval, duck the source by `duck_source_db` only during that interval, mix, and limit. A cue cannot be added unless it exists in the v2 plan.

Use a deterministic per-sample channel-preserving envelope: smoothstep attack<=5ms and release<=10ms, each<=duration/4 and entirely inside the cue. Overlaps use the strongest currently ramped reduction, not additive dB. No cue means no duck evaluator. Explicit amix normalize0/dropout_transition0 retains authored gains; reject unqualified cue/source channel mappings. See the technical authority for equations and strict receipt/clock boundaries. Production bounded seeking and final executor assembly remain Task6.

- [ ] **Step 5: Run audio tests**

Run:

```powershell
python -m pytest clipperx/tests/test_audio_mix.py clipperx/tests/test_visual_compiler.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/audio_mix.py clipperx/visual_compiler.py clipperx/tests/test_audio_mix.py clipperx/tests/test_visual_compiler.py
git commit -m "feat: compile synchronized mixed-track audio"
```

---

### Task 6: Render v2 in one encode and preserve explicit legacy behavior

Execution amendment,2026-10-15: first execute separately reviewed6a0 (clock/
observed decode),6a0-audio-seam (shared audio-only preparation),
6a1 (new analyzer generations/provenance),6a2 (global frame/z
assembly), then6b (host executor/metrics/CLI). Binding technical authority is
`docs/superpowers/specs/2026-10-15-source-clock-and-observed-decode-design.md`;
6a0 exact ownership/steps are `.superpowers/sdd/phase3-task-6a0-brief.md`.
The audio-only prerequisite has adopted design
`docs/superpowers/specs/2026-10-15-audio-only-observation-design.md` and exact brief
`.superpowers/sdd/phase3-task-6a0-audio-seam-brief.md`; it preserves combined8s
limits and requires separate review before6a1 integration.
6a1-A now has adopted design
`docs/superpowers/specs/2026-10-15-observed-visual-index-design.md` and exact brief
`.superpowers/sdd/phase3-task-6a1-A-brief.md`; it precedes separately reviewed
6a1-B Director-image/manifest association. Public Path contracts stay unchanged.
6a1-B is now adopted under
`docs/superpowers/specs/2026-10-15-director-image-provenance-design.md` and exact
`.superpowers/sdd/phase3-task-6a1-B-brief.md`: SIX existing paths/ORIGINAL36,
one integrated review for collector/consumer compatibility, fake ordinary tests
and ONE guarded tiny synthetic HOST control outside automatic pytest collection.
No new model/API campaign or renderer/6a2 implementation is implied.
Finalize each later unit's ownership/snapshot/brief from its preceding handoff
before dispatch. The requirements below remain final acceptance requirements,
not permission to skip those prerequisites or to modify original source times.

After pure6a2 review38, the separately reviewed streaming observer prerequisite
6a3 precedes a pure compiler adapter and general6b execution. Adopted authority:
`docs/superpowers/specs/2026-10-15-streamed-source-window-design.md`; exact FOUR
clock/decoder ownership and pure/fake verification are in
`.superpowers/sdd/phase3-task-6a3-brief.md`, ORIGINAL39. Old materialized receipts
are unchanged, no real child/media/model campaign is authorized by that unit,
and actual writer/OS/encoder topology remains6b's separately qualified scope.

The pureadapter6a4 is now separately adopted afterreview39 under
`docs/superpowers/specs/2026-10-15-stream-binding-design.md` and exact
`.superpowers/sdd/phase3-task-6a4-brief.md`, ORIGINAL40. TWO compiler paths only;
no actualobserver/runtime/media/encoder campaign or6b implementation is hidden
in its metadata-only cursor/finalization joins.

After independentreview40, separately adopted6b1 isolates the lazy Windows native
call adapter under `docs/superpowers/specs/2026-10-15-windows-sync-io-adapter-design.md`,
exact `.superpowers/sdd/phase3-task-6b1-brief.md`, ORIGINAL41 absence markers. Only
TWOnew adapter/test paths, fake-native gate. No actualWindows operation, full
supervisor, observer/audio/font change or encoder campaign is hidden in this unit.

After independentreview41, adopted6b2 adds versioned float audio observation via
`docs/superpowers/specs/2026-10-15-float-audio-observer-design.md`, exact
`.superpowers/sdd/phase3-task-6b2-brief.md`, ORIGINAL42. FOURclock/decoderpaths;
old s16/combined/streamed behavior and source origin IDs stay unchanged. This
pure/fake gate does not qualify actual float decode, prepared audio, Windows
ownership, rendered timing or final outputs. All original Task6/7 gates remain.

After independentreview42, adopted6b3 adds pure exact float sample selection at
`docs/superpowers/specs/2026-10-15-float-audio-selection-design.md`, exact
`.superpowers/sdd/phase3-task-6b3-brief.md`, ORIGINAL43 absence markers. TWOnew
selection/test paths; actual schedule/Qa/42receipt+PCM validation reuse, compact
maximal-group map and actual per-group byte application. No existing decoder,
audio_mix, owner, files/bus/normalization/CLI/metrics edit or actual campaign.

After independentreview43, adopted6b4 adds original-byte font staging under
`docs/superpowers/specs/2026-10-15-original-font-stage-design.md`, exact
`.superpowers/sdd/phase3-task-6b4-brief.md`, ORIGINAL44 absence markers. TWOnew
fontstage/test paths; preserve original-byte/style/name data, exclusive private
copies+rechecked identities and fixed selected-file statusunverified. Synthetic
temporary-file gate only, no real font/render/host integration campaign.

After independentreview44, adopted6b5 adds shared hosted source ownership under
`docs/superpowers/specs/2026-10-15-hosted-source-ownership-design.md`, exact
`.superpowers/sdd/phase3-task-6b5-brief.md`, ORIGINAL45. FOURowner/decoder/test
paths; opt-in driver/jobwatchdog/perchild lifecycle for both39stream/42float.
Pure/fake gate only, no actualWindows/source/transport/render qualification and
no original default/receipt/policy/limit change. Later OS gate remains separate.

**Files:**
- Modify: `clipperx/plan_render.py`
- Modify: `clipperx/cli.py`
- Create: `clipperx/render_metrics.py`
- Create: `clipperx/tests/test_render_metrics.py`
- Create: `clipperx/tests/test_plan_render_v2.py`
- Modify: `clipperx/tests/test_plan_render.py`

- [ ] **Step 1: Write failing renderer-routing tests**

Write these exact routing tests:

- `test_render_visual_plan_routes_once_to_v2_compiler_and_never_to_legacy`;
- `test_v2_renderer_import_graph_excludes_face_editing_and_legacy_subtitles`;
- `test_explicit_legacy_entry_still_accepts_a_v1_fixture_plan`.

- [ ] **Step 2: Confirm the current regressions**

Run:

```powershell
python -m pytest clipperx/tests/test_plan_render_v2.py clipperx/tests/test_plan_render.py clipperx/tests/test_editing.py -q
```

Expected before implementation: v2 routing is missing. Existing v1 behavior is characterized but remains isolated and is not imported by the new compiler.

- [ ] **Step 3: Implement explicit render entry points**

Implement these exact entry points:

- `render_visual_composition_plan(source_video: Path, plan_path: Path, output_dir: Path, *, transcript_path: Path | None = None, assets: RenderAssetRegistry, profile: RenderProfile | None = None) -> list[Path]`
- `render_legacy_composition_plan(source_video: Path, plan_path: Path, output_dir: Path, *, transcripts: Sequence[TranscriptSegment] | None = None, preset: str = "default", watermark_text: str | None = None) -> list[Path]`

The CLI becomes:

```text
clipperx render-visual-plan VIDEO PLAN --output-dir DIR
    [--transcript FILE] [--encoder h264_nvenc|libx264]
clipperx render-legacy-plan VIDEO PLAN --output-dir DIR
```

Keep `render-plan` as a deprecation error that tells the user which explicit command matches the contract version; do not guess from malformed JSON.

- [ ] **Step 4: Enforce import and dispatch isolation**

The v2 entry point uses the provider-free compiler/assets/overlay/audio plus the amended source-clock/runtime/metrics dependencies. It must not import legacy editing, subtitles, face, providers, model runtimes or settings by module initialization. Keep legacy module behavior and v1 subtitle mapping values unchanged; the explicit legacy entry remains the only caller of that path.

- [ ] **Step 5: Persist reproducibility sidecars**

For each MP4 write:

```text
<clip>.visual-plan.json
<clip>.render-spec.json
<clip>.metadata.json
<clip>.render-metrics.json
<clip>.ffmpeg.log
```

Metadata includes source fingerprint, plan hash, model/style card IDs from the plan manifest, FFmpeg version, encoder/profile, duration, A/V stream probe, loudness measurements, and render elapsed time.

`render-metrics.json` records every resolved source crop and output box by `shot_id`/`layer_id`, actual overlay boxes, safe/forbidden regions, video/audio start and duration, FPS, pixel format, and plan-versus-render deltas. Phase 4 consumes these measured values and must not reconstruct them.

- [ ] **Step 6: Run routing and regression tests**

Run:

```powershell
python -m pytest clipperx/tests/test_plan_render_v2.py clipperx/tests/test_render_metrics.py clipperx/tests/test_plan_render.py clipperx/tests/test_editing.py clipperx/tests/test_cli.py -q
```

Expected: all tests pass and v2 never enters legacy editing functions.

- [ ] **Step 7: Commit**

```powershell
git add clipperx/plan_render.py clipperx/render_metrics.py clipperx/cli.py clipperx/tests/test_plan_render_v2.py clipperx/tests/test_render_metrics.py clipperx/tests/test_plan_render.py clipperx/tests/test_cli.py
git commit -m "feat: render visual plans with deterministic compiler"
```

---

### Task 7: Add synthetic and real render verification

**Files:**
- Create: `clipperx/tests/media_factory.py`
- Create: `clipperx/tests/test_visual_render_integration.py`
- Create: `scripts/render_visual_smoke.py`
- Create: `data/validation_visual/manual_plans/`
- Modify: `docs/style-corpus-audit.md`

- [ ] **Step 1: Generate a portable synthetic source**

The fixture generator uses FFmpeg lavfi to create:

- 1920x1080 color/gameplay-like background;
- a labeled top-left webcam rectangle;
- timestamp and moving marker;
- stereo sine/audio pulse;
- 12-second H.264/AAC MP4.

It must not store a binary fixture in Git.

- [ ] **Step 2: Write integration assertions**

Render a v2 plan containing:

- two non-contiguous source ranges;
- content plus enlarged webcam;
- a full-camera shot;
- black and contain-blur fills;
- title, brand, and two subtitle positions;
- one crop move;
- one audio cue.

Assert with FFprobe and decoded frames:

```python
assert width == 1080 and height == 1920
assert pixel_format == "yuv420p"
assert abs(duration - plan.duration) <= 1 / 60 + 1024 / 48000
assert audio_sample_rate == 48000
assert first_source_marker == expected_first_marker
assert second_source_marker == expected_second_marker
assert overlays_respect_forbidden_regions(render_metrics) is True
```

- [ ] **Step 3: Run synthetic render integration**

```powershell
python -m pytest clipperx/tests/test_visual_render_integration.py -q
```

Expected: one MP4 and five sidecars; all assertions pass.

- [ ] **Step 4: Author three manual real-window v2 plans**

Use local source mappings for:

1. stable gameplay plus webcam;
2. full camera;
3. one visual state transition.

The plans use timeline evidence and no LLM. Store portable plan templates under `data/validation_visual/manual_plans/`; store absolute source mappings only in ignored local configuration.

- [ ] **Step 5: Render the real smoke set**

```powershell
python scripts/render_visual_smoke.py --sources data/validation_visual/local_sources.json --plans data/validation_visual/manual_plans --output data/validation_visual/renders/phase3 --encoder h264_nvenc
```

Expected: three playable 1080x1920 MP4 files, no face crop violation reported by preflight, and FFprobe duration error below 40 ms.

- [ ] **Step 6: Record measured evidence**

Append exact source windows, plans, render time, encoder, output probes, and unresolved visual issues to `docs/style-corpus-audit.md`.

- [ ] **Step 7: Final phase verification**

Run:

```powershell
python -m pytest clipperx/tests -q
python -m mypy clipperx
python -m ruff check clipperx scripts
python -m compileall clipperx scripts
```

Expected: all commands exit 0.

- [ ] **Step 8: Commit**

```powershell
git add clipperx/tests/media_factory.py clipperx/tests/test_visual_render_integration.py scripts/render_visual_smoke.py data/validation_visual/manual_plans docs/style-corpus-audit.md
git commit -m "test: verify deterministic visual renders"
```

## Phase 3 Exit Gate

Phase 4 may start only when:

- v2 plans never fall through legacy layout logic;
- non-contiguous source ranges remain frame-synchronized with audio;
- fill modes produce visibly distinct, tested graphs;
- titles, PNG brand, and subtitles obey per-shot geometry;
- the compiler emits H.264/AAC 1080x1920 `yuv420p` with faststart;
- synthetic integration and three real-window renders pass;
- every output has plan, render spec, render metrics, metadata, and FFmpeg log sidecars.
