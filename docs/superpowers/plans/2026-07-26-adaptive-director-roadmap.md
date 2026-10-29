# Adaptive Visual Director Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved adaptive stream-editing architecture in four independently testable phases without breaking the existing narrative planner or pretending that historical edited clips are paired training data.

**Architecture:** The v1 editorial planner remains the source of narrative ranges. A provider-free source timeline and identity index feed a stateless visual director. The director emits a strict v2 plan; a separate one-pass FFmpeg compiler renders it; local/VL QA may patch failed shots at most twice. User approval creates the first genuine source-result pairs.

**Tech Stack:** Python 3.10+, Pydantic 2, OpenCV YuNet/SFace, SciPy, sentence-transformers, OpenAI-compatible structured outputs, FFmpeg/FFprobe, ASS, Typer, Textual, pytest.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-07-26-adaptive-source-final-director-design.md`.
- Implementation order is strict: Phase 1 → Phase 2 → Phase 3 → Phase 4.
- A phase cannot start until the prior phase exit gate has current verification evidence.
- Do not process a full two-to-six-hour VOD with a model during implementation validation.
- Do not enrich all 301 style clips until a three-case model benchmark passes.
- V1 and v2 remain separate. V2 never imports the legacy face/layout/subtitle renderer.
- Source paths outside the workspace are read-only inputs; generated artifacts remain under `data/processed`, `data/model_validation`, or `data/qa_scenarios`.
- No Git repository is currently operational in this workspace. If that remains true during execution, record intended commit boundaries without fabricating commits.

---

### Task 1: Execute provider-free foundations

**Plan:** `docs/superpowers/plans/2026-07-26-adaptive-director-phase-1-foundations.md`

- [ ] Implement strict source-timeline and v2 visual contracts while preserving all v1 models.
- [ ] Install and validate one unambiguous OpenCV runtime plus official YuNet/SFace assets.
- [ ] Track faces and webcam rectangles independently.
- [ ] Keep ambiguous streamer identity explicit.
- [ ] Build sparse-to-dense source states without an LLM.
- [ ] Pass synthetic tests and all ten real source scenarios.

Gate output:

```text
<job>/source_visual_timeline.json
<job>/face_tracks.json
<job>/streamer_identity.json
<job>/webcam_tracks.json
data/validation_visual/runs/phase1/
```

---

### Task 2: Execute style retrieval and visual direction

**Plan:** `docs/superpowers/plans/2026-07-26-adaptive-director-phase-2-style-director.md`

- [ ] Extract objective evidence from historical edited clips.
- [ ] Build StyleCards with relative references and no invented VOD pairing.
- [ ] Retrieve two or three compatible references per source state.
- [ ] Keep all model calls stateless and limited to local transcript/state slices.
- [ ] Generate a complete `VisualCompositionPlan` v2 after persisted narrative Composition.
- [ ] Benchmark a small real model set on exactly three cases and select the cheapest passing model.

Model-call envelope per visual-director request:

```text
source images: <= 5
StyleCard images: <= 3
total images: <= 8
transcript: intersecting range plus 2 s padding, <= 8000 characters
follow-ups: <= 2 per editorial segment
calls: <= 6 per clip
```

Gate output:

```text
data/processed/style_library/
<job>/planning/runs/<run-id>/composition_plan.json
<job>/planning/runs/<run-id>/visual/visual-plan-manifest.json
<job>/planning/runs/<run-id>/visual/<clip-id>.visual-plan.v2.json
data/model_validation/visual-director-benchmark/
```

---

### Task 3: Execute deterministic one-pass rendering

**Plan:** `docs/superpowers/plans/2026-07-26-adaptive-director-phase-3-ffmpeg-compiler.md`

- [ ] Resolve only host-registered brand/font/SFX asset IDs.
- [ ] Compile normalized crops, layers, fill modes, overlays, motion, and audio.
- [ ] Remap non-contiguous source transcript/audio to contiguous output.
- [ ] Encode once at 1080x1920, 60 FPS, H.264/AAC, `yuv420p`, faststart.
- [ ] Preserve A/V alignment within one frame plus one AAC packet.
- [ ] Keep legacy editing, face detection, and subtitles outside the v2 import path.
- [ ] Pass synthetic marker synchronization and three real-window renders.

Gate output per clip:

```text
<clip>.mp4
<clip>.visual-plan.json
<clip>.render-spec.json
<clip>.render-metrics.json
<clip>.metadata.json
<clip>.ffmpeg.log
```

---

### Task 4: Execute post-render QA, corrections, and user workflow

**Plan:** `docs/superpowers/plans/2026-07-26-adaptive-director-phase-4-qa-ui.md`

- [ ] Check media, decoding, black runs, timing, and measured safe-zone geometry locally.
- [ ] Review bounded rendered frame sets with the selected VL model.
- [ ] Permit only failed-shot replacements and at most two correction renders.
- [ ] Mark remaining failures `needs_review`; never hide them behind another fallback.
- [ ] Expose the `Athan adaptativo` storyboard and scoped actions in CLI/TUI.
- [ ] Run all ten real scenarios.
- [ ] Save a source-result pair only after explicit user approval.

Final statuses remain distinct:

```text
planned
rendered
local_qa_passed
vl_qa_passed
needs_review
user_approved
```

---

### Task 5: Verify the integrated adaptive path

**Files:**
- Modify during execution: `README.md`
- Modify during execution: `docs/style-corpus-audit.md`
- Create during execution: `data/qa_scenarios/runs/<run-id>/`

- [ ] **Step 1: Run the complete default regression**

```powershell
python -m pytest clipperx/tests -q
python -m mypy clipperx
python -m ruff check clipperx scripts
python -m compileall clipperx scripts
python -m clipperx.cli doctor --json
```

Expected: all commands exit 0. A sandbox permission failure is recorded as blocked verification, not as a passing test.

- [ ] **Step 2: Run FFmpeg E2E**

```powershell
python -m pytest clipperx/tests -m e2e -q
```

Expected: synthetic non-contiguous visual/audio markers align within `1/60 + 1024/48000` seconds.

- [ ] **Step 3: Run the ten local source scenarios**

```powershell
python scripts/run_qa_scenarios.py --manifest data/qa_scenarios/manifest.json --sources data/qa_scenarios/local_sources.json --output data/qa_scenarios/runs/latest
```

Expected: ten explicit outcomes with input, timeline, plan, MP4, and QA artifacts.

- [ ] **Step 4: Run a bounded real-model contract test**

```powershell
python scripts/run_qa_scenarios.py --manifest data/qa_scenarios/manifest.json --sources data/qa_scenarios/local_sources.json --output data/qa_scenarios/runs/model-review --vl --model-from data/model_validation/selected_visual_model.json --confirm-inference --limit 3
```

Expected: exactly three scenarios, valid structured responses, usage recorded, and no full VOD/transcript payload.

- [ ] **Step 5: Obtain user acceptance**

Present the absolute MP4 paths. User feedback covers framing, context boundaries, visual priority, overlays, fill choice, cuts, and retained gaps. Store approved pairs; retain rejected renders and reasons as regression evidence without treating them as good targets.

- [ ] **Step 6: Record truthful readiness**

Update README and the audit with:

- installed/configured/tested status for local models and provider;
- exact unit/E2E/model results;
- scenario pass/fail/ambiguous counts;
- render times and model usage;
- number of corrections;
- user-approved pair IDs;
- unresolved limitations.

## Final Acceptance Gate

ClipperX is ready for launch in the adaptive mode only when:

- one complete input VOD reaches MP4 outputs through the v2 path;
- no visual decision falls through a legacy default;
- identity, camera movement, multi-person, no-camera, browser, reacted-video, and non-contiguous-range cases have real evidence;
- the selected model passes its vision/schema contract;
- compiler and QA gates pass on the target RTX 3060 system;
- the user approves at least one rendered result;
- the interface clearly exposes failures and does not label mocked or skipped checks as operational.
