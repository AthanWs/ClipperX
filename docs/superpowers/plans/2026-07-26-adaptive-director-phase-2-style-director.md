# Adaptive Visual Director Phase 2 Style and Direction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert historical edited clips into a searchable style library and generate complete `VisualCompositionPlan` v2 documents for narrative clips without treating unrelated VODs and edits as supervised pairs.

**Architecture:** Local analyzers create objective clip descriptors and representative keyframes. An optional provider-neutral structured model enriches them into `StyleCard` records. Retrieval first enforces scene compatibility, then ranks style similarity and recency. A stateless visual director consumes one editorial clip, its local source timeline, selected frames, and two or three StyleCards; it either returns a valid v2 plan or a bounded evidence request.

**Tech Stack:** Python 3.10+, Pydantic 2, existing OpenAI-compatible structured client, OpenCV, FFprobe, sentence-transformers, NumPy, Typer, pytest.

## Global Constraints

- Requires the Phase 1 exit gate in `docs/superpowers/plans/2026-07-26-adaptive-director-phase-1-foundations.md`.
- Historical edited clips are style references only. Never infer their original source timestamps.
- `visual_quality` and `narrative_quality` are distinct fields; unknown narrative quality stays `null`.
- The model must never receive a full two-to-six-hour transcript or video.
- Every model call is self-contained and includes all required IDs, local transcript excerpt, schema, frames, and constraints.
- No server-side chat/session state is assumed for Ollama, CLIProxyAPI, or another provider.
- Model inference remains opt-in through `--confirm-inference`.
- The director may subdivide an editorial segment on visual events but may not add, delete, or reorder narrative source ranges.
- A missing or invalid plan is an error. Do not map it to `SPLIT`, `ADAPTIVE`, or another legacy layout.
- If `git rev-parse --is-inside-work-tree` is false, record intended commit messages and skip commit commands.

---

### Task 1: Define StyleCard, retrieval, and director-response contracts

**Files:**
- Create: `clipperx/style_contracts.py`
- Create: `clipperx/tests/test_style_contracts.py`
- Modify: `clipperx/visual_contracts.py`
- Modify: `clipperx/tests/test_visual_contracts.py`

- [ ] **Step 1: Write failing StyleCard validation tests**

```python
from pydantic import ValidationError
import pytest

from clipperx.style_contracts import (
    BackgroundTreatment,
    SceneFamily,
    StyleCard,
    StyleKeyframe,
)


def test_style_card_does_not_require_a_fake_source_pair() -> None:
    card = StyleCard(
        card_id="style-1",
        clip_fingerprint="abc",
        scene_family=SceneFamily.GAMEPLAY_WITH_CAMERA,
        keyframes=[StyleKeyframe(timestamp=1.0, path="frames/1.jpg")],
        background_treatment=BackgroundTreatment.BLACK,
        visual_quality=0.9,
        narrative_quality=None,
        description="Gameplay arriba, cámara amplia abajo, espacios negros.",
    )
    assert not hasattr(card, "source_vod_timestamp")


def test_style_card_requires_visual_evidence() -> None:
    with pytest.raises(ValidationError):
        StyleCard(
            card_id="empty",
            clip_fingerprint="abc",
            scene_family="unknown",
            keyframes=[],
            background_treatment="black",
            visual_quality=0.5,
            description="sin evidencia",
        )
```

- [ ] **Step 2: Confirm the tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_style_contracts.py -q
```

Expected: import failure for `clipperx.style_contracts`.

- [ ] **Step 3: Implement complete style contracts**

Public enums:

```python
class SceneFamily(str, Enum):
    GAMEPLAY_WITH_CAMERA = "gameplay_with_camera"
    FULL_CAMERA = "full_camera"
    BROWSER_WITH_CAMERA = "browser_with_camera"
    REACTED_VIDEO_WITH_CAMERA = "reacted_video_with_camera"
    CONTENT_ONLY = "content_only"
    MULTI_PERSON = "multi_person"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class BackgroundTreatment(str, Enum):
    COVER = "cover"
    CONTAIN_BLUR = "contain_blur"
    BLACK = "black"
    SOLID = "solid"
    MIXED = "mixed"
```

`StyleCard` must include:

```python
contract_version: Literal[1] = 1
card_id: str
clip_fingerprint: str
source_clip_ref: str  # relative to the style-library manifest
duration: float
created_at: datetime | None
scene_family: SceneFamily
keyframes: list[StyleKeyframe]  # 1..8
camera_regions: list[NormalizedRect]
content_regions: list[NormalizedRect]
background_treatment: BackgroundTreatment
title_regions: list[NormalizedRect]
brand_regions: list[NormalizedRect]
subtitle_regions: list[NormalizedRect]
subtitle_palette: list[str]
cut_rate_per_minute: float
zoom_count: int
special_techniques: list[str]
visual_quality: float
narrative_quality: float | None
approval: Literal["approved", "unreviewed", "rejected"]
description: str
evidence: list[str]
```

The model-facing output is a smaller contract with no host-controlled identity or paths:

```python
class StyleCardAnalysisResponse(BaseModel):
    contract_version: Literal[1] = 1
    scene_family: SceneFamily
    camera_regions: list[NormalizedRect]
    content_regions: list[NormalizedRect]
    background_treatment: BackgroundTreatment
    title_regions: list[NormalizedRect]
    brand_regions: list[NormalizedRect]
    subtitle_regions: list[NormalizedRect]
    subtitle_palette: list[str]
    special_techniques: list[str]
    description: str
    evidence_refs: list[str]
```

Add `VisualEvidenceRequest` and `VisualDirectorResponse` to `visual_contracts.py`:

```python
class VisualDirectorStatus(str, Enum):
    READY = "ready"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    REVIEW_REQUIRED = "review_required"


class VisualEvidenceRequest(BaseModel):
    request_id: str
    clip_id: str
    source: TimeRange
    question: str
    sample_interval_seconds: float = Field(ge=0.25, le=2.0)
    reason: Literal[
        "identity_ambiguous",
        "camera_boundary",
        "important_region",
        "state_transition",
        "crop_motion",
    ]
    iteration: int = Field(ge=1, le=3)


class VisualDirectorResponse(BaseModel):
    contract_version: Literal[1] = 1
    status: VisualDirectorStatus
    plan: VisualCompositionPlan | None = None
    evidence_requests: list[VisualEvidenceRequest] = Field(default_factory=list)
    style_card_ids: list[str] = Field(default_factory=list)
    reasoning: str
```

Validator rules:

- `ready` requires a plan and forbids evidence requests;
- `need_more_evidence` requires one to three requests and forbids a plan;
- `review_required` forbids a plan;
- every request window is inside one of the editorial source ranges expanded by at most 15 seconds;
- the same request signature cannot repeat at a later iteration.
- `source_clip_ref` and every keyframe ref are relative, cannot escape the library root, and cannot contain a drive or URI scheme;
- `StyleCardAnalysisResponse` contains no card ID, source path, VOD association, approval, or narrative-quality field;

- [ ] **Step 4: Run contract tests**

Run:

```powershell
python -m pytest clipperx/tests/test_style_contracts.py clipperx/tests/test_visual_contracts.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add clipperx/style_contracts.py clipperx/visual_contracts.py clipperx/tests/test_style_contracts.py clipperx/tests/test_visual_contracts.py
git commit -m "feat: define style and visual director contracts"
```

---

### Task 2: Extract objective style evidence from edited clips

**Files:**
- Create: `clipperx/style_analysis.py`
- Create: `clipperx/tests/test_style_analysis.py`
- Modify: `clipperx/jobs.py`
- Modify: `clipperx/config.py`

- [ ] **Step 1: Write failing local-analysis tests**

```python
def test_black_bars_are_not_reported_as_blur() -> None:
    frames = synthetic_black_panel_frames()
    result = analyze_style_frames(frames)
    assert result.background_treatment == BackgroundTreatment.BLACK


def test_chicken_yellow_subtitle_palette_is_detected() -> None:
    frame = frame_with_subtitle(rgb=(255, 213, 26), outline=(0, 0, 0))
    result = analyze_style_frames([frame])
    assert "#FFD51A" in result.subtitle_palette


def test_analysis_keeps_visual_and_narrative_quality_separate() -> None:
    result = analyze_edited_clip(fake_clip, narrative_quality=None)
    assert result.visual_quality is not None
    assert result.narrative_quality is None
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_style_analysis.py -q
```

Expected: import failure for `clipperx.style_analysis`.

- [ ] **Step 3: Implement deterministic evidence extraction**

`analyze_edited_clip(path)` must:

1. fingerprint and probe the MP4;
2. sample the first frame, last frame, scene changes, and evenly spaced frames up to eight;
3. detect black/solid/blurred background regions;
4. estimate title, subtitle, and brand regions from temporal text-like overlays;
5. detect stable camera/content rectangles;
6. estimate cut rate, zoom/pan events, and overlay persistence;
7. save representative JPEGs and `style_evidence.json`.

Do not classify humor, virality, or narrative quality locally. Objective quality covers clipping, unreadable overlays, severe face cuts, black-frame errors, and visual consistency.

Cache under:

```text
data/processed/style_library/evidence/<clip-fingerprint>/
```

and include analyzer version plus input fingerprint.

- [ ] **Step 4: Run analysis tests**

Run:

```powershell
python -m pytest clipperx/tests/test_style_analysis.py clipperx/tests/test_jobs.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add clipperx/style_analysis.py clipperx/jobs.py clipperx/config.py clipperx/tests/test_style_analysis.py clipperx/tests/test_jobs.py
git commit -m "feat: extract objective style evidence"
```

---

### Task 3: Build provider-neutral StyleCards

**Files:**
- Create: `clipperx/style_library.py`
- Create: `clipperx/style_prompts.py`
- Create: `clipperx/tests/test_style_library.py`
- Create: `clipperx/tests/test_style_prompts.py`
- Modify: `clipperx/structured_client.py`

- [ ] **Step 1: Write failing structured-card tests**

Create a fake client that returns `StyleCardAnalysisResponse` and records images/prompts. Write:

- `test_card_builder_sends_three_to_five_saved_keyframes`, asserting the exact image count and host-preserved fingerprint;
- `test_card_builder_rejects_absolute_keyframe_refs`, using `C:/outside/frame.jpg`;
- `test_prompt_forbids_source_pair_and_narrative_inference`, asserting the two explicit prohibitions in the system prompt.

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_style_library.py clipperx/tests/test_style_prompts.py -q
```

Expected: missing modules.

- [ ] **Step 3: Implement enrichment and persistence**

`build_style_card()` receives objective evidence plus three to five keyframes. Its system prompt must state:

```text
Describe exclusivamente decisiones visibles de edición.
No inventes el VOD fuente ni una relación temporal con otro archivo.
No evalúes la calidad narrativa; el host la conserva como null salvo etiqueta humana.
Devuelve exactamente StyleCardAnalysisResponse, sin paths, IDs ni asociaciones con un VOD.
Usa solo las regiones y técnicas que puedas sostener con los keyframes y métricas.
Describe solamente los campos visuales solicitados por el contrato.
```

The caller combines the response with immutable host fields (`card_id`, fingerprint, relative `source_clip_ref`, duration, approval, narrative quality, and objective metrics) so the model cannot rewrite them.

Persist one JSON per card and an atomic `library_manifest.json`. Re-running with the same evidence hash and model ID is idempotent.

- [ ] **Step 4: Keep the generic client stateless**

Only extend `OpenAICompatibleStructuredClient.generate()` if required to attach image labels. Preserve its current single request with system/user content and schema. Add a test proving that two calls do not reuse prior messages.

- [ ] **Step 5: Test with fake clients**

Run:

```powershell
python -m pytest clipperx/tests/test_style_library.py clipperx/tests/test_style_prompts.py clipperx/tests/test_structured_client.py -q
```

Expected: all tests pass without network.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/style_library.py clipperx/style_prompts.py clipperx/structured_client.py clipperx/tests/test_style_library.py clipperx/tests/test_style_prompts.py clipperx/tests/test_structured_client.py
git commit -m "feat: create provider-neutral style cards"
```

---

### Task 4: Retrieve compatible style references

**Files:**
- Create: `clipperx/style_retrieval.py`
- Create: `clipperx/tests/test_style_retrieval.py`
- Modify: `clipperx/config.py`

- [ ] **Step 1: Write failing retrieval tests**

```python
def test_gameplay_query_excludes_full_camera_only_card() -> None:
    results = retrieve_style_cards(
        query(scene_family="gameplay_with_camera"),
        [card("gameplay"), card("full_camera")],
        limit=3,
    )
    assert [item.card.card_id for item in results] == ["gameplay"]


def test_recent_approved_style_wins_only_after_compatibility() -> None:
    newer_incompatible = card("new", scene_family="full_camera", approval="approved")
    older_compatible = card("old", scene_family="gameplay_with_camera", approval="approved")
    results = retrieve_style_cards(
        query(scene_family="gameplay_with_camera"),
        [newer_incompatible, older_compatible],
        limit=3,
    )
    assert [item.card.card_id for item in results] == ["old"]


def test_retrieval_returns_reasons_and_scores() -> None:
    result = retrieve_style_cards(
        query(scene_family="gameplay_with_camera"),
        [card("gameplay", scene_family="gameplay_with_camera")],
        limit=3,
    )[0]
    assert result.compatibility_score >= 0
    assert result.style_score >= 0
    assert result.reasons
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_style_retrieval.py -q
```

Expected: missing module.

- [ ] **Step 3: Implement two-stage retrieval**

Stage A rejects incompatible cards using scene family, presence of camera, content preservation needs, and number of visible people. Stage B ranks remaining cards:

```python
final_score = (
    0.35 * scene_similarity
    + 0.20 * geometry_similarity
    + 0.15 * overlay_similarity
    + 0.15 * semantic_description_similarity
    + 0.10 * visual_quality
    + 0.05 * recency_score
)
```

Use the existing sentence-transformer setting for description embeddings and cache vectors in `style_index.npz`. Retrieval returns at most three diverse cards; no two returned cards may share the same clip fingerprint.

If fewer than two compatible cards exist, return what exists and mark `coverage="sparse"` rather than adding incompatible examples.

- [ ] **Step 4: Run retrieval tests**

Run:

```powershell
python -m pytest clipperx/tests/test_style_retrieval.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add clipperx/style_retrieval.py clipperx/config.py clipperx/tests/test_style_retrieval.py
git commit -m "feat: retrieve compatible historical styles"
```

---

### Task 5: Define common safe zones without shifting visual centering

**Files:**
- Create: `clipperx/platform_safety.py`
- Create: `clipperx/tests/test_platform_safety.py`

- [ ] **Step 1: Write failing safe-zone tests**

```python
def test_common_title_band_starts_below_top_ten_percent() -> None:
    zones = common_vertical_safe_zones()
    assert zones.title.y >= 0.10


def test_right_controls_restrict_overlays_not_canvas_center() -> None:
    zones = common_vertical_safe_zones()
    assert zones.canvas_center_x == 0.5
    assert zones.overlay_forbidden_right.x >= 0.80


def test_subtitle_region_avoids_bottom_ui() -> None:
    zones = common_vertical_safe_zones()
    assert zones.subtitle_allowed.y + zones.subtitle_allowed.height <= 0.82
```

- [ ] **Step 2: Implement named profiles**

Add `tiktok`, `reels`, `shorts`, and `common_vertical` profiles. The common profile is the union of UI exclusions:

```python
title = NormalizedRect(x=0.08, y=0.10, width=0.84, height=0.12)
subtitle_allowed = NormalizedRect(x=0.10, y=0.48, width=0.72, height=0.34)
overlay_forbidden_right = NormalizedRect(x=0.82, y=0.36, width=0.18, height=0.52)
overlay_forbidden_bottom = NormalizedRect(x=0, y=0.82, width=1, height=0.18)
```

These restrictions apply to title, subtitle, and brand overlays. They must not change the output canvas center, move gameplay left, or offset the streamer panel.

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m pytest clipperx/tests/test_platform_safety.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```powershell
git add clipperx/platform_safety.py clipperx/tests/test_platform_safety.py
git commit -m "feat: model cross-platform overlay safe zones"
```

---

### Task 6: Implement the stateless visual director

**Files:**
- Create: `clipperx/visual_director.py`
- Create: `clipperx/visual_prompts.py`
- Create: `clipperx/tests/test_visual_director.py`
- Create: `clipperx/tests/test_visual_prompts.py`
- Modify: `clipperx/evidence.py`
- Modify: `clipperx/jobs.py`

- [ ] **Step 1: Write failing director tests with fake responses**

Cover these exact tests:

- `test_director_preserves_editorial_ranges_when_splitting_visual_shots`: the union and order of source ranges are identical before/after;
- `test_director_rechecks_camera_boundary_then_finishes`: the fake client is called twice and only `(59.0, 63.0)` is collected;
- `test_director_stops_after_three_evidence_iterations`: status is `REVIEW_REQUIRED` and no fourth call occurs;
- `test_director_never_receives_full_transcript`: a distant sentinel transcript line is absent from every user prompt.

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_visual_director.py clipperx/tests/test_visual_prompts.py -q
```

Expected: missing modules.

- [ ] **Step 3: Implement prompt assembly**

The system prompt must define:

- the director chooses explicit geometry, not layout names;
- source ranges are immutable;
- visual subdivisions must align to source-state events;
- streamer identity must use supplied track IDs and status;
- important regions, face, HUD, and OCR cannot be cut;
- `cover`, blur, and black are evidence-based;
- headline, brand, and subtitles remain centered within overlay-safe regions;
- no random effects or unrequested SFX;
- evidence is requested when identity or a boundary is unresolved;
- output is exactly `VisualDirectorResponse`.

The user payload contains only:

```json
{
  "clip": {"clip_id": "clip-0001", "headline": {}, "segments": []},
  "local_transcript": [],
  "source_spans": [],
  "face_tracks": [],
  "webcam_tracks": [],
  "audio_windows": [],
  "safe_zones": {},
  "style_cards": [],
  "frame_manifest": []
}
```

Cap each request at five source images plus one keyframe from each of at most three StyleCards, eight images total. Prefer event, start, end, identity, and important-region frames; do not sample at fixed cadence.

- [ ] **Step 4: Implement iterative orchestration**

```python
@dataclass(frozen=True, slots=True)
class VisualDirectorLimits:
    max_calls_per_clip: int = 6
    max_followups_per_segment: int = 2
    max_source_images_per_call: int = 5
    style_cards_per_segment: int = 3
    max_style_images_per_card: int = 1
    transcript_padding_seconds: float = 2.0
    max_transcript_chars: int = 8000
```

Implement the exact public signature `run_visual_director(source_video: Path, editorial_clip: ClipPlan, source_timeline: SourceVisualTimeline, *, client: StructuredClient, style_library: StyleLibrary, limits: VisualDirectorLimits = VisualDirectorLimits(), jobs_root: Path | None = None) -> Path`.

Persist each request, selected images, response, token usage, and evidence bundle. Deduplicate by `(reason, rounded start/end, question)` rather than request ID alone. Validate that the final plan covers exactly the editorial source ranges and does not exceed 179 seconds.

- [ ] **Step 5: Run director tests**

Run:

```powershell
python -m pytest clipperx/tests/test_visual_director.py clipperx/tests/test_visual_prompts.py clipperx/tests/test_evidence.py clipperx/tests/test_jobs.py -q
```

Expected: all tests pass without network or model calls.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/visual_director.py clipperx/visual_prompts.py clipperx/evidence.py clipperx/jobs.py clipperx/tests/test_visual_director.py clipperx/tests/test_visual_prompts.py clipperx/tests/test_evidence.py clipperx/tests/test_jobs.py
git commit -m "feat: direct temporal visual composition"
```

---

### Task 7: Integrate the director after narrative composition

**Files:**
- Modify: `clipperx/planner_engine.py`
- Modify: `clipperx/cli.py`
- Create: `clipperx/tests/test_visual_planner_integration.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing integration tests**

Write these exact integration tests:

- `test_visual_planning_consumes_persisted_composition_without_mutating_it`, comparing canonical model dumps before and after;
- `test_engine_calls_visual_director_only_after_composition_path_exists`, asserting the file exists inside the fake director;
- `test_visual_plan_command_requires_confirmation_for_inference`, invoking a complete fixture command without `--confirm-inference` and asserting failure before client construction.

- [ ] **Step 2: Implement an explicit post-composition command**

Add:

```text
clipperx build-style-library DIR --output-dir DIR
    [--model MODEL --context-window N --confirm-inference]
clipperx plan-visual VIDEO EDITORIAL_PLAN SOURCE_TIMELINE
    --style-library DIR --model MODEL --context-window N
    --confirm-inference
```

`build-style-library` may run local evidence extraction without a model. Enrichment requires confirmation. `plan-visual` creates one v2 JSON per editorial clip plus a manifest linking both plan versions.

- [ ] **Step 3: Add an explicit integrated wrapper after persisted Composition**

Keep `run_editorial_planner()` and its return type unchanged. Add:

```python
@dataclass(frozen=True, slots=True)
class PlannerRunArtifacts:
    composition_plan: Path
    visual_plan_manifest: Path
    run_summary: Path


def run_editorial_and_visual_planner(
    video_path: Path,
    transcript_path: Path,
    *,
    editorial_client: StructuredClient,
    visual_client: StructuredClient,
    model_name: str,
    context_window_tokens: int,
    max_output_tokens: int,
    source_timeline_path: Path,
    style_library: StyleLibrary,
    editorial_limits: PlannerRunLimits,
    visual_limits: VisualDirectorLimits,
    jobs_root: Path | None = None,
) -> PlannerRunArtifacts:
    raise NotImplementedError
```

The wrapper calls the existing planner, verifies that `composition_plan.json` is persisted, and only then invokes the visual director. Existing tests that expect a v1 `GlobalCompositionResponse` remain valid.

- [ ] **Step 4: Run integration and regression tests**

Run:

```powershell
python -m pytest clipperx/tests/test_visual_planner_integration.py clipperx/tests/test_planner_engine.py clipperx/tests/test_cli.py -q
python -m pytest clipperx/tests -q
```

Expected: all tests pass; old `plan-with-model` output remains editorial v1.

- [ ] **Step 5: Commit**

```powershell
git add clipperx/planner_engine.py clipperx/cli.py clipperx/tests/test_visual_planner_integration.py README.md
git commit -m "feat: integrate visual direction after composition"
```

---

### Task 8: Benchmark a small real model set before batch enrichment

**Files:**
- Create: `scripts/benchmark_visual_models.py`
- Create: `data/model_validation/visual_director_cases.json`
- Create: `data/model_validation/candidate_models.json`
- Create: `data/model_validation/selected_visual_model.json`
- Create: `clipperx/tests/test_visual_model_cases.py`
- Modify: `docs/style-corpus-audit.md`

- [ ] **Step 1: Define three bounded benchmark cases**

The cases are:

1. gameplay with stable top-left webcam;
2. full camera;
3. a camera/browser or browser/gameplay transition.

Each case stores editorial plan, local source timeline subset, expected invariants, and no more than eight image paths. It does not store a complete transcript.

- [ ] **Step 2: Test the cases without inference**

Run:

```powershell
python -m pytest clipperx/tests/test_visual_model_cases.py -q
```

Expected: three valid cases, each below configured text/image budget.

- [ ] **Step 3: Discover candidates and run the bounded benchmark**

```powershell
python scripts/benchmark_visual_models.py --cases data/model_validation/visual_director_cases.json --discover-models --base-url http://localhost:8317/v1 --exclude-family gemini --candidate-output data/model_validation/candidate_models.json
python scripts/benchmark_visual_models.py --cases data/model_validation/visual_director_cases.json --models-from data/model_validation/candidate_models.json --selection-output data/model_validation/selected_visual_model.json --confirm-inference
```

The first command performs no inference. The second records schema success, invariant pass rate, prompt/completion tokens, latency, evidence iterations, estimated provider cost, and the cheapest passing model. It renders nothing and processes only the three cases.

- [ ] **Step 4: Select the cheapest model that passes**

Required pass conditions:

- 100% valid JSON schema;
- 100% preservation of editorial ranges;
- no identity substitution;
- no geometry outside 0..1;
- correct visual split in the transition case;
- no more than one extra evidence iteration per case.

If no tested model passes, do not enrich all 301 clips and do not process a full VOD.

- [ ] **Step 5: Record exact results**

Append model IDs, provider, date, context setting, token usage, pass/fail, and selected role to `docs/style-corpus-audit.md`.

- [ ] **Step 6: Final phase verification**

Run:

```powershell
python -m pytest clipperx/tests -q
python -m mypy clipperx
python -m ruff check clipperx scripts
python -m compileall clipperx scripts
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit**

```powershell
git add scripts/benchmark_visual_models.py data/model_validation/visual_director_cases.json clipperx/tests/test_visual_model_cases.py docs/style-corpus-audit.md
git commit -m "test: benchmark visual director models"
```

## Phase 2 Exit Gate

Phase 3 may start only when:

- StyleCards contain no invented source pairing;
- retrieval excludes incompatible scene families;
- model calls are stateless, local, and bounded;
- the director preserves narrative source ranges exactly;
- `VisualCompositionPlan` v2 validates with explicit geometry and evidence;
- a three-case model benchmark identifies a passing model or explicitly records that none passed.
