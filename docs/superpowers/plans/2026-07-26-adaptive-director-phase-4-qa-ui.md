# Adaptive Visual Director Phase 4 QA and Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inspect rendered MP4s, correct only failed shots at most twice, expose the adaptive workflow in CLI/TUI, and save user-approved source-result pairs for future regression and learning.

**Architecture:** Local QA handles media invariants and measured geometry first. A bounded stateless VL review then inspects selected rendered frames and emits structured issues. A correction orchestrator allows at most two plan revisions and enforces shot-level write scope. The interface presents storyboard, evidence, warnings, attempts, and approval actions; automatic QA never implies human approval.

**Tech Stack:** Python 3.10+, Pydantic 2, FFmpeg/FFprobe, OpenCV, existing structured client, Textual, Typer, pytest.

## Global Constraints

- Requires the Phase 3 exit gate and its render sidecars.
- Phase 4 consumes `VisualCompositionPlan` v2 and compiler metrics; it does not reconstruct geometry.
- Attempt 0 plus at most two correction renders means a maximum of three MP4s per clip.
- Correction 1 may touch only failed shots. Correction 2 may touch only shots still failing after correction 1.
- Source/output ranges and narrative order are immutable during QA correction.
- Codec, muxing, missing-audio, and decode failures are renderer failures; do not spend a model revision on them.
- Every VL request is self-contained, capped at eight images, and requires explicit inference authorization.
- A technical `qa_passed` result is not a user `approved` result.
- Historical style clips are never converted into source-result pairs.
- The TUI keeps legacy and `Athan adaptativo` workflows visibly separate.
- If the workspace is not a Git repository, skip commit commands and record intended messages.

---

### Task 1: Define QA, revision, and cycle contracts

**Files:**
- Create: `clipperx/render_review_models.py`
- Create: `clipperx/tests/test_render_review_models.py`
- Modify: `clipperx/visual_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_review_attempt_is_limited_to_zero_one_or_two() -> None:
    with pytest.raises(ValidationError):
        RenderQAReport(
            clip_id="c1",
            attempt=3,
            local_passed=False,
            vl_status="not_run",
            issues=[],
            mp4_path="render.mp4",
            plan_path="plan.json",
            metrics_path="metrics.json",
        )


def test_patch_requires_failed_shot_ids() -> None:
    with pytest.raises(ValidationError):
        PlanRevision(replacements=[], reasoning="nothing")


def test_automatic_pass_is_not_user_approval() -> None:
    result = ReviewCycleResult(
        status="qa_passed",
        corrections=0,
        final_plan_path="plan.json",
        final_mp4_path="render.mp4",
        final_report_path="qa.json",
    )
    assert not hasattr(result, "approved")
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_render_review_models.py -q
```

Expected: missing review models.

- [ ] **Step 3: Implement issue and report models**

Minimum issue codes:

```python
class QAIssueCode(str, Enum):
    WRONG_RESOLUTION = "wrong_resolution"
    WRONG_CODEC = "wrong_codec"
    MISSING_AUDIO = "missing_audio"
    AV_DESYNC = "av_desync"
    DURATION_MISMATCH = "duration_mismatch"
    DECODE_ERROR = "decode_error"
    BLACK_FRAME = "black_frame"
    CUT_DISCONTINUITY = "cut_discontinuity"
    UNSAFE_OVERLAY = "unsafe_overlay"
    CROPPED_FACE = "cropped_face"
    WRONG_IDENTITY = "wrong_identity"
    OCCLUDED_TEXT = "occluded_text"
    CRITICAL_CONTENT_LOST = "critical_content_lost"
    ABSURD_GAP = "absurd_gap"
    ABRUPT_LAYOUT_CHANGE = "abrupt_layout_change"
```

Public report types:

```python
class RenderQAIssue(BaseModel):
    code: QAIssueCode
    severity: Literal["warning", "error"]
    shot_id: str | None = None
    output_start: float | None = None
    output_end: float | None = None
    evidence_frame_ids: list[str] = Field(default_factory=list)
    detail: str
    patchable: bool


class RenderQAReport(BaseModel):
    contract_version: Literal[1] = 1
    clip_id: str
    attempt: int = Field(ge=0, le=2)
    local_passed: bool
    vl_status: Literal["not_run", "passed", "patch_required", "needs_review"]
    issues: list[RenderQAIssue]
    mp4_path: str
    plan_path: str
    metrics_path: str
    review_frame_ids: list[str] = Field(default_factory=list)


class PlanRevision(BaseModel):
    contract_version: Literal[1] = 1
    replacements: list[VisualShot] = Field(min_length=1)
    reasoning: str


class ReviewCycleResult(BaseModel):
    status: Literal["qa_passed", "needs_review", "renderer_failed"]
    corrections: int = Field(ge=0, le=2)
    final_plan_path: str
    final_mp4_path: str
    final_report_path: str
```

All models use `extra="forbid"`.

- [ ] **Step 4: Run contract tests**

Run:

```powershell
python -m pytest clipperx/tests/test_render_review_models.py clipperx/tests/test_visual_contracts.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add clipperx/render_review_models.py clipperx/visual_contracts.py clipperx/tests/test_render_review_models.py
git commit -m "feat: define post-render review contracts"
```

---

### Task 2: Persist review attempts atomically

**Files:**
- Create: `clipperx/render_review_store.py`
- Create: `clipperx/tests/test_render_review_store.py`
- Modify: `clipperx/jobs.py`
- Modify: `clipperx/config.py`

- [ ] **Step 1: Write failing store tests**

```python
def test_review_store_creates_numbered_attempts(tmp_path) -> None:
    store = RenderReviewStore(tmp_path, clip_id="clip-1", cycle_id="cycle-1")
    assert store.attempt(0).root.name == "attempt_00"
    assert store.attempt(2).root.name == "attempt_02"


def test_fourth_attempt_is_impossible(tmp_path) -> None:
    store = RenderReviewStore(tmp_path, clip_id="clip-1", cycle_id="cycle-1")
    with pytest.raises(ValueError, match="0..2"):
        store.attempt(3)


def test_report_write_is_atomic(tmp_path) -> None:
    store = RenderReviewStore(tmp_path, clip_id="clip-1", cycle_id="cycle-1")
    output = store.write_report(store.attempt(0), report(attempt=0))
    assert RenderQAReport.model_validate_json(output.read_text(encoding="utf-8")).attempt == 0
    assert list(output.parent.glob("*.tmp")) == []
```

- [ ] **Step 2: Implement the store**

Artifact tree:

```text
<job>/qa/<clip_id>/<cycle_id>/
  attempt_00/
    plan.json
    render.mp4
    render-spec.json
    render-metrics.json
    local-qa.json
    vl-qa.json
    frames/
  attempt_01/
  attempt_02/
  cycle-result.json
```

Add `qa_dir`, `feedback_dir`, and `review_cycle_dir()` to `JobStore`. Add global `feedback_pairs_dir` under `data/processed/feedback/pairs`.

- [ ] **Step 3: Run store tests**

Run:

```powershell
python -m pytest clipperx/tests/test_render_review_store.py clipperx/tests/test_jobs.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```powershell
git add clipperx/render_review_store.py clipperx/jobs.py clipperx/config.py clipperx/tests/test_render_review_store.py clipperx/tests/test_jobs.py
git commit -m "feat: persist bounded render review attempts"
```

---

### Task 3: Implement deterministic local render QA

**Files:**
- Create: `clipperx/local_render_qa.py`
- Create: `clipperx/tests/test_local_render_qa.py`
- Modify: `clipperx/media.py`
- Modify: `clipperx/tests/test_media.py`

- [ ] **Step 1: Write failing QA tests using generated media**

Cover these generated-media tests:

- `test_valid_vertical_mp4_passes_local_qa`;
- `test_720x1280_reports_wrong_resolution`;
- `test_missing_audio_reports_missing_audio`;
- `test_av_offset_of_41_ms_reports_desync`;
- `test_half_second_black_run_reports_black_frame`;
- `test_overlay_outside_compiled_safe_zone_reports_unsafe_overlay`;
- `test_decode_failure_is_not_patchable`.

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_local_render_qa.py -q
```

Expected: missing local QA module.

- [ ] **Step 3: Extend media probing**

`probe_media()` must include:

```json
{
  "video": {
    "codec": "h264",
    "width": 1080,
    "height": 1920,
    "pix_fmt": "yuv420p",
    "start_time": 0.0,
    "duration": 37.48,
    "frame_rate": 60.0
  },
  "audio": {
    "codec": "aac",
    "sample_rate": 48000,
    "start_time": 0.0,
    "duration": 37.48
  }
}
```

Preserve all existing keys.

- [ ] **Step 4: Implement exact local checks**

Implement `run_local_render_qa(mp4_path: Path, plan: VisualCompositionPlan, render_metrics_path: Path, *, attempt: int, output_path: Path) -> RenderQAReport`.

Checks:

- H.264/AAC, 1080x1920, `yuv420p`;
- duration at most 179.0 seconds;
- planned/rendered duration difference at most one 60 FPS frame plus one AAC packet (`1/60 + 1024/48000`, under 0.040 seconds);
- A/V start and end differences use the same under-0.040-second tolerance;
- complete FFmpeg decode to null succeeds;
- all-black run at least 0.250 seconds is an error unless the plan marks that interval as intentional black;
- frames immediately before/after each shot boundary decode;
- actual compiled overlay/layer boxes from `render-metrics.json` obey allowed/forbidden regions;
- unplanned abrupt visual changes are warnings for VL review, not automatic plan mutations.

Local QA must not use face recognition or claim `wrong_identity`.

- [ ] **Step 5: Run QA and media tests**

Run:

```powershell
python -m pytest clipperx/tests/test_local_render_qa.py clipperx/tests/test_media.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/local_render_qa.py clipperx/media.py clipperx/tests/test_local_render_qa.py clipperx/tests/test_media.py
git commit -m "feat: verify rendered media locally"
```

---

### Task 4: Select rendered evidence and run bounded VL review

**Files:**
- Create: `clipperx/render_review_frames.py`
- Create: `clipperx/render_review_prompts.py`
- Create: `clipperx/vl_render_review.py`
- Create: `clipperx/tests/test_render_review_frames.py`
- Create: `clipperx/tests/test_render_review_prompts.py`
- Create: `clipperx/tests/test_vl_render_review.py`

- [ ] **Step 1: Write failing frame-selection tests**

Write these exact tests:

- `test_review_frames_include_start_end_boundaries_keyframes_and_emotion`, asserting all six reason labels;
- `test_review_frames_deduplicate_timestamps_within_200_ms_but_merge_reasons`;
- `test_vl_calls_never_exceed_eight_images`, asserting every recorded fake-client batch size.

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_render_review_frames.py clipperx/tests/test_render_review_prompts.py clipperx/tests/test_vl_render_review.py -q
```

Expected: missing modules.

- [ ] **Step 3: Implement review-frame selection**

Select:

- 100 ms after output start and 100 ms before output end;
- 100 ms before/after each shot boundary;
- every crop-track keyframe;
- source high-energy/emotion events mapped through `OutputTimeline`;
- timestamps associated with local warnings.

Deduplicate timestamps within 200 ms while retaining all reason labels. Save JPEGs with stable frame IDs and a manifest. Do not send the MP4.

- [ ] **Step 4: Implement the VL review prompt**

The prompt asks only whether the rendered result violates the supplied plan/evidence:

- face or head cut;
- selected identity wrong;
- essential gameplay/browser/reacted-video region lost;
- subtitle/title/brand unsafe or occluding content;
- meaningless black/blur area;
- abrupt visual discontinuity;
- retained gap unrelated to the reaction;
- cut that loses visual context.

It must not rewrite clips, add effects, change narrative ranges, or score virality. It returns `RenderQAReport` issues with `shot_id`, evidence frame IDs, and patchability.

- [ ] **Step 5: Implement stateless batches**

Implement `run_vl_render_review(report: RenderQAReport, plan: VisualCompositionPlan, frames: Sequence[ReviewFrame], *, client: StructuredGenerator, max_images_per_call: int = 8) -> RenderQAReport`.

Each batch repeats clip/shot geometry, frame manifest, local issues, identity IDs, and relevant source evidence. Merge identical issue signatures and preserve the highest severity.

- [ ] **Step 6: Run review tests**

Run:

```powershell
python -m pytest clipperx/tests/test_render_review_frames.py clipperx/tests/test_render_review_prompts.py clipperx/tests/test_vl_render_review.py -q
```

Expected: all tests pass with fake clients and no network.

- [ ] **Step 7: Commit**

```powershell
git add clipperx/render_review_frames.py clipperx/render_review_prompts.py clipperx/vl_render_review.py clipperx/tests/test_render_review_frames.py clipperx/tests/test_render_review_prompts.py clipperx/tests/test_vl_render_review.py
git commit -m "feat: review rendered frames with bounded vision calls"
```

---

### Task 5: Enforce the two-correction shot-scoped cycle

**Files:**
- Create: `clipperx/post_render_cycle.py`
- Create: `clipperx/plan_revision.py`
- Create: `clipperx/tests/test_post_render_cycle.py`
- Create: `clipperx/tests/test_plan_revision.py`

- [ ] **Step 1: Write failing cycle tests**

Add these exact tests:

- `test_clean_initial_render_stops_at_attempt_zero`: use a renderer spy and a clean local report; assert `corrections == 0` and `renderer.calls == 1`.
- `test_two_failed_attempts_never_trigger_fourth_render`: return a failing report for attempts 0, 1, and 2; assert `status == "needs_review"` and `renderer.calls == 3`.
- `test_second_revision_cannot_touch_recovered_shot`: mark `shot-1` recovered after revision 1, attempt to revise it again, and assert `PlanRevisionError`.
- `test_revision_cannot_change_source_or_output_ranges`: alter either canonical range in a permitted shot and assert `PlanRevisionError` with a message containing `rangos`.

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_post_render_cycle.py clipperx/tests/test_plan_revision.py -q
```

Expected: missing modules.

- [ ] **Step 3: Define renderer and revision ports**

```python
class ClipRenderPort(Protocol):
    def render(
        self,
        source_video: Path,
        plan: VisualCompositionPlan,
        output_dir: Path,
    ) -> RenderArtifact:
        raise NotImplementedError


class PlanRevisionPort(Protocol):
    def revise(
        self,
        plan: VisualCompositionPlan,
        report: RenderQAReport,
        *,
        allowed_shot_ids: frozenset[str],
    ) -> PlanRevision:
        raise NotImplementedError
```

- [ ] **Step 4: Implement canonical patch validation**

`apply_plan_revision()` replaces complete shots by `shot_id`, then verifies:

- replacement ID is allowed;
- source and output ranges are byte-for-byte equal to the original canonical JSON;
- narrative reason and evidence references are preserved;
- unaffected shots are canonical-JSON identical;
- revised plan fully validates.

The revision prompt receives only failed shots, their issues, adjacent shot summaries, source evidence, and review frames.

- [ ] **Step 5: Implement the bounded orchestrator**

Implement `finalize_render(source_video: Path, initial_plan: VisualCompositionPlan, *, renderer: ClipRenderPort, local_qa: LocalQAPort, vl_reviewer: VLReviewPort | None, reviser: PlanRevisionPort | None, store: RenderReviewStore, max_corrections: int = 2) -> ReviewCycleResult`.

Rules:

1. render attempt 0;
2. stop as `renderer_failed` on non-patchable structural errors;
3. run VL only when configured/confirmed;
4. if no errors, return `qa_passed`;
5. revision 1 may touch current failed shots;
6. revision 2 may touch only still-failed shots;
7. after attempt 2, return `needs_review` if errors remain;
8. reuse source timeline, identity, and StyleCards; never reindex the VOD.

`max_corrections` accepts only 0, 1, or 2.

- [ ] **Step 6: Run cycle tests**

Run:

```powershell
python -m pytest clipperx/tests/test_post_render_cycle.py clipperx/tests/test_plan_revision.py -q
```

Expected: all tests pass; a fourth render is impossible.

- [ ] **Step 7: Commit**

```powershell
git add clipperx/post_render_cycle.py clipperx/plan_revision.py clipperx/tests/test_post_render_cycle.py clipperx/tests/test_plan_revision.py
git commit -m "feat: constrain post-render corrections by shot"
```

---

### Task 6: Build the ten-scenario QA corpus and approved feedback pairs

**Files:**
- Create: `clipperx/qa_corpus.py`
- Create: `clipperx/feedback_store.py`
- Create: `clipperx/tests/test_qa_corpus.py`
- Create: `clipperx/tests/test_feedback_store.py`
- Create: `data/qa_scenarios/manifest.json`

- [ ] **Step 1: Define exact scenario IDs**

The manifest must contain exactly:

```text
stable_webcam_dark_gameplay
full_camera
full_camera_to_browser
browser_to_gameplay
foreign_faces_in_content
second_person
no_webcam
reacted_vertical_video
browser_small_text
camera_move_or_disappearance
```

Committed records contain expected states/invariants. Machine-local source, start, and end mappings stay in ignored `data/qa_scenarios/local_sources.json` until each range is visually verified. Contact sheets alone are insufficient evidence for exact temporal ranges.

- [ ] **Step 2: Write manifest tests**

Add these exact tests:

- `test_manifest_contains_exactly_ten_scenarios`: assert the committed manifest contains the ten required IDs once each and no extras.
- `test_each_local_scenario_has_positive_non_overlapping_range`: load the ignored local mapping, assert `end_s > start_s >= 0`, and reject overlapping ranges from the same source unless the records explicitly declare the same validation window.
- `test_no_absolute_drive_paths_are_committed`: recursively inspect the committed manifest and reject Windows drive roots, UNC roots, and `file://` URIs.

- [ ] **Step 3: Implement scenario artifact storage**

Each executed scenario stores:

```text
scenario.json
input/source_ref.json
timeline.json
plan.json
renders/attempt_XX.mp4
qa/attempt_XX.local.json
qa/attempt_XX.vl.json
final_result.json
```

- [ ] **Step 4: Implement human-approved feedback pairs**

Implement `approve_render_pair(*, source_video: Path, initial_plan: Path, final_plan: Path, output_mp4: Path, qa_report: Path, correction_history: Path, user_decision: UserDecision, output_root: Path) -> Path`.

Only `UserDecision(decision="approved")` creates:

```text
data/processed/feedback/pairs/<pair_id>/
  manifest.json
  source_ranges/<shot_id>.mp4
  initial_plan.json
  final_plan.json
  output.mp4
  qa_report.json
  correction_history.json
  user_decision.json
```

Extract exact source ranges as H.264/AAC clips and record their source fingerprint. A QA pass without user approval creates no pair.

- [ ] **Step 5: Run corpus and feedback tests**

Run:

```powershell
python -m pytest clipperx/tests/test_qa_corpus.py clipperx/tests/test_feedback_store.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/qa_corpus.py clipperx/feedback_store.py clipperx/tests/test_qa_corpus.py clipperx/tests/test_feedback_store.py data/qa_scenarios/manifest.json
git commit -m "feat: store QA scenarios and approved render pairs"
```

---

### Task 7: Expose QA and approval commands

**Files:**
- Modify: `clipperx/cli.py`
- Create: `clipperx/tests/test_review_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

Add these exact CLI tests:

- `test_qa_render_local_does_not_create_model_client`: invoke `qa-render` without `--vl`, assert exit code 0, and assert the injected model-client factory has zero calls.
- `test_vl_review_requires_explicit_confirmation`: invoke `qa-render sample.mp4 plan.json --metrics render-metrics.json --vl --model configured-model` without `--confirm-inference`; assert a nonzero exit code and output containing `confirm-inference`.
- `test_max_corrections_three_is_rejected`: invoke `finalize-render` with `--max-corrections 3`; assert a CLI validation error before renderer construction.
- `test_approve_render_requires_explicit_decision`: omit `--decision`; assert a nonzero exit code and no feedback-pair directory creation.

- [ ] **Step 2: Implement commands**

```text
clipperx qa-render MP4 [PLAN] [--metrics FILE] [--media-only]
    [--vl --model MODEL --context-window N --confirm-inference]
    # PLAN and --metrics are mandatory unless --media-only is selected
clipperx finalize-render VIDEO PLAN --output-dir DIR
    [--vl --model MODEL --context-window N --confirm-inference]
    [--max-corrections 0|1|2]
clipperx approve-render CYCLE_RESULT --decision approved
clipperx regenerate-shot CYCLE_RESULT --shot-id ID
clipperx correct-identity CYCLE_RESULT --shot-id ID --track-id ID
```

`regenerate-shot` and `correct-identity` create a scoped revision request; they do not edit MP4 files in place.

- [ ] **Step 3: Register test markers**

Add:

```toml
"e2e: tests that invoke real FFmpeg and render media"
"model: opt-in tests that call an external or local model"
```

- [ ] **Step 4: Document status distinctions**

README distinguishes:

- planned;
- rendered;
- local-QA passed;
- VL-QA passed;
- needs review;
- user approved/pair saved.

- [ ] **Step 5: Run CLI tests**

Run:

```powershell
python -m pytest clipperx/tests/test_review_cli.py clipperx/tests/test_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/cli.py clipperx/tests/test_review_cli.py pyproject.toml README.md
git commit -m "feat: expose render QA and approval workflow"
```

---

### Task 8: Add the `Athan adaptativo` storyboard to the TUI

**Files:**
- Create: `clipperx/interface_presenter.py`
- Create: `clipperx/tests/test_interface_presenter.py`
- Modify: `clipperx/interface.py`
- Modify: `clipperx/runner.py`
- Create: `clipperx/tests/test_interface.py`
- Modify: `clipperx/tests/test_runner.py`

- [ ] **Step 1: Write presenter tests before Textual widgets**

```python
def test_storyboard_exposes_exact_shot_identity_and_style_refs() -> None:
    view = present_cycle(cycle)
    shot = view.clips[0].shots[0]
    assert shot.shot_id == "shot-1"
    assert shot.streamer_track_id == "face-track-2"
    assert shot.style_card_ids == ["style-4", "style-8"]


def test_actions_keep_clip_and_shot_scope() -> None:
    action = build_regenerate_action("clip-1", "shot-2")
    assert action.payload == {"clip_id": "clip-1", "shot_id": "shot-2"}
```

- [ ] **Step 2: Implement a UI-independent presenter**

The presenter exposes:

- clip and shot IDs;
- source/output ranges;
- dominant source;
- streamer track/crop;
- used StyleCards;
- title, brand, subtitle and safe-zone boxes;
- confidence and QA warnings;
- attempt history and final status;
- action payloads for approve, regenerate shot, and correct identity.

- [ ] **Step 3: Add the TUI workflow**

Add preset `Athan adaptativo` with stages:

```text
Indexar fuente → Plan narrativo → Dirección visual → Render → QA → Resultado
```

The storyboard uses centered thumbnails and does not alter render geometry. Legacy modes remain labeled and separate.

- [ ] **Step 4: Extend runner messages**

Replace raw path-only results for the adaptive mode with a typed result containing job ID, composition path, visual plan path, MP4 paths, QA status, and warnings. Preserve legacy result handling.

- [ ] **Step 5: Run presenter, runner, and Textual smoke tests**

Run:

```powershell
python -m pytest clipperx/tests/test_interface_presenter.py clipperx/tests/test_runner.py clipperx/tests/test_interface.py -q
```

Expected: all tests pass, including `Textual.run_test()` navigation without invoking a model.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/interface_presenter.py clipperx/interface.py clipperx/runner.py clipperx/tests/test_interface_presenter.py clipperx/tests/test_interface.py clipperx/tests/test_runner.py
git commit -m "feat: present adaptive storyboard and review actions"
```

---

### Task 9: Run full synthetic, real, and human validation

**Files:**
- Create: `scripts/run_qa_scenarios.py`
- Create: `clipperx/tests/test_review_e2e.py`
- Modify: `docs/style-corpus-audit.md`

- [ ] **Step 1: Run deterministic E2E fixtures**

Generate and test:

1. valid 1080x1920 H.264/AAC;
2. missing audio;
3. 720x1280;
4. 0.5-second black run;
5. 41 ms A/V offset;
6. fake VL issue plus one successful revision;
7. fake VL issue persisting through two revisions.

Run:

```powershell
python -m pytest clipperx/tests/test_review_e2e.py -m e2e -q
```

Expected: exact issue codes and maximum three render calls.

- [ ] **Step 2: Run local QA on the existing v8 render**

```powershell
python -m clipperx.cli qa-render data/model_validation/rendered/luna_low_1clip_v8_adaptive/le-robaron-la-silla-en-un-segundo-3321-3359.mp4 --media-only
```

If that legacy render lacks v2 sidecars, record it as not eligible for geometric QA rather than fabricating data. Media-only checks may still run.

- [ ] **Step 3: Execute all ten real scenarios**

```powershell
python scripts/run_qa_scenarios.py --manifest data/qa_scenarios/manifest.json --sources data/qa_scenarios/local_sources.json --output data/qa_scenarios/runs/latest
```

Expected: input reference, timeline, plan, MP4, local report, optional VL report, and final result for every scenario.

- [ ] **Step 4: Run opt-in model review on the same small bundle**

Only after selecting a model in Phase 2:

```powershell
python scripts/run_qa_scenarios.py --manifest data/qa_scenarios/manifest.json --sources data/qa_scenarios/local_sources.json --output data/qa_scenarios/runs/model-review --vl --model-from data/model_validation/selected_visual_model.json --confirm-inference --limit 3
```

Expected: exactly three bounded scenarios and recorded usage. Do not start with all ten or a full VOD.

- [ ] **Step 5: Request human review of the rendered clips**

Present absolute paths and ask the user to inspect:

- face/head framing;
- whether visual context starts early enough and ends correctly;
- camera/content priority;
- subtitle/title/brand position;
- black/blur/cover choice;
- whether cuts and gaps feel coherent.

Only approved outputs call `approve-render`.

- [ ] **Step 6: Record truthful readiness evidence**

Append:

- suite results;
- synthetic E2E results;
- ten-scenario pass/fail counts;
- model/provider and token usage;
- correction counts;
- user approvals/rejections;
- created pair IDs;
- unresolved edge cases.

- [ ] **Step 7: Final release verification**

Run:

```powershell
python -m pytest clipperx/tests -q
python -m pytest clipperx/tests -m e2e -q
python -m mypy clipperx
python -m ruff check clipperx scripts
python -m compileall clipperx scripts
python -m clipperx.cli doctor --json
```

Expected: every command exits 0. Model-marked tests remain opt-in and must have a separately recorded passing run before claiming VL operation.

- [ ] **Step 8: Commit**

```powershell
git add scripts/run_qa_scenarios.py clipperx/tests/test_review_e2e.py docs/style-corpus-audit.md
git commit -m "test: validate adaptive render review end to end"
```

## Phase 4 Exit Gate

The adaptive workflow is ready for user acceptance only when:

- local QA detects media and measured-geometry failures;
- VL review is bounded, stateless, and contract-tested with the selected provider;
- no clip can exceed two correction renders;
- a correction cannot alter narrative ranges or healthy shots;
- all ten real scenarios have stored artifacts and explicit outcomes;
- the TUI exposes storyboard, confidence, issues, and scoped actions;
- at least one user-approved output creates a genuine source-result pair;
- “rendered”, “QA passed”, and “user approved” remain separate statuses.
