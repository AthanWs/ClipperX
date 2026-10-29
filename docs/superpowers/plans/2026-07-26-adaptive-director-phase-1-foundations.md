# Adaptive Visual Director Phase 1 Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the provider-free visual foundation: strict v2 composition contracts, temporal source states, persistent webcam regions, and streamer identity tracks that do not select the largest face by default.

**Architecture:** Preserve the existing editorial planner and `timeline.json` contract. Add a separate `SourceVisualTimeline` artifact built in two passes: sparse local analysis over the VOD, then dense local probes around detected changes and approved clip boundaries. YuNet supplies landmarks and SFace embeddings; temporal association links faces and webcam rectangles. Ambiguous identity remains explicit data and can request more evidence later.

**Tech Stack:** Python 3.10+, Pydantic 2, OpenCV 4.12 YuNet/SFace, NumPy, SciPy Hungarian matching, FFprobe, existing audio feature extractor, Typer, pytest.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-07-26-adaptive-source-final-director-design.md`.
- This phase must not invoke an LLM or VL model.
- Do not alter narrative ranges, discovery, boundary search, or composition.
- Do not replace an ambiguous streamer identity with the largest visible face.
- Keep `clipperx.timeline.build_timeline()` and its current v2 artifact working for existing callers.
- Every cache key must include source fingerprint, analyzer version, model hashes, and sampling configuration.
- All normalized rectangles use the source frame coordinate system with origin at the top-left.
- If `git rev-parse --is-inside-work-tree` is false, record the intended commit message in the execution log and skip the commit command; do not claim a commit exists.

---

### Task 1: Define strict visual timeline and composition contracts

**Files:**
- Create: `clipperx/visual_contracts.py`
- Create: `clipperx/tests/test_visual_contracts.py`
- Modify: `clipperx/planner_contracts.py`
- Modify: `clipperx/tests/test_planner_contracts.py`

- [ ] **Step 1: Write failing geometry and timeline contract tests**

```python
from pydantic import ValidationError
import pytest

from clipperx.planner_contracts import TimeRange
from clipperx.visual_contracts import (
    FillMode,
    LayerKind,
    NormalizedRect,
    SourceState,
    SourceVisualSpan,
    VisualCompositionPlan,
    VisualLayer,
    VisualShot,
)


def test_normalized_rect_rejects_out_of_frame_geometry() -> None:
    with pytest.raises(ValidationError, match="sale del frame"):
        NormalizedRect(x=0.8, y=0.1, width=0.3, height=0.4)


def test_visual_plan_rejects_output_gap_or_overlap() -> None:
    layer = VisualLayer(
        layer_id="content",
        kind=LayerKind.SOURCE_VIDEO,
        source_ref="source",
        source_crop=NormalizedRect(x=0, y=0, width=1, height=1),
        destination=NormalizedRect(x=0, y=0, width=1, height=1),
        fill=FillMode.COVER,
        z_index=0,
    )
    with pytest.raises(ValidationError, match="continuos"):
        VisualCompositionPlan(
            clip_id="c1",
            source_fingerprint="source-fp",
            composition_fingerprint="composition-fp",
            source_timeline_fingerprint="timeline-fp",
            identity_fingerprint="identity-fp",
            style_library_fingerprint="style-fp",
            shots=[
                VisualShot(
                    shot_id="s1",
                    source=TimeRange(start=10, end=12),
                    output=TimeRange(start=0, end=2),
                    primary_source="content",
                    reason="hook",
                    layers=[layer],
                    evidence_ids=["sample-1"],
                    confidence=1.0,
                ),
                VisualShot(
                    shot_id="s2",
                    source=TimeRange(start=20, end=22),
                    output=TimeRange(start=2.2, end=4.2),
                    primary_source="streamer",
                    reason="payoff",
                    layers=[layer],
                    evidence_ids=["sample-2"],
                    confidence=1.0,
                ),
            ],
        )


def test_source_span_keeps_ambiguous_identity_explicit() -> None:
    span = SourceVisualSpan(
        span_id="state-0001",
        source=TimeRange(start=0, end=10),
        state=SourceState.MULTI_PERSON,
        confidence=0.7,
        streamer_track_id=None,
        identity_status="ambiguous",
        webcam_track_id=None,
        important_regions=[],
        text_regions=[],
        evidence_ids=["sample-1"],
        change_reasons=[],
    )
    assert span.streamer_track_id is None
    assert span.identity_status == "ambiguous"
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
python -m pytest clipperx/tests/test_visual_contracts.py -q
```

Expected: collection fails because `clipperx.visual_contracts` does not exist.

- [ ] **Step 3: Implement enums and source-timeline models**

Implement these public types in `clipperx/visual_contracts.py`:

```python
class SourceState(str, Enum):
    FULL_CAMERA = "full_camera"
    CONTENT_WITH_CAMERA = "content_with_camera"
    CONTENT_ONLY = "content_only"
    BROWSER_TEXT = "browser_text"
    REACTED_VIDEO = "reacted_video"
    MULTI_PERSON = "multi_person"
    UNKNOWN = "unknown"


class IdentityStatus(str, Enum):
    CONFIDENT = "confident"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


class FillMode(str, Enum):
    COVER = "cover"
    CONTAIN_BLUR = "contain_blur"
    BLACK = "black"
    SOLID = "solid"
    INHERIT = "inherit"


class LayerKind(str, Enum):
    SOURCE_VIDEO = "source_video"
    STREAMER_CAMERA = "streamer_camera"
    SOLID = "solid"


class NormalizedRect(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "NormalizedRect":
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("El rectángulo normalizado sale del frame")
        return self
```

Add `TemporalRectObservation`, `FaceTrackSummary`, `WebcamTrack`, `SourceVisualSample`, `SourceVisualSpan`, and `SourceVisualTimeline`. Use these exact public fields:

```python
class TemporalRectObservation(BaseModel):
    at: float = Field(ge=0)
    rect: NormalizedRect
    confidence: float = Field(ge=0, le=1)
    evidence_id: str


class FaceTrackSummary(BaseModel):
    track_id: str
    first_seen: float = Field(ge=0)
    last_seen: float = Field(gt=0)
    observation_count: int = Field(ge=1)
    embedding_centroid: list[float] = Field(min_length=1)
    positions: list[NormalizedRect] = Field(min_length=1)
    recurrent_states: list[SourceState] = Field(default_factory=list)
    full_camera_seconds: float = Field(default=0.0, ge=0)
    mean_confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)


class WebcamTrack(BaseModel):
    track_id: str
    source_ranges: list[TimeRange] = Field(min_length=1)
    observations: list[TemporalRectObservation] = Field(min_length=1)
    face_track_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)


class SourceVisualSample(BaseModel):
    sample_id: str
    at: float = Field(ge=0)
    histogram_change_score: float = Field(ge=0, le=1)
    structural_change_score: float = Field(ge=0, le=1)
    letterbox_bounds: NormalizedRect | None = None
    persistent_rectangles: list[NormalizedRect] = Field(default_factory=list)
    text_density: float = Field(ge=0, le=1)
    text_regions: list[NormalizedRect] = Field(default_factory=list)
    face_track_ids: list[str] = Field(default_factory=list)
    webcam_track_id: str | None = None
    webcam_rect: NormalizedRect | None = None
    brightness: float = Field(ge=0, le=1)
    sharpness: float = Field(ge=0)
    motion: float = Field(ge=0, le=1)
    audio_features: dict[str, float] = Field(default_factory=dict)
    audio_window: TimeRange | None = None
    transcript_window: TimeRange | None = None
    diarization_window: TimeRange | None = None
    evidence_ids: list[str] = Field(min_length=1)
```

A span must carry:

```python
span_id: str
source: TimeRange
state: SourceState
confidence: float
streamer_track_id: str | None
identity_status: IdentityStatus
webcam_track_id: str | None
important_regions: list[NormalizedRect]
text_regions: list[NormalizedRect]
evidence_ids: list[str]
change_reasons: list[str]
```

`SourceVisualTimeline` must use these exact fields:

```python
contract_version: Literal[1] = 1
source_fingerprint: str
duration_seconds: float = Field(gt=0)
analyzer_version: str
model_hashes: dict[str, str]
sampling_config: dict[str, float]
samples: list[SourceVisualSample]
spans: list[SourceVisualSpan]
face_tracks: list[FaceTrackSummary]
webcam_tracks: list[WebcamTrack]
```

Timeline validators must reject duplicate IDs, non-increasing sample timestamps, out-of-order or overlapping spans, references to unknown face/webcam tracks, and timestamps or ranges outside `duration_seconds`.

- [ ] **Step 4: Implement the complete v2 visual composition contract**

Add these public types:

```python
class CropKeyframe(BaseModel):
    at: float = Field(ge=0)
    crop: NormalizedRect
    easing: Literal["linear", "ease_in_out"] = "ease_in_out"


class Anchor(str, Enum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER = "center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


class VisualLayer(BaseModel):
    layer_id: str
    kind: LayerKind
    source_ref: str
    anchor: Anchor = Anchor.TOP_LEFT
    source_crop: NormalizedRect | None = None
    destination: NormalizedRect
    fill: FillMode
    solid_color: str | None = None
    blur_sigma: float | None = Field(default=None, gt=0, le=50)
    z_index: int = Field(ge=0)
    opacity: float = Field(default=1.0, ge=0, le=1)
    crop_track: list[CropKeyframe] = Field(default_factory=list)
    preserve_regions: list[NormalizedRect] = Field(default_factory=list)


class OverlayPlan(BaseModel):
    text: str | None = None
    asset_id: str | None = None
    font_asset_id: str | None = None
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    destination: NormalizedRect
    z_index: int = Field(ge=0)


class SubtitlePlan(BaseModel):
    destination: NormalizedRect
    font_asset_id: str = "subtitle_bold"
    font_size: int = Field(default=58, ge=36, le=84)
    primary_color: str = "#FFD51A"
    outline_color: str = "#000000"
    outline_width: int = Field(default=4, ge=0, le=8)
    max_words_per_event: int = Field(default=6, ge=1, le=10)
    max_lines: int = Field(default=2, ge=1, le=3)
    emphasize_active_word: bool = False


class AudioCue(BaseModel):
    asset_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    gain_db: float = Field(default=-12.0, ge=-36, le=0)
    duck_source_db: float = Field(default=-6.0, ge=-18, le=0)


class AudioMixPlan(BaseModel):
    source_gain_db: float = Field(default=0.0, ge=-24, le=12)
    normalize_dialogue: bool = True
    limiter: bool = True
    cues: list[AudioCue] = Field(default_factory=list)


class VisualShot(BaseModel):
    shot_id: str
    source: TimeRange
    output: TimeRange
    primary_source: Literal["content", "streamer", "shared"]
    reason: str
    layers: list[VisualLayer] = Field(min_length=1)
    subtitle_allowed_regions: list[NormalizedRect] = Field(default_factory=list)
    forbidden_regions: list[NormalizedRect] = Field(default_factory=list)
    subtitles: SubtitlePlan | None = None
    title: OverlayPlan | None = None
    brand: OverlayPlan | None = None
    transition_in: Literal["cut"] = "cut"
    transition_out: Literal["cut"] = "cut"
    audio: AudioMixPlan = Field(default_factory=AudioMixPlan)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class VisualCompositionPlan(BaseModel):
    contract_version: Literal[2] = 2
    clip_id: str
    width: Literal[1080] = 1080
    source_fingerprint: str
    composition_fingerprint: str
    source_timeline_fingerprint: str
    identity_fingerprint: str
    style_library_fingerprint: str
    height: Literal[1920] = 1920
    shots: list[VisualShot] = Field(min_length=1)
```

Validators must enforce:

- each shot has equal source and output duration within 1 ms;
- output shots start at 0 and are contiguous within 1 ms;
- `shot_id` and `layer_id` values are unique in their scope;
- crop keyframes are ordered and unique; moving tracks start at 0 and end at the shot duration;
- overlay times are relative to the shot and inside its duration;
- subtitle, title, and brand destinations do not intersect forbidden regions;
- audio cue times are relative to the shot, ordered, and inside its duration;
- every layer `source_ref` is the sentinel `source` for the source frame or a Phase 1 track ID;
- exactly one of `text` or `asset_id` is present in an overlay;
- `solid` requires `solid_color`; other fill modes forbid it;
- a first-shot layer cannot use `inherit`;
- asset, font, and audio IDs must match `^[a-z][a-z0-9_]{0,63}$`; paths, drive roots, separators, and URI-like values are forbidden here, while actual host-registry membership is enforced by the Phase 3 preflight before FFmpeg starts;
- total output duration is at most `MAX_CLIP_SECONDS`;
- at least one visible layer covers every shot;
- no unknown fill/layout value is accepted.

Keep `NormalizedCrop` and every other v1 planner model unchanged. Export the v2 schema alongside v1; do not add required fields to `EditSegment`.

- [ ] **Step 5: Run contract tests**

Run:

```powershell
python -m pytest clipperx/tests/test_visual_contracts.py clipperx/tests/test_planner_contracts.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/visual_contracts.py clipperx/planner_contracts.py clipperx/tests/test_visual_contracts.py clipperx/tests/test_planner_contracts.py
git commit -m "feat: add strict visual composition contracts"
```

---

### Task 2: Add reproducible YuNet and SFace model assets

**Files:**
- Create: `clipperx/model_assets.py`
- Create: `clipperx/tests/test_model_assets.py`
- Modify: `clipperx/config.py`
- Modify: `clipperx/diagnostics.py`
- Modify: `clipperx/cli.py`
- Create: `clipperx/tests/test_diagnostics.py`
- Create: `clipperx/tests/test_cli.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing asset registry tests**

```python
from clipperx.model_assets import FACE_MODELS, validate_model_file


def test_face_models_use_named_onnx_files() -> None:
    assert FACE_MODELS["yunet"].filename == "face_detection_yunet_2023mar.onnx"
    assert FACE_MODELS["sface"].filename == "face_recognition_sface_2021dec.onnx"


def test_lfs_pointer_is_rejected(tmp_path) -> None:
    model = tmp_path / FACE_MODELS["yunet"].filename
    model.write_text("version https://git-lfs.github.com/spec/v1", encoding="ascii")
    assert validate_model_file(model, FACE_MODELS["yunet"]) is False
```

- [ ] **Step 2: Confirm the tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_model_assets.py -q
```

Expected: import failure for `clipperx.model_assets`.

- [ ] **Step 3: Implement the model registry and atomic downloader**

Use these official OpenCV Zoo assets:

```python
FACE_MODELS = {
    "yunet": ModelAsset(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        minimum_bytes=200_000,
    ),
    "sface": ModelAsset(
        filename="face_recognition_sface_2021dec.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        minimum_bytes=30_000_000,
    ),
}
```

`ensure_face_models(model_dir, allow_download)` must:

1. validate extension, minimum size, and reject Git LFS pointer text;
2. download to `*.part`;
3. load YuNet with `cv2.FaceDetectorYN_create()` and SFace with `cv2.FaceRecognizerSF_create()`;
4. compute and persist SHA-256 in `face_models.manifest.json`;
5. atomically rename only after both models load;
6. raise an actionable error when download is disabled.
7. serialize setup with an inter-process lock and recover or preserve enough evidence to restore an interrupted transaction;
8. verify an existing installed bundle against its persisted manifest instead of silently blessing changed files.

Filesystem failures exposed by the public setup API must become actionable `ModelAssetError` failures. A failed rollback must preserve its backup directory for a later recovery attempt.

Add:

```python
face_model_dir: str = "data/models/face_identity"
allow_model_downloads: bool = False
```

to `EnvSettings`. Add diagnostics that report API availability, asset presence, hashes, and loadability without downloading.

Diagnostics must also fail when both `opencv-python` and `opencv-contrib-python` distributions are installed in the selected runtime; Phase 1 needs one unambiguous `cv2` build before ONNX calibration.
Diagnostics must compare current filenames, byte sizes, and SHA-256 values with the manifest when present, and report mismatch or a missing manifest without rewriting it.


- [ ] **Step 4: Test validation without network**

Run:

```powershell
python -m pytest clipperx/tests/test_model_assets.py clipperx/tests/test_diagnostics.py -q
```

Expected: all tests pass; no network call occurs.

- [ ] **Step 5: Run the explicit one-time model setup**

Run only with user-approved network access:

```powershell
$env:ALLOW_MODEL_DOWNLOADS="true"
python -m clipperx.cli setup-face-models
python -m clipperx.cli doctor --json
```

`setup-face-models` is the only command in this flow authorized to download. `doctor`
must remain strictly offline. Expected: setup installs the validated pair atomically;
then diagnostics report `FaceDetectorYN_create=true`, `FaceRecognizerSF_create=true`,
both assets loadable, and two SHA-256 values.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/model_assets.py clipperx/config.py clipperx/diagnostics.py clipperx/cli.py clipperx/tests/test_model_assets.py clipperx/tests/test_diagnostics.py clipperx/tests/test_cli.py .gitignore
git commit -m "feat: manage face identity model assets"
```

---

### Task 3: Track faces and resolve streamer identity without size bias

**Files:**
- Create: `clipperx/face_identity.py`
- Create: `clipperx/webcam_tracking.py`
- Create: `clipperx/tests/test_face_identity.py`
- Create: `clipperx/tests/test_webcam_tracking.py`
- Modify: `clipperx/tests/test_face.py`
- Modify: `clipperx/jobs.py`

- [ ] **Step 1: Write failing association and identity tests**

Use synthetic detections and embeddings; do not require ONNX files in unit tests.

```python
def test_larger_foreign_face_does_not_replace_persistent_streamer() -> None:
    observations = [
        obs(0, box=(0.02, 0.02, 0.18, 0.25), embedding=unit(0)),
        obs(5, box=(0.02, 0.02, 0.18, 0.25), embedding=unit(0)),
        obs(10, box=(0.30, 0.05, 0.60, 0.85), embedding=unit(1)),
        obs(10, box=(0.02, 0.02, 0.18, 0.25), embedding=unit(0)),
    ]
    tracks = associate_face_tracks(observations)
    identity = resolve_streamer_identity(tracks)
    assert identity.track_id == tracks[0].track_id


def test_close_identity_scores_return_ambiguous() -> None:
    identity = resolve_streamer_identity(
        [track("a", score_inputs=(0.8, 0.8)), track("b", score_inputs=(0.79, 0.81))]
    )
    assert identity.status == IdentityStatus.AMBIGUOUS
    assert identity.track_id is None
```

Preflight clarification:

- `LandmarkedFace` is a frame-local result: normalized box, exactly five normalized
  landmarks, detector confidence, and an optional L2-normalized embedding.
- `FaceObservation` adds `at`, `source_state`, `evidence_id`, and embedding-quality
  weight. Area may affect that quality weight only; it must not enter identity rank.
- Public association is `associate_face_tracks(observations, *, sample_timestamps=(),
  max_missed_samples=3) -> list[FaceTrack]`. Empty sampled timestamps must be passed
  explicitly. A track survives up to three consecutive missing samples; a detection
  after those three misses starts a new track.
- Public resolution is `resolve_streamer_identity(tracks, *, saved_profile_embedding=None,
  ambiguity_margin=0.12) -> StreamerIdentityResult`.
- Rank is the active-weight-normalized sum of duration `0.35`, recurrence across
  source states `0.20`, full-camera ratio `0.20`, and saved-profile cosine mapped to
  `[0,1]` at `0.25`. Omit and renormalize the profile term when no profile exists.
  Face area is forbidden from this score. No tracks returns `unavailable`; one valid
  track returns `confident`; a top-two gap below the margin returns `ambiguous`.
- `FaceTrack.to_summary()` is the serialization boundary to `FaceTrackSummary`; keep
  quality-weighted observations internal and serialize a normalized centroid.
- `WebcamRegionObservation` contains a persistent rectangle independently from
  `face_track_ids`. `associate_webcam_tracks(..., sample_timestamps=(),
  max_missed_samples=3)` must continue a rectangle track when face evidence is absent.
  Shared face-track evidence may associate a relocated rectangle; otherwise require
  rectangle overlap or bounded center motion.

- [ ] **Step 2: Confirm the focused tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_face_identity.py clipperx/tests/test_webcam_tracking.py clipperx/tests/test_face.py -q
```

Expected: missing identity/webcam modules; the legacy face test remains unchanged.

- [ ] **Step 3: Implement YuNet landmark detections and SFace embeddings**

Add a `LandmarkedFace` type containing normalized box, five landmarks, detector confidence, and optional embedding. Implement:

```python
class FaceIdentityBackend(Protocol):
    def detect_and_embed(self, frame: np.ndarray) -> list[LandmarkedFace]:
        raise NotImplementedError
    def cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        raise NotImplementedError
```

The OpenCV backend must call `FaceDetectorYN.setInputSize()`, use `detect()`, pass the 15-value YuNet detection to `FaceRecognizerSF.alignCrop()`, then call `feature()`. L2-normalize embeddings before serialization.

Retain `clipperx.face` unchanged for legacy callers. New identity code uses YuNet landmarks and SFace and must never call legacy `_largest_box`.

- [ ] **Step 4: Implement temporal association**

Use `scipy.optimize.linear_sum_assignment` per sampled timestamp. The cost is:

```python
cost = 0.55 * (1.0 - cosine_similarity) + 0.30 * (1.0 - box_iou) + 0.15 * motion_distance
```

Reject an assignment if cosine similarity is below `0.32` or normalized center motion exceeds `0.45`. A missing observation does not end a track until three sampled timestamps pass. Persist per-track embedding centroid, duration, recurrence across source states, positions, and quality-weighted samples.

Streamer ranking may use duration, recurrence, full-camera presence, and match to a saved face profile. Face area is only a quality signal for embedding selection and must not rank identity. If the first two scores differ by less than `0.12`, return `ambiguous`.

- [ ] **Step 5: Separate webcam tracks from face tracks**

Implement `webcam_tracking.py` as a new v2 path based on persistent rectangular regions plus supporting face-track evidence. A track may remain alive temporarily when the streamer turns away; store `face_track_ids` separately from rectangle observations.

Do not modify `detect_webcam_region()` or `_largest_box()` in Phase 1. Add a
characterization test proving they remain legacy-only. The v2 builder is created in
Task 4, so its import-path proof belongs to `test_visual_timeline.py`, not Task 3.

- [ ] **Step 6: Add job artifact paths**

Add atomic paths:

```python
face_tracks_path -> root / "face_tracks.json"
streamer_identity_path -> root / "streamer_identity.json"
webcam_tracks_path -> root / "webcam_tracks.json"
```

- [ ] **Step 7: Run tests**

Run:

```powershell
python -m pytest clipperx/tests/test_face_identity.py clipperx/tests/test_webcam_tracking.py clipperx/tests/test_face.py clipperx/tests/test_jobs.py -q
```

Expected: all tests pass, including the larger-foreign-face regression.

- [ ] **Step 8: Commit**

```powershell
git add clipperx/face_identity.py clipperx/webcam_tracking.py clipperx/jobs.py clipperx/tests/test_face_identity.py clipperx/tests/test_webcam_tracking.py clipperx/tests/test_face.py clipperx/tests/test_jobs.py
git commit -m "feat: track streamer identity across source layouts"
```

---

### Task 4: Build the sparse-to-dense source visual timeline

**Files:**
- Create: `clipperx/visual_timeline.py`
- Create: `clipperx/tests/test_visual_timeline.py`
- Modify: `clipperx/timeline.py`
- Modify: `clipperx/audio_features.py`
- Modify: `clipperx/jobs.py`

- [ ] **Step 1: Write failing state segmentation tests**

Tests must use generated frames and injected analyzers:

```python
def test_camera_move_splits_source_state_near_event() -> None:
    samples = [
        sample(0, webcam=(0.00, 0.00, 0.25, 0.30)),
        sample(10, webcam=(0.00, 0.00, 0.25, 0.30)),
        sample(20, webcam=(0.72, 0.00, 0.25, 0.30)),
        sample(30, webcam=(0.72, 0.00, 0.25, 0.30)),
    ]
    spans = segment_visual_states(samples)
    assert [span.source.start for span in spans] == [0, 15]
    assert "webcam_moved" in spans[1].change_reasons


def test_person_inside_reacted_video_is_not_labeled_webcam() -> None:
    samples = reacted_video_samples_with_large_face()
    spans = segment_visual_states(samples)
    assert all(span.state != SourceState.CONTENT_WITH_CAMERA for span in spans)
```

- [ ] **Step 2: Confirm the tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_visual_timeline.py -q
```

Expected: import failure for `clipperx.visual_timeline`.

- [ ] **Step 3: Implement local sample features**

For each sparse sample, compute and serialize:

- histogram and structural change score;
- letterbox/pillarbox bounds;
- persistent straight-edge rectangles;
- MSER text-region density and bounding boxes;
- YuNet/SFace observations;
- webcam-region observations;
- brightness, sharpness, and motion;
- existing audio features;
- transcript and diarization references by timestamp, not copied full text.

Default sparse interval is 10 seconds. Dense windows use 0.5-second samples from three seconds before to three seconds after:

- visual change score crossing `0.18`;
- webcam appearing, disappearing, or moving more than 8% of frame width/height;
- face-track set changing;
- letterbox bounds changing;
- text density changing by at least `0.12`;
- an editorial candidate boundary supplied by the caller.

Merge overlapping dense windows before decoding frames.

- [ ] **Step 4: Implement state classification and segmentation**

The local classifier produces evidence, not a creative layout. Implement these exact public signatures:

- `classify_source_state(sample: SourceVisualSample) -> tuple[SourceState, float, list[str]]`
- `segment_visual_states(samples: Sequence[SourceVisualSample], *, minimum_span_seconds: float = 1.0) -> list[SourceVisualSpan]`

Use temporal consistency across adjacent samples. Do not label a region as webcam from a face alone; require a persistent rectangle or a confident streamer track recurring in the same bounded region. Mark unresolved cases `unknown` or `multi_person`.

- [ ] **Step 5: Implement the builder and preserve the legacy timeline**

Implement `build_source_visual_timeline(video_path: Path, *, editorial_boundaries: Sequence[float] = (), sparse_interval: float = 10.0, dense_interval: float = 0.5, allow_model_download: bool = False, force: bool = False) -> Path`.

Write atomically to `JobStore.source_visual_timeline_path`. Do not change the return format of `timeline.build_timeline()`. Reuse frame decodes and audio caches where signatures match.
Add a builder characterization proving this v2 path imports and uses
`webcam_tracking`; monkeypatch the legacy `detect_webcam_region()` / `_largest_box()`
path to fail if the builder touches it.

- [ ] **Step 6: Run timeline tests**

Run:

```powershell
python -m pytest clipperx/tests/test_visual_timeline.py clipperx/tests/test_timeline.py clipperx/tests/test_audio_features.py clipperx/tests/test_jobs.py -q
```

Expected: all tests pass and the legacy timeline tests remain unchanged.

- [ ] **Step 7: Commit**

```powershell
git add clipperx/visual_timeline.py clipperx/timeline.py clipperx/audio_features.py clipperx/jobs.py clipperx/tests/test_visual_timeline.py clipperx/tests/test_timeline.py clipperx/tests/test_audio_features.py clipperx/tests/test_jobs.py
git commit -m "feat: build temporal source visual states"
```

---

### Task 5: Expose local indexing and inspection commands

**Files:**
- Modify: `clipperx/cli.py`
- Create: `clipperx/tests/test_visual_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

Use `typer.testing.CliRunner` and monkeypatch the builder:

```python
def test_index_visual_does_not_require_provider(monkeypatch, tmp_path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    timeline = tmp_path / "source_visual_timeline.json"
    timeline.write_text("{}", encoding="utf-8")
    captured = {}
    def fake_builder(video_path, **kwargs):
        captured.update(kwargs)
        return timeline
    monkeypatch.setattr("clipperx.visual_timeline.build_source_visual_timeline", fake_builder)
    result = runner.invoke(app, ["index-visual", str(video)])
    assert result.exit_code == 0
    assert "Source visual timeline:" in result.stdout
    assert captured["allow_model_download"] is False


def test_index_visual_enables_download_only_with_explicit_flag(monkeypatch, tmp_path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    timeline = tmp_path / "source_visual_timeline.json"
    timeline.write_text("{}", encoding="utf-8")
    captured = {}
    def fake_builder(video_path, **kwargs):
        captured.update(kwargs)
        return timeline
    monkeypatch.setattr("clipperx.visual_timeline.build_source_visual_timeline", fake_builder)
    result = runner.invoke(
        app,
        ["index-visual", str(video), "--allow-model-download"],
    )
    assert result.exit_code == 0
    assert captured["allow_model_download"] is True
```

- [ ] **Step 2: Confirm the tests fail**

Run:

```powershell
python -m pytest clipperx/tests/test_visual_cli.py -q
```

Expected: `index-visual` is not a registered command.

- [ ] **Step 3: Implement commands**

Add:

```text
clipperx index-visual VIDEO [--sparse-interval 10] [--dense-interval 0.5]
                              [--allow-model-download] [--force]
clipperx inspect-visual VIDEO --start S --end E [--sample-interval 0.5]
clipperx export-visual-schemas --output-dir DIR
```

`inspect-visual` must cap a single window at 300 seconds and emit a standalone `SourceVisualTimeline` subset. None of these commands accepts or reads provider credentials.

- [ ] **Step 4: Document artifacts and costs**

README must explain:

- sparse vs. dense passes;
- model asset sizes and one-time download;
- expected artifact paths;
- that this phase performs zero LLM calls;
- how ambiguity is represented instead of silently choosing a face.

- [ ] **Step 5: Run CLI and full unit tests**

Run:

```powershell
python -m pytest clipperx/tests/test_visual_cli.py -q
python -m pytest clipperx/tests -q
python -m compileall clipperx
```

Expected: all tests pass and compileall exits 0.

- [ ] **Step 6: Commit**

```powershell
git add clipperx/cli.py clipperx/tests/test_visual_cli.py README.md
git commit -m "feat: expose provider-free visual indexing"
```

---

### Task 6: Validate on synthetic and real source windows

**Files:**
- Create: `data/validation_visual/scenarios.json`
- Create: `scripts/validate_source_timeline.py`
- Create: `clipperx/tests/test_validation_manifest.py`
- Modify: `docs/style-corpus-audit.md`

- [ ] **Step 1: Define a portable scenario manifest**

The committed manifest contains scenario IDs and expected properties, not machine-specific drive paths:

```json
{
  "version": 1,
  "scenarios": [
    {"id": "stable_gameplay_webcam", "expected_states": ["content_with_camera"]},
    {"id": "full_camera", "expected_states": ["full_camera"]},
    {"id": "full_camera_to_browser", "ordered_states": ["full_camera", "content_with_camera"], "boundary_near_event_seconds": 2.0},
    {"id": "browser_to_gameplay", "boundary_near_event_seconds": 2.0},
    {"id": "foreign_faces_in_content", "forbid_streamer_switch": true},
    {"id": "multi_person", "require_multiple_faces": true},
    {"id": "no_webcam", "forbid_webcam_track": true},
    {"id": "reacted_vertical_video", "require_vertical_region": true},
    {"id": "browser_small_text", "require_text_regions": true},
    {"id": "webcam_move_or_disappear", "ordered_states": ["content_with_camera", "full_camera"], "boundary_near_event_seconds": 2.0}
  ]
}
```

Machine-local source path/start/end mappings live in ignored
`data/validation_visual/local_sources.json`. Transition scenarios also carry a
human-confirmed `event_at`; the gate requires the detected visual boundary to
land within the declared tolerance instead of accepting an unrelated change.

- [ ] **Step 2: Write and run manifest tests**

Run:

```powershell
python -m pytest clipperx/tests/test_validation_manifest.py -q
```

Expected: ten unique scenarios, every scenario has an enforceable assertion, and no `R:\` or other absolute source path is committed.

- [ ] **Step 3: Run a no-LLM real-window smoke**

```powershell
python scripts/validate_source_timeline.py --sources data/validation_visual/local_sources.json --output data/validation_visual/runs/phase1
```

Expected:

- ten scenario reports are written;
- no provider endpoint is contacted;
- each output records runtime, decoded samples when provable, model hashes, state spans, identity status, and failed assertions;
- the command exits 1 if any required assertion fails and 2 when evidence is
  ambiguous or unavailable.

- [ ] **Step 4: Record measured results**

Append the exact run timestamp, hardware, scenario pass count, ambiguous identities, false webcam detections, and runtime to `docs/style-corpus-audit.md`. Do not call Phase 1 complete if the real-window smoke was skipped.

- [ ] **Step 5: Final phase verification**

Run:

```powershell
python -m pytest clipperx/tests -q
python -m mypy clipperx
python -m ruff check clipperx scripts
python -m compileall clipperx scripts
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit**

```powershell
git add data/validation_visual/scenarios.json scripts/validate_source_timeline.py clipperx/tests/test_validation_manifest.py docs/style-corpus-audit.md
git commit -m "test: validate source visual timeline scenarios"
```

## Phase 1 Exit Gate

Phase 2 may start only when:

- v1 editorial plans still validate;
- v2 visual plans reject incomplete geometry;
- the source timeline is cached and provider-free;
- a foreign larger face does not replace the persistent streamer in tests;
- webcam movement produces a temporal split;
- the ten-scenario real-window report exists and distinguishes pass, fail, ambiguous, and unavailable.
