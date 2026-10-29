from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import subprocess
from typing import Iterator, cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from clipperx.style_analysis import ANALYZER_VERSION, analyze_edited_clip, analyze_style_frames
from clipperx.style_contracts import BackgroundTreatment


def _panel_frame(color: tuple[int, int, int] = (0, 0, 0)) -> NDArray[np.uint8]:
    frame: NDArray[np.uint8] = np.full((320, 180, 3), color, dtype=np.uint8)
    rng = np.random.default_rng(41)
    frame[80:240] = rng.integers(30, 230, (160, 180, 3), dtype=np.uint8)
    return frame


def test_black_bars_are_not_reported_as_blur() -> None:
    result = analyze_style_frames([_panel_frame(), _panel_frame()])
    assert result.background_treatment == BackgroundTreatment.BLACK
    assert any(region.kind == "black" for region in result.background_regions)


def test_solid_bars_and_blur_remain_distinct() -> None:
    solid = analyze_style_frames([_panel_frame((70, 110, 160))])
    assert solid.background_treatment == BackgroundTreatment.SOLID
    frame = _panel_frame()
    rng = np.random.default_rng(3)
    blur = cast(NDArray[np.uint8], cv2.GaussianBlur(
        rng.integers(30, 220, frame.shape, dtype=np.uint8), (31, 31), 9
    ))
    blur[80:240] = frame[80:240]
    result = analyze_style_frames([blur])
    assert result.background_treatment == BackgroundTreatment.CONTAIN_BLUR
    assert any(region.kind == "blur_candidate" for region in result.background_regions)


def test_chicken_yellow_subtitle_palette_is_detected() -> None:
    frame = _panel_frame()
    cv2.putText(frame, "HOLA", (27, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(frame, "HOLA", (27, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 213, 26), 2)
    result = analyze_style_frames([frame])
    assert "#FFD51A" in result.subtitle_palette
    assert result.subtitle_regions
    assert result.overlay_candidates


def test_bright_subtitle_palette_survives_many_temporal_candidate_colors() -> None:
    frames = []
    colors = [(190 + i, 90 + i, 30 + i) for i in range(15)] + [(255, 213, 26)]
    for color in colors:
        frame = _panel_frame()
        cv2.putText(frame, "HOLA", (27, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(frame, "HOLA", (27, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        frames.append(frame)
    result = analyze_style_frames(frames)
    assert "#FFD51A" in result.subtitle_palette


def test_temporal_overlay_persistence_and_panels_are_estimates() -> None:
    frame = _panel_frame()
    cv2.putText(frame, "TITLE", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    result = analyze_style_frames([frame, frame], timestamps=[0.0, 1.0])
    assert result.title_regions
    assert max(item.persistence for item in result.overlay_candidates) == 1.0
    assert result.panel_candidates
    assert result.camera_regions == []
    assert result.quality_metrics["severe_face_cuts"] is None


def test_separate_content_panels_are_measured_without_camera_identity() -> None:
    rng = np.random.default_rng(82)
    frames = []
    for _ in range(3):
        frame: NDArray[np.uint8] = np.zeros((400, 240, 3), dtype=np.uint8)
        frame[45:210] = rng.integers(20, 230, (165, 240, 3), dtype=np.uint8)
        frame[250:355, 45:195] = rng.integers(20, 230, (105, 150, 3), dtype=np.uint8)
        frames.append(frame)
    result = analyze_style_frames(frames)
    assert len(result.panel_candidates) >= 2
    assert any(
        region.rect.y > 0.6 and region.rect.width < 0.8 for region in result.panel_candidates
    )
    assert result.camera_regions == []


def test_quality_penalizes_black_frames_without_narrative_inference() -> None:
    result = analyze_style_frames([np.zeros((320, 180, 3), dtype=np.uint8)])
    assert result.visual_quality < 0.6
    assert result.quality_metrics["black_frame_fraction"] == 1.0
    assert result.narrative_quality is None


@pytest.mark.parametrize(
    "frames", [[], [np.zeros((10, 10), dtype=np.uint8)], [np.zeros((10, 10, 3), dtype=np.float64)]]
)
def test_invalid_frames_fail_clearly(frames: list[np.ndarray]) -> None:
    with pytest.raises(ValueError):
        analyze_style_frames(frames)


def test_invalid_timestamps_fail_clearly() -> None:
    with pytest.raises(ValueError):
        analyze_style_frames([_panel_frame(), _panel_frame()], timestamps=[1.0, 0.0])


def test_motion_candidates_measure_translation_without_semantic_roles() -> None:
    rng = np.random.default_rng(12)
    frame = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
    moved = cast(NDArray[np.uint8], cv2.warpAffine(
        frame, np.asarray([[1, 0, 12], [0, 1, 0]], dtype=np.float32), (320, 320)
    ))
    result = analyze_style_frames([frame, moved], timestamps=[0.0, 0.5])
    assert result.pan_count >= 1
    assert any(item.kind == "pan_candidate" for item in result.motion_candidates)
    assert result.camera_regions == []


def test_jobs_and_config_expose_the_same_style_library_root(tmp_path: Path) -> None:
    from clipperx.config import Paths
    from clipperx.jobs import JobStore

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = JobStore.for_video(source, jobs_root=tmp_path / "jobs")
    assert Paths(base_dir=tmp_path).style_library_dir == tmp_path / "data/processed/style_library"
    assert job.style_evidence_path.name == "style_evidence.json"
    assert job.style_evidence_path.parent.name == job.fingerprint


def _video(path: Path) -> Path:
    # Real valid synthetic MP4 with a sharp cut away from evenly spaced samples.
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=180x320:r=10:d=0.7",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=180x320:r=10:d=1.3",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    return path


def test_analysis_saves_endpoints_actual_scene_and_library_relative_paths(tmp_path: Path) -> None:
    clip = _video(tmp_path / "clip.mp4")
    root = tmp_path / "library"
    result = analyze_edited_clip(clip, style_library_root=root, narrative_quality=None)
    assert result.visual_quality is not None
    assert result.narrative_quality is None
    assert result.analyzer_version == ANALYZER_VERSION
    assert 1 <= len(result.keyframes) <= 8
    assert result.keyframes[0].timestamp == 0.0
    assert result.keyframes[-1].timestamp >= 1.8
    assert any(abs(frame.timestamp - 0.7) < 0.11 for frame in result.keyframes)
    assert any(abs(at - 0.7) < 0.11 for at in result.scene_change_timestamps)
    for keyframe in result.keyframes:
        assert keyframe.path.startswith(f"evidence/{result.clip_fingerprint}/frames/")
        assert cv2.imread(str(root / keyframe.path)) is not None
    assert (root / result.evidence_path).is_file()
    assert result.media_properties["video"]["height"] == 320


def test_valid_cache_reused_and_invalid_version_rebuilt(tmp_path: Path) -> None:
    clip = _video(tmp_path / "clip.mp4")
    root = tmp_path / "library"
    first = analyze_edited_clip(clip, style_library_root=root)
    evidence_path = root / first.evidence_path
    modified = evidence_path.stat().st_mtime_ns
    cached = analyze_edited_clip(clip, style_library_root=root, narrative_quality=0.8)
    assert evidence_path.stat().st_mtime_ns == modified
    assert cached.narrative_quality == 0.8
    # Caller-owned narrative scores are never cached as local evidence.
    assert json.loads(evidence_path.read_text())["narrative_quality"] is None
    payload = json.loads(evidence_path.read_text())
    payload["analyzer_version"] = "outdated"
    evidence_path.write_text(json.dumps(payload))
    rebuilt = analyze_edited_clip(clip, style_library_root=root)
    assert rebuilt.analyzer_version == ANALYZER_VERSION
    assert json.loads(evidence_path.read_text())["analyzer_version"] == ANALYZER_VERSION


def test_corrupt_or_missing_jpeg_rebuilds_cache(tmp_path: Path) -> None:
    clip = _video(tmp_path / "clip.mp4")
    root = tmp_path / "library"
    first = analyze_edited_clip(clip, style_library_root=root)
    jpeg = root / first.keyframes[0].path
    jpeg.write_bytes(b"corrupt")
    analyze_edited_clip(clip, style_library_root=root)
    assert cv2.imread(str(jpeg)) is not None


@pytest.mark.parametrize("damage", ["json", "fingerprint", "timestamps", "traversal"])
def test_invalid_cache_metadata_rebuilds(tmp_path: Path, damage: str) -> None:
    clip = _video(tmp_path / "clip.mp4")
    root = tmp_path / "library"
    first = analyze_edited_clip(clip, style_library_root=root)
    evidence_path = root / first.evidence_path
    payload = json.loads(evidence_path.read_text())
    if damage == "json":
        evidence_path.write_text("invalid json")
    else:
        if damage == "fingerprint":
            payload["clip_fingerprint"] = "wrong"
        elif damage == "timestamps":
            payload["sampled_timestamps"] = [-1]
        else:
            payload["keyframes"][0]["path"] = "../outside.jpg"
        evidence_path.write_text(json.dumps(payload))
    rebuilt = analyze_edited_clip(clip, style_library_root=root)
    assert rebuilt == first


def test_fingerprint_changes_with_source_and_original_remains_unchanged(tmp_path: Path) -> None:
    clip = _video(tmp_path / "clip.mp4")
    before = clip.read_bytes()
    first = analyze_edited_clip(clip, style_library_root=tmp_path / "library")
    assert clip.read_bytes() == before
    clip.touch()
    second = analyze_edited_clip(clip, style_library_root=tmp_path / "library")
    assert second.clip_fingerprint != first.clip_fingerprint


def test_missing_source_and_out_of_range_narrative_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze_edited_clip(tmp_path / "absent.mp4", style_library_root=tmp_path / "library")
    clip = _video(tmp_path / "clip.mp4")
    with pytest.raises(ValueError):
        analyze_edited_clip(clip, style_library_root=tmp_path / "library", narrative_quality=1.1)


@pytest.mark.parametrize("stage", ["lock", "cache"])
def test_source_replacement_during_cache_lookup_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    import clipperx.style_analysis as style

    clip = _video(tmp_path / "clip.mp4")
    root = tmp_path / "library"
    analyze_edited_clip(clip, style_library_root=root)
    if stage == "lock":
        original_lock = style._InterProcessFileLock

        @contextmanager
        def changed_lock(path: Path) -> Iterator[None]:
            with original_lock(path):
                clip.write_bytes(clip.read_bytes() + b"replacement")
                yield

        monkeypatch.setattr(style, "_InterProcessFileLock", changed_lock)
    else:
        original_cache = style._cache

        def changed_cache(root: Path, fingerprint: str, evidence_path: str) -> object:
            cached = original_cache(root, fingerprint, evidence_path)
            clip.write_bytes(clip.read_bytes() + b"replacement")
            return cached

        monkeypatch.setattr(style, "_cache", changed_cache)
    with pytest.raises(RuntimeError, match="source changed"):
        analyze_edited_clip(clip, style_library_root=root)


@pytest.mark.parametrize(
    "damage", ["overlay_negative", "overlay_future", "overlay_unordered", "overlay_nan",
               "motion_reversed", "motion_future", "motion_negative", "motion_nan"],
)
def test_nested_temporal_corruption_rebuilds(tmp_path: Path, damage: str) -> None:
    clip = _video(tmp_path / "clip.mp4")
    root = tmp_path / "library"
    first = analyze_edited_clip(clip, style_library_root=root)
    evidence_path = root / first.evidence_path
    payload = json.loads(evidence_path.read_text())
    if damage.startswith("overlay"):
        times = {"overlay_negative": [-100], "overlay_future": [20],
                 "overlay_unordered": [1, 0], "overlay_nan": [float("nan")]}
        payload["overlay_candidates"] = [{
            "role": "subtitle_candidate", "rect": {"x": 0, "y": 0, "width": 1, "height": 1},
            "persistence": 1, "observed_timestamps": times[damage], "palette": [],
            "glyph_height_pixels_estimate": 10,
        }]
    else:
        start, end = {"motion_reversed": (20, 1), "motion_future": (0, 20),
                      "motion_negative": (-1, 1), "motion_nan": (0, float("nan"))}[damage]
        payload["motion_candidates"] = [{
            "start": start, "end": end, "kind": "pan_candidate", "scale_estimate": 1,
            "translation_fraction": 0.1, "inlier_fraction": 1,
        }]
    evidence_path.write_text(json.dumps(payload))
    assert analyze_edited_clip(clip, style_library_root=root) == first


def test_vfr_final_frame_and_late_scene_are_preserved(tmp_path: Path) -> None:
    clip = tmp_path / "vfr.mkv"
    # Five real presentation positions: 0, .1, .2, 2.8, 3.8 seconds.
    # Nominal frame metadata is not a presentation timeline for this clip.
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
        "color=red:s=64x64:r=10:d=0.5", "-vf",
        "setpts='if(lt(N,3),N,if(eq(N,3),28,38))/(10*TB)',"
        "drawbox=color=blue:t=fill:enable='eq(n,4)'",
        "-fps_mode", "vfr", "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p", str(clip),
    ], check=True, capture_output=True, timeout=20)
    decoded = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "frame=pts_time", "-of", "json", str(clip),
    ], check=True, capture_output=True, text=True, timeout=20)
    assert [float(frame["pts_time"]) for frame in json.loads(decoded.stdout)["frames"]] == [
        0.0, 0.1, 0.2, 2.8, 3.8,
    ]
    root = tmp_path / "library"
    evidence = analyze_edited_clip(clip, style_library_root=root)
    assert evidence.sampled_timestamps[-1] == pytest.approx(3.8, abs=0.01)
    assert evidence.keyframes[-1].timestamp == pytest.approx(3.8, abs=0.01)
    assert any(abs(t - 3.8) < 0.01 for t in evidence.scene_change_timestamps)
    final = cv2.imread(str(root / evidence.keyframes[-1].path))
    assert final is not None
    assert final[:, :, 0].mean() > 200  # Actual blue final image, not just a timestamp.
    assert final[:, :, 2].mean() < 30


def test_jpegs_are_published_by_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _video(tmp_path / "clip.mp4")
    original = Path.write_bytes

    def no_direct_jpeg_write(path: Path, data: bytes) -> int:
        assert path.suffix != ".jpg", "published JPEG must not be truncated in place"
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", no_direct_jpeg_write)
    root = tmp_path / "library"
    result = analyze_edited_clip(clip, style_library_root=root)
    assert all(cv2.imread(str(root / frame.path)) is not None for frame in result.keyframes)
    assert not list(root.rglob("*.tmp"))
