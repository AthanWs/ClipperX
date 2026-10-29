"""Pure sample-grid and generation provenance contracts; no native processes."""

from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from clipperx.render_clock import SourceAudioDecodeReceipt, SourceTimeOrigin, audio_decode_argv
from clipperx.tests.test_render_clock import audio_payload, make_origin
from clipperx.visual_contracts import SourceVisualTimeline


def provenance() -> Any:
    assert importlib.util.find_spec("clipperx.visual_provenance"), "pure provenance API absent"
    from clipperx import visual_provenance
    return visual_provenance


def sample_receipt(*, phase: F = F(0)) -> tuple[SourceAudioDecodeReceipt, SourceTimeOrigin]:
    base = make_origin()
    # A fractional common origin yields an exact nonzero source phase without
    # inventing PCM or an inconsistent output sample-grid origin.
    origin = SourceTimeOrigin(base.path, base.size, base.mtime_ns, base.fingerprint,
                              base.origin - phase, base.duration, base.streams, base.decoder_id, {})
    payload = audio_payload()
    payload["audio"][0]["source_phase"] = str(phase)
    payload["audio"][0]["coverage"][0]["status"] = "unavailable" if phase > 0 else "covered"
    payload["argv"] = [audio_decode_argv("ffmpeg", origin, F(0), F(1, 10), "render48k-mono")]
    return SourceAudioDecodeReceipt(origin.identity, origin.decoder_id, "HOST;audio-only;v1", payload), origin


def test_audio_mapping_half_ties_and_absolute_endpoint_count() -> None:
    p = provenance()
    receipt, origin = sample_receipt(phase=F(1, 100000))
    phase = F(1, 100000)
    start, end = phase + F(1, 96000), phase + F(3, 96000)
    mapped = p.map_source_audio_window(receipt, origin, start, end)
    assert (mapped.first, mapped.count, mapped.rate) == (1, 1, 48000)
    assert mapped.requested_start == start and mapped.requested_end == end
    assert mapped.included_start == phase + F(1, 48000)
    assert mapped.included_end == phase + F(2, 48000)
    assert mapped.start_delta == mapped.end_delta == F(1, 96000)
    assert mapped.observation_id == json.loads(receipt.to_json())["sha256"]
    # A duration of 1.02 samples rounds to one; absolute endpoints give two.
    mapped = p.map_source_audio_window(receipt, origin, phase + F(49, 4800000), phase + F(151, 4800000))
    assert (mapped.first, mapped.count) == (0, 2)


@pytest.mark.parametrize("start,end", [(F(0), F(1, 100)), (F(1, 100000), F(11, 1000000)),
                                       (F(1, 100), F(11, 100))])
def test_audio_mapping_uncovered_or_empty_is_explicit(start: F, end: F) -> None:
    p = provenance()
    receipt, origin = sample_receipt(phase=F(1, 100000))
    with pytest.raises(p.AudioUnavailableError):
        p.map_source_audio_window(receipt, origin, start, end)


def test_audio_mapping_stale_receipt_errors_are_not_silence() -> None:
    p = provenance()
    receipt, _ = sample_receipt(phase=F(1, 100000))
    with pytest.raises(ValueError, match="origin"):
        p.map_source_audio_window(receipt, make_origin(), .01, .02)


def test_identical_pcm_hash_different_clock_has_distinct_observation_identity() -> None:
    p = provenance()
    first, origin1 = sample_receipt()
    shifted, origin2 = sample_receipt(phase=F(1, 100000))
    assert json.loads(first.payload_json)["audio"][0]["pcm_sha256"] == json.loads(shifted.payload_json)["audio"][0]["pcm_sha256"]
    left = p.map_source_audio_window(first, origin1, .01, .02)
    right = p.map_source_audio_window(shifted, origin2, .01, .02)
    assert left.observation_id != right.observation_id
    assert left.included_start != right.included_start


def test_timeline_fingerprint_matches_existing_unicode_default_spaces(tmp_path: Path) -> None:
    p = provenance()
    timeline = SourceVisualTimeline(source_fingerprint="évidénce", duration_seconds=2,
        analyzer_version="héritage", model_hashes={}, sampling_config={}, samples=[],
        spans=[], face_tracks=[], webcam_tracks=[])
    raw = json.dumps(timeline.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, allow_nan=False)
    assert p.visual_timeline_fingerprint(timeline) == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"host source")
    path = tmp_path / "legacy.json"
    path.write_text(timeline.model_dump_json(), encoding="utf-8")
    context = p.load_visual_provenance(path, source_video=source)
    assert context.status == "unverified_legacy"
    detached = context.to_record()
    detached["status"] = "mutated"
    assert context.to_record()["status"] == "unverified_legacy"


def _generation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, SourceTimeOrigin]:
    from clipperx.tests.test_visual_timeline import _install_observed_boundary
    from clipperx.visual_timeline import build_source_visual_timeline_result
    source, origin, _ = _install_observed_boundary(monkeypatch, tmp_path)
    return build_source_visual_timeline_result(source).artifact_path, source, origin


def _reseal(path: Path, *, timeline_changed: bool = False) -> None:
    p = provenance()
    from clipperx.render_clock import canonical, digest
    root = path.parent
    marker = json.loads((root / "complete.json").read_text())
    marker["catalog"] = p.generation_catalog(root)
    marker["clock_receipt_sha256"] = p.file_sha256(root / "clock-receipt.json")
    marker["timeline_sha256"] = p.file_sha256(path)
    if timeline_changed:
        marker["timeline_fingerprint"] = p.visual_timeline_fingerprint(SourceVisualTimeline.model_validate_json(path.read_text()))
    marker["sha256"] = digest(canonical({k: v for k, v in marker.items() if k != "sha256"}))
    (root / "complete.json").write_text(canonical(marker), encoding="utf-8")


@pytest.mark.parametrize("fault", ["pts", "sample_id", "preview", "preview_swap", "audio_count", "audio_bool", "timeline_extra", "wav_body",
    "missing_association", "duplicate_association", "config_bool", "missing_origin", "missing_complete"])
def test_observed_generation_checks_associations_beyond_catalog_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str,
) -> None:
    p = provenance()
    path, source, origin = _generation(monkeypatch, tmp_path)
    clock_path = path.parent / "clock-receipt.json"
    clock = json.loads(clock_path.read_text())
    if fault == "pts":
        clock["sample_associations"][0]["raw_pts"] += 1
    elif fault == "sample_id":
        clock["sample_associations"][0]["sample_id"] = "invented"
    elif fault == "preview":
        clock["sample_associations"][0]["preview"] = clock["sample_associations"][1]["preview"]
    elif fault == "preview_swap":
        first, second = clock["sample_associations"][:2]
        for field in ("preview", "preview_sha256"):
            first[field], second[field] = second[field], first[field]
    elif fault == "audio_count":
        clock["audio_associations"][0]["slice"]["count"] += 1
    elif fault == "audio_bool":
        clock["audio_associations"][0]["slice"]["first"] = False
    elif fault == "timeline_extra":
        timeline = json.loads(path.read_text())
        timeline["unrecognized_qualification"] = True
        path.write_text(json.dumps(timeline))
    elif fault == "wav_body":
        wav = next(path.parent.glob("*.wav"))
        data = bytearray(wav.read_bytes())
        data[-1] ^= 1
        wav.write_bytes(data)
    elif fault == "missing_association":
        clock["sample_associations"].pop()
    elif fault == "duplicate_association":
        clock["sample_associations"].append(clock["sample_associations"][0])
    elif fault == "config_bool":
        # A bool can equal 0.0 numerically; exact emitted type must still fail.
        clock["config"]["sampling"]["editorial_boundary_count"] = False
        marker = json.loads((path.parent / "complete.json").read_text())
        marker["config"] = clock["config"]
        (path.parent / "complete.json").write_text(json.dumps(marker))
    elif fault == "missing_complete":
        (path.parent / "complete.json").unlink()
    if fault not in ("missing_origin", "missing_complete"):
        clock_path.write_text(json.dumps(clock))
        _reseal(path)
    with pytest.raises(ValueError):
        p.load_visual_provenance(path, source_video=source, current_origin=None if fault == "missing_origin" else origin)


@pytest.mark.parametrize("relative", ["../escape", "C:/escape", "/escape", "nested/../escape", "a\\b", "a//b", "./a"])
def test_generation_paths_reject_traversal_and_absolute(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError):
        provenance().generation_file(tmp_path, relative)


def test_generation_paths_reject_escaped_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "generation"
    root.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit test symlink creation")
    with pytest.raises(ValueError, match="escapes"):
        provenance().generation_file(root, "link/file")


def test_provenance_does_not_reopen_receipts_with_unbounded_text_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    p = provenance()
    path, source, origin = _generation(monkeypatch, tmp_path)
    def forbidden(*args: Any, **kwargs: Any) -> str:
        pytest.fail("unbounded read_text bypasses bounded generation JSON reader")
    monkeypatch.setattr(Path, "read_text", forbidden)
    assert p.load_visual_provenance(path, source_video=source, current_origin=origin).status == "observed_samples"


@pytest.mark.parametrize("fault", ["missing_receipt", "video_receipt", "bool_reason", "empty_reason", "false_unavailable", "partial_covered"])
def test_review_i1_audio_negative_association_requires_actual_mapper_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str,
) -> None:
    from clipperx.tests.test_visual_timeline import _install_observed_boundary
    from clipperx.visual_timeline import build_source_visual_timeline_result
    p = provenance()
    valid_negative = fault not in ("false_unavailable", "partial_covered")
    source, origin, _ = _install_observed_boundary(monkeypatch, tmp_path,
        audio_prefix=F(1, 16000) if valid_negative else F(0))
    path = build_source_visual_timeline_result(source, sparse_interval=1).artifact_path
    clock_path = path.parent / "clock-receipt.json"
    clock = json.loads(clock_path.read_text())
    timeline = json.loads(path.read_text())
    row = clock["audio_associations"][0]
    if fault == "missing_receipt":
        row["receipt"] = "does-not-exist.json"
    elif fault == "video_receipt":
        row["receipt"] = clock["sample_associations"][0]["receipt"]
    elif fault == "bool_reason":
        row["reason"] = True
    elif fault == "empty_reason":
        row["reason"] = ""
    elif fault == "false_unavailable":
        row.update(status="unavailable", slice=None, reason="Requested unrounded audio coverage unavailable")
        timeline["samples"][0].update(audio_features={}, audio_window=None)
    else:
        audio_row = next(o for o in clock["observations"] if o["kind"] == "audio")
        receipt_path = path.parent / audio_row["path"]
        envelope = json.loads(receipt_path.read_text())
        receipt = SourceAudioDecodeReceipt.from_json(receipt_path.read_text(), origin=origin, policy=envelope["policy"])
        mapped = p.map_source_audio_window(receipt, origin, 0, F(1, 2))
        row["slice"] = mapped.to_record()
        timeline["samples"][0]["audio_window"]["end"] = .5
        timeline["samples"][0]["audio_features"]["end"] = .5
    path.write_text(json.dumps(timeline))
    clock_path.write_text(json.dumps(clock))
    _reseal(path, timeline_changed=True)
    with pytest.raises(ValueError):
        p.load_visual_provenance(path, source_video=source, current_origin=origin)


@pytest.mark.parametrize("empty", [False, True])
def test_review_i1_valid_unavailable_audio_prefix_or_empty_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, empty: bool,
) -> None:
    from clipperx.tests.test_visual_timeline import _install_observed_boundary, _NoRectangleAnalyzer
    from clipperx.visual_timeline import build_source_visual_timeline_result
    source, origin, _ = _install_observed_boundary(monkeypatch, tmp_path,
        duration=.000119 if empty else 4., audio_prefix=F(0) if empty else F(1, 16000),
        local_analyzer=_NoRectangleAnalyzer())
    path = build_source_visual_timeline_result(source, sparse_interval=.000002 if empty else 1.).artifact_path
    assert provenance().load_visual_provenance(path, source_video=source, current_origin=origin).status == "observed_samples"
    clock = json.loads((path.parent / "clock-receipt.json").read_text())
    assert clock["audio_associations"][0]["status"] == "unavailable"


@pytest.mark.parametrize("fault", ["covered_overlap", "invented_gap", "wrong_reason", "bool_reason", "duplicate", "mask_hole", "deleted_covered"])
def test_review_i2_negative_video_requires_supported_noncontradictory_intervals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str,
) -> None:
    from clipperx.tests.test_visual_timeline import _install_observed_boundary
    from clipperx.visual_timeline import build_source_visual_timeline_result
    p = provenance()
    prefix = .015 if fault in ("wrong_reason", "bool_reason", "duplicate", "mask_hole") else 0.
    source, origin, _ = _install_observed_boundary(monkeypatch, tmp_path, prefix=prefix)
    path = build_source_visual_timeline_result(source).artifact_path
    clock_path = path.parent / "clock-receipt.json"
    clock, timeline = json.loads(clock_path.read_text()), json.loads(path.read_text())
    def unknown(start: float, end: float, reason: str = "missing_predecessor") -> dict[str, Any]:
        return {"span_id": f"unknown_{start}_{end}", "source": {"start": start, "end": end}, "state": "unknown",
            "confidence": 0., "streamer_track_id": None, "identity_status": "unavailable", "webcam_track_id": None,
            "important_regions": [], "text_regions": [], "evidence_ids": [], "change_reasons": ["observed_unavailable:" + reason]}
    if fault in ("covered_overlap", "invented_gap"):
        a, b = ("0/1", "4/1") if fault == "covered_overlap" else ("1/2", "3/4")
        clock["unavailable_intervals"] = [{"start": a, "end": b, "reason": "missing_predecessor"}]
        timeline["spans"] = [unknown(0., 4.)]
    elif fault in ("wrong_reason", "bool_reason"):
        clock["unavailable_intervals"][0]["reason"] = "unproven_gap" if fault == "wrong_reason" else True
    elif fault == "duplicate":
        clock["unavailable_intervals"].append(clock["unavailable_intervals"][0])
    elif fault == "mask_hole":
        timeline["spans"] = [unknown(0., .005, "before_video_start"), unknown(.01, 4., "before_video_start")]
    else:
        for preview in path.parent.glob("*.jpg"):
            preview.unlink()
        timeline.update(samples=[], face_tracks=[], webcam_tracks=[], spans=[unknown(0., 4., "no_observed_samples")])
        clock.update(sample_associations=[], audio_associations=[],
            unavailable_intervals=[{"start": "0/1", "end": "4/1", "reason": "no_observed_samples"}])
        marker_path = path.parent / "complete.json"
        marker = json.loads(marker_path.read_text())
        marker["creation_work"]["indexed_samples"] = 0
        marker_path.write_text(json.dumps(marker))
    path.write_text(json.dumps(timeline))
    clock_path.write_text(json.dumps(clock))
    _reseal(path, timeline_changed=True)
    with pytest.raises(ValueError):
        p.load_visual_provenance(path, source_video=source, current_origin=origin)


@pytest.mark.parametrize("case", ["prefix15", "prefix59", "rational", "rational_scope", "internal", "terminal", "no_samples"])
def test_review_i2_valid_supported_absence_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str,
) -> None:
    from clipperx.tests.test_visual_timeline import _install_observed_boundary, _NoRectangleAnalyzer
    from clipperx.visual_timeline import build_source_visual_timeline_result
    prefix = F(1, 3) if case == "rational" else .015 if case == "prefix15" else .059 if case == "prefix59" else 0.
    source, origin, _ = _install_observed_boundary(monkeypatch, tmp_path, prefix=prefix,
        duration=F(1, 3) if case == "rational_scope" else .1 if case == "no_samples" else 8. if case == "internal" else 4.,
        terminal=case in ("terminal", "no_samples", "rational_scope"), late_window_at=F(3) if case == "internal" else None,
        local_analyzer=_NoRectangleAnalyzer())
    path = build_source_visual_timeline_result(source, sparse_interval=3 if case == "internal" else 1).artifact_path
    assert provenance().load_visual_provenance(path, source_video=source, current_origin=origin).status == "observed_samples"
    clock = json.loads((path.parent / "clock-receipt.json").read_text())
    intervals = clock["unavailable_intervals"]
    expected = "no_observed_samples" if case in ("no_samples", "rational_scope") else "missing_predecessor" if case == "internal" else "unknown_terminal" if case == "terminal" else "before_video_start"
    assert any(expected in row["reason"] for row in intervals)


def test_review_i2_no_samples_cannot_carry_unobserved_tracks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from clipperx.tests.test_visual_timeline import _install_observed_boundary
    from clipperx.visual_timeline import build_source_visual_timeline_result
    source, origin, _ = _install_observed_boundary(monkeypatch, tmp_path, duration=.1, terminal=True)
    path = build_source_visual_timeline_result(source).artifact_path
    timeline = json.loads(path.read_text())
    timeline["face_tracks"] = [{"track_id": "invented", "first_seen": 0., "last_seen": .05, "observation_count": 1,
        "embedding_centroid": [1.], "positions": [{"x": .1, "y": .1, "width": .1, "height": .1}],
        "recurrent_states": ["unknown"], "full_camera_seconds": 0., "mean_confidence": .9, "evidence_ids": ["invented"]}]
    path.write_text(json.dumps(timeline))
    _reseal(path, timeline_changed=True)
    with pytest.raises(ValueError, match="tracks"):
        provenance().load_visual_provenance(path, source_video=source, current_origin=origin)
