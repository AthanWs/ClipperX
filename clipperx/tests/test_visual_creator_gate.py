from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from clipperx.jobs import JobStore
from clipperx.tests.test_visual_director import FakeClient, ready, scene as director_scene  # noqa: F401
from clipperx.visual_contracts import SourceVisualTimeline
from clipperx.visual_director import (
    DirectorPublicationError, VisualDirectorLimits, recover_visual_director_publication,
    run_visual_director,
)


@pytest.fixture
def scene(director_scene: Any, tmp_path: Path) -> Any:
    # Historical prompt tests deliberately use unsorted text. Creator captions
    # consume valid chronological ASR without repairing or sorting production input.
    path = JobStore.for_video(director_scene[0], jobs_root=tmp_path / "jobs").transcript_path
    rows = json.loads(path.read_text())
    path.write_text(json.dumps(sorted(rows, key=lambda row: row["start"])), encoding="utf-8")
    return director_scene


def creator_ready(payload: dict[str, Any]) -> dict[str, Any]:
    response = ready(payload)
    for index, shot in enumerate(response["plan"]["shots"]):
        shot["evidence_ids"] = [next(span["span_id"] for span in payload["source_spans"]
            if span["source"]["start"] <= shot["source"]["start"] < span["source"]["end"])]
        duration = shot["output"]["end"] - shot["output"]["start"]
        shot["brand"] = dict(asset_id="kick_brand", start=0, end=duration,
            destination=dict(x=.35, y=.25, width=.3, height=.05), z_index=4)
        if index == 0:
            shot["title"] = dict(text=payload["clip"]["headline"]["overlay_text"],
                font_asset_id="headline_impact", start=0, end=min(1, duration),
                destination=dict(x=.1, y=.1, width=.8, height=.1), z_index=3)
        shot["subtitles"] = dict(destination=dict(x=.2, y=.7, width=.6, height=.08),
            font_asset_id="subtitle_bold", font_size=72, primary_color="#FFFF00",
            outline_color="#000000", outline_width=8, max_words_per_event=4)
    return response


def run_creator(scene: Any, tmp_path: Path, response: Any = creator_ready,
                **kwargs: Any) -> Path:
    video, clip, timeline, library = scene
    return run_visual_director(video, clip, timeline, client=FakeClient([response]),
        style_library=library, limits=VisualDirectorLimits(require_creator_overlays=True),
        jobs_root=tmp_path / "jobs", **kwargs)


def write_transcript(scene: Any, tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = JobStore.for_video(scene[0], jobs_root=tmp_path / "jobs").transcript_path
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_limits_preserve_historical_default_and_reject_non_bool() -> None:
    assert getattr(VisualDirectorLimits(), "require_creator_overlays", None) is False
    assert asdict(VisualDirectorLimits(require_creator_overlays=True))["require_creator_overlays"] is True
    for value in (0, 1, "true", None):
        with pytest.raises(ValueError, match="bool"):
            VisualDirectorLimits(require_creator_overlays=value)  # type: ignore[arg-type]


def test_conservative_geometry_ready_is_creator_review(scene: Any, tmp_path: Path) -> None:
    path = run_creator(scene, tmp_path, ready)
    assert path.name == "receipt.json"
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "review_required"
    assert "brand" in receipt["reason"]
    assert not (path.parent / "visual_plan.json").exists()


def test_valid_creator_plan_publishes_and_recovers(scene: Any, tmp_path: Path) -> None:
    path = run_creator(scene, tmp_path)
    assert path.name == "visual_plan.json"
    assert recover_visual_director_publication(scene[0], path.parent,
        jobs_root=tmp_path / "jobs") == path
    authority = json.loads((path.parent / "authority.json").read_text())
    assert authority["limits"]["require_creator_overlays"] is True
    payload = json.loads((path.parent / "call-01" / "request.json").read_text())["user_payload"]
    requirements = payload["clip"]["host_context"]["creator_overlays"]
    assert requirements["required"] is True
    assert requirements["brand"]["full_shot"] is True
    assert requirements["subtitles"]["when_local_speech"] is True


@pytest.mark.parametrize("defect", ["missing_brand", "partial_brand", "late_brand",
    "missing_title", "late_title", "missing_captions", "wrong_color", "small_font",
    "weak_outline", "wrong_outline", "wrong_font"])
def test_creator_rejects_incomplete_or_weak_overlays(scene: Any, tmp_path: Path,
                                                    defect: str) -> None:
    def broken(payload: Any) -> Any:
        response = creator_ready(payload)
        first = response["plan"]["shots"][0]
        if defect == "missing_brand":
            response["plan"]["shots"][-1]["brand"] = None
        elif defect == "partial_brand":
            first["brand"]["end"] -= .1
        elif defect == "late_brand":
            first["brand"]["start"] = .1
        elif defect == "missing_title":
            first["title"] = None
        elif defect == "late_title":
            first["title"]["start"] = .1
        elif defect == "missing_captions":
            first["subtitles"] = None
        else:
            key, value = {"wrong_color": ("primary_color", "#FFD51A"),
                "small_font": ("font_size", 58), "weak_outline": ("outline_width", 4),
                "wrong_outline": ("outline_color", "#FFFFFF"),
                "wrong_font": ("font_asset_id", "headline_impact")}[defect]
            first["subtitles"][key] = value
        return response
    path = run_creator(scene, tmp_path, broken)
    assert path.name == "receipt.json"
    assert json.loads(path.read_text())["status"] == "review_required"


def test_known_silence_allows_no_subtitles_but_missing_transcript_does_not(
        scene: Any, tmp_path: Path) -> None:
    text = write_transcript(scene, tmp_path, [])
    def silent(payload: Any) -> Any:
        response = creator_ready(payload)
        for shot in response["plan"]["shots"]:
            shot["subtitles"] = None
        return response
    assert run_creator(scene, tmp_path, silent).name == "visual_plan.json"
    text.unlink()
    path = run_creator(scene, tmp_path, silent)
    assert path.name == "receipt.json"
    assert "transcript" in json.loads(path.read_text())["reason"]


def test_explicit_transcript_is_consumed_without_replacing_canonical(
        scene: Any, tmp_path: Path) -> None:
    canonical = JobStore.for_video(scene[0], jobs_root=tmp_path / "jobs").transcript_path
    before = canonical.read_bytes()
    explicit = tmp_path / "chosen.json"
    explicit.write_text(json.dumps([dict(start=62, end=63, text="explicit chosen speech")]),
        encoding="utf-8")
    path = run_creator(scene, tmp_path, transcript_path=explicit)
    assert path.name == "visual_plan.json"
    assert canonical.read_bytes() == before
    authority = json.loads((path.parent / "authority.json").read_text())
    assert authority["host_context"]["text"]["path"] == str(explicit.resolve())
    assert authority["host_context"]["text"]["sha256"] == hashlib.sha256(explicit.read_bytes()).hexdigest()
    payload = json.loads((path.parent / "call-01" / "request.json").read_text())["user_payload"]
    assert payload["local_transcript"][0]["text"] == "explicit chosen speech"
    explicit.write_text("[]", encoding="utf-8")
    with pytest.raises(DirectorPublicationError):
        recover_visual_director_publication(scene[0], path.parent, jobs_root=tmp_path / "jobs")


def test_explicit_missing_path_does_not_fall_back_to_canonical(scene: Any, tmp_path: Path) -> None:
    path = run_creator(scene, tmp_path, transcript_path=tmp_path / "absent.json")
    assert path.name == "receipt.json"
    assert json.loads(path.read_text())["calls"] == 0


@pytest.mark.parametrize("row,required", [
    (dict(start=60, end=64, text="coarse", words=[dict(start=60, end=60.5, word="first")]), [0]),
    (dict(start=60, end=64, text="coarse", words=[dict(start=61, end=61.5, word="middle")]), [1]),
    (dict(start=60, end=64, text="coarse", words=[dict(start=63, end=63.5, word="last")]), [2]),
    (dict(start=60, end=64, text="coarse", words=[dict(start=61, end=61, word="point")]), [1]),
    (dict(start=60, end=64, text="coarse", words=[dict(start=62, end=62, word="point")]), [2]),
    (dict(start=60, end=64, text="coarse", words=[dict(start=64, end=64, word="outside")]), []),
    (dict(start=60, end=64, text="coarse", words=[dict(start=60.9, end=61, word="edge")]), [0]),
    (dict(start=60, end=64, text="coarse", words=[dict(start=60.9, end=61.1, word="cross")]), [0, 1]),
    (dict(start=61, end=62, text="segment only"), [1]),
    (dict(start=60, end=64, text="  ", words=[]), []),
])
def test_creator_speech_uses_first_middle_last_and_half_open_word_ownership(
        scene: Any, tmp_path: Path, row: dict[str, Any], required: list[int]) -> None:
    video, clip, timeline, library = scene
    spans = []
    for index, (a, b) in enumerate([(0, 61), (61, 62), (62, 100)]):
        spans.append(dict(timeline.spans[0].model_dump(), span_id=f"part-{index}",
            source=dict(start=a, end=b), evidence_ids=[f"span-e-{index}"]))
    local_scene = (video, clip, SourceVisualTimeline.model_validate(dict(timeline.model_dump(), spans=spans)), library)
    write_transcript(local_scene, tmp_path, [row])
    def proposed(payload: Any) -> Any:
        response = creator_ready(payload)
        for index, shot in enumerate(response["plan"]["shots"]):
            if index not in required:
                shot["subtitles"] = None
        return response
    assert run_creator(local_scene, tmp_path, proposed).name == "visual_plan.json"
    if required:
        def omitted(payload: Any) -> Any:
            response = proposed(payload)
            response["plan"]["shots"][required[-1]]["subtitles"] = None
            return response
        assert run_creator(local_scene, tmp_path, omitted).name == "receipt.json"


def reseal_plan(root: Path, plan: dict[str, Any]) -> None:
    import clipperx.visual_director as director
    director._write(root / "visual_plan.json", plan)
    companion = json.loads((root / "provenance.json").read_text())
    companion["plan"] = dict(director._file_record(root / "visual_plan.json"),
        fingerprint=director._hash(plan))
    director._write(root / "provenance.json", companion)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["plan_sha256"] = companion["plan"]["sha256"]
    receipt["companion_sha256"] = director.file_sha256(root / "provenance.json")
    director._write(root / "receipt.json", receipt)


@pytest.mark.parametrize("defect", ["brand", "captions", "title"])
def test_resealed_creator_ready_still_checks_overlay_requirements(scene: Any, tmp_path: Path,
                                                                 defect: str) -> None:
    import clipperx.visual_director as director
    path = run_creator(scene, tmp_path)
    assert path.name == "visual_plan.json"
    plan = json.loads(path.read_text())
    key = "subtitles" if defect == "captions" else defect
    plan["shots"][0][key] = None
    reseal_plan(path.parent, plan)
    with pytest.raises(DirectorPublicationError):
        director._load_terminal(path.parent)
    with pytest.raises(DirectorPublicationError):
        recover_visual_director_publication(scene[0], path.parent, jobs_root=tmp_path / "jobs")


def test_authority_flag_tampering_cannot_downgrade_creator_mode(scene: Any, tmp_path: Path) -> None:
    import clipperx.visual_director as director
    path = run_creator(scene, tmp_path)
    root = path.parent
    authority = json.loads((root / "authority.json").read_text())
    authority["limits"]["require_creator_overlays"] = False
    director._write(root / "authority.json", authority)
    companion = json.loads((root / "provenance.json").read_text())
    companion["authority"] = director._file_record(root / "authority.json")
    director._write(root / "provenance.json", companion)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["companion_sha256"] = director.file_sha256(root / "provenance.json")
    director._write(root / "receipt.json", receipt)
    with pytest.raises(DirectorPublicationError):
        director._load_terminal(root)


def test_historical_missing_creator_flag_remains_geometry_only(scene: Any, tmp_path: Path) -> None:
    import clipperx.visual_director as director
    path = run_visual_director(scene[0], scene[1], scene[2], client=FakeClient([ready]),
        style_library=scene[3], jobs_root=tmp_path / "jobs")
    assert path.name == "visual_plan.json"
    root = path.parent
    authority = json.loads((root / "authority.json").read_text())
    authority["limits"].pop("require_creator_overlays", None)
    authority["host_context"].pop("creator_overlays", None)
    director._write(root / "authority.json", authority)
    companion = json.loads((root / "provenance.json").read_text())
    companion["host_context"] = authority["host_context"]
    companion["authority"] = director._file_record(root / "authority.json")
    director._write(root / "provenance.json", companion)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["host_context"] = authority["host_context"]
    receipt["companion_sha256"] = director.file_sha256(root / "provenance.json")
    director._write(root / "receipt.json", receipt)
    assert director._load_terminal(root)[0] == path


def test_creator_review_recovery_does_not_read_changed_transcript(scene: Any, tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.visual_director as director
    path = run_creator(scene, tmp_path, ready)
    assert path.name == "receipt.json"
    write_transcript(scene, tmp_path, [])
    monkeypatch.setattr(director, "_transcript", lambda *a: pytest.fail("REVIEW opened transcript"))
    monkeypatch.setattr(director, "_check_dependencies", lambda *a: pytest.fail("REVIEW reopened dependencies"))
    assert recover_visual_director_publication(scene[0], path.parent,
        jobs_root=tmp_path / "jobs") == path


def _clock_test_plan() -> tuple[Any, Any]:
    from clipperx.tests.test_output_timeline import shot
    from clipperx.visual_contracts import VisualShot

    shots = []
    for index, (start, end, brand_end) in enumerate([(0, .1, .1), (.1, .4, .3)]):
        value = shot((start, end), (start, end), str(index)).model_dump()
        rect = dict(x=.1, y=.1, width=.2, height=.1)
        value["brand"] = dict(asset_id="kick_brand", start=0, end=brand_end,
                              destination=rect, z_index=1)
        if index == 0:
            value["title"] = dict(text="Title", font_asset_id="headline_impact",
                                  start=0, end=.05, destination=rect, z_index=2)
        shots.append(VisualShot.model_validate(value))
    return SimpleNamespace(shots=shots), SimpleNamespace(headline=SimpleNamespace(overlay_text="Title"))


def test_creator_brand_uses_exact_decimal_shot_duration() -> None:
    from fractions import Fraction
    from clipperx.output_timeline import OutputTimeline
    from clipperx.visual_director import _validate_creator_overlays

    plan, clip = _clock_test_plan()
    timeline = OutputTimeline.from_shots(plan.shots)
    assert timeline.overlay_output_interval("1", 0, .3) == (Fraction(1, 10), Fraction(2, 5))
    _validate_creator_overlays(plan, clip, [])

    for endpoint in (.2999999999999999, .30000000000000004):
        plan.shots[1].brand.end = endpoint
        with pytest.raises(ValueError, match="brand"):
            _validate_creator_overlays(plan, clip, [])


def test_creator_brand_accepts_decimal_end_when_float_duration_rounds_down() -> None:
    from fractions import Fraction
    from clipperx.output_timeline import OutputTimeline
    from clipperx.visual_contracts import VisualShot
    from clipperx.visual_director import _validate_creator_overlays

    plan, clip = _clock_test_plan()
    value = plan.shots[1].model_dump()
    value["source"]["end"] = value["output"]["end"] = .3
    value["brand"]["end"] = .2
    plan.shots[1] = VisualShot.model_validate(value)
    assert OutputTimeline.from_shots(plan.shots).overlay_output_interval("1", 0, .2) == (
        Fraction(1, 10), Fraction(3, 10))
    _validate_creator_overlays(plan, clip, [])


@pytest.mark.parametrize("captions", [False, True])
def test_creator_rejects_unalignable_segment_with_only_blank_words(captions: bool) -> None:
    from clipperx.visual_director import _validate_creator_overlays

    plan, clip = _clock_test_plan()
    plan.shots = plan.shots[:1]
    if captions:
        from clipperx.visual_contracts import SubtitlePlan
        plan.shots[0].subtitles = SubtitlePlan(destination=dict(x=.1, y=.7, width=.8, height=.1),
            font_size=72, primary_color="#FFFF00", outline_color="#000000", outline_width=8)
    rows = [dict(start=0, end=.1, text="spoken text",
                 words=[dict(start=.02, end=.04, word="  ")])]
    with pytest.raises(ValueError, match="align"):
        _validate_creator_overlays(plan, clip, rows)


def test_creator_unalignable_transcript_publishes_review(scene: Any, tmp_path: Path) -> None:
    write_transcript(scene, tmp_path, [dict(start=60, end=61, text="spoken text",
        words=[dict(start=60.2, end=60.4, word=" ")])])
    path = run_creator(scene, tmp_path)
    assert path.name == "receipt.json"
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "review_required"
    assert "align" in receipt["reason"]


def test_creator_blank_words_and_blank_text_remain_known_silence() -> None:
    from clipperx.visual_director import _validate_creator_overlays

    plan, clip = _clock_test_plan()
    plan.shots = plan.shots[:1]
    _validate_creator_overlays(plan, clip, [dict(start=0, end=.1, text=" ",
        words=[dict(start=.02, end=.04, word=" ")])])


@pytest.mark.parametrize("output_end,duration", [(.3, .2), (.4, .3)])
@pytest.mark.parametrize("item", ["brand", "title", "cue", "crop"])
@pytest.mark.parametrize("outside", [False, True])
def test_decimal_shot_limits_preserve_exact_endpoints(output_end: float, duration: float,
                                                       item: str, outside: bool) -> None:
    import math
    from pydantic import ValidationError
    from clipperx.tests.test_output_timeline import shot
    from clipperx.visual_contracts import VisualShot

    value = shot((.1, output_end), (.1, output_end), "fractional").model_dump()
    endpoint = math.nextafter(duration, math.inf) if outside else duration
    rect = dict(x=.1, y=.1, width=.2, height=.1)
    if item in ("brand", "title"):
        identity = {"asset_id": "kick_brand"} if item == "brand" else {"text": "Title"}
        value[item] = dict(**identity, start=0, end=endpoint, destination=rect, z_index=1)
    elif item == "cue":
        value["audio"]["cues"] = [dict(asset_id="sound", start=0, end=endpoint)]
    else:
        crop = dict(x=0, y=0, width=1, height=1)
        value["layers"][0]["crop_track"] = [dict(at=0, crop=crop), dict(at=endpoint, crop=crop)]
    if outside:
        with pytest.raises(ValidationError, match="shot"):
            VisualShot.model_validate(value)
    else:
        VisualShot.model_validate(value)


@pytest.mark.parametrize("end,accepted", [(.1999999995, True), (.199999998, False)])
def test_decimal_crop_track_preserves_existing_closure_tolerance(end: float, accepted: bool) -> None:
    from pydantic import ValidationError
    from clipperx.tests.test_output_timeline import shot
    from clipperx.visual_contracts import VisualShot

    value = shot((.1, .3), (.1, .3), "fractional").model_dump()
    crop = dict(x=0, y=0, width=1, height=1)
    value["layers"][0]["crop_track"] = [dict(at=0, crop=crop), dict(at=end, crop=crop)]
    if accepted:
        VisualShot.model_validate(value)
    else:
        with pytest.raises(ValidationError, match="crop_track"):
            VisualShot.model_validate(value)


def test_unalignable_text_outside_all_shots_does_not_block_creator() -> None:
    from clipperx.visual_director import _validate_creator_overlays

    plan, clip = _clock_test_plan()
    _validate_creator_overlays(plan, clip, [dict(start=1, end=2, text="outside",
        words=[dict(start=1.1, end=1.2, word=" ")])])
