from clipperx.plan_render import candidate_from_plan
from pathlib import Path
from typing import Any
import pytest
from clipperx.planner_contracts import (
    ClipPlan,
    EditSegment,
    HeadlinePlan,
    NormalizedCrop,
    TimeRange,
)


def test_plan_keeps_non_chronological_output_order() -> None:
    plan = ClipPlan(
        clip_id="clip-1",
        topic_id="topic-1",
        headline=HeadlinePlan(overlay_text="Hook", platform_title="Título"),
        segments=[
            EditSegment(
                source=TimeRange(start=20, end=22),
                output_order=1,
                narrative_role="payoff",
                layout="full_content",
                reason="payoff",
            ),
            EditSegment(
                source=TimeRange(start=5, end=8),
                output_order=0,
                narrative_role="hook",
                layout="picture_in_picture",
                camera_crop=NormalizedCrop(x=0, y=0, width=0.25, height=0.25),
                reason="hook",
            ),
        ],
        reasoning="test",
    )

    candidate = candidate_from_plan(plan)

    assert candidate.time_ranges == [(5.0, 8.0), (20.0, 22.0)]
    assert candidate.layout_segments is not None
    assert candidate.layout_segments[0][2] == "PICTURE_IN_PICTURE"


def test_explicit_legacy_uses_current_creator_defaults_and_keeps_source_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.plan_render import render_legacy_composition_plan
    import clipperx.editing as editing
    import clipperx.face as face
    import clipperx.media as media
    from clipperx.planner_contracts import GlobalCompositionResponse
    clip = ClipPlan(clip_id="legacy", topic_id="topic", headline=HeadlinePlan(overlay_text="Hook", platform_title="Title"),
        segments=[EditSegment(source=TimeRange(start=5, end=6), output_order=0,
            narrative_role="hook", layout="full_content", reason="legacy")], reasoning="fixture")
    payload = GlobalCompositionResponse(clips=[clip]).model_dump_json()
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: payload)
    sidecars: list[str] = []
    monkeypatch.setattr(Path, "write_text", lambda self, text, **kwargs: sidecars.append(text))
    monkeypatch.setattr(media, "probe_media", lambda _: {"video": {"width": 1920, "height": 1080}})
    monkeypatch.setattr(face, "load_face_model", lambda: pytest.fail("full content needs no face model"))
    calls: list[Any] = []
    def render(*args: Any, **kwargs: Any) -> list[Path]:
        calls.append((args, kwargs))
        return [Path("O:/HOST/legacy.mp4")]
    monkeypatch.setattr(editing, "render_batch", render)
    assert render_legacy_composition_plan(Path("source"), Path("plan"), Path("out")) == [Path("O:/HOST/legacy.mp4")]
    assert calls[0][0][1][0].time_ranges == [(5, 6)]
    assert calls[0][1]["layout"].caption_template == "default" and len(sidecars) == 1


def test_rejected_segment_camera_is_not_inherited_from_previous_segment(monkeypatch, tmp_path):
    from clipperx.plan_render import render_legacy_composition_plan
    from clipperx.planner_contracts import GlobalCompositionResponse
    from clipperx import face, media, editing
    clip = ClipPlan(clip_id="moving", topic_id="t", headline=HeadlinePlan(overlay_text="Hook", platform_title="Title"),
        segments=[EditSegment(source=TimeRange(start=a, end=b), output_order=i,
            narrative_role="hook", layout="split", reason="camera control")
            for i, (a, b) in enumerate([(0, 4), (4, 8)])], reasoning="fixture")
    path = tmp_path/"plan.json"
    path.write_text(GlobalCompositionResponse(clips=[clip]).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(media, "probe_media", lambda _: {"video": {"width":1280,"height":720}})
    class Detector:
        def close(self):
            pass
    monkeypatch.setattr(face, "load_face_model", Detector)
    responses = iter([face.WebcamRegion(0, 0, 230, 190, "top-left", .05), None])
    monkeypatch.setattr(face, "detect_webcam_region", lambda *a, **k: next(responses))
    captured = []
    layouts = []
    def render(source, clips, *a, **k):
        captured.extend(clips)
        layouts.append(k["layout"])
        return []
    monkeypatch.setattr(editing, "render_batch", render)
    render_legacy_composition_plan(Path("source.mp4"), path, tmp_path/"out")
    assert captured[0].segment_face_crops == {0:(0,0,230,190), 1:None}
    assert layouts[0].camera_fit == "contain"
