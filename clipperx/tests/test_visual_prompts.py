import json

from clipperx.visual_prompts import VISUAL_DIRECTOR_SYSTEM_PROMPT, bounded_transcript


def test_sfx_transient_anchors_to_observed_event_without_shifting_or_truncating() -> None:
    guide = VISUAL_DIRECTOR_SYSTEM_PROMPT
    assert len(guide) <= 6500
    for term in ["transient_offset_seconds", "cue.start = anchor - transient_offset_seconds",
                 "anchor is shot-relative", "start<0", "full asset cannot fit", "omit",
                 "never shift the event", "authorized_sfx", "duck_source_db=0"]:
        assert term in guide


def test_serialized_director_context_carries_current_creator_preferences() -> None:
    from clipperx.tests.test_visual_contracts import make_timeline
    from clipperx.planner_contracts import ClipPlan, TimeRange
    from clipperx.visual_prompts import build_visual_director_prompt
    clip = ClipPlan.model_validate(dict(clip_id="clip", topic_id="topic", reasoning="source-grounded moment",
        headline=dict(overlay_text="Original headline", platform_title="title"),
        segments=[dict(source=dict(start=0, end=4), output_order=0, narrative_role="payoff",
                       layout="adaptive", reason="preserve the payoff")]))
    payload = json.loads(build_visual_director_prompt(clip, make_timeline(), windows=[TimeRange(start=0,end=4)],
        transcript=[], style_cards=[], frame_manifest=[], host_context={}, safe_zones={}))
    prefs = payload.get("creator_preferences")
    assert isinstance(prefs, dict)
    assert prefs["brand"] == {"asset_id":"kick_brand", "persistent":True, "independent_of_title":True}
    assert prefs["subtitles"] == {
        "primary_color":"#FFFF00", "outline_color":"#000000", "outline_width":8,
        "font_size":72,"max_words_per_event":4,
        "placement_policy":"near_content_action_without_occlusion",
        "avoid_forcing_caption_to_camera":True,"preserve_game_hud":True,
        "keep_stable_within_scene":True,
    }
    assert prefs["camera"]["policy"] == "adaptive_to_moment"
    assert prefs["emphasis"]["requires_source_reason"] is True
    assert prefs["silence"]["preserve_reaction_suspense_and_payoff"] is True
    assert payload["clip"]["headline"]["overlay_text"] == "Original headline"
    assert payload["clip"].get("reasoning") == "source-grounded moment"
    assert payload["clip"]["segments"][0]["source"] == {"start":0.0,"end":4.0}


def test_system_prompt_defines_geometry_and_immutable_sources() -> None:
    for term in ["geometry", "immutable", "track", "HUD", "OCR", "cover",
                 "centered", "SFX", "VisualDirectorResponse", "evidence"]:
        assert term in VISUAL_DIRECTOR_SYSTEM_PROMPT


def test_ready_protocol_guide_covers_host_cross_field_requirements_within_budget() -> None:
    guide = VISUAL_DIRECTOR_SYSTEM_PROMPT
    assert len(guide) <= 6500
    for term in ["blur_sigma > 0 and <= 50", "blurred cover backdrop", "sharp contained foreground",
                 "black destination", "opacity=1", "inherit", "solid", "review_required",
                 "full-canvas", "split foregrounds alone", "higher layers", "179",
                 "RELATIVE TO SHOT", "start at 0", "end at shot duration", "continuous",
                 "mandatory", "exactly headline.overlay_text", "text XOR asset_id",
                 "headline_impact", "subtitle_bold", "kick_brand", "No SFX cues",
                 "five host fingerprints", "supplied", "unreviewed compatible",
                 "visual inspiration", "human approval", "narrative score", "original-VOD pairing",
                 "content axis", "source dimensions"]:
        assert term in guide


def test_system_prompt_explains_shot_local_source_image_citation_authority() -> None:
    guide = VISUAL_DIRECTOR_SYSTEM_PROMPT
    assert len(guide) <= 6500
    for term in ["local_source_image_ids_by_shot", "SOURCE IMAGE", "[start, end)",
                 "captured", "sample_refs.requested", "representative at", "lookahead",
                 "does not authorize", "metadata span/sample", "style-image IDs",
                 "empty", "invent", "request more evidence", "small relevant subset"]:
        assert term in guide


def test_system_prompt_explains_exact_source_reference_semantics() -> None:
    from typing import cast

    import pytest
    from requests import Session

    from clipperx.structured_client import (
        OpenAICompatibleStructuredClient, StructuredModelConfig, StructuredResponseError,
    )
    from clipperx.tests.test_structured_client import FakeSession
    from clipperx.visual_contracts import VisualDirectorResponse

    session = FakeSession()
    client = OpenAICompatibleStructuredClient(StructuredModelConfig(
        provider="cliproxy", base_url="http://localhost:8317/v1", api_key="fake",
        model="light-model", context_window_tokens=65536,
    ), session=cast(Session, session))
    # The fake returns another contract; inspect what the real client serialized.
    with pytest.raises(StructuredResponseError) as caught:
        client.generate(VisualDirectorResponse, system_prompt=VISUAL_DIRECTOR_SYSTEM_PROMPT,
                        user_prompt="local evidence")
    assert caught.value.failure_kind == "model_validation"
    assert session.payload is not None
    prompt = session.payload["messages"][0]["content"]
    for term in ['source_ref="source"', 'kind="source_video"', '"source_video"',
                 "Every non-streamer-camera", "STREAMER_CAMERA alone", "locally applicable",
                 "confident identity", "own valid crop", "empty or null"]:
        assert term in prompt
    schema = session.payload["response_format"]["json_schema"]["schema"]
    assert 'source_ref="source"' in schema["$defs"]["VisualLayer"]["properties"][
        "source_ref"]["description"]


def test_transcript_filters_distant_lines_words_and_biometric_arrays() -> None:
    lines = [dict(start=60, end=61, text="x" * 100, embedding=[9], words=[
        dict(start=60, end=61, word="local"), dict(start=5, end=6, word="SECRET")]),
        dict(start=1, end=2, text="DISTANT")]
    result = bounded_transcript(lines, [(58, 66)], max_chars=350)
    encoded = json.dumps(result)
    assert "SECRET" not in encoded and "DISTANT" not in encoded and "embedding" not in encoded
    assert len(encoded) <= 350


def test_transcript_coarse_row_never_exports_distant_text() -> None:
    result = bounded_transcript([
        dict(start=0, end=90, text="DISTANT_TEXT", words=[
            dict(start=60, end=61, word="local"), dict(start=5, end=6, word="DISTANT_WORD")])
    ], [(59, 63)], max_chars=8000)
    assert "DISTANT" not in json.dumps(result)
    assert result[0]["text"] == "local"


def test_prompt_rejects_unbounded_local_observations() -> None:
    import pytest
    from clipperx.tests.test_visual_contracts import make_timeline, make_webcam_track
    from clipperx.planner_contracts import ClipPlan, TimeRange
    from clipperx.visual_prompts import build_visual_director_prompt
    webcam = make_webcam_track()
    webcam.observations *= 257
    timeline = make_timeline(webcam_tracks=[webcam])
    clip = ClipPlan.model_validate(dict(clip_id="clip", topic_id="topic", reasoning="reason",
        headline=dict(overlay_text="headline", platform_title="title"),
        segments=[dict(source=dict(start=0, end=4), output_order=0, narrative_role="payoff",
                       layout="adaptive", reason="reason")]))
    with pytest.raises(ValueError, match="bounded"):
        build_visual_director_prompt(clip, timeline, windows=[TimeRange(start=0, end=4)],
            transcript=[], style_cards=[], frame_manifest=[], host_context={}, safe_zones={})


def test_prompt_compacts_dense_regions_without_losing_protection() -> None:
    from clipperx.tests.test_visual_contracts import make_timeline, make_span
    from clipperx.planner_contracts import ClipPlan, TimeRange
    from clipperx.visual_contracts import NormalizedRect
    from clipperx.visual_prompts import build_visual_director_prompt
    regions = [NormalizedRect(x=(i % 20) / 25, y=(i // 20) / 25, width=.04, height=.04)
               for i in range(400)]
    timeline = make_timeline(spans=[make_span(text_regions=regions, important_regions=regions)])
    clip = ClipPlan.model_validate(dict(clip_id="clip", topic_id="topic", reasoning="reason",
        headline=dict(overlay_text="headline", platform_title="title"),
        segments=[dict(source=dict(start=0, end=4), output_order=0, narrative_role="payoff",
                       layout="adaptive", reason="reason")]))
    payload = json.loads(build_visual_director_prompt(clip, timeline,
        windows=[TimeRange(start=0, end=4)], transcript=[], style_cards=[], frame_manifest=[],
        host_context={}, safe_zones={}))
    span = payload["source_spans"][0]
    assert len(span["text_regions"]) == 1 and len(span["important_regions"]) == 1
    assert span["region_summary"]["text_candidate_count"] == 400
    bounds = span["text_regions"][0]
    assert all(bounds["x"] <= r.x and bounds["y"] <= r.y
               and bounds["x"] + bounds["width"] >= r.x + r.width
               and bounds["y"] + bounds["height"] >= r.y + r.height for r in regions)
    assert "not verified OCR" in span["region_summary"]["semantics"]
