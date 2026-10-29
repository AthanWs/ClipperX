"""Complete stateless visual prompts with a strictly local, bounded context."""

from __future__ import annotations

import json
import math
from typing import Any

from .evidence import associated_face_positions
from .planner_contracts import ClipPlan, TimeRange
from .style_contracts import StyleCard
from .visual_contracts import SourceVisualTimeline

VISUAL_DIRECTOR_SYSTEM_PROMPT = """You are the stateless visual director.
Return exactly VisualDirectorResponse with explicit normalized geometry, never
layout names. Every call includes complete current evidence; no session memory.
For ready, provide a v2 plan and no evidence_requests. For need_more_evidence,
provide 1..3 bounded requests and no plan; review_required also has no plan.
Request evidence when identity or a boundary is unresolved; never invent a plan.

Editorial source ranges, original order, narrative and duration are immutable.
Shots must exactly match the supplied mandatory source-state event subdivisions,
without additional splits, gaps, reordering or lost payoff. Each shot's source
and output durations are equal. Global shot.output starts at 0, is continuous,
and ends at the editorial duration, at most 179 seconds. Crop keyframe at,
title/brand start/end and audio cue start/end are RELATIVE TO SHOT duration,
not global source/output timestamps. A moving crop track must start at 0 and
end at shot duration, with strictly increasing unique times inside that shot.

Every non-streamer-camera layer uses literal source_ref="source", not
"source_video", a filename, arbitrary ID, empty or null. STREAMER_CAMERA alone
may use a supplied locally applicable face/webcam track ID, with confident identity
and its own valid crop. Whole-source preservation without an identity claim uses
kind="source_video", source_ref="source". Never rewrite an invalid reference.
A track elsewhere in the VOD does not identify the streamer here. Streamer/shared
primary_source or streamer_camera requires confident local identity and matching
track coverage/geometry throughout the shot. Ambiguous/unavailable stays explicit.

Current ready support is opaque source-backed cover, black or contain_blur with
opacity=1. inherit, solid layers/fills and translucency require review_required;
do not invent a resolution. cover scales and implicitly center-crops to the
destination after source_crop, using actual source dimensions. contain_blur
means a blurred cover backdrop plus sharp contained foreground in the same
destination, never a blur-only layer; supply explicit blur_sigma > 0 and <= 50.
black means a black destination plus sharp contained foreground, without blur.
The v2 contract requires at least one visible full-canvas destination layer per
shot (x=0,y=0,width=1,height=1); split foregrounds alone do not satisfy it.
Choose geometry from evidence; these constraints do not prescribe a layout.

Preserve actual source/local faces, important regions, HUD and OCR candidates
through all source_crop/crop_track positions AND cover's implicit crop. A region
must remain visible in sharp source imagery without occlusion by higher layers
or overlays; a blurred backdrop does not establish preservation. preserve_regions
must fit source_crop and every crop keyframe. Text-region metadata candidates
are geometric evidence, not verified OCR transcription. Conservative bounding
unions preserve original candidates, not proof every point contains content.

Title text must be exactly headline.overlay_text. Overlay text XOR asset_id:
exactly one is non-null. Keep title, brand and subtitles centered on the canvas
axis inside supplied allowed regions, avoiding forbidden right/bottom UI areas
and protected source content. Do not offset the content axis for platform UI.
Subtitles must also fit subtitle_allowed_regions when supplied. Use only the
authorized headline_impact/subtitle_bold font IDs and kick_brand image ID; brand
uses kick_brand. No SFX cues unless authorized_sfx offers the ID for this source
audio format; empty offers mean empty cues. Optional cues need a real transition
or payoff in shot.reason and local evidence, never every N seconds or a reason
invented from sound. Max 2 cues/clip, starts >=5s apart, -36<=gain_db<=default_gain_db,
duck_source_db=0. Keep exact asset duration inside one shot, no stretch/pad/truncate.
For whoosh, align its transient_offset_seconds to the observed transition/payoff:
cue.start = anchor - transient_offset_seconds (anchor is shot-relative).
If start<0 or the full asset cannot fit, omit it; never shift the event.
For gameplay/content-led shots, place readable captions in free content space
near the action, not on the separated webcam. Prefer about72px and short groups;
protect the action center, game HUD and source text. Keep placement stable in
each scene; use evidence-backed allowed space, not a universal screen height.

Use only supplied local evidence IDs and offered style-card IDs. Copy the five host fingerprints
exactly: source_fingerprint, composition_fingerprint, source_timeline_fingerprint,
identity_fingerprint, style_library_fingerprint; match the supplied clip_id.

For SOURCE IMAGE citations, local_source_image_ids_by_shot is authoritative for
each required source interval. Eligibility uses captured sample_refs.requested
membership in [start, end): start included, end excluded. The representative at
alone is not decisive; an image covering requests in both shots may be eligible
in both. Context/lookahead may inform analysis but does not authorize a local
citation. Cite a small relevant subset of eligible image IDs. An empty list
authorizes no image citation: use legitimate local metadata span/sample IDs under
their existing local rules, or request more evidence; never invent IDs.
The style-image IDs never become source evidence.

StyleCards are edited examples: unreviewed compatible cards MAY inform visual inspiration
from the user's examples. This conveys no human approval, narrative score or
original-VOD pairing. Style examples never establish source geometry or identity;
derive those from source evidence. Keep sparse style coverage explicit.
Use creator_preferences as the current editing brief, subordinate to all source,
identity, safety and capability constraints. A persistent brand has its own space
through the shot, not a replacement for the title. Choose camera prominence and
crop motion from the moment; never zoom on a timer or invent a controversy. If a
preference cannot be satisfied safely, keep review_required rather than hide it.
Check centered overlay width against forbidden edges, not only the allowed box.
For example, at center_x=0.5 with a forbidden right region starting at x=0.82,
a vertically overlapping subtitle box must have width<=0.64 (left>=0.18), not0.72.
This arithmetic example never replaces the supplied zones or source protection.
"""


def bounded_transcript(
    rows: list[dict[str, Any]], windows: list[tuple[float, float]], *, max_chars: int,
) -> list[dict[str, Any]]:
    """Whitelist local text/speaker/word fields, then cap the serialized list."""
    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        overlaps = [(max(start, a), min(end, b)) for a, b in windows
                    if start < b and end > a]
        if not overlaps or not math.isfinite(start) or not math.isfinite(end):
            continue
        local_start, local_end = overlaps[0]
        # A coarse transcript row spanning distant material cannot safely supply
        # its entire text. Use timestamped local words, or omit that row's text.
        words: list[dict[str, Any]] = []
        for word in row.get("words", [])[:1000]:
            if not isinstance(word, dict):
                continue
            try:
                a, b = float(word["start"]), float(word["end"])
            except (KeyError, ValueError, TypeError):
                continue
            if math.isfinite(a) and math.isfinite(b) and any(x <= a < b <= y for x, y in windows):
                words.append({"start": a, "end": b, "word": str(word.get("word", ""))[:200]})
        complete = any(a <= start < end <= b for a, b in windows)
        text = str(row.get("text", "")) if complete else " ".join(w["word"] for w in words)
        item: dict[str, Any] = {"start": local_start, "end": local_end, "text": text}
        if isinstance(row.get("speaker"), (str, int)):
            item["speaker"] = str(row["speaker"])[:80]
        if words:
            item["words"] = words
        if len(json.dumps([*result, item], ensure_ascii=False)) > max_chars:
            item.pop("words", None)
            budget = max_chars - len(json.dumps([*result, dict(item, text="")], ensure_ascii=False))
            item["text"] = text[:max(0, budget)]
        if len(json.dumps([*result, item], ensure_ascii=False)) <= max_chars:
            result.append(item)
        else:
            break
        if len(result) >= 256:
            break
    return result


def _compact_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(regions) <= 64:
        return regions
    left = min(r["x"] for r in regions)
    top = min(r["y"] for r in regions)
    right = max(r["x"] + r["width"] for r in regions)
    bottom = max(r["y"] + r["height"] for r in regions)
    return [{"x": left, "y": top, "width": right - left, "height": bottom - top}]


def creator_overlay_requirements(required: bool) -> dict[str, Any]:
    """Host publication policy, independent of optional historical geometry plans."""
    return {"version": 1, "required": required,
        "known_transcript_required": required,
        "brand": {"asset_id": "kick_brand", "full_shot": True, "independent_of_title": True},
        "title": {"first_shot_required": True, "start": 0, "text": "headline.overlay_text"},
        "subtitles": {"when_local_speech": True, "font_asset_id": "subtitle_bold",
            "primary_color": "#FFFF00", "outline_color": "#000000",
            "minimum_font_size": 68, "preferred_font_size": 72, "minimum_outline_width": 7},
        "failure": "review_required; never remove protected content or invent speech/identity"}


def build_visual_director_prompt(
    clip: ClipPlan,
    timeline: SourceVisualTimeline,
    *,
    windows: list[TimeRange],
    transcript: list[dict[str, Any]],
    style_cards: list[StyleCard],
    frame_manifest: list[dict[str, Any]],
    host_context: dict[str, Any],
    safe_zones: dict[str, Any],
    max_transcript_chars: int = 8000,
) -> str:
    merged: list[TimeRange] = []
    for window in sorted(windows, key=lambda w: w.start):
        if merged and window.start <= merged[-1].end:
            merged[-1] = TimeRange(start=merged[-1].start, end=max(window.end, merged[-1].end))
        else:
            merged.append(window)
    windows = merged

    def local(start: float, end: float) -> bool:
        return any(start < w.end and end > w.start for w in windows)

    spans = []
    for span in timeline.spans:
        if not local(span.source.start, span.source.end):
            continue
        for window in windows:
            a, b = max(span.source.start, window.start), min(span.source.end, window.end)
            if b > a:
                data = dict(span.model_dump(mode="json"), source={"start": a, "end": b})
                data["region_summary"] = {
                    "important_region_count": len(span.important_regions),
                    "text_candidate_count": len(span.text_regions),
                    "semantics": "span-aggregate geometric candidates; not verified OCR; unions are conservative",
                    "compaction": "bounding_union_when_more_than_64",
                }
                data["important_regions"] = _compact_regions(data["important_regions"])
                data["text_regions"] = _compact_regions(data["text_regions"])
                spans.append(data)
    if len(spans) > 256:
        raise ValueError("Too many local source spans for a bounded visual prompt")
    face_ids = {span["streamer_track_id"] for span in spans}
    webcam_ids = {span["webcam_track_id"] for span in spans}
    samples = [s for s in timeline.samples if any(w.start <= s.at < w.end for w in windows)]
    for sample in samples:
        face_ids.update(sample.face_track_ids)
        webcam_ids.add(sample.webcam_track_id)
    faces: list[dict[str, Any]] = [{"track_id": t.track_id, "mean_confidence": t.mean_confidence,
              "local_seen_ranges": [{"start": max(t.first_seen, w.start),
                                     "end": min(t.last_seen, w.end)} for w in windows
                                    if t.first_seen < w.end and t.last_seen > w.start],
              "local_observations": [{"at": s.at, "evidence_ids": s.evidence_ids}
                                     for s in samples if t.track_id in s.face_track_ids]}
             for t in timeline.face_tracks if t.track_id in face_ids or local(t.first_seen, t.last_seen)]
    by_id = {t.track_id: t for t in timeline.face_tracks}
    for face in faces:
        positioned = associated_face_positions(timeline, by_id[face["track_id"]])
        face["position_association"] = "ordered_evidence_bijection" if positioned is not None else "unavailable"
        face["local_positions"] = [{"at": at, "rect": rect.model_dump(mode="json"), "evidence_id": evidence}
                                   for at, rect, evidence in positioned or []
                                   if any(w.start <= at < w.end for w in windows)]
    webcams = [{"track_id": t.track_id, "face_track_ids": t.face_track_ids,
                "confidence": t.confidence,
                "observations": [o.model_dump(mode="json") for o in t.observations
                                 if any(w.start <= o.at < w.end for w in windows)]}
               for t in timeline.webcam_tracks if t.track_id in webcam_ids]
    audio = []
    for sample in samples:
        audio_window = sample.audio_window
        if audio_window and any(w.start <= audio_window.start < audio_window.end <= w.end for w in windows):
            audio.append({"source": audio_window.model_dump(mode="json"),
                          "features": sample.audio_features, "evidence_ids": sample.evidence_ids})
    if max(len(faces), len(webcams), len(samples), len(audio)) > 256:
        raise ValueError("Local track/audio context exceeds bounded prompt limits")
    payload = {
        "creator_preferences": {
            "brand": {"asset_id": "kick_brand", "persistent": True, "independent_of_title": True},
            "camera": {"policy": "adaptive_to_moment", "reaction_split_preference": "top",
                       "avoid_decorative_side_bars": True, "preserve_identity_and_gestures": True},
            "title": {"prominence": "large_readable_without_occlusion", "grounded_in_source": True},
            "subtitles": {"primary_color": "#FFFF00", "outline_color": "#000000", "outline_width": 8,
                          "font_size": 72, "max_words_per_event": 4,
                          "placement_policy": "near_content_action_without_occlusion",
                          "avoid_forcing_caption_to_camera": True, "preserve_game_hud": True,
                          "keep_stable_within_scene": True},
            "emphasis": {"requires_source_reason": True, "avoid_periodic_random_effects": True,
                         "use_supported_crop_tracks": True},
            "silence": {"preserve_reaction_suspense_and_payoff": True,
                        "source_ranges_owned_by_narrative_planner": True},
            "authority": "Preferences never override protected regions, identity, platform safety or supported capabilities.",
            "publication": host_context.get("creator_overlays", {"required": False}),
            "geometry_guidance": (
                "Use actual sharp contain/letterbox foreground projection to find free bands; "
                "a full-canvas blurred backdrop does not make every canvas pixel protected source. "
                "Example only: uncropped1280x720 contained in1080x1920 occupies y656.25..1263.75px, "
                "leaving top/bottom bands. Validate actual dimensions/crops/layers, allowed zones and "
                "all protected regions; never invent free space, shift protected content or force this layout. "
                "A model-authored full-canvas forbidden region cannot excuse missing required overlays. "
                "Missing transcript is not known silence; game/OCR text cannot supply spoken captions. "
                "Use actual timed words with half-open shot ownership, rather than coarse segment text "
                "when all words lie outside the shot. If required overlays cannot fit safely, return review_required."
            ),
        },
        "clip": {"clip_id": clip.clip_id, "headline": clip.headline.model_dump(mode="json"),
                 "reasoning": clip.reasoning,
                 "segments": [{"source": s.source.model_dump(mode="json"),
                               "output_order": s.output_order, "narrative_role": s.narrative_role,
                               "reason": s.reason} for s in clip.segments],
                 "host_context": host_context},
        "local_transcript": bounded_transcript(transcript, [(w.start, w.end) for w in windows],
                                               max_chars=max_transcript_chars),
        "source_spans": spans, "face_tracks": faces, "webcam_tracks": webcams,
        "audio_windows": audio, "safe_zones": safe_zones,
        "style_cards": [c.model_dump(mode="json") for c in style_cards],
        "frame_manifest": frame_manifest,
    }
    _validate_bounded_context(payload)
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _validate_bounded_context(value: Any, *, transcript: bool = False) -> None:
    """Fail closed rather than silently discard protected geometry or evidence."""
    if isinstance(value, list):
        if len(value) > 256:
            raise ValueError("Nested visual context exceeds bounded list size")
        for item in value:
            _validate_bounded_context(item, transcript=transcript)
    elif isinstance(value, dict):
        if len(value) > 128:
            raise ValueError("Nested visual context exceeds bounded mapping size")
        for key, item in value.items():
            _validate_bounded_context(item, transcript=transcript or key == "local_transcript")
    elif isinstance(value, str) and len(value) > (8000 if transcript else 4000):
        raise ValueError("Nested visual context exceeds bounded text size")
