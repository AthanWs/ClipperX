"""Translate validated editorial plans into rendered MP4 clips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

from .planner_contracts import ClipPlan, GlobalCompositionResponse
from .schema import ClipCandidate, TranscriptSegment

if TYPE_CHECKING:
    from .render_assets import RenderAssetRegistry
    from .render_contracts import RenderProfile

LAYOUT_MAP = {
    "full_content": "FULL_CONTENT",
    "full_camera": "FULL_CAM",
    "split": "SPLIT",
    "picture_in_picture": "PICTURE_IN_PICTURE",
    "adaptive": "ADAPTIVE",
}

SUBTITLE_MAP = {
    "dynamic_clean": "default",
    "karaoke": "hormozi",
    "minimal": "minimal",
    "podcast": "podcast",
}


def candidate_from_plan(plan: ClipPlan) -> ClipCandidate:
    ordered = sorted(plan.segments, key=lambda segment: segment.output_order)
    ranges = [(segment.source.start, segment.source.end) for segment in ordered]
    layout_segments = [
        (segment.source.start, segment.source.end, LAYOUT_MAP[segment.layout])
        for segment in ordered
    ]
    return ClipCandidate(
        start=min(start for start, _ in ranges),
        end=max(end for _, end in ranges),
        text="",
        coherence=1.0,
        headline=plan.headline.overlay_text,
        keywords=plan.headline.keywords,
        layout_segments=layout_segments,
        time_ranges=ranges,
    )


def render_legacy_composition_plan(
    source_video: Path,
    plan_path: Path,
    output_dir: Path,
    *,
    transcripts: Sequence[TranscriptSegment] | None = None,
    preset: str = "default",
    watermark_text: str | None = None,
) -> list[Path]:
    from .editing import LayoutConfig, render_batch
    from .config import settings
    from .face import detect_webcam_region, load_face_model
    from .media import probe_media

    plan = GlobalCompositionResponse.model_validate_json(plan_path.read_text(encoding="utf-8"))
    media = probe_media(source_video)
    source_width = int(media.get("video", {}).get("width") or 0)
    source_height = int(media.get("video", {}).get("height") or 0)
    requires_camera = any(
        segment.layout in {"full_camera", "split", "picture_in_picture", "adaptive"}
        for clip in plan.clips
        for segment in clip.segments
    )
    detector = load_face_model() if requires_camera else None
    outputs: list[Path] = []
    try:
        for clip_plan in plan.clips:
            candidate = candidate_from_plan(clip_plan)
            segment_crops: dict[int, tuple[int, int, int, int] | None] = {}
            ordered_segments = sorted(clip_plan.segments, key=lambda item: item.output_order)
            for index, segment in enumerate(ordered_segments):
                if segment.camera_crop is not None and source_width > 0 and source_height > 0:
                    crop = segment.camera_crop
                    segment_crops[index] = (
                        round(crop.x * source_width),
                        round(crop.y * source_height),
                        round(crop.width * source_width),
                        round(crop.height * source_height),
                    )
                    continue
                if detector is None or segment.layout not in {
                    "full_camera", "split", "picture_in_picture", "adaptive"
                }:
                    continue
                webcam = detect_webcam_region(
                    source_video,
                    start=segment.source.start,
                    end=segment.source.end,
                    max_samples=16,
                    classifier=detector,
                )
                if webcam:
                    segment_crops[index] = webcam.as_tuple()[:4]
                else:
                    segment_crops[index] = None
            if segment_crops:
                candidate.segment_face_crops = segment_crops
            default_crop = next(iter(segment_crops.values()), None)
            layout = LayoutConfig(
                face_crop=default_crop,
                camera_fit="contain",
                brand_image_path=settings.brand_image_path,
                watermark_text=watermark_text,
                caption_template=SUBTITLE_MAP.get(clip_plan.subtitle_style, "default"),
            )
            rendered = render_batch(
                source_video,
                [candidate],
                output_dir,
                layout=layout,
                transcripts=transcripts,
                preset=preset,
            )
            if not rendered:
                continue
            output_path = rendered[0]
            metadata = {
            "version": 1,
            "clip_id": clip_plan.clip_id,
            "topic_id": clip_plan.topic_id,
            "output": str(output_path),
            "overlay_text": clip_plan.headline.overlay_text,
            "platform_title": clip_plan.headline.platform_title,
            "caption": clip_plan.headline.caption,
            "keywords": clip_plan.headline.keywords,
            "duration_seconds": clip_plan.duration,
            "segments": [segment.model_dump(mode="json") for segment in clip_plan.segments],
                "reasoning": clip_plan.reasoning,
                "camera_crops": {str(key): value for key, value in segment_crops.items()},
            }
            sidecar = output_path.with_suffix(".metadata.json")
            sidecar.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            outputs.append(output_path)
    finally:
        if detector is not None:
            detector.close()
    return outputs


def render_composition_plan(
    source_video: Path, plan_path: Path, output_dir: Path, *,
    transcripts: Sequence[TranscriptSegment] | None = None, preset: str = "default",
    watermark_text: str | None = None,
) -> list[Path]:
    """Strict v1 compatibility alias; v2 callers use the explicit visual route."""
    return render_legacy_composition_plan(source_video, plan_path, output_dir,
        transcripts=transcripts, preset=preset, watermark_text=watermark_text)


def render_visual_composition_plan(
    source_video: Path, plan_path: Path, output_dir: Path, *,
    transcript_path: Path | None = None, assets: RenderAssetRegistry,
    profile: RenderProfile | None = None,
) -> list[Path]:
    """Execute exactly one validated v2 plan using the real default HOST."""
    from ._visual_render_host import _render_visual_plan_host
    return [_render_visual_plan_host(source_video, plan_path, output_dir,
        transcript_path=transcript_path, assets=assets, profile=profile)]


def load_transcript(path: Path | None) -> list[TranscriptSegment] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSegment.from_dict(item) for item in payload]
