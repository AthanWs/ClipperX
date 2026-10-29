"""Video editing utilities leveraging ffmpeg-python."""

from __future__ import annotations

import tempfile
import textwrap
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

from ._media_types import FFmpegStream, load_ffmpeg
ffmpeg = load_ffmpeg()

from PIL import Image, ImageFont  # noqa: E402 - ffmpeg remains eager before PIL

from .helpers import segment_overlaps_ranges  # noqa: E402
from .schema import ClipCandidate, TranscriptSegment  # noqa: E402
from .subtitles import clip_to_ass  # noqa: E402
from .summaries import build_filename  # noqa: E402


DEFAULT_CODEC = "h264_nvenc"
DEFAULT_CRF = 23  # Used as CQ for NVENC
EXPORT_FPS = 60


# ---------------------------------------------------------------------------
# Export presets
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExportPreset:
    name: str
    width: int
    height: int
    video_bitrate: str    # e.g. "10M"
    audio_bitrate: str    # e.g. "192k"
    codec: str            # "h264_nvenc" | "libx264"
    max_duration: float   # seconds

EXPORT_PRESETS: dict[str, ExportPreset] = {
    "tiktok": ExportPreset("tiktok", 1080, 1920, "10M", "192k", "h264_nvenc", 179),
    "reels": ExportPreset("reels", 1080, 1920, "12M", "192k", "h264_nvenc", 179),
    "shorts": ExportPreset("shorts", 1080, 1920, "10M", "192k", "h264_nvenc", 179),
    "default": ExportPreset("default", 1080, 1920, "8M", "128k", "h264_nvenc", 179),
}


@dataclass(slots=True)
class LayoutConfig:
    width: int = 1080
    height: int = 1920
    face_crop: tuple[int, int, int, int] | None = None  # x, y, w, h
    background_color: str = "#000000"
    font_file: str | None = "C:/Windows/Fonts/arial.ttf"
    title_font_file: str | None = "C:/Windows/Fonts/impact.ttf"
    title_font_size: int = 88
    face_height_ratio: float = 0.35
    face_min_height_ratio: float = 0.27
    camera_width_ratio: float = 1.0
    top_safe_ratio: float = 0.10
    top_bar_height: int = 208
    bottom_bar_height: int = 0
    panel_gap: int = 12
    brand_image_path: str | None = None
    brand_max_width: int = 420
    brand_max_height: int = 56
    brand_y: int = 8
    watermark_text: str | None = None
    watermark_x: float = 0.083
    watermark_y: float = 0.057
    safe_left: int = 170
    safe_right: int = 170
    safe_top: int = 192
    safe_bottom: int = 420
    title_duration_seconds: float = 4.0
    caption_template: str = "default"
    content_crop: tuple[int, int, int, int] | None = None  # independent evidence-backed content ROI
    camera_position: str = "top"
    title_persistent: bool = True
    title_color: str = "white"
    integrated_overlays: bool = True
    camera_fit: str = "cover"


def render_clip(
    source_video: Path,
    output_path: Path,
    clip: ClipCandidate,
    *,
    layout: LayoutConfig | None = None,
    transcripts: Sequence[TranscriptSegment] | None = None,
    title_text: str | None = None,
    preset: str = "default",
) -> Path:
    """Compose 9:16 clip based on AI-suggested layout(s), supporting dynamic transitions."""

    cfg = layout or LayoutConfig()
    export = EXPORT_PRESETS.get(preset, EXPORT_PRESETS["default"])

    # Normalize segments
    segments = _resolve_render_segments(clip)
    frame_counts = _segment_frame_counts(segments)
    source_size = _probe_dimensions(source_video)
    caption_margins: list[int] = []
    brand_path = _resolve_brand_path(cfg.brand_image_path)
    reserve_header = bool(title_text or brand_path is not None or cfg.watermark_text)
    if not reserve_header:
        cfg = replace(cfg, top_bar_height=0, brand_image_path=None, watermark_text=None)

    # Temporary directory for sub-clips
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        segment_paths = []
        segment_configs: list[LayoutConfig] = []

        # 1. Render each segment individually
        for idx, (start, end, layout_mode) in enumerate(segments):
            seg_out = temp_path / f"seg_{idx}.mp4"
            # Use per-segment face_crop if available
            seg_cfg = cfg
            seg_face_crops = clip.segment_face_crops
            if seg_face_crops and idx in seg_face_crops:
                seg_cfg = replace(cfg, face_crop=seg_face_crops[idx])
            if clip.segment_content_crops and idx in clip.segment_content_crops:
                seg_cfg = replace(seg_cfg, content_crop=clip.segment_content_crops[idx])
            _render_segment(
                source_video,
                seg_out,
                start,
                end,
                layout_mode,
                seg_cfg,
                frame_count=frame_counts[idx],
                reserve_header=reserve_header,
            )
            caption_margins.append(_caption_margin_for_segment(
                seg_cfg, layout_mode, seg_cfg.face_crop, source_size,
                reserve_header=reserve_header))
            segment_paths.append(seg_out)
            segment_configs.append(seg_cfg)

        # 2. Concatenate segments
        if len(segment_paths) == 1:
            intermediate = segment_paths[0]
        else:
            inputs = [ffmpeg.input(str(p)) for p in segment_paths]
            joined_video = ffmpeg.concat(*(item.video for item in inputs), v=1, a=0)
            joined_video = joined_video.filter("settb", "1/60").setpts("N")

            # Simple concat render to apply join
            join_out = temp_path / "joined.mp4"
            out = ffmpeg.output(
                joined_video,
                str(join_out),
                vcodec="libx264",
                an=None,
                crf=18,
                preset="fast",
                r=EXPORT_FPS,
                fps_mode="cfr",
            )
            ffmpeg.run(out, quiet=True, overwrite_output=True)
            intermediate = join_out

        # 2.5 Insert B-roll if available
        if clip.broll_inserts:
            intermediate = _composite_broll(intermediate, clip, temp_path)

        # 3. Apply Overlays (Subtitles, Title, Watermark) on the full joined video
        base_stream = ffmpeg.input(str(intermediate))
        video_stream = base_stream.video
        # Read the original timeline once, never accumulated subclip AAC padding.
        original_audio = [ffmpeg.input(str(source_video), ss=start, t=end-start).audio
                          .filter("asetpts", "PTS-STARTPTS")
                          for start, end in _source_audio_ranges(segments)]
        audio_stream = (original_audio[0] if len(original_audio) == 1 else
                        ffmpeg.concat(*original_audio, v=0, a=1))
        brand_streams = None
        if brand_path is not None:
            crop_x, crop_y, crop_width, crop_height = _image_content_box(brand_path)
            brand_width, brand_height = _fit_size(
                crop_width,
                crop_height,
                cfg.brand_max_width,
                cfg.brand_max_height,
            )
            brand_stream = (
                ffmpeg.input(str(brand_path), loop=1, framerate=1)
                .video.filter("crop", crop_width, crop_height, crop_x, crop_y)
                .filter("scale", brand_width, brand_height)
            )
            brand_streams = brand_stream.split() if len(segments) > 1 else None

        fitted_title = _fit_title(title_text, cfg) if title_text else None
        frame_cursor = 0
        for idx, ((_, _, mode), seg_cfg, frames) in enumerate(zip(segments, segment_configs, frame_counts)):
            enable = f"between(n,{frame_cursor},{frame_cursor+frames-1})"
            header_y, header_h, title_y, brand_y = _header_geometry(seg_cfg, mode, seg_cfg.face_crop, source_size)
            if reserve_header and not cfg.integrated_overlays:
                video_stream = video_stream.filter("drawbox", x=0, y=header_y, w=cfg.width,
                    h=header_h, color=cfg.background_color, t="fill", enable=enable)
            if fitted_title:
                wrapped_title, fitted_title_size = fitted_title
                lines = wrapped_title.splitlines()
                row_height = cfg.top_bar_height / len(lines)
                for line_index, line in enumerate(lines):
                    title_line_y = (f"{title_y}+({cfg.top_bar_height}-text_h)/2" if len(lines) == 1 else
                        f"{title_y}+{line_index}*{row_height:.6f}+({row_height:.6f}-text_h)/2")
                    drawtext_args = dict(text=line, fontsize=fitted_title_size,
                        fontcolor=cfg.title_color, x="(w-text_w)/2",
                        y=title_line_y, line_spacing=0,
                        shadowcolor="black", shadowx=2, shadowy=2,
                        bordercolor="black", borderw=4 if cfg.integrated_overlays else 1,
                        enable=enable if cfg.title_persistent else
                            f"{enable}*lt(t,{max(0.1,cfg.title_duration_seconds):.3f})")
                    if cfg.title_font_file:
                        drawtext_args["fontfile"] = cfg.title_font_file
                    video_stream = video_stream.drawtext(**drawtext_args)
            if brand_path is not None:
                item = brand_streams[idx] if brand_streams is not None else brand_stream
                brand_x = str(cfg.safe_left) if _integrated_top_camera(seg_cfg, mode, seg_cfg.face_crop, source_size) else "(main_w-overlay_w)/2"
                video_stream = ffmpeg.overlay(video_stream, item, x=brand_x,
                    y=str(brand_y), eof_action="repeat", shortest=1, enable=enable)
            elif cfg.watermark_text:
                wm_args = dict(text=cfg.watermark_text, fontsize=36, fontcolor=cfg.title_color,
                    x="(w-text_w)/2", y=str(brand_y), enable=enable,
                    shadowcolor=cfg.background_color, shadowx=1, shadowy=1, borderw=1)
                if cfg.font_file:
                    wm_args["fontfile"] = cfg.font_file
                video_stream = video_stream.drawtext(**wm_args)
            frame_cursor += frames

        if transcripts is not None and transcripts:
            subtitle_path = output_path.with_suffix(".ass")
            caption_clip = replace(clip, time_ranges=[(a, b) for a, b, _ in segments])
            clip_to_ass(caption_clip, transcripts, subtitle_path,
                        template=cfg.caption_template, margins_v=caption_margins)
            video_stream = video_stream.filter("ass", filename=str(subtitle_path))

        # Apply scaling as part of the filter chain, NOT as an output argument
        video_stream = video_stream.filter("scale", export.width, export.height)
        video_stream = video_stream.filter("trim", end_frame=sum(frame_counts))
        video_stream = video_stream.filter("setsar", 1).filter("settb", "1/60").setpts("N")

        # Final Output settings — use export preset
        if export.codec.endswith("_nvenc"):
            output_kwargs = {
                "vcodec": export.codec,
                "acodec": "aac",
                "preset": "p6",
                "tune": "hq",
                "rc": "vbr",
                "cq": DEFAULT_CRF,
                "b:v": export.video_bitrate,
                "b:a": export.audio_bitrate,
                "pix_fmt": "yuv420p",
                "movflags": "+faststart",
            }
        else:
            output_kwargs = {
                "vcodec": export.codec,
                "acodec": "aac",
                "crf": 18,
                "b:a": export.audio_bitrate,
                "preset": "fast",
                "pix_fmt": "yuv420p",
                "movflags": "+faststart",
            }

        out = ffmpeg.output(
            video_stream,
            audio_stream,
            str(output_path),
            ar=48000,
            r=EXPORT_FPS,
            fps_mode="cfr",
            **output_kwargs,
        )
        out = out.overwrite_output()
        ffmpeg.run(out, quiet=True, overwrite_output=True)

    return output_path


def _composite_broll(
    intermediate: Path,
    clip: ClipCandidate,
    temp_path: Path,
) -> Path:
    """Replace visuals; render_clip independently assembles original source audio."""
    from .logger import get_logger
    logger = get_logger(__name__)

    current = intermediate
    # Sort inserts by timestamp descending so offsets don't shift
    sorted_inserts = sorted(
        [b for b in clip.broll_inserts if b.video_path and b.video_path.exists()],
        key=lambda b: b.timestamp,
        reverse=True,
    )

    for idx, broll in enumerate(sorted_inserts):
        try:
            out_path = temp_path / f"broll_{idx}.mp4"
            main = ffmpeg.input(str(current))
            broll_in = ffmpeg.input(str(broll.video_path), t=broll.duration)
            broll_v = (
                broll_in.video
                .filter("scale", 1080, 1920, force_original_aspect_ratio="increase")
                .filter("crop", 1080, 1920)
                .setpts(f"PTS-STARTPTS+{broll.timestamp:.6f}/TB")
            )
            video = ffmpeg.overlay(main.video, broll_v, eof_action="pass")

            ffmpeg.output(
                video, str(out_path), an=None,
                vcodec="libx264", crf=18, preset="fast", r=EXPORT_FPS, fps_mode="cfr",
            ).run(quiet=True, overwrite_output=True)
            current = out_path
        except Exception as exc:
            logger.warning("[editing] broll insert failed", extra={"error": str(exc)})
            continue

    return current


def _fit_title(value: str, cfg: LayoutConfig) -> tuple[str, int]:
    """Reuse the modern measured title fitter; retain text instead of truncating it."""
    if not cfg.title_font_file:
        return _wrap_overlay_text(value), cfg.title_font_size
    from .filtergraph import PixelRect
    from .overlay_ass import _title_event
    width = cfg.width-cfg.safe_left-cfg.safe_right
    event = _title_event(text=value, path=Path(cfg.title_font_file),
        destination=PixelRect(0, 0, width, cfg.top_bar_height), shot_id="compat-title",
        start=0, end=max(.1, cfg.title_duration_seconds), layer=1,
        max_font_size=cfg.title_font_size, outline_width=4 if cfg.integrated_overlays else 2)
    size = min(event.font_size, cfg.title_font_size)
    if size < min(28, cfg.title_font_size):
        raise ValueError("Title is too long for readable mobile text; shorten the headline")
    lines = event.lines
    if len(lines) == 2:
        font = ImageFont.truetype(cfg.title_font_file, size)
        stroke = 4 if cfg.integrated_overlays else 2

        def measured(line: str) -> float:
            left, _, right, _ = font.getbbox(line, stroke_width=stroke)
            return max(right-left, font.getlength(line)+2*stroke)

        words = event.text.split()
        alternatives = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
        fitting = [pair for pair in alternatives if max(map(measured, pair)) <= width]
        if fitting:
            lines = min(fitting, key=lambda pair: abs(measured(pair[0])-measured(pair[1])))
    return "\n".join(lines), size


def _wrap_overlay_text(value: str, *, width: int = 20, max_lines: int = 3) -> str:
    lines = textwrap.wrap(
        " ".join(value.split()),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = lines[: max_lines - 1]
    tail = " ".join(lines[max_lines - 1 :])
    return "\n".join([*head, tail[:width].rstrip()])


def _resolve_brand_path(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parent.parent / candidate
    return candidate if candidate.is_file() else None


def _image_content_box(path: Path) -> tuple[int, int, int, int]:
    with Image.open(path) as image:
        width, height = image.size
        if "A" not in image.getbands():
            return 0, 0, width, height
        box = image.getchannel("A").getbbox()
        if box is None:
            return 0, 0, width, height
        left, top, right, bottom = box
        return left, top, right - left, bottom - top


def _fit_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    scale = min(max_width / max(1, width), max_height / max(1, height))
    fitted_width = max(2, round(width * scale))
    fitted_height = max(2, round(height * scale))
    return fitted_width - fitted_width % 2, fitted_height - fitted_height % 2


def _top_safe_margin(cfg: LayoutConfig) -> int:
    ratio = max(0.0, min(cfg.top_safe_ratio, 0.3))
    return round(cfg.height * ratio)


def _content_without_camera(
    stream: FFmpegStream,
    camera_crop: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
) -> FFmpegStream:
    content_crop = _content_crop_without_camera(camera_crop, source_width, source_height)
    if content_crop is None:
        return stream
    x, y, width, height = content_crop
    return stream.filter("crop", width, height, x, y)


def _content_crop_without_camera(
    camera_crop: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int] | None:
    x, y, width, height = _clamp_crop(camera_crop, source_width, source_height)
    if width / source_width > 0.45 or height / source_height > 0.65:
        return None
    horizontal_margin = max(8, int(source_width * 0.01))
    vertical_margin = max(8, int(source_height * 0.01))
    if x / source_width < 0.25 and (x + width) / source_width < 0.55:
        crop_x = min(source_width - 64, x + width + horizontal_margin)
        return crop_x, 0, source_width - crop_x, source_height
    if x / source_width > 0.45:
        crop_width = max(64, x - horizontal_margin)
        return 0, 0, crop_width, source_height
    if y / source_height < 0.2 and (y + height) / source_height < 0.55:
        crop_y = min(source_height - 64, y + height + vertical_margin)
        return 0, crop_y, source_width, source_height - crop_y
    if y / source_height > 0.45:
        crop_height = max(64, y - vertical_margin)
        return 0, 0, source_width, crop_height
    return None


def _probe_dimensions(source: Path) -> tuple[int, int]:
    """Return (width, height) of the source video."""
    try:
        probe = ffmpeg.probe(str(source))
        video_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "video"), None
        )
        if video_stream:
            return int(video_stream["width"]), int(video_stream["height"])
    except Exception:
        pass
    return 1920, 1080  # default assumption


def _render_segment(
    source: Path,
    output: Path,
    start: float,
    end: float,
    layout_mode: str,
    cfg: LayoutConfig,
    *,
    frame_count: int | None = None,
    reserve_header: bool = False,
) -> None:
    """Helper: Render a single time segment with a specific layout."""

    duration = end - start
    if frame_count is None:
        frame_count = _segment_frame_counts([(start, end, layout_mode)])[0]
    base = ffmpeg.input(str(source), ss=start, t=duration)
    source_video = base.video.setpts("PTS-STARTPTS")

    layout_mode = layout_mode.upper()

    # Normalize layout tags
    if "TALKING" in layout_mode:
        layout_mode = "FULL_CAM"
    if "GAMEPLAY" in layout_mode:
        layout_mode = "SPLIT"

    # Detect source orientation
    src_w, src_h = _probe_dimensions(source)
    is_landscape = src_w > src_h
    composed = None

    if not is_landscape:
        # Source is vertical or square — simpler processing, no landscape crop logic
        # Just scale/pad to target dimensions
        composed = _compose_full_frame(source_video, cfg, reserve_header)

    # --- Landscape source: original layout logic ---
    # --- 1. FULL CONTENT ---
    elif "FULL_CONTENT" in layout_mode or "CINEMATIC" in layout_mode:
        composed = _compose_full_frame(source_video, cfg, reserve_header)

    # --- 2. FULL CAM ---
    elif "FULL_CAM" in layout_mode:
        if cfg.face_crop:
            x, y, width, height = _clamp_crop(cfg.face_crop, src_w, src_h)
            camera = source_video.filter("crop", width, height, x, y)
            composed = _compose_full_frame(camera, cfg, reserve_header)
        else:
            composed = _compose_full_frame(source_video, cfg, reserve_header)

    elif "PICTURE_IN_PICTURE" in layout_mode or "PIP" in layout_mode:
        if not cfg.face_crop:
            composed = _compose_full_frame(source_video, cfg, reserve_header)
        else:
            split = source_video.split()
            x, y, width, height = _clamp_crop(cfg.face_crop, src_w, src_h)
            content_source = split[0]
            if cfg.content_crop is not None:
                cx, cy, cw, ch = _clamp_crop(cfg.content_crop, src_w, src_h)
                content_source = content_source.filter("crop", cw, ch, cx, cy)
            content = _compose_full_frame(content_source, cfg, reserve_header)
            pip_x, pip_y, camera_width, camera_height = _pip_geometry(cfg, width, height)
            camera = _compose_contained(split[1].filter("crop", width, height, x, y),
                camera_width, camera_height, cfg.background_color)
            composed = ffmpeg.overlay(
                content,
                camera,
                x=pip_x,
                y=pip_y,
            )

    # --- 3. SPLIT (Default) ---
    else:
        if not cfg.face_crop:
            composed = _compose_full_frame(source_video, cfg, reserve_header)
        else:
            split = source_video.split()
            x, y, width, height = _clamp_crop(cfg.face_crop, src_w, src_h)
            # Generic desktop/reacted-video material must not silently lose its
            # left column merely because the webcam is in the left corner.
            content_crop = None if cfg.content_crop is None else _clamp_crop(cfg.content_crop, src_w, src_h)
            if content_crop is None:
                content_source = split[0]
                content_width, content_height = src_w, src_h
            else:
                content_x, content_y, content_width, content_height = content_crop
                content_source = split[0].filter(
                    "crop", content_width, content_height, content_x, content_y
                )

            top_y, gameplay_target_height, camera_y, face_target_height = (
                _split_panel_geometry(
                    cfg,
                    content_width,
                    content_height,
                    width,
                    height,
                )
            )
            gameplay_stream = _compose_contained(
                content_source,
                cfg.width,
                gameplay_target_height,
                cfg.background_color,
                top_inset=_split_title_inset(cfg, content_width, content_height,
                    gameplay_target_height) if reserve_header else 0,
            )
            face_stream = (
                split[1]
                .filter("crop", width, height, x, y)
            )
            face_stream = _compose_camera_panel(
                face_stream,
                cfg.width,
                face_target_height,
                width,
                height,
                fit=cfg.camera_fit,
            )

            background = ffmpeg.input(
                f"color=c={cfg.background_color}:s={cfg.width}x{cfg.height}:r={EXPORT_FPS}",
                f="lavfi",
                t=duration,
            )
            composed = ffmpeg.overlay(background, gameplay_stream, x=0, y=top_y)
            composed = ffmpeg.overlay(composed, face_stream, x=0, y=camera_y)

    composed = (composed.filter("fps", fps=EXPORT_FPS, start_time=0)
                .filter("tpad", stop_mode="clone", stop_duration=1 / EXPORT_FPS)
                .filter("trim", end_frame=frame_count).filter("setsar", 1)
                .filter("settb", "1/60").setpts("N"))
    out = ffmpeg.output(
        composed,
        str(output),
        vcodec=DEFAULT_CODEC,
        an=None,
        preset="p4",
        tune="hq",
        rc="vbr",
        cq=19,
        vframes=frame_count,
        r=EXPORT_FPS,
        fps_mode="cfr",
        **{"b:v": "12M"},
    )
    ffmpeg.run(out, quiet=True, overwrite_output=True)


def prepare_render_layout(layout: LayoutConfig | None, *, caption_template: str,
        brand_image_path: str | None) -> LayoutConfig:
    """Preserve every caller override while selecting the requested caption style."""
    if layout is not None:
        return replace(layout, caption_template=caption_template)
    return LayoutConfig(brand_image_path=brand_image_path, caption_template=caption_template,
                        camera_fit="contain")


def render_batch(
    source_video: Path,
    clips: Iterable[ClipCandidate],
    output_dir: Path,
    *,
    layout: LayoutConfig | None = None,
    transcripts: Sequence[TranscriptSegment] | None = None,
    preset: str = "default",
) -> list[Path]:
    paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    export = EXPORT_PRESETS.get(preset, EXPORT_PRESETS["default"])

    for clip in clips:
        if clip.duration > export.max_duration:
            raise ValueError(
                f"El clip {clip.headline or 'sin título'} dura {clip.duration:.3f}s y supera "
                f"el máximo de {export.max_duration:.0f}s"
            )

        filename = build_filename(clip.headline or "clip", clip)
        output_path = output_dir / filename
        clip_transcripts: Sequence[TranscriptSegment] | None = transcripts
        if transcripts is not None:
            clip_transcripts = _filter_segments_for_clip(clip, transcripts)

        clip_layout = layout or LayoutConfig()
        if clip.webcam_region:
            clip_layout = replace(clip_layout, face_crop=clip.webcam_region[:4])
        elif clip.face_crop:
            clip_layout = replace(clip_layout, face_crop=clip.face_crop)

        render_clip(
            source_video,
            output_path,
            clip,
            layout=clip_layout,
            transcripts=clip_transcripts,
            title_text=clip.headline,
            preset=preset,
        )
        paths.append(output_path)
    return paths


def _filter_segments_for_clip(
    clip: ClipCandidate,
    segments: Sequence[TranscriptSegment],
) -> list[TranscriptSegment]:
    ranges = clip.time_ranges or [(clip.start, clip.end)]
    filtered: list[TranscriptSegment] = []
    for segment in segments:
        if segment_overlaps_ranges(segment, ranges):
            filtered.append(segment)
    return filtered


def _resolve_render_segments(clip: ClipCandidate) -> list[tuple[float, float, str]]:
    ranges = clip.time_ranges or [(clip.start, clip.end)]
    layout_mode = (clip.layout or "SPLIT").upper()
    if not clip.layout_segments:
        return [(start, end, layout_mode) for start, end in ranges if end > start]
    resolved: list[tuple[float, float, str]] = []
    for layout_start, layout_end, mode in clip.layout_segments:
        for range_start, range_end in ranges:
            start = max(layout_start, range_start)
            end = min(layout_end, range_end)
            if end > start:
                resolved.append((start, end, mode))
    if not resolved:
        raise ValueError("Los layout_segments no intersectan los rangos del clip")
    return resolved


def _segment_frame_counts(segments: Sequence[tuple[float, float, str]]) -> list[int]:
    """Partition one cumulative 60Hz output clock; no rounding once per cut."""
    if not segments:
        raise ValueError("At least one video frame range is required")
    duration = Fraction(0)
    previous = 0
    counts: list[int] = []
    for start, end, _ in segments:
        a, b = Fraction(str(start)), Fraction(str(end))
        if a < 0 or b <= a:
            raise ValueError("Positive finite video frame ranges required")
        duration += b - a
        boundary = int(duration * EXPORT_FPS + Fraction(1, 2))
        if boundary <= previous:
            raise ValueError("A segment has no output frame at60FPS")
        counts.append(boundary - previous)
        previous = boundary
    return counts


def _source_audio_ranges(segments: Sequence[tuple[float, float, str]]) -> list[tuple[float, float]]:
    """Coalesce only touching edits, without sorting or losing repeated ranges."""
    ranges: list[tuple[float, float]] = []
    for start, end, _ in segments:
        if ranges and Fraction(str(ranges[-1][1])) == Fraction(str(start)):
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    return ranges


def _caption_margin_for_segment(cfg: LayoutConfig, mode: str,
        crop: tuple[int, int, int, int] | None, source_size: tuple[int, int], *,
        reserve_header: bool = True) -> int:
    if (crop is not None and source_size[0] > source_size[1]
            and ("PICTURE_IN_PICTURE" in mode.upper() or "PIP" in mode.upper())):
        _, _, width, height = _clamp_crop(crop, *source_size)
        _, top, _, _ = _pip_geometry(cfg, width, height)
        return cfg.height - top + 24
    if (crop is not None and source_size[0] > source_size[1]
            and any(tag in mode.upper() for tag in ("SPLIT", "ADAPTIVE", "GAMEPLAY"))):
        x, y, width, height = _clamp_crop(crop, *source_size)
        content = cfg.content_crop
        content_size = source_size if content is None else _clamp_crop(content, *source_size)[2:]
        content_y, panel_h, camera_y, camera_h = _split_panel_geometry(cfg, *content_size, width, height)
        if cfg.integrated_overlays:
            content_w, content_h = content_size
            inset = _split_title_inset(cfg, content_w, content_h, panel_h) if reserve_header else 0
            scale = min(cfg.width/content_w, (panel_h-inset)/content_h)
            sharp_h = content_h*scale
            sharp_top = content_y+inset+(panel_h-inset-sharp_h)/2
            # Keep reading in the upper content lane, above the center/typical
            # avatar area. This is a deterministic fallback, not HUD detection;
            # the visual director supplies evidence-backed protected regions.
            bottom = min(cfg.height-cfg.safe_bottom, round(sharp_top+.19*sharp_h))
            return cfg.height-max(cfg.safe_top, bottom)
        bottom = camera_y+camera_h if cfg.camera_position == "top" else camera_y
        return cfg.height - bottom + 24
    return max(cfg.safe_bottom, round(cfg.height * .24))


def _pip_geometry(cfg: LayoutConfig, width: int, height: int) -> tuple[int, int, int, int]:
    camera_width = max(2, int(cfg.width * .34)//2*2)
    camera_height = max(2, min(int(cfg.height * .25), round(camera_width*height/max(1, width)))//2*2)
    top = min(round(cfg.height * .64), round(cfg.height * .8)-camera_height)//2*2
    return 32, max(0, top), camera_width, camera_height


def _compose_full_frame(stream: FFmpegStream, cfg: LayoutConfig, reserve_header: bool) -> FFmpegStream:
    if cfg.integrated_overlays:
        inset = min(_top_safe_margin(cfg)+_overlay_lane_height(cfg), cfg.height-2) if reserve_header else 0
        return _compose_contained(stream, cfg.width, cfg.height, cfg.background_color, top_inset=inset)
    header = max(0, min(_top_safe_margin(cfg)+_header_height(cfg), cfg.height-2))//2*2 if reserve_header else 0
    panel = _compose_contained(stream, cfg.width, cfg.height-header, cfg.background_color)
    return (panel.filter("pad", cfg.width, cfg.height, 0, header, color=cfg.background_color)
            if header else panel)


def _compose_contained(
    stream: FFmpegStream, width: int, height: int, background_color: str,
    *, top_inset: int = 0,
) -> FFmpegStream:
    top_inset = max(0, min(int(top_inset), height-2))//2*2
    split = stream.split()
    background = (
        split[0]
        .filter("scale", width, height, force_original_aspect_ratio="increase")
        .filter("crop", width, height)
        .filter("boxblur", 12, 2)
    )
    foreground = split[1].filter(
        "scale",
        width,
        height-top_inset,
        force_original_aspect_ratio="decrease",
    )
    return ffmpeg.overlay(
        background,
        foreground,
        x="(main_w-overlay_w)/2",
        y=f"{top_inset}+(main_h-{top_inset}-overlay_h)/2" if top_inset else "(main_h-overlay_h)/2",
    )


def _compose_cover(stream: FFmpegStream, width: int, height: int) -> FFmpegStream:
    return (
        stream.filter("scale", width, height, force_original_aspect_ratio="increase")
        .filter("crop", width, height)
    )


def _compose_camera_panel(
    stream: FFmpegStream,
    width: int,
    height: int,
    source_width: int,
    source_height: int,
    *, fit: str = "cover",
) -> FFmpegStream:
    if fit not in {"cover", "contain"}:
        raise ValueError("camera_fit must be cover or contain")
    if fit == "contain" or source_width < source_height:
        return _compose_contained(stream, width, height, "black")
    return (stream.filter("scale", width, height, force_original_aspect_ratio="increase")
            .filter("crop", width, height, "(in_w-out_w)/2", "0"))


def _camera_inner_width(
    canvas_width: int,
    panel_height: int,
    camera_width: int,
    camera_height: int,
) -> int:
    aspect = camera_width / max(1, camera_height)
    if aspect < 0.9:
        minimum_ratio = 0.62
    elif aspect < 1.1:
        minimum_ratio = 0.86
    else:
        minimum_ratio = 0.94
    natural_width = round(panel_height * aspect)
    minimum_width = round(canvas_width * minimum_ratio)
    maximum_width = canvas_width
    inner_width = max(minimum_width, min(natural_width, maximum_width))
    return max(2, inner_width - inner_width % 2)


def _split_panel_geometry(
    cfg: LayoutConfig,
    content_width: int,
    content_height: int,
    camera_width: int | None = None,
    camera_height: int | None = None,
) -> tuple[int, int, int, int]:
    if cfg.camera_position not in {"top", "bottom"}:
        raise ValueError("camera_position must be top or bottom")
    top_y = 0 if cfg.integrated_overlays else max(0, min(_top_safe_margin(cfg) + _header_height(cfg), cfg.height-256))
    panel_gap = 0 if cfg.integrated_overlays else max(0, min(cfg.panel_gap, cfg.height // 4))
    bottom_bar = max(0, min(cfg.bottom_bar_height, cfg.height // 3))
    max_ratio = max(0.1, min(cfg.face_height_ratio, 0.6))
    min_ratio = max(0.1, min(cfg.face_min_height_ratio, max_ratio))
    maximum_face_height = max(64, int(cfg.height * max_ratio))
    minimum_face_height = max(64, int(cfg.height * min_ratio))
    if camera_width and camera_height:
        camera_aspect = camera_width / max(1, camera_height)
        desired_face_height = round(
            cfg.width * cfg.camera_width_ratio / max(camera_aspect, 0.1)
        )
        face_height = max(
            minimum_face_height,
            min(desired_face_height, maximum_face_height),
        )
    else:
        face_height = maximum_face_height
    face_height = max(2, face_height//2*2)
    if cfg.camera_position == "top":
        content_y = face_height + _header_height(cfg) + panel_gap
        return content_y, max(64, cfg.height-content_y-bottom_bar), 0, face_height
    available_gameplay = cfg.height - top_y - panel_gap - bottom_bar - face_height
    gameplay_height = max(64, available_gameplay)
    camera_y = top_y + gameplay_height + panel_gap
    return top_y, gameplay_height, camera_y, face_height


def _header_height(cfg: LayoutConfig) -> int:
    return 0 if cfg.integrated_overlays else _overlay_lane_height(cfg)


def _overlay_lane_height(cfg: LayoutConfig) -> int:
    brand_lane = cfg.brand_max_height + 2*cfg.brand_y if cfg.brand_image_path or cfg.watermark_text else 0
    return cfg.top_bar_height + brand_lane


def _split_title_inset(cfg: LayoutConfig, width: int, height: int, panel_height: int) -> int:
    if not cfg.integrated_overlays:
        return 0
    if cfg.camera_position == "bottom":
        return _top_safe_margin(cfg)+_overlay_lane_height(cfg)
    # Reuse natural contain margins; only move the sharp viewport for tall ROIs.
    sharp_height = min(panel_height, cfg.width*height/max(1, width))
    return 0 if (panel_height-sharp_height)/2 >= cfg.top_bar_height-2 else cfg.top_bar_height


def _integrated_top_camera(cfg: LayoutConfig, mode: str,
        crop: tuple[int, int, int, int] | None, source_size: tuple[int, int]) -> bool:
    return (cfg.integrated_overlays and cfg.camera_position == "top" and crop is not None
        and source_size[0] > source_size[1]
        and any(tag in mode.upper() for tag in ("SPLIT", "ADAPTIVE", "GAMEPLAY")))


def _header_geometry(cfg: LayoutConfig, mode: str, crop: tuple[int, int, int, int] | None,
        source_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if _integrated_top_camera(cfg, mode, crop, source_size):
        assert crop is not None
        _, _, camera_y, camera_h = _split_panel_geometry(cfg, *source_size, *_clamp_crop(crop, *source_size)[2:])
        return camera_y+camera_h, 0, camera_y+camera_h, _top_safe_margin(cfg)
    y = _top_safe_margin(cfg)
    if (crop is not None and source_size[0] > source_size[1] and cfg.camera_position == "top"
            and any(tag in mode.upper() for tag in ("SPLIT", "ADAPTIVE", "GAMEPLAY"))):
        _, _, camera_y, camera_h = _split_panel_geometry(cfg, *source_size, *_clamp_crop(crop, *source_size)[2:])
        y = camera_y+camera_h
    brand_lane = _overlay_lane_height(cfg)-cfg.top_bar_height
    return y, _header_height(cfg), y+brand_lane, y+cfg.brand_y


def _clamp_crop(
    crop: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = crop
    x = max(0, min(int(x), source_width - 2))
    y = max(0, min(int(y), source_height - 2))
    width = max(2, min(int(width), source_width - x))
    height = max(2, min(int(height), source_height - y))
    return x, y, width, height
