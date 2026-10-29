from pathlib import Path
from typing import Any

import pytest

import clipperx.editing as editing
from clipperx.schema import ClipCandidate, TranscriptSegment
from clipperx.subtitles import clip_to_ass


def test_top_camera_uses_canvas_top_and_separate_title_brand_lane() -> None:
    cfg = editing.LayoutConfig(brand_image_path="assets/branding/KickAthanW.png", integrated_overlays=False)
    assert getattr(cfg, "camera_position", None) == "top"
    assert editing._split_panel_geometry(cfg, 934, 720, 334, 271) == (964, 956, 0, 672)
    geometry = getattr(editing, "_header_geometry", None)
    assert callable(geometry)
    assert geometry(cfg, "SPLIT", (0, 0, 334, 271), (1280, 720)) == (672, 280, 744, 680)
    assert editing._caption_margin_for_segment(cfg, "SPLIT", (0, 0, 334, 271), (1280, 720)) == 1272


def test_bottom_camera_remains_explicitly_selectable_without_overlap() -> None:
    cfg = editing.LayoutConfig(brand_image_path="assets/branding/KickAthanW.png", integrated_overlays=False)
    assert hasattr(cfg, "camera_position")
    cfg.camera_position = "bottom"
    assert editing._split_panel_geometry(cfg, 934, 720, 334, 271) == (472, 764, 1248, 672)
    assert editing._header_geometry(cfg, "SPLIT", (0, 0, 334, 271), (1280, 720)) == (192, 280, 264, 200)
    assert editing._caption_margin_for_segment(cfg, "SPLIT", (0, 0, 334, 271), (1280, 720)) == 696


def test_title_measurement_can_be_larger_without_changing_existing_fitter_default() -> None:
    cfg = editing.LayoutConfig()
    wrapped, size = editing._fit_title("¿DE DÓNDE SALIÓ TANTA GENTE?", cfg)
    assert wrapped.replace("\n", " ") == "¿DE DÓNDE SALIÓ TANTA GENTE?"
    assert 64 <= size <= 88
    from clipperx.filtergraph import PixelRect
    from clipperx.overlay_ass import _title_event
    event = _title_event(text="Short title", path=Path(cfg.title_font_file),
        destination=PixelRect(0, 0, 740, 208), shot_id="old-v2", start=0, end=1, layer=1)
    assert event.font_size <= 46


def test_real_graph_brand_is_present_with_title_from_first_frame_to_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(editing.ffmpeg, "run", lambda graph, **kw: commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing, "_probe_dimensions", lambda source: (1280, 720))
    cfg = editing.LayoutConfig(face_crop=(0, 0, 334, 271), brand_image_path="assets/branding/KickAthanW.png")
    clip = ClipCandidate(1, 8, "", 1, layout_segments=[(1, 3, "SPLIT"), (3, 8, "FULL_CAM")])
    editing.render_clip(Path("source.mp4"), tmp_path/"out.mp4", clip, layout=cfg, title_text="Hook")
    graph = " ".join(commands[-1])
    assert "gte(t\\,4.000)" not in graph
    assert "lt(t\\,4.000)" not in graph
    assert "between(n\\,0\\,119)" in graph
    assert "between(n\\,120\\,419)" in graph
    assert "drawbox=" not in graph
    first = " ".join(commands[0])
    assert "pad=1080:672" not in first, "Landscape camera must not get side bars"


def test_bright_yellow_caption_has_opaque_heavy_black_outline(tmp_path: Path) -> None:
    out = tmp_path/"caption.ass"
    clip_to_ass(ClipCandidate(0, 1, "", 1), [TranscriptSegment(0, 1, "Test")], out)
    style = next(line for line in out.read_text(encoding="utf-8").splitlines() if line.startswith("Style:"))
    fields = style.split(",")
    assert fields[3] == "&H0000FFFF"
    assert fields[5] == "&H00000000"
    assert int(fields[16]) >= 7


def test_pipeline_layout_preparation_keeps_every_creator_override() -> None:
    prepare = getattr(editing, "prepare_render_layout", None)
    assert callable(prepare)
    cfg = editing.LayoutConfig(content_crop=(12, 20, 340, 500), camera_position="bottom",
        background_color="#F4F1E8", title_color="black", title_persistent=False,
        title_duration_seconds=2.5, title_font_size=76)
    result = prepare(cfg, caption_template="neon", brand_image_path="unrelated.png")
    from dataclasses import asdict
    assert asdict(result) == dict(asdict(cfg), caption_template="neon")
    assert cfg.caption_template == "default"
    fresh = prepare(None, caption_template="default", brand_image_path="assets/branding/KickAthanW.png")
    assert fresh.brand_image_path == "assets/branding/KickAthanW.png"


def test_camera_fill_does_not_center_crop_away_hair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    commands: list[list[str]]=[]
    monkeypatch.setattr(editing.ffmpeg,"run",lambda graph,**kw:commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing,"_probe_dimensions",lambda source:(1280,720))
    editing.render_clip(Path("source.mp4"),tmp_path/"out.mp4",ClipCandidate(0,1,"",1),
        layout=editing.LayoutConfig(face_crop=(0,0,334,271)),title_text="Hook")
    assert "crop=1080:672:(in_w-out_w)/2:0" in " ".join(commands[0])


@pytest.mark.parametrize("width,height", [(500,540),(486,540),(484,540)])
def test_all_portrait_camera_ratios_keep_the_complete_person(width: int,height: int) -> None:
    stream=editing._compose_camera_panel(editing.ffmpeg.input("source.mp4").video,1080,672,width,height)
    graph=" ".join(editing.ffmpeg.compile(editing.ffmpeg.output(stream,"out.mp4",an=None)))
    assert "scale=1080:672:force_original_aspect_ratio=decrease" in graph
    assert "boxblur=" in graph
