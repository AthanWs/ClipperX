from pathlib import Path

import pytest

from clipperx import editing
from clipperx.schema import ClipCandidate


@pytest.mark.parametrize("position,want", [
    ("top", (672, 1248, 0, 672)),
    ("bottom", (0, 1248, 1248, 672)),
])
def test_default_panels_touch_without_header_or_gap(position, want):
    cfg = editing.LayoutConfig(camera_position=position,
        brand_image_path="assets/branding/KickAthanW.png")
    assert editing._split_panel_geometry(cfg, 934, 720, 334, 271) == want


def test_integrated_title_and_brand_do_not_add_opaque_strip(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(editing.ffmpeg, "run", lambda graph, **kw: commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing, "_probe_dimensions", lambda source: (1280, 720))
    cfg = editing.LayoutConfig(face_crop=(0, 0, 334, 271), content_crop=(346, 0, 934, 720),
        brand_image_path="assets/branding/KickAthanW.png")
    editing.render_clip(Path("source.mp4"), tmp_path/"out.mp4", ClipCandidate(0, 1, "", 1),
        layout=cfg, title_text="Hook")
    final = " ".join(commands[-1])
    assert "drawbox=" not in final
    assert "borderw=4" in final and "bordercolor=black" in final
    assert "x=170:y=192" in final, "Brand overlays camera wall, not a new center strip"
    assert "y=672+(208-text_h)/2" in final
    assert "scale=1080:1248:force_original_aspect_ratio=decrease" in " ".join(commands[0])


def test_portrait_sharp_viewport_is_below_overlays_without_black_pad():
    cfg = editing.LayoutConfig(brand_image_path="assets/branding/KickAthanW.png")
    result = editing._compose_full_frame(editing.ffmpeg.input("source.mp4").video, cfg, True)
    graph = " ".join(editing.ffmpeg.compile(editing.ffmpeg.output(result, "out.mp4")))
    assert "pad=" not in graph
    assert "scale=1080:1920:force_original_aspect_ratio=increase" in graph
    assert "scale=1080:1448:force_original_aspect_ratio=decrease" in graph
    assert "y=472+(main_h-472-overlay_h)/2" in graph


def test_tall_split_content_preserves_complete_sharp_roi_below_title(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(editing.ffmpeg, "run", lambda graph, **kw: commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing, "_probe_dimensions", lambda source: (1280, 720))
    editing.render_clip(Path("source.mp4"), tmp_path/"out.mp4", ClipCandidate(0, 1, "", 1),
        layout=editing.LayoutConfig(face_crop=(0, 0, 340, 270), content_crop=(560, 120, 260, 550)),
        title_text="Hook")
    graph = " ".join(commands[0])
    assert "crop=260:550:560:120" in graph
    assert "scale=1080:1040:force_original_aspect_ratio=decrease" in graph
    assert "y=208+(main_h-208-overlay_h)/2" in graph


def test_legacy_banded_layout_is_explicitly_available():
    cfg = editing.LayoutConfig(integrated_overlays=False, brand_image_path="assets/branding/KickAthanW.png")
    assert editing._split_panel_geometry(cfg, 934, 720, 334, 271) == (964, 956, 0, 672)


@pytest.mark.parametrize("text", ["¿DE DÓNDE SALIÓ TANTA GENTE?", "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"])
def test_title_measurement_includes_rendered_four_pixel_outline(text):
    from PIL import ImageFont
    cfg = editing.LayoutConfig()
    wrapped, size = editing._fit_title(text, cfg)
    font = ImageFont.truetype(cfg.title_font_file, size)
    assert max(font.getlength(line)+8 for line in wrapped.splitlines()) <= 740
    assert len(wrapped.splitlines())*(sum(font.getmetrics())+8) <= 208


def test_two_line_headline_is_balanced_instead_of_leaving_last_word_alone():
    from PIL import ImageFont
    cfg = editing.LayoutConfig()
    wrapped, size = editing._fit_title("¿DE DÓNDE SALIÓ TANTA GENTE?", cfg)
    font = ImageFont.truetype(cfg.title_font_file, size)
    widths = [font.getlength(line) for line in wrapped.splitlines()]
    assert len(widths) == 2 and min(widths)/max(widths) >= .7


def test_each_title_line_is_centered_independently(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(editing.ffmpeg, "run", lambda graph, **kw: commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing, "_probe_dimensions", lambda _: (1280, 720))
    editing.render_clip(Path("s.mp4"),tmp_path/"out.mp4",ClipCandidate(0,1,"",1),
        layout=editing.LayoutConfig(face_crop=(0,0,334,271)),title_text="¿DE DÓNDE SALIÓ TANTA GENTE?")
    final = " ".join(commands[-1])
    assert final.count("drawtext=") == 2
    assert final.count("x=(w-text_w)/2") == 2


def test_detected_face_roi_can_be_fitted_without_cover_cutting_its_chin():
    result = editing._compose_camera_panel(editing.ffmpeg.input("s.mp4").video,
        1080,672,228,175,fit="contain")
    graph = " ".join(editing.ffmpeg.compile(editing.ffmpeg.output(result,"out.mp4")))
    assert "scale=1080:672:force_original_aspect_ratio=decrease" in graph
    assert "crop=1080:672:(in_w-out_w)/2:0" not in graph
    assert "boxblur=" in graph


def test_automatic_pipeline_preserves_whole_detected_crop_but_keeps_explicit_override():
    config = editing.prepare_render_layout(None,caption_template="default",brand_image_path=None)
    assert config.camera_fit == "contain"
    override = editing.LayoutConfig()
    assert editing.prepare_render_layout(override,caption_template="default",brand_image_path=None).camera_fit == "cover"
