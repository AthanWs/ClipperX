from pathlib import Path
from typing import Any

import pytest

import clipperx.editing as editing
from clipperx.schema import ClipCandidate, ViralityScore


def test_compact_camera_and_padding() -> None:
    cfg = editing.LayoutConfig()
    top, game_h, camera_y, camera_h = editing._split_panel_geometry(cfg, 934, 720, 334, 271)
    assert camera_y == 0
    assert top + game_h == cfg.height
    assert camera_h >= 650
    assert top == camera_h
    assert editing._camera_inner_width(1080, camera_h, 334, 271) >= 970


def test_mobile_title_guard_and_measured_fit_preserve_all_words() -> None:
    cfg = editing.LayoutConfig()
    assert editing._top_safe_margin(cfg) >= 192
    fit = getattr(editing, "_fit_title", None)
    assert callable(fit)
    from PIL import ImageFont
    for text in ["¿DE DÓNDE SALIÓ TANTA GENTE?", "PRUEBA DERIVADA: VERTICAL SIN WEBCAM"]:
        wrapped, size = fit(text, cfg)
        assert wrapped.replace("\n", " ") == text
        assert 28 <= size <= 88
        font = ImageFont.truetype(cfg.title_font_file, size)
        assert max(font.getlength(line)+4 for line in wrapped.splitlines()) <= cfg.width-cfg.safe_left-cfg.safe_right
        assert len(wrapped.splitlines())*(sum(font.getmetrics())+4) <= cfg.top_bar_height


@pytest.mark.parametrize("use_brand", [True, False])
def test_optional_short_title_does_not_hide_persistent_brand(
    use_brand: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(editing.ffmpeg, "run", lambda graph, **kw: commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing, "_probe_dimensions", lambda source: (1280, 720))
    cfg = editing.LayoutConfig(
        brand_image_path="assets/branding/KickAthanW.png" if use_brand else None,
        watermark_text=None if use_brand else "ATHANW",
        title_persistent=False,
    )
    editing.render_clip(Path("source.mp4"), tmp_path/"out.mp4",
                        ClipCandidate(1, 6, "", 1), layout=cfg, title_text="Hook")
    graph = " ".join(commands[-1])
    assert "*lt(t\\,4.000)" in graph, "Optional title endpoint remains exclusive"
    assert "gte(t\\,4.000)" not in graph, "Brand is independent of title duration"


def test_new_crop_field_preserves_legacy_positional_constructor() -> None:
    score = ViralityScore(total=7)
    clip = ClipCandidate(0, 1, "", 1, None, [], None, None, None, None, None, None, None, score, [])
    assert clip.virality is score
    assert clip.broll_inserts == []
    assert clip.segment_content_crops is None


def test_layout_config_keeps_legacy_positional_background_color() -> None:
    config = editing.LayoutConfig(1080, 1920, None, "#102030")
    assert config.background_color == "#102030"
    assert config.content_crop is None


def test_segment_frame_counts_share_one_output_clock() -> None:
    calculate = getattr(editing, "_segment_frame_counts", None)
    assert callable(calculate)
    assert calculate([(1, 1.1, "SPLIT"), (4, 4.1, "SPLIT"), (2, 2.1, "SPLIT")]) == [6, 6, 6]
    assert calculate([(0, .024, "SPLIT"), (1, 1.024, "SPLIT"), (2, 2.024, "SPLIT")]) == [1, 2, 1]
    with pytest.raises(ValueError, match="frame"):
        calculate([(0, .001, "SPLIT")])


def test_source_audio_ranges_only_merge_adjacent_ordered_edits() -> None:
    ranges = getattr(editing, "_source_audio_ranges", None)
    assert callable(ranges)
    assert ranges([(5, 6, "SPLIT"), (1, 2, "FULL_CAM"), (2, 3, "SPLIT")]) == [(5, 6), (1, 3)]
    assert ranges([(1, 2, "SPLIT"), (1, 2, "SPLIT")]) == [(1, 2), (1, 2)]


def test_caption_band_follows_segment_content_geometry() -> None:
    placement = getattr(editing, "_caption_margin_for_segment", None)
    assert callable(placement)
    cfg = editing.LayoutConfig()
    normal = placement(cfg, "SPLIT", (0, 0, 334, 271), (1280, 720))
    wide = placement(cfg, "SPLIT", (900, 0, 320, 180), (1280, 720))
    assert normal != wide
    for crop, margin in [((0, 0, 334, 271), normal), ((900, 0, 320, 180), wide)]:
        size = (1280, 720)  # no explicit content crop: preserve the whole source
        top, game_h, camera_y, camera_h = editing._split_panel_geometry(cfg, *size, *crop[2:])
        baseline = cfg.height - margin
        assert top < baseline < top+game_h*.65
        assert not camera_y < baseline < camera_y+camera_h


def test_real_graph_uses_60hz_canvas_and_final_original_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    def capture(graph: Any, **kwargs: Any) -> None:
        commands.append(editing.ffmpeg.compile(graph))
    monkeypatch.setattr(editing.ffmpeg, "run", capture)
    monkeypatch.setattr(editing, "_probe_dimensions", lambda source: (1280, 720))
    clip = ClipCandidate(start=1, end=4.1, text="", coherence=1,
        time_ranges=[(1, 1.1), (4, 4.1)],
        layout_segments=[(1, 1.1, "SPLIT"), (4, 4.1, "SPLIT")],
        segment_face_crops={0: (0, 0, 334, 271), 1: (900, 0, 320, 180)})
    editing.render_clip(Path("source.mp4"), tmp_path / "clip.mp4", clip)
    segment_commands = commands[:2]
    for command in segment_commands:
        assert any("color=c=#000000:s=1080x1920:r=60" in token for token in command)
        assert "fps=fps=60" in " ".join(command) or "fps=60" in " ".join(command)
        assert "-vframes" in command and command[command.index("-vframes") + 1] == "6"
        assert "tpad=" in " ".join(command), "A final partial frame needs bounded endpoint hold"
        assert "force_original_aspect_ratio=decrease" in " ".join(command), "Sharp game panel must not crop its lower HUD"
        assert "setpts=PTS-STARTPTS" in " ".join(command), "Source must start at zero before the canvas overlays, not one black lead frame"
    final = commands[-1]
    assert "source.mp4" in final, "Final audio must not be read from segment/join AAC"
    assert "concat=" in " ".join(final) and "a=1:n=2:v=0" in " ".join(final)
    assert "fps=" not in " ".join(final), "Do not resample an already exact60Hz segment clock again at EOF"
    assert "trim=end_frame=12" in " ".join(final)
    assert final[final.index("-ar") + 1] == "48000"


def test_video_join_does_not_wait_for_aac_segment_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(editing.ffmpeg, "run", lambda graph, **kw: commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing, "_probe_dimensions", lambda source: (1280, 720))
    clip = ClipCandidate(start=1, end=4.1, text="", coherence=1,
        time_ranges=[(1, 1.1), (4, 4.1)])
    editing.render_clip(Path("source.mp4"), tmp_path / "clip.mp4", clip)
    assert "concat=a=0:n=2:v=1" in " ".join(commands[-2])


def test_content_crop_is_independent_of_camera_and_safe_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(editing.ffmpeg, "run", lambda graph, **kw: commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing, "_probe_dimensions", lambda source: (1280, 720))
    clip = ClipCandidate(start=1, end=2, text="", coherence=1,
                         segment_face_crops={0: (0, 0, 334, 271)})
    editing.render_clip(Path("browser.mp4"), tmp_path / "safe.mp4", clip)
    assert "crop=934:720:346:0" not in " ".join(commands[0]), "A generic browser must not lose its left text column"
    commands.clear()
    clip.segment_content_crops = {0: (100, 280, 560, 360)}
    clip.segment_face_crops = {0: (20, 10, 292, 230)}
    editing.render_clip(Path("browser.mp4"), tmp_path / "focused.mp4", clip)
    assert "crop=560:360:100:280" in " ".join(commands[0])
    assert "crop=292:230:20:10" in " ".join(commands[0])


def test_pip_captions_stay_above_the_camera() -> None:
    cfg = editing.LayoutConfig()
    margin = editing._caption_margin_for_segment(cfg, "PICTURE_IN_PICTURE", (0, 0, 334, 271), (1280, 720))
    assert cfg.height - margin < round(cfg.height * .64)


def test_pip_caption_uses_the_same_clamped_roi_as_rendering() -> None:
    cfg = editing.LayoutConfig()
    margin = editing._caption_margin_for_segment(cfg, "PIP", (1180, 0, 1000, 500), (1280, 720))
    assert margin == 888  # actual clamp is100x500, PIP top1056, caption bottom1032


@pytest.mark.parametrize("mode", ["FULL_CONTENT", "FULL_CAM", "PICTURE_IN_PICTURE"])
def test_full_canvas_content_is_fitted_below_required_header(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(editing.ffmpeg, "run", lambda graph, **kw: commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing, "_probe_dimensions", lambda source: (1280, 720))
    clip = ClipCandidate(1, 2, "", 1, layout=mode, segment_face_crops={0: (0, 0, 340, 270)})
    editing.render_clip(Path("source.mp4"), tmp_path/"out.mp4", clip,
        layout=editing.LayoutConfig(integrated_overlays=False), title_text="Hook")
    graph = " ".join(commands[0])
    assert "scale=1080:1520:force_original_aspect_ratio=decrease" in graph
    assert "pad=1080:1920:0:400" in graph


def test_final_audio_is_not_truncated_by_video_frame_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(editing.ffmpeg, "run", lambda graph, **kw: commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing, "_probe_dimensions", lambda source: (1280, 720))
    editing.render_clip(Path("source.mp4"), tmp_path/"out.mp4", ClipCandidate(1, 1.2, "", 1))
    assert "-vframes" not in commands[-1]
    assert "trim=end_frame=12" in " ".join(commands[-1])


def test_encoded_layouts_have_square_pixels_and_explicit_60fps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(editing.ffmpeg, "run", lambda graph, **kw: commands.append(editing.ffmpeg.compile(graph)))
    monkeypatch.setattr(editing, "_probe_dimensions", lambda source: (1280, 720))
    clip = ClipCandidate(1, 4.1, "", 1, time_ranges=[(1, 1.1), (4, 4.1)])
    editing.render_clip(Path("source.mp4"), tmp_path/"out.mp4", clip)
    assert "setsar=1" in " ".join(commands[0])
    for command in commands:
        assert command[command.index("-r")+1] == "60"


@pytest.mark.parametrize("camera_size", [(340, 270), (300, 500), (640, 180)])
def test_pip_follows_aspect_and_keeps_caption_above_actual_camera(camera_size: tuple[int, int]) -> None:
    geometry = getattr(editing, "_pip_geometry", None)
    assert callable(geometry)
    cfg = editing.LayoutConfig()
    x, y, w, h = geometry(cfg, *camera_size)
    assert x == 32 and x+w < cfg.width*.5
    assert y+h <= cfg.height*.8
    assert w % 2 == h % 2 == y % 2 == 0
    margin = editing._caption_margin_for_segment(cfg, "PIP", (0, 0, *camera_size), (1280, 720))
    assert cfg.height-margin < y
