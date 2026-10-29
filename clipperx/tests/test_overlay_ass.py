"""HOST-authored overlay plans and literal text/layout checks."""

from pathlib import Path
import subprocess
from fractions import Fraction as F

import pytest

from clipperx.overlay_ass import (
    athan_adaptive_caption, build_overlay_ass, compile_overlay_ass, escape_ass_text,
    write_overlay_ass,
)
from clipperx.render_assets import RenderAssetRegistry, default_render_assets
from clipperx.schema import TranscriptSegment, Word
from clipperx.tests.test_output_timeline import shot
from clipperx.visual_contracts import (
    NormalizedRect, OverlayPlan, SubtitlePlan, VisualCompositionPlan,
)


def plan_with_two_subtitle_regions() -> VisualCompositionPlan:
    shots = [shot((10, 12), (0, 2)), shot((20, 22), (2, 4), "b")]
    for item, y in zip(shots, (.5, .65)):
        item.subtitles = SubtitlePlan(destination={"x": .1, "y": y, "width": .8,
                                                   "height": .1})  # type: ignore[arg-type]
    return VisualCompositionPlan.model_validate({
        "clip_id": "host", "source_fingerprint": "host", "composition_fingerprint": "host",
        "source_timeline_fingerprint": "host", "identity_fingerprint": "host",
        "style_library_fingerprint": "host", "shots": [s.model_dump() for s in shots]})


def test_subtitle_position_changes_by_shot() -> None:
    plan = plan_with_two_subtitle_regions()
    ass = build_overlay_ass(plan, [TranscriptSegment(10, 12, "Hola"),
                                   TranscriptSegment(20, 22, "Mundo")])
    assert r"\pos(540,1056)" in ass
    assert r"\pos(540,1344)" in ass
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass


def test_default_subtitle_is_chicken_yellow_with_black_outline() -> None:
    style = athan_adaptive_caption()
    assert style.primary_color == "&H001AD5FF"
    assert style.outline_color == "&H00000000"
    assert style.bold is True


def test_word_limits_wrapping_and_punctuation_have_real_layout_metadata() -> None:
    plan = plan_with_two_subtitle_regions()
    sub = plan.shots[0].subtitles
    assert sub is not None
    sub.max_words_per_event = 3
    sub.max_lines = 2
    words = [Word(10 + i / 5, 10 + (i + 1) / 5, t)
             for i, t in enumerate(["Hola", ",", "mundo", "estupendo", "hoy"])]
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "unused", words)])
    assert len(result.events) == 5  # one reached prefix per timed token
    assert list(dict.fromkeys(e.text for e in result.events)) == ["Hola, mundo", "estupendo hoy"]
    assert result.events[0].lines == ("Hola, mundo",)
    for event in result.events:
        assert event.word_count <= 3 and len(event.lines) <= 2
        assert event.measured_width <= event.destination.width
        assert event.measured_height <= event.destination.height


def test_title_autofits_impact_at_most_46_and_shot_local_time() -> None:
    plan = plan_with_two_subtitle_regions()
    plan.shots[1].title = OverlayPlan(text="Un titular largo con palabras completas " * 3,
        start=.25, end=1.75, destination=NormalizedRect(x=.2, y=.1, width=.6, height=.08),
        z_index=4)
    result = compile_overlay_ass(plan, [])
    event, = result.events
    assert (event.start, event.end) == (2.25, 3.75)
    assert event.font_family == "Impact"
    assert 1 <= event.font_size <= 46
    assert event.measured_width <= event.destination.width
    assert event.measured_height <= event.destination.height
    assert plan.shots[1].title.text is not None
    assert " ".join(event.lines) == plan.shots[1].title.text.strip()
    assert "…" not in result.document
    plan.shots[1].title.text = "Corto"
    short, = compile_overlay_ass(plan, []).events
    assert short.font_size == 46  # choose the largest fitting registered face


def test_active_word_intervals_only_when_requested_and_gaps_stay_unemphasized() -> None:
    plan = plan_with_two_subtitle_regions()
    words = [Word(10.1, 10.4, "uno"), Word(10.6, 11, "dos")]
    transcript = [TranscriptSegment(10, 12, "uno dos", words)]
    plain = compile_overlay_ass(plan, transcript)
    assert len(plain.events) == 2 and all(e.active_words == () for e in plain.events)
    assert plan.shots[0].subtitles is not None
    plan.shots[0].subtitles.emphasize_active_word = True
    result = compile_overlay_ass(plan, transcript)
    assert [(round(e.start, 3), round(e.end, 3), e.active_words) for e in result.events] == [
        (.1, .4, (0,)), (.4, .6, ()), (.6, 1, (1,))]
    assert r"\1c&H00FFFFFF&" in result.document
    active_line = next(line for line in result.document.splitlines() if line.startswith("Dialogue"))
    assert r"{\alpha&HFF&} dos" in active_line  # future suffix keeps its measured space
    with pytest.raises(ValueError, match="word"):
        compile_overlay_ass(plan, [TranscriptSegment(10, 12, "missing timestamps")])


@pytest.mark.parametrize("emphasis", [False, True])
def test_timed_captions_reveal_only_reached_words_with_stable_full_run_layout(
    emphasis: bool,
) -> None:
    from clipperx.overlay_ass import _event_text

    plan = plan_with_two_subtitle_regions()
    sub = plan.shots[0].subtitles
    assert sub is not None
    sub.emphasize_active_word = emphasis
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "uno dos tres", [
        Word(10.1, 10.4, "uno"), Word(10.5, 10.8, "dos"), Word(11, 11.4, "tres")])])
    expected = {F(1, 10): "uno", F(1, 2): "uno dos", F(1): "uno dos tres"}
    for start, prefix in expected.items():
        event = next(e for e in result.events if e.exact_interval[0] == start)
        assert event.text == "uno dos tres" and event.lines == ("uno dos tres",)
        assert event.visible_end == len(prefix)
        rendered = _event_text(event)
        if prefix != event.text:
            assert r"{\alpha&HFF&}" in rendered
            assert rendered.split(r"{\alpha&HFF&}", 1)[1] == event.text[len(prefix):]
    assert len({(e.line_ranges, e.measured_width, e.measured_height) for e in result.events}) == 1
    assert all(e.exact_interval[1] <= F(7, 5) for e in result.events)


def test_timed_captions_end_before_a_long_pause_instead_of_showing_stale_text() -> None:
    plan = plan_with_two_subtitle_regions()
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "uno dos", [
        Word(10.1, 10.3, "uno"), Word(11.5, 11.8, "dos")])])
    assert [(e.exact_interval, e.text) for e in result.events] == [
        ((F(1, 10), F(3, 10)), "uno"), ((F(3, 2), F(9, 5)), "dos")]
    assert all(not e.start <= .9 < e.end for e in result.events)


def test_short_pause_keeps_coherent_caption_until_next_reveal() -> None:
    result = compile_overlay_ass(plan_with_two_subtitle_regions(), [
        TranscriptSegment(10, 12, "uno dos", [
            Word(10.1, 10.3, "uno"), Word(10.65, 10.9, "dos")])])
    assert [e.text for e in result.events] == ["uno dos", "uno dos"]
    assert [e.exact_interval for e in result.events] == [(F(1, 10), F(13, 20)),
                                                       (F(13, 20), F(9, 10))]


@pytest.mark.parametrize("emphasis", [False, True])
def test_terminal_native_point_word_gets_bounded_display_dwell_without_retiming(
    emphasis: bool,
) -> None:
    plan = plan_with_two_subtitle_regions()
    sub = plan.shots[0].subtitles
    assert sub is not None
    sub.emphasize_active_word = emphasis
    transcript = [TranscriptSegment(10, 11, "uno fin", [
        Word(10.1, 10.4, "uno"), Word(10.5, 10.5, "fin")])]
    result = compile_overlay_ass(plan, transcript)
    terminal = result.events[-1]
    assert terminal.visible_end == len("uno fin")
    assert terminal.exact_interval == (F(1, 2), F(1, 2) + F(1, 30))
    assert terminal.active_words == ()  # a point timestamp is not invented acoustic duration
    line = next(line for line in result.document.splitlines()
                if line.startswith(f"Dialogue: {terminal.layer},0:00:00.50,"))
    assert ",0:00:00.53," in line
    assert transcript[0].words[-1].start == transcript[0].words[-1].end == 10.5


def test_point_only_caption_uses_its_anchor_and_stays_inside_the_shot() -> None:
    plan = plan_with_two_subtitle_regions()
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "fin", [
        Word(11.98, 11.98, "fin")])])
    event, = result.events
    assert event.exact_interval == (F(99, 50), F(2))
    assert event.visible_end == 3 and event.text == "fin"


@pytest.mark.parametrize("emphasis", [False, True])
@pytest.mark.parametrize("following_point", [False, True])
def test_point_display_dwell_stops_at_next_caption_in_another_segment(
    emphasis: bool, following_point: bool,
) -> None:
    plan = plan_with_two_subtitle_regions()
    assert plan.shots[0].subtitles is not None
    plan.shots[0].subtitles.emphasize_active_word = emphasis
    transcript = [
        TranscriptSegment(10, 11, "fin", [Word(10.9, 10.9, "fin")]),
        # The segment envelope is not the acoustic onset of its first word.
        TranscriptSegment(10.91, 11.5, "nuevo", [
            Word(10.92, 10.92 if following_point else 11.2, "nuevo")]),
    ]
    result = compile_overlay_ass(plan, transcript)
    first = next(e for e in result.events if e.text == "fin")
    second = next(e for e in result.events if e.text == "nuevo")
    assert first.exact_interval == (F(9, 10), F(23, 25))
    assert first.exact_interval[1] == second.exact_interval[0]
    assert first.active_words == ()
    assert transcript[0].words[0].start == transcript[0].words[0].end == 10.9


def test_point_display_dwell_stops_before_following_untimed_segment() -> None:
    result = compile_overlay_ass(plan_with_two_subtitle_regions(), [
        TranscriptSegment(10, 11, "fin", [Word(10.9, 10.9, "fin")]),
        TranscriptSegment(10.92, 11.5, "nuevo"),
    ])
    assert [e.exact_interval for e in result.events] == [
        (F(9, 10), F(23, 25)), (F(23, 25), F(3, 2))]


def test_point_display_limits_follow_each_run_not_coarse_segment_order() -> None:
    result = compile_overlay_ass(plan_with_two_subtitle_regions(), [
        TranscriptSegment(10, 12, "uno fin", [
            Word(10.1, 10.1, "uno"), Word(11.5, 11.5, "fin")]),
        TranscriptSegment(10.11, 11, "otro", [Word(10.12, 10.4, "otro")]),
    ])
    assert {e.text: e.exact_interval for e in result.events} == {
        "uno": (F(1, 10), F(3, 25)),
        "fin": (F(3, 2), F(3, 2) + F(1, 30)),
        "otro": (F(3, 25), F(2, 5)),
    }


def test_point_display_limit_sees_later_run_in_an_earlier_segment() -> None:
    result = compile_overlay_ass(plan_with_two_subtitle_regions(), [
        TranscriptSegment(10, 12, "uno nuevo", [
            Word(10.1, 10.2, "uno"), Word(10.92, 11.2, "nuevo")]),
        TranscriptSegment(10.8, 11.5, "fin", [Word(10.9, 10.9, "fin")]),
    ])
    assert {e.text: e.exact_interval for e in result.events} == {
        "uno": (F(1, 10), F(1, 5)),
        "nuevo": (F(23, 25), F(6, 5)),
        "fin": (F(9, 10), F(23, 25)),
    }


def test_point_display_limit_does_not_shorten_known_acoustic_overlap() -> None:
    with pytest.raises(ValueError, match="Overlapping emitted subtitle events"):
        compile_overlay_ass(plan_with_two_subtitle_regions(), [
            TranscriptSegment(10, 11, "uno fin", [
                Word(10.8, 10.95, "uno"), Word(10.9, 10.9, "fin")]),
            TranscriptSegment(10.92, 11.5, "nuevo", [Word(10.92, 11.2, "nuevo")]),
        ])


@pytest.mark.parametrize("prefix", [True, -1, 4, .5, "1"])
def test_visible_prefix_rejects_invalid_offsets_even_after_frozen_record_tampering(
    prefix: object,
) -> None:
    from dataclasses import replace
    from clipperx.overlay_ass import serialize_overlay_events

    event, = compile_overlay_ass(plan_with_two_subtitle_regions(), [
        TranscriptSegment(10, 11, "uno")]).events
    with pytest.raises(ValueError, match="visible prefix"):
        replace(event, visible_end=prefix)  # type: ignore[arg-type]
    object.__setattr__(event, "visible_end", prefix)
    with pytest.raises(ValueError, match="visible prefix"):
        serialize_overlay_events([event])


def test_title_cannot_use_caption_reveal_metadata_to_hide_known_headline_text() -> None:
    from dataclasses import replace
    from clipperx.overlay_ass import serialize_overlay_events

    event, = compile_overlay_ass(plan_with_two_subtitle_regions(), [
        TranscriptSegment(10, 11, "uno")]).events
    with pytest.raises(ValueError, match="subtitle"):
        replace(event, kind="title", visible_end=1)
    title = replace(event, kind="title")
    object.__setattr__(title, "visible_end", 1)
    with pytest.raises(ValueError, match="subtitle"):
        serialize_overlay_events([title])


@pytest.mark.parametrize("anchor", [10.001, 10.009, 10.015, 10.501, 10.999, 11.98])
def test_point_display_dwell_survives_ass_and_absolute_native_raster_projection(
    anchor: float,
) -> None:
    from dataclasses import replace
    from clipperx.output_timeline import OutputTimeline
    from clipperx.overlay_ass import serialize_overlay_events
    from clipperx.visual_compiler import _ass_boundary, _validate_overlays

    plan = plan_with_two_subtitle_regions()
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "fin", [
        Word(anchor, anchor, "fin")])])
    event, = result.events
    (left, right), = _validate_overlays(plan, OutputTimeline.from_shots(plan.shots), result)
    start, end = _ass_boundary(left), _ass_boundary(right)
    assert right > left and any(start <= F(k, 60) < end for k in range(left, right))
    assert event.exact_interval[1] <= F(2)
    projected = replace(event, start=float(start), end=float(end), exact_interval=(start, end))
    assert "fin" in serialize_overlay_events([projected])


def test_point_too_close_to_final_cut_fails_closed_without_retiming_or_omission() -> None:
    transcript = [TranscriptSegment(10, 12, "fin", [Word(11.999, 11.999, "fin")])]
    with pytest.raises(ValueError, match="ASS centisecond precision"):
        compile_overlay_ass(plan_with_two_subtitle_regions(), transcript)
    assert transcript[0].words[0].start == transcript[0].words[0].end == 11.999


def test_untrusted_text_escaped_before_host_tags_and_file_is_host_addressed(tmp_path: Path) -> None:
    text = "{\\pos(0,0)} literal \\N \\h\nnext"
    escaped = escape_ass_text(text)
    assert r"\{" in escaped and r"\}" in escaped
    assert "\\\u2060N" in escaped and escaped.endswith(r"\Nnext")
    plan = plan_with_two_subtitle_regions()
    plan.clip_id = "../not-a-path"
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, text)])
    path = write_overlay_ass(result.document, output_dir=tmp_path)
    assert path.parent == tmp_path and path.suffix == ".ass"
    assert path.read_text(encoding="utf-8-sig") == result.document
    assert write_overlay_ass(result.document, output_dir=tmp_path) == path
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_explicit_fonts_resolve_and_invalid_layout_does_not_truncate() -> None:
    plan = plan_with_two_subtitle_regions()
    assert plan.shots[0].subtitles is not None
    plan.shots[0].subtitles.font_asset_id = "missing"
    with pytest.raises(ValueError, match="Unknown font"):
        compile_overlay_ass(plan, [TranscriptSegment(10, 12, "Hi")])
    plan.shots[0].subtitles.font_asset_id = "subtitle_bold"
    with pytest.raises(ValueError, match="fit"):
        compile_overlay_ass(plan, [TranscriptSegment(10, 12, "W" * 100)])
    assert default_render_assets().fonts["subtitle_bold"].name == "arialbd.ttf"


def test_unsupported_italic_face_fails_instead_of_mismeasuring_regular_output() -> None:
    registry = RenderAssetRegistry({}, {"subtitle_bold": Path("C:/Windows/Fonts/arialbi.ttf")}, {})
    with pytest.raises(ValueError, match="italic"):
        compile_overlay_ass(plan_with_two_subtitle_regions(),
                            [TranscriptSegment(10, 11, "known text")], assets=registry)


def test_missing_vs_silent_and_compiled_result_detaches_plan() -> None:
    plan = plan_with_two_subtitle_regions()
    with pytest.raises(ValueError, match="Missing"):
        compile_overlay_ass(plan, None)
    assert compile_overlay_ass(plan, []).events == ()
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "hello")])
    before = result.document
    plan.shots[0].subtitles.destination.y = .1  # type: ignore[union-attr]
    assert result.events[0].destination.y == 960 and result.document == before
    from clipperx.overlay_ass import OverlayCompilation
    events = list(result.events)
    fonts = list(result.font_paths)
    detached = OverlayCompilation(result.document, events, fonts)  # type: ignore[arg-type]
    events.clear()
    fonts.clear()
    assert detached.events == result.events and detached.font_paths == result.font_paths


def test_segment_only_limit_error_requires_word_timestamps_without_omission() -> None:
    plan = plan_with_two_subtitle_regions()
    text = "uno dos tres cuatro cinco seis siete"
    with pytest.raises(ValueError, match="word timestamps"):
        compile_overlay_ass(plan, [TranscriptSegment(10, 12, text)])
    sub = plan.shots[0].subtitles
    assert sub is not None
    sub.max_words_per_event = 10
    sub.max_lines = 1
    with pytest.raises(ValueError, match="word timestamps.*fit"):
        compile_overlay_ass(plan, [TranscriptSegment(10, 12, "WWW " * 10)])
    sub.max_words_per_event = 2
    words = [Word(10 + i / 5, 10 + (i + 1) / 5, token)
             for i, token in enumerate(text.split())]
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, text, words)])
    assert " ".join(dict.fromkeys(event.text for event in result.events)) == text


def test_multiword_asr_token_cannot_bypass_word_limit_or_invent_times() -> None:
    plan = plan_with_two_subtitle_regions()
    assert plan.shots[0].subtitles is not None
    plan.shots[0].subtitles.max_words_per_event = 2
    phrase = "uno dos tres"
    with pytest.raises(ValueError, match="word timestamps"):
        compile_overlay_ass(plan, [TranscriptSegment(10, 11, phrase, [Word(10, 11, phrase)])])
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "unused", [
        Word(10, 10.5, "uno dos"), Word(10.5, 11, "tres")])])
    assert [e.lines for e in result.events] == [("uno dos",), ("tres",)]
    assert [e.word_count for e in result.events] == [2, 1]


def test_repeated_ranges_produce_subtitles_in_both_output_occurrences() -> None:
    plan = plan_with_two_subtitle_regions()
    plan.shots[1].source = plan.shots[0].source.model_copy(deep=True)
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 11, "same")])
    assert [(e.shot_id, e.start, e.end, e.lines) for e in result.events] == [
        ("a", 0, 1, ("same",)), ("b", 2, 3, ("same",))]


def test_brand_is_only_registered_png_and_global_output_metadata(tmp_path: Path) -> None:
    plan = plan_with_two_subtitle_regions()
    plan.shots[1].brand = OverlayPlan(asset_id="kick_brand", start=.1, end=1.5,
        destination=NormalizedRect(x=.25, y=.8, width=.5, height=.05), z_index=4)
    with pytest.raises(ValueError, match="host-provided"):
        compile_overlay_ass(plan, [])
    result = compile_overlay_ass(plan, [], brand_cache_dir=tmp_path)
    brand, = result.brands
    assert (brand.start, brand.end, brand.shot_id, brand.z_index) == (2.1, 3.5, "b", 4)
    assert brand.asset.path.is_file()
    assert "kick.com" not in result.document and "drawtext" not in result.document
    assert "Dialogue:" not in result.document


def test_overlay_event_detaches_nested_ranges_and_lines() -> None:
    from dataclasses import replace

    plan = plan_with_two_subtitle_regions()
    event, = compile_overlay_ass(plan, [TranscriptSegment(10, 11, "hello")]).events
    lines, line_ranges = list(event.lines), [list(pair) for pair in event.line_ranges]
    active = [0]
    words = [[0, 5]]
    detached = replace(event, lines=lines, line_ranges=line_ranges, active_words=active,  # type: ignore[arg-type]
                       word_ranges=words)  # type: ignore[arg-type]
    lines.clear()
    line_ranges[0].clear()
    active.clear()
    words[0].clear()
    assert detached.lines == ("hello",) and detached.line_ranges == ((0, 5),)
    assert detached.active_words == (0,) and detached.word_ranges == ((0, 5),)


def _decode_ass(document: str, directory: Path, *, at: float = .5) -> tuple[bytes, str]:
    """Bounded synthetic frame only, explicit already-provisioned Task3 binary."""
    ffmpeg = Path(__file__).resolve().parents[2] / "data/tools/ffmpeg/9.0.2/bin/ffmpeg.exe"
    assert ffmpeg.is_file(), "Task3's explicit runtime is required for this decoded diagnostic"
    ass = write_overlay_ass(document, output_dir=directory)
    result = subprocess.run([
        str(ffmpeg), "-hide_banner", "-nostdin", "-f", "lavfi", "-i",
        "color=c=0x607080:s=1080x1920:r=10:d=0.1,format=rgba",
        "-vf", f"setpts=PTS+{at}/TB,ass=filename={ass.name}",
        "-frames:v", "1", "-fps_mode", "passthrough",
        "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1",
    ], cwd=directory, capture_output=True, check=True, timeout=30)
    assert len(result.stdout) == 1080 * 1920 * 4
    log = result.stderr.decode("utf-8", errors="replace")
    (directory / f"{ass.stem}-{at}.log").write_text(log, encoding="utf-8")
    return result.stdout, log


@pytest.mark.parametrize("wrapped", [False, True])
def test_decoded_timed_prefix_hides_future_words_and_preserves_the_first_word_position(
    tmp_path: Path, wrapped: bool,
) -> None:
    from dataclasses import replace
    from PIL import Image, ImageChops
    from clipperx.overlay_ass import serialize_overlay_events

    plan = plan_with_two_subtitle_regions()
    sub = plan.shots[0].subtitles
    assert sub is not None
    sub.destination.width = .4 if wrapped else .8
    tokens = ["palabras", "medidas", "dos"] if wrapped else ["uno", "dos", "tres"]
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, " ".join(tokens), [
        Word(10.1, 10.5, tokens[0]), Word(10.6, 11, tokens[1]), Word(11.1, 11.5, tokens[2])])])
    assert len(result.events[0].lines) == (2 if wrapped else 1)
    early, _ = _decode_ass(result.document, tmp_path, at=.3)
    middle, _ = _decode_ass(result.document, tmp_path, at=.8)
    full, _ = _decode_ass(result.document, tmp_path, at=1.3)
    # Independent ASS expectation keeps full layout and hides only the suffix.
    reference = serialize_overlay_events([replace(result.events[-1], start=.1, end=1.5,
        exact_interval=(F(1, 10), F(3, 2)), visible_end=None)])
    hidden = reference.replace(tokens[0], tokens[0] + r"{\alpha&HFF&}")
    expected_early, _ = _decode_ass(hidden, tmp_path, at=.3)
    assert early == expected_early and early != middle != full
    first = Image.frombytes("RGBA", (1080, 1920), early)
    last = Image.frombytes("RGBA", (1080, 1920), full)
    background = Image.new("RGBA", first.size, first.getpixel((0, 0)))
    bbox = ImageChops.difference(first, background).convert("RGB").getbbox()
    assert bbox is not None and first.crop(bbox).tobytes() == last.crop(bbox).tobytes()
    for name, pixels in (("early", early), ("middle", middle), ("full", full)):
        Image.frombytes("RGBA", first.size, pixels).save(tmp_path / f"reveal-{name}.png")
    print(f"decoded prefix evidence: {tmp_path}; first_word_bbox={bbox}")


def test_decoded_libass_literals_registered_fonts_wrapping_and_bounds(tmp_path: Path) -> None:
    from PIL import Image, ImageChops

    plan = plan_with_two_subtitle_regions()
    sub = plan.shots[0].subtitles
    assert sub is not None
    sub.max_words_per_event = 10
    sub.destination.width = .8
    # The malicious-looking commands and backslash sequences must be visible text.
    text = r"{\fs150} A\NB \h C \\ end"
    compiled = compile_overlay_ass(plan, [TranscriptSegment(10, 12, text)])
    event, = compiled.events
    decoded, log = _decode_ass(compiled.document, tmp_path)
    image = Image.frombytes("RGBA", (1080, 1920), decoded)
    background = Image.new("RGBA", image.size, image.getpixel((0, 0)))
    bbox = ImageChops.difference(image, background).convert("RGB").getbbox()
    assert bbox is not None
    p = event.destination
    assert p.x <= bbox[0] < bbox[2] <= p.x + p.width
    assert p.y <= bbox[1] < bbox[3] <= p.y + p.height
    assert bbox[2] - bbox[0] > 450  # literal command/backslash glyphs actually occupy the line
    assert bbox[3] - bbox[1] < 80   # raw backslash-N did not create an extra text line
    assert "Arial" in log and "700" in log  # registered Arial Bold selected by libass
    image.crop((p.x, p.y, p.x + p.width, p.y + p.height)).save(tmp_path / "literal-subtitle.png")

    # Independent reference uses ZERO WIDTH SPACE instead of compiler WORD JOINER.
    # Equal decoded pixels plus the raw trailing-slash oracle prove actual visible
    # literals, rather than only escaped string fragments or a hidden glyph.
    independent = compiled.document.replace("\\\u2060", "\\\u200b")
    reference, _ = _decode_ass(independent, tmp_path)
    assert decoded == reference
    slash = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "\\")])
    guarded, _ = _decode_ass(slash.document, tmp_path)
    trailing_slash, _ = _decode_ass(slash.document.replace("\\\u2060", "\\"), tmp_path)
    assert guarded == trailing_slash

    # Force two genuine measured lines, then check both full outline bounds.
    sub.destination.width = .4
    words = [Word(10 + i / 5, 10 + (i + 1) / 5, token)
             for i, token in enumerate(["palabras", "medidas", "dos", "líneas"])]
    wrapped = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "unused", words)])
    assert len(wrapped.events[0].lines) == 2
    assert " ".join(dict.fromkeys(e.text for e in wrapped.events)) == "palabras medidas dos líneas"
    assert "unused" not in wrapped.document  # use known timed words, not unrelated metadata
    pixels, _ = _decode_ass(wrapped.document, tmp_path)
    wrapped_image = Image.frombytes("RGBA", image.size, pixels)
    wrap_bbox = ImageChops.difference(wrapped_image, background).convert("RGB").getbbox()
    assert wrap_bbox is not None
    rect = wrapped.events[0].destination
    assert rect.x <= wrap_bbox[0] < wrap_bbox[2] <= rect.x + rect.width
    assert rect.y <= wrap_bbox[1] < wrap_bbox[3] <= rect.y + rect.height
    assert wrap_bbox[3] - wrap_bbox[1] > 75
    wrapped_image.crop((rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)).save(
        tmp_path / "wrapped-subtitle.png")

    plan.shots[0].subtitles = None
    plan.shots[1].subtitles = None
    plan.shots[1].title = OverlayPlan(text="Titular completo con palabras medidas " * 3,
        start=0, end=2, destination=NormalizedRect(x=.25, y=.1, width=.5, height=.08), z_index=2)
    title = compile_overlay_ass(plan, None)
    title_event, = title.events
    title_pixels, title_log = _decode_ass(title.document, tmp_path, at=2.5)
    title_image = Image.frombytes("RGBA", image.size, title_pixels)
    title_bbox = ImageChops.difference(title_image, background).convert("RGB").getbbox()
    assert title_bbox is not None and "Impact" in title_log
    r = title_event.destination
    assert r.x <= title_bbox[0] < title_bbox[2] <= r.x + r.width
    assert r.y <= title_bbox[1] < title_bbox[3] <= r.y + r.height
    assert title_event.font_size < 46 and len(title_event.lines) > 1
    title_image.crop((r.x, r.y, r.x + r.width, r.y + r.height)).save(tmp_path / "fitted-title.png")
    print(f"decoded ASS diagnostics: {tmp_path}; literal_bbox={bbox}; wrap_bbox={wrap_bbox}; "
          f"title_bbox={title_bbox}; title_size={title_event.font_size}")


def test_ass_projection_uses_same_immutable_events_without_double_escaping() -> None:
    from clipperx.overlay_ass import serialize_overlay_events

    plan = plan_with_two_subtitle_regions()
    text = r"{\fs150} A\N"
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 11, text),
                                       TranscriptSegment(20, 21, "second")])
    assert serialize_overlay_events(result.events) == result.document
    projection = serialize_overlay_events([result.events[0]])
    dialogue = [line for line in projection.splitlines() if line.startswith("Dialogue:")]
    canonical = [line for line in result.document.splitlines() if line.startswith("Dialogue:")]
    assert dialogue == canonical[:1]
    assert "second" not in projection
    assert "\\\u2060N" in projection and "\\\u2060\u2060N" not in projection
    assert result.events[0].text == text  # raw literals retained; serializer owns escaping


def test_host_authored_ass_declares_rgb_matrix_for_every_serialized_document() -> None:
    from clipperx.overlay_ass import serialize_overlay_events

    plan = plan_with_two_subtitle_regions()
    result = compile_overlay_ass(plan, [TranscriptSegment(10.25, 10.75, "RGB text")])
    projection = serialize_overlay_events(result.events)
    empty = compile_overlay_ass(plan, []).document
    old_header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
    )

    for document in (result.document, projection, empty):
        assert document.count("YCbCr Matrix: None\n") == 1
        assert document.index("PlayResY: 1920\n") < document.index("YCbCr Matrix: None\n")
        assert document.index("YCbCr Matrix: None\n") < document.index("WrapStyle: 2\n")
        style_prefix_end = document.index("[V4+ Styles]\n") + len("[V4+ Styles]\n")
        prefix = document[:style_prefix_end]
        assert prefix.replace("YCbCr Matrix: None\n", "", 1) == old_header

    assert result.document == projection
    event, = result.events
    assert event.primary_color == "&H001AD5FF"
    assert event.exact_interval == (F(1, 4), F(3, 4))
    assert "Dialogue: 1,0:00:00.25,0:00:00.75," in result.document
    assert "RGB text" in result.document


@pytest.mark.parametrize("emphasis", [False, True])
def test_overlapping_words_split_by_word_limit_reject_emitted_collision(emphasis: bool) -> None:
    plan = plan_with_two_subtitle_regions()
    sub = plan.shots[0].subtitles
    assert sub is not None
    sub.max_words_per_event = 1
    sub.emphasize_active_word = emphasis
    transcript = [TranscriptSegment(10, 12, "uno dos", [
        Word(10, 11, "uno"), Word(10.5, 11.5, "dos")])]
    before = [(w.start, w.end, w.word) for w in transcript[0].words]
    with pytest.raises(ValueError, match="Overlapping emitted subtitle events"):
        compile_overlay_ass(plan, transcript)
    assert [(w.start, w.end, w.word) for w in transcript[0].words] == before


def test_overlapping_words_split_by_geometry_reject_emitted_collision() -> None:
    plan = plan_with_two_subtitle_regions()
    sub = plan.shots[0].subtitles
    assert sub is not None
    sub.destination.width = .3
    sub.max_lines = 1
    # Each word fits, but their combined measured line does not.
    assert len(compile_overlay_ass(plan, [TranscriptSegment(10, 12, "palabras medidas", [
        Word(10, 10.5, "palabras"), Word(10.5, 11, "medidas")])]).events) == 2
    with pytest.raises(ValueError, match="Overlapping emitted subtitle events"):
        compile_overlay_ass(plan, [TranscriptSegment(10, 12, "palabras medidas", [
            Word(10, 11, "palabras"), Word(10.5, 11.5, "medidas")])])


@pytest.mark.parametrize("with_words", [False, True])
def test_separate_transcript_segments_reject_emitted_collision(with_words: bool) -> None:
    plan = plan_with_two_subtitle_regions()
    transcript = [TranscriptSegment(10, 11, "uno", [Word(10, 11, "uno")] if with_words else []),
                  TranscriptSegment(10.5, 11.5, "dos",
                                    [Word(10.5, 11.5, "dos")] if with_words else [])]
    with pytest.raises(ValueError, match="Overlapping emitted subtitle events"):
        compile_overlay_ass(plan, transcript)


@pytest.mark.parametrize("emphasis", [False, True])
def test_overlapping_words_in_one_coherent_caption_remain_valid(emphasis: bool) -> None:
    plan = plan_with_two_subtitle_regions()
    sub = plan.shots[0].subtitles
    assert sub is not None
    sub.max_words_per_event = 2
    sub.emphasize_active_word = emphasis
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "uno dos", [
        Word(10, 11, "uno"), Word(10.5, 11.5, "dos")])])
    assert all(e.lines == ("uno dos",) and e.word_count == 2 for e in result.events)
    if emphasis:
        assert [(e.start, e.end, e.active_words) for e in result.events] == [
            (0, .5, (0,)), (.5, 1, (0, 1)), (1, 1.5, (1,))]
    else:
        assert [(e.start, e.end, e.active_words) for e in result.events] == [
            (0, .5, ()), (.5, 1.5, ())]


@pytest.mark.parametrize("emphasis", [False, True])
def test_adjacent_emitted_caption_intervals_are_valid(emphasis: bool) -> None:
    plan = plan_with_two_subtitle_regions()
    sub = plan.shots[0].subtitles
    assert sub is not None
    sub.max_words_per_event = 1
    sub.emphasize_active_word = emphasis
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 12, "uno dos", [
        Word(10, 11, "uno"), Word(11, 12, "dos")])])
    assert [(e.start, e.end, e.lines) for e in result.events] == [
        (0, 1, ("uno",)), (1, 2, ("dos",))]


def test_projection_rejects_emitted_caption_collision_regardless_of_layer_or_input_order() -> None:
    from dataclasses import replace
    from clipperx.overlay_ass import serialize_overlay_events

    first, = compile_overlay_ass(plan_with_two_subtitle_regions(),
                                 [TranscriptSegment(10, 11, "uno")]).events
    second = replace(first, start=.5, end=1.5, layer=first.layer + 1)
    for events in ([first, second], [second, first]):
        with pytest.raises(ValueError, match="Overlapping emitted subtitle events"):
            serialize_overlay_events(events)
    # The validation concerns captions, not planned title/caption interleaving.
    assert serialize_overlay_events([first, replace(second, kind="title")]).count("Dialogue:") == 2
    assert serialize_overlay_events([second]).count("Dialogue:") == 1


@pytest.mark.parametrize("kind", ["title", "subtitle"])
def test_registered_face_rejects_uncovered_codepoint_before_measured_layout(
    kind: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import import_module

    assets = default_render_assets()
    font_id = "headline_impact" if kind == "title" else "subtitle_bold"
    # Derive an actually uncovered scalar from the registered face's real cmap;
    # do not assume a particular emoji or CJK character is absent on this host.
    with import_module("fontTools.ttLib").TTFont(str(assets.fonts[font_id]),
                                                fontNumber=0, lazy=True) as face:
        cmap = face.getBestCmap()
        assert cmap is not None
        codepoint = next(cp for cp in range(0x1F300, 0x1FB00) if cp not in cmap)
    text = "A" + chr(codepoint)
    plan = plan_with_two_subtitle_regions()
    if kind == "title":
        plan.shots[0].subtitles = None
        plan.shots[1].subtitles = None
        plan.shots[0].title = OverlayPlan(text=text, start=0, end=1,
            destination=NormalizedRect(x=.1, y=.1, width=.8, height=.1), z_index=2)
        transcript = None
    else:
        transcript = [TranscriptSegment(10, 11, text)]

    def unexpected_font_sizing(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unsupported text reached font sizing before coverage guard")

    monkeypatch.setattr("clipperx.overlay_ass._font", unexpected_font_sizing)
    with pytest.raises(ValueError, match=rf"Unsupported glyph U\+{codepoint:04X}"):
        compile_overlay_ass(plan, transcript, assets=assets)


def test_valid_spanish_registered_glyphs_and_host_ass_guards_remain_literal() -> None:
    plan = plan_with_two_subtitle_regions()
    text = "¿Sí? ¡ÁÉÍÓÚ ñ ü!"
    result = compile_overlay_ass(plan, [TranscriptSegment(10, 11, text)])
    assert result.events[0].text == text and text in result.document
    plan.shots[0].title = OverlayPlan(text=text, start=0, end=1,
        destination=NormalizedRect(x=.1, y=.1, width=.8, height=.1), z_index=2)
    assert next(e for e in compile_overlay_ass(plan, []).events if e.kind == "title").text == text
    literal = r"{\fs150} A\N \h"
    escaped = compile_overlay_ass(plan, [TranscriptSegment(10, 11, literal)])
    caption = next(e for e in escaped.events if e.kind == "subtitle")
    assert caption.text == literal and "\u2060" not in caption.text
    assert "\\\u2060N" in escaped.document and r"\{" in escaped.document


def test_uncovered_format_or_variation_character_is_not_whitelisted() -> None:
    from importlib import import_module

    path = default_render_assets().fonts["subtitle_bold"]
    with import_module("fontTools.ttLib").TTFont(str(path), fontNumber=0, lazy=True) as face:
        cmap = face.getBestCmap()
        assert cmap is not None
        unsupported = next(cp for cp in (0x200D, 0xFE0F, 0xE0100) if cp not in cmap)
    with pytest.raises(ValueError, match="Unsupported glyph"):
        compile_overlay_ass(plan_with_two_subtitle_regions(),
                            [TranscriptSegment(10, 11, "A" + chr(unsupported))])


@pytest.mark.parametrize("mode", ["no_cmap", "notdef_name", "glyph_zero"])
def test_unestablished_cmap_or_missing_glyph_is_explicit_error(
    mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import import_module

    font_class = import_module("fontTools.ttLib").TTFont
    if mode == "no_cmap":
        monkeypatch.setattr(font_class, "getBestCmap", lambda self: None)
    elif mode == "notdef_name":
        monkeypatch.setattr(font_class, "getBestCmap", lambda self: {ord("A"): ".notdef"})
    else:
        original = font_class.getGlyphID
        monkeypatch.setattr(font_class, "getGlyphID",
                            lambda self, name: 0 if name == "A" else original(self, name))
    with pytest.raises(ValueError, match="glyph coverage|Unsupported glyph"):
        compile_overlay_ass(plan_with_two_subtitle_regions(), [TranscriptSegment(10, 11, "A")])


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_font_reader_load_failure_becomes_explicit_coverage_error(
    error_type: type[Exception], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import import_module

    def unreadable_font(*args: object, **kwargs: object) -> None:
        raise error_type("synthetic font reader failure")

    monkeypatch.setattr(import_module("fontTools.ttLib"), "TTFont", unreadable_font)
    with pytest.raises(ValueError, match="glyph coverage.*synthetic font reader failure"):
        compile_overlay_ass(plan_with_two_subtitle_regions(), [TranscriptSegment(10, 11, "A")])


def exact_precision_plan() -> VisualCompositionPlan:
    plan = plan_with_two_subtitle_regions()
    first, second = shot((1, 1.15), (0, .15)), shot((1e-18, .2), (.15, .35), "b")
    second.subtitles = plan.shots[0].subtitles
    plan.shots = [first, second]
    return VisualCompositionPlan.model_validate(plan.model_dump())


def test_caption_packing_carries_exact_intervals_and_max_ends() -> None:
    plan = exact_precision_plan()
    sub = plan.shots[1].subtitles
    assert sub is not None
    sub.max_words_per_event = 2
    result = compile_overlay_ass(plan, [TranscriptSegment(.075, .15, "known", [
        Word(.075, .12500000000000003, "uno"), Word(.1, .125, "dos"),
        Word(.13, .15, "tres")])])
    offset = F(3, 20) - F(1, 10**18)
    assert [e.exact_interval for e in result.events] == [
        (offset + F(3, 40), offset + F(1, 10)),
        (offset + F(1, 10), offset + F(".12500000000000003")),
        (offset + F(13, 100), offset + F(3, 20))]
    assert [e.text for e in result.events] == ["uno dos", "uno dos", "tres"]
    for event in result.events:
        assert event.exact_interval is not None
        assert (event.start, event.end) == tuple(float(v) for v in event.exact_interval)
        assert event.word_count <= 2 and len(event.lines) <= sub.max_lines


@pytest.mark.parametrize("relation", ["gap", "overlap", "adjacent"])
def test_real_caption_producer_uses_exact_active_boundaries(relation: str) -> None:
    from clipperx.output_timeline import OutputTimeline
    from clipperx.overlay_ass import _caption_events

    plan = exact_precision_plan()
    sub = plan.shots[1].subtitles
    assert sub is not None
    sub.emphasize_active_word = True
    first_end = .10000000000000002 if relation == "overlap" else .1
    second_start = .10000000000000002 if relation == "gap" else .1
    mapped, = OutputTimeline.from_shots(plan.shots).remap([
        TranscriptSegment(.075, .125, "uno dos", [Word(.075, first_end, "uno"),
                                                   Word(second_start, .125, "dos")])])
    events = _caption_events(mapped, sub, path=default_render_assets().fonts["subtitle_bold"],
                             layer=1)
    offset = F(3, 20) - F(1, 10**18)
    a, b = offset + F(str(first_end)), offset + F(str(second_start))
    if relation == "adjacent":
        assert [e.exact_interval for e in events] == [(offset + F(3, 40), a),
                                                     (a, offset + F(1, 8))]
        assert [e.active_words for e in events] == [(0,), (1,)]
    else:
        assert float(a) == float(b)  # float boundary set would erase the middle slice
        assert [e.exact_interval for e in events] == [(offset + F(3, 40), min(a, b)),
            (min(a, b), max(a, b)), (max(a, b), offset + F(1, 8))]
        assert [e.active_words for e in events] == [(0,), () if relation == "gap" else (0, 1), (1,)]
    assert all(a.exact_interval is not None and b.exact_interval is not None
               and a.exact_interval[1] == b.exact_interval[0] for a, b in zip(events, events[1:]))


def test_sub_ulp_caption_collision_hidden_by_equal_float_endpoints_rejects() -> None:
    plan = exact_precision_plan()
    sub = plan.shots[1].subtitles
    assert sub is not None
    sub.max_words_per_event = 1
    offset = F(3, 20) - F(1, 10**18)
    assert offset + F(".10000000000000002") > offset + F(1, 10)
    assert float(offset + F(".10000000000000002")) == float(offset + F(1, 10))
    with pytest.raises(ValueError, match="Overlapping emitted subtitle events"):
        compile_overlay_ass(plan, [TranscriptSegment(.075, .125, "uno dos", [
            Word(.075, .10000000000000002, "uno"), Word(.1, .125, "dos")])])


def test_title_preserves_float_expression_and_adds_original_exact_sum() -> None:
    plan = exact_precision_plan()
    plan.shots[1].title = OverlayPlan(text="title", start=.075, end=.1,
        destination=NormalizedRect(x=.1, y=.1, width=.8, height=.1), z_index=2)
    title, = compile_overlay_ass(plan, []).events
    assert title.start == .15 + .075 == .22499999999999998
    assert title.end == .15 + .1
    assert title.exact_interval == (F(9, 40), F(1, 4))


def test_overlay_exact_metadata_detaches_validates_and_keeps_canonical_none() -> None:
    from dataclasses import replace
    from clipperx.overlay_ass import serialize_overlay_events

    event, = compile_overlay_ass(plan_with_two_subtitle_regions(), [
        TranscriptSegment(10, 11, "uno")]).events
    interval = [F(0), F(1)]
    detached = replace(event, exact_interval=interval)  # type: ignore[arg-type]
    interval.clear()
    assert detached.exact_interval == (F(0), F(1))
    canonical = replace(event, exact_interval=None)
    assert canonical.exact_interval is None
    for bad in [(0.0, 1.0), (False, F(1)), (F(1), F(0)), (F(-1), F(1)), (F(0),)]:
        with pytest.raises(ValueError):
            replace(event, exact_interval=bad)  # type: ignore[arg-type]
    for other in [replace(canonical, start=.5, end=1.5),
                  replace(event, start=.5, end=1.5, exact_interval=(F(1, 2), F(3, 2)))]:
        with pytest.raises(ValueError, match="Overlapping emitted subtitle events"):
            serialize_overlay_events([canonical, other])
    adjacent = replace(canonical, start=1, end=2)
    assert serialize_overlay_events([canonical, adjacent]).count("Dialogue:") == 2


def test_caption_producer_rejects_missing_exact_metadata_instead_of_rebuilding_floats() -> None:
    from clipperx.output_timeline import MappedEvent, MappedWord
    from clipperx.overlay_ass import _caption_events

    sub = plan_with_two_subtitle_regions().shots[0].subtitles
    assert sub is not None
    path = default_render_assets().fonts["subtitle_bold"]
    for mapped in (MappedEvent("a", 0, 1, "x"),
                   MappedEvent("a", 0, 1, "x", (MappedWord(0, 1, "x"),), (F(0), F(1)))):
        with pytest.raises(ValueError, match="exact"):
            _caption_events(mapped, sub, path=path, layer=1)
