from pathlib import Path

import pytest

from clipperx.schema import ClipCandidate, TranscriptSegment, Word
from clipperx.subtitles import clip_to_ass, clip_to_srt


def test_default_caption_style_reserves_right_ui_space() -> None:
    from clipperx.caption_templates import get_template
    template = get_template("default")
    assert 68 <= template.font_size <= 78
    assert template.margin_l >= 210 and template.margin_r >= 210


def test_clip_to_srt(tmp_path: Path) -> None:
    clip = ClipCandidate(start=5.0, end=15.0, text="hola mundo", coherence=0.8)
    segments = [
        TranscriptSegment(start=0.0, end=6.0, text="intro"),
        TranscriptSegment(start=6.0, end=12.0, text="hola mundo"),
        TranscriptSegment(start=12.0, end=18.0, text="despedida"),
    ]

    out = tmp_path / "clip.srt"
    clip_to_srt(clip, segments, out)

    content = out.read_text(encoding="utf-8")
    assert "hola mundo" in content
    assert "00:00:01,000" in content


def test_clip_to_ass_restores_spaces_between_words(tmp_path: Path) -> None:
    clip = ClipCandidate(start=0, end=2, text="", coherence=1.0)
    segments = [
        TranscriptSegment(
            start=0,
            end=2,
            text="¿Cómo se metió?",
            words=[
                Word(start=0, end=0.2, word="¿"),
                Word(start=0.2, end=0.6, word="Cómo"),
                Word(start=0.6, end=0.9, word="se"),
                Word(start=0.9, end=1.4, word="metió"),
                Word(start=1.4, end=1.5, word="?"),
            ],
        )
    ]

    output = tmp_path / "clip.ass"
    clip_to_ass(clip, segments, output)

    assert "¿Cómo se metió?" in output.read_text(encoding="utf-8")


def test_ass_placement_follows_each_output_range(tmp_path: Path) -> None:
    clip = ClipCandidate(start=1, end=4, text="", coherence=1, time_ranges=[(1, 2), (3, 4)])
    rows = [TranscriptSegment(start=1, end=2, text="Primero"),
            TranscriptSegment(start=3, end=4, text="Después")]
    out = tmp_path / "moving.ass"
    clip_to_ass(clip, rows, out, margins_v=[876, 764])
    events = [line for line in out.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue:")]
    assert ",876,,Primero" in events[0]
    assert ",764,,Después" in events[1]
    assert ",0:00:01.00,0:00:02.00," in events[1]
    with pytest.raises(ValueError, match="margin"):
        clip_to_ass(clip, rows, tmp_path / "invalid.ass", margins_v=[1])


def test_ass_reuses_generator_transcript_for_later_ranges(tmp_path: Path) -> None:
    clip = ClipCandidate(start=1, end=4, text="", coherence=1, time_ranges=[(1, 2), (3, 4)])
    rows = (r for r in [TranscriptSegment(start=1, end=2, text="Primero"),
                       TranscriptSegment(start=3, end=4, text="Después")])
    out = tmp_path / "generator.ass"
    clip_to_ass(clip, rows, out, margins_v=[876, 764])
    assert ",764,,Después" in out.read_text(encoding="utf-8")


def test_default_captions_do_not_span_a_multi_second_pause(tmp_path: Path) -> None:
    clip=ClipCandidate(0,4,"",1)
    transcript=[TranscriptSegment(0,4,"¿Ese quién es?",words=[Word(.1,.4,"¿Ese"),
        Word(3.,3.2,"quién"),Word(3.2,3.4,"es?")])]
    out=tmp_path/"pauses.ass"
    clip_to_ass(clip,transcript,out)
    events=[s for s in out.read_text(encoding="utf-8").splitlines() if s.startswith("Dialogue:")]
    assert events[0].split(",")[2] == "0:00:00.40"
    assert all(not (s.split(",")[1] < "0:00:02.00" < s.split(",")[2]) for s in events)


def test_default_caption_reveals_future_words_only_at_their_native_start(tmp_path: Path) -> None:
    out=tmp_path/"reveal.ass"
    clip_to_ass(ClipCandidate(0,2,"",1),[TranscriptSegment(0,2,"Hola parcero",words=[
        Word(.1,.4,"Hola"),Word(.6,.9,"parcero")])],out)
    events=[s for s in out.read_text(encoding="utf-8").splitlines() if s.startswith("Dialogue:")]
    assert len(events)==2
    assert events[0].endswith(r"Hola{\alpha&HFF&} parcero")
    assert events[1].split(",")[1] == "0:00:00.60"
    assert events[1].endswith("Hola parcero")


def test_srt_millisecond_timestamp_does_not_use_ass_centisecond_carry() -> None:
    from clipperx.subtitles import _format_timestamp
    assert _format_timestamp(.999) == "00:00:00,999"
    assert _format_timestamp(59.999) == "00:00:59,999"
    assert _format_timestamp(59.999, ass=True) == "0:01:00.00"


def test_zero_duration_native_final_word_is_not_silently_lost(tmp_path: Path) -> None:
    out=tmp_path/"zero-word.ass"
    clip_to_ass(ClipCandidate(0,1,"",1),[TranscriptSegment(0,1,"Sí parce.",words=[
        Word(.1,.3,"Sí"),Word(.3,.3,"parce.")])],out)
    rows=[s for s in out.read_text(encoding="utf-8").splitlines() if s.startswith("Dialogue:")]
    assert rows[-1].endswith("Sí parce.")
    assert rows[-1].split(",")[1] < rows[-1].split(",")[2]


def test_zero_duration_caption_interval_contains_an_actual_60fps_frame(tmp_path: Path) -> None:
    out=tmp_path/"between-frames.ass"
    clip_to_ass(ClipCandidate(.064,1,"",1),[
        TranscriptSegment(.08,.081,"sí",words=[Word(.08,.08,"sí")])],out)
    row=next(s for s in out.read_text(encoding="utf-8").splitlines() if s.startswith("Dialogue:"))
    start,end=(float(value.rsplit(":",1)[1]) for value in row.split(",")[1:3])
    assert any(start <= frame/60 < end for frame in range(60))

