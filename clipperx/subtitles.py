"""Subtitle generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from .caption_templates import CaptionTemplate, get_template
from .schema import TranscriptSegment, ClipCandidate, Word


def _build_ass_header(template: CaptionTemplate) -> str:
    """Generate an ASS header dynamically from a CaptionTemplate."""
    bold_flag = -1 if template.bold else 0
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: ClipperX,{template.font_name},{template.font_size},"
        f"{template.primary_color},{template.highlight_color},"
        f"{template.outline_color},{template.back_color},"
        f"{bold_flag},0,0,0,100,100,0,0,1,"
        f"{template.outline_width},{template.shadow_depth},"
        f"2,{template.margin_l},{template.margin_r},{template.margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


# Keep the legacy header as a constant for backwards compatibility
ASS_HEADER = _build_ass_header(get_template("default"))


def clip_to_ass(
    clip: ClipCandidate,
    segments: Iterable[TranscriptSegment],
    output_path: Path,
    *,
    margin_v: int | None = None,
    margins_v: Sequence[int] | None = None,
    template: str = "default",
) -> Path:
    tpl = get_template(template)
    effective_margin = tpl.margin_v if margin_v is None else margin_v
    header = _build_ass_header(tpl)

    events: list[str] = []
    segments = tuple(segments)
    ranges = _clip_ranges(clip)
    if margins_v is not None and (len(margins_v) != len(ranges)
            or any(type(m) is not int or not 0 <= m <= 1920 for m in margins_v)):
        raise ValueError("One valid caption margin per output range is required")
    offset = 0.0
    for range_index, (range_start, range_end) in enumerate(ranges):
        for segment in segments:
            if segment.start >= range_end or segment.end <= range_start:
                continue
            events.extend(
                _segment_to_events(
                    segment,
                    range_start=range_start,
                    range_end=range_end,
                    offset=offset,
                    margin_v=effective_margin if margins_v is None else margins_v[range_index],
                    template=tpl,
                )
            )
        offset += range_end - range_start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + "\n" + "\n".join(events), encoding="utf-8")
    return output_path


def clip_to_srt(
    clip: ClipCandidate,
    segments: Iterable[TranscriptSegment],
    output_path: Path,
) -> Path:
    """Minimal SRT export for tests/CLI. Uses segment-level timing within clip."""
    lines: list[str] = []
    index = 1
    ranges = _clip_ranges(clip)
    offset = 0.0
    for range_start, range_end in ranges:
        for segment in segments:
            if segment.start >= range_end or segment.end <= range_start:
                continue
            overlap_start = max(segment.start, range_start)
            overlap_end = min(segment.end, range_end)
            if overlap_end <= overlap_start:
                continue
            start = offset + (overlap_start - range_start)
            end = offset + (overlap_end - range_start)
            lines.append(str(index))
            lines.append(f"{_format_timestamp(start)} --> {_format_timestamp(end)}")
            lines.append(segment.text.strip())
            lines.append("")
            index += 1
        offset += range_end - range_start

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _clip_ranges(clip: ClipCandidate) -> list[tuple[float, float]]:
    if clip.time_ranges:
        return clip.time_ranges
    return [(clip.start, clip.end)]


def _format_timestamp(seconds: float, *, ass: bool = False) -> str:
    resolution = 100 if ass else 1000
    whole, fraction = divmod(round(seconds * resolution), resolution)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if ass:
        return f"{hours:01}:{minutes:02}:{secs:02}.{fraction:02}"
    return f"{hours:02}:{minutes:02}:{secs:02},{fraction:03}"


def _segment_to_events(
    segment: TranscriptSegment,
    *,
    range_start: float,
    range_end: float,
    offset: float,
    margin_v: int,
    template: CaptionTemplate | None = None,
) -> List[str]:
    events: List[str] = []
    words = segment.words
    if not words:
        text = segment.text.strip()
        if not text:
            return events
        overlap_start = max(segment.start, range_start)
        overlap_end = min(segment.end, range_end)
        if overlap_end <= overlap_start:
            return events
        events.append(
            _build_event(
                text,
                offset + (overlap_start - range_start),
                offset + (overlap_end - range_start),
                margin_v,
                template=template,
            )
        )
        return events

    bucket: List[Word] = []
    for word in words:
        if word.end <= range_start or word.start >= range_end:
            continue
        if bucket and (word.start-bucket[-1].end > .35 or word.end-bucket[0].start > 2.5):
            events.extend(_bucket_events(bucket,range_start,range_end,offset,margin_v,template))
            bucket=[]
        bucket.append(word)
        bucket_text = _join_words(bucket)
        if len(bucket_text) >= 28 or len(bucket) >= 5 or word.word.strip().endswith((".", "?", "!", ";", ":")):
            events.extend(_bucket_events(bucket,range_start,range_end,offset,margin_v,template))
            bucket = []

    if bucket:
        events.extend(_bucket_events(bucket,range_start,range_end,offset,margin_v,template))

    return events


def _bucket_events(words: List[Word], range_start: float, range_end: float,
        offset: float, margin_v: int, template: CaptionTemplate | None) -> List[str]:
    text=_join_words(words)
    if template is None or template.animation != "reveal":
        return [_build_event(text,offset+max(words[0].start-range_start,0),
            offset+min(words[-1].end,range_end)-range_start,margin_v,
            template=template,words=words,range_start=range_start,offset=offset)]
    events=[]
    for index,word in enumerate(words):
        start=max(word.start,range_start)
        end=min(words[index+1].start if index+1 < len(words) else word.end,range_end)
        if word.end == word.start and end <= start:
            # TDT may emit a zero-duration lexical token. Two frame periods
            # survive ASS centisecond rounding and include a 60 Hz frame;
            # native acoustic timestamps remain unchanged.
            end=min(start+2/60,range_end)
        if end <= start:
            continue
        visible=_join_words(words[:index+1])
        hidden=text[len(visible):]
        contents=_escape_ass_text(visible)
        if hidden:
            contents+=r"{\alpha&HFF&}"+_escape_ass_text(hidden)
        events.append(f"Dialogue: 0,{_format_timestamp(offset+start-range_start,ass=True)},"
            f"{_format_timestamp(offset+end-range_start,ass=True)},ClipperX,,0000,0000,{margin_v},,{contents}")
    return events


def _build_event(
    text: str,
    start: float,
    end: float,
    margin_v: int,
    *,
    template: CaptionTemplate | None = None,
    words: List[Word] | None = None,
    range_start: float = 0.0,
    offset: float = 0.0,
) -> str:
    text = _escape_ass_text(text.replace("\n", " "))
    start_ts = _format_timestamp(start, ass=True)
    end_ts = _format_timestamp(end, ass=True)

    animation = template.animation if template else "none"

    if animation == "karaoke" and words and template:
        # Build karaoke text with {\kf<duration>} per word for progressive highlight
        parts: list[str] = []
        rendered_text = ""
        for w in words:
            word_dur_cs = max(1, int((w.end - w.start) * 100))  # centiseconds
            token = w.word.strip()
            prefix = _word_prefix(rendered_text, token)
            rendered_text += prefix + token
            parts.append(f"{{\\kf{word_dur_cs}}}{_escape_ass_text(prefix + token)}")
        animated_text = "".join(parts).strip()
        return (
            f"Dialogue: 0,{start_ts},{end_ts},ClipperX,,0000,0000,"
            f"{margin_v},,{animated_text}"
        )

    elif animation == "pop" and template:
        # Scale up effect on the entire group
        animated_text = (
            f"{{\\fscx120\\fscy120\\t(0,100,\\fscx100\\fscy100)}}{text}"
        )
        return (
            f"Dialogue: 0,{start_ts},{end_ts},ClipperX,,0000,0000,"
            f"{margin_v},,{animated_text}"
        )

    elif animation == "fade" and template:
        # Fade in/out per group
        animated_text = f"{{\\fad(200,200)}}{text}"
        return (
            f"Dialogue: 0,{start_ts},{end_ts},ClipperX,,0000,0000,"
            f"{margin_v},,{animated_text}"
        )

    # "none" or fallback
    return f"Dialogue: 0,{start_ts},{end_ts},ClipperX,,0000,0000,{margin_v},,{text}"


def _escape_ass_text(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _join_words(words: Iterable[Word]) -> str:
    text = ""
    for word in words:
        token = word.word.strip()
        if not token:
            continue
        prefix = _word_prefix(text, token)
        text += prefix + token
    return text


def _word_prefix(existing: str, token: str) -> str:
    if not existing or not token:
        return ""
    if token[0] in ".,!?;:%)]}»" or existing[-1] in "¿¡([{«":
        return ""
    return " "
