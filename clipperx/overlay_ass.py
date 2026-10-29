"""Deterministic v2 subtitle/title ASS and brand assembly metadata.

Only known transcript text and validated shot overlays are compiled. No legacy
renderer, tools, model, media decoding, or credential/configuration imports.
"""

from bisect import bisect_right
from collections.abc import Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from importlib import import_module
from io import BytesIO
import math
from pathlib import Path
import re
from typing import Literal, Protocol, cast, TYPE_CHECKING, Iterator, TypedDict

from PIL import ImageFont

from .filtergraph import PixelRect, rect_to_pixels
from .output_timeline import MappedEvent, MappedWord, OutputTimeline, _exact_interval, join_words
from .render_assets import (
    BrandVisualLayer, RenderAssetRegistry, default_render_assets, place_brand_asset,
    prepare_brand_asset, publish_render_bytes, _read_asset,
)
from .schema import TranscriptSegment
from .visual_contracts import SubtitlePlan, VisualCompositionPlan

if TYPE_CHECKING:
    from ._retained_files import RetainedFiles


class _FileOptions(TypedDict, total=False):
    _files: "RetainedFiles"


@dataclass(frozen=True, slots=True)
class CaptionStyle:
    primary_color: str = "&H001AD5FF"
    outline_color: str = "&H00000000"
    bold: bool = True


def athan_adaptive_caption() -> CaptionStyle:
    return CaptionStyle()


def _clean_text(text: str, *, strip: bool = True) -> str:
    if not isinstance(text, str) or any(ord(c) < 32 and c not in "\t\r\n" for c in text):
        raise ValueError("Overlay text requires printable text or line breaks")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[^\S\n]+", " ", text)
    return normalized.strip() if strip else normalized


def escape_ass_text(text: str) -> str:
    """Escape literals before host tags, targeting the required libass renderer.

    Libass supports escaped literal braces. ASS has no general backslash escape;
    U+2060 WORD JOINER is invisible and prevents a following N/n/h or brace from
    becoming syntax, while retaining the actual backslash glyph. Newlines alone
    become host-generated hard breaks. Never interpolate this into a shell.
    """
    return (_clean_text(text, strip=False).replace("\\", "\\\u2060").replace("{", r"\{")
            .replace("}", r"\}").replace("\n", r"\N"))


@dataclass(frozen=True, slots=True)
class OverlayEvent:
    shot_id: str
    kind: Literal["subtitle", "title"]
    start: float
    end: float
    destination: PixelRect
    lines: tuple[str, ...]
    font_family: str
    font_size: int
    bold: bool
    outline_width: int
    primary_color: str
    outline_color: str
    layer: int
    measured_width: float
    measured_height: float
    word_count: int = 0
    active_words: tuple[int, ...] = ()
    # Offsets into the normalized complete text, retained for active-word tags.
    text: str = ""
    line_ranges: tuple[tuple[int, int], ...] = ()
    word_ranges: tuple[tuple[int, int], ...] = ()
    exact_interval: tuple[Fraction, Fraction] | None = None
    # Complete text/layout stay fixed; only this reached prefix is opaque.
    visible_end: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lines", tuple(self.lines))
        object.__setattr__(self, "active_words", tuple(self.active_words))
        for name in ("line_ranges", "word_ranges"):
            object.__setattr__(self, name, tuple(tuple(pair) for pair in getattr(self, name)))
        object.__setattr__(self, "exact_interval", _exact_interval(self.exact_interval))
        if (self.visible_end is not None
                and (type(self.visible_end) is not int or not 0 <= self.visible_end <= len(self.text))):
            raise ValueError("Overlay visible prefix requires an in-range integer offset")
        if self.visible_end is not None and self.kind != "subtitle":
            raise ValueError("Only a subtitle can declare a visible prefix")


@dataclass(frozen=True, slots=True)
class OverlayCompilation:
    document: str
    events: tuple[OverlayEvent, ...]
    font_paths: tuple[Path, ...]
    brands: tuple[BrandVisualLayer, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "font_paths", tuple(Path(p) for p in self.font_paths))
        object.__setattr__(self, "brands", tuple(self.brands))


def _color(value: str) -> str:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError("ASS color requires #RRGGBB")
    return "&H00" + value[5:7].upper() + value[3:5].upper() + value[1:3].upper()


class _FontFace(Protocol):
    """Small verified API boundary for FontTools, which ships no py.typed."""

    def getBestCmap(self) -> dict[int, str] | None: ...

    def getGlyphID(self, glyphName: str) -> int: ...


class _FontFaceLoader(Protocol):
    def __call__(self, file: str | BytesIO, *, fontNumber: int,
                 lazy: bool) -> AbstractContextManager[_FontFace]: ...


@contextmanager
def _font_input(path: Path, files: "RetainedFiles | None") -> Iterator[str | BytesIO]:
    if files is None:
        yield str(path)
    else:
        with BytesIO(_read_asset(path, files)) as memory:
            yield memory


def _validate_glyph_coverage(path: Path, text: str, *, _files: "RetainedFiles | None" = None) -> None:
    """Reject missing glyphs in the same face index measured by Pillow.

    The factory cast is a local interface to the documented real FontTools API,
    not a claim that its untyped implementation has been statically verified.
    No fallback, cache or general Unicode-format/variation exemption is applied.
    Only normalized line breaks are host ASS syntax, not font glyphs. The proven
    host backslash guard is inserted later by escaping, never accepted here as
    an exemption for an unsupported character in supplied text.
    """
    try:
        loader = cast(_FontFaceLoader, import_module("fontTools.ttLib").TTFont)
        with _font_input(path, _files) as font_input, loader(font_input, fontNumber=0, lazy=True) as face:
            cmap = face.getBestCmap()
            if cmap is None:
                raise ValueError("No Unicode cmap")
            for codepoint in sorted({ord(c) for c in text if c != "\n"}):
                glyph = cmap.get(codepoint)
                if not glyph or glyph == ".notdef" or face.getGlyphID(glyph) == 0:
                    raise ValueError(f"Unsupported glyph U+{codepoint:04X}")
    except Exception as exc:
        # Normalize failures of this external font-reading boundary, retaining
        # the cause. Context exit closes its reader even for invalid cmap data.
        raise ValueError(f"Cannot establish registered font glyph coverage: {path}: {exc}") from exc


def _font(path: Path, size: int, *, require_bold: bool = False, _files: "RetainedFiles | None" = None
          ) -> tuple[ImageFont.FreeTypeFont, str, bool]:
    try:
        with _font_input(path, _files) as font_input:
            font = ImageFont.truetype(font_input, size, index=0)
        family, style = font.getname()
    except OSError as exc:
        raise ValueError(f"Font asset is not loadable: {path}") from exc
    if not family or style is None or any(c in family for c in ",\\{}\r\n"):
        raise ValueError("Registered font family cannot be represented in ASS")
    if any(name in style.lower() for name in ("italic", "oblique")):
        raise ValueError("Registered italic/oblique face is unsupported by v2 upright text styles")
    bold = "bold" in style.lower()
    if require_bold and not bold:
        raise ValueError("Subtitle font requires a registered bold face for measured bold metrics")
    return font, family, bold


def _text_width(font: ImageFont.FreeTypeFont, text: str, outline: int) -> float:
    left, _, right, _ = font.getbbox(text, stroke_width=outline)
    return max(float(right - left), float(font.getlength(text)) + 2 * outline)


def _layout(text: str, font: ImageFont.FreeTypeFont, destination: PixelRect,
            outline: int, max_lines: int) -> tuple[tuple[tuple[int, int], ...], float, float] | None:
    """Greedy word wrapping with explicit hard breaks and conservative TTF bounds.

    Pillow measures the registered face at the requested em size including its
    outline. Libass's ASS cell size is no larger; retain the full ascent/descent
    per line as a conservative vertical bound, with no stretch or truncation.
    """
    ranges: list[tuple[int, int]] = []
    offset = 0
    for paragraph in text.split("\n"):
        matches = list(re.finditer(r"\S+", paragraph))
        if not matches:
            ranges.append((offset, offset))
        else:
            start, end = matches[0].start(), matches[0].end()
            if _text_width(font, paragraph[start:end], outline) > destination.width:
                return None
            for token in matches[1:]:
                if _text_width(font, paragraph[start:token.end()], outline) <= destination.width:
                    end = token.end()
                else:
                    ranges.append((offset + start, offset + end))
                    start, end = token.start(), token.end()
                    if _text_width(font, paragraph[start:end], outline) > destination.width:
                        return None
            ranges.append((offset + start, offset + end))
        offset += len(paragraph) + 1
    ascent, descent = font.getmetrics()
    height = len(ranges) * (ascent + descent + 2 * outline)
    if len(ranges) > max_lines or height > destination.height:
        return None
    width = max((_text_width(font, text[a:b], outline) for a, b in ranges), default=0)
    return tuple(ranges), width, float(height)


def _word_text(words: Sequence[MappedWord]) -> tuple[str, tuple[tuple[int, int], ...]]:
    text = ""
    ranges: list[tuple[int, int]] = []
    for word in words:
        token = _clean_text(word.text)
        joined = join_words([text, token])
        ranges.append((len(joined) - len(token), len(joined)))
        text = joined
    return text, tuple(ranges)


def _caption_events(mapped: MappedEvent, subtitle: SubtitlePlan, *, path: Path,
                     layer: int, caption_starts: Sequence[Fraction] = (),
                     _files: "RetainedFiles | None" = None) -> tuple[OverlayEvent, ...]:
    if mapped.exact_interval is None or any(w.exact_interval is None for w in mapped.words):
        raise ValueError("Subtitle production requires exact mapped intervals")
    options: _FileOptions = {} if _files is None else {"_files": _files}
    _validate_glyph_coverage(path, _clean_text(mapped.text), **options)
    destination = rect_to_pixels(subtitle.destination, width=1080, height=1920)
    font, family, bold = _font(path, subtitle.font_size, require_bold=True, **options)
    outline = subtitle.outline_width
    events: list[OverlayEvent] = []
    remaining = mapped.words
    while True:
        words: tuple[MappedWord, ...] = ()
        word_ranges: tuple[tuple[int, int], ...] = ()
        if remaining:
            if len(_clean_text(remaining[0].text).split()) > subtitle.max_words_per_event:
                raise ValueError("Subtitle token requires separate word timestamps for word limit")
            limit = min(len(remaining), subtitle.max_words_per_event)
            first_interval = remaining[0].exact_interval
            assert first_interval is not None
            covered_end = first_interval[1]
            for index, word in enumerate(remaining[1:limit], 1):
                interval = word.exact_interval
                assert interval is not None
                if interval[0] - covered_end > Fraction(7, 20):
                    limit = index
                    break
                covered_end = max(covered_end, interval[1])
            for count in range(limit, 0, -1):
                words = remaining[:count]
                if sum(len(_clean_text(w.text).split()) for w in words) > subtitle.max_words_per_event:
                    continue
                text, word_ranges = _word_text(words)
                layout = _layout(text, font, destination, outline, subtitle.max_lines)
                if layout is not None:
                    break
            else:
                raise ValueError("Subtitle word cannot fit planned destination/max_lines")
            intervals = tuple(w.exact_interval for w in words)
            first = words[0].exact_interval
            assert first is not None
            start, end = first[0], max(i[1] for i in intervals if i is not None)
            points = tuple(i[0] for i in intervals if i is not None and i[0] == i[1])
            if points:
                bound = (mapped.output_bounds or mapped.exact_interval)[1]
                following_index = bisect_right(caption_starts, max(points))
                if following_index < len(caption_starts):
                    # Only bound synthetic point dwell, never trim a known
                    # acoustic end to repair overlapping source captions.
                    bound = min(bound, caption_starts[following_index])
                if len(remaining) > len(words):
                    following = remaining[len(words)].exact_interval
                    assert following is not None
                    bound = min(bound, following[0])
                # Two frame periods survive ASS rounding with >=one60Hz frame.
                # This changes display only, never native acoustic word times.
                end = max(end, min(max(points) + Fraction(1, 30), bound))
            if start >= end:
                raise ValueError("Native word anchor has no positive display interval within its shot")
        else:
            text = _clean_text(mapped.text)
            if subtitle.emphasize_active_word:
                raise ValueError("Active-word subtitles require word timestamps")
            if len(text.split()) > subtitle.max_words_per_event:
                raise ValueError("Subtitle requires word timestamps to satisfy max_words_per_event")
            layout = _layout(text, font, destination, outline, subtitle.max_lines)
            if layout is None:
                raise ValueError("Subtitle requires word timestamps to fit destination/max_lines")
            start, end = mapped.exact_interval
        assert layout is not None
        line_ranges, width, height = layout
        event = OverlayEvent(mapped.shot_id, "subtitle", float(start), float(end), destination,
            tuple(text[a:b] for a, b in line_ranges), family, subtitle.font_size, bold,
            outline, _color(subtitle.primary_color), _color(subtitle.outline_color),
            layer, width, height, sum(len(_clean_text(w.text).split()) for w in words)
            if words else len(text.split()),
            text=text, line_ranges=line_ranges, word_ranges=word_ranges, exact_interval=(start, end))
        if words:
            boundaries = {start, end, *(i[0] for i in intervals if i is not None)}
            if subtitle.emphasize_active_word:
                boundaries.update(i[1] for i in intervals if i is not None)
            ordered = sorted(boundaries)
            for left, right in zip(ordered, ordered[1:]):
                active = tuple(i for i, interval in enumerate(intervals)
                               if subtitle.emphasize_active_word and interval is not None
                               and interval[0] < right and interval[1] > left)
                reached = max(b for (_, b), interval in zip(word_ranges, intervals)
                              if interval is not None and interval[0] <= left)
                events.append(replace(event, start=float(left), end=float(right), active_words=active,
                                      exact_interval=(left, right), visible_end=reached))
        else:
            events.append(event)
        if not remaining:
            break
        remaining = remaining[len(words):]
        if not remaining:
            break
    return tuple(events)


def _title_event(*, text: str, path: Path, destination: PixelRect, shot_id: str,
                  start: float, end: float, layer: int,
                  exact_interval: tuple[Fraction, Fraction] | None = None,
                  max_font_size: int = 46,
                  outline_width: int = 2,
                  _files: "RetainedFiles | None" = None) -> OverlayEvent:
    text = _clean_text(text)
    if not text:
        raise ValueError("Title text must not be empty")
    options: _FileOptions = {} if _files is None else {"_files": _files}
    _validate_glyph_coverage(path, text, **options)
    if type(max_font_size) is not int or not 1 <= max_font_size <= 128:
        raise ValueError("Title maximum font size must be an integer in1..128")
    if type(outline_width) is not int or not 0 <= outline_width <= 12:
        raise ValueError("Title outline width must be an integer in0..12")
    for size in range(max_font_size, 0, -1):
        font, family, bold = _font(path, size, **options)
        layout = _layout(text, font, destination, outline_width, destination.height)
        if layout is not None:
            ranges, width, height = layout
            return OverlayEvent(shot_id, "title", start, end, destination,
                tuple(text[a:b] for a, b in ranges), family, size, bold, outline_width,
                "&H00FFFFFF", "&H00000000", layer, width, height,
                text=text, line_ranges=ranges, exact_interval=exact_interval)
    raise ValueError("Title cannot fit planned destination without truncation")


def _timestamp(seconds: float) -> str:
    centiseconds = math.floor(seconds * 100 + .5)
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds_int, hundredths = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{seconds_int:02}.{hundredths:02}"


def _event_text(event: OverlayEvent) -> str:
    visible_end = len(event.text) if event.visible_end is None else event.visible_end
    if type(visible_end) is not int or not 0 <= visible_end <= len(event.text):
        raise ValueError("Overlay visible prefix requires an in-range integer offset")
    if event.visible_end is not None and event.kind != "subtitle":
        raise ValueError("Only a subtitle can declare a visible prefix")
    rendered: list[str] = []
    for left, right in event.line_ranges:
        # Add host color tags only AFTER escaping each untrusted text slice.
        cuts = {left, right}
        cuts.add(max(left, min(right, visible_end)))
        for index in event.active_words:
            a, b = event.word_ranges[index]
            cuts.update((max(left, min(right, a)), max(left, min(right, b))))
        boundaries = sorted(cuts)
        pieces: list[str] = []
        for a, b in zip(boundaries, boundaries[1:]):
            active = any(event.word_ranges[i][0] <= a < event.word_ranges[i][1]
                         for i in event.active_words)
            literal = escape_ass_text(event.text[a:b])
            if a >= visible_end:
                literal = r"{\alpha&HFF&}" + literal
            elif active:
                literal = r"{\1c&H00FFFFFF&}" + literal + r"{\1c" + event.primary_color + "&}"
            pieces.append(literal)
        rendered.append("".join(pieces))
    return r"\N".join(rendered)


def _validate_caption_collisions(events: Sequence[OverlayEvent]) -> None:
    """Reject distinct emitted captions sharing time and planned pixel area.

    This checks packed output events, not source-word overlap. Simultaneous
    words in one coherent caption (including active-word slices) remain valid.
    Half-open adjacent intervals and disjoint planned destinations do not collide.
    Do not repair a collision by changing known timing, text, limits or geometry.
    """
    def interval(event: OverlayEvent) -> tuple[Fraction, Fraction] | tuple[float, float]:
        return event.exact_interval if event.exact_interval is not None else (event.start, event.end)

    active: list[OverlayEvent] = []
    for current in sorted((e for e in events if e.kind == "subtitle"), key=lambda e: interval(e)[0]):
        active = [previous for previous in active if interval(previous)[1] > interval(current)[0]]
        p = current.destination
        for previous in active:
            q = previous.destination
            if (p.x < q.x + q.width and q.x < p.x + p.width
                    and p.y < q.y + q.height and q.y < p.y + p.height):
                raise ValueError(
                    "Overlapping emitted subtitle events cannot be represented coherently "
                    "within planned limits/region without changing known timing/text"
                )
        active.append(current)


def serialize_overlay_events(events: Sequence[OverlayEvent]) -> str:
    """Pure ordered ASS view of immutable events, with exactly one escaping pass.

    The canonical clip document uses every event. Task6 may project an ordered
    subset around intervening PNG/source z-layers using this same serializer;
    it must not interpolate raw event text into a separate serializer or filter.
    Distinct captions that overlap in output time and planned destination are
    rejected here for both canonical output and host-requested projections.
    """
    events = tuple(events)
    _validate_caption_collisions(events)
    # Host-authored RGB colors follow the target range, not an implicit historical TV matrix.
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "YCbCr Matrix: None\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    )
    styles: list[str] = []
    dialogue: list[str] = []
    for index, event in enumerate(events):
        start, end = _timestamp(event.start), _timestamp(event.end)
        if start == end:
            raise ValueError("Overlay interval cannot be represented at ASS centisecond precision")
        style = f"e{index}"
        styles.append(f"Style: {style},{event.font_family},{event.font_size},"
            f"{event.primary_color},{event.primary_color},{event.outline_color},&H00000000,"
            f"{-1 if event.bold else 0},0,0,0,100,100,0,0,1,{event.outline_width},0,5,0,0,0,1")
        p = event.destination
        x, y = p.x + p.width / 2, p.y + p.height / 2
        tags = rf"{{\an5\pos({x:g},{y:g})\q2}}"
        dialogue.append(f"Dialogue: {event.layer},{start},{end},{style},,0,0,0,,"
                        + tags + _event_text(event))
    return (header + "\n".join(styles) + "\n\n[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            + "\n".join(dialogue) + "\n")


def compile_overlay_ass(plan: VisualCompositionPlan,
                        transcript: Sequence[TranscriptSegment] | None, *,
                        assets: RenderAssetRegistry | None = None,
                        brand_cache_dir: Path | None = None,
                        _files: "RetainedFiles | None" = None) -> OverlayCompilation:
    """Compile one clip; the caller has already bound shot title to overlay_text.

    Brands need an explicit host cache directory. Returned timing is global output
    time; every nested shot overlay time is shifted by that shot's output start.
    Materialization of the ASS is separate. No external renderer is invoked.
    """
    plan = VisualCompositionPlan.model_validate(plan.model_dump())
    timeline = OutputTimeline.from_shots(plan.shots)
    registry = assets if assets is not None else default_render_assets()
    subtitle_shots = {s.shot_id: s for s in plan.shots if s.subtitles is not None}
    mapped = timeline.remap(transcript) if subtitle_shots else ()
    events: list[OverlayEvent] = []
    fonts: set[Path] = set()
    brands: list[BrandVisualLayer] = []
    options: _FileOptions = {} if _files is None else {"_files": _files}
    for shot in plan.shots:
        if shot.subtitles is not None:
            path = registry.resolve("font", shot.subtitles.font_asset_id)
            shot_events = tuple(event for event in mapped if event.shot_id == shot.shot_id)
            caption_starts = []
            for event in shot_events:
                # Include later word starts too: wrapping/pauses can begin a
                # new display run inside an earlier, overlapping envelope.
                intervals = tuple(w.exact_interval for w in event.words) if event.words else (event.exact_interval,)
                for interval in intervals:
                    if interval is None:
                        raise ValueError("Subtitle production requires exact mapped intervals")
                    caption_starts.append(interval[0])
            caption_starts.sort()
            if not shot_events:
                _font(path, shot.subtitles.font_size, require_bold=True, **options)
            fonts.add(path)
            layer = max([layer.z_index for layer in shot.layers]
                        + [o.z_index for o in (shot.title, shot.brand) if o is not None]) + 1
            for event in shot_events:
                events.extend(_caption_events(event, shot.subtitles, path=path, layer=layer,
                                              caption_starts=caption_starts, **options))
        if shot.title is not None:
            title = shot.title
            if title.text is None:
                raise ValueError("Title overlay requires validated headline.overlay_text")
            path = registry.resolve("font", title.font_asset_id or "headline_impact")
            fonts.add(path)
            events.append(_title_event(text=title.text, path=path,
                destination=rect_to_pixels(title.destination, width=1080, height=1920),
                shot_id=shot.shot_id, start=shot.output.start + title.start,
                end=shot.output.start + title.end, layer=title.z_index,
                exact_interval=timeline.overlay_output_interval(shot.shot_id, title.start, title.end), **options))
        if shot.brand is not None:
            brand = shot.brand
            if brand.asset_id is None:
                raise ValueError("Brand requires a registered image asset_id")
            if brand_cache_dir is None:
                raise ValueError("Planned brand requires a host-provided brand_cache_dir")
            prepared = prepare_brand_asset(registry, brand.asset_id, cache_dir=brand_cache_dir, **options)
            brands.append(place_brand_asset(prepared,
                destination=rect_to_pixels(brand.destination, width=1080, height=1920),
                start=shot.output.start + brand.start, end=shot.output.start + brand.end,
                z_index=brand.z_index, shot_id=shot.shot_id))
    return OverlayCompilation(serialize_overlay_events(events), tuple(events),
                              tuple(sorted(fonts)), tuple(brands))


def build_overlay_ass(plan: VisualCompositionPlan,
                      transcript: Sequence[TranscriptSegment] | None, *,
                      assets: RenderAssetRegistry | None = None,
                      brand_cache_dir: Path | None = None, _files: "RetainedFiles | None" = None) -> str:
    return compile_overlay_ass(plan, transcript, assets=assets,
                               brand_cache_dir=brand_cache_dir, **({} if _files is None else {"_files": _files})).document


def write_overlay_ass(document: str, *, output_dir: Path, _files: "RetainedFiles | None" = None) -> Path:
    return publish_render_bytes(document.encode("utf-8-sig"), output_dir=output_dir, suffix=".ass",
        **({} if _files is None else {"_files": _files}))
