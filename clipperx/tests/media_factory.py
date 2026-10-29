"""Finite recipes only; importing this module never creates media or loads fonts."""

from fractions import Fraction as F
from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any
import zlib

FLOAT48_SAMPLES = 96000
FLOAT44_SAMPLES = 132300
EXPECTED_PULSES = (F(1, 2), F(2), F(3), F(5), F(13, 2))
SOURCE_PULSES = (F(1), F(5, 2), F(11, 2), F(17, 2), F(10))
DIAGNOSTIC_K = (0, 119, 120, 121, 149, 150, 269, 270, 300, 479)
RECIPE_VERSION = "native57-v1-phase-envelope-y010-opaque-q4-aac-editlist-preparation-tiny-probe-isolation-v3-decimal-source"


def impact_wav() -> bytes:
    """Fixed cue PCM with an explicit IEEE-float front-left/front-right layout."""
    cue_pcm = b"".join(struct.pack("<2f", value, -value) for n in range(4800)
        for value in [math.sin(2*math.pi*3000*n/48000)*(.5-.5*math.cos(2*math.pi*n/4799))])
    fmt = (struct.pack("<HHIIHHHHI", 0xfffe, 2, 48000, 48000*8, 8, 32, 22, 32, 3)
           +bytes.fromhex("0300000000001000800000aa00389b71"))
    wav = (b"WAVEfmt "+struct.pack("<I", 40)+fmt+b"fact"+struct.pack("<II", 4, 4800)
           +b"data"+struct.pack("<I", len(cue_pcm))+cue_pcm)
    return b"RIFF"+struct.pack("<I", len(wav))+wav


def prepare_aac_fixture_mov(raw: bytes) -> tuple[bytes, dict[str, Any]]:
    """Shift only the finite generated three-second v0 MOV, preserving AAC trim."""
    cap = 16*1024*1024
    if not 0 < len(raw) <= cap-24:
        raise ValueError("AAC fixture input/output cap")
    containers = {b"moov", b"trak", b"mdia", b"edts", b"minf", b"stbl", b"dinf"}
    fragments = {b"moof", b"traf", b"mvex", b"mfra", b"tfra", b"mfhd", b"tfhd", b"trun", b"trex"}
    children: dict[int, list[tuple[int, int, bytes]]] = {}

    def parse(start: int, end: int, depth: int) -> list[tuple[int, int, bytes]]:
        if depth > 8:
            raise ValueError("AAC fixture container depth")
        rows: list[tuple[int, int, bytes]] = []
        while start < end:
            if len(rows) >= 128 or end-start < 8:
                raise ValueError("AAC fixture atom count/truncation")
            size, tag = struct.unpack_from(">I4s", raw, start)
            if size < 8 or size > end-start or tag in fragments:
                raise ValueError("AAC fixture unsupported atom size/fragment")
            row = (start, start+size, tag)
            rows.append(row)
            if tag in containers:
                children[start] = parse(start+8, start+size, depth+1)
            start += size
        return rows

    def one(rows: list[tuple[int, int, bytes]], tag: bytes) -> tuple[int, int, bytes]:
        found = [row for row in rows if row[2] == tag]
        if len(found) != 1:
            raise ValueError("AAC fixture missing/duplicate "+tag.decode("ascii"))
        return found[0]

    def header(row: tuple[int, int, bytes], minimum: int) -> int:
        start, end, _ = row
        if end-start < minimum or raw[start+8] != 0:
            raise ValueError("AAC fixture v0 header bounds/version")
        return start

    top = parse(0, len(raw), 0)
    parsed = top+[row for rows in children.values() for row in rows]
    for tag, count in ((b"moov", 1), (b"mdat", 1), (b"mvhd", 1), (b"trak", 2),
                       (b"tkhd", 2), (b"mdia", 2), (b"mdhd", 2),
                       (b"edts", 2), (b"elst", 2)):
        if sum(row[2] == tag for row in parsed) != count:
            raise ValueError("AAC fixture required atom multiplicity")
    moov = one(top, b"moov")
    mdat = one(top, b"mdat")
    if top[-1] != moov or mdat[1] > moov[0]:
        raise ValueError("AAC fixture requires mdat before terminal moov")
    movie = children[moov[0]]
    mvhd = header(one(movie, b"mvhd"), 108)
    scale, duration = struct.unpack_from(">II", raw, mvhd+20)
    delta = scale*5//4
    if (scale == 0 or scale % 4 or duration != 3*scale
            or duration+delta > 0xffffffff):
        raise ValueError("AAC fixture integral movie timescale/duration/overflow")
    tracks = [row for row in movie if row[2] == b"trak"]
    if len(tracks) != 2:
        raise ValueError("AAC fixture exactly two tracks")
    replacements: dict[int, bytes] = {}
    patches: dict[int, bytes] = {mvhd+24: struct.pack(">I", duration+delta)}
    facts: list[dict[str, Any]] = []
    accepted_handlers: set[int] = set()
    for track in tracks:
        rows = children[track[0]]
        tkhd = header(one(rows, b"tkhd"), 92)
        if struct.unpack_from(">I", raw, tkhd+28)[0] != duration:
            raise ValueError("AAC fixture track presentation duration")
        patches[tkhd+28] = struct.pack(">I", duration+delta)
        mdia = one(rows, b"mdia")
        media = children[mdia[0]]
        mdhd = header(one(media, b"mdhd"), 32)
        media_scale, media_duration = struct.unpack_from(">II", raw, mdhd+20)
        hdlr = header(one(media, b"hdlr"), 32)
        handler = raw[hdlr+16:hdlr+20]
        if handler not in (b"vide", b"soun") or raw[hdlr+12:hdlr+16] not in (bytes(4), b"mhlr"):
            raise ValueError("AAC fixture unsupported handler")
        accepted_handlers.add(hdlr)
        minfs = [row for row in media if row[2] == b"minf"]
        if len(minfs) > 1:
            raise ValueError("AAC fixture duplicate media information")
        for minf in minfs:
            data_handlers = [row for row in children[minf[0]] if row[2] == b"hdlr"]
            if len(data_handlers) > 1:
                raise ValueError("AAC fixture duplicate data handler")
            for data_row in data_handlers:
                data_hdlr = header(data_row, 32)
                if raw[data_hdlr+12:data_hdlr+20] != b"dhlrurl ":
                    raise ValueError("AAC fixture unsupported data handler")
                accepted_handlers.add(data_hdlr)
        trim = 1024 if handler == b"soun" else 0
        if (media_scale == 0 or (handler == b"soun" and media_scale != 44100)
                or media_duration < 3*media_scale+trim):
            raise ValueError("AAC fixture media timescale/duration")
        edts = one(rows, b"edts")
        elst_row = one(children[edts[0]], b"elst")
        elst = header(elst_row, 28)
        count = struct.unpack_from(">I", raw, elst+12)[0]
        edit_duration, media_time, rate = struct.unpack_from(">IiI", raw, elst+16)
        if (elst_row[1]-elst != 28 or count != 1 or edit_duration != duration
                or media_time != trim or rate != 0x10000):
            raise ValueError("AAC fixture original single unit-rate priming edit")
        original = raw[elst+16:elst+28]
        empty = struct.pack(">IiI", delta, -1, 0x10000)
        replacements[elst] = (struct.pack(">I4s", 40, b"elst")+raw[elst+8:elst+12]
                              +struct.pack(">I", 2)+empty+original)
        facts.append(dict(handler=handler.decode("ascii"), original_media_time=media_time,
            original_duration=edit_duration, original_entry_hex=original.hex(),
            added_entry_hex=empty.hex(), media_timescale=media_scale,
            media_duration=media_duration))
    if {row["handler"] for row in facts} != {"vide", "soun"}:
        raise ValueError("AAC fixture distinct video/audio handlers")
    if {row[0] for row in parsed if row[2] == b"hdlr"} != accepted_handlers:
        raise ValueError("AAC fixture unsupported handler location")

    def rebuild(row: tuple[int, int, bytes]) -> bytes:
        start, end, tag = row
        if start in replacements:
            return replacements[start]
        if start in children:
            payload = b"".join(rebuild(child) for child in children[start])
            return struct.pack(">I4s", len(payload)+8, tag)+payload
        result = bytearray(raw[start:end])
        for offset in (start+24, start+28):
            if offset in patches:
                result[offset-start:offset-start+4] = patches[offset]
        return bytes(result)

    output = b"".join(rebuild(row) for row in top)
    prefix = raw[:moov[0]]
    media_bytes = raw[mdat[0]:mdat[1]]
    output_prefix = output[:moov[0]]
    output_media = output[mdat[0]:mdat[1]]
    def sha(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()
    if (len(output) != len(raw)+24 or len(output) > cap or prefix != output_prefix
            or media_bytes != output_media or sha(prefix) != sha(output_prefix)
            or sha(media_bytes) != sha(output_media)):
        raise ValueError("AAC fixture byte preservation/growth")
    return output, dict(input_bytes=len(raw), output_bytes=len(output),
        input_sha256=sha(raw), output_sha256=sha(output), growth_bytes=24,
        movie_timescale=scale, offset_movie_units=delta, offset_seconds="5/4",
        prefix_sha256=sha(prefix), output_prefix_sha256=sha(output_prefix),
        mdat_sha256=sha(media_bytes), output_mdat_sha256=sha(output_media), tracks=facts)


@dataclass(frozen=True, slots=True)
class OverlayBoundaryBatch:
    name: str
    region: tuple[int, int, int, int]
    first_active: int
    last_active: int  # exclusive authored global K
    frames: tuple[int, ...]

    @property
    def active(self) -> tuple[bool, ...]:
        return tuple(self.first_active <= k < self.last_active for k in self.frames)


def overlay_boundary_batches() -> tuple[OverlayBoundaryBatch, ...]:
    """Five six-cell crops, independent of compiled event boxes or mapper output."""
    return (
        OverlayBoundaryBatch("subtitle_s1", (108, 1574, 864, 192), 12, 72, (11, 12, 13, 71, 72, 73)),
        OverlayBoundaryBatch("subtitle_s3", (108, 308, 864, 192), 282, 342, (281, 282, 283, 341, 342, 343)),
        OverlayBoundaryBatch("brand_s1", (842, 1382, 130, 154), 30, 120, (29, 30, 31, 119, 120, 121)),
        OverlayBoundaryBatch("brand_s2", (842, 1382, 130, 154), 150, 270, (149, 150, 151, 269, 270, 271)),
        OverlayBoundaryBatch("brand_s3", (842, 1382, 130, 154), 285, 465, (284, 285, 286, 464, 465, 466)))


def float48_pcm(channels: int) -> bytes:
    if channels not in (1, 2):
        raise ValueError("mono or stereo fixture")
    words = [0, 0x80000000, 0x3FA00000, 0xBFC00000, 0x35800000, 0xB5800000, 1]
    words += [struct.unpack("<I", struct.pack("<f", (i-125)/256))[0] for i in range(250)]
    result = bytearray()
    for n in range(FLOAT48_SAMPLES):
        for ch in range(channels):
            word = words[(n+37*ch) % 257] ^ (0x80000000 if ch and n % 2 else 0)
            if 36000 <= n < 36032:
                word = 0x3FC00000 if ch == 0 else 0
            elif 60000 <= n < 60032:
                word = 0xBFC00000 if ch == 1 else 0
            result.extend(struct.pack("<I", word))
    return bytes(result)


def tonal_pcm(rate: int, seconds: int, centers: tuple[F, ...]) -> bytes:
    result = bytearray()
    for n in range(rate*seconds):
        t = n/rate
        for ch, hz in enumerate((997, 1499)):
            value = ((n//(rate//8)+3*ch) % 8-4)/2048 + math.sin(2*math.pi*hz*t)/256
            for center in centers:
                d = t-float(center)
                if abs(d) <= .005:
                    value += (.75 if ch == 0 else -.5)*(1+math.cos(math.pi*d/.005))/2
            result.extend(struct.pack("<f", value))
    return bytes(result)


def float44_pcm() -> bytes:
    return tonal_pcm(44100, 3, (F(11, 10), F(8, 5)))


def alpha_pixel(x: int, y: int) -> tuple[int, int, int, int]:
    if not 0 <= x < 32 or not 0 <= y < 32:
        raise ValueError("32x32 fixture pixel bounds")
    return ((255, 0, 0, 255) if y < 16 and x < 16 else
            (0, 255, 0, 128) if y < 16 else
            (0, 0, 255, 0) if x < 16 else (255, 255, 255, 255))


def alpha_png() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data))+tag+data+struct.pack(">I", zlib.crc32(tag+data))
    rows = b"".join(b"\0"+bytes(v for x in range(32) for v in alpha_pixel(x, y)) for y in range(32))
    return (b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 6, 0, 0, 0))
            +chunk(b"IDAT", zlib.compress(rows))+chunk(b"IEND", b""))


def tiny_color(ordinal: int) -> tuple[int, int, int]:
    return (32+ordinal, (64+2*ordinal) % 256, 192-ordinal)


def tiny_rgb(frames: int) -> bytes:
    if not 1 <= frames <= 180:
        raise ValueError("tiny finite video recipe <=180 frames")
    return b"".join(bytes((255, 255, 255) if n & (1 << (x-4)) else (0, 0, 0))
                    if y in (12, 13) and 4 <= x < 12 else bytes(tiny_color(n))
                    for n in range(frames) for y in range(16) for x in range(16))


def tiny_frame_ordinals() -> tuple[int | None, ...]:
    return 0, 1, None, None, 8, 9


def synthetic_source_ordinal(k: int) -> int | None:
    if not 0 <= k < 480:
        raise ValueError("synthetic output frame bound")
    return k+30 if k < 150 else None if k < 270 else k+210


def synthetic_crop(n: int) -> tuple[F, F, F, F]:
    if not 0 <= n <= 150:
        raise ValueError("whole-shot crop ordinal bound")
    x = F(1, 4)+F(n, 960) if n <= 120 else F(3, 8)-F(n-120, 240)
    return x, F(1, 4), F(1, 2), F(1, 2)


def marker_center(ordinal: int) -> tuple[int, int]:
    return 960+(ordinal*7) % 200, 400+(ordinal*3) % 240


def rect(x: float, y: float, width: float, height: float) -> dict[str, float]:
    return dict(x=x, y=y, width=width, height=height)


def layer(name: str, *, fill: str = "cover", z: int = 0,
          destination: dict[str, float] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = dict(layer_id=name, kind="solid" if fill == "solid" else "source_video", source_ref="source",
        destination=destination or rect(0, 0, 1, 1), fill=fill, z_index=z)
    if fill == "solid":
        item["solid_color"] = "#203040"
    if fill == "contain_blur":
        item["blur_sigma"] = 12
    return item


def shot(name: str, start: float, end: float, output: float, *, solid: bool = False
         ) -> dict[str, Any]:
    return dict(shot_id=name, source=dict(start=start, end=end),
        output=dict(start=output, end=float(F(str(output))+F(str(end))-F(str(start)))), primary_source="content",
        reason="finite authored qualification fixture", confidence=1., evidence_ids=["fixture"],
        layers=[layer(name+"-full", fill="solid" if solid else "cover")],
        audio=dict(normalize_dialogue=False, limiter=True, cues=[]))


def plan_payload(fingerprint: str, shots: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(contract_version=2, clip_id="native57", source_fingerprint=fingerprint,
        composition_fingerprint=RECIPE_VERSION, source_timeline_fingerprint=RECIPE_VERSION,
        identity_fingerprint="no-face-identity-claim", style_library_fingerprint=RECIPE_VERSION,
        shots=shots)


def float_plan(fingerprint: str, *, rate: int) -> Any:
    from clipperx.visual_contracts import VisualCompositionPlan
    start, end = (.625, 1.375) if rate == 48000 else (1.013, 1.713)
    return VisualCompositionPlan.model_validate(plan_payload(fingerprint, [shot("float", start, end, 0)]))


def tiny_plan(fingerprint: str, *, all_solid: bool = False) -> Any:
    from clipperx.visual_contracts import VisualCompositionPlan
    shots = [shot("solid", 0, 2/60, 0, solid=True)] if all_solid else [
        shot("first", 0, 2/60, 0), shot("solid", 2/60, 4/60, 2/60, solid=True),
        shot("last", 0.134, 0.16733333333333333, 4/60)]
    if not all_solid:
        # Decimal float endpoints cannot exactly encode 1/30. Choose the inward
        # adjacent representable local endpoint BEFORE native; it rasterizes to
        # the authored two frames and never exceeds its exact decimal occurrence.
        durations = [math.nextafter(float(F(str(item["output"]["end"]))
                    - F(str(item["output"]["start"]))), -math.inf) for item in shots]
        shots[-1]["brand"] = dict(asset_id="alpha_brand", start=0., end=durations[-1],
                                  destination=rect(.6, .3, .2, .2), z_index=30)
        for item, duration in zip(shots, durations):
            item["title"] = dict(text="NATIVE 012", font_asset_id="headline_impact", start=0.,
                end=duration,
                destination=rect(.1, .04, .8, .12), z_index=10)
    return VisualCompositionPlan.model_validate(plan_payload(fingerprint, shots))


def synthetic_plan(fingerprint: str) -> Any:
    from clipperx.visual_contracts import VisualCompositionPlan
    shots = [shot("s1", .5, 3., 0), shot("s2", 5., 7., 2.5, solid=True),
             shot("s3", 8., 11.5, 4.5)]
    shots[0]["layers"][0]["fill"] = "black"
    shots[0]["layers"][0]["source_crop"] = rect(.25, .25, .5, .5)
    shots[0]["layers"][0]["crop_track"] = [dict(at=t, crop=rect(x, .25, .5, .5), easing="linear")
        for t, x in ((0., .25), (2., .375), (2.5, .25))]
    camera = layer("camera", z=20, destination=rect(.1, .10, .8, .25))
    camera["source_crop"] = rect(0, 0, .25, .25)
    shots[0]["layers"].append(camera)
    shots[2]["layers"] = [layer("camera-full", fill="contain_blur")]
    shots[2]["layers"][0]["source_crop"] = rect(0, 0, .25, .25)
    for i, item in enumerate(shots):
        duration = item["source"]["end"]-item["source"]["start"]
        item["forbidden_regions"] = [rect(0, .94, 1, .06)]
        item["title"] = dict(text="NATIVE 012", font_asset_id="headline_impact", start=0.,
            end=duration, destination=rect(.1, .04, .8, .12), z_index=10)
        first, last = ((.5, 2.), (0., 2.), (.25, 3.25))[i]
        item["brand"] = dict(asset_id="alpha_brand", start=first, end=last,
            destination=rect(.78, .72, .12, .08), z_index=30)
        item["audio"]["normalize_dialogue"] = i != 1
        if i != 1:
            region = rect(.1, .82 if i == 0 else .16, .8, .1)
            item["subtitle_allowed_regions"] = [region]
            item["subtitles"] = dict(destination=region, emphasize_active_word=False)
    shots[1]["audio"]["cues"] = [dict(asset_id="impact", start=.5, end=.6,
                                             gain_db=-12., duck_source_db=-6.)]
    return VisualCompositionPlan.model_validate(plan_payload(fingerprint, shots))


def transcript() -> list[dict[str, Any]]:
    return [dict(start=.7, end=1.7, text="CAMARA UNO"),
            dict(start=8.2, end=9.2, text="CAMARA DOS")]


def reference_argv(ffmpeg: str, source: str, output: str, *, channels: int,
                   input_rate: int, delivery_tags: bool = False) -> tuple[str, ...]:
    layout = "mono" if channels == 1 else "stereo"
    # Independently authored no-seek reference; never uses observer argv builder.
    # Delivery diagnostic tags use fixed4800-sample frames with p=0: grouping
    # only, no padding/sample/phase change. All8s tags then fit the64KiB tail.
    grouping = "asetnsamples=n=4800:p=0," if delivery_tags else ""
    graph = (f"asettb=1/{input_rate},aresample=48000:async=0:osf=flt:tsf=fltp,"
             f"aformat=sample_fmts=flt:channel_layouts={layout},{grouping}asettb=1/48000,ashowinfo@qref")
    return (ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info", "-n", "-copyts",
        "-i", source, "-map", "0:a:0", "-vn", "-sn", "-dn", "-af", graph,
        "-c:a", "pcm_f32le", "-threads", "1", "-f", "f32le", output)
