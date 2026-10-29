"""Independent finite qualification oracles; no compiler, binder or mapper imports.

Every expectation comes from authored fixture clocks/geometry. A missing measurement
is never positive evidence. These functions do no file, process or font IO.
"""

from fractions import Fraction
import hashlib
import math
import re
import struct
from typing import Any, Callable, Iterable, Sequence
import zlib

RATE = 48000


def finite_pcm(data: bytes, channels: int, samples: int | None = None) -> tuple[float, ...]:
    if channels not in (1, 2) or len(data) % (4 * channels):
        raise ValueError("unaligned PCM bytes/channels")
    if samples is not None and len(data) != samples * channels * 4:
        raise ValueError("PCM sample count/EOF mismatch")
    if len(data) > 16 * 1024**2:
        raise ValueError("reference PCM ceiling")
    values = tuple(v[0] for v in struct.iter_unpack("<f", data))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite PCM")
    return values


def bit_identity(observed: bytes, expected: bytes, *, channels: int, samples: int) -> str:
    finite_pcm(observed, channels, samples)
    finite_pcm(expected, channels, samples)
    if observed != expected:
        raise ValueError("float bit identity failure including signed zero")
    return hashlib.sha256(observed).hexdigest()


def grid_offset(reference_grid: Fraction, observed_grid: Fraction) -> int:
    offset = (observed_grid - reference_grid) * RATE
    if offset.denominator != 1:
        raise ValueError("nonintegral reference/observer phase")
    return int(offset)


def reference_indices(origin: Fraction, source_start: Fraction, output_start: Fraction,
                      first: int, count: int, reference_grid: Fraction) -> tuple[int, ...]:
    if count < 1 or count > 384000 or first < 0:
        raise ValueError("finite independent index count required")
    # The sample whose half-open interval contains the authored requested instant.
    return tuple(math.floor((origin + source_start + Fraction(n, RATE) - output_start
                             - reference_grid) * RATE) for n in range(first, first + count))


def indexed_pcm(reference: bytes, indices: Sequence[int], channels: int) -> bytes:
    finite_pcm(reference, channels)
    stride = channels * 4
    if not indices or min(indices) < 0 or max(indices) >= len(reference) // stride:
        raise ValueError("reference index outside measured EOF")
    return b"".join(reference[j * stride:(j + 1) * stride] for j in indices)


def phase_envelope(observed: bytes, reference: bytes, indices: Sequence[int], *,
                   channels: int) -> list[dict[str, float]]:
    actual = finite_pcm(observed, channels, len(indices))
    ref = finite_pcm(reference, channels)
    if not indices or min(indices) < 1 or max(indices) >= len(ref) // channels - 1:
        raise ValueError("both independent reference neighbors required")
    result = []
    for channel in range(channels):
        raw, residual = [], []
        for n, j in enumerate(indices):
            value = ref[j * channels + channel]
            error = abs(actual[n * channels + channel] - value)
            delta = max(abs(value - ref[(j - 1) * channels + channel]),
                        abs(ref[(j + 1) * channels + channel] - value))
            raw.append(error)
            residual.append(max(0.0, error - delta))
        row = {"raw_max": max(raw), "raw_rms": math.sqrt(sum(x*x for x in raw)/len(raw)),
               "residual_max": max(residual),
               "residual_rms": math.sqrt(sum(x*x for x in residual)/len(residual))}
        if row["residual_max"] > 2e-5 or row["residual_rms"] > 2e-6:
            raise ValueError(f"44.1 residual envelope failure: {row}")
        result.append(row)
    return result


def numeric_errors(observed: bytes, expected: bytes, *, channels: int) -> list[dict[str, float]]:
    actual, reference = finite_pcm(observed, channels), finite_pcm(expected, channels)
    if not actual or len(actual) != len(reference):
        raise ValueError("full observer numerical association/count mismatch")
    result = []
    for ch in range(channels):
        errors = [abs(a-b) for a, b in zip(actual[ch::channels], reference[ch::channels])]
        result.append(dict(raw_max=max(errors), raw_rms=math.sqrt(sum(e*e for e in errors)/len(errors))))
    return result


def reference_tags(tail: bytes, *, truncated: bool, channels: int,
                   pcm: bytes) -> tuple[Fraction, int]:
    """Require every output block to match exact bytes/count/sequence/continuous grid."""
    if truncated or len(tail) > 65536:
        raise ValueError("full reference tags unavailable/truncated")
    pattern = (rb"ashowinfo@qref[^\n]*?n:(\d+) pts:(-?\d+)[^\n]*?channels:(\d+)"
               rb"[^\n]*?rate:(\d+)[^\n]*?nb_samples:(\d+) checksum:([A-Fa-f0-9]{8})")
    rows = re.findall(pattern, tail)
    if not rows:
        raise ValueError("missing full reference tags")
    first = int(rows[0][1])
    count = 0
    for ordinal, row in enumerate(rows):
        n, pts, ch, rate, samples = map(int, row[:5])
        if n != ordinal or pts != first + count or ch != channels or rate != RATE or samples <= 0:
            raise ValueError("reference tag sequence/grid/count association")
        block = pcm[count*channels*4:(count+samples)*channels*4]
        if zlib.adler32(block, 0) != int(row[5], 16):
            raise ValueError("reference tag PCM checksum association")
        count += samples
    finite_pcm(pcm, channels, count)
    return Fraction(first, RATE), count


def frame_grid(pts: Sequence[str | Fraction], count: int) -> None:
    if len(pts) != count:
        raise ValueError("natural frame count mismatch")
    for n, value in enumerate(pts):
        try:
            parsed = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("invalid rational PTS") from error
        # ffprobe decimal serialization is limited to six fractional places.
        if abs(parsed - Fraction(n, 60)) > Fraction(1, 1000000):
            raise ValueError("noncontinuous K/60 PTS")


def marker(observed_ordinal: int, expected_ordinal: int,
           observed_center: tuple[float, float], expected_center: tuple[float, float]) -> None:
    if observed_ordinal != expected_ordinal or any(
            abs(a-b) > 3 for a, b in zip(observed_center, expected_center)):
        raise ValueError("source ordinal/marker projection mismatch")


def blend(foreground: tuple[int, int, int, int], background: tuple[int, int, int]
          ) -> tuple[int, int, int]:
    alpha = foreground[3]
    return tuple((foreground[c]*alpha + background[c]*(255-alpha) + 127)//255
                 for c in range(3))  # type: ignore[return-value]


def color(pixel: Sequence[int], expected: Sequence[int], *, tolerance: int,
          context: str = "") -> None:
    if len(pixel) != 4 or pixel[3] != 255 or any(
            abs(pixel[c]-expected[c]) > tolerance for c in range(3)):
        actual_rgba = tuple(pixel[c] for c in range(min(4, len(pixel))))
        expected_rgb = tuple(expected[c] for c in range(min(3, len(expected))))
        bounded_context = "".join(c if " " <= c <= "~" else "?" for c in context[:96])
        raise ValueError("color/alpha observation mismatch "
                         f"actual_rgba={actual_rgba} actual_length={len(pixel)} "
                         f"expected_rgb={expected_rgb} tolerance={tolerance}"
                         + (f" context={bounded_context}" if bounded_context else ""))


def events(expected: Sequence[Fraction], observed: Sequence[Fraction], tolerance: Fraction) -> None:
    if len(expected) != len(observed) or any(abs(a-b) > tolerance
                                           for a, b in zip(expected, observed)):
        raise ValueError("pulse/channel/lag mismatch")


def burst_centers(pcm: bytes, channels: int, centers: Sequence[Fraction], *,
                  grid: Fraction, radius: int = 240) -> tuple[tuple[Fraction, ...], ...]:
    values = finite_pcm(pcm, channels)
    result = []
    for ch in range(channels):
        measured = []
        for center in centers:
            j = math.floor((center-grid)*RATE)
            if j-radius < 0 or j+radius >= len(values)//channels:
                raise ValueError("burst window exceeds observed region")
            # Energy centroid preserves channel-specific finite burst timing.
            weights = [(n, values[n*channels+ch]**2) for n in range(j-radius, j+radius+1)]
            energy = sum(w for _, w in weights)
            if energy < 1e-12:
                raise ValueError("missing channel burst")
            centroid = sum(n*w for n, w in weights)/energy
            measured.append(grid+Fraction(round(centroid*1000000), 1000000*RATE))
        result.append(tuple(measured))
    return tuple(result)


def faststart(data: bytes) -> tuple[int, int]:
    offset, moov, mdat = 0, None, None
    for offset, size, tag in atoms(data):
        if tag == b"moov":
            moov = offset
        if tag == b"mdat" and mdat is None:
            mdat = offset
    if moov is None or mdat is None or moov >= mdat:
        raise ValueError("MP4 lacks moov-before-mdat faststart")
    return moov, mdat


def atoms(data: bytes) -> Iterable[tuple[int, int, bytes]]:
    offset = 0
    while offset < len(data):
        if len(data)-offset < 8:
            raise ValueError("truncated MP4 atom")
        size, tag = struct.unpack_from(">I4s", data, offset)
        header = 8
        if size == 1:
            if len(data)-offset < 16:
                raise ValueError("truncated extended MP4 atom")
            size = struct.unpack_from(">Q", data, offset+8)[0]
            header = 16
        elif size == 0:
            size = len(data)-offset
        if size < header or size > len(data)-offset:
            raise ValueError("MP4 atom bounds")
        yield offset, size, tag
        offset += size


def aac_presentation(first_pts: Fraction, decoded: int, skip: int, padding: int,
                     expected: int) -> tuple[Fraction, Fraction]:
    if min(decoded, skip, padding, expected) < 0 or skip+padding >= decoded:
        raise ValueError("AAC skip/padding bounds")
    start = first_pts+Fraction(skip, RATE)
    end = first_pts+Fraction(decoded-padding, RATE)
    if start != 0 or decoded-skip-padding != expected:
        raise ValueError("AAC audible presentation count/start mismatch")
    return start, end


def aac_packet_presentation(pts: Sequence[int], durations: Sequence[int], skip: int,
                            padding: int, expected: int) -> tuple[Fraction, Fraction]:
    """Validate the native 48k AAC-LC 1024-sample grid before counting coded samples.

    A short final packet duration already excludes the separately signaled tail
    padding. Restore that padding only for this representation, so the scalar
    oracle always receives coded samples and subtracts each trim exactly once.
    """
    if (not pts or len(pts) != len(durations)
            or any(type(value) is not int for value in (*pts, *durations, skip, padding, expected))):
        raise ValueError("AAC packet sequences/integers required")
    if not 0 <= skip <= 1024 or not 0 <= padding <= 1024 or expected <= 0:
        raise ValueError("AAC packet skip/padding/expected bounds")
    if any(value != pts[0]+1024*k for k, value in enumerate(pts)):
        raise ValueError("AAC packet PTS must follow the 1024-sample grid")
    if any(value != 1024 for value in durations[:-1]) or not 1 <= durations[-1] <= 1024:
        raise ValueError("AAC packet duration bounds/grid")
    coded = sum(durations)
    if durations[-1] != 1024:
        if padding <= 0 or durations[-1]+padding != 1024:
            raise ValueError("AAC short final duration requires exact tail padding")
        coded += padding
    if coded != 1024*len(pts):
        raise ValueError("AAC coded frame count mismatch")
    return aac_presentation(Fraction(pts[0], RATE), coded, skip, padding, expected)


def require_measured(values: dict[str, Any]) -> None:
    if any(value is None for value in values.values()):
        raise ValueError("unmeasured independent oracle; acceptance remains pending")


def barcode(pixels: Sequence[Sequence[int]], *, bits: int = 10) -> int:
    if bits not in (8, 10) or len(pixels) != bits or any(len(pixel) < 3 for pixel in pixels):
        raise ValueError("finite measured barcode bit count required")
    value = 0
    for bit, pixel in enumerate(pixels):
        if min(pixel[:3]) >= 190:
            value |= 1 << bit
        elif min(pixel[:3]) >= 100:
            raise ValueError("ambiguous encoded ordinal bit")
    return value


def pulse_peaks(pcm: bytes, centers: Sequence[Fraction], *, grid: Fraction
                ) -> tuple[tuple[Fraction, ...], ...]:
    values = finite_pcm(pcm, 2)
    result = []
    for ch in range(2):
        measured = []
        for center in centers:
            j = math.floor((center-grid)*RATE)
            first, last = j-1800, j+1800
            if first < 0 or last >= len(values)//2:
                raise ValueError("pulse window outside decoded presentation")
            index = max(range(first, last+1), key=lambda n: abs(values[n*2+ch]))
            amplitude = values[index*2+ch]
            if abs(amplitude) < .01 or (amplitude > 0) != (ch == 0):
                raise ValueError("pulse channel code/sign missing or swapped")
            measured.append(grid+Fraction(index, RATE))
        result.append(tuple(measured))
    return tuple(result)


def flash_ordinals(rgb: bytes, frames: int) -> tuple[int, ...]:
    if len(rgb) != frames*8*8*3:
        raise ValueError("flash patch byte/EOF/frame count")
    return tuple(n for n in range(frames)
                 if sum(rgb[n*192:(n+1)*192])/192 >= 220)


def median_rgb(pixel: Callable[[int, int], Sequence[int]], x: int, y: int) -> tuple[int, int, int]:
    samples = [pixel(x+dx, y+dy) for dy in range(-2, 3) for dx in range(-2, 3)]
    if any(len(value) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in value)
           for value in samples):
        raise ValueError("actual decoded RGB sample shape/range")
    medians = [sorted(value[ch] for value in samples)[12] for ch in range(3)]
    return medians[0], medians[1], medians[2]


def tiny_encoded_pixels(observations: Iterable[tuple[int, Callable[[int, int], Sequence[int]]]],
                        *, all_solid: bool = False) -> list[dict[str, Any]]:
    """Public delivered pixels, independent of the component and source generator."""
    ordinals = (None, None) if all_solid else (0, 1, None, None, 8, 9)
    colors = ((32, 48, 64),)*2 if all_solid else (
        (32, 64, 192), (33, 66, 191), (32, 48, 64), (32, 48, 64), (40, 80, 184), (41, 82, 183))
    records = []
    for expected_k, (k, pixel) in enumerate(observations):
        if expected_k >= len(ordinals) or type(k) is not int or k != expected_k:
            raise ValueError("tiny delivered diagnostic ordinal/count/order association")
        expected, ordinal = colors[k], ordinals[k]
        base = median_rgb(pixel, 500, 1000)
        color((*base, 255), expected, tolerance=12)
        if ordinal is not None:
            observed = barcode([median_rgb(pixel, 120*(bit+1), 1500) for bit in range(8)], bits=8)
            if observed != ordinal:
                raise ValueError("tiny delivered source ordinal barcode mismatch")
        png_active = not all_solid and k >= 4
        medians = []
        for x, y, foreground in ((702, 714, (255, 0, 0, 255)),
                                 (810, 714, (0, 255, 0, 128)),
                                 (702, 822, (0, 0, 255, 0)),
                                 (810, 822, (255, 255, 255, 255))):
            median = median_rgb(pixel, x, y)
            target = blend(foreground, expected) if png_active else expected
            color((*median, 255), target, tolerance=12)
            medians.append(median)
        records.append(dict(k=k, source_ordinal=ordinal, base_rgb_median=base,
                            png_active=png_active, png_rgb_medians=medians, rgb_tolerance=12,
                            scope="independent public encoded pixels, not pre-H264 component"))
    if len(records) != len(ordinals):
        raise ValueError("missing tiny delivered diagnostic frames")
    return records


def overlay_contact_sheet(name: str, rgb: bytes, width: int, height: int,
                          tail: bytes, *, truncated: bool) -> list[dict[str, Any]]:
    # Separately authored literal expectations; never compiler events, extracted
    # tag ordinals or mutable expected labels supplied by the producing command.
    literals = {
        "subtitle_s1": ((108, 1574, 864, 192), (11, 12, 13, 71, 72, 73)),
        "subtitle_s3": ((108, 308, 864, 192), (281, 282, 283, 341, 342, 343)),
        "brand_s1": ((842, 1382, 130, 154), (29, 30, 31, 119, 120, 121)),
        "brand_s2": ((842, 1382, 130, 154), (149, 150, 151, 269, 270, 271)),
        "brand_s3": ((842, 1382, 130, 154), (284, 285, 286, 464, 465, 466))}
    if name not in literals or truncated or len(tail) > 65536:
        raise ValueError("unknown overlay batch or incomplete bounded crop tags")
    region, frames = literals[name]
    crop_width, crop_height = region[2:]
    if (width, height) != (crop_width*6, crop_height) or len(rgb) != width*height*3:
        raise ValueError("overlay contact-sheet dimensions/bytes/count")
    config = re.findall(rb"showinfo@qoverlay[^\n]*config in time_base:\s*(\d+/\d+)", tail)
    rows = re.findall(rb"showinfo@qoverlay[^\n]*n:\s*(\d+)\s+pts:\s*(-?\d+)"
        rb"[^\n]*fmt:rgb24[^\n]*s:(\d+)x(\d+)[^\n]*checksum:([A-Fa-f0-9]{8})", tail)
    if len(config) != 1 or len(rows) != 6:
        raise ValueError("complete six-frame crop tag association required")
    time_base = Fraction(config[0].decode())
    expected_active = (False, True, True, True, False, False)
    records = []
    for cell, (k, active, row) in enumerate(zip(frames, expected_active, rows)):
        n, pts, tag_width, tag_height = map(int, row[:4])
        if n != cell or pts*time_base != Fraction(k, 60) or (tag_width, tag_height) != region[2:]:
            raise ValueError("decoded crop ordinal/PTS/geometry association")
        crop = b"".join(rgb[(y*width+cell*crop_width)*3:(y*width+(cell+1)*crop_width)*3]
                        for y in range(crop_height))
        if zlib.adler32(crop, 0) != int(row[4], 16):
            raise ValueError("decoded crop bytes/showinfo checksum association")
        if name.startswith("subtitle"):
            # Fixed authored yellow ink in these synthetic ROIs only; semantic
            # glyph, clipping, forbidden-area and z-order acceptance stays Root's.
            observed_active = any(all(abs(pixel[ch]-(255, 213, 26)[ch]) <= 12 for ch in range(3))
                                  for pixel in struct.iter_unpack("BBB", crop))
            measurement = "presence of authored yellow RGB ink within12, not text identity"
        else:
            def pixel(x: int, y: int) -> tuple[int, ...]:
                j = (y*crop_width+x)*3
                return tuple(crop[j:j+3])
            red = median_rgb(pixel, 32, 44)
            white = median_rgb(pixel, 97, 109)
            observed_active = all(abs(red[ch]-(255, 0, 0)[ch]) <= 12 for ch in range(3)) and min(white) >= 243
            measurement = "opaque red and white PNG patch RGB medians within12"
        if observed_active != active:
            raise ValueError(f"synthetic overlay activation mismatch at {name} K{k}")
        records.append(dict(name=name, global_k=k, expected_active=active, observed_active=observed_active,
            authored_region=region, sheet_cell=(cell*crop_width, 0, crop_width, crop_height),
            rgb_bytes=len(crop), rgb_sha256=hashlib.sha256(crop).hexdigest(),
            decoded_pts=str(pts*time_base), measurement=measurement,
            glyph_clipping_safe_zone_z_order_acceptance=None))
    return records
