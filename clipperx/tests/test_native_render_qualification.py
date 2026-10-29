"""Fake-only qualification tests. Literal expectations are independent of the renderer."""

from fractions import Fraction as F
import importlib
import struct
from typing import Any, Callable, cast

import pytest


@pytest.fixture(autouse=True)
def no_native(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import ctypes
    import io
    import os
    from pathlib import Path
    import subprocess
    import threading

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("native/process/worker boundary forbidden in source tests")

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)
    monkeypatch.setattr(threading.Timer, "start", forbidden)
    monkeypatch.setattr(threading.Event, "wait", forbidden)
    monkeypatch.setattr(ctypes, "WinDLL", forbidden, raising=False)
    def media_path(path: Any) -> bool:
        if not isinstance(path, (str, bytes, os.PathLike)):
            return False
        name = os.fsdecode(path).replace("\\", "/").lower()
        return ("/data/" in name or "/fonts/" in name or name.startswith("o:/host")
                or name.endswith((".mov", ".mp4", ".png", ".wav", ".f32", ".bgr", ".rgba", ".ttf", ".guard")))
    for name in ("open", "stat", "lstat", "resolve", "mkdir", "unlink", "rmdir", "exists", "is_dir", "iterdir"):
        original = getattr(Path, name)
        def guarded(path: Path, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            if media_path(path):
                forbidden()
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(Path, name, guarded)
    for module, name in ((builtins, "open"), (io, "open"), (os, "open"), (os, "stat"),
                         (os, "lstat"), (os, "scandir")):
        original = getattr(module, name)
        def guarded_open(path: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            if media_path(path):
                forbidden()
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(module, name, guarded_open)


def modules() -> tuple[Any, Any, Any]:
    return cast(tuple[Any, Any, Any], tuple(importlib.import_module(name) for name in (
        "clipperx.tests.media_factory", "clipperx.tests.native_render_oracles",
        "scripts.render_visual_smoke")))


def literal_wav_chunks(raw: bytes) -> list[tuple[bytes, bytes]]:
    """Independent RIFF walk for the captured fixture, including exact bounds."""
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    assert int.from_bytes(raw[4:8], "little") == len(raw)-8
    rows = []
    offset = 12
    while offset < len(raw):
        assert offset % 2 == 0 and offset+8 <= len(raw)
        size = int.from_bytes(raw[offset+4:offset+8], "little")
        end = offset+8+size
        assert end <= len(raw)
        rows.append((raw[offset:offset+4], raw[offset+8:end]))
        offset = end+(size % 2)
    assert offset == len(raw)
    return rows


def test_prepare_synthetic_declares_explicit_float_stereo_cue(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace
    from clipperx import local_paths
    m, _, s = modules()
    captured: dict[str, tuple[bytes, dict[str, Any]]] = {}
    outputs: list[tuple[Any, ...]] = []
    probes: list[tuple[Any, ...]] = []
    def put(leaf: str, raw: bytes, **kwargs: Any) -> Path:
        assert leaf not in captured
        captured[leaf] = (raw, kwargs)
        return Path("memory")/leaf
    def file_output(*args: Any, **kwargs: Any) -> tuple[Path, None]:
        outputs.append((*args, kwargs))
        return Path("memory/synthetic.mp4"), None
    def probe(*args: Any) -> dict[str, Any]:
        probes.append(args)
        return {"streams": [{"codec_type": "video", "nb_read_frames": "720",
                             "width": 1920, "height": 1080}]}
    manifest = {"external": {key: {"path": key+".fixture"}
                            for key in ("headline_impact", "subtitle_bold")}}
    session = SimpleNamespace(put=put, file_output=file_output, probe=probe, manifest=manifest,
        runtime=SimpleNamespace(ffmpeg=Path("fake-ffmpeg"), graph_file_option="-filter_complex_script"),
        read=lambda *args: b"fake staged font", deadline=lambda seconds: seconds,
        files=SimpleNamespace(stage_deadline=lambda deadline: nullcontext()))
    monkeypatch.setattr(m, "tonal_pcm", lambda *args: b"fake source PCM")
    monkeypatch.setattr(m, "alpha_png", lambda: b"fake brand")
    monkeypatch.setattr(local_paths, "source_fingerprint", lambda *args, **kwargs: "source-stat-fingerprint")
    prepared = s.prepare_synthetic(session)
    raw, options = captured["impact.wav"]
    chunks = literal_wav_chunks(raw)
    fmt = dict(chunks)[b"fmt "]
    assert (int.from_bytes(fmt[:2], "little"), len(fmt)) == (0xfffe, 40)
    assert int.from_bytes(fmt[20:24], "little") == 3
    assert fmt[24:] == bytes.fromhex("0300000000001000800000aa00389b71")
    assert options == {"role": "asset", "retain": True}
    assert prepared["cue"] == Path("memory/impact.wav")
    assert s.assets_for(manifest, prepared["brand"], prepared["cue"]).sfx == {"impact": prepared["cue"]}
    assert len(outputs) == 1 and outputs[0][0] == "synthetic.mp4"
    assert probes == [(prepared["source"], "-count_frames", "-show_streams")]
    assert set(captured) == {"synthetic.f32", "alpha.png", "fixture.ttf", "synthetic.txt",
                             "impact.wav", "synthetic.json", "transcript.json"}


def test_impact_wav_literal_extensible_fields_and_unchanged_pcm() -> None:
    import hashlib
    m, _, _ = modules()
    raw = m.impact_wav()
    assert len(raw) == 38480 and int.from_bytes(raw[4:8], "little") == 38472
    chunks = literal_wav_chunks(raw)
    assert [tag for tag, _ in chunks] == [b"fmt ", b"fact", b"data"]
    fmt, fact, pcm = [payload for _, payload in chunks]
    assert len(fmt) == 40
    assert struct.unpack("<HHIIHHHHI", fmt[:24]) == (0xfffe, 2, 48000, 384000, 8, 32, 22, 32, 3)
    assert fmt[24:] == bytes.fromhex("0300000000001000800000aa00389b71")
    assert fact == b"\xc0\x12\x00\x00"  # 4800 frames, independent literal
    assert len(pcm) == 38400
    assert hashlib.sha256(pcm).hexdigest() == "14ff18f321b86b9aa4786a49de927ae4664b6755f48c96ff5cfa23afc544c4d1"


@pytest.mark.parametrize("fault", ("", "absent-normalization", "owner", "files", "normalization_files", "stages"))
def test_public_failure_keeps_original_and_readonly_host_observations(
        monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    from pathlib import Path
    from types import SimpleNamespace
    from clipperx._visual_render_host import VisualRenderFailure
    _, _, s = modules()
    campaign = s.Campaign("synthetic")
    seed = {"scope": "existing campaign observation"}
    campaign.records.append(seed)
    events: list[str] = []
    measured = {"owner": SimpleNamespace(own0=False), "files": SimpleNamespace(files_clear=False),
                "normalization_files": SimpleNamespace(files_clear=False)}
    snapshot_error = PermissionError("precise snapshot failure "+"x"*3000)
    def observe(name: str) -> Any:
        events.append(name)
        if fault == name:
            raise snapshot_error
        return measured[name]
    class Tools:
        @property
        def evidence(self) -> list[Any]:
            events.append("stages")
            if fault == "stages":
                raise snapshot_error
            return [seed]
    host = SimpleNamespace(root=Path("memory/public"), tools=Tools(),
        job=SimpleNamespace(snapshot=lambda: observe("owner")),
        files=SimpleNamespace(snapshot=lambda: observe("files")),
        normalization_files=None if fault == "absent-normalization" else
            SimpleNamespace(snapshot=lambda: observe("normalization_files")))
    original = ValueError("original cue qualification failure")
    failure = VisualRenderFailure(original, cast(Any, host))
    failure.__cause__ = original
    def public(*args: Any, **kwargs: Any) -> Any:
        events.append("public")
        raise failure
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no following oracle or lifecycle action")
    for port in (host, host.job, host.files, host.normalization_files):
        if port is not None:
            for name in ("service_cleanup", "check", "request_stop", "retire", "cleanup"):
                setattr(port, name, forbidden)
    monkeypatch.setattr(s, "bounded_read", lambda *args: s.json_bytes(
        {"shots": [{"output": {"start": "0", "end": "1/30"}}]}))
    monkeypatch.setattr(s, "assets_for", lambda *args: {})
    monkeypatch.setattr(s, "public_render", public)
    monkeypatch.setattr(s, "FixtureSession", forbidden)
    monkeypatch.setattr(s, "verify_delivery", forbidden)
    with pytest.raises(s.QualificationFailure) as caught:
        s.render_and_verify(Path("memory"), {}, campaign, source=Path("source.fixture"),
            plan=Path("plan.fixture"), brand=Path("brand.fixture"), output=Path("result.fixture"),
            frames=2, scenario="all-solid")
    assert caught.value.__cause__ is failure and failure.cause is original and failure.__cause__ is original
    assert caught.value.evidence is campaign.records and campaign.records[0] is seed
    assert len(campaign.records) == 3
    record = campaign.records[-1]
    assert record["scope"] == "public B failed; controller observation, not delivery success"
    assert events == ["public", "owner", "files", *([] if fault == "absent-normalization" else
                                                      ["normalization_files"]), "stages"]
    for name, value in measured.items():
        if name == fault or name == "normalization_files" and fault == "absent-normalization":
            assert name not in record["snapshots"]
        else:
            assert record["snapshots"][name] is value
    if fault == "stages":
        assert "stages" not in record["snapshots"]
    else:
        assert record["snapshots"]["stages"] == (seed,)
    errors = record["snapshot_errors"]
    if fault in (*measured, "stages"):
        assert set(errors) == {fault}
        assert errors[fault]["type"] == "PermissionError" and len(errors[fault]["message"]) == 2048
    else:
        assert errors == {}


def literal_aac_mov(*, shifted: bool = False, reverse: bool = False,
                    fault: str = "", movie_scale: int = 1000, data_handlers: bool = False) -> bytes:
    """Hand-authored v0 boxes, independent of the fixture preparation parser."""
    def box(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I4s", 8+len(payload), tag)+payload
    duration = 3*movie_scale+(5*movie_scale//4 if shifted else 0)
    mvhd = box(b"mvhd", struct.pack(">6I", 0, 0, 0, movie_scale, duration, 0x10000)+bytes(76))
    tracks = []
    for index, handler, scale, trim in ((1, b"vide", 15360, 0), (2, b"soun", 44100, 1024)):
        if fault == "audio-trim" and index == 2:
            trim = 0
        if fault == "video-trim" and index == 1:
            trim = 1
        tkhd = box(b"tkhd", struct.pack(">7I", 3, 0, 0, index, 0,
            duration+int(fault == "track-duration"), 0)+bytes(56))
        mdhd = box(b"mdhd", struct.pack(">6I", 0, 0, 0,
            scale+int(fault == "audio-scale" and index == 2),
            3*scale+trim-int(fault == "media-duration"), 0))
        hdlr = box(b"hdlr", bytes(8)+(b"vide" if fault == "handlers" else handler)+bytes(12))
        data_handler = b""
        if data_handlers:
            hdlr = box(b"hdlr", bytes(4)+(b"dhlr" if fault == "media-role" else b"mhlr")
                +handler+bytes(12)+b"\x0cVideoHandler")
            data_handler = box(b"hdlr", (b"\1\0\0\0" if fault == "data-version" else bytes(4))
                +(b"mhlr" if fault == "data-role" else b"dhlr")
                +(b"vide" if fault == "data-subtype" else b"url ")+bytes(12)+b"\x0bDataHandler")
            if fault == "data-bounds":
                data_handler = box(b"hdlr", bytes(4)+b"dhlrurl ")
        entry = struct.pack(">IiI", 3*movie_scale+int(fault == "edit-duration"), trim,
                            0 if fault == "rate" else 0x10000)
        entries = (struct.pack(">IiI", 1250, -1, 0x10000) if shifted else b"")+entry
        elst = box(b"elst", struct.pack(">II", 0, 2 if shifted else 1)+entries)
        if fault == "entry-count":
            elst = elst[:12]+struct.pack(">I", 2)+elst[16:]
        edts = box(b"edts", elst+(elst if fault == "duplicate-elst" else b""))
        stco = box(b"stco", struct.pack(">III", 0, 1, 32))
        co64 = box(b"co64", struct.pack(">IIQ", 0, 1, 32))
        sample_tables = stco+co64+box(b"stsd", b"opaque-codec")
        if fault == "data-in-stbl":
            sample_tables += data_handler
        minf_handler = b"" if fault in ("data-in-mdia", "data-in-stbl", "data-in-trak") else data_handler
        tables = box(b"minf", minf_handler+box(b"stbl", sample_tables)
            +(data_handler if fault == "duplicate-data" else b""))
        mdia = box(b"mdia", mdhd+hdlr+tables+(hdlr if fault == "duplicate-hdlr" else b""))
        if fault == "data-in-mdia":
            mdia = box(b"mdia", mdhd+hdlr+tables+data_handler)
        children = tkhd+edts+mdia+box(b"uuid", b"keep-track")
        if fault == "data-in-trak":
            children += data_handler
        if fault == "missing-edts":
            children = tkhd+mdia
        if fault == "duplicate-tkhd":
            children += tkhd
        if fault == "duplicate-edts":
            children += edts
        tracks.append(box(b"trak", children))
    if reverse:
        tracks.reverse()
    moov_body = mvhd+b"".join(tracks)+box(b"udta", b"keep-movie")
    if fault == "missing-track":
        moov_body = mvhd+tracks[0]
    if fault == "duplicate-track":
        moov_body += tracks[0]
    if fault == "duplicate-mvhd":
        moov_body += mvhd
    if fault == "child-count":
        moov_body += box(b"free", b"")*128
    if fault == "fragment":
        moov_body += box(b"mvex", b"")
    if fault == "nested-track":
        moov_body += box(b"minf", tracks[0])
    if fault == "depth":
        nested = box(b"free", b"")
        for _ in range(10):
            nested = box(b"minf", nested)
        moov_body += nested
    prefix = box(b"ftyp", b"qt  "+bytes(4))+box(b"wide", b"")+box(b"mdat", b"literal-samples")
    moov = box(b"moov", moov_body)
    if fault == "moov-first":
        return moov+prefix
    if fault == "nonterminal":
        return prefix+moov+box(b"free", b"")
    if fault == "duplicate-mdat":
        prefix += box(b"mdat", b"x")
    if fault == "missing-mdat":
        prefix = box(b"ftyp", b"qt  "+bytes(4))
    if fault == "duplicate-moov":
        moov += moov
    return prefix+moov


@pytest.mark.parametrize("reverse", (False, True))
def test_aac_editlist_literal_bytes_preserve_priming_and_all_media(reverse: bool) -> None:
    m, _, _ = modules()
    original = literal_aac_mov(reverse=reverse)
    shifted, facts = m.prepare_aac_fixture_mov(original)
    assert shifted == literal_aac_mov(shifted=True, reverse=reverse)
    assert len(shifted) == len(original)+24
    assert shifted[:47] == original[:47]  # entire prefix, including mdat and its payload
    assert shifted.count(struct.pack(">IiI", 1250, -1, 0x10000)) == 2
    assert struct.pack(">IiI", 3000, 1024, 0x10000) in shifted
    assert facts["growth_bytes"] == 24 and facts["offset_movie_units"] == 1250
    assert facts["prefix_sha256"] == facts["output_prefix_sha256"]
    assert facts["mdat_sha256"] == facts["output_mdat_sha256"]
    assert sorted(row["original_media_time"] for row in facts["tracks"]) == [0, 1024]


@pytest.mark.parametrize("reverse", (False, True))
def test_aac_editlist_accepts_literal_quicktime_data_handlers_verbatim(reverse: bool) -> None:
    m, _, _ = modules()
    original = literal_aac_mov(reverse=reverse, data_handlers=True)
    shifted, facts = m.prepare_aac_fixture_mov(original)
    assert shifted == literal_aac_mov(shifted=True, reverse=reverse, data_handlers=True)
    literal_data = (b"\0\0\0,hdlr"+bytes(4)+b"dhlrurl "+bytes(12)+b"\x0bDataHandler")
    assert original.count(literal_data) == shifted.count(literal_data) == 2
    assert shifted[:47] == original[:47] and len(shifted) == len(original)+24
    assert struct.pack(">IiI", 3000, 1024, 0x10000) in shifted
    assert facts["prefix_sha256"] == facts["output_prefix_sha256"]
    assert facts["mdat_sha256"] == facts["output_mdat_sha256"]


@pytest.mark.parametrize("fault", (
    "duplicate-data", "data-role", "data-subtype", "data-version", "data-bounds", "media-role",
    "data-in-mdia", "data-in-stbl", "data-in-trak", "duplicate-hdlr", "nested-track"))
def test_aac_editlist_rejects_misplaced_or_invalid_quicktime_handlers(fault: str) -> None:
    m, _, _ = modules()
    with pytest.raises(ValueError):
        m.prepare_aac_fixture_mov(literal_aac_mov(data_handlers=True, fault=fault))


@pytest.mark.parametrize("fault", (
    "audio-trim", "video-trim", "audio-scale", "rate", "track-duration", "edit-duration",
    "entry-count", "duplicate-elst", "duplicate-hdlr", "handlers", "missing-edts",
    "duplicate-tkhd", "missing-track", "duplicate-track", "duplicate-mvhd", "child-count",
    "fragment", "moov-first", "nonterminal", "duplicate-mdat", "missing-mdat", "duplicate-moov",
    "duplicate-edts", "media-duration", "nested-track", "depth"))
def test_aac_editlist_rejects_unsupported_literal_layouts(fault: str) -> None:
    m, _, _ = modules()
    with pytest.raises(ValueError):
        m.prepare_aac_fixture_mov(literal_aac_mov(fault=fault))


@pytest.mark.parametrize("fault", (
    "zero", "extended", "truncated", "overflow", "version", "mdhd-version", "tkhd-version",
    "elst-version", "movie-scale", "fractional-offset", "movie-duration", "short-header",
    "short-hdlr", "missing-mdhd", "duplicate-mdhd", "cap", "output-cap", "duration-overflow",
    "missing-mvhd", "missing-tkhd", "missing-hdlr", "missing-elst", "missing-moov", "already-shifted"))
def test_aac_editlist_rejects_bounds_versions_and_unrepresentable_values(fault: str) -> None:
    m, _, _ = modules()
    data = literal_aac_mov()
    if fault in ("zero", "extended", "overflow"):
        data = struct.pack(">I", {"zero": 0, "extended": 1, "overflow": 0xffffffff}[fault])+data[4:]
    elif fault == "truncated":
        data = data[:-1]
    elif fault.endswith("version"):
        tag = {"version": b"mvhd", "mdhd-version": b"mdhd", "tkhd-version": b"tkhd",
               "elst-version": b"elst"}[fault]
        offset = data.index(tag)+4
        data = data[:offset]+b"\1"+data[offset+1:]
    elif fault == "duration-overflow":
        data = literal_aac_mov(movie_scale=1100000000)
    elif fault == "fractional-offset":
        data = literal_aac_mov(movie_scale=999)
    elif fault == "already-shifted":
        data = literal_aac_mov(shifted=True)
    elif fault.startswith("missing-") and fault != "missing-mdhd":
        data = data.replace(fault.removeprefix("missing-").encode(), b"free", 1)
    elif fault in ("movie-scale", "movie-duration"):
        offset = data.index(b"mvhd")+(16 if fault != "movie-duration" else 20)
        value = {"movie-scale": 0, "movie-duration": 2999}[fault]
        data = data[:offset]+struct.pack(">I", value)+data[offset+4:]
    elif fault in ("short-header", "short-hdlr", "missing-mdhd", "duplicate-mdhd"):
        tag = b"hdlr" if fault == "short-hdlr" else b"mdhd"
        offset = data.index(tag)
        # Keep outer sizes valid while changing the required box's identity/declared boundary.
        if fault == "duplicate-mdhd":
            data = data.replace(b"hdlr", b"mdhd", 1)
        elif fault == "missing-mdhd":
            data = data[:offset]+b"free"+data[offset+4:]
        else:
            data = data[:offset-4]+struct.pack(">I", 12)+data[offset:]
    else:
        extra = (16*1024*1024+1 if fault == "cap" else 16*1024*1024-12)-len(data)
        data = struct.pack(">I4s", extra, b"free")+bytes(extra-8)+data
    with pytest.raises(ValueError):
        m.prepare_aac_fixture_mov(data)


def test_source_modules_exist() -> None:
    for name in ("clipperx.tests.media_factory", "clipperx.tests.native_render_oracles",
                 "scripts.render_visual_smoke"):
        assert importlib.util.find_spec(name) is not None


def test_recipes_freeze_literal_counts_and_corrections() -> None:
    m, _, _ = modules()
    assert m.FLOAT48_SAMPLES == 96000 and m.FLOAT44_SAMPLES == 132300
    assert m.tiny_frame_ordinals() == (0, 1, None, None, 8, 9)
    assert m.synthetic_source_ordinal(0) == 30
    assert m.synthetic_source_ordinal(149) == 179
    assert m.synthetic_source_ordinal(150) is None
    assert m.synthetic_source_ordinal(270) == 480
    assert m.synthetic_source_ordinal(479) == 689
    assert m.synthetic_crop(119) == (F(359, 960), F(1, 4), F(1, 2), F(1, 2))
    assert m.synthetic_crop(120)[0] == F(3, 8)
    assert m.synthetic_crop(121)[0] == F(89, 240)
    assert m.alpha_pixel(24, 24) == (255, 255, 255, 255)
    assert m.alpha_pixel(8, 24) == (0, 0, 255, 0)
    plan = m.synthetic_plan("source-stat-fingerprint")
    assert plan.shots[0].layers[1].destination.y == .10
    assert [s.audio.normalize_dialogue for s in plan.shots] == [True, False, True]
    assert plan.shots[1].audio.cues[0].start == .5
    assert plan.shots[2].brand.start == .25
    assert m.EXPECTED_PULSES == (F(1, 2), F(2), F(3), F(5), F(13, 2))


def test_48k_literal_bits_preserve_signed_zero_subnormal_and_channels() -> None:
    m, o, _ = modules()
    mono = m.float48_pcm(1)
    assert struct.unpack_from("<7I", mono) == (
        0, 0x80000000, 0x3FA00000, 0xBFC00000, 0x35800000, 0xB5800000, 1)
    assert len(mono) == 384000
    stereo = m.float48_pcm(2)
    assert len(stereo) == 768000 and stereo[0:4] != stereo[4:8]
    o.bit_identity(mono, mono, channels=1, samples=96000)
    for corrupt in (b"\0" + mono, mono[:-1], bytes.fromhex("00000080") + mono[4:],
                    struct.pack("<I", 0x7F800000) + mono[4:],
                    struct.pack("<I", 0x7FC00000) + mono[4:]):
        with pytest.raises(ValueError):
            o.bit_identity(corrupt, mono, channels=1, samples=96000)


def test_independent_containing_indices_negative_phase_and_first_split() -> None:
    _, o, _ = modules()
    assert o.reference_indices(F(5, 4), F(5, 8), F(0), 0, 4, F(5, 4)) == (
        30000, 30001, 30002, 30003)
    assert o.reference_indices(F(-1), F(1, 48000), F(0), 0, 3,
                               F(-1) + F(1, 96000)) == (0, 1, 2)
    assert o.grid_offset(F(-1), F(-1) + F(1, 48000)) == 1
    with pytest.raises(ValueError, match="phase"):
        o.grid_offset(F(0), F(1, 96000))
    assert o.reference_indices(F(0), F(1, 96000), F(0), 0, 3, F(0)) == (0, 1, 2)


def test_phase_envelope_scores_all_samples_and_records_raw_error() -> None:
    _, o, _ = modules()
    ref = struct.pack("<5f", 0, .125, .25, .375, .5)
    obs = struct.pack("<3f", .25, .375, .5)
    result = o.phase_envelope(obs, ref, (1, 2, 3), channels=1)
    assert result[0]["raw_max"] == .125 and result[0]["residual_max"] == 0
    with pytest.raises(ValueError, match="residual"):
        o.phase_envelope(struct.pack("<3f", .251, .375, .5), ref, (1, 2, 3), channels=1)
    with pytest.raises(ValueError, match="neighbor"):
        o.phase_envelope(obs, ref, (0, 1, 2), channels=1)
    with pytest.raises(ValueError):
        o.phase_envelope(obs + b"x", ref, (1, 2, 3), channels=1)


def test_frame_pts_count_identity_and_projection() -> None:
    m, o, _ = modules()
    o.frame_grid(["0", "1/60", "1/30"], 3)
    for bad in (["0", "1/60"], ["0", "1/60", "1/30", "1/20"],
                ["0", "1/30", "1/60"], ["0", "nan", "1/30"]):
        with pytest.raises(ValueError):
            o.frame_grid(bad, 3)
    o.marker(30, 30, (100, 200), (102, 198))
    for ordinal, center in ((31, (100, 200)), (30, (104, 200))):
        with pytest.raises(ValueError):
            o.marker(ordinal, 30, center, (100, 200))
    assert m.synthetic_crop(119) != m.synthetic_crop(120) != m.synthetic_crop(121)


def test_color_failure_reports_observed_rgba_expected_rgb_and_tolerance() -> None:
    _, o, _ = modules()
    with pytest.raises(ValueError, match="^color/alpha observation mismatch") as caught:
        o.color((53, 138, 20, 255), (50, 138, 20), tolerance=2)
    message = str(caught.value)
    assert "actual_rgba=(53, 138, 20, 255)" in message
    assert "actual_length=4" in message
    assert "expected_rgb=(50, 138, 20)" in message
    assert "tolerance=2" in message


@pytest.mark.parametrize("pixel", [(), (50, 138, 20), (50, 138, 20, 255, 99),
                                  (50, 138, 20, 254), (53, 138, 20, 255)])
def test_color_failure_context_preserves_shape_alpha_and_rgb_rejection(pixel: tuple[int, ...]) -> None:
    _, o, s = modules()
    with pytest.raises(ValueError, match="^color/alpha observation mismatch") as caught:
        o.color(pixel, (50, 138, 20), tolerance=2, context="K=5 x=702 y=714")
    message = str(caught.value)
    assert f"actual_rgba={pixel[:4]}" in message
    assert f"actual_length={len(pixel)}" in message
    assert "expected_rgb=(50, 138, 20) tolerance=2 context=K=5 x=702 y=714" in message
    assert s.error_details(caught.value)["message"] == message


def test_color_context_and_metadata_stay_bounded_on_failure_only() -> None:
    _, o, _ = modules()
    with pytest.raises(ValueError) as caught:
        o.color(tuple(range(10000)), tuple(range(10000)), tolerance=2,
                context="K=5\n\N{SNOWMAN}" + "x"*10000)
    message = str(caught.value)
    assert "actual_rgba=(0, 1, 2, 3) actual_length=10000 expected_rgb=(0, 1, 2) tolerance=2" in message
    context = message.split(" context=", 1)[1]
    assert context == "K=5??" + "x"*91
    assert len(context) == 96 and context.isascii()
    assert len(message) < 256
    # Rejected alpha/shape still short-circuit RGB access, including an empty expectation.
    for pixel in ((), (50, 138, 20, 254)):
        with pytest.raises(ValueError, match="^color/alpha observation mismatch"):
            o.color(pixel, (), tolerance=2)
    with pytest.raises(IndexError):
        o.color((50, 138, 20, 255), (), tolerance=2)
    for pixel in ((50, 138, 20, 255), (48, 140, 18, 255), (52, 136, 22, 255)):
        assert o.color(pixel, (50, 138, 20), tolerance=2, context="\N{SNOWMAN}"*10000) is None


def test_tiny_component_color_call_context_has_authored_global_k_and_coordinates() -> None:
    import ast
    import inspect

    _, _, s = modules()
    tree = ast.parse(inspect.getsource(s.tiny_component))
    calls = sorted((node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "color"),
                   key=lambda node: node.lineno)
    assert len(calls) == 2
    for call, coordinates in zip(calls, (((500, 1000),),
            ((702, 714), (810, 714), (702, 822), (810, 822)))):
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert ast.literal_eval(keywords["tolerance"]) == 2
        expression = compile(ast.Expression(keywords["context"]), "<color-context>", "eval")
        for i in range(3):
            for local in range(2):
                for x, y in coordinates:
                    assert eval(expression, {}, dict(i=i, local=local, x=x, y=y)) == (
                        f"K={i*2+local} x={x} y={y}")


@pytest.mark.parametrize("name", ("tiny_component", "tiny_encoded_pixels"))
def test_tiny_probe_coordinates_and_foregrounds_are_independent_literals(name: str) -> None:
    import ast
    import inspect

    _, o, s = modules()
    tree = ast.parse(inspect.getsource(getattr(s if name == "tiny_component" else o, name)))
    probes = [ast.literal_eval(node.iter) for node in ast.walk(tree)
              if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
              and [item.id for item in node.target.elts if isinstance(item, ast.Name)]
              in (["x", "y", "fore"], ["x", "y", "foreground"])]
    assert probes == [((702, 714, (255, 0, 0, 255)), (810, 714, (0, 255, 0, 128)),
                      (702, 822, (0, 0, 255, 0)), (810, 822, (255, 255, 255, 255)))]


def test_tiny_probe_isolation_recipe_identity() -> None:
    m, _, _ = modules()
    assert m.RECIPE_VERSION == (
        "native57-v1-phase-envelope-y010-opaque-q4-aac-editlist-preparation"
        "-tiny-probe-isolation-v3-decimal-source")


@pytest.mark.parametrize("global_k,source_start,source_end", ((4, F(8, 60), F(9, 60)),
                                                           (5, F(9, 60), F(10, 60))))
def test_tiny_roundtripped_decimal_requests_stay_inside_intended_source_frames(
        global_k: int, source_start: F, source_end: F) -> None:
    from clipperx.output_timeline import OutputTimeline
    from clipperx.visual_contracts import VisualCompositionPlan

    m, _, _ = modules()
    plan = VisualCompositionPlan.model_validate_json(m.tiny_plan("0"*16).model_dump_json())
    request = OutputTimeline.from_shots(plan.shots).video_request(global_k)
    assert source_start <= request.requested_source < source_end


def test_tiny_probe_isolation_recipe_and_six_frame_schedule() -> None:
    import hashlib
    import math

    from clipperx.output_timeline import OutputTimeline
    from clipperx.visual_contracts import VisualCompositionPlan

    m, _, _ = modules()
    plan = VisualCompositionPlan.model_validate_json(m.tiny_plan("0"*16).model_dump_json())
    shots = plan.model_dump()["shots"]
    assert [row["shot_id"] for row in shots] == ["first", "solid", "last"]
    assert [row["source"] for row in shots] == [
        dict(start=0., end=2/60), dict(start=2/60, end=4/60),
        dict(start=0.134, end=0.16733333333333333)]
    assert repr(plan.shots[2].source.start) == "0.134"
    assert repr(plan.shots[2].source.end) == "0.16733333333333333"
    assert [row["output"] for row in shots] == [
        dict(start=0., end=2/60), dict(start=2/60, end=4/60), dict(start=4/60, end=.1)]
    timeline = OutputTimeline.from_shots(plan.shots)
    assert [(row.k_start, row.k_end, row.frame_count) for row in timeline.raster_spans()] == [
        (0, 2, 2), (2, 4, 2), (4, 6, 2)]
    assert tuple(timeline.video_request(k).requested_source for k in (0, 1)) == (F(0), F(1, 60))
    assert [row["brand"] is not None for row in shots] == [False, False, True]
    assert shots[2]["brand"]["destination"] == dict(x=.6, y=.3, width=.2, height=.2)
    for row in shots:
        duration = math.nextafter(float(F(str(row["output"]["end"]))
                                 - F(str(row["output"]["start"]))), -math.inf)
        assert row["title"]["start"] == 0. and row["title"]["end"] == duration
        assert row["title"]["destination"] == dict(x=.1, y=.04, width=.8, height=.12)
    assert shots[2]["brand"]["start"] == 0.
    assert shots[2]["brand"]["end"] == shots[2]["title"]["end"]
    solid_plan = VisualCompositionPlan.model_validate_json(
        m.tiny_plan("0"*16, all_solid=True).model_dump_json())
    solid = solid_plan.model_dump()["shots"]
    assert len(solid) == 1 and solid[0]["brand"] is None and solid[0]["title"] is None
    assert solid[0]["source"] == dict(start=0., end=2/60)
    assert solid[0]["output"] == dict(start=0., end=2/60)
    assert solid[0]["layers"][0]["kind"] == "solid"
    assert solid[0]["layers"][0]["destination"] == dict(x=0., y=0., width=1., height=1.)
    assert solid[0]["layers"][0]["solid_color"] == "#203040"
    assert [(row.k_start, row.k_end, row.frame_count)
            for row in OutputTimeline.from_shots(solid_plan.shots).raster_spans()] == [(0, 2, 2)]
    assert m.tiny_frame_ordinals() == (0, 1, None, None, 8, 9)
    for frames, expected in (
            (120, "a415be1e09e7598e9ee6efde00ddcaf3b299279418ef46ea0f0456e18c4babc6"),
            (180, "d9c39be77236f9fadbcea8d4523b4fe678907e9c1f4d0d10707362882e750437")):
        assert hashlib.sha256(m.tiny_rgb(frames)).hexdigest() == expected
    assert hashlib.sha256(m.alpha_png()).hexdigest() == (
        "df2d67576878455279d8954bb1513a410600a7b89e87603b5f76383ed7b3e461")


def test_tiny_decimal_source_endpoints_keep_1600_nearest_48k_samples() -> None:
    m, _, _ = modules()
    last = m.tiny_plan("0"*16).shots[2]
    endpoints = tuple(F(str(value)) for value in (last.source.start, last.source.end))
    assert endpoints == (F("0.134"), F("0.16733333333333333"))
    # Independent nearest-integer arithmetic on authored decimals, without the audio mapper.
    qa = tuple((2*value.numerator*48000+value.denominator)//(2*value.denominator)
               for value in endpoints)
    assert qa == (6432, 8032)
    assert qa[1]-qa[0] == 1600
    assert abs(endpoints[1]-endpoints[0]-F(2, 60)) <= F("1e-9")
    assert abs((last.source.end-last.source.start)
               - (last.output.end-last.output.start)) <= 1e-9


def test_tiny_probe_footprints_clear_barcode_and_png_boundaries() -> None:
    # Literal pixel-center cover transform: 16 square -> 1080x1920,
    # scale120, x offset -420. Barcode texel cells occupy x3.5..11.5/y11.5..13.5.
    # Static tiny cover uses CPU scale, whose documented default bicubic support
    # is two source pixels. This is not pinned-build coefficient provenance.
    radius = 2
    for x, y in ((702, 1290), (810, 1290), (702, 1398), (810, 1398)):
        source_x, source_y = (F(x)+F(841, 2))/120-F(1, 2), (F(y)+F(1, 2))/120-F(1, 2)
        assert F(7, 2) < source_x < F(23, 2)
        assert 0 < F(23, 2)-source_y < radius
    clearances: list[F] = []
    png_clearances: list[F] = []
    for x, y in ((702, 714), (810, 714), (702, 822), (810, 822)):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                source_y = (F(y+dy)+F(1, 2))/120-F(1, 2)
                clearances.append(F(23, 2)-source_y)
                png_x = (F(x+dx-648)+F(1, 2))*32/216-F(1, 2)
                png_y = (F(y+dy-660)+F(1, 2))*32/216-F(1, 2)
                png_clearances.extend(abs(value-edge) for value in (png_x, png_y)
                                      for edge in (F(-1, 2), F(31, 2), F(63, 2)))
    assert min(clearances) == F(1231, 240) > 4 > radius
    assert min(png_clearances) == F(206, 27) > 4
    # Entire aspect-preserved PNG x648..864/y660..876 also clears the barcode.
    assert F(23, 2)-((F(876)+F(1, 2))/120-F(1, 2)) > F(469, 100)
    # The unchanged center's 5x5 footprint exceeds the published radius,
    # but does not meet the extra four-source-pixel reserve used for the PNG.
    center_clearance = F(23, 2)-((F(1002)+F(1, 2))/120-F(1, 2))
    assert center_clearance == F(175, 48)
    assert radius < center_clearance < 4
    assert 290 < 660 < 876 < 1500


def test_pixel_alpha_pulse_and_unmeasured_oracles_fail_closed() -> None:
    _, o, _ = modules()
    assert o.blend((0, 255, 0, 128), (100, 20, 40)) == (50, 138, 20)
    o.color((50, 138, 20, 255), (50, 138, 20), tolerance=2)
    for pixel in ((53, 138, 20, 255), (50, 138, 20, 254)):
        with pytest.raises(ValueError):
            o.color(pixel, (50, 138, 20), tolerance=2)
    o.events((F(1), F(2)), (F(1) + F(1, 48000), F(2)), F(1, 48000))
    with pytest.raises(ValueError):
        o.events((F(1), F(2)), (F(1), F(2) + F(2, 48000)), F(1, 48000))
    with pytest.raises(ValueError, match="unmeasured"):
        o.require_measured({"pixels": None, "loudness": None})


def test_mp4_atoms_and_aac_presentation_are_independent() -> None:
    _, o, _ = modules()
    def atom(tag: bytes) -> bytes:
        return struct.pack(">I4s", 8, tag)
    assert o.faststart(atom(b"ftyp") + atom(b"moov") + atom(b"mdat")) == (8, 16)
    for bad in (atom(b"mdat") + atom(b"moov"), b"bad", struct.pack(">I4s", 99, b"moov")):
        with pytest.raises(ValueError):
            o.faststart(bad)
    assert o.aac_presentation(F(-1024, 48000), 385024, 1024, 0, 384000) == (F(0), F(8))
    for skip, pad in ((0, 0), (1024, 1024), (-1, 0)):
        with pytest.raises(ValueError):
            o.aac_presentation(F(-1024, 48000), 385024, skip, pad, 384000)


@pytest.mark.parametrize("mode", ("next-decode", "complete", "gap", "wrong-padding"))
def test_delivery_a11_trimmed_aac_reaches_next_decode_stage(mode: str) -> None:
    from contextlib import nullcontext
    from io import BytesIO
    from pathlib import Path
    from types import SimpleNamespace

    _, _, s = modules()
    packets: dict[str, Any] = {"packets": [
        {"pts": -1024, "duration": 1024, "side_data_list": [
            {"side_data_type": "Skip Samples", "skip_samples": 1024, "discard_padding": 0}]},
        {"pts": 0, "duration": 1024}, {"pts": 1024, "duration": 1024},
        {"pts": 2048, "duration": 1024}, {"pts": 3072, "duration": 1024},
        {"pts": 4096, "duration": 704, "side_data_list": [
            {"side_data_type": "Skip Samples", "skip_samples": 0, "discard_padding": 320}]},
    ]}
    if mode == "gap":
        packets["packets"][2]["pts"] = 1025
    elif mode == "wrong-padding":
        packets["packets"][-1]["side_data_list"][0]["discard_padding"] = 319
    saved: list[tuple[str, bytes, dict[str, Any]]] = []
    probes: list[tuple[str, ...]] = []
    runs: list[tuple[Any, ...]] = []
    metadata: dict[str, Any] = {"streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1080,
         "height": 1920, "pix_fmt": "yuv420p", "nb_read_frames": "6"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
    ]}
    atoms = b"".join(struct.pack(">I4s", 8, tag) for tag in (b"ftyp", b"moov", b"mdat"))

    class MemoryCandidate(BytesIO):
        def fstat(self) -> Any:
            return SimpleNamespace(st_size=len(atoms))

    class NextDecodeStage(Exception):
        pass

    class FakeSession:
        runtime = SimpleNamespace(ffmpeg="fake-ffmpeg")
        files = SimpleNamespace(stage_deadline=lambda deadline: nullcontext(),
                                open_binary=lambda path: MemoryCandidate(atoms))

        def deadline(self, seconds: int) -> int:
            return seconds

        def register(self, *args: Any, **kwargs: Any) -> None:
            pass

        def probe(self, path: Path, *options: str) -> dict[str, Any]:
            probes.append(options)
            if "-count_frames" in options:
                return metadata
            if "-show_frames" in options:
                return {"frames": [{"best_effort_timestamp_time": value} for value in
                                   ("0.000000", "0.016667", "0.033333", "0.050000", "0.066667", "0.083333")]}
            assert options == ("-select_streams", "a:0", "-show_packets", "-show_entries",
                "packet=pts,duration:packet_side_data=side_data_type,skip_samples,discard_padding")
            return packets

        def put(self, leaf: str, raw: bytes, **kwargs: Any) -> Path:
            saved.append((leaf, raw, kwargs))
            return Path(leaf)

        def run(self, *args: Any, **kwargs: Any) -> None:
            runs.append(args)
            assert kwargs == {"kind": "probe", "inputs": (Path("fake.mp4"),)}
            if mode == "next-decode":
                raise NextDecodeStage("stop at following independent complete decode")

    if mode == "next-decode":
        with pytest.raises(NextDecodeStage, match="following independent complete decode"):
            s.verify_delivery(FakeSession(), Path("fake.mp4"), 6, ())
    elif mode in ("gap", "wrong-padding"):
        with pytest.raises(ValueError, match="AAC"):
            s.verify_delivery(FakeSession(), Path("fake.mp4"), 6, ())
    else:
        result = s.verify_delivery(FakeSession(), Path("fake.mp4"), 6, ())
        assert result["metadata"] is metadata
        assert result["natural_frames"] == 6
        assert result["presentation"] == ("0", "1/10")
        assert result["aac_skip"] == 1024 and result["aac_discard_padding"] == 320
    assert len(runs) == (0 if mode in ("gap", "wrong-padding") else 1)
    assert len(probes) == 3
    assert saved == [("delivery-aac-packets.json", s.json_bytes(packets),
                      {"role": "sidecar", "cap": 1_000_000, "retain": True})]


@pytest.mark.parametrize("pts,durations,skip,padding,expected,end", (
    ((-1024, 0, 1024, 2048, 3072, 4096), (1024, 1024, 1024, 1024, 1024, 704), 1024, 320, 4800, F(1, 10)),
    ((-1024, 0, 1024, 2048, 3072, 4096), (1024,)*6, 1024, 320, 4800, F(1, 10)),
    ((-1024, 0, 1024), (1024, 1024, 576), 1024, 448, 1600, F(1, 30)),
    (tuple(range(-1024, 384000, 1024)), (1024,)*376, 1024, 0, 384000, F(8)),
))
def test_aac_packet_presentation_counts_coded_frames(
        pts: tuple[int, ...], durations: tuple[int, ...], skip: int,
        padding: int, expected: int, end: F) -> None:
    _, o, _ = modules()
    assert o.aac_packet_presentation(pts, durations, skip, padding, expected) == (F(0), end)


@pytest.mark.parametrize("change", (
    {"padding": 0}, {"padding": 319}, {"padding": -1}, {"padding": 1025},
    {"skip": 0}, {"skip": 1023}, {"skip": -1}, {"skip": 1025},
    {"expected": 4799}, {"expected": 4801}, {"expected": 0}, {"expected": -1},
    {"durations": (1024, 1023, 1024, 1024, 1024, 704)},
    {"durations": (1024, 1024, 1024, 1024, 1024, 1025)},
    {"durations": (1024, 1024, 1024, 1024, 1024, 0)},
    {"durations": (1024, 1024, 1024, 1024, 1024, -1)},
    {"durations": (1024, 1024, 1024, 1024, 1024, 703)},
    {"pts": ()}, {"durations": ()}, {"pts": (), "durations": ()},
    {"durations": (1024,)*5}, {"pts": (-1024, 0, 1024, 2048, 3072)},
    {"pts": (-1024, 0, 1025, 2048, 3072, 4096)},
    {"pts": (-1024, 0, 0, 2048, 3072, 4096)},
    {"pts": (-1024, 1024, 0, 2048, 3072, 4096)},
    {"pts": (0, 1024, 2048, 3072, 4096, 5120)},
    {"pts": (-1024, False, 1024, 2048, 3072, 4096)},
    {"pts": (-1024, 0.0, 1024, 2048, 3072, 4096)},
    {"pts": (-1024, "0", 1024, 2048, 3072, 4096)},
    {"durations": (1024, 1024, 1024, 1024, 1024, True)},
    {"durations": (1024, 1024, 1024, 1024, 1024, 704.0)},
    {"durations": (1024, 1024, 1024, 1024, 1024, "704")},
    {"skip": True}, {"skip": 1024.0}, {"skip": "1024"},
    {"padding": False}, {"padding": 320.0}, {"padding": "320"},
    {"expected": True}, {"expected": 4800.0}, {"expected": "4800"},
))
def test_aac_packet_presentation_rejects_unknown_or_inconsistent_counts(change: dict[str, Any]) -> None:
    _, o, _ = modules()
    values: dict[str, Any] = dict(pts=(-1024, 0, 1024, 2048, 3072, 4096),
        durations=(1024, 1024, 1024, 1024, 1024, 704), skip=1024, padding=320, expected=4800)
    with pytest.raises(ValueError):
        o.aac_packet_presentation(**{**values, **change})


def test_manifest_guard_optin_budget_and_failure_stop_are_pure() -> None:
    _, _, s = modules()
    with pytest.raises(ValueError, match="opt-in"):
        s.validate_activation(False, True)
    with pytest.raises(ValueError, match="confirm-host"):
        s.validate_activation(True, False)
    manifest = s.example_manifest()
    s.validate_manifest(manifest)
    changes: tuple[dict[str, Any], ...] = ({"pins": {}}, {"accepted_b": "head"}, {"version": 2},
                   {"cases": ["synthetic"]}, {"limits": {"probe_seconds": 900}})
    for change in changes:
        with pytest.raises(ValueError):
            s.validate_manifest({**manifest, **change})
    for directory, guard in ((True, False), (False, True), (True, True)):
        with pytest.raises(ValueError):
            s.require_absence(directory, guard)
    s.require_absence(False, False)
    assert s.disk_budget(1920, 1080, (150, 120, 210), (True, False, True))["raw"] == 5287680000
    for kwargs in ({"children": 97}, {"generations": 262145}, {"paths": 32769},
                   {"live": 17}, {"free": 0}, {"seconds": 2701}):
        with pytest.raises(ValueError):
            s.admit("prerequisites", **kwargs)
    assert s.fs_rejected(1025, 6) and s.fs_rejected(900, 5)
    assert not s.fs_rejected(900, 6)
    with pytest.raises(ValueError):
        s.cancelled_result(admitted=False, acquired=True, before_eof=True,
                           stop_code="cancelled", own0=True, files_clear=True)
    for own0, files_clear in ((False, True), (True, False)):
        with pytest.raises(ValueError):
            s.cancelled_result(admitted=True, acquired=True, before_eof=True,
                               stop_code="cancelled", own0=own0, files_clear=files_clear)
    seen: list[str] = []
    def fail(case: str) -> None:
        seen.append(case)
        raise ValueError("first failure")
    with pytest.raises(ValueError, match="first failure"):
        s.sequence(("A1", "A2"), fail)
    assert seen == ["A1"]


@pytest.mark.parametrize("fault", ("late-open", "close", "consumer", "short-write"))
def test_real_retained_custody_blocks_false_clearance(fault: str) -> None:
    from clipperx._retained_files import RetainedFiles
    from clipperx.tests.test_retained_files import MemoryFS, ROOT
    from clipperx.tests.test_source_host_ownership import admitted, finish_job
    h, fs = admitted(), MemoryFS()
    files = RetainedFiles(h.job, root=ROOT, clock=h.job._clock,
        _open=fs.open, _stat=fs.stat, _path_action=fs.action, _fsync=lambda _: None)
    path = ROOT/"qualification.raw"
    files.register(path, role="raw", owned=True, max_bytes=8)
    if fault == "late-open":
        def late(op: str, _: Any) -> None:
            if op == "open":
                h.now = h.job._deadline
        fs.hook = late
        with pytest.raises(ValueError):
            files.open_file(path, "xb").__enter__()
        assert not files.snapshot().files_clear
        assert files.retire().files_clear
    elif fault == "close":
        def uncertain(op: str, _: Any) -> None:
            if op == "close":
                raise OSError("unknown close")
        fs.hook = uncertain
        with pytest.raises(OSError):
            with files.open_file(path, "xb"):
                pass
        assert not files.retire().files_clear
        assert finish_job(h).own0
        assert not files.snapshot().files_clear
        return
    elif fault == "consumer":
        token = files.acquire_consumer((), outputs=(path,))
        fs.data[path] = b"partial"
        h.job.request_stop("cancelled")
        files.release_consumer(token, retired=False)
        assert not files.retire().files_clear and path in fs.data
        files.release_consumer(token, retired=True)
        assert files.retire().files_clear
    else:
        from scripts.render_visual_smoke import FixtureSession, Campaign
        s = FixtureSession(ROOT, {}, Campaign("prerequisites"))
        s.job, s.files, s.clock = h.job, files, h.job._clock
        fs.write_chunk = 2
        assert s.put("qualification.raw", b"abcd", role="raw", cap=8) == path
        assert fs.data[path] == b"abcd"
        assert files.retire().files_clear
    assert finish_job(h).own0


def test_reference_tags_require_full_sequence_checksum_and_eof() -> None:
    _, o, _ = modules()
    data = struct.pack("<2f", .125, -.25)
    import zlib
    checksum = zlib.adler32(data, 0)
    tags = (f"[ashowinfo@qref @ 00] n:0 pts:60000 fmt:flt channels:1 "
            f"rate:48000 nb_samples:2 checksum:{checksum:08X}\n").encode()
    assert o.reference_tags(tags, truncated=False, channels=1, pcm=data) == (F(5, 4), 2)
    for bad in (tags.replace(b"n:0", b"n:1"), tags.replace(b"nb_samples:2", b"nb_samples:1"),
                tags.replace(f"{checksum:08X}".encode(), b"00000000")):
        with pytest.raises(ValueError):
            o.reference_tags(bad, truncated=False, channels=1, pcm=data)
    with pytest.raises(ValueError):
        o.reference_tags(tags, truncated=True, channels=1, pcm=data)


def test_recipes_use_real_compiler_and_audio_schedule_without_native() -> None:
    m, _, _ = modules()
    from clipperx.visual_compiler import compile_visual_assembly
    from clipperx.audio_mix import schedule_audio
    from clipperx.tests.test_visual_compiler import pure_overlays
    for plan, counts in ((m.tiny_plan("0"*16), (2, 2, 2)),
                         (m.tiny_plan("0"*16, all_solid=True), (2,)),
                         (m.synthetic_plan("0"*16), (150, 120, 210))):
        assembly = compile_visual_assembly(plan, source_size=(1920, 1080), overlays=pure_overlays(plan))
        assert tuple(shot.frame_count for shot in assembly.shots) == counts
        if len(counts) == 3:
            assert assembly.shots[1].transport is None
        else:
            assert assembly.source_input is None
    synthetic = m.synthetic_plan("0"*16)
    assert schedule_audio(synthetic).total_samples == 384000
    assert schedule_audio(m.float_plan("0"*16, rate=44100)).total_samples == 33600


def test_pin_mismatch_precedes_setup_and_public_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path
    _, _, s = modules()
    data = s.example_manifest()
    monkeypatch.setattr(s, "bounded_read", lambda *args: s.json_bytes(data))
    monkeypatch.setattr(s, "file_hash", lambda *args: "f"*64)
    with pytest.raises(ValueError, match="current source pin mismatch"):
        s.load_manifest(Path("manifest.json"), Path("O:/HOST/repository"))


def test_literal_barcode_pulse_and_flash_classification() -> None:
    _, o, _ = modules()
    assert o.barcode([(255, 255, 255) if 30 & (1 << n) else (32, 64, 128)
                      for n in range(10)]) == 30
    data = bytearray(48000*8)
    struct.pack_into("<2f", data, 24000*8, .75, -.5)
    assert o.pulse_peaks(bytes(data), (F(1, 2),), grid=F(0)) == ((F(1, 2),), (F(1, 2),))
    swapped = bytearray(data)
    struct.pack_into("<2f", swapped, 24000*8, -.5, .75)
    with pytest.raises(ValueError, match="channel"):
        o.pulse_peaks(bytes(swapped), (F(1, 2),), grid=F(0))
    assert o.flash_ordinals(bytes([20, 30, 40])*64+bytes([255, 255, 255])*64, 2) == (1,)
    with pytest.raises(ValueError):
        o.flash_ordinals(bytes(385), 2)


def test_new_root_approved_current_pin_with_explicit_lineage(monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    from io import BytesIO
    from pathlib import Path
    _, _, s = modules()
    data = s.example_manifest()
    # Trusted external bytes only. The actual SHA reader and manifest validator
    # remain real; no renderer/runtime/receipt validator is replaced.
    payload = b"new separately reviewed current source"
    original_open = Path.open
    def external_bytes(path: Path, *args: Any, **kwargs: Any) -> Any:
        if "snapshots" in path.parts and "task-56" in path.parts:
            return original_open(path, *args, **kwargs)
        return BytesIO(payload)
    monkeypatch.setattr(Path, "open", external_bytes)
    with pytest.raises(ValueError, match="current source pin mismatch"):
        s.check_source_pins(data, Path("O:/Escritorio/ClipperX"), 2**63-1)
    data["pins"] = {key: hashlib.sha256(payload).hexdigest() for key in data["pins"]}
    s.validate_manifest(data)
    # Exact immutable source snapshot evidence is read normally; only current
    # source bytes are synthetic. There is no old B source byte-lock.
    s.check_source_pins(data, Path("O:/Escritorio/ClipperX"), 2**63-1)
    current_only = {**data, "lineage": {}}
    with pytest.raises(ValueError):
        s.validate_manifest(current_only)


def test_tiny_180_frame_recipe_has_valid_bytes_and_independent_barcode() -> None:
    m, o, _ = modules()
    rgb = m.tiny_rgb(180)
    assert len(rgb) == 180*768
    assert rgb[(12*16+4)*3:(12*16+5)*3] == bytes((0, 0, 0))
    offset = 768+(12*16+4)*3
    assert rgb[offset:offset+3] == bytes((255, 255, 255))
    for n in (0, 1, 8, 9, 119, 179):
        pixels = [rgb[n*768+(12*16+4+b)*3:n*768+(12*16+5+b)*3] for b in range(8)]
        assert o.barcode(pixels, bits=8) == n


def test_actual_public_evidence_requires_whole_shots_and_n120() -> None:
    _, _, s = modules()
    meta: dict[str, Any] = dict(source_bindings=[], stages=[
        dict(kind="spatial", stdout_bytes=16588800, natural_eof=True, retired=True),
        dict(kind="encoder", stdout_bytes=0, natural_eof=True, retired=True)], audio={"normalization": None},
                                maximum_active_media_children=1, process_counts={"source_stream": 0})
    s.validate_public_evidence(meta, 2, all_solid=True)
    bad = {**meta, "source_bindings": [{"slots": []}]}
    with pytest.raises(ValueError):
        s.validate_public_evidence(bad, 2, all_solid=True)
    with pytest.raises(ValueError):
        s.validate_public_evidence(meta, 480, synthetic=True)


def literal_a13_all_solid_metadata() -> dict[str, Any]:
    return dict(maximum_active_media_children=1, process_count=17,
        process_counts=dict(render_tools=16, source_stream=0, source_float=1, normalization=0),
        source_bindings=[], audio={"normalization": None}, stages=[
            dict(kind="probe", argv=["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc2=s=16x16:r=1", "-frames:v", "1", "-an", "-threads", "1",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"], stdout_bytes=768,
                natural_eof=True, retired=True),
            dict(kind="spatial", argv=["ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=60",
                "-frames:v", "2", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                stdout_bytes=16588800, natural_eof=True, retired=True),
            dict(kind="encoder", argv=["ffmpeg", "-pixel_format", "rgba", "-i", "sealed.rgba"],
                stdout_bytes=0, natural_eof=True, retired=True)])


def test_all_solid_accepts_literal_a13_bgr24_capability_output() -> None:
    _, _, s = modules()
    s.validate_public_evidence(literal_a13_all_solid_metadata(), 2, all_solid=True)


@pytest.mark.parametrize("fault", ("bindings", "stream", "missing-counts", "missing-stream",
    "none", "string", "float", "false", "true", "spatial-input", "spatial-output"))
def test_all_solid_requires_zero_source_stream_and_no_spatial_bgr24(fault: str) -> None:
    _, _, s = modules()
    meta = literal_a13_all_solid_metadata()
    # Isolate each invalid source observation from the independently valid probe.
    meta["stages"] = meta["stages"][1:]
    if fault == "bindings":
        meta["source_bindings"] = [{"slots": []}]
    elif fault == "missing-counts":
        del meta["process_counts"]
    elif fault == "missing-stream":
        del meta["process_counts"]["source_stream"]
    elif fault.startswith("spatial-"):
        meta["stages"][0]["argv"] = (["ffmpeg", "-f", "rawvideo", "-pixel_format", "bgr24",
            "-i", "generated-inlet", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"]
            if fault == "spatial-input" else ["ffmpeg", "-pix_fmt", "bgr24", "pipe:1"])
    else:
        meta["process_counts"]["source_stream"] = {
            "stream": 1, "none": None, "string": "0", "float": 0.0, "false": False, "true": True}[fault]
    with pytest.raises(ValueError, match="all-SOLID unexpectedly admitted video observer/BGR inlet") as caught:
        s.validate_public_evidence(meta, 2, all_solid=True)
    assert type(caught.value) is ValueError


def test_all_solid_ignores_bgr_substrings_in_spatial_paths_and_labels() -> None:
    _, _, s = modules()
    meta = literal_a13_all_solid_metadata()
    meta["stages"] = meta["stages"][1:]
    meta["stages"][0]["label"] = "bgr-labelled spatial stage"
    meta["stages"][0]["argv"].extend(["-metadata", "title=bgr24 capability", "generated-bgr24.rgba"])
    s.validate_public_evidence(meta, 2, all_solid=True)


def test_non_all_solid_keeps_existing_source_observations() -> None:
    _, _, s = modules()
    meta = literal_a13_all_solid_metadata()
    meta["source_bindings"] = [{"slots": []}]
    meta["process_counts"]["source_stream"] = 1
    meta["stages"][1]["argv"] = ["ffmpeg", "-pixel_format", "bgr24", "-i", "generated-inlet"]
    s.validate_public_evidence(meta, 2)


@pytest.mark.parametrize("fault", ("bytes", "eof", "retired", "encoder", "serial"))
def test_capability_probe_does_not_relax_spatial_delivery_contract(fault: str) -> None:
    _, _, s = modules()
    meta = literal_a13_all_solid_metadata()
    if fault == "bytes":
        meta["stages"][1]["stdout_bytes"] -= 1
    elif fault == "encoder":
        meta["stages"].append(dict(meta["stages"][-1]))
    elif fault == "serial":
        meta["maximum_active_media_children"] = 2
    else:
        meta["stages"][1]["natural_eof" if fault == "eof" else "retired"] = False
    with pytest.raises(ValueError, match="serial child|whole-shot spatial"):
        s.validate_public_evidence(meta, 2, all_solid=True)


def test_post_return_validator_failure_keeps_original_cause_and_campaign_records(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path
    _, _, s = modules()
    campaign = s.Campaign("synthetic")
    seed = {"scope": "existing fake campaign observation"}
    records = campaign.records
    records.append(seed)
    meta = literal_a13_all_solid_metadata()
    meta["source_bindings"] = [{"slots": []}]  # real validator failure, not a fabricated exception
    candidate = Path("controller/returned.mp4")
    plan = Path("all-solid.json")
    suffixes = (".visual-plan.json", ".render-spec.json", ".metadata.json", ".render-metrics.json", ".ffmpeg.log")
    returned = [candidate, *(candidate.with_suffix(suffix) for suffix in suffixes)]
    calls: list[str] = []
    errors: list[Exception] = []
    def read(path: Path, limit: int) -> bytes:
        assert limit == 8*s.MiB
        if path == plan:
            return s.json_bytes({"shots": [{"output": {"start": "0", "end": "1/30"}}]})
        assert path == candidate.with_suffix(".metadata.json")
        return s.json_bytes(meta)
    def public(*args: Any, **kwargs: Any) -> list[Path]:
        calls.append("public")
        return [candidate]
    validate = s.validate_public_evidence
    def observe(*args: Any, **kwargs: Any) -> None:
        calls.append("validator")
        try:
            validate(*args, **kwargs)
        except Exception as error:
            errors.append(error)
            raise
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("following oracle/session")
        raise AssertionError("must stop at post-return evidence failure")
    monkeypatch.setattr(s, "bounded_read", read)
    monkeypatch.setattr(s, "public_render", public)
    monkeypatch.setattr(s, "assets_for", lambda *args: {})
    monkeypatch.setattr(Path, "iterdir", lambda path: iter(returned))
    monkeypatch.setattr(Path, "is_file", lambda path: path in returned)
    monkeypatch.setattr(s, "validate_public_evidence", observe)
    monkeypatch.setattr(s, "FixtureSession", forbidden)
    monkeypatch.setattr(s, "verify_delivery", forbidden)
    with pytest.raises(s.QualificationFailure) as caught:
        s.render_and_verify(Path("controller"), {}, campaign, source=Path("source.fixture"),
            plan=plan, brand=Path("brand.fixture"), output=candidate, frames=2, scenario="all-solid")
    assert len(errors) == 1 and type(errors[0]) is ValueError
    assert caught.value.__cause__ is errors[0]
    assert str(caught.value) == str(errors[0]) == "all-SOLID unexpectedly admitted video observer/BGR inlet"
    assert caught.value.evidence is records is campaign.records
    assert records[0] is seed and len(records) == 3
    assert records[-1]["actual_children"] == 17
    assert calls == ["public", "validator"]


def test_excess_after_public_return_stops_before_next_case(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, s = modules()
    now = [0]
    monkeypatch.setattr(s.time, "monotonic_ns", lambda: now[0])
    campaign = s.Campaign("synthetic")
    campaign.charge(48)
    seen: list[str] = []
    def run(case: str) -> None:
        seen.append(case)
        now[0] = 3601*10**9  # exact B call has returned, beyond campaign ceiling
        s.record_public_return(campaign, 30, 48)
    with pytest.raises(ValueError, match="admission"):
        s.sequence(("public-return", "next-case"), run)
    assert seen == ["public-return"]


def test_full_observer_prefix_suffix_errors_are_recorded_without_exclusions() -> None:
    _, o, _ = modules()
    ref = struct.pack("<3f", 0, .125, .25)
    obs = struct.pack("<3f", .5, .125, .75)
    measured = o.numeric_errors(obs, ref, channels=1)
    assert measured[0]["raw_max"] == .5
    assert measured[0]["raw_rms"] == pytest.approx((.5/3)**.5)


def fixture_adapter(monkeypatch: pytest.MonkeyPatch, *, cancel: bool = False,
                    parent_seconds: int | None = None,
                    cancel_status: Callable[[Any], Any] | None = None,
                    cancel_after_payload: bool = False, cancel_abort_read: bool = False) -> Any:
    import hashlib
    import io
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from clipperx import render_runtime as rr, source_decode as sd
    from clipperx._retained_files import RetainedFiles
    from clipperx.tests.test_retained_files import MemoryFS, ROOT
    from clipperx.tests.test_source_host_ownership import admitted, scripted_host, ScriptProcess
    from scripts import render_visual_smoke as smoke
    if parent_seconds is None:
        h = admitted()
    else:
        from clipperx._source_host_ownership import SourceHostJob
        h = scripted_host()
        original = h.job
        h.job = SourceHostJob(h.driver, deadline_ns=h.now+parent_seconds*10**9,
            cleanup_grace_ns=100, clock=original._clock, io_factory=h.io_factory,
            worker_factory=original._workers, event_factory=original._events,
            current_worker=original._current_worker, process_factory=h.process_factory)
        h.job.start_driver()
        h.current = h.driver
        h.job.register_current_driver()
    fs = MemoryFS()
    def action(operation: str, path: Path, destination: Path | None) -> Any:
        return (1 << 40, 0, 1 << 40) if operation == "disk_usage" else fs.action(operation, path, destination)
    files = RetainedFiles(h.job, root=ROOT, clock=h.job._clock,
        _open=fs.open, _stat=fs.stat, _path_action=action, _fsync=lambda _: None)
    files.register(ROOT, role="checked", owned=False, max_bytes=2**48)
    monkeypatch.setattr(smoke.time, "monotonic_ns", h.job._clock)
    s = smoke.FixtureSession(ROOT, {}, smoke.Campaign("prerequisites"))
    s.job, s.files, s.clock = h.job, files, h.job._clock
    pair = (Path("O:/HOST/ffmpeg.exe"), Path("O:/HOST/ffprobe.exe"))
    hashes = tuple(hashlib.sha256(body).hexdigest().upper() for body in (b"mpeg", b"probe"))
    for path, body in zip(pair, (b"mpeg", b"probe")):
        fs.data[path] = body
        files.register(path, role="runtime", owned=False, max_bytes=100)
    for module in (rr, sd):
        monkeypatch.setattr(module, "FFMPEG_SHA256", hashes[0])
        monkeypatch.setattr(module, "FFPROBE_SHA256", hashes[1])
    s.runtime = rr.RenderRuntime(pair[0], pair[1], rr.NATIVE_RUNTIME,
        hashes[0], hashes[1], hashes[0], hashes[1], "-/filter_complex", None)
    source = ROOT/"source.mov"
    fs.data[source] = b"finite fixture"
    files.register(source, role="original", owned=False, max_bytes=100)
    metadata = dict(format=dict(start_time="1.25", duration="2"), streams=[
        dict(index=0, codec_type="video", time_base="1/60", start_pts=75,
             duration_ts=120, width=16, height=16),
        dict(index=1, codec_type="audio", time_base="1/48000", start_pts=60000,
             duration_ts=96000, sample_rate="48000", channels=1, sample_fmt="flt")])
    stops: list[str] = []
    snapshots: list[Any] = []
    request_stop = h.job.request_stop
    def recorded_stop(code: str) -> None:
        stops.append(code)
        request_stop(code)
    monkeypatch.setattr(h.job, "request_stop", recorded_stop)
    def factory(argv: tuple[str, ...]) -> Any:
        child = ScriptProcess(json.dumps(metadata).encode() if argv[0] == str(pair[1]) else b"abcd", b"")
        if cancel:
            class CancelPipe(io.BytesIO):
                def read(self, size: int | None = -1) -> bytes:
                    if not snapshots:
                        status = h.job.snapshot()
                        current = status.current
                        assert s.cancel_ready and status.failure is None and current is not None
                        assert current.kind == "spatial" and current.phase == "active"
                        assert current.scope_open and current.consumer_active
                        assert len(current.diagnostics) == 1
                        diagnostic = current.diagnostics[0]
                        assert diagnostic.kind == "spatial" and diagnostic.exit_code is None
                        assert not diagnostic.natural_eof and not diagnostic.transfer_uncertain
                        assert any(row.role == "process" and row.generation == diagnostic.generation
                                   and row.pid is not None and row.alive is None
                                   and row.phase == "live" and not row.uncertain for row in current.resources)
                        assert any(row.path.endswith("cancel.rgba") and row.operation == "fstat"
                                   and row.phase == "returned" and not row.uncertain
                                   for row in files.snapshot().rows)
                        snapshots.append(status)
                        if not cancel_after_payload:
                            observed = status if cancel_status is None else cancel_status(status)
                            s._observe_paced_cancel(observed)
                            s._observe_paced_cancel(observed)  # repeated active poll must stay one-shot
                            if cancel_status is None:
                                assert s.cancel_observed and stops == ["cancelled"]
                            if cancel_abort_read:
                                raise OSError(995, "trusted fake cancelled read")
                    raw = super().read(size)
                    if cancel_after_payload:
                        if self.tell() == 120*16*16*4:
                            assert not stops
                            s._observe_paced_cancel(snapshots[0])
                            s._observe_paced_cancel(snapshots[0])
                            assert stops == ["cancelled"]
                        else:
                            assert self.tell() == 65536 and not stops
                    return raw
            child.stdout = CancelPipe(bytes(120*16*16*4) if cancel_after_payload else b"abcd")
        h.children.append(child)
        return child
    h.job._processes = factory
    s.tools = s.make_tools()
    return SimpleNamespace(session=s, h=h, fs=fs, files=files, source=source,
                           stops=stops, snapshots=snapshots)


@pytest.mark.parametrize("path", ("probe", "raw", "cancel"))
def test_actual_fixture_adapter_composes_deadlines(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    from clipperx.tests.test_source_host_ownership import finish_job
    from clipperx._source_host_ownership import HostedObservationFailure
    c = fixture_adapter(monkeypatch, cancel=path == "cancel")
    s = c.session
    if path == "probe":
        origin = s.source_probe(c.source)
        assert origin.origin == F(5, 4) and origin.streams[0].width == 16
        assert origin.streams[1].sample_format == "flt"
        assert any(name.name.startswith("retirement") for name in c.fs.data)
    elif path == "raw":
        output = s.raw_output("raw.rgba", (str(s.runtime.ffmpeg), "finite"), 4, ())
        assert c.fs.data[output] == b"abcd"
        assert any(name.name.startswith("retirement") for name in c.fs.data)
    else:
        with pytest.raises(HostedObservationFailure) as caught:
            s.paced_cancellation(c.source)
        assert c.stops == ["cancelled"] and s.cancel_observed
        assert caught.value.failure.code == "cancelled"
        assert s.state["cancellation_window"] is c.snapshots[0].current
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


def test_complete_payload_hosted_cancel_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx._source_host_ownership import HostedObservationFailure
    from clipperx.tests.test_source_host_ownership import finish_job
    from scripts import render_visual_smoke as smoke
    c = fixture_adapter(monkeypatch, cancel=True, cancel_after_payload=True)
    c.session.run_driver(lambda: c.session.paced_cancellation(c.source))
    error = c.session.state["error"]
    assert isinstance(error, HostedObservationFailure) and error.failure.code == "cancelled"
    status = finish_job(c.h)
    diagnostic = status.current.diagnostics[0]
    assert diagnostic.stdout_bytes == 122880 and not diagnostic.natural_eof
    assert status.own0 and c.files.snapshot().files_clear and c.stops == ["cancelled"]
    assert c.session.state["cancellation_window"] is c.snapshots[0].current
    assert c.snapshots[0].current.diagnostics[0].stdout_bytes == 0
    with pytest.raises(smoke.QualificationFailure) as caught:
        c.session.finish_result(cancel=True)
    assert caught.value.__cause__ is error
    evidence = caught.value.evidence[-1]["cancellation"]
    assert evidence["classification"] == "ineligible_window" and not evidence["incomplete_payload"]
    assert evidence["terminal"].diagnostics[0].stdout_bytes == 122880


@pytest.mark.parametrize("case", ("not_ready", "already_observed", "job_failure", "missing_current",
    "other_kind", "not_active", "closed_scope", "not_consumer", "no_diagnostic", "two_diagnostics",
    "natural_eof", "exit_code", "transfer_uncertain", "diagnostic_kind", "mismatched_generation", "complete_capture",
    "missing_process", "pid_unknown", "reserved_process", "closed_process", "uncertain_process",
    "wrong_resource_role"))
def test_paced_cancel_rejects_unadmitted_or_finished_window(monkeypatch: pytest.MonkeyPatch,
                                                         case: str) -> None:
    from dataclasses import replace
    from clipperx._source_host_ownership import HostFailure
    from clipperx.tests.test_source_host_ownership import finish_job
    def edit(status: Any) -> Any:
        current = status.current
        if case == "not_ready":
            c.session.cancel_ready = False
        elif case == "already_observed":
            c.session.cancel_observed = True
        elif case == "job_failure":
            return replace(status, failure=HostFailure("validation", "driver", 1, 100, None, 0, None))
        elif case == "missing_current":
            return replace(status, current=None)
        elif case in ("other_kind", "not_active", "closed_scope", "not_consumer"):
            key, value = {"other_kind": ("kind", "encoder"), "not_active": ("phase", "validating"),
                          "closed_scope": ("scope_open", False), "not_consumer": ("consumer_active", False)}[case]
            current = replace(current, **{key: value})
        elif case == "no_diagnostic":
            current = replace(current, diagnostics=())
        elif case == "two_diagnostics":
            current = replace(current, diagnostics=current.diagnostics*2)
        elif case in ("natural_eof", "exit_code", "transfer_uncertain", "diagnostic_kind", "mismatched_generation",
                      "complete_capture"):
            diagnostic = current.diagnostics[0]
            key, value = {"natural_eof": ("natural_eof", True), "exit_code": ("exit_code", 0),
                "transfer_uncertain": ("transfer_uncertain", True), "diagnostic_kind": ("kind", "encoder"),
                "mismatched_generation": ("generation", diagnostic.generation+1),
                "complete_capture": ("stdout_bytes", 122880)}[case]
            current = replace(current, diagnostics=(replace(diagnostic, **{key: value}),))
        else:
            process = next(row for row in current.resources if row.role == "process")
            resources = tuple(row for row in current.resources if row is not process)
            if case != "missing_process":
                key, value = {"pid_unknown": ("pid", None), "reserved_process": ("phase", "reserved"),
                    "closed_process": ("phase", "closed"), "uncertain_process": ("uncertain", True),
                    "wrong_resource_role": ("role", "stdout")}[case]
                resources += (replace(process, **{key: value}),)
            current = replace(current, resources=resources)
        return replace(status, current=current)
    c = fixture_adapter(monkeypatch, cancel=True, cancel_status=edit)
    c.session.run_driver(lambda: c.session.paced_cancellation(c.source))
    assert len(c.snapshots) == 1 and c.stops == []
    assert c.session.cancel_observed is (case == "already_observed")
    assert "cancellation_window" not in c.session.state
    assert c.files.snapshot().files_clear and finish_job(c.h).own0


@pytest.mark.parametrize("phase,alive", (("live", False), ("in_call", None)))
def test_paced_cancel_admission_uses_process_phase_not_alive(monkeypatch: pytest.MonkeyPatch,
                                                           phase: str, alive: bool | None) -> None:
    from dataclasses import replace
    from clipperx.tests.test_source_host_ownership import finish_job
    from scripts import render_visual_smoke as smoke
    def edit(status: Any) -> Any:
        current = status.current
        return replace(status, current=replace(current, resources=tuple(
            replace(row, phase=phase, alive=alive) if row.role == "process" else row
            for row in current.resources)))
    c = fixture_adapter(monkeypatch, cancel=True, cancel_status=edit)
    c.session.run_driver(lambda: c.session.paced_cancellation(c.source))
    assert c.stops == ["cancelled"] and c.session.cancel_observed
    assert c.files.snapshot().files_clear and finish_job(c.h).own0
    assert c.session.finish_result(cancel=True) is None
    window = c.session.state["cancellation_window"]
    assert next(row for row in window.resources if row.role == "process").phase == phase
    assert len(smoke.json_bytes(c.session.campaign.records[-1])) < 8*1024**2


@pytest.mark.parametrize("case", ("missing_capture", "missing_terminal", "observation_id", "generation",
    "captured_kind", "terminal_kind", "captured_no_diagnostic", "terminal_no_diagnostic",
    "captured_two_diagnostics", "terminal_two_diagnostics", "captured_diagnostic_kind",
    "terminal_diagnostic_kind", "captured_complete", "terminal_complete", "captured_natural_eof",
    "terminal_natural_eof", "captured_exit", "captured_uncertain"))
def test_cancel_terminal_requires_same_incomplete_window(monkeypatch: pytest.MonkeyPatch,
                                                       case: str) -> None:
    from dataclasses import replace
    from clipperx._source_host_ownership import HostedObservationFailure
    from clipperx.tests.test_source_host_ownership import finish_job
    from scripts import render_visual_smoke as smoke
    c = fixture_adapter(monkeypatch, cancel=True)
    c.session.run_driver(lambda: c.session.paced_cancellation(c.source))
    error = c.session.state["error"]
    assert isinstance(error, HostedObservationFailure) and error.failure.code == "cancelled"
    status = finish_job(c.h)
    assert status.own0 and c.files.snapshot().files_clear
    assert status.current.diagnostics[0].stdout_bytes < 122880 and c.stops == ["cancelled"]
    if case == "missing_capture":
        c.session.state.pop("cancellation_window")
    elif case == "missing_terminal":
        status = replace(status, current=None)
    else:
        captured = case.startswith("captured_")
        current = c.session.state["cancellation_window"] if captured else status.current
        field = case.removeprefix("captured_").removeprefix("terminal_")
        if field == "observation_id":
            current = replace(current, observation_id=current.observation_id+1)
        elif field == "kind":
            current = replace(current, kind="encoder")
        elif field == "no_diagnostic":
            current = replace(current, diagnostics=())
        elif field == "two_diagnostics":
            current = replace(current, diagnostics=current.diagnostics*2)
        else:
            diagnostic = current.diagnostics[0]
            key, value = {"generation": ("generation", diagnostic.generation+1),
                "diagnostic_kind": ("kind", "encoder"), "complete": ("stdout_bytes", 122880),
                "natural_eof": ("natural_eof", True), "exit": ("exit_code", 0),
                "uncertain": ("transfer_uncertain", True)}[field]
            current = replace(current, diagnostics=(replace(diagnostic, **{key: value}),))
        if captured:
            c.session.state["cancellation_window"] = current
        else:
            status = replace(status, current=current)
    monkeypatch.setattr(c.session.job, "snapshot", lambda: status)
    with pytest.raises(smoke.QualificationFailure) as caught:
        c.session.finish_result(cancel=True)
    assert caught.value.__cause__ is error
    evidence = caught.value.evidence[-1]
    assert evidence["cancellation"]["classification"] == "ineligible_window"
    assert evidence["cancellation"]["terminal"] is status.current
    assert evidence["owner"].own0 and evidence["files"].files_clear


def test_cancelled_aborted_read_accepts_terminal_transfer_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx._source_host_ownership import HostedObservationFailure
    from clipperx.tests.test_source_host_ownership import finish_job
    c = fixture_adapter(monkeypatch, cancel=True, cancel_abort_read=True)
    c.session.run_driver(lambda: c.session.paced_cancellation(c.source))
    error = c.session.state["error"]
    assert isinstance(error, HostedObservationFailure) and error.failure.code == "cancelled"
    status = finish_job(c.h)
    diagnostic = status.current.diagnostics[0]
    assert diagnostic.stdout_bytes == 0 and diagnostic.stdout_bytes < 122880
    assert diagnostic.transfer_uncertain and not diagnostic.natural_eof
    assert status.failure.code == "cancelled" and status.own0 and c.files.snapshot().files_clear
    assert c.stops == ["cancelled"]
    captured = c.session.state["cancellation_window"]
    assert captured is c.snapshots[0].current and not captured.diagnostics[0].transfer_uncertain
    assert c.session.finish_result(cancel=True) is None
    evidence = c.session.campaign.records[-1]["cancellation"]
    assert evidence["classification"] == "observed_cancelled" and evidence["incomplete_payload"]
    assert evidence["window"] is captured and evidence["terminal"].diagnostics[0].transfer_uncertain


def tiny_encoded_samples(*, corrupt: str = "") -> list[Any]:
    # Independent literal decoded medians and ordinals, not recipe/compiler output.
    colors = ((32, 64, 192), (33, 66, 191), (32, 48, 64), (32, 48, 64), (40, 80, 184), (41, 82, 183))
    ordinals = (0, 1, None, None, 8, 9)
    result = []
    for k, (base, ordinal) in enumerate(zip(colors, ordinals)):
        patches = {(500, 1000): base}
        for bit in range(8):
            patches[(120*(bit+1), 1500)] = (255, 255, 255) if ordinal is not None and ordinal & (1 << bit) else (0, 0, 0)
        fore = ((255, 0, 0), (20, 168, 92) if k == 4 else (20, 169, 91), base, (255, 255, 255))
        for center, value in zip(((702, 714), (810, 714), (702, 822), (810, 822)), fore):
            patches[center] = value if k >= 4 else base
        if corrupt == "ordinal" and k == 0:
            patches[(120, 1500)] = (255, 255, 255)
        if corrupt == "color" and k == 1:
            patches[(500, 1000)] = (46, 66, 191)
        if corrupt == "early" and k == 3:
            patches[(702, 714)] = (255, 0, 0)
        if corrupt == "missing" and k == 5:
            patches[(810, 714)] = base
        def pixel(x: int, y: int, values: Any = patches, default: Any = base) -> Any:
            return next((rgb for (cx, cy), rgb in values.items() if abs(x-cx) <= 3 and abs(y-cy) <= 3), default)
        result.append((k, pixel))
    return result


def test_tiny_public_encoded_pixels_are_independent_and_sensitive() -> None:
    _, o, _ = modules()
    records = o.tiny_encoded_pixels(tiny_encoded_samples())
    assert tuple(row["source_ordinal"] for row in records) == (0, 1, None, None, 8, 9)
    assert records[4]["png_active"] and records[5]["png_active"]
    for corruption in ("ordinal", "color", "early", "missing"):
        with pytest.raises(ValueError):
            o.tiny_encoded_pixels(tiny_encoded_samples(corrupt=corruption))
    with pytest.raises(ValueError):
        o.tiny_encoded_pixels(tiny_encoded_samples()[:-1])
    swapped = tiny_encoded_samples()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError):
        o.tiny_encoded_pixels(swapped)


def test_tiny_all_solid_encoded_probes_stay_inactive_at_new_coordinates() -> None:
    _, o, _ = modules()
    def solid(x: int, y: int) -> tuple[int, int, int]:
        return 32, 48, 64
    records = o.tiny_encoded_pixels(((0, solid), (1, solid)), all_solid=True)
    assert [row["source_ordinal"] for row in records] == [None, None]
    assert [row["png_active"] for row in records] == [False, False]
    for cx, cy in ((702, 714), (810, 714), (702, 822), (810, 822)):
        def unexpected_png(x: int, y: int) -> tuple[int, int, int]:
            return (255, 0, 0) if abs(x-cx) <= 2 and abs(y-cy) <= 2 else (32, 48, 64)
        with pytest.raises(ValueError, match="color/alpha observation mismatch"):
            o.tiny_encoded_pixels(((0, solid), (1, unexpected_png)), all_solid=True)


def test_overlay_boundary_schedule_is_literal_and_finite() -> None:
    m, _, _ = modules()
    batches = m.overlay_boundary_batches()
    assert [(row.name, row.frames) for row in batches] == [
        ("subtitle_s1", (11, 12, 13, 71, 72, 73)),
        ("subtitle_s3", (281, 282, 283, 341, 342, 343)),
        ("brand_s1", (29, 30, 31, 119, 120, 121)),
        ("brand_s2", (149, 150, 151, 269, 270, 271)),
        ("brand_s3", (284, 285, 286, 464, 465, 466))]
    assert [row.active for row in batches] == [(False, True, True, True, False, False)]*5
    assert len(m.DIAGNOSTIC_K) == 10 and len(batches) == 5


def boundary_contact_sheet(name: str, *, corrupt_activation: bool = False) -> Any:
    import zlib
    literals = {
        "subtitle_s1": (864, 192, (11, 12, 13, 71, 72, 73)),
        "subtitle_s3": (864, 192, (281, 282, 283, 341, 342, 343)),
        "brand_s1": (130, 154, (29, 30, 31, 119, 120, 121)),
        "brand_s2": (130, 154, (149, 150, 151, 269, 270, 271)),
        "brand_s3": (130, 154, (284, 285, 286, 464, 465, 466))}
    width, height, frames = literals[name]
    pictures, lines = [], ["[showinfo@qoverlay @ 00] config in time_base: 1/60, frame_rate: 60/1"]
    for n, k in enumerate(frames):
        active = n in (1, 2, 3) and not (corrupt_activation and n == 1)
        picture = bytearray(bytes((10, 20, 30))*width*height)
        patches = ((30, 30, (255, 213, 26)),) if name.startswith("subtitle") else (
            (32, 44, (255, 0, 0)), (97, 109, (255, 255, 255)))
        if active:
            for cx, cy, rgb in patches:
                for y in range(cy-3, cy+4):
                    for x in range(cx-3, cx+4):
                        j = (y*width+x)*3
                        picture[j:j+3] = bytes(rgb)
        pictures.append(bytes(picture))
        lines.append(f"[showinfo@qoverlay @ 00] n:{n} pts:{k} fmt:rgb24 s:{width}x{height} checksum:{zlib.adler32(picture, 0):08X}")
    tiled = b"".join(frame[y*width*3:(y+1)*width*3] for y in range(height) for frame in pictures)
    return tiled, width*6, height, ("\n".join(lines)+"\n").encode()


@pytest.mark.parametrize("name", ("subtitle_s1", "subtitle_s3", "brand_s1", "brand_s2", "brand_s3"))
def test_overlay_crops_associate_bytes_tags_labels_and_activation(name: str) -> None:
    _, o, _ = modules()
    rgb, width, height, tail = boundary_contact_sheet(name)
    measured = o.overlay_contact_sheet(name, rgb, width, height, tail, truncated=False)
    assert len(measured) == 6
    assert [row["expected_active"] for row in measured] == [False, True, True, True, False, False]
    assert all(row["observed_active"] == row["expected_active"] for row in measured)
    for bad_rgb, bad_width, bad_tail, truncated in (
            (rgb[:-1], width, tail, False), (rgb, width+1, tail, False),
            (rgb, width, tail.replace(b"n:1 ", b"n:0 "), False),
            (rgb, width, tail, True)):
        with pytest.raises(ValueError):
            o.overlay_contact_sheet(name, bad_rgb, bad_width, height, bad_tail, truncated=truncated)
    corrupt, cw, ch, valid_tags = boundary_contact_sheet(name, corrupt_activation=True)
    with pytest.raises(ValueError, match="activation"):
        o.overlay_contact_sheet(name, corrupt, cw, ch, valid_tags, truncated=False)


def test_overlay_crop_controls_fit_current_fixed_synthetic_count() -> None:
    m, _, s = modules()
    assert s.oracle_child_reservation("synthetic", 10) == 22
    assert 2+39+s.oracle_child_reservation("synthetic", 10) == 63
    assert s.oracle_child_reservation("tiny", 6) == 10
    assert s.oracle_child_reservation("all-solid", 2) == 6
    graph = s.overlay_crop_graph(m.overlay_boundary_batches()[0])
    assert "crop=864:192:108:1574" in graph and "showinfo@qoverlay" in graph
    assert "tile=layout=6x1:nb_frames=6:padding=0:margin=0" in graph
    assert all(f"eq(n\\,{k})" in graph for k in (11, 12, 13, 71, 72, 73))
    assert "fps=" not in graph and "setpts=" not in graph


@pytest.mark.parametrize("parent_seconds", (10, 90))
def test_fixture_float_creates_real_bounded_observer(monkeypatch: pytest.MonkeyPatch,
                                                    parent_seconds: int) -> None:
    import json
    import zlib
    from pathlib import Path
    from clipperx.tests import media_factory as m
    from clipperx.tests.test_source_host_ownership import ScriptProcess, finish_job
    from clipperx._source_host_ownership import HostedObservationFailure
    from scripts import render_visual_smoke as smoke
    c = fixture_adapter(monkeypatch, parent_seconds=parent_seconds)
    s = c.session
    original = c.h.job._processes
    raw = m.float48_pcm(1)
    attempted: list[tuple[str, ...]] = []
    stopped: list[str] = []
    def stop_before_observer_launch(operation: str, path: Path) -> None:
        control = s.job.snapshot().current
        if (operation == "read" and path == s.runtime.ffmpeg and control is not None
                and control.kind == "materialized_f32" and not stopped):
            stopped.append("real float control admitted before fake input-pin read")
            s.job.request_stop("cancelled")
    c.fs.hook = stop_before_observer_launch
    def factory(argv: tuple[str, ...]) -> Any:
        attempted.append(argv)
        if argv[0] == str(s.runtime.ffprobe):
            return original(argv)
        if argv[-1].endswith(".mov"):
            c.fs.data[Path(argv[-1])] = b"finite MOV fixture"
            child = ScriptProcess(b"", b"")
        elif argv[-1].endswith(".reference.f32"):
            c.fs.data[Path(argv[-1])] = raw
            tags = []
            for n, offset in enumerate(range(0, 96000, 1024)):
                block = raw[offset*4:min(offset+1024, 96000)*4]
                tags.append(f"[ashowinfo@qref @ 00] n:{n} pts:{60000+offset} fmt:flt channels:1 "
                            f"rate:48000 nb_samples:{len(block)//4} checksum:{zlib.adler32(block, 0):08X}")
            child = ScriptProcess(b"", ("\n".join(tags)+"\n").encode())
        else:
            raise AssertionError("observer child must not start after fake pre-launch stop")
        c.h.children.append(child)
        return child
    c.h.job._processes = factory
    with pytest.raises(HostedObservationFailure) as caught:
        smoke.fixture_float(s, 48000, 1)
    assert caught.value.failure.code == "cancelled" and len(stopped) == 1
    control = s.job.snapshot().current
    assert control is not None and control.kind == "materialized_f32"
    assert control.deadline_ns == c.h.now+min(parent_seconds, 45)*10**9
    assert len(attempted) == 3  # full reference/planner/control reached; no observer child
    assert attempted[0] == (str(s.runtime.ffmpeg), "-hide_banner", "-nostats", "-loglevel", "info", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "16x16", "-framerate", "60",
        "-i", str(s.root/"float48000-1.rgb"), "-f", "f32le", "-ar", "48000", "-ac", "1",
        "-i", str(s.root/"float48000-1.f32"), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "qtrle",
        "-c:a", "pcm_f32le", "-output_ts_offset", "1.25", "-use_editlist", "1",
        "-avoid_negative_ts", "disabled", str(s.root/"float48000-1.mov"))
    assert "-af" in attempted[-1] and json.loads(s.records[1]["result"].stdout)["format"]["start_time"] == "1.25"
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


def test_float_metadata_origin_is_retained_before_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from pathlib import Path
    from clipperx.tests.test_source_host_ownership import ScriptProcess, finish_job
    from scripts import render_visual_smoke as smoke
    c = fixture_adapter(monkeypatch)
    original = c.h.job._processes
    def process(argv: tuple[str, ...]) -> Any:
        if argv[0] != str(c.session.runtime.ffprobe) and argv[-1].endswith(".mov"):
            c.fs.data[Path(argv[-1])] = literal_aac_mov()
            child = ScriptProcess(b"", b"")
            c.h.children.append(child)
            return child
        return original(argv)
    c.h.job._processes = process
    with pytest.raises(ValueError, match="UNVERIFIED MOV recipe"):
        smoke.fixture_float(c.session, 44100, 1)
    path = c.session.root/"float44100-1.origin.json"
    record = json.loads(c.fs.data[path])
    assert record["origin"] == "5/4" and len(record["streams"]) == 2
    assert path in c.session.keep
    assert next(p for p in c.files.snapshot().paths if p.path == str(path)).role == "sidecar"
    assert c.files.retire(retain=tuple(c.session.keep)).files_clear
    assert finish_job(c.h).own0


@pytest.mark.parametrize("unsupported", (False, True))
def test_aac_fixture_retained_preparation_precedes_single_truthful_probe(
        monkeypatch: pytest.MonkeyPatch, unsupported: bool) -> None:
    import hashlib
    import json
    from pathlib import Path
    from clipperx.tests.test_source_host_ownership import ScriptProcess, finish_job
    from scripts import render_visual_smoke as smoke
    m, _, _ = modules()
    c = fixture_adapter(monkeypatch)
    s = c.session
    original = c.h.job._processes
    attempted: list[tuple[str, ...]] = []
    zero = s.root/"float44100-2.zero.mov"
    final = s.root/"float44100-2.mov"
    sidecar = s.root/"float44100-2.preparation.json"
    at_probe: list[tuple[dict[Any, bytes], tuple[Any, ...]]] = []
    producer_rows: list[Any] = []
    def process(argv: tuple[str, ...]) -> Any:
        attempted.append(argv)
        if argv[0] == str(s.runtime.ffprobe):
            at_probe.append((dict(c.fs.data), tuple(s.keep)))
            return original(argv)  # truthful 48k mono metadata; original predicate must reject
        producer_rows.append(next(row for row in c.files.snapshot().paths if row.path == argv[-1]))
        c.fs.data[Path(argv[-1])] = b"unsupported layout" if unsupported else literal_aac_mov()
        child = ScriptProcess(b"", b"")
        c.h.children.append(child)
        return child
    c.h.job._processes = process
    with pytest.raises(ValueError, match="AAC fixture" if unsupported else "UNVERIFIED MOV recipe"):
        smoke.fixture_float(s, 44100, 2)
    assert attempted[0] == (str(s.runtime.ffmpeg), "-hide_banner", "-nostats", "-loglevel", "info", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "16x16", "-framerate", "60",
        "-i", str(s.root/"float44100-2.rgb"), "-f", "f32le", "-ar", "44100", "-ac", "2",
        "-i", str(s.root/"float44100-2.f32"), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "qtrle",
        "-c:a", "aac", "-output_ts_offset", "0", "-use_editlist", "1",
        "-avoid_negative_ts", "disabled", str(zero))
    assert producer_rows[0].role == "asset" and producer_rows[0].max_bytes == 16*1024*1024
    assert c.fs.data[s.root/"float44100-2.f32"] == m.float44_pcm()
    assert c.fs.data[s.root/"float44100-2.rgb"] == m.tiny_rgb(180)
    assert len(attempted) == (1 if unsupported else 2)
    assert zero in s.keep and zero in c.fs.data
    if unsupported:
        assert final not in c.fs.data and sidecar not in c.fs.data
    else:
        snapshot, kept = at_probe[0]
        assert attempted[1][-1] == str(final) and snapshot[final] == literal_aac_mov(shifted=True)
        facts = json.loads(snapshot[sidecar])
        assert facts["input_sha256"] == hashlib.sha256(literal_aac_mov()).hexdigest()
        assert facts["output_sha256"] == hashlib.sha256(snapshot[final]).hexdigest()
        assert facts["growth_bytes"] == 24 and facts["offset_seconds"] == "5/4"
        assert "origin" not in facts  # no measured metadata invented by preparation
        assert {zero, final, sidecar} <= set(kept)
        assert json.loads(c.fs.data[s.root/"float44100-2.origin.json"])["streams"][1]["sample_rate"] == 48000
        roles = {row.path: row for row in c.files.snapshot().paths}
        assert roles[str(final)].role == "asset" and roles[str(final)].max_bytes == 16*1024*1024
        assert roles[str(sidecar)].role == "sidecar"
    assert c.files.retire(retain=tuple(s.keep)).files_clear
    assert finish_job(c.h).own0


@pytest.mark.parametrize("cleanup", ("none", "retire-contract", "delete"))
def test_driver_and_cleanup_causes_survive_cancel_wrapper(monkeypatch: pytest.MonkeyPatch,
                                                         cleanup: str) -> None:
    from clipperx.tests.test_source_host_ownership import finish_job
    from scripts import render_visual_smoke as smoke
    c = fixture_adapter(monkeypatch)
    s = c.session
    original = TypeError("original missing deadline argument before cancellation")
    delete_errors: list[PermissionError] = []
    scratch = s.put("source.f32", b"abcd", role="pcm")
    if cleanup == "retire-contract":
        s.keep.append(scratch)  # intentionally forbidden real retirement role
    elif cleanup == "delete":
        action = c.files._path_action
        def fail_delete(operation: str, path: Any, destination: Any) -> Any:
            if operation == "unlink" and path == scratch:
                error = PermissionError(13, "precise trusted fake cleanup failure")
                error.winerror = 5
                delete_errors.append(error)
                raise error
            return action(operation, path, destination)
        c.files._path_action = s.record_path_action(fail_delete)
    def work() -> None:
        raise original
    s.run_driver(work)
    assert finish_job(c.h).own0
    with pytest.raises(smoke.QualificationFailure) as caught:
        s.finish_result(cancel=True)
    assert caught.value.__cause__ is original
    evidence = caught.value.evidence[-1]
    assert evidence["driver_error"]["type"] == "TypeError"
    assert evidence["driver_error"]["message"] == str(original)
    assert evidence["driver_error"]["frames"][-1]["function"] == "work"
    assert evidence["cancellation"]["classification"] == "not_reached"
    assert evidence["owner"].own0
    if cleanup == "none":
        assert evidence["cleanup_error"] is None and evidence["files"].files_clear
    elif cleanup == "retire-contract":
        assert evidence["cleanup_error"]["type"] == "ValueError"
        assert "settled owned output" in evidence["cleanup_error"]["message"]
        assert not evidence["files"].files_clear
    else:
        record = evidence["cleanup_actions"][0]
        assert record["operation"] == "unlink" and record["errno"] == 13 and record["winerror"] == 5
        assert len(delete_errors) == 1 and record["error"]["message"] == str(delete_errors[0])
        assert record["error"]["frames"][-1]["function"] == "fail_delete"
        assert not evidence["files"].files_clear
        assert sum(op == "unlink" and path == scratch for op, path in c.fs.events) == 0


def test_expected_cancel_is_separate_from_unreached_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.tests.test_source_host_ownership import finish_job
    c = fixture_adapter(monkeypatch, cancel=True)
    c.session.run_driver(lambda: c.session.paced_cancellation(c.source))
    status = finish_job(c.h)
    assert status.own0
    assert status.current.diagnostics[0].stdout_bytes < 122880
    assert not status.current.diagnostics[0].natural_eof and c.stops == ["cancelled"]
    assert c.session.finish_result(cancel=True) is None
    assert c.session.campaign.records[-1]["cancellation"]["classification"] == "observed_cancelled"
    assert c.session.campaign.records[-1]["cancellation"]["window"] is c.snapshots[0].current
    assert c.session.campaign.records[-1]["cancellation"]["terminal"] == status.current
    assert c.session.campaign.records[-1]["cancellation"]["incomplete_payload"]


@pytest.mark.parametrize("observed", (False, True))
def test_natural_eof_cannot_become_successful_cancel(monkeypatch: pytest.MonkeyPatch,
                                                  observed: bool) -> None:
    import io
    from clipperx.tests.test_source_host_ownership import finish_job
    from scripts import render_visual_smoke as smoke
    c = fixture_adapter(monkeypatch)
    original = c.h.job._processes
    def complete(argv: tuple[str, ...]) -> Any:
        child = original(argv)
        child.stdout = io.BytesIO(bytes(120*16*16*4))
        return child
    c.h.job._processes = complete
    c.session.cancel_observed = observed
    c.session.run_driver(lambda: c.session.paced_cancellation(c.source))
    assert c.stops == [] and c.session.cancel_ready
    assert isinstance(c.session.state["error"], ValueError)
    assert "natural EOF is not cancellation evidence" in str(c.session.state["error"])
    status = finish_job(c.h)
    assert status.own0 and c.files.snapshot().files_clear
    assert status.current.diagnostics[0].natural_eof and status.current.diagnostics[0].exit_code == 0
    with pytest.raises(smoke.QualificationFailure) as caught:
        c.session.finish_result(cancel=True)
    assert caught.value.__cause__ is c.session.state["error"]
    assert caught.value.evidence[-1]["cancellation"]["classification"] == (
        "unexpected_cause" if observed else "window_not_observed")


def test_fixture_durable_purposes_transfer_and_scratch_retires(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path
    from clipperx.tests.test_source_host_ownership import ScriptProcess, finish_job
    c = fixture_adapter(monkeypatch)
    s = c.session
    at_launch = []
    def producer(argv: tuple[str, ...]) -> Any:
        path = Path(argv[-1])
        at_launch.append(next(row.role for row in c.files.snapshot().paths if row.path == str(path)))
        c.fs.data[path] = b"trusted fake file-output bytes"
        child = ScriptProcess(b"", b"")
        c.h.children.append(child)
        return child
    c.h.job._processes = producer
    for name in ("fixture.mov", "fixture.mp4", "diagnostic.png"):
        s.file_output(name, (str(s.runtime.ffmpeg), "finite"), (), retain=True)
    s.put("tiny.json", b"{}", retain=True)
    s.put("transcript.json", b"[]", retain=True)
    pcm = s.put("scratch.f32", b"abcd", role="pcm")
    raw = s.put("scratch.rgba", b"abcd", role="raw")
    negative, _ = s.file_output("negative.mp4", (str(s.runtime.ffmpeg), "finite"), ())
    assert c.files.retire(retain=tuple(s.keep)).files_clear
    roles = {Path(row.path).name: row.role for row in c.files.snapshot().paths}
    assert at_launch == ["asset", "asset", "asset", "temporary"]
    assert roles["tiny.json"] == roles["transcript.json"] == "sidecar"
    assert all(path not in c.fs.data for path in (pcm, raw, negative))
    assert all(path in c.fs.data for path in s.keep)
    assert finish_job(c.h).own0


def test_unknown_retained_purpose_fails_before_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    c = fixture_adapter(monkeypatch)
    for name, role in (("unknown.bin", None), ("wrong.json", "temporary"), ("bad.f32", "pcm")):
        with pytest.raises(ValueError, match="purpose"):
            if role is None:
                c.session.put(name, b"data", retain=True)
            else:
                c.session.put(name, b"data", role=role, retain=True)
        assert c.session.root/name not in c.fs.data
    assert c.files.retire().files_clear
    from clipperx.tests.test_source_host_ownership import finish_job
    assert finish_job(c.h).own0


@pytest.mark.parametrize("ready,observed,label", ((False, False, "not_reached"),
    (True, False, "window_not_observed"), (True, True, "unexpected_cause")))
def test_unreached_missed_wrong_cause_never_become_expected_cancel(monkeypatch: pytest.MonkeyPatch,
        ready: bool, observed: bool, label: str) -> None:
    from clipperx.tests.test_source_host_ownership import finish_job
    from scripts import render_visual_smoke as smoke
    c = fixture_adapter(monkeypatch)
    c.session.cancel_ready, c.session.cancel_observed = ready, observed
    error = RuntimeError("concrete earlier failure remains first")
    def work() -> None:
        raise error
    c.session.run_driver(work)
    assert finish_job(c.h).own0
    with pytest.raises(smoke.QualificationFailure) as caught:
        c.session.finish_result(cancel=True)
    assert caught.value.__cause__ is error
    assert caught.value.evidence[-1]["cancellation"]["classification"] == label


def test_cleanup_callback_evidence_is_bounded_and_delegates_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import render_visual_smoke as smoke
    from clipperx.tests.test_source_host_ownership import finish_job
    c = fixture_adapter(monkeypatch)
    s = c.session
    failure = OSError(13, "x"*5000)
    calls: list[str] = []
    def external(operation: str, path: Any, destination: Any) -> object:
        calls.append(operation)
        raise failure
    wrapper = s.record_path_action(external)
    s._retiring = True
    for _ in range(10):
        with pytest.raises(OSError) as caught:
            wrapper("unlink", s.root/"scratch.f32", None)
        assert caught.value is failure
    with pytest.raises(OSError):
        wrapper("lstat", s.root/"missing", None)
    s._retiring = False
    assert len(calls) == 11 and s.state["cleanup_action_count"] == 10
    assert len(s.state["cleanup_actions"]) == 8
    assert all(len(row["error"]["message"]) == 2048 and len(row["error"]["frames"]) <= 8
               for row in s.state["cleanup_actions"])
    assert len(smoke.json_bytes(s.state["cleanup_actions"])) < 8*1024**2
    assert c.files.retire().files_clear and finish_job(c.h).own0


@pytest.mark.parametrize("count", (0, 1, 8, 9, 16, 32, 33))
def test_error_details_truncation_boundaries(count: int) -> None:
    import sys
    from types import TracebackType
    from scripts.render_visual_smoke import error_details
    frame = sys._getframe()
    trace: TracebackType | None = None
    for lineno in range(count, 0, -1):
        trace = TracebackType(trace, frame, frame.f_lasti, lineno)
    error = ValueError("literal bounded trace")
    cause = KeyError("unchanged cause")
    error.__traceback__, error.__cause__ = trace, cause
    record = error_details(error)
    traversed = min(count, 32)
    assert record["trace_truncated"] is (count > 8)
    assert [row["lineno"] for row in record["frames"]] == list(range(max(1, traversed-7), traversed+1))
    assert len(record["frames"]) <= 8
    assert record["type"] == "ValueError" and record["message"] == "literal bounded trace"
    assert record["cause"] == {"type": "KeyError", "message": "'unchanged cause'"}
    assert error.__traceback__ is trace and error.__cause__ is cause

