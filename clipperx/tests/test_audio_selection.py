"""Tiny, pure actual42 byte/metadata selection gates; no host fixtures."""

from fractions import Fraction as F
from dataclasses import replace
import struct
import json
import subprocess
import threading
from typing import Any
import zlib

import pytest

import clipperx.audio_selection as selection_module
import clipperx.render_clock as clock_module
import clipperx.source_decode as decode_module

from clipperx.audio_selection import (
    AudioSelectionLimits, apply_float_audio_group, map_float_audio_group,
    plan_float_audio_selection,
)
from clipperx.audio_mix import AudioPreparationError
from clipperx.render_clock import (
    SourceFloatAudioDecodeReceipt, SourceTimeOrigin, StreamClock, digest,
    float_audio_decode_argv, source_decoder_identity,
    rational, canonical,
)
from clipperx.source_decode import DecodedFloatAudioWindow, FloatAudioDecodeLimits
from clipperx.tests.test_visual_contracts import make_plan, make_shot
from clipperx.visual_contracts import VisualCompositionPlan


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> Any:
        raise AssertionError("No child/thread/timer/native/source operation is authorized")

    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(threading.Thread, "start", fail)
    monkeypatch.setattr(threading.Timer, "start", fail)
    monkeypatch.setattr(decode_module, "_check_observation_source", fail)
    monkeypatch.setattr(decode_module, "_runtime_identity", fail)


def tiny_case(channels: int) -> tuple[
    VisualCompositionPlan, SourceTimeOrigin, dict[str, object], FloatAudioDecodeLimits,
    SourceFloatAudioDecodeReceipt, DecodedFloatAudioWindow, bytes,
]:
    runtime: dict[str, object] = dict(runtime_id="HOST", ffmpeg="O:/HOST/ffmpeg.exe",
        ffprobe="O:/HOST/ffprobe.exe", ffmpeg_sha256="A"*64, ffprobe_sha256="B"*64)
    origin = SourceTimeOrigin("O:/HOST/float.nut", 16, 1, "0123456789abcdef",
        F(0), F(1), (
            StreamClock(0, "video", F(1, 60), F(0), F(1), width=1, height=1),
            StreamClock(1, "audio", F(1, 48000), F(0), F(1), sample_rate=48000,
                        channels=channels, sample_format="flt")),
        source_decoder_identity(runtime), {})
    dl = FloatAudioDecodeLimits(preroll=F(0), postroll=F(0), max_window=F(1),
        timeout=1, max_blocks=2, max_samples=8, max_block_samples=8,
        max_decoded_bytes=32, max_retained_bytes=32, max_log_bytes=4096,
        max_metadata_bytes=8192, io_chunk_bytes=16)
    pcm = (bytes.fromhex("00000000 0000803f 000080bf 00008035") if channels == 1
        else bytes.fromhex("00000000 00000080 0000803f 01000000 "
                           "000080bf 0000c03f 00008035 000000b5"))
    policy = "render48k-f32-mono-v1" if channels == 1 else "render48k-f32-stereo-v1"
    block = dict(n=0, pts=0, time_base="1/48000", rate=48000, channels=channels,
        samples=4, format="flt", checksum=f"{zlib.adler32(pcm, 0):08X}", stage="cxain")
    window = ["0/1", "1/12500"]
    audio = dict(policy=policy, input=[block], output=[dict(block, stage="cxaout",
        bytes=len(pcm), sha256=digest(pcm))], samples=4, bytes=len(pcm),
        pcm_sha256=digest(pcm), sample_grid_origin="0/1", source_phase="0/1",
        coverage=[dict(window=window, status="covered")])
    payload = dict(windows=[window], limits=dict(dl.record(), decoder_start="0/1",
        decoder_end="1/12500"), audio=[audio], argv=[float_audio_decode_argv(
            str(runtime["ffmpeg"]), origin, F(0), F(1, 12500), policy)])
    receipt = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=dl.record())
    backing = bytearray(pcm)
    observed = DecodedFloatAudioWindow(receipt, backing)
    backing[:] = bytes(len(backing))
    plan = make_plan(source_fingerprint=origin.fingerprint, shots=[make_shot(
        source={"start": 0.0, "end": 0.00008}, output={"start": 0.0, "end": 0.00008})])
    return plan, origin, runtime, dl, receipt, observed, pcm


@pytest.mark.parametrize("channels", [1, 2])
def test_selection_real42_mono_stereo_bytes(channels: int) -> None:
    plan, origin, runtime, dl, receipt, observed, pcm = tiny_case(channels)
    limits = AudioSelectionLimits()
    selection = plan_float_audio_selection(plan, origin=origin, runtime=runtime, limits=limits)
    mapping = map_float_audio_group(plan, 0, receipt, origin=origin, runtime=runtime,
                                    decode_limits=dl, limits=limits)
    selected = apply_float_audio_group(plan, mapping, observed, origin=origin, runtime=runtime,
                                     decode_limits=dl, limits=limits)
    assert selection.total_samples == 4
    assert mapping.selected_samples == 4 and mapping.selected_bytes == 16*channels
    assert [(r.first, r.last, r.source_first, r.residual) for r in mapping.runs] == [(0, 4, 0, F(0))]
    assert not mapping.reused_first_interval
    assert selected.pcm == pcm == observed.pcm
    assert selected.pcm_sha256 == digest(pcm) and selected.map_sha256 == mapping.sha256
    assert mapping.selection_sha256 == selection.sha256
    assert selection.record()["schedule_id"] == selection.schedule.identity
    for n in range(4):
        request = F(n, 48000)
        assert F(0) <= request < F(1, 12500)
        assert F(n, 48000) <= request < F(n+1, 48000)
        assert selected.pcm[n*4*channels:(n+1)*4*channels] == pcm[n*4*channels:(n+1)*4*channels]
    detached = selection.record()
    detached["spans"] = []
    assert len(selection.spans) == 1
    # Pure audio projection deliberately excludes unrelated creative prose.
    changed = plan.model_copy(update={"shots": [plan.shots[0].model_copy(update={"reason": "other"})]})
    assert plan_float_audio_selection(changed, origin=origin, runtime=runtime,
                                      limits=limits).sha256 == selection.sha256


def group_case(a: F, s: F, length: F, *, offset: F = F(0),
               predecessor: int = 0) -> tuple[
    VisualCompositionPlan, SourceTimeOrigin, dict[str, object], FloatAudioDecodeLimits,
    SourceFloatAudioDecodeReceipt, DecodedFloatAudioWindow, bytes,
]:
    _, old_origin, runtime, _, _, _, _ = tiny_case(1)
    origin = SourceTimeOrigin(old_origin.path, 16, 1, old_origin.fingerprint, offset, F(1), (
        StreamClock(0, "video", F(1, 60), offset, offset+1, width=1, height=1),
        StreamClock(1, "audio", F(1, 48000), offset, offset+1,
                    sample_rate=48000, channels=1, sample_format="flt")), old_origin.decoder_id, {})
    e = s+length
    pts = ((offset+s)*48000).__floor__()-predecessor
    g = F(pts, 48000)
    count = -(-((offset+e-g)*48000).numerator // ((offset+e-g)*48000).denominator)
    pcm = b"".join(struct.pack("<f", n+0.25) for n in range(count))
    dl = FloatAudioDecodeLimits(preroll=F(1, 10000), postroll=F(1, 10000), max_window=F(1),
        timeout=1, max_blocks=2, max_samples=32, max_block_samples=32,
        max_decoded_bytes=128, max_retained_bytes=128, max_log_bytes=4096,
        max_metadata_bytes=8192, io_chunk_bytes=16)
    start, end = max(F(0), s-dl.preroll), min(F(1), e+dl.postroll)
    block = dict(n=0, pts=pts, time_base="1/48000", rate=48000, channels=1,
        samples=count, format="flt", checksum=f"{zlib.adler32(pcm, 0):08X}", stage="cxain")
    window = [rational(s), rational(e)]
    audio = dict(policy="render48k-f32-mono-v1", input=[block],
        output=[dict(block, stage="cxaout", bytes=len(pcm), sha256=digest(pcm))],
        samples=count, bytes=len(pcm), pcm_sha256=digest(pcm), sample_grid_origin=rational(g),
        source_phase=rational(g-offset), coverage=[dict(window=window, status="covered")])
    payload = dict(windows=[window], limits=dict(dl.record(), decoder_start=rational(start),
        decoder_end=rational(end)), audio=[audio], argv=[float_audio_decode_argv(
            str(runtime["ffmpeg"]), origin, start, end, "render48k-f32-mono-v1")])
    receipt = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=dl.record())
    shots = []
    if a:
        shots.append(make_shot("preceding", source={"start": .3, "end": float(F(3, 10)+a)},
                               output={"start": 0., "end": float(a)}))
    shots.append(make_shot("selected", source={"start": float(s), "end": float(e)},
        output={"start": float(a), "end": float(a+length)}))
    plan = make_plan(source_fingerprint=origin.fingerprint, shots=shots)
    return plan, origin, runtime, dl, receipt, DecodedFloatAudioWindow(receipt, pcm), pcm


def test_selection_groups_preserve_occurrences_and_processing_slices() -> None:
    plan, origin, runtime, _, _, _, _ = tiny_case(1)
    shots = [
        make_shot("visual1", source={"start": 0., "end": .000026},
                  output={"start": 0., "end": .000026}),
        make_shot("visual2", source={"start": .000026, "end": .00008},
                  output={"start": .000026, "end": .00008}),
        make_shot("flags", source={"start": .00008, "end": .00012},
                  output={"start": .00008, "end": .00012},
                  audio={"normalize_dialogue": False, "source_gain_db": -3., "limiter": False,
                         "cues": [{"asset_id": "cue_hit", "start": 0., "end": .00003}]}),
        make_shot("repeat", source={"start": 0., "end": .00008},
                  output={"start": .00012, "end": .00020}),
    ]
    plan = make_plan(source_fingerprint=origin.fingerprint, shots=shots)
    selection = plan_float_audio_selection(plan, origin=origin, runtime=runtime,
                                            limits=AudioSelectionLimits())
    assert [(g.shot_first, g.shot_last, g.schedule_first, g.schedule_last)
            for g in selection.groups] == [(0, 3, 0, 2), (3, 4, 2, 3)]
    assert [s.schedule_span_index for s in selection.spans] == [0, 0, 1, 2]
    assert [s.group_index for s in selection.spans] == [0, 0, 0, 1]
    assert [(s.first, s.last) for s in selection.spans] == [(0, 1), (1, 4), (4, 6), (6, 10)]
    assert selection.spans[1].start == F(26, 1000000)
    assert selection.schedule.cues[0].start == F(8, 100000)
    assert selection.schedule.cues[0].first == 4
    assert len(selection.schedule.spans) == 3 and selection.total_samples == 10
    # At n=1 the continuous group's request precedes the internal exact .000026 boundary.
    p, o, r, dl, receipt, observed, pcm = group_case(F(0), F(0), F(12, 100000))
    split = make_plan(source_fingerprint=o.fingerprint, shots=shots[:3])
    mapping = map_float_audio_group(split, 0, receipt, origin=o, runtime=r,
                                   decode_limits=dl, limits=AudioSelectionLimits())
    selected = apply_float_audio_group(split, mapping, observed, origin=o, runtime=r,
                                      decode_limits=dl, limits=AudioSelectionLimits())
    assert selected.pcm == pcm
    assert mapping.runs[0].source_first == 0 and mapping.selected_samples == 6
    for delta in (-F(1, 10**10), F(1, 10**10)):
        # OutputTimeline's representation tolerance remains; exact adjacency alone groups.
        shifted = make_plan(source_fingerprint=origin.fingerprint, shots=[
            make_shot("left", source={"start": 0., "end": .00008},
                      output={"start": 0., "end": .00008}),
            make_shot("right", source={"start": .5, "end": .50008},
                      output={"start": float(F(8, 100000)+delta), "end": float(F(16, 100000)+delta)}),
        ])
        accepted = plan_float_audio_selection(shifted, origin=origin, runtime=runtime,
                                              limits=AudioSelectionLimits())
        assert [row.schedule_span_index for row in accepted.spans] == [0, 1]
        assert len(accepted.groups) == 2 and accepted.spans[1].start == F(str(shifted.shots[1].output.start))


@pytest.mark.parametrize("a,s,length,offset,predecessor,run_count,reuse", [
    (F(26, 1000000), F(2, 1000000), F(60, 1000000), F(0), 0, 2, True),
    (F(26, 1000000), F(16, 1000000), F(60, 1000000), F(0), 0, 1, False),
    (F(26, 1000000), F(2, 1000000), F(10, 1000000), F(0), 0, 1, False),
    (F(1, 32000), F(1, 4), F(60, 1000000), F(2), 1, 1, False),
    (F(1, 32000), F(3, 4), F(60, 1000000), F(-2), 1, 1, False),
    (F(0), F(0), F(80, 1000000), F(0), 0, 1, False),
])
def test_selection_fractional_first_interval_runs(a: F, s: F, length: F, offset: F,
        predecessor: int, run_count: int, reuse: bool) -> None:
    plan, origin, runtime, dl, receipt, observed, pcm = group_case(
        a, s, length, offset=offset, predecessor=predecessor)
    limits = AudioSelectionLimits()
    mapping = map_float_audio_group(plan, int(bool(a)), receipt, origin=origin, runtime=runtime,
                                   decode_limits=dl, limits=limits)
    selected = apply_float_audio_group(plan, mapping, observed, origin=origin, runtime=runtime,
                                      decode_limits=dl, limits=limits)
    expected = bytearray()
    indices = []
    for n in range(mapping.group.first, mapping.group.last):
        x = max(a, F(n, 48000))
        tau = offset+s+x-a
        j = (48000*(tau-mapping.sample_grid_origin)).__floor__()
        rho = mapping.sample_grid_origin+F(j, 48000)-offset-s+a-F(n, 48000)
        run = next(r for r in mapping.runs if r.first <= n < r.last)
        assert j == run.source_first+n-run.first and rho == run.residual
        assert a <= x < a+length and s <= tau-offset < s+length
        assert mapping.sample_grid_origin+F(j, 48000) <= tau < mapping.sample_grid_origin+F(j+1, 48000)
        assert -F(1, 48000) < rho <= F(1, 96000)
        indices.append(j)
        expected.extend(pcm[j*4:(j+1)*4])
    assert selected.pcm == bytes(expected)
    assert len(mapping.runs) == run_count and mapping.reused_first_interval is reuse
    assert (len(indices) > 1 and indices[0] == indices[1]) is reuse


def bypass(value: Any, **changes: object) -> Any:
    result = object.__new__(type(value))
    for name in value.__dataclass_fields__:
        object.__setattr__(result, name, changes.get(name, getattr(value, name)))
    return result


def with_payload(receipt: SourceFloatAudioDecodeReceipt, payload: dict[str, Any]) -> SourceFloatAudioDecodeReceipt:
    return bypass(receipt, payload_json=canonical(payload))


def test_selection_missing_coverage_and_eof() -> None:
    plan, origin, runtime, dl, receipt, observed, _ = tiny_case(1)
    limits = AudioSelectionLimits()
    payload = json.loads(receipt.payload_json)
    # A valid42 unavailable observation: move both stages one sample forward.
    for stage in ("input", "output"):
        payload["audio"][0][stage][0]["pts"] = 1
        payload["audio"][0][stage][0]["samples"] = 3
    payload["audio"][0]["output"][0]["bytes"] = 12
    payload["audio"][0].update(samples=3, bytes=12)
    payload["audio"][0].update(sample_grid_origin="1/48000", source_phase="1/48000",
                              coverage=[dict(window=payload["windows"][0], status="unavailable")])
    unavailable = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=dl.record())
    with pytest.raises(AudioPreparationError, match="SELECTION_UNAVAILABLE"):
        map_float_audio_group(plan, 0, unavailable, origin=origin, runtime=runtime,
                              decode_limits=dl, limits=limits)
    # Receipt requests cannot substitute an earlier/later source window.
    p2 = make_plan(source_fingerprint=origin.fingerprint, shots=[make_shot(
        source={"start": .000001, "end": .000081}, output={"start": 0., "end": .00008})])
    with pytest.raises(AudioPreparationError, match="SELECTION_ASSOCIATION"):
        map_float_audio_group(p2, 0, receipt, origin=origin, runtime=runtime, decode_limits=dl, limits=limits)
    # Tolerated declaration difference remains exact.
    p3 = make_plan(source_fingerprint=origin.fingerprint, shots=[make_shot(
        source={"start": 0., "end": .0000625}, output={"start": 0., "end": .0000625001})])
    payload = json.loads(receipt.payload_json)
    payload["windows"] = [["0/1", "1/16000"]]
    payload["limits"]["decoder_end"] = "1/16000"
    payload["audio"][0]["coverage"][0]["window"] = payload["windows"][0]
    payload["argv"] = [float_audio_decode_argv(str(runtime["ffmpeg"]), origin, F(0), F(1, 16000),
                                               "render48k-f32-mono-v1")]
    eof = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=dl.record())
    selection = plan_float_audio_selection(p3, origin=origin, runtime=runtime, limits=limits)
    assert selection.spans[0].source_end == F(1, 16000)
    assert selection.spans[0].end == F(".0000625001")
    assert map_float_audio_group(p3, 0, eof, origin=origin, runtime=runtime,
                                decode_limits=dl, limits=limits).selected_samples == 3
    # n=3 requests exactly the observed right EOF, never substitute n=2.
    payload = json.loads(receipt.payload_json)
    audio = payload["audio"][0]
    for stage in ("input", "output"):
        audio[stage][0]["samples"] = 3
    audio["output"][0]["bytes"] = 12
    audio.update(samples=3, bytes=12)
    audio["coverage"][0]["status"] = "unavailable"
    eof = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=dl.record())
    assert F(3, 48000) == F(audio["samples"], 48000)
    with pytest.raises(AudioPreparationError, match="SELECTION_UNAVAILABLE"):
        map_float_audio_group(plan, 0, eof, origin=origin, runtime=runtime, decode_limits=dl, limits=limits)
    # No valid PCM substitution for missing bytes, and no normalization of Qa holes.
    mapping = map_float_audio_group(plan, 0, receipt, origin=origin, runtime=runtime,
                                    decode_limits=dl, limits=limits)
    for raw in (observed.pcm[:-4], observed.pcm+b"\0\0\0\0"):
        with pytest.raises(AudioPreparationError):
            apply_float_audio_group(plan, mapping, bypass(observed, pcm=raw), origin=origin,
                                    runtime=runtime, decode_limits=dl, limits=limits)
    bad = make_plan(source_fingerprint=origin.fingerprint, shots=[make_shot(
        source={"start": 0., "end": .00008}, output={"start": 0.000000001, "end": .000080001})])
    # A sub-nanosecond first declaration is retained when its Qa partition starts0.
    assert plan_float_audio_selection(bad, origin=origin, runtime=runtime, limits=limits).spans[0].start == F(1, 10**9)


def test_selection_association_and_hostile_records() -> None:
    plan, origin, runtime, dl, receipt, observed, pcm = tiny_case(1)
    limits = AudioSelectionLimits()
    mapping = map_float_audio_group(plan, 0, receipt, origin=origin, runtime=runtime,
                                    decode_limits=dl, limits=limits)
    for name in ("origin_id", "origin_decoder_id", "operation_id", "policy", "runtime_json"):
        bad_receipt = bypass(receipt, **{name: "wrong"})
        with pytest.raises(AudioPreparationError):
            map_float_audio_group(plan, 0, bad_receipt, origin=origin, runtime=runtime,
                                  decode_limits=dl, limits=limits)
    for name, value in (("policy", "wrong"), ("schedule_id", "0"*64),
            ("selection_sha256", "0"*64), ("runtime_sha256", "0"*64),
            ("operation_id", "0"*64), ("observed_samples", True), ("channels", True),
            ("runs", list(mapping.runs)), ("reused_first_interval", True),
            ("selected_bytes", 15), ("group", replace(mapping.group, index=1))):
        with pytest.raises(AudioPreparationError):
            apply_float_audio_group(plan, bypass(mapping, **{name: value}), observed, origin=origin,
                                    runtime=runtime, decode_limits=dl, limits=limits)
    selection = plan_float_audio_selection(plan, origin=origin, runtime=runtime, limits=limits)
    for bad in (bypass(selection, policy="wrong"), bypass(selection, total_samples=0),
                bypass(selection, audio_format=bypass(selection.audio_format, layout="surround")),
                bypass(selection, schedule=bypass(selection.schedule, source_fingerprint=[])),
                bypass(mapping, selection_sha256="short"),
                bypass(mapping, runs=(replace(mapping.runs[0], last=3),))):
        with pytest.raises(AudioPreparationError):
            bad.record()
    with pytest.raises(AudioPreparationError):
        map_float_audio_group(plan, True, receipt, origin=origin, runtime=runtime,
                              decode_limits=dl, limits=limits)
    with pytest.raises(AudioPreparationError):
        map_float_audio_group(plan, 0, receipt, origin=origin, runtime=runtime,
                              decode_limits=replace(dl, max_samples=3), limits=limits)
    other_runtime = dict(runtime, runtime_id="OTHER")
    with pytest.raises(AudioPreparationError):
        map_float_audio_group(plan, 0, receipt, origin=origin, runtime=other_runtime,
                              decode_limits=dl, limits=limits)
    with pytest.raises(AudioPreparationError):
        map_float_audio_group(plan, 0, receipt, origin=origin,
                              runtime=dict(runtime, ffmpeg="O:/OTHER/ffmpeg.exe"),
                              decode_limits=dl, limits=limits)
    for field, origin_value in (("fingerprint", "fedcba9876543210"), ("origin", F(1)),
                         ("decoder_id", "0"*64), ("streams", list(origin.streams))):
        with pytest.raises(AudioPreparationError):
            map_float_audio_group(plan, 0, receipt, origin=bypass(origin, **{field: origin_value}),
                                  runtime=runtime, decode_limits=dl, limits=limits)
    for field, value in (("pts", 1), ("n", 1), ("time_base", "1/44100"),
                         ("channels", 2), ("samples", True), ("format", "s16"), ("bytes", 12)):
        payload = json.loads(receipt.payload_json)
        payload["audio"][0]["output"][0][field] = value
        with pytest.raises(AudioPreparationError):
            map_float_audio_group(plan, 0, with_payload(receipt, payload), origin=origin,
                                  runtime=runtime, decode_limits=dl, limits=limits)
    class ReceiptSubclass(SourceFloatAudioDecodeReceipt):
        pass

    child = object.__new__(ReceiptSubclass)
    for field in receipt.__dataclass_fields__:
        object.__setattr__(child, field, getattr(receipt, field))
    with pytest.raises(AudioPreparationError):
        map_float_audio_group(plan, 0, child, origin=origin, runtime=runtime,
                              decode_limits=dl, limits=limits)
    for raw, corrupt in ((pcm[:-4]+bytes.fromhex("0000807f"), "finite"),
                         (pcm[:-4]+bytes.fromhex("0100c07f"), "finite"),
                         (pcm[:-4]+bytes.fromhex("000080ff"), "finite"),
                         (pcm[:-4]+bytes.fromhex("0000003f"), "block"),
                         (pcm, "checksum"), (pcm, "aggregate")):
        payload = json.loads(receipt.payload_json)
        audio = payload["audio"][0]
        if corrupt != "aggregate":
            audio["pcm_sha256"] = digest(raw)
        if corrupt == "finite":
            audio["output"][0].update(sha256=digest(raw), checksum=f"{zlib.adler32(raw, 0):08X}")
        if corrupt == "checksum":
            audio["output"][0]["checksum"] = f"{zlib.adler32(raw):08X}"
        if corrupt == "aggregate":
            audio["pcm_sha256"] = "0"*64
        changed = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=dl.record())
        forged = bypass(observed, receipt=changed, pcm=raw)
        changed_map = map_float_audio_group(plan, 0, changed, origin=origin, runtime=runtime,
                                          decode_limits=dl, limits=limits)
        with pytest.raises(AudioPreparationError, match="SELECTION_PCM"):
            apply_float_audio_group(plan, changed_map, forged, origin=origin, runtime=runtime,
                                    decode_limits=dl, limits=limits)


def test_selection_limits_precede_payload_work(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, origin, runtime, dl, receipt, observed, _ = tiny_case(1)
    limits = AudioSelectionLimits()
    mapping = map_float_audio_group(plan, 0, receipt, origin=origin, runtime=runtime,
                                    decode_limits=dl, limits=limits)
    reached: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> Any:
        reached.append("forbidden")
        raise AssertionError("forbidden payload/hash/strict traversal reached")

    # These all fail before current origin/runtime identity hashes or strict-model traversal.
    invalids = [
        (plan, bypass(limits, max_shots=True)), (plan, bypass(limits, max_groups=4097)),
        (plan, replace(limits, max_runs=1)), (plan, replace(limits, max_metadata_bytes=100)),
        (plan, replace(limits, max_selected_samples=3)), (plan, replace(limits, max_selected_bytes=15)),
        (plan, replace(limits, max_plan_bytes=10)),
        (plan.model_copy(update={"shots": [plan.shots[0], plan.shots[0]]}), replace(limits, max_shots=1)),
        (plan.model_copy(update={"extra": "unrequested"}), limits),
        (plan.model_copy(update={"clip_id": "x"*2001}), replace(limits, max_plan_bytes=2000)),
        (plan.model_copy(update={"clip_id": 1 << 4097}), limits),
        (plan.model_copy(update={"clip_id": object()}), limits),
    ]
    cyclic: list[object] = []
    cyclic.append(cyclic)
    invalids.append((plan.model_copy(update={"clip_id": cyclic}), limits))
    deep: object = "leaf"
    for _ in range(12):
        deep = [deep]
    invalids.append((plan.model_copy(update={"clip_id": deep}), limits))
    with monkeypatch.context() as patch:
        patch.setattr(clock_module, "digest", forbidden)
        for invalid, caps in invalids:
            with pytest.raises(AudioPreparationError):
                plan_float_audio_selection(invalid, origin=origin, runtime=runtime, limits=caps)
    assert reached == []
    with monkeypatch.context() as patch:
        patch.setattr(selection_module, "_strict_model_input", forbidden)
        for invalid, caps in invalids[:2]+invalids[6:]:
            with pytest.raises(AudioPreparationError):
                plan_float_audio_selection(invalid, origin=origin, runtime=runtime, limits=caps)
    assert reached == []
    # Receipt string guards precede its record()/JSON loader or real42 reconstruction.
    for bad in (bypass(receipt, payload_json="x"*8193), bypass(receipt, runtime_json=[]),
                bypass(receipt, operation_id="x"*8193)):
        with monkeypatch.context() as patch:
            patch.setattr(selection_module, "_bounded_json", forbidden)
            with pytest.raises(AudioPreparationError):
                map_float_audio_group(plan, 0, bad, origin=origin, runtime=runtime,
                                      decode_limits=dl, limits=limits)
    assert reached == []
    # Malformed wire uses bounded39, including duplicates, exponent, huge int and depth.
    for text in ('{"windows":[],"windows":[]}', '{"x":1e999999999}',
                 '{"x":'+"9"*1236+'}', '['*11+'0'+']'*11):
        with pytest.raises(AudioPreparationError):
            map_float_audio_group(plan, 0, bypass(receipt, payload_json=text), origin=origin,
                                  runtime=runtime, decode_limits=dl, limits=limits)
    with monkeypatch.context() as patch:
        patch.setattr(decode_module, "_validate_pcm", forbidden)
        patch.setattr(selection_module, "_copy_pcm", forbidden)
        for badmap, badpcm, caps in (
            (bypass(mapping, selected_bytes=True), observed.pcm, limits),
            (bypass(mapping, runs=list(mapping.runs)), observed.pcm, limits),
            (mapping, bytearray(observed.pcm), limits),
            (mapping, observed.pcm, replace(limits, max_observed_bytes=15)),
            (mapping, observed.pcm, replace(limits, max_selected_bytes=15)),
            (mapping, observed.pcm, replace(limits, max_metadata_bytes=100)),
        ):
            with pytest.raises(AudioPreparationError):
                apply_float_audio_group(plan, badmap, bypass(observed, pcm=badpcm), origin=origin,
                                        runtime=runtime, decode_limits=dl, limits=caps)
    assert reached == []
    # AssertionError itself must propagate; a swallowed sentinel cannot count as a guard.
    with monkeypatch.context() as patch:
        patch.setattr(selection_module, "_strict_model_input", forbidden)
        with pytest.raises(AssertionError, match="forbidden"):
            plan_float_audio_selection(plan, origin=origin, runtime=runtime, limits=limits)
    assert reached == ["forbidden"]
    with monkeypatch.context() as patch:
        patch.setattr(FloatAudioDecodeLimits, "record", forbidden)
        for tightened in (bypass(dl, max_samples=9_000_001), bypass(dl, max_samples=True),
                          bypass(dl, preroll=F(1 << 4097)), bypass(dl, preroll=F(1))):
            with pytest.raises(AudioPreparationError):
                map_float_audio_group(plan, 0, receipt, origin=origin, runtime=runtime,
                                      decode_limits=tightened, limits=limits)
    assert reached == ["forbidden"]
    # Bypassed public projection must reject before hashing or retaining all rows.
    selection = plan_float_audio_selection(plan, origin=origin, runtime=runtime, limits=limits)
    with monkeypatch.context() as patch:
        patch.setattr(selection_module, "digest", forbidden)
        for bad in (bypass(selection, total_samples=True),
                    bypass(selection, spans=list(selection.spans)),
                    bypass(mapping, runs=(bypass(mapping.runs[0], first=True),)),
                    bypass(mapping, observed_bytes=1 << 4097)):
            with pytest.raises(AudioPreparationError):
                _ = bad.sha256
    assert reached == ["forbidden"]
    many = tuple(replace(selection.spans[0], shot_id="x"*500) for _ in range(3))
    projection = bypass(selection, spans=many, limits=replace(limits, max_metadata_bytes=1600))
    projected: list[str] = []
    real_scalars = selection_module._scalars

    def counted(value: Any) -> dict[str, object]:
        if type(value) is type(selection.spans[0]):
            projected.append(value.shot_id)
        return real_scalars(value)

    with monkeypatch.context() as patch:
        patch.setattr(selection_module, "_scalars", counted)
        with pytest.raises(AudioPreparationError):
            projection.record()
    assert len(projected) < 3
    class Trap(dict[str, object]):
        def items(self) -> Any:
            return forbidden()

    for changed_origin, changed_runtime in (
        (bypass(origin, path=Trap()), runtime),
        (bypass(origin, color=(("x", Trap()),)), runtime),
        (origin, dict(runtime, runtime_id=Trap())),
    ):
        with pytest.raises(AudioPreparationError):
            plan_float_audio_selection(plan, origin=changed_origin, runtime=changed_runtime, limits=limits)
    assert reached == ["forbidden"]
