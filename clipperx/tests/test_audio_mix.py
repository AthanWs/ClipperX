"""HOST synthetic PCM proofs; no source video, GPU, or delivered AAC claims."""

from array import array
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import wave

import pytest

from clipperx import audio_mix as audio
from clipperx.render_assets import RenderAssetRegistry
from clipperx.render_runtime import RenderRequirements, RenderRuntime, resolve_render_runtime
from clipperx.tests.test_visual_compiler import host_plan
from clipperx.visual_contracts import VisualCompositionPlan


def plan(tmp_path: Path, parts: tuple[float, ...] = (1.,), *, normalize: bool = False,
         limiter: bool = False) -> VisualCompositionPlan:
    source = tmp_path / "host.bin"
    if not source.exists():
        source.write_bytes(b"HOST synthetic identity only")
    base = host_plan(source).model_dump()
    template = base["shots"][0]
    base["shots"] = []
    offset = 0.
    for i, duration in enumerate(parts):
        shot = json.loads(json.dumps(template))
        shot.update(shot_id=f"s{i}", source={"start": offset, "end": offset + duration},
                    output={"start": offset, "end": offset + duration},
                    audio={"normalize_dialogue": normalize, "limiter": limiter})
        base["shots"].append(shot)
        offset += duration
    return VisualCompositionPlan.model_validate(base)


def test_schedule_detaches_merges_visual_splits_and_rounds_absolute_clock(tmp_path: Path) -> None:
    p = plan(tmp_path, (.10001, .10001, .79998))
    schedule = audio.schedule_audio(p)
    assert schedule.total_samples == 48000
    assert len(schedule.spans) == 1
    assert audio.quantize_time(.10001) == 4800
    p.shots[0].audio.source_gain_db = -9
    assert schedule.spans[0].gain_db == 0
    assert schedule == audio.schedule_audio(plan(tmp_path))


def test_exact_subset_and_zero_sample_interval_rejected(tmp_path: Path) -> None:
    p = plan(tmp_path)
    p.shots[0].output.end += .0001
    with pytest.raises(ValueError, match="durations must match exactly"):
        audio.schedule_audio(p)
    p = plan(tmp_path, (.000001, .999999))
    with pytest.raises(ValueError, match="zero samples"):
        audio.schedule_audio(p)


def test_graph_original_windows_gain_and_disabled_authority(tmp_path: Path) -> None:
    p = plan(tmp_path, (1., 1.))
    p.shots[0].source.start, p.shots[0].source.end = 10., 11.
    p.shots[1].source.start, p.shots[1].source.end = 20., 21.
    p.shots[1].audio.source_gain_db = -6
    result = audio.compile_audio(p, audio_format=audio.AudioFormat(48000, "stereo"))
    assert "atrim=start=10:end=11" in result.filter_complex
    assert "atrim=start=20:end=21" in result.filter_complex
    assert "asetpts=PTS-10/TB" in result.filter_complex
    assert "volume=-6dB" in result.filter_complex
    assert "loudnorm" not in result.filter_complex and "aeval=" not in result.filter_complex
    assert result.normalization_scope == "mixed_track_normalization"


def test_late_fractional_source_and_cue_decimal_clocks_are_not_rounded(tmp_path: Path) -> None:
    p = plan(tmp_path)
    p.shots[0].source.start, p.shots[0].source.end = 10000.025000000001, 10001.025000000001
    graph = audio.compile_audio(p, audio_format=audio.AudioFormat(48000, "mono")).filter_complex
    assert "atrim=start=10000.025000000001:end=10001.025000000001" in graph
    assert "asetpts=PTS-10000.025000000001/TB" in graph


def test_schedule_and_receipt_collection_fields_detach(tmp_path: Path) -> None:
    schedule = audio.schedule_audio(plan(tmp_path))
    mutable = list(schedule.spans)
    detached = audio.AudioSchedule(schedule.source_fingerprint, mutable, [])
    mutable.clear()
    assert detached.spans == schedule.spans and detached.cues == ()


def test_boolean_rate_and_nonfinite_stats_are_rejected() -> None:
    with pytest.raises(ValueError, match="48000"):
        audio.AudioFormat(True, "mono")
    with pytest.raises(audio.AudioPreparationError, match="malformed"):
        audio.select_normalization_mode(audio.LoudnessStats(float("nan"), 0, -1, -30, 0))
    with pytest.raises(audio.AudioPreparationError, match="malformed"):
        audio.parse_loudness(json.dumps({"input_i": -20, "input_lra": True, "input_tp": -10,
            "input_thresh": -30, "target_offset": 0, "normalization_type": "linear"}))


def add_cue(p: VisualCompositionPlan, start: float, end: float, duck: float = -6,
            gain: float = 0) -> None:
    from clipperx.visual_contracts import AudioCue
    p.shots[0].audio.cues.append(AudioCue(asset_id="impact", start=start, end=end,
                                        gain_db=gain, duck_source_db=duck))


def test_cue_registry_channel_qualification_and_exact_envelope(tmp_path: Path) -> None:
    p = plan(tmp_path)
    add_cue(p, .10001, .30001)
    add_cue(p, .20001, .40001, -12)
    schedule = audio.schedule_audio(p)
    assert audio.duck_db(schedule, .1) == 0
    assert audio.duck_db(schedule, .15) == -6
    assert audio.duck_db(schedule, .20251) == pytest.approx(-6)
    assert audio.duck_db(schedule, .21) == -12
    assert audio.duck_db(schedule, .40001) == 0
    with pytest.raises(ValueError, match="registered"):
        audio.compile_audio(p, audio_format=audio.AudioFormat(48000, "stereo"))
    assets = RenderAssetRegistry({}, {}, {"impact": tmp_path / "impact.wav"})
    with pytest.raises(ValueError, match="layout"):
        audio.compile_audio(p, audio_format=audio.AudioFormat(48000, "stereo"), assets=assets,
                            cue_formats={"impact": audio.AudioFormat(48000, "mono")})


@pytest.fixture(scope="module")
def runtime() -> RenderRuntime:
    # Explicit paired pinned runtime. No qualification, GPU device, or video work.
    from clipperx.render_runtime import NATIVE_RUNTIME
    return resolve_render_runtime(RenderRequirements(NATIVE_RUNTIME, True))


def pcm(seconds: float = 1., *, amplitude: float = .1, channels: int = 2) -> bytes:
    return array("f", (amplitude * math.sin(2 * math.pi * 997 * n / 48000)
                       for n in range(round(seconds * 48000)) for _ in range(channels))).tobytes()


def binding(p: VisualCompositionPlan, data: bytes) -> audio.SourceClockBinding:
    return audio.SourceClockBinding(p.source_fingerprint, "HOST original PTS at 0; selected PCM",
                                    hashlib.sha256(data).hexdigest())


def decode(runtime: RenderRuntime, graph: str, data: bytes, channels: int = 2,
           cue_data: bytes | None = None, tmp_path: Path | None = None) -> array:
    args = [str(runtime.ffmpeg), "-hide_banner", "-loglevel", "info", "-filter_complex_threads",
            "1", "-f", "f32le", "-ar", "48000", "-ac", str(channels), "-i", "pipe:0"]
    if cue_data is not None:
        assert tmp_path is not None
        cue_file = tmp_path / "impact.wav"
        samples = array("f")
        samples.frombytes(cue_data)
        with wave.open(str(cue_file), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(array("h", (round(value*32768) for value in samples)).tobytes())
        args += ["-i", str(cue_file)]
    args += ["-filter_complex", graph, "-map", "[audio_out]", "-c:a", "pcm_f32le",
             "-f", "f32le", "pipe:1"]
    result = subprocess.run(args, input=data, capture_output=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    values = array("f")
    values.frombytes(result.stdout)
    return values


def test_real_duck_fractional_ramps_overlap_channels_cue_padding(runtime: RenderRuntime, tmp_path: Path) -> None:
    p = plan(tmp_path)
    add_cue(p, .10001, .30001)
    add_cue(p, .20001, .40001, -12)
    assets = RenderAssetRegistry({}, {}, {"impact": tmp_path / "impact.wav"})
    result = audio.compile_audio(p, audio_format=audio.AudioFormat(48000, "stereo"), assets=assets,
                                cue_formats={"impact": audio.AudioFormat(48000, "stereo")})
    data = array("f", (.1 if ch == 0 else -.2 for _ in range(48000) for ch in range(2))).tobytes()
    # One explicitly registered asset input shared by both planned cue occurrences.
    out = decode(runtime, result.filter_complex, data,
                 cue_data=array("f", [0.] * 20).tobytes(), tmp_path=tmp_path)
    assert len(out) == 96000
    for n in range(48000):
        gain = 10 ** (audio.duck_db(result.schedule, n / 48000) / 20)
        assert out[2*n] == pytest.approx(.1 * gain, abs=3e-7)
        assert out[2*n+1] == pytest.approx(-.2 * gain, abs=3e-7)


def test_real_limiter_no_boost_timing_eof_and_subdivision(runtime: RenderRuntime, tmp_path: Path) -> None:
    p = plan(tmp_path, limiter=True)
    data = array("f", [.125] * 96000).tobytes()
    graph = audio.compile_audio(p, audio_format=audio.AudioFormat(48000, "stereo")).filter_complex
    assert "alimiter=limit=0.944:level=0:latency=1" in graph
    out = decode(runtime, graph, data)
    assert len(out) == 96000
    assert min(out) == pytest.approx(.125) and max(out) == pytest.approx(.125)
    split = plan(tmp_path, (.33333, .66667), limiter=True)
    assert graph == audio.compile_audio(split, audio_format=audio.AudioFormat(48000, "stereo")).filter_complex


def test_real_limiter_first_last_impulses_and_disabled_runs(runtime: RenderRuntime, tmp_path: Path) -> None:
    p = plan(tmp_path, (.4, .2, .4), limiter=True)
    p.shots[1].audio.limiter = False
    values = array("f", [0.] * 96000)
    for n in [0, 19199, 19200, 28799, 28800, 47999]:
        values[2*n], values[2*n+1] = .25, -.5
    graph = audio.compile_audio(p, audio_format=audio.AudioFormat(48000, "stereo")).filter_complex
    assert graph.count("alimiter=") == 2
    out = decode(runtime, graph, values.tobytes())
    assert len(out) == len(values)
    assert out == values


def test_real_cue_exact_gain_padding_and_zero_duck(runtime: RenderRuntime, tmp_path: Path) -> None:
    p = plan(tmp_path)
    add_cue(p, .10001, .3, duck=0, gain=-12)
    result = audio.compile_audio(p, audio_format=audio.AudioFormat(48000, "stereo"),
        assets=RenderAssetRegistry({}, {}, {"impact": tmp_path / "impact.wav"}),
        cue_formats={"impact": audio.AudioFormat(48000, "stereo")})
    source = array("f", [.1, -.2] * 48000).tobytes()
    cue = array("f", [.25, .5] * 100).tobytes()
    out = decode(runtime, result.filter_complex, source, cue_data=cue, tmp_path=tmp_path)
    first = audio.quantize_time(.10001)
    for n in [0, first-1, first, first+99, first+100, 14399, 14400, 47999]:
        for ch, base in enumerate([.1, -.2]):
            extra = [.25, .5][ch] * 10 ** (-12/20) if first <= n < first+100 else 0
            assert out[2*n+ch] == pytest.approx(base+extra, abs=1e-7)


def test_measured_shared_authority_modes_staleness_and_real_pcm(runtime: RenderRuntime, tmp_path: Path) -> None:
    p = plan(tmp_path, normalize=True)
    schedule = audio.schedule_audio(p)
    data = pcm()
    clock = binding(p, data)
    fmt = audio.AudioFormat(48000, "stereo")
    receipt = audio.measure_normalization(schedule, data, binding=clock, audio_format=fmt, runtime=runtime)
    assert receipt.mode == "linear"
    observed = audio.process_normalized_pcm(schedule, data, receipt=receipt, binding=clock,
                                            audio_format=fmt, runtime=runtime)
    assert observed.observed_mode == receipt.mode
    assert len(observed.pcm) == len(data)
    result = audio.compile_audio(p, audio_format=fmt, binding=clock, runtime=runtime, receipt=receipt)
    assert result.filter_complex.count("loudnorm=") == 1
    split = plan(tmp_path, (.33333, .66667), normalize=True)
    assert result == audio.compile_audio(split, audio_format=fmt, binding=clock, runtime=runtime,
                                        receipt=receipt)
    with pytest.raises(audio.AudioPreparationError, match="identity"):
        audio.compile_audio(p, audio_format=fmt, binding=replace(clock, decoder_identity="stale"),
                            runtime=runtime, receipt=receipt)
    stats = replace(receipt.stats, input_i=-30., input_tp=-2.)
    assert audio.select_normalization_mode(stats)[0] == "dynamic"
    with pytest.raises(audio.AudioPreparationError, match="mode"):
        audio.verify_normalization_observation(receipt, "dynamic", schedule.enabled_samples)


def test_real_dynamic_processing_mode_rate_count_and_observations(runtime: RenderRuntime, tmp_path: Path) -> None:
    p = plan(tmp_path, (4.,), normalize=True)
    values = array("f")
    values.frombytes(pcm(4., amplitude=.002))
    values[20000] = .95
    data = values.tobytes()
    schedule = audio.schedule_audio(p)
    clock, fmt = binding(p, data), audio.AudioFormat(48000, "stereo")
    receipt = audio.measure_normalization(schedule, data, binding=clock, audio_format=fmt, runtime=runtime)
    assert receipt.mode == "dynamic"
    observed = audio.process_normalized_pcm(schedule, data, receipt=receipt, binding=clock,
                                            audio_format=fmt, runtime=runtime)
    assert observed.samples == 192000 and observed.observed_mode == "dynamic"
    assert all(math.isfinite(value) for value in observed.output_loudness)


def test_real_true_false_true_mapping_gain_after_shared_normalization(runtime: RenderRuntime, tmp_path: Path) -> None:
    p = plan(tmp_path, (.4, .2, .4), normalize=True)
    p.shots[0].audio.source_gain_db = -6
    p.shots[1].audio.normalize_dialogue = False
    p.shots[1].audio.source_gain_db = -12
    p.shots[2].audio.source_gain_db = -3
    source = pcm()
    enabled = source[:19200*8] + source[28800*8:]
    schedule = audio.schedule_audio(p)
    clock, fmt = binding(p, enabled), audio.AudioFormat(48000, "stereo")
    receipt = audio.measure_normalization(schedule, enabled, binding=clock, audio_format=fmt, runtime=runtime)
    normalized = audio.process_normalized_pcm(schedule, enabled, receipt=receipt, binding=clock,
                                               audio_format=fmt, runtime=runtime)
    compiled = audio.compile_audio(p, audio_format=fmt, binding=clock, runtime=runtime, receipt=receipt)
    assert compiled.filter_complex.count("loudnorm=") == 1
    out = decode(runtime, compiled.filter_complex, source)
    raw, norm = array("f"), array("f")
    raw.frombytes(source)
    norm.frombytes(normalized.pcm)
    for n, ref, gain in [(1000, norm[2000], -6), (25000, raw[50000], -12),
                         (40000, norm[(40000-9600)*2], -3)]:
        assert out[2*n] == pytest.approx(ref * 10 ** (gain/20), abs=2e-7)


def test_stale_receipt_format_runtime_and_sample_count_fail_before_tools(runtime: RenderRuntime, tmp_path: Path) -> None:
    p = plan(tmp_path, normalize=True)
    schedule, data = audio.schedule_audio(p), pcm()
    clock, fmt = binding(p, data), audio.AudioFormat(48000, "stereo")
    stats = audio.LoudnessStats(-24, 2, -15, -34, 0)
    mode, reason = audio.select_normalization_mode(stats)
    receipt = audio.NormalizationReceipt(schedule.identity, clock, audio._runtime_identity(runtime),
                                           fmt, 48000, stats, mode, reason)
    for stale in [replace(receipt, samples=47999), replace(receipt, runtime_identity=("stale",)),
                  replace(receipt, audio_format=audio.AudioFormat(48000, "mono"))]:
        with pytest.raises(audio.AudioPreparationError, match="identity"):
            audio.compile_audio(p, audio_format=fmt, binding=clock, runtime=runtime, receipt=stale)
    with pytest.raises(audio.AudioPreparationError, match="sample_count"):
        audio.verify_normalization_observation(receipt, mode, 47999)


@pytest.mark.parametrize("seconds,amplitude,code", [(.1, .1, "insufficient_duration"),
    (1., 0., "digital_silence"), (1., 1e-8, "undefined_measurement")])
def test_real_unmeasurable_has_typed_outcome(runtime: RenderRuntime, tmp_path: Path, seconds: float,
                                            amplitude: float, code: str) -> None:
    p = plan(tmp_path, (seconds,), normalize=True)
    data = pcm(seconds, amplitude=amplitude)
    with pytest.raises(audio.AudioPreparationError) as error:
        audio.measure_normalization(audio.schedule_audio(p), data, binding=binding(p, data),
                                     audio_format=audio.AudioFormat(48000, "stereo"), runtime=runtime)
    assert error.value.code == code


def test_real_short_dynamic_filter_mode_reversion_is_rejected(runtime: RenderRuntime, tmp_path: Path) -> None:
    p = plan(tmp_path, normalize=True)
    values = array("f")
    values.frombytes(pcm(amplitude=.002))
    values[20000] = .95
    data, fmt = values.tobytes(), audio.AudioFormat(48000, "stereo")
    schedule = audio.schedule_audio(p)
    clock = binding(p, data)
    receipt = audio.measure_normalization(schedule, data, binding=clock, audio_format=fmt, runtime=runtime)
    assert receipt.mode == "dynamic"
    with pytest.raises(audio.AudioPreparationError, match="Expected dynamic; observed linear"):
        audio.process_normalized_pcm(schedule, data, receipt=receipt, binding=clock,
                                     audio_format=fmt, runtime=runtime)


def test_real_limiter_threshold_and_mono_channel_preservation(runtime: RenderRuntime, tmp_path: Path) -> None:
    p = plan(tmp_path, limiter=True)
    values = array("f", [1.2] * 48000).tobytes()
    graph = audio.compile_audio(p, audio_format=audio.AudioFormat(48000, "mono")).filter_complex
    out = decode(runtime, graph, values, channels=1)
    assert len(out) == 48000
    assert max(out) == pytest.approx(.944, abs=2e-7)
    assert max(abs(value) for value in out) <= .9440002


@pytest.mark.parametrize("expression", ["volume=-6.123456789dB[gain0]",
    "volume=-12.345678901dB,adelay=", "(-9.876543219)*"])
def test_fractional_authored_gains_and_duck_preserve_decimal_precision(tmp_path: Path,
                                                                      expression: str) -> None:
    p = plan(tmp_path)
    p.shots[0].audio.source_gain_db = -6.123456789
    add_cue(p, .1, .3, duck=-9.876543219, gain=-12.345678901)
    compiled = audio.compile_audio(p, audio_format=audio.AudioFormat(48000, "stereo"),
        assets=RenderAssetRegistry({}, {}, {"impact": tmp_path / "impact.wav"}),
        cue_formats={"impact": audio.AudioFormat(48000, "stereo")})
    assert expression in compiled.filter_complex
    assert audio.duck_db(compiled.schedule, .2) == -9.876543219


@pytest.fixture
def normalization_case(tmp_path: Path) -> tuple[audio.AudioSchedule, bytes, audio.SourceClockBinding,
                                               audio.AudioFormat, RenderRuntime,
                                               audio.NormalizationReceipt]:
    p = plan(tmp_path, normalize=True)
    schedule, data = audio.schedule_audio(p), bytes(48000*2*4)
    clock, fmt = binding(p, data), audio.AudioFormat(48000, "stereo")
    path = tmp_path / "host.bin"
    digest = audio._hash(path)
    runtime = RenderRuntime(path, path, "system", digest, digest, None, None,
                            "-filter_complex_script", None)
    stats = audio.LoudnessStats(-24, 2, -15, -34, 0)
    mode, reason = audio.select_normalization_mode(stats)
    receipt = audio.NormalizationReceipt(schedule.identity, clock, audio._runtime_identity(runtime),
                                           fmt, 48000, stats, mode, reason)
    return schedule, data, clock, fmt, runtime, receipt


def loudnorm_observation() -> dict[str, object]:
    return {"input_i": "-24", "input_lra": "2", "input_tp": "-15",
            "input_thresh": "-34", "target_offset": "0", "normalization_type": "linear",
            "output_i": "-14.0123456789", "output_lra": "1.234567891", "output_tp": "-5.0123456789"}


@pytest.mark.parametrize("field", ["output_i", "output_lra", "output_tp"])
@pytest.mark.parametrize("bad,code", [("MISSING", "malformed_measurement"),
    (True, "malformed_measurement"), ("not-numeric", "malformed_measurement"),
    (None, "malformed_measurement"), ([], "malformed_measurement"),
    ("nan", "undefined_measurement"), ("-inf", "undefined_measurement")])
def test_pass_two_output_boundary_has_typed_failures(
    normalization_case: tuple[audio.AudioSchedule, bytes, audio.SourceClockBinding,
                              audio.AudioFormat, RenderRuntime, audio.NormalizationReceipt],
    monkeypatch: pytest.MonkeyPatch, field: str, bad: object, code: str,
) -> None:
    schedule, data, clock, fmt, runtime, receipt = normalization_case
    payload = loudnorm_observation()
    if bad == "MISSING":
        payload.pop(field)
    else:
        payload[field] = bad
    result = subprocess.CompletedProcess([], 0, data, json.dumps(payload).encode())
    monkeypatch.setattr(audio, "_run_pcm", lambda *args, **kwargs: result)
    with pytest.raises(audio.AudioPreparationError) as error:
        audio.process_normalized_pcm(schedule, data, receipt=receipt, binding=clock,
                                     audio_format=fmt, runtime=runtime)
    assert error.value.code == code


def test_pass_two_output_boundary_retains_valid_observation(
    normalization_case: tuple[audio.AudioSchedule, bytes, audio.SourceClockBinding,
                              audio.AudioFormat, RenderRuntime, audio.NormalizationReceipt],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule, data, clock, fmt, runtime, receipt = normalization_case
    result = subprocess.CompletedProcess([], 0, data, json.dumps(loudnorm_observation()).encode())
    monkeypatch.setattr(audio, "_run_pcm", lambda *args, **kwargs: result)
    observed = audio.process_normalized_pcm(schedule, data, receipt=receipt, binding=clock,
                                           audio_format=fmt, runtime=runtime)
    assert observed.output_loudness == (-14.0123456789, 1.234567891, -5.0123456789)
    assert observed.pcm == data and observed.observed_mode == receipt.mode
    assert observed.samples == 48000 and observed.upstream_observation == receipt.stats


def test_normalization_filter_retains_supported_reported_decimal_precision(
    normalization_case: tuple[audio.AudioSchedule, bytes, audio.SourceClockBinding,
                              audio.AudioFormat, RenderRuntime, audio.NormalizationReceipt],
) -> None:
    receipt = normalization_case[-1]
    stats = audio.LoudnessStats(-24.123456789, 2.234567891, -15.345678912, -34.456789123, .123456789)
    receipt = replace(receipt, stats=stats)
    graph = audio.normalization_filter(receipt)
    for parameter, value in [("measured_I", stats.input_i), ("measured_LRA", stats.input_lra),
                             ("measured_TP", stats.input_tp), ("measured_thresh", stats.input_thresh),
                             ("offset", stats.target_offset)]:
        assert f"{parameter}={value}:" in graph
    # No binary-expansion artifacts from .17g for decimal values such as .1.
    graph = audio.normalization_filter(replace(receipt, stats=replace(stats, target_offset=.1)))
    assert "offset=0.1:" in graph
