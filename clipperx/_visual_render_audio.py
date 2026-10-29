"""Source-clock float selection followed by the existing measured effect tail."""

from contextlib import ExitStack
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

from . import source_decode as sd
from ._normalization_files import NormalizationFiles
from ._normalization_transport import prepare_normalized_audio
from .audio_mix import AudioSchedule, SourceClockBinding, compile_prepared_audio, schedule_audio
from .audio_selection import (
    AudioSelectionGroup, AudioSelectionLimits, apply_float_audio_group,
    map_float_audio_group, plan_float_audio_selection,
)
from .render_assets import RenderAssetRegistry
from .render_clock import SourceTimeOrigin
from .render_runtime import RenderRuntime, combine_render_requirements, qualify_render_runtime
from .visual_compiler import probe_audio_format
from .visual_contracts import VisualCompositionPlan

PCM_CAP = 68_736_000


def enabled_slices(schedule: AudioSchedule, group: AudioSelectionGroup,
                   stride: int) -> tuple[tuple[int, int], ...]:
    return tuple(((max(span.first, group.first)-group.first)*stride,
                  (min(span.last, group.last)-group.first)*stride)
        for span in schedule.spans if span.normalize and max(span.first, group.first) < min(span.last, group.last))


def prepare_audio(host: Any, plan: VisualCompositionPlan, origin: SourceTimeOrigin,
                  runtime: RenderRuntime, assets: RenderAssetRegistry, requirements: Any
                  ) -> tuple[Path, Any, dict[str, Any]]:
    schedule = schedule_audio(plan)
    limits, decode_limits = AudioSelectionLimits(), sd.FloatAudioDecodeLimits()
    observation_runtime = replace(runtime, device_index=None)
    record = sd._runtime_record(observation_runtime)
    selection = plan_float_audio_selection(plan, origin=origin, runtime=record, limits=limits)
    fmt = selection.audio_format
    stride = 4*fmt.channels
    if max(schedule.total_samples*stride, schedule.enabled_samples*stride) > PCM_CAP:
        raise ValueError("packed PCM bus exceeds 68736000 bytes")
    selected_path = host.root / "selected.f32"
    needs_selected = schedule.enabled_samples < schedule.total_samples
    enabled = bytearray(schedule.enabled_samples*stride)
    enabled_offset = 0
    maps = []
    with ExitStack() as stack:
        selected_sink = None
        if needs_selected:
            host.register(selected_path, "pcm", schedule.total_samples*stride)
            selected_sink = stack.enter_context(host.sink(selected_path))
        for group in selection.groups:
            host.recheck_space()
            host.require_log_capacity(decode_limits.max_log_bytes)
            control = host.job.new_observation("materialized_f32", deadline_ns=host.deadline(45))
            observed = sd.observe_float_audio_window(observation_runtime, origin,
                sd.Window(group.source_start, group.source_end), audio_policy="render48k-f32-"+fmt.layout+"-v1",
                limits=decode_limits, _host=control, _runtime_files=host.files)
            mapping = map_float_audio_group(plan, group.index, observed.receipt, origin=origin,
                runtime=record, decode_limits=decode_limits, limits=limits)
            selected = apply_float_audio_group(plan, mapping, observed, origin=origin,
                runtime=record, decode_limits=decode_limits, limits=limits)
            view = memoryview(selected.pcm)
            try:
                with host.files.stage_deadline(host.deadline(120)):
                    if selected_sink is not None:
                        host.write_all(selected_sink, view)
                    for first, last in enabled_slices(schedule, group, stride):
                        enabled[enabled_offset:enabled_offset+last-first] = view[first:last]
                        enabled_offset += last-first
            finally:
                view.release()
            maps.append({"group": group.index, "mapping_sha256": selected.map_sha256,
                "selected_bytes": len(selected.pcm), "selected_sha256": selected.pcm_sha256,
                "receipt_sha256": mapping.receipt_sha256,
                "source_control": control.ownership_snapshot(), "source_receipt": observed.receipt.record()})
            host.account_logs(sum(d.log_bytes for d in control.ownership_snapshot().diagnostics))
            host.record_auxiliary("float", maps[-1])
            del observed, selected
    if enabled_offset != len(enabled):
        raise ValueError("complete enabled bus sample association mismatch")
    if needs_selected:
        host.seal(selected_path, schedule.total_samples*stride, finite=True)
    enabled_bytes = bytes(enabled)
    del enabled
    norm = NormalizationFiles(host.job, runtime, host.root / "normalization", clock=host.job._clock)
    host.normalization_files = norm
    host.require_log_capacity(2*65536 if schedule.enabled_samples else 0)
    with host.files.stage_deadline(host.deadline(120)):
        binding = (SourceClockBinding(plan.source_fingerprint, origin.decoder_id,
            hashlib.sha256(enabled_bytes).hexdigest()) if schedule.enabled_samples else None)
        prepared = prepare_normalized_audio(schedule, enabled_bytes, binding=binding, audio_format=fmt,
            runtime=runtime, job=host.job, files=norm)
    host.account_logs(sum(d.log_bytes for c in norm._input_consumers for d in c.ownership_snapshot().diagnostics))
    if prepared is not None:
        host.record_auxiliary("normalization", {"passes": prepared.passes,
            "controls": tuple(c.ownership_snapshot() for c in norm._input_consumers)})
    del enabled_bytes
    inputs = []
    argv = [str(runtime.ffmpeg), "-hide_banner", "-nostats", "-loglevel", "info"]
    labels: dict[str, str] = {}
    if needs_selected:
        labels["selected_input_label"] = "0:a"
        inputs.append(selected_path)
    if prepared is not None:
        normalized_path = norm.output_path(prepared)
        host.files.register(normalized_path, role="original", owned=False, max_bytes=PCM_CAP)
        labels["normalized_input_label"] = f"{len(inputs)}:a"
        inputs.append(normalized_path)
    for path in inputs:
        argv.extend(("-f", "f32le", "-ar", "48000", "-ac", str(fmt.channels),
            "-channel_layout", fmt.layout, "-i", str(path)))
    cue_formats = {key: probe_audio_format(path, runtime=runtime,
        _runtime_files=host.files, _tool_runner=host.tools) for key, path in assets.sfx.items()
        if any(c.asset_id == key for c in schedule.cues)}
    compiled = compile_prepared_audio(plan, schedule_identity=schedule.identity, audio_format=fmt,
        assets=assets, cue_formats=cue_formats, binding=binding, runtime=runtime,
        receipt=None if prepared is None else prepared.receipt, cue_input_start=len(inputs), **labels)
    qualified = qualify_render_runtime(runtime, combine_render_requirements(requirements, compiled.requirements),
        _files=host.files, _runner=host.tools)
    for _, path, _ in compiled.cue_inputs:
        argv.extend(("-i", str(path)))
        inputs.append(path)
    graph_path = host.root / "assets" / "audio.txt"
    host.publish(graph_path, compiled.filter_complex.encode(), role="graph")
    argv.extend((runtime.graph_file_option, "assets/audio.txt", "-map", f"[{compiled.output_label}]",
        "-vn", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"))
    finalmix = host.root / "finalmix.f32"
    size = schedule.total_samples*stride
    host.register(finalmix, "pcm", size)
    with host.sink(finalmix) as sink:
        host.run(tuple(argv), kind="finalmix", timeout=120, max_stdout_bytes=size,
            expected_stdout_bytes=size, sink=sink, inputs=tuple([*inputs, graph_path]))
    host.seal(finalmix, size, finite=True)
    with host.io_scope(120):
        if prepared is not None:
            norm.release_output(prepared)
        if not norm.retire().files_clear:
            raise ValueError("normalization files retain unresolved custody")
    return finalmix, fmt, {"schedule_identity": schedule.identity, "selection_sha256": selection.sha256,
        "groups": maps, "binding": binding, "normalization": None if prepared is None else {
            "receipt": prepared.receipt, "passes": prepared.passes,
            "normalized_file": prepared.normalized_file,
            "controls": tuple(c.ownership_snapshot() for c in norm._input_consumers)}, "qualification": qualified,
        "graph": compiled.filter_complex, "cue_inputs": compiled.cue_inputs}
