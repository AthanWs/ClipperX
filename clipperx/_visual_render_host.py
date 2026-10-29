"""Concrete serial RAW executor with one registered driver and owned services."""

from dataclasses import fields, is_dataclass, replace
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import struct
import subprocess
import sys
import threading
import time
from typing import Any, cast, Iterator, ContextManager

from . import font_staging, render_runtime as rr, source_decode as sd
from ._retained_files import RetainedBinary, RetainedFiles
from ._render_services import OwnedRenderTools
from ._source_host_ownership import Process, SourceHostJob
from ._visual_render_audio import prepare_audio
from ._visual_render_video import render_shots, source_windows
from .audio_mix import schedule_audio
from .overlay_ass import compile_overlay_ass
from .render_assets import RenderAssetRegistry
from .render_contracts import FFmpegRenderSpec, RenderProfile
from .render_metrics import parse_frame_observations, validate_output_observations
from .schema import TranscriptSegment
from .visual_compiler import (
    CompiledVisualAssembly, compile_visual_assembly, encoder_arguments,
    preflight_visual_plan, resolve_visual_plan, _bounded_json_metadata,
)
from .visual_contracts import VisualCompositionPlan

MiB = 1024**2
GiB = 1024**3


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: ({"text": getattr(value, f.name).decode("utf-8", errors="replace"),
            "representation": "bounded stderr tail", **_json_value(getattr(value, f.name))}
            if f.name == "stderr_tail" and isinstance(getattr(value, f.name), bytes)
            else _json_value(getattr(value, f.name))) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (Path, Fraction, Enum)):
        return value.value if isinstance(value, Enum) else str(value)
    return value


def _json_bytes(value: Any) -> bytes:
    raw = json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > 512*MiB:
        raise ValueError("bounded evidence bytes exceeded")
    return raw


class RecordedTools(OwnedRenderTools):
    """Driver-owned bounded diagnostic snapshots; metadata workers never write files."""

    def __init__(self, job: SourceHostJob, files: RetainedFiles) -> None:
        super().__init__(job, files, clock=job._clock)
        self.evidence: list[dict[str, Any]] = []
        self.on_record: Any = None
        self.before_run: Any = None

    def run(self, argv: tuple[str, ...], **kwargs: Any) -> Any:
        if self.before_run is not None:
            self.before_run(kwargs["max_stderr_bytes"])
        started = self._job._now()
        result = super().run(argv, **kwargs)
        if len(self.evidence) >= 32768:
            raise ValueError("stage evidence count cap")
        record = _json_value(result)
        record["stderr_tail_text"] = result.stderr_tail.decode("utf-8", errors="replace")
        record["stderr_is_tail"] = True
        record["elapsed_ns"] = self._job._now()-started
        self.evidence.append(record)
        if self.on_record is not None:
            self.on_record(len(self.evidence), record)
        return result


class HostContext:
    def __init__(self, job: SourceHostJob, files: RetainedFiles, root: Path) -> None:
        self.job, self.files, self.root = job, files, root
        self.tools = RecordedTools(job, files)
        self.tools.before_run = self.require_log_capacity
        self.normalization_files: Any = None
        self.font_paths: tuple[Path, ...] = ()
        self.budget: dict[str, int] = {}
        self.seals: dict[str, dict[str, Any]] = {}
        self.diagnostic_paths: list[Path] = []
        self.evidence_bytes = 0
        self.log_bytes = 0
        self.auxiliary_index = 0

    def record_auxiliary(self, kind: str, value: Any) -> None:
        if kind not in ("source", "float", "normalization"):
            raise ValueError("finite auxiliary diagnostic kind required")
        path = self.root / "diagnostics" / f"{kind}{self.auxiliary_index:06d}.json"
        self.auxiliary_index += 1
        self.publish(path, _json_bytes(value))
        self.diagnostic_paths.append(path)

    def require_log_capacity(self, reservation: int) -> None:
        if self.log_bytes+reservation > 256*MiB:
            raise ValueError("aggregate256MiB child log admission exhausted")

    def account_logs(self, actual: int) -> None:
        self.require_log_capacity(actual)
        self.log_bytes += actual

    def record_progress(self, index: int, record: dict[str, Any]) -> None:
        self.account_logs(record.get("stderr_bytes", 0))
        path = self.root / "diagnostics" / f"{index:06d}.json"
        self.publish(path, _json_bytes(record))
        self.diagnostic_paths.append(path)

    def deadline(self, seconds: int) -> int:
        return min(self.job._deadline, self.job._now()+seconds*10**9)

    def io_scope(self, seconds: int) -> ContextManager[None]:
        active = self.job._driver_io_deadline
        deadline = self.deadline(seconds)
        if active is not None:
            if active > deadline:
                raise ValueError("existing file deadline would exceed requested stage budget")
            return nullcontext()
        return self.files.stage_deadline(deadline)

    @contextmanager
    def sink(self, path: Path) -> Iterator[RetainedBinary]:
        context = self.files.open_file(path, "xb")
        with self.io_scope(120):
            sink = context.__enter__()
        try:
            yield sink
        except BaseException:
            context.__exit__(*sys.exc_info())
            raise
        else:
            with self.io_scope(120):
                try:
                    sink.flush()
                    sink.fsync()
                finally:
                    context.__exit__(*sys.exc_info())

    def register(self, path: Path, role: str, cap: int) -> None:
        self.files.register(path, role=role, owned=True, max_bytes=cap)

    @staticmethod
    def write_all(sink: RetainedBinary, view: memoryview) -> None:
        offset = 0
        while offset < len(view):
            count = sink.write(view[offset:offset+65536])
            offset += count

    def read(self, path: Path, cap: int) -> bytes:
        data = bytearray()
        with self.files.stage_deadline(self.deadline(120)), self.files.open_binary(path) as source:
            while True:
                raw = source.read(min(MiB, cap+1-len(data)))
                if not raw:
                    break
                data.extend(raw)
                if len(data) > cap:
                    raise ValueError("bounded input byte cap")
        return bytes(data)

    def publish(self, path: Path, raw: bytes, *, role: str = "sidecar") -> None:
        if len(raw) > 512*MiB:
            raise ValueError("published byte cap")
        if role == "sidecar":
            if self.evidence_bytes+len(raw) > 512*MiB:
                raise ValueError("aggregate evidence byte cap")
            self.evidence_bytes += len(raw)
        temporary_final = path.with_name(path.name+".tmp")
        self.register(temporary_final, "temporary", max(len(raw), 1))
        self.register(path, role, max(len(raw), 1))
        with self.io_scope(120):
            with self.files.open_file(temporary_final, "xb") as stream:
                view = memoryview(raw)
                try:
                    self.write_all(stream, view)
                finally:
                    view.release()
                stream.flush()
                stream.fsync()
            self.files.path_action("replace", temporary_final, destination=path)

    def seal(self, path: Path, expected: int, *, finite: bool = False) -> dict[str, Any]:
        digest, size = hashlib.sha256(), 0
        with self.files.stage_deadline(self.deadline(900)), self.files.open_binary(path) as source:
            while True:
                raw = source.read(MiB)
                if not raw:
                    break
                size += len(raw)
                if size > expected or finite and (len(raw) % 4 or any(
                        not math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw))):
                    raise ValueError("RAW/PCM count/alignment/finite readback failure")
                digest.update(raw)
        if size != expected:
            raise ValueError("RAW/PCM short readback")
        record = {"bytes": size, "sha256": digest.hexdigest(), "readback": True}
        self.seals[str(path.relative_to(self.root))] = record
        return record

    def run(self, argv: tuple[str, ...], **kwargs: Any) -> Any:
        return self.tools.run(argv, max_stderr_bytes=16*MiB, **kwargs)

    def admission(self, assembly: CompiledVisualAssembly, audio_groups: int) -> dict[str, int]:
        frames = sum(s.frame_count for s in assembly.shots)
        w, h = assembly.source_size
        raw = 8294400*frames+max((3*w*h*s.frame_count for s in assembly.shots if s.transport), default=0)
        windows = sum(len(source_windows(assembly, s)) for s in assembly.shots)
        # Acquisition/path actions reserve generations; repeated reads/writes use
        # their retained descriptor row. These conservative reservations include
        # repeated pin opens, source pre/post checks, stage readbacks and cleanup.
        generations = (2048+128*windows+64*len(assembly.shots)+64*audio_groups
            +32*len(assembly.brand_inputs)+32*len(assembly.ass_documents))
        paths = (len(self.files.snapshot().paths)+128+8*len(assembly.shots)+2*windows+2*audio_groups
            +4*len(assembly.ass_documents)+6*len(assembly.brand_inputs))
        # One retained RAW sink, one pin/readback descriptor, one independent
        # path-action row. Inputs are claimed as consumer tokens, not descriptors.
        live = 3
        if paths > 32768 or generations > 262144 or live > 16:
            raise ValueError("plan exceeds retained path/generation admission bounds")
        subtotal = raw+4*68_736_000+512*MiB+512*MiB+256*MiB+4*GiB
        margin = max(8*GiB, (subtotal+9)//10)
        self.budget = {"raw_peak": raw, "subtotal": subtotal, "margin": margin,
            "required": subtotal+margin, "path_reservation": paths,
            "generation_reservation": generations, "live_reservation": live}
        return dict(self.budget)

    def recheck_space(self) -> None:
        with self.io_scope(120):
            facts = cast(tuple[int, int, int], self.files.path_action("disk_usage", self.root))
        # Actual stat sizes are conservative credited retained allocations. A
        # fresh FS measurement already includes their occupied bytes.
        retained = sum(p.size_bytes or 0 for p in self.files.snapshot().paths if p.owned and p.present)
        required = max(0, self.budget.get("required", 0)-retained)
        if facts[2] < required:
            raise ValueError(f"insufficient free disk: need {required} bytes, available {facts[2]}; no quality fallback")


class VisualRenderFailure(ValueError):
    def __init__(self, cause: BaseException, host: HostContext) -> None:
        super().__init__(f"unpublished visual render at {host.root}: {cause}")
        self.cause, self.host = cause, host


def _faststart(host: HostContext, candidate: Path, size: int) -> bool:
    remaining, moov, mdat = size, None, None
    with host.files.stage_deadline(host.deadline(900)), host.files.open_binary(candidate) as stream:
        while remaining:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("truncated MP4 atom")
            atom_size, tag = struct.unpack(">I4s", header)
            consumed = 8
            if atom_size == 1:
                extension = stream.read(8)
                if len(extension) != 8:
                    raise ValueError("truncated large MP4 atom")
                atom_size = struct.unpack(">Q", extension)[0]
                consumed = 16
            elif atom_size == 0:
                atom_size = remaining
            if atom_size < consumed or atom_size > remaining:
                raise ValueError("MP4 atom bounds mismatch")
            offset = size-remaining
            if tag == b"moov":
                moov = offset
            if tag == b"mdat" and mdat is None:
                mdat = offset
            skip = atom_size-consumed
            while skip:
                chunk = stream.read(min(MiB, skip))
                if not chunk:
                    raise ValueError("truncated MP4 body")
                skip -= len(chunk)
            remaining -= atom_size
        if stream.read(1):
            raise ValueError("MP4 grew during readback")
    return moov is not None and mdat is not None and moov < mdat


def _verify_candidate(host: HostContext, runtime: rr.RenderRuntime, candidate: Path,
                      frames: int, samples: int, channels: int) -> dict[str, Any]:
    with host.io_scope(120):
        size = host.files.stat_path(candidate).st_size
    if not 0 < size <= 2147483648:
        raise ValueError("actual candidate exceeds accepted MP4 size cap")
    faststart = _faststart(host, candidate, size)
    def probe(args: tuple[str, ...]) -> bytes:
        result = host.tools((str(runtime.ffprobe), "-v", "error", *args, str(candidate)), timeout=45,
            max_stdout_bytes=1_000_000, max_stderr_bytes=1_000_000)
        return result.stdout
    metadata = json.loads(probe(("-count_frames", "-show_streams", "-show_format", "-of", "json")))
    video_pts = parse_frame_observations(probe(("-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json")), audio=False)
    audio = parse_frame_observations(probe(("-select_streams", "a:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time,nb_samples", "-of", "json")), audio=True)
    # Distinct complete output decode, bounded by the existing probe ceiling.
    host.tools.run((str(runtime.ffmpeg), "-hide_banner", "-nostats", "-v", "error", "-xerror",
        "-i", str(candidate), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"),
        kind="probe", timeout=45, max_stdout_bytes=1, max_stderr_bytes=1_000_000,
        expected_stdout_bytes=0, inputs=(candidate,))
    metrics = validate_output_observations(metadata, video_pts, audio, expected_frames=frames,
        expected_samples=samples, channels=channels, actual_size=size, faststart=faststart)
    metrics["content_decode_verification_deadline_seconds"] = 45
    return metrics


def _execute(host: HostContext, source: Path, plan_path: Path, transcript_path: Path | None,
             assets: RenderAssetRegistry, profile: RenderProfile) -> tuple[Path, tuple[Path, ...]]:
    files = host.files
    # Every traversal is declared before any path IO.
    paths = [(source, 2**48), (plan_path, 8_000_000)]
    if transcript_path is not None:
        paths.append((transcript_path, 8_000_000))
    paths.extend((p, 16*MiB) for p in (*assets.fonts.values(), *assets.brand_images.values()))
    paths.extend((p, 512*MiB) for p in assets.sfx.values())
    for path, cap in paths:
        files.register(path, role="original", owned=False, max_bytes=cap)
        for ancestor in path.parents:
            files._register_checked(ancestor)
    plan_bytes = host.read(plan_path, 8_000_000)
    payload = _bounded_json_metadata(plan_bytes.decode("utf-8"), cap=8_000_000, max_depth=24)
    if type(payload) is not dict or payload.get("contract_version") != 2:
        raise ValueError("render-visual-plan requires explicit contract_version 2")
    plan = resolve_visual_plan(VisualCompositionPlan.model_validate(payload))
    transcript = None
    if transcript_path is not None:
        raw_transcript = host.read(transcript_path, 8_000_000)
        payload = _bounded_json_metadata('{"transcript":'+raw_transcript.decode("utf-8")+'}',
            cap=8_000_016, max_depth=24)["transcript"]
        if type(payload) is not list or len(payload) > 32768:
            raise ValueError("bounded transcript list required")
        transcript = [TranscriptSegment.from_dict(item) for item in payload]
    requirements = rr.combine_render_requirements(sd.decode_requirements(), rr.requirements_for_plan(plan))
    for path in rr._runtime_paths(requirements):
        files.register(path, role="runtime", owned=False, max_bytes=268435456)
    with host.io_scope(120):
        runtime = rr.resolve_render_runtime(requirements, device_index=0, _files=files)
        observation_runtime = replace(runtime, device_index=None)
        origin = sd.probe_source(source, observation_runtime, _files=files, _runner=host.tools)
    # Conservative encoded asset ceiling before any generated brand bytes. The
    # compiler's complete generated-text reservation is separately bounded8MB.
    brands = len({s.brand.asset_id for s in plan.shots if s.brand is not None})
    if brands*16*MiB+32*MiB+8_000_000 > 512*MiB:
        raise ValueError("brand/font/generated asset reservation exceeds512MiB")
    with files.stage_deadline(host.deadline(120)):
        if sum(files.stat_path(p).st_size for p in set(assets.sfx.values())) > 512*MiB:
            raise ValueError("registered cue input aggregate exceeds512MiB")
        stage = font_staging.stage_original_fonts(assets, stage_dir=host.root / "fonts", _files=files)
        font_staging.verify_font_stage(stage, _files=files)
        host.font_paths = tuple(f.staged_path for f in stage.fonts)
        overlays = compile_overlay_ass(plan, transcript, assets=stage.registry,
            brand_cache_dir=host.root / "assets", _files=files)
        video = next(s for s in origin.streams if s.kind == "video")
        assembly = compile_visual_assembly(plan, source_size=(video.width, video.height), overlays=overlays)
    with host.io_scope(120):
        preflight = preflight_visual_plan(plan, source=source, assets=stage.registry, profile=profile,
            transcript=transcript, runtime=runtime, _assembly=assembly, _runtime_files=files, _tool_runner=host.tools)
    requirements = rr.combine_render_requirements(requirements, assembly.requirements)
    schedule = schedule_audio(plan)
    host.admission(assembly, len(schedule.spans))
    host.recheck_space()
    finalmix, audio_format, audio_evidence = prepare_audio(host, plan, origin, runtime, stage.registry, requirements)
    rgba, bindings, graphs = render_shots(host, assembly, runtime, origin)
    host.recheck_space()
    listing = host.root / "shots.list"
    raw_list = "".join(f"shots/{p.name}\n" for p in rgba).encode("ascii")
    if len(rgba) > 4096:
        raise ValueError("concatf list line cap")
    host.publish(listing, raw_list, role="graph")
    candidate = host.root / "clip.mp4"
    host.register(candidate, "candidate", 4*GiB)
    with files.stage_deadline(host.deadline(120)):
        font_staging.verify_font_stage(stage, _files=files)
    argv = (str(runtime.ffmpeg), "-hide_banner", "-nostats", "-loglevel", "info", "-n",
        "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", "1080x1920", "-framerate", "60",
        "-protocol_whitelist", "file,concatf", "-i", "concatf:shots.list",
        "-f", "f32le", "-ar", "48000", "-ac", str(audio_format.channels),
        "-channel_layout", audio_format.layout, "-i", str(finalmix), "-map", "0:v:0", "-map", "1:a:0",
        *encoder_arguments(profile), "-fs", "2147483648", str(candidate))
    host.run(argv, kind="encoder", timeout=900, max_stdout_bytes=1, expected_stdout_bytes=0,
        inputs=(*rgba, listing, finalmix), outputs=(candidate,))
    metrics = _verify_candidate(host, runtime, candidate, sum(s.frame_count for s in assembly.shots),
        schedule.total_samples, audio_format.channels)
    with files.stage_deadline(host.deadline(120)):
        sd._check_observation_source(observation_runtime, origin,
            "render48k-mono" if audio_format.channels == 1 else "render48k-stereo", _files=files)
        font_staging.verify_font_stage(stage, _files=files)
    metrics["compiler_resolved"] = json.loads(assembly.compiler_metadata_json)
    metrics["overlay_measured_font_boxes"] = _json_value(overlays.events)
    metrics["selected_file_identity_status"] = stage.selected_file_identity_status
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    spec = FFmpegRenderSpec(source=str(source), output=str(candidate), plan_hash=plan_hash,
        command=argv, filter_complex=";".join(graphs), expected_duration=float(assembly.timeline.spans[-1].output_end),
        profile=profile, assets=tuple(str(p) for p in (*host.font_paths, *(b.asset.path for b in assembly.brand_inputs))))
    source_count = sum(len(b["source_control"].diagnostics) for b in bindings)
    float_count = sum(len(g["source_control"].diagnostics) for g in audio_evidence["groups"])
    norm_count = 0 if audio_evidence["normalization"] is None else len(audio_evidence["normalization"]["passes"])
    metadata = {"version": 2, "status": "rendered", "plan_hash": plan_hash,
        "source_fingerprint": origin.fingerprint, "source_origin": origin.record(),
        "model_id": None, "model_id_reason": "No model ID supplied in validated plan",
        "style_id": None, "style_id_reason": "No style ID supplied in validated plan",
        "plan_provenance": {k: v for k, v in plan.model_dump(mode="json").items() if "fingerprint" in k},
        "runtime": runtime, "observation_runtime": observation_runtime,
        "preflight": preflight, "audio": audio_evidence,
        "source_bindings": bindings, "font_stage": stage, "limits": host.budget,
        "stages": host.tools.evidence, "sealed_files": host.seals,
        "aggregate_child_log_bytes": host.log_bytes,
        "process_count": len(host.tools.evidence)+source_count+float_count+norm_count,
        "process_counts": {"render_tools": len(host.tools.evidence), "source_stream": source_count,
            "source_float": float_count, "normalization": norm_count}, "maximum_active_media_children": 1,
        "publication": "Return after controller verifies owner and file clearance; no persisted final own0"}
    documents = {"visual-plan.json": plan.model_dump(mode="json"), "render-spec.json": spec.model_dump(mode="json"),
        "metadata.json": metadata, "render-metrics.json": metrics}
    retained = [candidate]
    for suffix, payload in documents.items():
        path = candidate.with_suffix("."+suffix)
        host.publish(path, _json_bytes(payload))
        retained.append(path)
    log = candidate.with_suffix(".ffmpeg.log")
    host.publish(log, _json_bytes({"diagnostics": host.tools.evidence,
        "source_stream": [b["source_control"] for b in bindings],
        "source_float": [g["source_control"] for g in audio_evidence["groups"]],
        "normalization": None if audio_evidence["normalization"] is None else audio_evidence["normalization"]["controls"],
        "stderr_representation": "bounded tails; truncation recorded per child"}))
    retained.append(log)
    return candidate, tuple(retained)


def _render_visual_plan_host(source_video: Path, plan_path: Path, output_dir: Path, *,
        transcript_path: Path | None, assets: RenderAssetRegistry, profile: RenderProfile | None) -> Path:
    initial_cwd = Path.cwd()
    def detached(path: Path) -> Path:
        result = path if path.is_absolute() else initial_cwd / path
        if ".." in result.parts:
            raise ValueError("caller paths must not contain parent traversal")
        return result
    source, plan_path, output = (detached(Path(p)) for p in (source_video, plan_path, output_dir))
    transcript_path = None if transcript_path is None else detached(transcript_path)
    assets = RenderAssetRegistry(*({k: detached(v) for k, v in table.items()}
        for table in (assets.brand_images, assets.fonts, assets.sfx)))
    profile = RenderProfile.model_validate((profile or RenderProfile()).model_dump())
    root = output / ("render_"+secrets.token_hex(16))
    state: dict[str, Any] = {}
    def process(argv: tuple[str, ...]) -> Process:
        return cast(Process, subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, text=False, shell=False, close_fds=True, cwd=root))
    def driver_main() -> None:
        retain: tuple[Path, ...] = ()
        try:
            job.register_current_driver()
            with files.stage_deadline(host.deadline(120)):
                for path in (output.parent, *output.parent.parents):
                    files._register_checked(path)
                # A caller-selected output parent must exist. Generated child
                # creation remains exclusive and never replaces an old package.
                if not stat.S_ISDIR(files.stat_path(output.parent).st_mode):
                    raise ValueError("output parent directory missing")
                host.register(output, "directory", 0)
                try:
                    output_stat = cast(os.stat_result, files.path_action("lstat", output))
                    if not stat.S_ISDIR(output_stat.st_mode):
                        raise ValueError("output directory is not a directory")
                except FileNotFoundError:
                    files.path_action("mkdir", output)
                host.register(root, "directory", 0)
                files.path_action("mkdir", root)
                for name in ("shots", "assets", "normalization", "diagnostics"):
                    path = root / name
                    host.register(path, "directory", 0)
                    files.path_action("mkdir", path)
            host.record_progress(0, {"status": "started", "source": str(source), "plan": str(plan_path)})
            host.tools.on_record = host.record_progress
            result, retain = _execute(host, source, plan_path, transcript_path, assets, profile)
            state["result"] = result
        except Exception as error:
            state["error"] = error
            retain = tuple(host.diagnostic_paths)
            if job._failure is None:
                job.fail("validation", "driver")
        finally:
            try:
                cleanup_scope = nullcontext() if job._failure is not None else host.io_scope(5)
                with cleanup_scope:
                    if host.normalization_files is not None:
                        host.normalization_files.retire()
                    files.retire(retain=retain)
            finally:
                job.driver_done()
    driver = threading.Thread(target=driver_main, name="visual-render-driver", daemon=False)
    clock = time.monotonic_ns
    job = SourceHostJob(driver, deadline_ns=clock()+7200*10**9, clock=clock, process_factory=process)
    files = RetainedFiles(job, root=output.parent, clock=clock)
    host = HostContext(job, files, root)
    try:
        try:
            job.start_driver()
        except Exception as startup_error:
            state.setdefault("error", startup_error)
        while True:
            status = job.service_cleanup()
            if status.own0:
                break
            if status.failure is not None and clock() >= status.failure.cleanup_deadline_ns:
                raise ValueError("original cleanup grace exhausted; resources retained")
            driver.join(.01)
        if ("error" in state or status.failure is not None or not files.snapshot().files_clear
                or host.normalization_files is not None and not host.normalization_files.snapshot().files_clear):
            raise state.get("error", ValueError("render ownership/file clearance failed"))
        return cast(Path, state["result"])
    except Exception as error:
        raise VisualRenderFailure(error, host) from error
