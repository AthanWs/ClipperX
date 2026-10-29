"""Provider-free immutable host provenance and exact observed audio slices."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any
import wave
import zlib

from .local_paths import source_fingerprint
from .render_clock import SourceAudioDecodeReceipt, SourceDecodeReceipt, SourceTimeOrigin
from .render_clock import canonical, digest, exact, rational
from .visual_contracts import SourceVisualTimeline


OBSERVED_ANALYZER_VERSION = "source-visual-local-v8-observed"
MAX_SCHEDULE_POINTS = 50_000
MAX_CATALOG_ARTIFACTS = 100_000
MAX_ARTIFACT_BYTES = 64_000_000
MAX_GENERATION_BYTES = 16_000_000_000


def emitted_text_size(text: str) -> int:
    """Actual UTF-8 bytes from the existing platform text writer (newline=None)."""
    return len(text.encode("utf-8")) + text.count("\n") * (len(os.linesep) - 1)


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate generation JSON field")
        result[key] = value
    return result


def read_generation_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Missing or oversized generation JSON")
    with path.open("rb") as handle:
        encoded = handle.read(MAX_ARTIFACT_BYTES + 1)
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("Oversized generation JSON")
    data = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_fields,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if type(data) is not dict:
        raise ValueError("Generation JSON must be an object")
    canonical(data)  # also rejects nonfinite overflowing floats
    return data


def _schema(data: object, fields: set[str], label: str) -> dict[str, Any]:
    if type(data) is not dict or set(data) != fields:
        raise ValueError(f"Invalid {label} schema")
    return data


def _sha(value: object) -> str:
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid SHA256 identity")
    return value


def _integer(value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid generation integer")
    return value


def generation_file(root: Path, relative: object) -> Path:
    if (type(relative) is not str or not relative or "\\" in relative or ":" in relative
            or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).is_absolute()
            or any(part in ("..", ".", "") for part in relative.split("/"))):
        raise ValueError("Unsafe generation artifact path")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Generation artifact escapes through symlink/junction")
    return target


def generation_catalog(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "complete.json":
            continue
        checked = generation_file(root, relative)
        size = checked.stat().st_size
        total += size
        if size > MAX_ARTIFACT_BYTES or total > MAX_GENERATION_BYTES or len(rows) >= MAX_CATALOG_ARTIFACTS:
            raise ValueError("Generation artifact/catalog budget exceeded")
        rows.append({"path": relative, "bytes": size, "sha256": file_sha256(checked)})
    return rows


def _verify_wav(path: Path, receipt: SourceAudioDecodeReceipt) -> None:
    audio = json.loads(receipt.payload_json)["audio"][0]
    first = audio["output"][0]
    aggregate = hashlib.sha256()
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()) != (
                first["channels"], 2, first["rate"], audio["samples"]):
            raise ValueError("WAV sample layout/count conflicts with observation")
        for block in audio["output"]:
            body = handle.readframes(block["samples"])
            if (len(body) != block["bytes"] or digest(body) != block["sha256"]
                    or f"{zlib.adler32(body, 0):08X}" != block["checksum"]):
                raise ValueError("WAV body/block identity conflicts with observation")
            aggregate.update(body)
        if handle.readframes(1) or aggregate.hexdigest() != audio["pcm_sha256"]:
            raise ValueError("WAV body aggregate identity conflicts with observation")


def source_time(value: object) -> Fraction:
    """Convert external floats once, through decimal text, into the exact domain."""
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Source time must be finite")
        return exact(str(value))
    return exact(value)


class AudioUnavailableError(ValueError):
    """Valid receipt cannot cover a nonempty requested sample slice."""


@dataclass(frozen=True, slots=True)
class AudioSampleSlice:
    observation_id: str
    first: int
    count: int
    rate: int
    requested_start: Fraction
    requested_end: Fraction
    included_start: Fraction
    included_end: Fraction
    start_delta: Fraction
    end_delta: Fraction

    def to_record(self) -> dict[str, Any]:
        return {name: rational(value) if isinstance(value, Fraction) else value
                for name in self.__dataclass_fields__ if (value := getattr(self, name)) is not None}


def map_source_audio_window(
    receipt: SourceAudioDecodeReceipt, origin: SourceTimeOrigin,
    start: object, end: object,
) -> AudioSampleSlice:
    receipt.validate_origin(origin)
    a, b = source_time(start), source_time(end)
    data = json.loads(receipt.payload_json)
    audio = data["audio"][0]
    phase = exact(audio["source_phase"])
    rate, total = audio["output"][0]["rate"], audio["samples"]
    if (not 0 <= a < b <= origin.duration
            or not any(exact(w[0]) <= a < b <= exact(w[1]) for w in data["windows"])
            or a < phase or b > phase + Fraction(total, rate)):
        raise AudioUnavailableError("Requested unrounded audio coverage unavailable")
    first = math.floor((a - phase) * rate + Fraction(1, 2))
    last = math.floor((b - phase) * rate + Fraction(1, 2))
    if not 0 <= first < last <= total:
        raise AudioUnavailableError("Requested audio slice is empty or outside the observed grid")
    included_start, included_end = phase + Fraction(first, rate), phase + Fraction(last, rate)
    return AudioSampleSlice(json.loads(receipt.to_json())["sha256"], first, last - first, rate,
                            a, b, included_start, included_end, included_start - a, included_end - b)


def visual_timeline_fingerprint(timeline: SourceVisualTimeline) -> str:
    return digest(json.dumps(timeline.model_dump(mode="json"), ensure_ascii=False,
                             sort_keys=True, allow_nan=False))


def supported_unavailable_intervals(
    origin: SourceTimeOrigin, start: Fraction, end: Fraction,
    receipts: Iterable[SourceDecodeReceipt], *, no_samples: bool,
) -> tuple[tuple[Fraction, Fraction, str], ...]:
    """One exact absence policy shared by generation masking and its loader."""
    video = next(s for s in origin.streams if s.kind == "video")
    intervals: list[tuple[Fraction, Fraction, str]] = []
    requested: set[Fraction] = set()
    covered: set[Fraction] = set()
    def add(a: Fraction, b: Fraction, reason: str) -> None:
        if start <= a < b <= end:
            intervals.append((a, b, reason))
    if video.start is not None:
        add(start, min(end, video.start - origin.origin), "before_video_start")
    if video.end is not None:
        add(max(start, video.end - origin.origin), end, "after_video_end")
    for receipt in receipts:
        receipt.validate_origin(origin)
        data = json.loads(receipt.payload_json)
        windows = [(exact(w[0]), exact(w[1])) for w in data["windows"]]
        if any(not start <= a < b <= end for a, b in windows):
            raise ValueError("Observed video window outside generation scope")
        for outcome in data["outcomes"]:
            point = exact(outcome["requested"])
            window = next(((a, b) for a, b in windows if a <= point < b), None)
            if point in requested or window is None:
                raise ValueError("Duplicate or out-of-window observed request")
            requested.add(point)
            a, b = window
            status = outcome["status"]
            if status == "covered":
                covered.add(point)
            elif status == "before_video_start":
                if video.start is None or origin.to_raw(point) >= video.start:
                    raise ValueError("Before-start outcome contradicts selected stream bounds")
            elif status == "missing_predecessor":
                if data["video"]:
                    first = data["video"][0]
                    first_at = first["pts"] * exact(first["time_base"]) - origin.origin
                    if point >= first_at:
                        raise ValueError("Missing predecessor contradicts observed video")
                    add(a, min(b, first_at), status)
                else:
                    add(a, b, status)
            elif status == "unknown_terminal":
                if not data["video"]:
                    raise ValueError("Terminal outcome requires its observed final frame")
                last = data["video"][-1]
                last_at = last["pts"] * exact(last["time_base"]) - origin.origin
                if point < last_at or last["coverage_end"] is not None:
                    raise ValueError("Terminal outcome contradicts observed video")
                add(max(a, last_at), b, status)
            else:
                raise ValueError("Unsupported negative video outcome")
    if not requested:
        raise ValueError("Observed generation requires actual sampled requests")
    if no_samples:
        if covered:
            raise ValueError("No-samples scope contradicts covered receipt requests")
        return ((start, end, "no_observed_samples"),)
    merged: list[tuple[Fraction, Fraction, str]] = []
    for a, b, reason in sorted(set(intervals)):
        if merged and a <= merged[-1][1]:
            old_a, old_b, old_reason = merged[-1]
            merged[-1] = old_a, max(b, old_b), ";".join(sorted(set(old_reason.split(";") + reason.split(";"))))
        else:
            merged.append((a, b, reason))
    ordered_covered, cursor = sorted(covered), 0
    for a, b, _ in merged:
        while cursor < len(ordered_covered) and ordered_covered[cursor] < a:
            cursor += 1
        if cursor < len(ordered_covered) and ordered_covered[cursor] < b:
            raise ValueError("Unavailable interval contradicts a covered request")
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class VisualProvenanceContext:
    status: str
    record_json: str

    def to_record(self) -> dict[str, Any]:
        return json.loads(self.record_json)

    @property
    def timeline_fingerprint(self) -> str:
        return str(self.to_record()["timeline_fingerprint"])

    @property
    def source_fingerprint(self) -> str:
        return str(self.to_record()["source_fingerprint"])

    @property
    def generation_id(self) -> str | None:
        return self.to_record()["generation_id"]

    @property
    def scope(self) -> dict[str, str] | None:
        return self.to_record()["scope"]

    @property
    def complete_sha256(self) -> str | None:
        return self.to_record()["complete_sha256"]

    @property
    def clock_receipt_sha256(self) -> str | None:
        return self.to_record()["clock_receipt_sha256"]


def validate_observed_generation(
    timeline_path: Path, *, source_video: Path, current_origin: SourceTimeOrigin,
    completion: dict[str, Any],
) -> VisualProvenanceContext:
    """Validate exactly this generation; also used before its commit marker exists."""
    root = timeline_path.parent
    marker = _schema(completion, {"version", "kind", "generation_id", "source_fingerprint", "origin_id",
        "config", "scope", "analyzer_version", "timeline_fingerprint", "timeline_sha256",
        "clock_receipt_sha256", "catalog", "creation_work", "sha256"}, "completion")
    unsigned = {k: v for k, v in marker.items() if k != "sha256"}
    if _sha(marker["sha256"]) != digest(canonical(unsigned)):
        raise ValueError("Completion self-hash mismatch")
    if (type(marker["version"]) is not int or marker["version"] != 1
            or marker["kind"] != "observed_visual_generation"
            or marker["generation_id"] != root.name or not re.fullmatch(r"[0-9a-f]{32}", root.name)
            or marker["analyzer_version"] != OBSERVED_ANALYZER_VERSION):
        raise ValueError("Invalid observed generation identity")
    if (current_origin.path != str(source_video.resolve())
            or source_fingerprint(source_video) != current_origin.fingerprint
            or marker["source_fingerprint"] != current_origin.fingerprint
            or marker["origin_id"] != current_origin.identity):
        raise ValueError("Generation source/current origin conflict")
    scope = _schema(marker["scope"], {"kind", "start", "end"}, "scope")
    if (scope["kind"] not in ("full", "window") or any(type(scope[k]) is not str for k in scope)
            or not 0 <= exact(scope["start"]) < exact(scope["end"]) <= current_origin.duration):
        raise ValueError("Invalid generation scope")
    if scope["kind"] == "full" and (exact(scope["start"]) != 0 or exact(scope["end"]) != current_origin.duration):
        raise ValueError("Conflicting full scope")
    config = _schema(marker["config"], {"sampling", "model_hashes", "dependencies", "legacy_context"}, "config")
    if any(type(config[k]) is not dict for k in config):
        raise ValueError("Invalid analyzer config")
    sampling = config["sampling"]
    if any(type(k) is not str or type(v) is not float or not math.isfinite(v) for k, v in sampling.items()):
        raise ValueError("Sampling configuration requires exact finite emitted float types")
    if scope["kind"] == "window":
        if (set(sampling) != {"inspection_start_seconds", "inspection_end_seconds", "inspection_interval_seconds"}
                or sampling["inspection_start_seconds"] != float(exact(scope["start"]))
                or sampling["inspection_end_seconds"] != float(exact(scope["end"]))
                or sampling["inspection_interval_seconds"] <= 0):
            raise ValueError("Invalid inspection sampling configuration")
    else:
        count = sampling.get("editorial_boundary_count", -1.)
        if count < 0 or count != int(count) or count > MAX_SCHEDULE_POINTS:
            raise ValueError("Invalid editorial boundary count")
        if (set(sampling) != {"sparse_interval_seconds", "dense_interval_seconds", "dense_radius_seconds",
                "editorial_boundary_count", *(f"editorial_boundary_{i:04d}" for i in range(int(count)))}
                or sampling["sparse_interval_seconds"] <= 0 or sampling["dense_interval_seconds"] <= 0
                or sampling["dense_radius_seconds"] != 3.0):
            raise ValueError("Invalid full analyzer sampling configuration")
    if (set(config["dependencies"]) != {"opencv", "numpy"}
            or any(type(v) is not str or not v for v in config["dependencies"].values())
            or set(config["legacy_context"]) != {"transcript", "diarization", "streamer_identity", "global_streamer_identity"}
            or set(config["model_hashes"]) != {"yunet", "sface"}):
        raise ValueError("Invalid analyzer/dependency/context identities")
    for value in config["model_hashes"].values():
        _sha(value)
    for label, entry in config["legacy_context"].items():
        _schema(entry, {"status", "sha256"}, "legacy context")
        if entry["status"] not in ("missing", "unverified_legacy"):
            raise ValueError("Legacy text/profile cannot be certified by observed PCM")
        if entry["status"] == "unverified_legacy":
            _sha(entry["sha256"])
        elif entry["sha256"] is not None:
            raise ValueError("Missing legacy context has a digest")
    catalog = marker["catalog"]
    if type(catalog) is not list or not 2 <= len(catalog) <= MAX_CATALOG_ARTIFACTS:
        raise ValueError("Invalid catalog length")
    catalog_by_path: dict[str, dict[str, Any]] = {}
    total = 0
    for row in catalog:
        _schema(row, {"path", "bytes", "sha256"}, "catalog row")
        target = generation_file(root, row["path"])
        size = _integer(row["bytes"])
        total += size
        if (row["path"] in catalog_by_path or row["path"] == "complete.json"
                or size > MAX_ARTIFACT_BYTES or total > MAX_GENERATION_BYTES
                or not target.is_file() or target.stat().st_size != size
                or file_sha256(target) != _sha(row["sha256"])):
            raise ValueError("Catalog duplicate/size/hash conflict")
        catalog_by_path[row["path"]] = row
    complete_path = root / "complete.json"
    complete_bytes = complete_path.stat().st_size if complete_path.is_file() else emitted_text_size(canonical(marker) + "\n")
    if complete_bytes > MAX_ARTIFACT_BYTES or total + complete_bytes > MAX_GENERATION_BYTES:
        raise ValueError("Completion-inclusive generation byte budget exceeded")
    if set(catalog_by_path) != {p.relative_to(root).as_posix() for p in root.rglob("*")
                               if p.is_file() and p.name != "complete.json"}:
        raise ValueError("Catalog does not exactly cover generation artifacts")
    if (timeline_path.name != "source_visual_timeline.json" or timeline_path.name not in catalog_by_path
            or catalog_by_path[timeline_path.name]["sha256"] != marker["timeline_sha256"]
            or "clock-receipt.json" not in catalog_by_path
            or catalog_by_path["clock-receipt.json"]["sha256"] != marker["clock_receipt_sha256"]):
        raise ValueError("Completion timeline/clock file association conflict")
    raw_timeline = read_generation_json(timeline_path)
    timeline = SourceVisualTimeline.model_validate(raw_timeline)
    if canonical(raw_timeline) != canonical(timeline.model_dump(mode="json")):
        raise ValueError("Observed timeline must match its exact emitted contract schema/types")
    if (timeline.analyzer_version != OBSERVED_ANALYZER_VERSION
            or timeline.source_fingerprint != current_origin.fingerprint
            or timeline.duration_seconds != float(current_origin.duration)
            or timeline.model_hashes != config["model_hashes"] or timeline.sampling_config != config["sampling"]
            or visual_timeline_fingerprint(timeline) != marker["timeline_fingerprint"]):
        raise ValueError("Timeline fingerprint/config/origin conflict")
    clock = _schema(read_generation_json(root / "clock-receipt.json"), {"version", "kind", "generation_id",
        "origin", "origin_id", "scope", "config", "observations", "sample_associations", "audio_associations",
        "unavailable_attempts", "unavailable_intervals"}, "clock receipt")
    if (type(clock["version"]) is not int or clock["version"] != 1 or clock["kind"] != "observed_visual_evidence"
            or clock["generation_id"] != root.name or canonical(clock["origin"]) != canonical(current_origin.record())
            or clock["origin_id"] != current_origin.identity or clock["scope"] != scope or clock["config"] != config):
        raise ValueError("Clock receipt generation/origin/config conflict")
    if any(type(clock[k]) is not list or len(clock[k]) > MAX_SCHEDULE_POINTS for k in (
            "observations", "sample_associations", "audio_associations", "unavailable_attempts", "unavailable_intervals")):
        raise ValueError("Clock receipt list budget/schema conflict")
    receipts: dict[str, SourceDecodeReceipt | SourceAudioDecodeReceipt] = {}
    referenced = {timeline_path.name, "clock-receipt.json"}
    for row in clock["observations"]:
        _schema(row, {"path", "kind", "observation_id", "wav"}, "observation reference")
        if row["path"] in receipts or row["path"] not in catalog_by_path:
            raise ValueError("Duplicate/missing observed receipt")
        path = generation_file(root, row["path"])
        envelope = read_generation_json(path)
        if envelope.get("sha256") != _sha(row["observation_id"]):
            raise ValueError("Observed receipt envelope identity conflict")
        if row["kind"] == "audio":
            receipt: SourceDecodeReceipt | SourceAudioDecodeReceipt = SourceAudioDecodeReceipt.from_json(
                canonical(envelope), origin=current_origin, policy=envelope["policy"])
            if row["wav"] not in catalog_by_path:
                raise ValueError("Missing generation PCM WAV")
            assert isinstance(receipt, SourceAudioDecodeReceipt)
            _verify_wav(generation_file(root, row["wav"]), receipt)
            referenced.add(row["wav"])
        elif row["kind"] == "video" and row["wav"] is None:
            receipt = SourceDecodeReceipt.from_json(canonical(envelope),
                                                    origin=current_origin, policy=envelope["policy"])
        else:
            raise ValueError("Invalid observation reference kind")
        receipts[row["path"]] = receipt
        referenced.add(row["path"])
    samples = {sample.sample_id: sample for sample in timeline.samples}
    associated: set[str] = set()
    for row in clock["sample_associations"]:
        _schema(row, {"sample_id", "requested", "receipt", "frame_n", "raw_pts", "time_base", "coverage_start",
            "coverage_end", "raw_bgr_sha256", "analysis_sha256", "analysis_width", "analysis_height",
            "transform", "preview", "preview_sha256"}, "sample association")
        sample = samples.get(row["sample_id"])
        video_receipt = receipts.get(row["receipt"])
        if sample is None or row["sample_id"] in associated or not isinstance(video_receipt, SourceDecodeReceipt):
            raise ValueError("Missing/duplicate sample receipt association")
        requested = exact(row["requested"])
        if float(requested) != sample.at or not exact(scope["start"]) <= requested < exact(scope["end"]):
            raise ValueError("Sample requested timestamp conflict")
        payload = json.loads(video_receipt.payload_json)
        frame_n = _integer(row["frame_n"])
        if frame_n >= len(payload["video"]):
            raise ValueError("Unknown observed frame")
        frame = payload["video"][frame_n]
        if (not any(exact(o["requested"]) == requested and o["frame_n"] == frame_n and o["status"] == "covered"
                    for o in payload["outcomes"])
                or row["raw_pts"] != frame["pts"] or row["time_base"] != frame["time_base"]
                or exact(row["coverage_start"]) != frame["pts"] * exact(frame["time_base"]) - current_origin.origin
                or exact(row["coverage_end"]) != exact(frame["coverage_end"]) - current_origin.origin
                or row["raw_bgr_sha256"] != frame["sha256"]):
            raise ValueError("Observed frame/sample coverage association conflict")
        width, height = frame["width"], frame["height"]
        expected_width = min(960, width)
        expected_height = height if width <= 960 else max(1, round(height * 960 / width))
        if (_integer(row["analysis_width"], 1) != expected_width or _integer(row["analysis_height"], 1) != expected_height
                or row["transform"] != {"maximum_width": 960, "resize": "INTER_AREA", "preview": "JPEG90"}
                or row["preview"] != f"sample-{int(round(sample.at * 1_000_000)):012d}.jpg"
                or row["preview"] not in catalog_by_path
                or row["preview_sha256"] != catalog_by_path[row["preview"]]["sha256"]):
            raise ValueError("Analysis/preview transform association conflict")
        _sha(row["analysis_sha256"])
        associated.add(row["sample_id"])
        referenced.add(row["preview"])
    if associated != set(samples):
        raise ValueError("Every visual sample requires observed coverage")
    if not samples and (timeline.face_tracks or timeline.webcam_tracks):
        raise ValueError("No-samples scope cannot have unobserved face/webcam tracks")
    expected_covered = {exact(o["requested"]) for receipt in receipts.values() if isinstance(receipt, SourceDecodeReceipt)
                        for o in json.loads(receipt.payload_json)["outcomes"] if o["status"] == "covered"}
    if {exact(row["requested"]) for row in clock["sample_associations"]} != expected_covered:
        raise ValueError("Covered receipt requests were deleted from the generation samples")
    audio_associated: set[str] = set()
    intended_audio = {sample.sample_id: (source_time(sample.at),
        source_time(timeline.samples[index + 1].at) if index + 1 < len(timeline.samples) else exact(scope["end"]))
        for index, sample in enumerate(timeline.samples)}
    for row in clock["audio_associations"]:
        _schema(row, {"sample_id", "status", "receipt", "slice", "reason"}, "audio association")
        sample = samples.get(row["sample_id"])
        if sample is None or row["sample_id"] in audio_associated:
            raise ValueError("Unknown/duplicate audio sample association")
        audio_receipt = receipts.get(row["receipt"])
        if not isinstance(audio_receipt, SourceAudioDecodeReceipt):
            raise ValueError("Audio availability requires its actual audio-only receipt")
        intended_start, intended_end = intended_audio[sample.sample_id]
        if row["status"] == "covered":
            if row["reason"] is not None:
                raise ValueError("Audio slice needs its audio-only receipt")
            _schema(row["slice"], set(AudioSampleSlice.__dataclass_fields__), "audio sample slice")
            mapped = map_source_audio_window(audio_receipt, current_origin,
                intended_start, intended_end)
            if canonical(mapped.to_record()) != canonical(row["slice"]) or sample.audio_window is None:
                raise ValueError("Audio sample-grid mapping conflict")
            if (sample.audio_features.get("start") != float(mapped.included_start)
                    or sample.audio_features.get("end") != float(mapped.included_end)
                    or sample.audio_window.start != float(mapped.requested_start)
                    or sample.audio_window.end != float(mapped.requested_end)):
                raise ValueError("Audio numerical timestamp association conflict")
        elif row["status"] == "unavailable":
            if (sample.audio_features or sample.audio_window is not None or row["slice"] is not None
                    or type(row["reason"]) is not str or not row["reason"]):
                raise ValueError("Unavailable audio cannot have invented measurements")
            try:
                map_source_audio_window(audio_receipt, current_origin, intended_start, intended_end)
            except AudioUnavailableError as exc:
                if row["reason"] != str(exc):
                    raise ValueError("Unavailable audio reason contradicts its exact mapper outcome") from exc
            else:
                raise ValueError("Unavailable audio contradicts covered complete feature interval")
        else:
            raise ValueError("Invalid audio coverage status")
        audio_associated.add(row["sample_id"])
    if audio_associated != set(samples):
        raise ValueError("Missing audio availability metadata")
    attempted: set[str] = set()
    expected_attempts = {(name, o["requested"], o["status"]) for name, receipt in receipts.items()
        if isinstance(receipt, SourceDecodeReceipt) for o in json.loads(receipt.payload_json)["outcomes"] if o["status"] != "covered"}
    for row in clock["unavailable_attempts"]:
        _schema(row, {"requested", "status", "receipt"}, "unavailable attempt")
        attempt_receipt = receipts.get(row["receipt"])
        if not isinstance(attempt_receipt, SourceDecodeReceipt) or row["requested"] in attempted:
            raise ValueError("Unavailable attempt duplicate/missing receipt")
        if row["status"] == "covered" or not any(o == {"requested": row["requested"], "status": row["status"], "frame_n": None}
                for o in json.loads(attempt_receipt.payload_json)["outcomes"]):
            raise ValueError("Unavailable attempt contradicts observed receipt")
        attempted.add(row["requested"])
    if {(row["receipt"], row["requested"], row["status"]) for row in clock["unavailable_attempts"]} != expected_attempts:
        raise ValueError("Negative receipt requests lack their exact attempt associations")
    supported = supported_unavailable_intervals(current_origin, exact(scope["start"]), exact(scope["end"]),
        (receipt for receipt in receipts.values() if isinstance(receipt, SourceDecodeReceipt)), no_samples=not samples)
    expected_intervals = [{"start": rational(a), "end": rational(b), "reason": reason} for a, b, reason in supported]
    if canonical(clock["unavailable_intervals"]) != canonical(expected_intervals):
        raise ValueError("Unavailable intervals are unsupported, duplicated or conflicting")
    for row in clock["unavailable_intervals"]:
        _schema(row, {"start", "end", "reason"}, "unavailable interval")
        a, b = exact(row["start"]), exact(row["end"])
        if (not exact(scope["start"]) <= a < b <= exact(scope["end"])
                or any(type(v) is not str or not v for v in row.values())):
            raise ValueError("Invalid unavailable interval")
        # Span schema remains v1 floats. Observed request/coverage and receipt
        # intervals above are checked exactly; match their float projections here.
        spans = [s for s in timeline.spans if s.source.start < float(b) and float(a) < s.source.end]
        if not spans or any(s.state.value != "unknown" or s.confidence != 0 or s.streamer_track_id is not None
            or s.webcam_track_id is not None or s.important_regions or s.text_regions
            or s.identity_status.value != "unavailable"
            or s.change_reasons != ["observed_unavailable:" + row["reason"]] for s in spans):
            raise ValueError("Unavailable scope must remain UNKNOWN after finalization")
        cursor = float(a)
        for span in spans:
            if span.source.start > cursor:
                raise ValueError("Unavailable UNKNOWN masking has a coverage hole")
            cursor = max(cursor, min(float(b), span.source.end))
        if cursor < float(b):
            raise ValueError("Unavailable UNKNOWN masking is incomplete")
    work = _schema(marker["creation_work"], {"indexed_samples"}, "creation work")
    if _integer(work["indexed_samples"]) != len(samples) or len(samples) > MAX_SCHEDULE_POINTS:
        raise ValueError("Generation creation work conflicts with analyzed sample count")
    if referenced != set(catalog_by_path):
        raise ValueError("Unassociated generation catalog artifact")
    record = {"status": "observed_samples", "source_fingerprint": current_origin.fingerprint,
        "timeline_fingerprint": marker["timeline_fingerprint"], "scope": scope, "generation_id": root.name,
        "complete_sha256": marker["sha256"], "clock_receipt_sha256": marker["clock_receipt_sha256"],
        "origin_id": current_origin.identity, "timeline_sha256": marker["timeline_sha256"]}
    return VisualProvenanceContext("observed_samples", canonical(record))


def load_visual_provenance(
    timeline_path: Path, *, source_video: Path,
    expected_timeline_fingerprint: str | None = None,
    current_origin: SourceTimeOrigin | None = None,
) -> VisualProvenanceContext:
    timeline_path, source_video = Path(timeline_path), Path(source_video)
    timeline = SourceVisualTimeline.model_validate(read_generation_json(timeline_path))
    fingerprint = visual_timeline_fingerprint(timeline)
    if expected_timeline_fingerprint is not None and fingerprint != expected_timeline_fingerprint:
        raise ValueError("Timeline fingerprint conflict")
    marked = (timeline.analyzer_version == OBSERVED_ANALYZER_VERSION
              or (timeline_path.parent / "complete.json").exists()
              or (timeline_path.parent / "clock-receipt.json").exists()
              or "visual_observed" in timeline_path.parts)
    if marked:
        if current_origin is None:
            raise ValueError("Observed generation requires current_origin")
        return validate_observed_generation(timeline_path, source_video=source_video, current_origin=current_origin,
                                             completion=read_generation_json(timeline_path.parent / "complete.json"))
    record = {"status": "unverified_legacy", "source_fingerprint": timeline.source_fingerprint,
              "timeline_fingerprint": fingerprint, "scope": None, "generation_id": None,
              "complete_sha256": None, "clock_receipt_sha256": None}
    return VisualProvenanceContext("unverified_legacy", json.dumps(record))
