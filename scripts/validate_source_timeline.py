"""Provider-free validation runner for source visual timelines.

Portable expectations live in ``data/validation_visual/scenarios.json``. Exact
video paths and human-confirmed windows belong in the ignored local mapping.
The runner builds each distinct source once, then evaluates every mapped window
against the resulting strict ``SourceVisualTimeline`` contract.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable, Sequence, cast
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clipperx.local_paths import atomic_write_text, source_job_root  # noqa: E402
from clipperx.media import probe_media  # noqa: E402
from clipperx.visual_contracts import (  # noqa: E402
    IdentityStatus,
    SourceState,
    SourceVisualTimeline,
)
from clipperx.visual_timeline import (  # noqa: E402
    VisualTimelineBuildResult, build_source_visual_timeline, build_source_visual_timeline_result,
)

_DEFAULT_PUBLIC_BUILDER = build_source_visual_timeline


EXPECTED_SCENARIO_IDS = frozenset(
    {
        "stable_gameplay_webcam",
        "full_camera",
        "full_camera_to_browser",
        "browser_to_gameplay",
        "foreign_faces_in_content",
        "multi_person",
        "no_webcam",
        "reacted_vertical_video",
        "browser_small_text",
        "webcam_move_or_disappear",
    }
)
ASSERTION_KEYS = frozenset(
    {
        "expected_states",
        "ordered_states",
        "event_transition",
        "forbid_streamer_switch",
        "forbid_webcam_track",
        "require_text_regions",
        "require_multiple_faces",
        "require_vertical_region",
    }
)
REPORT_STATUSES = ("pass", "fail", "ambiguous", "unavailable")
_COVERAGE_EPSILON = 0.001


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    assertions: dict[str, object]


@dataclass(frozen=True)
class ScenarioManifest:
    scenarios: tuple[Scenario, ...]

    @property
    def by_id(self) -> dict[str, Scenario]:
        return {scenario.scenario_id: scenario for scenario in self.scenarios}


@dataclass(frozen=True)
class SourceMapping:
    scenario_id: str
    source_reference: str
    source_path: Path
    start: float
    end: float
    event_at: float | None = None


@dataclass(frozen=True)
class BuildMetadata:
    artifact_path: Path
    cache_hit: bool
    indexed_samples_this_run: int | None

    def __post_init__(self) -> None:
        frames = self.indexed_samples_this_run
        if frames is not None and (isinstance(frames, bool) or frames < 0):
            raise ValueError("indexed_samples_this_run debe ser >= 0 o null")
        if self.cache_hit and frames not in (None, 0):
            raise ValueError("cache_hit no puede reportar frames decodificados")


Builder = Callable[..., Path | BuildMetadata | VisualTimelineBuildResult]


@dataclass(frozen=True)
class ValidationRunResult:
    exit_code: int
    summary: dict[str, object]


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{label} no existe: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} no es JSON válido: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} debe ser un objeto JSON")
    return payload


def _require_version_one(payload: dict[str, object], label: str) -> None:
    if payload.get("version") != 1:
        raise ValueError(f"{label}.version debe ser 1")


def _validate_assertion(name: str, value: object, scenario_id: str) -> None:
    if name in {"expected_states", "ordered_states"}:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{scenario_id}.{name} debe ser una lista no vacía")
        allowed = {state.value for state in SourceState}
        if any(not isinstance(item, str) or item not in allowed for item in value):
            raise ValueError(f"{scenario_id}.{name} contiene un SourceState inválido")
        return
    if name == "event_transition":
        if not isinstance(value, dict):
            raise ValueError(f"{scenario_id}.{name} debe ser un objeto")
        kind = value.get("kind")
        if kind == "state":
            expected_keys = {"kind", "within_seconds", "from_state", "to_state"}
            allowed_states = {state.value for state in SourceState}
            if set(value) != expected_keys:
                raise ValueError(f"{scenario_id}.{name} state tiene campos inválidos")
            if value.get("from_state") not in allowed_states or value.get("to_state") not in allowed_states:
                raise ValueError(f"{scenario_id}.{name} contiene un SourceState inválido")
            if value.get("from_state") == value.get("to_state"):
                raise ValueError(f"{scenario_id}.{name} requiere estados distintos")
        elif kind == "main_content":
            expected_keys = {"kind", "within_seconds", "minimum_change_score"}
            if set(value) != expected_keys:
                raise ValueError(f"{scenario_id}.{name} main_content tiene campos inválidos")
            score = value.get("minimum_change_score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0 < float(score) <= 1
            ):
                raise ValueError(f"{scenario_id}.{name}.minimum_change_score debe estar en (0, 1]")
        else:
            raise ValueError(f"{scenario_id}.{name}.kind no soportado")
        within = value.get("within_seconds")
        if (
            isinstance(within, bool)
            or not isinstance(within, (int, float))
            or not math.isfinite(float(within))
            or float(within) <= 0
        ):
            raise ValueError(f"{scenario_id}.{name}.within_seconds debe ser numérico > 0")
        return
    if value is not True:
        raise ValueError(f"{scenario_id}.{name} debe ser true para ser ejecutable")

def load_scenario_manifest(path: Path) -> ScenarioManifest:
    payload = _read_json(Path(path), "scenario manifest")
    _require_version_one(payload, "scenario manifest")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ValueError("scenario manifest.scenarios debe ser una lista")

    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_scenarios):
        if not isinstance(raw, dict):
            raise ValueError(f"scenario[{index}] debe ser un objeto")
        scenario_id = raw.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"scenario[{index}].id debe ser texto no vacío")
        if scenario_id in seen:
            raise ValueError(f"scenario id duplicado: {scenario_id}")
        seen.add(scenario_id)
        assertions = {key: value for key, value in raw.items() if key in ASSERTION_KEYS}
        if not assertions:
            raise ValueError(f"{scenario_id} requiere al menos una assertion ejecutable")
        unexpected = set(raw) - {"id"} - ASSERTION_KEYS
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"{scenario_id} contiene campos no ejecutables: {names}")
        for name, value in assertions.items():
            _validate_assertion(name, value, scenario_id)
        scenarios.append(Scenario(scenario_id=scenario_id, assertions=assertions))

    actual_ids = {scenario.scenario_id for scenario in scenarios}
    if len(scenarios) != 10 or actual_ids != EXPECTED_SCENARIO_IDS:
        missing = sorted(EXPECTED_SCENARIO_IDS - actual_ids)
        extra = sorted(actual_ids - EXPECTED_SCENARIO_IDS)
        raise ValueError(
            "scenario manifest debe contener exactamente los diez IDs; "
            f"missing={missing}, extra={extra}"
        )
    return ScenarioManifest(scenarios=tuple(scenarios))


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} debe ser numérico")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} debe ser finito")
    return result


def load_source_mappings(
    path: Path, manifest: ScenarioManifest
) -> tuple[SourceMapping, ...]:
    path = Path(path)
    payload = _read_json(path, "sources mapping")
    _require_version_one(payload, "sources mapping")
    raw_mappings = payload.get("mappings")
    if not isinstance(raw_mappings, list):
        raise ValueError("sources mapping.mappings debe ser una lista")

    mappings: list[SourceMapping] = []
    seen: set[str] = set()
    known_ids = set(manifest.by_id)
    for index, raw in enumerate(raw_mappings):
        if not isinstance(raw, dict):
            raise ValueError(f"mapping[{index}] debe ser un objeto")
        unexpected = set(raw) - {"scenario_id", "source", "start", "end", "event_at"}
        if unexpected:
            raise ValueError(f"mapping[{index}] contiene campos desconocidos")
        scenario_id = raw.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in known_ids:
            raise ValueError(f"mapping[{index}].scenario_id no existe en el manifest")
        if scenario_id in seen:
            raise ValueError(f"mapping scenario_id duplicado: {scenario_id}")
        seen.add(scenario_id)
        source = raw.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"mapping[{index}].source debe ser texto no vacío")
        if "://" in source:
            raise ValueError(f"mapping[{index}].source debe ser un path local")
        start = _finite_number(raw.get("start"), f"mapping[{index}].start")
        end = _finite_number(raw.get("end"), f"mapping[{index}].end")
        if start < 0:
            raise ValueError(f"mapping[{index}].start debe ser >= 0")
        if end <= start:
            raise ValueError(f"mapping[{index}].end debe ser mayor que start")
        raw_event = raw.get("event_at")
        event_at = (
            None
            if raw_event is None
            else _finite_number(raw_event, f"mapping[{index}].event_at")
        )
        if event_at is not None and not start < event_at < end:
            raise ValueError(f"mapping[{index}].event_at debe estar dentro de start/end")
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        mappings.append(
            SourceMapping(
                scenario_id=scenario_id,
                source_reference=source,
                source_path=source_path.resolve(),
                start=start,
                end=end,
                event_at=event_at,
            )
        )
    return tuple(mappings)


def _overlaps(start: float, end: float, other_start: float, other_end: float) -> bool:
    return other_end > start and other_start < end


def _rectangle_iou(left: Any, right: Any) -> float:
    intersection_width = max(
        0.0,
        min(left.x + left.width, right.x + right.width) - max(left.x, right.x),
    )
    intersection_height = max(
        0.0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _relevant_timeline_data(
    timeline: SourceVisualTimeline, mapping: SourceMapping
) -> tuple[list[Any], list[Any], list[Any]]:
    spans = [
        span
        for span in timeline.spans
        if _overlaps(mapping.start, mapping.end, span.source.start, span.source.end)
    ]
    samples = [
        sample for sample in timeline.samples if mapping.start <= sample.at < mapping.end
    ]
    webcam_tracks = [
        track
        for track in timeline.webcam_tracks
        if any(
            _overlaps(mapping.start, mapping.end, source.start, source.end)
            for source in track.source_ranges
        )
    ]
    return spans, samples, webcam_tracks


def _coverage_complete(spans: list[Any], mapping: SourceMapping) -> bool:
    cursor = mapping.start
    for span in spans:
        start = max(mapping.start, span.source.start)
        end = min(mapping.end, span.source.end)
        if end <= start:
            continue
        if start > cursor + _COVERAGE_EPSILON:
            return False
        cursor = max(cursor, end)
    return cursor >= mapping.end - _COVERAGE_EPSILON


def _evidence_result(
    name: str,
    expected: object,
    *,
    status: str,
    actual: object,
    message: str,
) -> dict[str, object]:
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "status": status,
        "passed": status == "pass",
        "message": message,
    }

def _assertion_result(
    name: str,
    expected: object,
    *,
    mapping: SourceMapping,
    spans: list[Any],
    samples: list[Any],
    webcam_tracks: list[Any],
) -> dict[str, object]:
    if not spans or not _coverage_complete(spans, mapping):
        return _evidence_result(
            name,
            expected,
            status="unavailable",
            actual={"span_count": len(spans)},
            message="incomplete temporal span coverage",
        )
    if not samples:
        return _evidence_result(
            name,
            expected,
            status="unavailable",
            actual={"sample_count": 0},
            message="no indexed samples in requested window",
        )

    observed_states = [span.state.value for span in spans]
    if name == "expected_states":
        required_states = set(cast(list[str], expected))
        observed_state_set = set(observed_states)
        missing = sorted(required_states - observed_state_set)
        unexpected = sorted(observed_state_set - required_states)
        status = "pass" if not missing and not unexpected else "fail"
        actual: object = sorted(observed_state_set)
        if status == "pass":
            message = "only expected states present"
        else:
            message = f"missing states: {missing}; unexpected states: {unexpected}"
    elif name == "ordered_states":
        collapsed_states: list[str] = []
        for state in observed_states:
            if not collapsed_states or collapsed_states[-1] != state:
                collapsed_states.append(state)
        expected_order = cast(list[str], expected)
        status = "pass" if collapsed_states == expected_order else "fail"
        actual = collapsed_states
        message = (
            "exact ordered state sequence present"
            if status == "pass"
            else f"expected exact state sequence {expected_order}"
        )
    elif name == "event_transition":
        event_at = mapping.event_at
        if event_at is None:
            return _evidence_result(
                name,
                expected,
                status="unavailable",
                actual=None,
                message="human-labeled event_at is missing",
            )
        expectation = cast(dict[str, object], expected)
        kind = cast(str, expectation["kind"])
        within = float(cast(float, expectation["within_seconds"]))
        candidates: list[dict[str, object]] = []
        if kind == "state":
            from_state = cast(str, expectation["from_state"])
            to_state = cast(str, expectation["to_state"])
            for left, right in zip(spans, spans[1:]):
                boundary = float(right.source.start)
                if (
                    abs(float(left.source.end) - boundary) <= _COVERAGE_EPSILON
                    and mapping.start < boundary < mapping.end
                    and left.state.value == from_state
                    and right.state.value == to_state
                ):
                    candidates.append(
                        {
                            "at": boundary,
                            "from_state": left.state.value,
                            "to_state": right.state.value,
                        }
                    )
        else:
            minimum_score = float(cast(float, expectation["minimum_change_score"]))
            for sample in samples:
                score = max(
                    float(sample.histogram_change_score),
                    float(sample.structural_change_score),
                )
                if score >= minimum_score:
                    candidates.append(
                        {
                            "at": float(sample.at),
                            "change_score": score,
                            "histogram_change_score": float(sample.histogram_change_score),
                            "structural_change_score": float(sample.structural_change_score),
                        }
                    )
        nearest = min(
            candidates,
            key=lambda item: abs(cast(float, item["at"]) - event_at),
            default=None,
        )
        nearest_at = cast(float, nearest["at"]) if nearest is not None else None
        distance = abs(nearest_at - event_at) if nearest_at is not None else None
        status = "pass" if distance is not None and distance <= within else "fail"
        actual = {
            "kind": kind,
            "event_at": event_at,
            "nearest_qualifying_event": nearest_at,
            "distance_seconds": distance,
            "evidence": nearest,
        }
        message = (
            f"nearest qualifying {kind} event is {distance} seconds from label"
            if nearest is not None
            else f"no qualifying {kind} event found"
        )
    elif name == "forbid_streamer_switch":
        if any(span.identity_status is IdentityStatus.AMBIGUOUS for span in spans):
            return _evidence_result(
                name,
                expected,
                status="ambiguous",
                actual={"identity_statuses": [span.identity_status.value for span in spans]},
                message="streamer identity is ambiguous in requested window",
            )
        if any(span.identity_status is IdentityStatus.UNAVAILABLE for span in spans):
            return _evidence_result(
                name,
                expected,
                status="unavailable",
                actual={"identity_statuses": [span.identity_status.value for span in spans]},
                message="streamer identity is unavailable in part of requested window",
            )
        track_ids = sorted(
            {
                span.streamer_track_id
                for span in spans
                if span.identity_status is IdentityStatus.CONFIDENT
                and span.streamer_track_id is not None
            }
        )
        identity_sample_count = sum(
            1
            for sample in samples
            if any(track_id in sample.face_track_ids for track_id in track_ids)
        )
        if not track_ids or identity_sample_count < 2:
            return _evidence_result(
                name,
                expected,
                status="unavailable",
                actual={
                    "streamer_track_ids": track_ids,
                    "identity_sample_count": identity_sample_count,
                },
                message="insufficient confident streamer identity evidence",
            )
        status = "pass" if len(track_ids) == 1 else "fail"
        actual = track_ids
        message = (
            "streamer identity stable"
            if status == "pass"
            else "streamer identity switched"
        )
    elif name == "forbid_webcam_track":
        track_ids = sorted(
            {
                *(
                    span.webcam_track_id
                    for span in spans
                    if span.webcam_track_id is not None
                ),
                *(
                    sample.webcam_track_id
                    for sample in samples
                    if sample.webcam_track_id is not None
                ),
                *(track.track_id for track in webcam_tracks),
            }
        )
        status = "pass" if not track_ids else "fail"
        actual = track_ids
        message = (
            "no webcam track"
            if status == "pass"
            else f"webcam tracks found: {track_ids}"
        )
    elif name == "require_text_regions":
        region_count = sum(len(span.text_regions) for span in spans) + sum(
            len(sample.text_regions) for sample in samples
        )
        status = "pass" if region_count > 0 else "fail"
        actual = region_count
        message = f"observed {region_count} text regions"
    elif name == "require_multiple_faces":
        multi_face_samples = [
            sample
            for sample in samples
            if len(set(sample.face_track_ids)) >= 2
        ]
        maximum = max(
            (len(set(sample.face_track_ids)) for sample in samples),
            default=0,
        )
        track_ids = sorted(
            {
                track_id
                for sample in multi_face_samples
                for track_id in sample.face_track_ids
            }
        )
        status = "pass" if multi_face_samples else "fail"
        actual = {"maximum_face_tracks": maximum, "track_ids": track_ids}
        message = f"observed {len(multi_face_samples)} multi-face samples"
    elif name == "require_vertical_region":
        vertical_by_sample = [
            (
                sample.at,
                [
                    rectangle
                    for rectangle in sample.rectangle_candidates
                    if rectangle.height >= 0.35
                    and rectangle.width * rectangle.height >= 0.05
                    and rectangle.width / rectangle.height <= 0.65
                ],
            )
            for sample in samples
        ]
        vertical_regions = [
            rectangle
            for _, rectangles in vertical_by_sample
            for rectangle in rectangles
        ]
        persistent_sample_count = max(
            (
                len(
                    {
                        at
                        for at, rectangles in vertical_by_sample
                        if any(
                            _rectangle_iou(anchor, rectangle) >= 0.5
                            for rectangle in rectangles
                        )
                    }
                )
                for anchor in vertical_regions
            ),
            default=0,
        )
        status = "pass" if persistent_sample_count >= 2 else "fail"
        actual = {
            "region_count": len(vertical_regions),
            "persistent_sample_count": persistent_sample_count,
            "minimum_aspect_ratio": (
                min(
                    rectangle.width / rectangle.height
                    for rectangle in vertical_regions
                )
                if vertical_regions
                else None
            ),
        }
        message = (
            f"observed {len(vertical_regions)} vertical regions across "
            f"{persistent_sample_count} matching samples"
        )
    else:  # Manifest validation makes this unreachable.
        raise ValueError(f"assertion no soportada: {name}")
    return _evidence_result(
        name,
        expected,
        status=status,
        actual=actual,
        message=message,
    )


def evaluate_scenario(
    scenario: Scenario,
    mapping: SourceMapping,
    timeline: SourceVisualTimeline,
    *,
    source_build_runtime_seconds: float,
    build_metadata: BuildMetadata,
) -> dict[str, object]:
    evaluation_started = perf_counter()
    spans, samples, webcam_tracks = _relevant_timeline_data(timeline, mapping)
    assertions = [
        _assertion_result(
            name,
            expected,
            spans=spans,
            mapping=mapping,
            samples=samples,
            webcam_tracks=webcam_tracks,
        )
        for name, expected in scenario.assertions.items()
    ]
    assertion_statuses = {str(result["status"]) for result in assertions}
    if "fail" in assertion_statuses:
        status = "fail"
    elif "ambiguous" in assertion_statuses:
        status = "ambiguous"
    elif "unavailable" in assertion_statuses:
        status = "unavailable"
    else:
        status = "pass"
    failures = [result["message"] for result in assertions if result["status"] == "fail"]
    evidence_issues = [result["message"] for result in assertions if result["status"] in {"ambiguous", "unavailable"}]
    identity_statuses = sorted({span.identity_status.value for span in spans})
    return {
        "scenario_id": scenario.scenario_id,
        "status": status,
        "source_reference": mapping.source_reference,
        "requested_window": {"start": mapping.start, "end": mapping.end},
        "source_indexed_sample_count": len(timeline.samples),
        "window_indexed_sample_count": len(samples),
        "indexed_samples_this_run": build_metadata.indexed_samples_this_run,
        "cache_hit": build_metadata.cache_hit,
        "source_build_runtime_seconds": source_build_runtime_seconds,
        "scenario_evaluation_runtime_seconds": perf_counter() - evaluation_started,
        "model_hashes": dict(timeline.model_hashes),
        "relevant_state_spans": [span.model_dump(mode="json") for span in spans],
        "identity_statuses": identity_statuses,
        "assertions": assertions,
        "failures": failures,
        "unavailable_reason": None,
        "evidence_issues": evidence_issues,
    }


def _unavailable_report(
    scenario: Scenario,
    *,
    reason: str,
    mapping: SourceMapping | None = None,
    source_build_runtime_seconds: float = 0.0,
    source_indexed_sample_count: int | None = None,
    window_indexed_sample_count: int | None = None,
    indexed_samples_this_run: int | None = None,
    cache_hit: bool = False,
    detail: str | None = None,
) -> dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "status": "unavailable",
        "source_reference": mapping.source_reference if mapping else None,
        "requested_window": (
            {"start": mapping.start, "end": mapping.end} if mapping else None
        ),
        "source_indexed_sample_count": source_indexed_sample_count,
        "window_indexed_sample_count": window_indexed_sample_count,
        "indexed_samples_this_run": indexed_samples_this_run,
        "cache_hit": cache_hit,
        "source_build_runtime_seconds": source_build_runtime_seconds,
        "scenario_evaluation_runtime_seconds": 0.0,
        "model_hashes": {},
        "relevant_state_spans": [],
        "identity_statuses": [],
        "assertions": [],
        "failures": [],
        "evidence_issues": [reason],
        "unavailable_reason": reason,
        "detail": detail,
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid4().hex}"


def _exit_code_for_counts(counts: dict[str, int]) -> int:
    if counts["fail"]:
        return 1
    if counts != {"pass": 10, "fail": 0, "ambiguous": 0, "unavailable": 0}:
        return 2
    return 0


def _publish_run(
    output_dir: Path,
    manifest: ScenarioManifest,
    reports: dict[str, dict[str, object]],
    *,
    sources_path: Path,
) -> ValidationRunResult:
    ordered_reports = [reports[scenario.scenario_id] for scenario in manifest.scenarios]
    counts = Counter(str(report["status"]) for report in ordered_reports)
    normalized_counts = {status: counts.get(status, 0) for status in REPORT_STATUSES}
    gate_passed = normalized_counts == {
        "pass": 10,
        "fail": 0,
        "ambiguous": 0,
        "unavailable": 0,
    }
    run_id = _new_run_id()
    summary: dict[str, object] = {
        "version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources_mapping": str(sources_path),
        "counts": normalized_counts,
        "gate_passed": gate_passed,
        "executed_scenarios": 10 - normalized_counts["unavailable"],
        "all_ten_real_windows_executed": normalized_counts["unavailable"] == 0,
        "scenario_reports": [f"{report['scenario_id']}.json" for report in ordered_reports],
    }

    staging_dir = output_dir / ".staging" / run_id
    generation_dir = output_dir / "generations" / run_id
    staging_dir.mkdir(parents=True, exist_ok=False)
    generation_dir.parent.mkdir(parents=True, exist_ok=True)
    for report in ordered_reports:
        _atomic_write_json(staging_dir / f"{report['scenario_id']}.json", report)
    _atomic_write_json(staging_dir / "summary.json", summary)
    staging_dir.replace(generation_dir)

    latest = {
        "version": 1,
        "run_id": run_id,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "generation": (Path("generations") / run_id).as_posix(),
        "summary": (Path("generations") / run_id / "summary.json").as_posix(),
    }
    _atomic_write_json(output_dir / "latest.json", latest)
    return ValidationRunResult(
        exit_code=_exit_code_for_counts(normalized_counts),
        summary=summary,
    )

def _file_snapshot(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _invoke_builder(
    active_builder: Builder,
    source_path: Path,
    *,
    boundaries: tuple[float, ...],
    sparse_interval: float,
    dense_interval: float,
    force: bool,
) -> BuildMetadata:
    if active_builder is _DEFAULT_PUBLIC_BUILDER:
        observed = build_source_visual_timeline_result(source_path,
            editorial_boundaries=boundaries, sparse_interval=sparse_interval, dense_interval=dense_interval,
            allow_model_download=False, force=force)
        return BuildMetadata(observed.artifact_path, observed.cache_hit, observed.indexed_samples_this_run)
    expected_artifact = source_job_root(source_path) / "source_visual_timeline.json"
    before = _file_snapshot(expected_artifact)
    raw_result = active_builder(
        source_path,
        editorial_boundaries=boundaries,
        sparse_interval=sparse_interval,
        dense_interval=dense_interval,
        allow_model_download=False,
        force=force,
    )
    if isinstance(raw_result, (BuildMetadata, VisualTimelineBuildResult)):
        frames = 0 if raw_result.cache_hit else raw_result.indexed_samples_this_run
        return BuildMetadata(
            artifact_path=Path(raw_result.artifact_path),
            cache_hit=raw_result.cache_hit,
            indexed_samples_this_run=frames,
        )

    artifact_path = Path(raw_result)
    after = _file_snapshot(artifact_path)
    cache_hit = (
        not force
        and before is not None
        and artifact_path.resolve() == expected_artifact.resolve()
        and before == after
    )
    indexed_samples_this_run: int | None = 0 if cache_hit else None
    if (
        not cache_hit
        and artifact_path.resolve() == expected_artifact.resolve()
        and after is not None
        and before != after
    ):
        rebuilt_timeline = SourceVisualTimeline.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
        indexed_samples_this_run = len(rebuilt_timeline.samples)
    return BuildMetadata(
        artifact_path=artifact_path,
        cache_hit=cache_hit,
        indexed_samples_this_run=indexed_samples_this_run,
    )


def _probe_duration(source_path: Path) -> float:
    metadata = probe_media(source_path)
    raw_duration = metadata.get("duration_seconds")
    if isinstance(raw_duration, bool) or not isinstance(raw_duration, (int, float)):
        raise ValueError("probe_media no devolvió duration_seconds")
    duration = float(raw_duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("probe_media devolvió una duración inválida")
    return duration


def run_validation(
    *,
    scenario_manifest_path: Path,

    sources_path: Path,
    output_dir: Path,
    sparse_interval: float = 10.0,
    dense_interval: float = 0.5,
    force: bool = False,
    builder: Builder | None = None,
) -> ValidationRunResult:
    if sparse_interval <= 0 or dense_interval <= 0:
        raise ValueError("sampling intervals deben ser positivos")
    scenario_manifest_path = Path(scenario_manifest_path)
    sources_path = Path(sources_path)
    output_dir = Path(output_dir)
    manifest = load_scenario_manifest(scenario_manifest_path)
    if not sources_path.is_file():
        reports = {
            scenario.scenario_id: _unavailable_report(
                scenario, reason="sources_mapping_missing"
            )
            for scenario in manifest.scenarios
        }
        return _publish_run(
            output_dir, manifest, reports, sources_path=sources_path
        )

    mappings = load_source_mappings(sources_path, manifest)
    mapping_by_id = {mapping.scenario_id: mapping for mapping in mappings}
    reports = {
        scenario.scenario_id: _unavailable_report(
            scenario, reason="scenario_not_mapped"
        )
        for scenario in manifest.scenarios
        if scenario.scenario_id not in mapping_by_id
    }
    grouped: dict[Path, list[SourceMapping]] = defaultdict(list)
    for mapping in mappings:
        grouped[mapping.source_path].append(mapping)

    active_builder = builder or build_source_visual_timeline
    for source_path, source_mappings in grouped.items():
        if not source_path.is_file():
            for mapping in source_mappings:
                reports[mapping.scenario_id] = _unavailable_report(
                    manifest.by_id[mapping.scenario_id],
                    reason="source_missing",
                    mapping=mapping,
                )
            continue
        try:
            media_duration = _probe_duration(source_path)
        except Exception as exc:
            for mapping in source_mappings:
                reports[mapping.scenario_id] = _unavailable_report(
                    manifest.by_id[mapping.scenario_id],
                    reason="media_probe_failed",
                    mapping=mapping,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            continue

        valid_mappings: list[SourceMapping] = []
        for mapping in source_mappings:
            if mapping.end > media_duration:
                reports[mapping.scenario_id] = _unavailable_report(
                    manifest.by_id[mapping.scenario_id],
                    reason="window_outside_media",
                    mapping=mapping,
                    detail=(
                        f"requested end {mapping.end:.3f}s exceeds "
                        f"media duration {media_duration:.3f}s"
                    ),
                )
            else:
                valid_mappings.append(mapping)
        if not valid_mappings:
            continue

        boundaries = tuple(
            sorted(
                {
                    value
                    for mapping in valid_mappings
                    for value in (mapping.start, mapping.event_at, mapping.end)
                    if value is not None
                }
            )
        )
        started = perf_counter()
        try:
            build_metadata = _invoke_builder(
                active_builder,
                source_path,
                boundaries=boundaries,
                sparse_interval=sparse_interval,
                dense_interval=dense_interval,
                force=force,
            )
            timeline = SourceVisualTimeline.model_validate_json(
                build_metadata.artifact_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            runtime = perf_counter() - started
            for mapping in valid_mappings:
                reports[mapping.scenario_id] = _unavailable_report(
                    manifest.by_id[mapping.scenario_id],
                    reason="source_build_failed",
                    mapping=mapping,
                    source_build_runtime_seconds=runtime,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            continue
        runtime = perf_counter() - started
        for mapping in valid_mappings:
            scenario = manifest.by_id[mapping.scenario_id]
            if mapping.end > timeline.duration_seconds:
                reports[mapping.scenario_id] = _unavailable_report(
                    scenario,
                    reason="window_outside_timeline",
                    mapping=mapping,
                    source_build_runtime_seconds=runtime,
                    source_indexed_sample_count=len(timeline.samples),
                    window_indexed_sample_count=0,
                    indexed_samples_this_run=build_metadata.indexed_samples_this_run,
                    cache_hit=build_metadata.cache_hit,
                )
                continue
            reports[mapping.scenario_id] = evaluate_scenario(
                scenario,
                mapping,
                timeline,
                source_build_runtime_seconds=runtime,
                build_metadata=build_metadata,
            )
    return _publish_run(output_dir, manifest, reports, sources_path=sources_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate provider-free SourceVisualTimeline scenarios."
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=ROOT / "data" / "validation_visual" / "scenarios.json",
    )
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-interval", type=float, default=10.0)
    parser.add_argument("--dense-interval", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_validation(
            scenario_manifest_path=args.scenarios,
            sources_path=args.sources,
            output_dir=args.output,
            sparse_interval=args.sparse_interval,
            dense_interval=args.dense_interval,
            force=args.force,
        )
    except ValueError as exc:
        print(f"Validation configuration error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
