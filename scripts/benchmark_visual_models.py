"""Bounded visual-director gate. Inventory is read-only; inference is opt-in.

Portable cases reference existing v7 caches. Machine-local source paths come from
the ignored Phase1 mapping. No analyzer, renderer, corpus batch or download runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys
from time import perf_counter
from typing import Any, Sequence, TypeVar
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, ValidationError  # noqa: E402
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError  # noqa: E402
from requests.exceptions import RequestException  # noqa: E402

from clipperx.evidence import associated_face_positions  # noqa: E402
from clipperx.local_paths import atomic_write_text, source_fingerprint  # noqa: E402
from clipperx.planner_contracts import ClipPlan  # noqa: E402
from clipperx.planner_engine import StructuredClient  # noqa: E402
from clipperx.structured_client import (  # noqa: E402
    InputBudgetError, OpenAICompatibleStructuredClient, ProviderHTTPError,
    StructuredModelConfig, StructuredResponseError, StructuredResult,
)
from clipperx.style_library import StyleLibrary  # noqa: E402
from clipperx.visual_contracts import SourceVisualTimeline, VisualCompositionPlan  # noqa: E402
from clipperx.visual_director import VisualDirectorLimits, run_visual_director  # noqa: E402

OWNER_MODEL = "gpt-6-luna(low)"
CONTEXT = 65536
OUTPUT = 4096
LIMITS = VisualDirectorLimits(max_calls_per_clip=2, max_followups_per_segment=1)
ModelT = TypeVar("ModelT", bound=BaseModel)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def portable(ref: str) -> bool:
    windows = PureWindowsPath(ref)
    return bool(ref) and not (Path(ref).is_absolute() or windows.drive or windows.anchor
                             or ".." in Path(ref.replace("\\", "/")).parts)


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    if data.get("version") != 1 or len(cases) != 3:
        raise ValueError("Exactly three v1 cases required")
    if data.get("image_budget_per_call") != 8:
        raise ValueError("Benchmark image budget must be eight")
    ids = set()
    for case in cases:
        serialized = json.dumps(case, ensure_ascii=False)
        if len(serialized) > min(data["text_budget_chars_per_case"], 16000):
            raise ValueError("Case exceeds text budget")
        if re.search(r'"(?:local_|full_)?transcript"\s*:', serialized):
            raise ValueError("Portable cases cannot contain transcript")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", case["case_id"]):
            raise ValueError("case_id must be a bounded portable identifier")
        refs = [case["source_ref"], case["timeline_subset"]["cache_ref"], *case["image_paths"]]
        if len(case["image_paths"]) > 8 or not all(portable(ref) for ref in refs):
            raise ValueError("Case references must be portable and bounded")
        clip = ClipPlan.model_validate(case["editorial_plan"])
        if len(clip.segments) != 1 or clip.duration > 60:
            raise ValueError("One bounded editorial window required")
        start, end = case["timeline_subset"]["context_window"]
        window = clip.segments[0].source
        duration = case["timeline_subset"]["duration_seconds"]
        if (start != max(0, window.start - 15) or end != min(duration, window.end + 15)
                or duration > 60):
            raise ValueError("Context must equal source-clamped editorial +/-15s")
        ranges = case["expected"]["source_ranges"]
        if (ranges[0][0] != window.start or ranges[-1][1] != window.end
                or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:]))):
            raise ValueError("Expected split must preserve editorial range")
        if case["case_id"] in ids:
            raise ValueError("Duplicate case ID")
        ids.add(case["case_id"])
    return list(cases)


def timeline_subset(timeline: SourceVisualTimeline, start: float, end: float
                    ) -> SourceVisualTimeline:
    """Retain original times and reconstruct local face summaries by proven IDs."""
    data = timeline.model_dump(mode="json")
    data["samples"] = [s for s in data["samples"] if start <= s["at"] < end]
    data["spans"] = [s for s in data["spans"]
                     if s["source"]["start"] < end and s["source"]["end"] > start]
    needed = {f for s in data["samples"] for f in s["face_track_ids"]}
    needed.update(s["streamer_track_id"] for s in data["spans"] if s["streamer_track_id"])
    faces = []
    for track in timeline.face_tracks:
        if track.track_id not in needed:
            continue
        positions = associated_face_positions(timeline, track)
        if positions is None:
            raise ValueError("Cached face positions have no proven temporal bijection")
        local = [(at, box, eid) for at, box, eid in positions if start <= at < end]
        if len(local) < 2:
            raise ValueError("Insufficient local face observations for strict summary")
        face = track.model_dump(mode="json")
        face.update(first_seen=local[0][0], last_seen=local[-1][0],
                    observation_count=len(local), positions=[box.model_dump() for _, box, _ in local],
                    evidence_ids=[eid for _, _, eid in local])
        faces.append(face)
    data["face_tracks"] = faces
    webcams = []
    camera_ids = {s["webcam_track_id"] for s in data["samples"] + data["spans"]
                  if s["webcam_track_id"]}
    for track in data["webcam_tracks"]:
        if track["track_id"] not in camera_ids:
            continue
        observations = [o for o in track["observations"] if start <= o["at"] < end]
        if not observations:
            raise ValueError("Missing local webcam observations")
        track["observations"] = observations
        track["source_ranges"] = [dict(start=max(start, r["start"]), end=min(end, r["end"]))
                                  for r in track["source_ranges"]
                                  if r["start"] < end and r["end"] > start]
        track["face_track_ids"] = [f for f in track["face_track_ids"] if f in needed]
        track["evidence_ids"] = list(dict.fromkeys(o["evidence_id"] for o in observations))
        webcams.append(track)
    data["webcam_tracks"] = webcams
    subset = SourceVisualTimeline.model_validate(data)
    for track in subset.face_tracks:
        if associated_face_positions(subset, track) is None:
            raise ValueError("Subset lost face temporal association")
    return subset


def prepare_case(case: dict[str, Any], mapping_path: Path
                 ) -> tuple[Path, ClipPlan, SourceVisualTimeline]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    matches = [m for m in mapping["mappings"] if m["scenario_id"] == case["source_ref"]]
    if len(matches) != 1:
        raise ValueError("Missing or duplicate machine-local source mapping")
    video = (mapping_path.parent / matches[0]["source"]).resolve()
    ref = case["timeline_subset"]
    timeline = SourceVisualTimeline.model_validate_json((ROOT / ref["cache_ref"]).read_text())
    if (source_fingerprint(video) != ref["source_fingerprint"]
            or timeline.source_fingerprint != ref["source_fingerprint"]
            or timeline.duration_seconds != ref["duration_seconds"]):
        raise ValueError("Source/cache authority differs from portable case")
    subset = timeline_subset(timeline, *ref["context_window"])
    actual = [[s.source.start, s.source.end] for s in subset.spans]
    if actual != ref["span_ranges"]:
        raise ValueError("Cached timeline span ranges changed")
    states = [s.state.value for i, s in enumerate(subset.spans)
              if i == 0 or subset.spans[i - 1].state != s.state]
    if states != ref["states"]:
        raise ValueError("Cached timeline states changed")
    return video, ClipPlan.model_validate(case["editorial_plan"]), subset


class MeasuredClient:
    def __init__(self, client: StructuredClient) -> None:
        self.client = client
        self.records: list[dict[str, Any]] = []

    def generate(self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
                 images: list[Path] | None = None, temperature: float = .2
                 ) -> StructuredResult[ModelT]:
        if len(self.records) >= 2:
            raise ValueError("Benchmark call limit")
        manifest = json.loads(user_prompt)["frame_manifest"]
        if (len(images or []) > 8 or sum(f["kind"] == "source" for f in manifest) > 5
                or sum(f["kind"] == "style" for f in manifest) > 3):
            raise ValueError("Benchmark image limit")
        record: dict[str, Any] = {"schema_valid": False, "prompt_tokens": None,
                                  "completion_tokens": None, "total_tokens": None}
        self.records.append(record)
        started = perf_counter()
        try:
            result = self.client.generate(output_model, system_prompt=system_prompt,
                user_prompt=user_prompt, images=images, temperature=temperature)
            output_model.model_validate(result.value.model_dump())
            record["schema_valid"] = True
            # The shared client substitutes zero for absent counters. Use only
            # actually reported raw provider fields, never that zero fallback.
            usage = result.raw_response.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if type(value) is int and value >= 0:
                    record[key] = value
            return result
        except ProviderHTTPError as error:
            record.update(failure_stage="provider_http",
                          provider_http_status=error.provider_http_status,
                          provider_error_code=error.provider_error_code)
            raise
        except InputBudgetError:
            record["failure_stage"] = "input_budget"
            raise
        except StructuredResponseError as error:
            record.update(failure_stage=("response_envelope" if error.failure_kind == "response_envelope"
                                         else "structured_schema"),
                          failure_kind=error.failure_kind, finish_reason=error.finish_reason,
                          validation_issues=error.issues)
            record.update(error.usage)
            raise
        except RequestsJSONDecodeError:
            # This also inherits ValueError; never label an HTTP body as model JSON.
            record["failure_stage"] = "response_envelope"
            raise
        except RequestException:
            # Transport exceptions may embed URLs/headers/keys; retain only category.
            record["failure_stage"] = "transport"
            raise
        except (ValidationError, json.JSONDecodeError):
            record["failure_stage"] = "structured_schema"
            raise
        finally:
            record["latency_seconds"] = perf_counter() - started


def run_case(case: dict[str, Any], video: Path, clip: ClipPlan,
             timeline: SourceVisualTimeline, library: StyleLibrary,
             client: StructuredClient, jobs_root: Path) -> dict[str, Any]:
    measured = MeasuredClient(client)
    started = perf_counter()
    artifact = run_visual_director(video, clip.model_copy(deep=True), timeline.model_copy(deep=True),
        client=measured, style_library=library, limits=LIMITS, jobs_root=jobs_root)
    content = json.loads(artifact.read_text(encoding="utf-8"))
    receipt = json.loads((artifact.parent / "receipt.json").read_text(encoding="utf-8"))
    ready = artifact.name == "visual_plan.json" and receipt.get("status") == "ready"
    ranges_ok = False
    if ready:
        plan = VisualCompositionPlan.model_validate(content)
        ranges_ok = [[s.source.start, s.source.end] for s in plan.shots] == case["expected"]["source_ranges"]
    calls = len(measured.records)
    schema_rate = sum(r["schema_valid"] for r in measured.records) / calls if calls else 0
    # All identity/geometry/evidence/source protection gates are enforced by the
    # public host before publishing a ready receipt, including final drift checks.
    invariants = {"host_validated": ready, "editorial_ranges_and_transition": ranges_ok,
                  "call_and_iteration_budget": calls <= 2 and receipt.get("calls", 0) <= 2}
    result: dict[str, Any] = {"case_id": case["case_id"], "artifact": str(artifact),
        "status": "pass" if ready and ranges_ok and schema_rate == 1 else "fail",
        "passed": ready and ranges_ok and schema_rate == 1,
        "schema_success_rate": schema_rate,
        "invariant_pass_rate": sum(invariants.values()) / len(invariants),
        "invariants": invariants, "calls": calls, "evidence_iterations": max(0, calls - 1),
        "call_metrics": measured.records, "elapsed_seconds": perf_counter() - started,
        "estimated_provider_cost": None, "billing": "unpriced",
        "failure_reason": receipt.get("reason"),
        "source_ranges": case["expected"]["source_ranges"]}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [r[key] for r in measured.records]
        result[key] = sum(values) if values and all(v is not None for v in values) else None
    write_json(artifact.parent / "benchmark_metrics.json", result)
    return result


def local_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Use the existing localhost proxy URL without credentials")
    return base_url.rstrip("/")


def discover_models(base_url: str, key: str, session: Any = None) -> dict[str, Any]:
    url = local_base_url(base_url) + "/models"
    headers = {"Authorization": f"Bearer {key}"}
    if session is not None:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    else:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            data = json.load(response)
    inventory = sorted({row["id"] for row in data["data"] if isinstance(row.get("id"), str)
                        and "gemini" not in row["id"].lower()})
    available = OWNER_MODEL in inventory or "gpt-6-luna" in inventory
    return {"version": 1, "status": "discovered", "date_utc": stamp(), "provider": "cliproxy",
        "base_url": base_url, "inventory_model_ids": inventory,
        "models": [OWNER_MODEL] if available else [], "inference_calls": 0,
        "billing": "unpriced", "selection_scope": "owner-selected target only",
        "target_availability": "advertised" if available else "not_advertised"}


def select_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [r for r in results if r["passed"]]
    return {"status": "passed" if passing else "none_passed",
        "selected_model": passing[0]["model"] if passing else None,
        "cheapest_passing_model": None,
        "selection_basis": "owner_selected_only_passing_target_unpriced" if passing else "none_passed",
        "estimated_provider_cost": None, "billing": "unpriced"}


def _pipeline_code_provenance(code_root: Path | None = None) -> dict[str, Any]:
    """Fingerprint complete package source and runner, never cache/output bytes.

    Relative paths and raw contents are length-prefixed to make the aggregate
    unambiguous. Per-file hashes explain the persisted provenance. The optional
    root is a test seam for isolated dependency edits, not a runtime CLI option.
    """
    root = (code_root or ROOT).resolve()
    package = root / "clipperx"
    runner = root / "scripts/benchmark_visual_models.py"
    files = [path for path in package.rglob("*.py") if "__pycache__" not in path.parts]
    if not files or not runner.is_file():
        raise ValueError("Complete pipeline code is unavailable")
    files.append(runner)
    digest = hashlib.sha256()
    manifest = []
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Pipeline code reference escapes code root")
        relative = path.relative_to(root).as_posix()
        name = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        manifest.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
    return {"algorithm": "sha256-length-prefixed-relative-path-and-bytes-v1",
            "digest_sha256": digest.hexdigest(), "files": manifest}


def ensure_run_config(root: Path, config: dict[str, Any]) -> str:
    signature = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    path = root / "run_config.json"
    expected = {"run_signature": signature, "inputs": config}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Run configuration changed; use a new run-dir")
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, expected)
    return signature


def aggregate_cases(root: Path, case_ids: list[str], signature: str) -> list[dict[str, Any]]:
    paths = [root / case_id / "result.json" for case_id in case_ids]
    if len(set(case_ids)) != 3 or not all(path.is_file() for path in paths):
        raise ValueError("All three completed case results are required")
    results = []
    for case_id, path in zip(case_ids, paths):
        result = json.loads(path.read_text(encoding="utf-8"))
        if (result.get("run_signature") != signature or result.get("case_id") != case_id
                or result.get("model") != OWNER_MODEL):
            raise ValueError("Case model/input/configuration signature differs")
        results.append(result)
    for case_id, result in zip(case_ids, results):
        artifact = Path(result.get("artifact", "")).resolve()
        if (not artifact.is_relative_to((root / case_id).resolve()) or not artifact.is_file()
                or artifact.name not in {"visual_plan.json", "receipt.json"}):
            raise ValueError("Missing or unrelated public director artifact")
        metrics = artifact.parent / "benchmark_metrics.json"
        saved = {k: v for k, v in result.items() if k not in {"model", "run_signature"}}
        if not metrics.is_file() or json.loads(metrics.read_text(encoding="utf-8")) != saved:
            raise ValueError("Saved metrics differ from director run metrics")
        receipt = json.loads((artifact.parent / "receipt.json").read_text(encoding="utf-8"))
        ready = artifact.name == "visual_plan.json" and receipt.get("status") == "ready"
        if (type(result.get("passed")) is not bool or receipt.get("calls") != result["calls"]
                or not 0 <= result["calls"] <= 2 or len(result["call_metrics"]) != result["calls"]):
            raise ValueError("Invalid completed case metrics")
        if ready:
            VisualCompositionPlan.model_validate_json(artifact.read_text(encoding="utf-8"))
        passing = (ready and result["schema_success_rate"] == 1
                   and all(result["invariants"].values()))
        if result["passed"] != passing:
            raise ValueError("Case pass disagrees with public artifact and metrics")
    return results


def append_audit(path: Path, result: dict[str, Any]) -> None:
    marker = f"<!-- visual-benchmark:{result['run_signature']} -->"
    existing = path.read_text(encoding="utf-8")
    if marker in existing:
        return
    lines = ["", marker, f"## Bounded director benchmark - {result['completed_utc']}", "",
        f"Provider cliproxy; model {OWNER_MODEL}; context {CONTEXT}; output {OUTPUT}.",
        "Only the owner-selected target was tested; proxy rates are unknown/unpriced.",
        f"Outcome: {result['status']}; selected: {result['selected_model']}; cheapest: unknown.",
        "Estimated provider cost: null. Source files are short Phase1 v7 references.", ""]
    for case in result["results"]:
        lines.append(f"- {case['case_id']}: {case['status']}; source {case['source_ranges']}; "
            f"schema {case['schema_success_rate']}; invariants {case['invariant_pass_rate']}; "
            f"calls {case['calls']}; evidence iterations {case['evidence_iterations']}; "
            f"tokens prompt/completion/total {case['prompt_tokens']}/{case['completion_tokens']}/"
            f"{case['total_tokens']}; elapsed {case['elapsed_seconds']:.3f}s; "
            f"call latencies {[round(c['latency_seconds'], 3) for c in case['call_metrics']]}s.")
    lines.extend(["", f"Measured artifacts: `{result['run_root']}`.",
        "This measures host/schema geometry and source preservation, not semantic editing,",
        "rendered quality or human approval. All three StyleCards remain unreviewed,",
        "narrative null and without invented source pairing. Missing style families stay sparse.",
        "No full VOD, all301 enrichment or MP4 rendering was performed.", ""])
    atomic_write_text(path, existing + "\n".join(lines))


def read_key() -> str:
    key = os.environ.get("CLIPPERX_BENCHMARK_API_KEY", "")
    if not key:
        raise ValueError("Set ephemeral CLIPPERX_BENCHMARK_API_KEY in process environment")
    return key


def make_client(model: str, base_url: str, key: str) -> OpenAICompatibleStructuredClient:
    return OpenAICompatibleStructuredClient(StructuredModelConfig(provider="cliproxy",
        base_url=base_url, api_key=key, model=model, context_window_tokens=CONTEXT,
        max_output_tokens=OUTPUT))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/model_validation/visual_director_cases.json")
    parser.add_argument("--discover-models", action="store_true")
    parser.add_argument("--base-url", default="http://localhost:8317/v1")
    parser.add_argument("--exclude-family", choices=["gemini"], default="gemini")
    parser.add_argument("--candidate-output", type=Path, default=ROOT / "data/model_validation/candidate_models.json")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--models", nargs="+")
    targets.add_argument("--models-from", type=Path)
    parser.add_argument("--selection-output", type=Path, default=ROOT / "data/model_validation/selected_visual_model.json")
    parser.add_argument("--confirm-inference", action="store_true")
    parser.add_argument("--local-mapping", type=Path, default=ROOT / "data/validation_visual/local_sources.json")
    parser.add_argument("--style-library", type=Path, default=ROOT / "data/processed/style_library")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "data/validation_visual/runs/model_benchmark")
    parser.add_argument("--run-dir", type=Path, help="Shared isolated directory for one three-case run")
    parser.add_argument("--case-id", help="Run exactly one case; requires --run-dir")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate three saved cases; no inference")
    parser.add_argument("--audit-output", type=Path, default=ROOT / "docs/style-corpus-audit.md")
    args = parser.parse_args(argv)
    if (args.case_id or args.aggregate) and args.run_dir is None:
        raise ValueError("--case-id and --aggregate require an explicit --run-dir")
    if args.aggregate and (args.case_id or args.discover_models):
        raise ValueError("--aggregate cannot be combined with case/discovery")
    if not args.discover_models and not args.aggregate and not args.confirm_inference:
        raise ValueError("Inference requires --confirm-inference before credentials/client construction")
    base_url = local_base_url(args.base_url)
    cases = load_cases(args.cases)
    if args.discover_models:
        write_json(args.candidate_output, discover_models(base_url, read_key()))
        print(f"Inventory saved: {args.candidate_output}; no inference")
        return 0
    models = args.models
    if args.models_from:
        models = json.loads(args.models_from.read_text(encoding="utf-8"))["models"]
    if models != [OWNER_MODEL]:
        raise ValueError(f"Only the owner-selected target is authorized: {OWNER_MODEL}")
    # Validate all local authority before credentials and before the first call.
    prepared = [prepare_case(case, args.local_mapping) for case in cases]
    library = StyleLibrary(args.style_library)
    style_fp = library.fingerprint
    run_root = (args.run_dir or (args.runs_root / uuid4().hex)).resolve()
    inputs = {"model": models[0], "base_url": base_url, "provider": "cliproxy",
        "context": CONTEXT, "output": OUTPUT, "style_fingerprint": style_fp,
        "case_manifest": cases, "limits": {"calls": 2, "followups": 1, "source_images": 5, "style_images": 3},
        "timeline_hashes": [hashlib.sha256(t.model_dump_json().encode()).hexdigest() for _, _, t in prepared],
        "source_fingerprints": [source_fingerprint(v) for v, _, _ in prepared],
        "pipeline_code": _pipeline_code_provenance()}
    if args.aggregate and not (run_root / "run_config.json").is_file():
        raise ValueError("Existing run-dir configuration required for aggregation")
    signature = ensure_run_config(run_root, inputs)
    result: dict[str, Any] = {"version": 1, "status": "incomplete", "date_utc": stamp(),
        "provider": "cliproxy", "base_url": base_url, "models_tested": models,
        "context_window_tokens": CONTEXT, "max_output_tokens": OUTPUT,
        "limits": {"max_calls_per_case": 2, "max_followups_per_case": 1,
                   "max_source_images_per_call": 5, "max_style_images_per_call": 3},
        "selection_scope": "owner-selected target only; no global cheapest claim",
        "selected_role": None, "selected_model": None, "cheapest_passing_model": None,
        "billing": "unpriced", "estimated_provider_cost": None,
        "run_root": str(run_root), "run_signature": signature, "results": []}
    if not args.aggregate:
        selected = [(c, p) for c, p in zip(cases, prepared)
                    if args.case_id is None or c["case_id"] == args.case_id]
        if not selected:
            raise ValueError("Unknown case-id")
        # A completed or interrupted case cannot accidentally be run again.
        if any((run_root / c["case_id"]).exists() for c, _ in selected):
            raise ValueError("Case already started; aggregate completed results or use a new run-dir")
        client = make_client(models[0], base_url, read_key())
        for case, (video, clip, timeline) in selected:
            case_root = run_root / case["case_id"]
            case_root.mkdir()
            write_json(case_root / "source_timeline_subset.json", timeline.model_dump(mode="json"))
            metrics = run_case(case, video, clip, timeline, library, client, case_root / "jobs")
            metrics.update(model=models[0], run_signature=signature)
            write_json(case_root / "result.json", metrics)
            result["results"].append(metrics)
        if args.case_id:
            write_json(args.selection_output, result)
            print(f"Case saved: {run_root / args.case_id / 'result.json'}; selection incomplete")
            return 0 if result["results"][0]["passed"] else 1
    result["results"] = aggregate_cases(run_root, [c["case_id"] for c in cases], signature)
    aggregate = {"model": models[0], "passed": all(c["passed"] for c in result["results"]),
                 "estimated_provider_cost": None}
    result.update(select_model([aggregate]))
    result["completed_utc"] = stamp()
    result["selected_role"] = "visual_director" if aggregate["passed"] else None
    write_json(run_root / "benchmark_result.json", result)
    write_json(args.selection_output, result)
    append_audit(args.audit_output, result)
    print(f"Benchmark {result['status']}: {args.selection_output}")
    return 0 if aggregate["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Provider exceptions can contain credentials; never print their text.
        print(f"Benchmark failed ({type(error).__name__}); inspect local artifacts", file=sys.stderr)
        raise SystemExit(2) from None
