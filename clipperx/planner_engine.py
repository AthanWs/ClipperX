"""Resumable staged editorial planner orchestration."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from .evidence import collect_evidence
from .jobs import JobStore
from .planner import prepare_transcript_batches
from .planner_contracts import (
    BoundaryReviewResponse,
    DiscoveryBatchResponse,
    EvidenceRequest,
    GlobalCompositionResponse,
)
from .planner_prompts import (
    BOUNDARY_SYSTEM_PROMPT,
    COMPOSITION_SYSTEM_PROMPT,
    DISCOVERY_SYSTEM_PROMPT,
    EDIT_DIRECTION_RULES,
)
from .planning_state import PlanningLedger
from .structured_client import StructuredResult, _strict_json_schema

if TYPE_CHECKING:
    from .style_library import StyleLibrary
    from .visual_director import VisualDirectorLimits

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredClient(Protocol):
    def generate(
        self,
        output_model: type[ModelT],
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[Path] | None = None,
        temperature: float = 0.2,
    ) -> StructuredResult[ModelT]: ...


@dataclass(frozen=True, slots=True)
class PlannerRunLimits:
    target_clips: int | None = 3
    max_candidates: int = 6
    max_initial_evidence_requests: int = 6
    max_boundary_calls: int = 8
    max_images_per_request: int = 4
    max_total_reported_tokens: int = 60_000

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "target_clips" and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError("Todos los límites del planner deben ser enteros positivos")
        if self.target_clips is not None and self.max_candidates < self.target_clips:
            raise ValueError("max_candidates no puede ser menor que target_clips")


class PlannerBudgetExceeded(RuntimeError):
    """A pending model stage cannot start under the run's reported-token ceiling."""

    def __init__(self, stage: str, reported_tokens: int, limit: int) -> None:
        self.stage = stage
        self.reported_tokens = reported_tokens
        self.limit = limit
        super().__init__(f"Planner token budget exhausted before {stage} ({reported_tokens}/{limit})")


@dataclass(frozen=True, slots=True)
class PlannerRunArtifacts:
    composition_plan: Path
    visual_plan_manifest: Path
    run_summary: Path


def run_editorial_and_visual_planner(
    video_path: Path,
    transcript_path: Path,
    *,
    editorial_client: StructuredClient,
    visual_client: StructuredClient,
    model_name: str,
    context_window_tokens: int,
    max_output_tokens: int,
    source_timeline_path: Path,
    style_library: StyleLibrary,
    editorial_limits: PlannerRunLimits,
    visual_limits: VisualDirectorLimits,
    jobs_root: Path | None = None,
) -> PlannerRunArtifacts:
    """Persist editorial v1 first, then direct detached copies of its clips."""
    composition_path = run_editorial_planner(
        video_path, transcript_path, discovery_client=editorial_client,
        evidence_client=editorial_client, model_name=model_name,
        context_window_tokens=context_window_tokens, max_output_tokens=max_output_tokens,
        limits=editorial_limits, jobs_root=jobs_root,
    )
    return run_visual_planner(
        video_path, composition_path, source_timeline_path=source_timeline_path,
        client=visual_client, style_library=style_library, limits=visual_limits, jobs_root=jobs_root,
        transcript_path=transcript_path,
    )


def _bounded_json(path: Path) -> Any:
    # Bound even injected/malformed references before parsing. No network resolution.
    with path.open("rb") as stream:
        content = stream.read(16 * 1024 * 1024 + 1)
    if len(content) > 16 * 1024 * 1024:
        raise ValueError("Planning JSON exceeds 16 MiB")
    return json.loads(content)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def _versioned_json(path: Path, version: int) -> dict[str, Any]:
    payload = _bounded_json(path)
    if (not isinstance(payload, dict) or type(payload.get("contract_version")) is not int
            or payload["contract_version"] != version):
        raise ValueError(f"Planning artifact requires explicit contract_version={version}")
    return payload


def _persisted_composition(path: Path) -> GlobalCompositionResponse:
    payload = _versioned_json(path, 1)
    if (not isinstance(payload.get("clips"), list)
            or set(payload) - set(GlobalCompositionResponse.model_fields)):
        raise ValueError("Persisted composition requires explicit clips and known fields")
    return GlobalCompositionResponse.model_validate(payload)


def run_visual_planner(
    video_path: Path,
    composition_plan_path: Path,
    *,
    source_timeline_path: Path,
    client: StructuredClient,
    style_library: StyleLibrary,
    limits: VisualDirectorLimits,
    jobs_root: Path | None = None,
    transcript_path: Path | None = None,
) -> PlannerRunArtifacts:
    """Link persisted editorial v1 to ready v2 artifacts or explicit review receipts.

    A fresh UUID manifest records progress after every clip. Malformed downstream
    output raises, while preserving prior successes and marking remaining clips pending.
    """
    from .local_paths import atomic_write_text, source_fingerprint
    from .visual_contracts import SourceVisualTimeline, VisualCompositionPlan
    from .visual_director import run_visual_director, DirectorPublicationError, _load_terminal
    from .visual_provenance import file_sha256, visual_timeline_fingerprint

    composition_path = composition_plan_path.resolve(strict=True)
    timeline_path = source_timeline_path.resolve(strict=True)
    source = video_path.resolve(strict=True)
    composition = _persisted_composition(composition_path)
    timeline = SourceVisualTimeline.model_validate(_versioned_json(timeline_path, 1))
    if len({clip.clip_id for clip in composition.clips}) != len(composition.clips):
        raise ValueError("Editorial clip IDs must be unique")
    source_fp = source_fingerprint(source)
    if source_fp != timeline.source_fingerprint:
        raise ValueError("Source timeline fingerprint differs from source")
    composition_fp = _canonical_hash(composition.model_dump(mode="json"))
    timeline_fp = visual_timeline_fingerprint(timeline)
    timeline_sha = file_sha256(timeline_path)
    selected_text = None if transcript_path is None else transcript_path.resolve(strict=True)
    text_sha = None if selected_text is None else file_sha256(selected_text)
    style_fp = style_library.fingerprint
    identity_fp = _canonical_hash({
        "face_tracks": [track.model_dump(mode="json") for track in timeline.face_tracks],
        "webcam_tracks": [track.model_dump(mode="json") for track in timeline.webcam_tracks],
        "local_identity": [(span.span_id, span.identity_status.value, span.streamer_track_id,
                            span.webcam_track_id) for span in timeline.spans],
    })
    root = composition_path.parent / "visual_runs" / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    artifacts = PlannerRunArtifacts(composition_path, root / "visual_plan_manifest.json",
                                    root / "run_summary.json")
    entries: list[dict[str, Any]] = [
        {"clip_id": clip.clip_id, "status": "pending", "visual_plan": None, "receipt": None}
        for clip in composition.clips
    ]
    manifest = {
        "manifest_version": 1, "editorial_contract_version": 1, "visual_contract_version": 2,
        "editorial_plan": str(composition_path), "composition_fingerprint": composition_fp,
        "source_timeline": str(timeline_path), "source_timeline_fingerprint": timeline_fp,
        "source_timeline_sha256":timeline_sha,
        "source_fingerprint": source_fp, "style_library_root": str(style_library.root),
        "style_library_fingerprint": style_fp, "status": "running", "clips": entries,
    }
    if selected_text is not None:
        manifest.update(transcript_path=str(selected_text), transcript_sha256=text_sha)

    def persist(status: str) -> None:
        manifest["status"] = status
        summary = {
            "status": status, "editorial_plan": str(composition_path),
            "visual_plan_manifest": str(artifacts.visual_plan_manifest),
            "total_clips": len(entries),
            **{f"{state}_clips": sum(entry["status"] == state for entry in entries)
               for state in ("ready", "review_required", "pending", "error")},
        }
        for path, payload in ((artifacts.visual_plan_manifest, manifest),
                              (artifacts.run_summary, summary)):
            atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def check_inputs() -> None:
        if (source_fingerprint(source) != source_fp
                or _canonical_hash(_persisted_composition(
                    composition_path).model_dump(mode="json")) != composition_fp
                or file_sha256(timeline_path)!=timeline_sha
                or visual_timeline_fingerprint(SourceVisualTimeline.model_validate(
                    _versioned_json(timeline_path, 1))) != timeline_fp
                or selected_text is not None and file_sha256(selected_text) != text_sha
                or style_library.fingerprint != style_fp):
            raise ValueError("Planning inputs changed during visual planning")

    persist("running")
    for index, clip in enumerate(composition.clips):
        try:
            check_inputs()
            bookkeeping_error=False
            try:
                output = Path(run_visual_director(
                    source, clip.model_copy(deep=True), timeline.model_copy(deep=True),
                    client=client, style_library=style_library, limits=limits, jobs_root=jobs_root,
                    source_timeline_path=timeline_path,
                    **({} if selected_text is None else {"transcript_path": selected_text}),
                )).resolve(strict=True)
            except DirectorPublicationError as exc:
                if exc.phase!="COMMITTED_BOOKKEEPING_FAILED" or exc.committed_path is None:
                    raise
                output=exc.committed_path.resolve(strict=True)
                committed=_load_terminal(exc.run_path)
                if committed is None or committed[0]!=output or committed[1]["status"]!="ready":
                    raise
                bookkeeping_error=True
            # Receipt is the Task6 publication marker; its filename is host-owned.
            if output.name not in {"visual_plan.json", "receipt.json"}:
                raise ValueError("Unknown visual director artifact")
            receipt_path = output.parent / "receipt.json"
            terminal=_load_terminal(output.parent)
            if terminal is None:
                raise ValueError("Missing terminal visual director receipt")
            _,receipt,companion=terminal
            receipt_sha=file_sha256(receipt_path)
            if (not isinstance(receipt, dict)
                    or receipt.get("clip_id") != clip.clip_id
                    or receipt.get("source_fingerprint") != source_fp
                    or type(receipt.get("calls")) is not int or receipt["calls"] < 0):
                raise ValueError("Malformed visual director receipt")
            host=companion.get("host_context")
            if selected_text is not None and (not isinstance(host, dict)
                    or host.get("text", {}).get("path") != str(selected_text)
                    or host.get("text", {}).get("sha256") != text_sha):
                raise ValueError("Visual terminal transcript association mismatch")
            if (not isinstance(host,dict) or host.get("upstream",{}).get("path")!=str(timeline_path)
                    or host["upstream"].get("sha256")!=timeline_sha
                    or host["upstream"].get("context",{}).get("timeline_fingerprint")!=timeline_fp
                    or host["upstream"]["context"].get("source_fingerprint")!=source_fp):
                raise ValueError("Visual terminal upstream association mismatch")
            status = receipt.get("status")
            if status == "ready" and output.name == "visual_plan.json":
                plan = VisualCompositionPlan.model_validate(_versioned_json(output, 2))
                expected = {
                    "clip_id": clip.clip_id, "source_fingerprint": source_fp,
                    "composition_fingerprint": _canonical_hash(clip.model_dump(mode="json")),
                    "source_timeline_fingerprint": timeline_fp, "identity_fingerprint": identity_fp,
                    "style_library_fingerprint": style_fp,
                }
                if (any(getattr(plan, name) != value for name, value in expected.items())
                        or not isinstance(receipt.get("plan_path"), str)
                        or Path(receipt["plan_path"]).resolve() != output):
                    raise ValueError("Visual plan association/fingerprint mismatch")
                check_inputs()
                _load_terminal(output.parent)
                if (file_sha256(receipt_path)!=receipt_sha
                        or file_sha256(output.parent/"provenance.json")!=receipt["companion_sha256"]):
                    raise ValueError("Visual terminal identity changed before manifest publication")
                entries[index].update(status="ready", visual_plan=str(output),
                                      receipt=str(receipt_path))
            elif (status == "review_required" and output.name == "receipt.json"
                  and isinstance(receipt.get("reason"), str) and receipt["reason"].strip()
                  and not (output.parent / "visual_plan.json").exists()):
                entries[index].update(status="review_required", receipt=str(receipt_path))
            else:
                raise ValueError("Unknown or malformed visual director output status")
            entries[index].update(provenance=str(output.parent/"provenance.json"),
                provenance_sha256=receipt["companion_sha256"],host_context=host)
            if bookkeeping_error:
                entries[index]["phase"]="bookkeeping_error"
            check_inputs()
        except Exception:
            entries[index].update(status="error")
            persist("error")
            raise
        persist("running")
    ready_count = sum(entry["status"] == "ready" for entry in entries)
    status = ("empty" if not entries else "ready" if ready_count == len(entries)
              else "partial" if ready_count else "review_required")
    persist(status)
    return artifacts


def run_editorial_planner(
    video_path: Path,
    transcript_path: Path,
    *,
    discovery_client: StructuredClient,
    evidence_client: StructuredClient | None,
    model_name: str,
    context_window_tokens: int,
    max_output_tokens: int | None = None,
    limits: PlannerRunLimits | None = None,
    force: bool = False,
    jobs_root: Path | None = None,
) -> Path:
    limits = limits or PlannerRunLimits()
    limits.validate()
    job = JobStore.for_video(video_path, jobs_root=jobs_root)
    root = job.planning_dir / "runs" / _planning_run_id(
        transcript_path,
        model_name=model_name,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        limits=limits,
    )
    root.mkdir(parents=True, exist_ok=True)
    if limits.target_clips is None:
        _validate_coverage_budget_increase(root, limits)
    if force:
        _archive_composition(root)
    _write_json(root / "run_config.json", {"limits": asdict(limits), "model": model_name})
    batches_dir = root / "batches"
    manifest_path = prepare_transcript_batches(
        transcript_path,
        batches_dir,
        model=model_name,
        context_window_tokens=context_window_tokens,
        reserved_output_tokens=max_output_tokens,
    )
    ledger_path = root / "ledger.json"
    coverage_identity = ({"source_fingerprint": job.fingerprint,
        "transcript_sha256": hashlib.sha256(transcript_path.read_bytes()).hexdigest(),
        "model": model_name} if limits.target_clips is None else None)
    if coverage_identity is not None:
        discovery_identity = {**coverage_identity, "system_prompt": DISCOVERY_SYSTEM_PROMPT,
            "output_schema": _strict_json_schema(DiscoveryBatchResponse),
            "limits": _coverage_semantic_limits(limits), "context_window_tokens": context_window_tokens,
            "max_output_tokens": max_output_tokens}
        identity_path = root / "discovery_identity.json"
        if identity_path.exists() and _bounded_json(identity_path) != discovery_identity:
            history = root / "discovery_history" / uuid4().hex
            history.mkdir(parents=True, exist_ok=False)
            for previous in (identity_path, ledger_path):
                if previous.exists():
                    shutil.copy2(previous, history / previous.name)
            PlanningLedger.create(manifest_path, ledger_path)
        _write_json(identity_path, discovery_identity)
    if force or not ledger_path.exists():
        ledger = PlanningLedger.create(manifest_path, ledger_path)
    else:
        ledger = PlanningLedger(ledger_path)
    usage_records: list[dict[str, Any]] = ([] if force and limits.target_clips is not None
        else _load_usage(root / "usage.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    coverage_candidate_ids: set[str] = set()
    discovery_deferred: list[dict[str, Any]] = []
    for batch_record in manifest["batches"]:
        state = ledger.load()
        batch_state = state["batches"][batch_record["batch_id"]]
        if coverage_identity is not None:
            response = _run_coverage_discovery_batch(root, ledger, batch_record,
                client=discovery_client, usage_records=usage_records, limits=limits,
                identity=discovery_identity, force=force, transcript_path=transcript_path,
                previous_ids=coverage_candidate_ids, deferred=discovery_deferred)
            if response is None:
                break
            ids = {candidate.candidate_id for candidate in response.candidates}
            if coverage_candidate_ids & ids:
                raise ValueError("candidate_id reused across discovery batches")
            coverage_candidate_ids.update(ids)
            continue
        if batch_state["status"] == "complete":
            continue
        _check_run_token_budget(root, usage_records, limits, stage="discovery")
        batch = json.loads(Path(batch_record["path"]).read_text(encoding="utf-8"))
        result = discovery_client.generate(
            DiscoveryBatchResponse,
            system_prompt=DISCOVERY_SYSTEM_PROMPT,
            user_prompt=_discovery_user_prompt(batch, limits),
            temperature=0.2,
        )
        _validate_discovery_limits(result.value, limits)
        ledger.apply_discovery(result.value)
        usage_records.append(_usage_record("discovery", batch["batch_id"], result.usage))
        _write_json(root / "usage.json", usage_records)

    state = ledger.load()
    if coverage_identity is not None and set(state.get("candidates", {})) != coverage_candidate_ids:
        raise ValueError("Discovery ledger contains candidates outside validated cached responses")
    candidates = list(state.get("candidates", {}).values())
    reviews, unresolved = _run_evidence_reviews(
        video_path,
        transcript_path,
        candidates,
        evidence_client=evidence_client or discovery_client,
        output_dir=root / "reviews",
        usage_records=usage_records,
        usage_path=root / "usage.json",
        limits=limits,
        force=force,
        cache_identity=({**coverage_identity, "streamer_identity": _load_optional(job.speaker_analysis_path)}
            if coverage_identity is not None else None),
    )
    if limits.target_clips is None:
        return _run_coverage_composition(root, candidates, reviews, unresolved,
            client=discovery_client, usage_records=usage_records, limits=limits,
            identity={**(coverage_identity or {}),
                "discovery_identity_sha256": _canonical_hash(discovery_identity),
                "boundary_prompt": BOUNDARY_SYSTEM_PROMPT,
                "boundary_schema": BoundaryReviewResponse.model_json_schema()},
            streamer_identity=_load_optional(job.speaker_analysis_path),
            model_name=model_name, context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens, force=force,
            discovery_deferred=discovery_deferred)
    system_prompt = f"{COMPOSITION_SYSTEM_PROMPT}\n\n{EDIT_DIRECTION_RULES}"
    user_prompt = json.dumps({
        "candidates": candidates,
        "boundary_reviews": reviews,
        "unresolved_evidence_due_to_limits": unresolved,
        "run_limits": asdict(limits),
        "streamer_identity": _load_optional(job.speaker_analysis_path),
    }, ensure_ascii=False)
    request_sha256 = _canonical_hash({
        "source_fingerprint": job.fingerprint,
        "transcript_sha256": hashlib.sha256(transcript_path.read_bytes()).hexdigest(),
        "model": model_name,
        "context_window_tokens": context_window_tokens,
        "max_output_tokens": max_output_tokens,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "output_schema": GlobalCompositionResponse.model_json_schema(),
        "temperature": 0.2,
    })
    plan_path = root / "composition_plan.json"
    if not force and _composition_cache_matches(plan_path, request_sha256):
        _write_json(root / "run_summary.json", _run_summary(usage_records, reviews, unresolved))
        return plan_path
    _check_run_token_budget(root, usage_records, limits, stage="composition",
        reviews=reviews, unresolved=unresolved)
    composition_result = discovery_client.generate(GlobalCompositionResponse,
        system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
    composition = GlobalCompositionResponse.model_validate(composition_result.value.model_dump())
    if not force:
        _archive_composition(root)
    _write_json(plan_path, composition.model_dump(mode="json"))
    usage_records.append(_usage_record("composition", "global", composition_result.usage))
    _write_json(root / "usage.json", usage_records)
    _write_json(root / "composition_receipt.json", {
        "version": 1, "kind": "editorial_composition_cache",
        "request_sha256": request_sha256,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    })
    _write_json(root / "run_summary.json", _run_summary(usage_records, reviews, unresolved))
    return plan_path


def _run_coverage_discovery_batch(root: Path, ledger: PlanningLedger,
        batch_record: dict[str, Any], *, client: StructuredClient,
        usage_records: list[dict[str, Any]], limits: PlannerRunLimits,
        identity: dict[str, Any], force: bool, transcript_path: Path,
        previous_ids: set[str], deferred: list[dict[str, Any]]) -> DiscoveryBatchResponse | None:
    from .context_budget import _segment_line
    from .schema import TranscriptSegment

    original = _bounded_json(Path(batch_record["path"]))
    segments = [TranscriptSegment.from_dict(item) for item in _bounded_json(transcript_path)]
    state = ledger.load()
    complete = state["batches"][batch_record["batch_id"]]["status"] == "complete"

    def discover(batch: dict[str, Any]) -> list[Any] | None:
        user_prompt = _discovery_user_prompt(batch, limits)
        request_sha = _canonical_hash({"identity": identity, "user_prompt": user_prompt})
        output = root / "discovery" / request_sha / "response.json"
        receipt_path = output.parent / "receipt.json"
        if receipt_path.exists() and not force:
            receipt = _bounded_json(receipt_path)
            if (not isinstance(receipt, dict) or set(receipt) != {
                    "version", "kind", "request_sha256", "plan_sha256"}
                    or type(receipt["version"]) is not int or receipt["version"] != 1
                    or receipt["kind"] != "editorial_discovery_cache"
                    or receipt["request_sha256"] != request_sha or not output.is_file()
                    or receipt["plan_sha256"] != hashlib.sha256(output.read_bytes()).hexdigest()):
                raise ValueError("Invalid discovery cache receipt/hash")
            response = DiscoveryBatchResponse.model_validate(_versioned_json(output, 1))
        else:
            if (complete or output.exists() or any(record.get("stage") == "discovery"
                    and record.get("request_sha256") == request_sha for record in usage_records)) and not force:
                raise ValueError("Discovery response or ledger has no valid cache receipt; refusing duplicate purchase")
            _check_run_token_budget(root, usage_records, limits, stage="discovery")
            result = client.generate(DiscoveryBatchResponse, system_prompt=DISCOVERY_SYSTEM_PROMPT,
                user_prompt=user_prompt, temperature=0.2)
            usage_records.append({**_usage_record("discovery", batch["batch_id"], result.usage),
                "request_sha256": request_sha})
            _write_json(root / "usage.json", usage_records)
            response = DiscoveryBatchResponse.model_validate(result.value.model_dump(warnings="error"))
            _validate_discovery_limits(response, limits)
            if (response.batch_id != batch["batch_id"] or response.cursor_before != batch["cursor_before"]
                    or response.cursor_after != batch["cursor_after"]):
                raise ValueError("Discovery response does not match offered batch/cursors")
            if force and output.exists():
                history = output.parent / "history" / uuid4().hex
                history.mkdir(parents=True, exist_ok=False)
                for previous in (output, receipt_path):
                    if previous.exists():
                        shutil.copy2(previous, history / previous.name)
            _write_json(output, response.model_dump(mode="json"))
            _write_json(receipt_path, {"version": 1, "kind": "editorial_discovery_cache",
                "request_sha256": request_sha,
                "plan_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        _validate_discovery_limits(response, limits)
        if (response.batch_id != batch["batch_id"] or response.cursor_before != batch["cursor_before"]
                or response.cursor_after != batch["cursor_after"]):
            raise ValueError("Discovery response does not match offered batch/cursors")
        if len(response.candidates) < limits.max_candidates:
            return list(response.candidates)
        before, after = batch["cursor_before"], batch["cursor_after"]
        if after - before <= 1:
            deferred.append({"batch_id": batch_record["batch_id"], "request_sha256": request_sha,
                "cursor_before": before, "cursor_after": after, "reason": "saturated_indivisible"})
            return None
        mid = (before + after) // 2
        children = []
        for child_before, child_after in ((before, mid), (mid, after)):
            child = dict(batch)
            child.update(batch_id=_canonical_hash({"parent": batch["batch_id"],
                "before": child_before, "after": child_after})[:16],
                new_start_index=child_before, new_end_index=child_after,
                cursor_before=child_before, cursor_after=child_after,
                text="\n".join(_segment_line(i, segments[i])
                    for i in range(batch["context_start_index"], child_after)))
            children.append(discover(child))
        if any(child is None for child in children):
            return None
        return [candidate for child in children for candidate in child]

    candidates = discover(original)
    if candidates is None:
        _write_json(root / "discovery_deferred.json", {"version": 1, "items": deferred})
        return None
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids) or set(ids) & previous_ids:
        raise ValueError("candidate_id reused across discovery leaves or batches")
    response = DiscoveryBatchResponse(batch_id=original["batch_id"],
        cursor_before=original["cursor_before"], cursor_after=original["cursor_after"],
        candidates=candidates)
    if complete and any(state["candidates"].get(candidate.candidate_id)
            != candidate.model_dump(mode="json") for candidate in candidates):
        raise ValueError("Discovery candidate ledger differs from cached response")
    ledger.apply_discovery(response)
    return response


def _run_coverage_composition(root: Path, candidates: list[dict[str, Any]],
        reviews: list[dict[str, Any]], unresolved: list[dict[str, Any]], *,
        client: StructuredClient, usage_records: list[dict[str, Any]], limits: PlannerRunLimits,
        identity: dict[str, Any], streamer_identity: Any, model_name: str,
        context_window_tokens: int, max_output_tokens: int | None, force: bool,
        discovery_deferred: list[dict[str, Any]]) -> Path:
    from .composition_groups import (build_composition_groups, merge_group_compositions,
        validate_group_composition)

    config = getattr(client, "config", None)
    reserved = max_output_tokens or min(4096, max(1024, context_window_tokens // 4))
    reserved = max(reserved, getattr(config, "max_output_tokens", reserved))
    groups = build_composition_groups(candidates, reviews, unresolved,
        system_prompt=f"{COMPOSITION_SYSTEM_PROMPT}\n\n{EDIT_DIRECTION_RULES}",
        identity=identity, run_limits=_coverage_semantic_limits(limits), streamer_identity=streamer_identity,
        context_window_tokens=min(context_window_tokens,
            getattr(config, "context_window_tokens", context_window_tokens)),
        max_output_tokens=reserved, model=model_name,
        input_utilization=min(0.85, getattr(config, "input_utilization", 0.85)))
    entries: list[dict[str, Any]] = []
    completed: dict[str, GlobalCompositionResponse] = {}
    plan_path = root / "composition_plan.json"
    archived = force

    def persist(status: str, *, publish: bool = False) -> None:
        _write_json(root / "coverage_groups.json", {"version": 1, "status": status,
            "groups": entries})
        summary = _run_summary(usage_records, reviews, unresolved)
        unassigned = sorted({candidate_id for entry in entries
            for candidate_id in entry.get("unassigned_candidate_ids", [])})
        if status == "complete" and (discovery_deferred or unassigned or unresolved):
            status = "partial"
            _write_json(root / "coverage_groups.json", {"version": 1, "status": status,
                "groups": entries})
        summary.update(coverage_mode=True, status=status, total_groups=len(entries),
            discovery_deferred=discovery_deferred, unassigned_candidate_ids=unassigned,
            completed_groups=len(completed),
            deferred_groups=sum(entry["status"] == "deferred" for entry in entries),
            pending_groups=sum(entry["status"] == "pending" for entry in entries),
            coverage_groups_manifest=str(root / "coverage_groups.json"))
        if status == "budget_exhausted":
            summary.update(stage="composition", max_total_reported_tokens=limits.max_total_reported_tokens)
        if publish:
            composition = merge_group_compositions([(group["group_id"], completed[group["group_id"]])
                for group in groups if group["group_id"] in completed], unresolved)
            request_sha = _canonical_hash({"mode": "coverage", "groups": entries})
            if force or not _composition_cache_matches(plan_path, request_sha):
                if not archived:
                    _archive_composition(root)
                _write_json(plan_path, composition.model_dump(mode="json"))
                _write_json(root / "composition_receipt.json", {"version": 1,
                    "kind": "editorial_composition_cache", "request_sha256": request_sha,
                    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest()})
        _write_json(root / "run_summary.json", summary)

    # Load and validate every available group before checking the new-call budget.
    for group in groups:
        directory = root / "composition_groups" / group["group_id"]
        request_path, output, receipt_path = (directory / name
            for name in ("request.json", "plan.json", "receipt.json"))
        if force and directory.exists():
            history = directory / "history" / uuid4().hex
            history.mkdir(parents=True, exist_ok=False)
            for path in (request_path, output, receipt_path):
                if path.is_file():
                    shutil.copy2(path, history / path.name)
        if request_path.exists() and _canonical_hash(_bounded_json(request_path)) != group["request_sha256"]:
            raise ValueError("Composition group request identity/hash mismatch")
        if not request_path.exists():
            _write_json(request_path, group["request"])
        entry = {"group_id": group["group_id"], "request_sha256": group["request_sha256"],
            "request_path": str(request_path), "plan_path": str(output),
            "candidate_ids": group["candidate_ids"], "status": "pending"}
        if group["deferred_reason"]:
            entry.update(status="deferred", reason=group["deferred_reason"])
            _write_json(receipt_path, {"version": 1, "kind": "editorial_group_deferred",
                "request_sha256": group["request_sha256"], "reason": group["deferred_reason"]})
        elif receipt_path.exists() and not force:
            receipt = _bounded_json(receipt_path)
            if (not isinstance(receipt, dict) or set(receipt) != {
                    "version", "kind", "request_sha256", "plan_sha256"}
                    or type(receipt["version"]) is not int or receipt["version"] != 1
                    or receipt["kind"] != "editorial_group_cache"
                    or receipt["request_sha256"] != group["request_sha256"]
                    or not output.is_file()
                    or receipt["plan_sha256"] != hashlib.sha256(output.read_bytes()).hexdigest()):
                raise ValueError("Invalid composition group receipt/hash")
            composition = _persisted_composition(output)
            validate_group_composition(composition, group["request"]["payload"])
            completed[group["group_id"]] = composition
            entry.update(status="complete", plan_sha256=receipt["plan_sha256"],
                unassigned_candidate_ids=_unassigned_coverage_candidates(composition, group["request"]["payload"]))
        elif output.exists() and not force:
            raise ValueError("Composition group plan has no valid receipt; refusing duplicate purchase")
        entries.append(entry)
    if not archived and plan_path.exists() and any(entry["status"] == "pending" for entry in entries):
        # Preserve the prior usage and terminal summary before progress/new model calls overwrite them.
        _composition_cache_matches(plan_path, "validate-existing-before-recomposition")
        _archive_composition(root)
        archived = True
    persist("running")
    for group, entry in zip(groups, entries):
        if entry["status"] != "pending":
            continue
        if _reported_tokens(usage_records) >= limits.max_total_reported_tokens:
            persist("budget_exhausted", publish=bool(completed))
            raise PlannerBudgetExceeded("composition", _reported_tokens(usage_records),
                limits.max_total_reported_tokens)
        try:
            result = client.generate(GlobalCompositionResponse,
                system_prompt=group["request"]["system_prompt"],
                user_prompt=group["request"]["user_prompt"], temperature=0.2)
            usage_records.append(_usage_record("composition", group["group_id"], result.usage))
            _write_json(root / "usage.json", usage_records)
            composition = GlobalCompositionResponse.model_validate(result.value.model_dump(warnings="error"))
            validate_group_composition(composition, group["request"]["payload"])
            # Reject cross-group duplicate selections before committing the new group.
            merge_group_compositions([*completed.items(), (group["group_id"], composition)], unresolved)
            output = Path(entry["plan_path"])
            _write_json(output, composition.model_dump(mode="json"))
            sha = hashlib.sha256(output.read_bytes()).hexdigest()
            _write_json(output.parent / "receipt.json", {"version": 1,
                "kind": "editorial_group_cache", "request_sha256": group["request_sha256"],
                "plan_sha256": sha})
            completed[group["group_id"]] = composition
            entry.update(status="complete", plan_sha256=sha,
                unassigned_candidate_ids=_unassigned_coverage_candidates(composition, group["request"]["payload"]))
        except Exception:
            entry["status"] = "error"
            persist("error")
            raise
        persist("running")
    persist("partial" if any(entry["status"] == "deferred" for entry in entries)
        else "complete", publish=True)
    return plan_path


def _unassigned_coverage_candidates(composition: GlobalCompositionResponse,
                                   payload: dict[str, Any]) -> list[str]:
    """Only uniquely matching topic and source ranges can be attributed automatically."""
    candidates = payload["candidates"]
    assigned = set(composition.rejected_candidate_ids)
    for clip in composition.clips:
        matches = {candidate["candidate_id"] for candidate in candidates
            if candidate.get("topic_id") == clip.topic_id and any(
                segment.source.start < float(item["end"])
                and segment.source.end > float(item["start"])
                for segment in clip.segments for item in candidate["transcript_ranges"])}
        if len(matches) == 1:
            assigned.update(matches)
    return sorted(candidate["candidate_id"] for candidate in candidates
        if candidate["candidate_id"] not in assigned)


def _run_evidence_reviews(
    video_path: Path,
    transcript_path: Path,
    candidates: list[dict[str, Any]],
    *,
    evidence_client: StructuredClient,
    output_dir: Path,
    usage_records: list[dict[str, Any]],
    usage_path: Path,
    limits: PlannerRunLimits,
    force: bool,
    cache_identity: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    queue = [
        EvidenceRequest.model_validate(request)
        for candidate in candidates
        for request in candidate.get("evidence_requests", [])
    ]
    processed: set[str] = set()
    reviews: list[dict[str, Any]] = []
    boundary_calls = sum(record.get("stage") == "boundary" for record in usage_records)
    unresolved: list[dict[str, Any]] = []
    while queue:
        request = queue.pop(0)
        request_key = (_review_cache_key(request, candidates, cache_identity)
            if cache_identity is not None else _request_key(request))
        if request_key in processed:
            continue
        review_path = output_dir / f"{_safe_name(request.request_id)}-{request_key[:12]}.json"
        receipt_path = review_path.with_suffix(".receipt.json")
        if review_path.exists() and not force:
            if cache_identity is not None:
                receipt = _bounded_json(receipt_path)
                if (not isinstance(receipt, dict) or set(receipt) != {
                        "version", "kind", "request_sha256", "plan_sha256"}
                        or type(receipt["version"]) is not int or receipt["version"] != 1
                        or receipt["kind"] != "editorial_review_cache"
                        or receipt["request_sha256"] != request_key
                        or receipt["plan_sha256"] != hashlib.sha256(review_path.read_bytes()).hexdigest()):
                    raise ValueError("Invalid boundary review cache receipt/hash")
            review_payload = _bounded_json(review_path)
            review = BoundaryReviewResponse.model_validate(review_payload)
        else:
            if (boundary_calls >= limits.max_boundary_calls
                    or _reported_tokens(usage_records) >= limits.max_total_reported_tokens):
                unresolved.append(request.model_dump(mode="json"))
                processed.add(request_key)
                continue
            evidence_path = collect_evidence(
                video_path,
                request,
                transcript_path=transcript_path,
                force=force,
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            result = evidence_client.generate(
                BoundaryReviewResponse,
                system_prompt=BOUNDARY_SYSTEM_PROMPT,
                user_prompt=_boundary_user_prompt(request, evidence, candidates),
                images=_select_evidence_images(
                    evidence,
                    limit=limits.max_images_per_request,
                ),
                temperature=0.1,
            )
            boundary_calls += 1
            review = BoundaryReviewResponse.model_validate(result.value.model_dump(warnings="error"))
            if review.candidate_id != request.candidate_id:
                raise ValueError("Boundary review candidate does not match its evidence request")
            if force and cache_identity is not None and review_path.exists():
                history = output_dir / "history" / uuid4().hex
                history.mkdir(parents=True, exist_ok=False)
                for previous in (review_path, receipt_path):
                    if previous.exists():
                        shutil.copy2(previous, history / previous.name)
            _write_json(review_path, review.model_dump(mode="json"))
            if cache_identity is not None:
                _write_json(receipt_path, {"version": 1, "kind": "editorial_review_cache",
                    "request_sha256": request_key,
                    "plan_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest()})
            usage_records.append(_usage_record("boundary", request.request_id, result.usage))
            _write_json(usage_path, usage_records)
        if review.candidate_id != request.candidate_id:
            raise ValueError("Boundary review candidate does not match its evidence request")
        reviews.append(review.model_dump(mode="json"))
        processed.add(request_key)
        for follow_up in review.evidence_requests:
            if follow_up.iteration <= request.iteration:
                raise RuntimeError(
                    "Cada EvidenceRequest de seguimiento debe incrementar iteration"
                )
            follow_up_key = (_review_cache_key(follow_up, candidates, cache_identity)
                if cache_identity is not None else _request_key(follow_up))
            if follow_up_key not in processed:
                queue.append(follow_up)
    return reviews, unresolved


def _discovery_user_prompt(batch: dict[str, Any], limits: PlannerRunLimits) -> str:
    return json.dumps(
        {
            "batch_id": batch["batch_id"],
            "cursor_before": batch["cursor_before"],
            "cursor_after": batch["cursor_after"],
            "context_start_index": batch["context_start_index"],
            "new_start_index": batch["new_start_index"],
            "new_end_index": batch["new_end_index"],
            "run_limits": {
                "target_clips": limits.target_clips,
                "max_candidates": limits.max_candidates,
                "max_initial_evidence_requests": limits.max_initial_evidence_requests,
            },
            "transcript": batch["text"],
        },
        ensure_ascii=False,
    )


def _boundary_user_prompt(
    request: EvidenceRequest,
    evidence: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    nearby = [
        candidate
        for candidate in candidates
        if any(
            float(item["end"]) >= request.window.start - 60
            and float(item["start"]) <= request.window.end + 60
            for item in candidate.get("transcript_ranges", [])
        )
    ]
    samples = [
        {
            "timestamp": sample.get("timestamp"),
            "visual_change": sample.get("visual_change"),
            "faces": sample.get("faces"),
            "audio": sample.get("audio"),
        }
        for sample in evidence.get("samples", [])
    ]
    return json.dumps(
        {
            "request": request.model_dump(mode="json"),
            "visual_summary": evidence.get("visual_summary"),
            "samples": samples,
            "transcript": evidence.get("transcript"),
            "speaker_analysis": evidence.get("speaker_analysis"),
            "streamer_profile": evidence.get("streamer_profile"),
            "nearby_candidates": nearby,
        },
        ensure_ascii=False,
    )


def _select_evidence_images(evidence: dict[str, Any], *, limit: int = 8) -> list[Path]:
    samples = evidence.get("samples", [])
    if len(samples) <= limit:
        return [Path(sample["frame"]) for sample in samples if sample.get("frame")]
    ranked_indices = sorted(
        range(len(samples)),
        key=lambda index: float(samples[index].get("visual_change") or 0.0),
        reverse=True,
    )
    selected = {0, len(samples) - 1, len(samples) // 2}
    for index in ranked_indices:
        selected.add(index)
        if len(selected) >= limit:
            break
    return [
        Path(samples[index]["frame"])
        for index in sorted(selected)
        if samples[index].get("frame")
    ]


def _usage_record(stage: str, item_id: str, usage: dict[str, int]) -> dict[str, Any]:
    return {"stage": stage, "item_id": item_id, **usage}


def _load_usage(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:80] or "item"


def _request_key(request: EvidenceRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _review_cache_key(request: EvidenceRequest, candidates: list[dict[str, Any]],
                      identity: dict[str, Any]) -> str:
    return _canonical_hash({"identity": identity, "request": request.model_dump(mode="json"),
        "candidates": candidates, "system_prompt": BOUNDARY_SYSTEM_PROMPT,
        "output_schema": _strict_json_schema(BoundaryReviewResponse), "temperature": 0.1})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _validate_discovery_limits(
    response: DiscoveryBatchResponse,
    limits: PlannerRunLimits,
) -> None:
    if len({candidate.candidate_id for candidate in response.candidates}) != len(response.candidates):
        raise ValueError("Duplicate candidate_id inside discovery response")
    if len(response.candidates) > limits.max_candidates:
        raise RuntimeError("Discovery excedió max_candidates")
    request_count = sum(len(candidate.evidence_requests) for candidate in response.candidates)
    if request_count > limits.max_initial_evidence_requests:
        raise RuntimeError("Discovery excedió max_initial_evidence_requests")


def _reported_tokens(records: list[dict[str, Any]]) -> int:
    return sum(int(record.get("total_tokens") or 0) for record in records)


def _check_run_token_budget(
    root: Path, usage_records: list[dict[str, Any]], limits: PlannerRunLimits, *,
    stage: str, reviews: list[dict[str, Any]] | None = None,
    unresolved: list[dict[str, Any]] | None = None,
) -> None:
    reported = _reported_tokens(usage_records)
    if reported < limits.max_total_reported_tokens:
        return
    summary = _run_summary(usage_records, reviews or [], unresolved or [])
    summary.update(status="budget_exhausted", stage=stage,
        max_total_reported_tokens=limits.max_total_reported_tokens)
    _write_json(root / "run_summary.json", summary)
    raise PlannerBudgetExceeded(stage, reported, limits.max_total_reported_tokens)


def _composition_cache_matches(plan_path: Path, request_sha256: str) -> bool:
    receipt_path = plan_path.parent / "composition_receipt.json"
    if not receipt_path.exists():
        return False
    receipt = _bounded_json(receipt_path)
    if (not isinstance(receipt, dict)
            or set(receipt) != {"version", "kind", "request_sha256", "plan_sha256"}
            or type(receipt["version"]) is not int or receipt["version"] != 1
            or receipt["kind"] != "editorial_composition_cache"
            or not isinstance(receipt["request_sha256"], str)
            or not isinstance(receipt["plan_sha256"], str)):
        raise ValueError("Invalid composition cache receipt")
    if not plan_path.is_file():
        raise ValueError("Missing composition cache plan")
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != receipt["plan_sha256"]:
        raise ValueError("Invalid composition cache plan hash")
    _persisted_composition(plan_path)
    return bool(receipt["request_sha256"] == request_sha256)


def _archive_composition(root: Path) -> None:
    existing = [root / name for name in ("composition_plan.json", "composition_receipt.json",
        "run_summary.json", "usage.json") if (root / name).is_file()]
    if not existing:
        return
    history = root / "composition_history" / uuid4().hex
    history.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, history / path.name)


def _run_summary(
    usage_records: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_calls": len(usage_records),
        "reported_tokens": _reported_tokens(usage_records),
        "reviews": len(reviews),
        "unresolved_due_to_limits": len(unresolved),
    }


def _coverage_semantic_limits(limits: PlannerRunLimits) -> dict[str, Any]:
    """Operating ceilings do not alter the editorial request or its identity."""
    return {key: value for key, value in asdict(limits).items()
        if key not in {"max_total_reported_tokens", "max_boundary_calls"}}


def _validate_coverage_budget_increase(root: Path, limits: PlannerRunLimits) -> None:
    config_path = root / "run_config.json"
    if not config_path.exists():
        return
    config = _bounded_json(config_path)
    previous = config.get("limits") if isinstance(config, dict) else None
    if not isinstance(previous, dict):
        raise ValueError("Invalid coverage run budget configuration")
    usage = _load_usage(root / "usage.json")
    spent = {"max_total_reported_tokens": _reported_tokens(usage),
        "max_boundary_calls": sum(record.get("stage") == "boundary" for record in usage)}
    for key in ("max_total_reported_tokens", "max_boundary_calls"):
        old = previous.get(key)
        new = getattr(limits, key)
        if type(old) is not int or old < 1:
            raise ValueError("Invalid previous coverage budget limit")
        if new < old or (new > old and new < spent[key]):
            raise ValueError(f"Coverage budget limit {key} cannot fall below prior limit or usage")


def _planning_run_id(
    transcript_path: Path,
    *,
    model_name: str,
    context_window_tokens: int,
    max_output_tokens: int | None,
    limits: PlannerRunLimits,
) -> str:
    transcript_hash = hashlib.sha256(transcript_path.read_bytes()).hexdigest()[:16]
    payload = json.dumps(
        {
            "transcript_hash": transcript_hash,
            "model": model_name,
            "context_window_tokens": context_window_tokens,
            "max_output_tokens": max_output_tokens,
            "limits": (_coverage_semantic_limits(limits) if limits.target_clips is None
                else asdict(limits)),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    run_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{_safe_name(transcript_path.stem)}-{run_hash}"
