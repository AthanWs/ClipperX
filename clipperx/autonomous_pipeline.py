"""Resumable preparation → editorial → visual → export, not automatic approval."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING, TypeVar

from .jobs import JobStore
from .local_paths import atomic_write_text, source_fingerprint
from .planner_contracts import GlobalCompositionResponse
from .planner_engine import PlannerBudgetExceeded, PlannerRunLimits
from .render_assets import RenderAssetRegistry, default_render_assets
from .render_contracts import RenderProfile
from .sfx_authority import capture_sfx_authority
from .visual_provenance import file_sha256

if TYPE_CHECKING:
    from .export_pipeline import ExportRunArtifacts
    from .planner_engine import PlannerRunArtifacts, StructuredClient
    from .style_library import StyleLibrary
    from .visual_director import VisualDirectorLimits

T = TypeVar("T")
_JSON_CAP = 16 * 1024**2


@dataclass(frozen=True, slots=True)
class AutonomousRunArtifacts:
    state_path: Path
    summary_path: Path
    composition_plan: Path | None
    visual_manifest: Path | None
    export_manifest: Path | None
    clips: tuple[Path, ...]


class AutonomousPipelineError(RuntimeError):
    """External stage failure with no provider message or credentials attached."""


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read(_JSON_CAP + 1)
    if len(raw) > _JSON_CAP:
        raise ValueError("Autonomous workflow JSON exceeds 16 MiB")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Autonomous workflow record must be an object")
    return value


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def _record(path: Path) -> dict[str, str]:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def _implementation_signature() -> str:
    # No import/config inspection, and no hashing of weights, media or caches.
    package = Path(__file__).resolve().parent
    return _hash([(p.name, file_sha256(p)) for p in sorted(package.glob("*.py"))])


def _ensure_transcript(source: Path) -> Path:
    from .stages import ensure_transcript
    return ensure_transcript(source)[1]


def _ensure_timeline(source: Path) -> Path:
    from .visual_timeline import build_source_visual_timeline
    return build_source_visual_timeline(source)


def _run_editorial(*args: Any, **kwargs: Any) -> Path:
    from .planner_engine import run_editorial_planner
    return run_editorial_planner(*args, **kwargs)


def _run_visual(*args: Any, **kwargs: Any) -> PlannerRunArtifacts:
    from .planner_engine import run_visual_planner
    return run_visual_planner(*args, **kwargs)


def _export(*args: Any, **kwargs: Any) -> ExportRunArtifacts:
    from .export_pipeline import export_visual_manifest
    return export_visual_manifest(*args, **kwargs)


def _invoke(stage: str, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    try:
        return function(*args, **kwargs)
    except PlannerBudgetExceeded:
        raise
    except Exception as error:
        raise AutonomousPipelineError(f"{stage} failed ({type(error).__name__})") from None


def run_autonomous_pipeline(
    source_video: Path, output_dir: Path, *, client: StructuredClient, model_name: str,
    style_library: StyleLibrary, context_window_tokens: int = 65536,
    max_output_tokens: int = 4096, transcript_path: Path | None = None,
    source_timeline_path: Path | None = None, editorial_limits: PlannerRunLimits | None = None,
    visual_limits: VisualDirectorLimits | None = None, assets: RenderAssetRegistry | None = None,
    profile: RenderProfile | None = None,
) -> AutonomousRunArtifacts:
    """Explicitly admitted caller supplies a client; passive imports never create one.

    Existing ASR/index stages own their model-aware caches. The workflow reuses
    unchanged planning references, but delegates export validation on every resume.
    Review receipts are not repurchased merely by invoking the operation again.
    """
    from .visual_director import VisualDirectorLimits
    from .visual_contracts import SourceVisualTimeline
    from .progress import reporter

    if not model_name.strip() or len(model_name) > 256 or context_window_tokens < 4096:
        raise ValueError("Explicit model and valid operational context required")
    if type(max_output_tokens) is not int or not 0 < max_output_tokens < context_window_tokens:
        raise ValueError("Invalid output-token reservation")
    source = Path(source_video).resolve(strict=True)
    source_fp = source_fingerprint(source)
    limits = editorial_limits or PlannerRunLimits(target_clips=None)
    limits.validate()
    visual = visual_limits or VisualDirectorLimits(require_creator_overlays=True)
    registry = assets if assets is not None else default_render_assets()
    render_profile = profile or RenderProfile()
    output = Path(output_dir).resolve()
    semantic_limits = asdict(limits)
    operational_limits = {key: semantic_limits[key] for key in
                          ("max_total_reported_tokens", "max_boundary_calls")}
    if limits.target_clips is None:
        for key in operational_limits:
            semantic_limits.pop(key)
    config = {"source": str(source), "source_fingerprint": source_fp,
        "model": model_name, "context_window_tokens": context_window_tokens,
        "max_output_tokens": max_output_tokens, "editorial_limits": semantic_limits,
        "visual_limits": asdict(visual), "style_fingerprint": style_library.fingerprint,
        "profile": render_profile.model_dump(mode="json"), "output_dir": str(output),
        "assets": {name: {key: _record(path) for key, path in table.items()}
            for name, table in (("brand", registry.brand_images), ("font", registry.fonts), ("sfx", registry.sfx))},
        "sfx_authority": capture_sfx_authority(), "implementation": _implementation_signature(),
        "provided_transcript": None if transcript_path is None else _record(transcript_path),
        "provided_timeline": None if source_timeline_path is None else _record(source_timeline_path)}
    run_id = _hash(config)[:24]
    job = JobStore.for_video(source)
    root = job.root / "autonomous" / run_id
    root.mkdir(parents=True, exist_ok=True)
    state_path, summary_path = root / "state.json", root / "autonomous-summary.json"
    with job.stage_lock(f"autonomous-{run_id}"):
        state: dict[str, Any] = _read(state_path) if state_path.exists() else {"version": 1, "config": config, "stages": {}}
        if state.get("version") != 1 or state.get("config") != config or not isinstance(state.get("stages"), dict):
            raise ValueError("Autonomous run state/input identity mismatch")
        previous_limits = state.get("operational_limits")
        if previous_limits is not None and limits.target_clips is None:
            if (not isinstance(previous_limits, dict) or set(previous_limits) != set(operational_limits)
                    or any(type(previous_limits[key]) is not int or operational_limits[key] < previous_limits[key]
                           for key in operational_limits)):
                raise ValueError("Coverage budget limits cannot decrease when resuming")
        increased_budget = bool(previous_limits and any(
            operational_limits[key] > previous_limits[key] for key in operational_limits))
        state["operational_limits"] = operational_limits
        stages: dict[str, Any] = state["stages"]
        clips: tuple[Path, ...] = ()
        phase = "preparation"

        def artifacts() -> AutonomousRunArtifacts:
            def path(name: str) -> Path | None:
                record = stages.get(name)
                return Path(record["artifact"]["path"]) if record else None
            return AutonomousRunArtifacts(state_path, summary_path, path("editorial"),
                path("visual"), path("export"), clips)

        def persist(status: str, *, error_type: str | None = None, export_summary: dict[str, Any] | None = None) -> None:
            state.update(status=status, stage=phase)
            atomic_write_text(state_path, json.dumps(state, ensure_ascii=False, indent=2))
            summary = {"version": 1, "status": status, "stage": phase, "source": str(source),
                "state": str(state_path), "clips": [str(p) for p in clips], "rendered_clips": len(clips),
                "full_qa_passed": False, "user_approved": False, "error_type": error_type,
                "export_summary": export_summary, "editorial_coverage": state.get("editorial_coverage"),
                "budget": state.get("budget")}
            atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2))

        def unchanged() -> None:
            if source_fingerprint(source) != source_fp or style_library.fingerprint != config["style_fingerprint"]:
                raise ValueError("Source or style library changed during automatic workflow")

        def cached(name: str, inputs: dict[str, Any]) -> Path | None:
            previous = stages.get(name)
            if previous is None:
                return None
            old = previous["artifact"]
            if _record(Path(old["path"])) != old:
                raise ValueError("Cached workflow artifact changed or is missing")
            return Path(old["path"]) if previous["inputs"] == inputs else None

        def save(name: str, path: Path, inputs: dict[str, Any]) -> None:
            if name in stages:
                state.setdefault("history", []).append({"stage": name, **stages[name]})
            stages[name] = {"artifact": _record(path), "inputs": inputs}
            persist("running")

        state.pop("budget", None)
        persist("running")
        try:
            reporter.set_percent("autonomous", 2, "Preparando transcripción e índice visual")
            text = Path(transcript_path) if transcript_path is not None else _invoke("transcription", _ensure_transcript, source)
            timeline = Path(source_timeline_path) if source_timeline_path is not None else _invoke("visual_index", _ensure_timeline, source)
            text, timeline = text.resolve(strict=True), timeline.resolve(strict=True)
            text_record, timeline_record = _record(text), _record(timeline)
            timeline_payload = _read(timeline)
            if type(timeline_payload.get("contract_version")) is not int or timeline_payload["contract_version"] != 1:
                raise ValueError("Visual timeline requires explicit contract_version 1")
            if SourceVisualTimeline.model_validate(timeline_payload).source_fingerprint != source_fp:
                raise ValueError("Visual timeline does not belong to the selected source")
            unchanged()
            phase = "editorial"
            reporter.set_percent("autonomous", 35, "Seleccionando todos los momentos justificados dentro del presupuesto")
            editorial_inputs = {"transcript": text_record}
            composition = cached("editorial", editorial_inputs)
            previous_coverage = state.get("editorial_coverage") or {}
            if increased_budget and any(previous_coverage.get(key) for key in
                                        ("pending_groups", "unresolved_due_to_limits")):
                composition = None
            if composition is None:
                composition = _invoke(phase, _run_editorial, source, text, discovery_client=client,
                    evidence_client=client, model_name=model_name, context_window_tokens=context_window_tokens,
                    max_output_tokens=max_output_tokens, limits=limits)
                unchanged()
                payload = _read(composition)
                if payload.get("contract_version") != 1:
                    raise ValueError("Explicit editorial v1 required")
                GlobalCompositionResponse.model_validate(payload)
                save("editorial", composition, editorial_inputs)
            coverage = composition.parent / "run_summary.json"
            state["editorial_coverage"] = _read(coverage) if coverage.is_file() else None
            phase = "visual"
            reporter.set_percent("autonomous", 60, "Editando geometría, marca y subtítulos con el director")
            if _record(text) != text_record or _record(timeline) != timeline_record:
                raise ValueError("Analysis input changed during automatic workflow")
            visual_inputs = {"composition": _record(composition), "transcript": text_record, "timeline": timeline_record}
            manifest = cached("visual", visual_inputs)
            if manifest is None:
                planned = _invoke(phase, _run_visual, source, composition,
                    source_timeline_path=timeline, transcript_path=text, client=client,
                    style_library=style_library, limits=visual)
                unchanged()
                manifest = planned.visual_plan_manifest
                save("visual", manifest, visual_inputs)
            phase = "export"
            reporter.set_percent("autonomous", 85, "Exportando y verificando referencias de cada clip")
            unchanged()
            exported = _invoke(phase, _export, source, manifest, output,
                transcript_path=text, assets=registry, profile=render_profile)
            clips = exported.clips
            save("export", exported.manifest, {"visual_manifest": _record(manifest)})
            export_summary = _read(exported.summary)
            status = export_summary.get("status")
            if status not in {"rendered_unreviewed", "partial", "review_required", "empty", "error"}:
                raise ValueError("Unknown export terminal status")
            coverage_status = state.get("editorial_coverage") or {}
            if any(coverage_status.get(key) for key in ("deferred_groups", "pending_groups", "unresolved_due_to_limits")):
                status = "partial"
            unchanged()
            persist(status, export_summary=export_summary)
            percent = 100 if status in {"rendered_unreviewed", "empty"} else 90
            reporter.set_percent("autonomous", percent, f"{len(clips)} clips exportados; estado {status}, revisión audiovisual pendiente")
        except PlannerBudgetExceeded as error:
            state["budget"] = {"stage": error.stage, "reported_tokens": error.reported_tokens, "limit": error.limit}
            persist("budget_exhausted", error_type=type(error).__name__)
        except Exception as error:
            persist("error", error_type=type(error).__name__)
            raise
        except BaseException:
            persist("cancelled")
            raise
        return artifacts()
