"""Resume validated visual manifests at the external render boundary.

Export success means retained bytes, not audiovisual review or user approval.
Director READY validation remains the authority for admitting a visual plan.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .jobs import JobStore
from .local_paths import atomic_write_text, source_fingerprint
from .plan_render import render_visual_composition_plan
from .planner_engine import _persisted_composition, _versioned_json
from .render_assets import RenderAssetRegistry, default_render_assets
from .render_contracts import RenderProfile
from .style_library import StyleLibrary
from .visual_contracts import SourceVisualTimeline, VisualCompositionPlan
from .visual_director import _fingerprints, _hash, _load_terminal
from .visual_provenance import (
    _unique_fields, emitted_text_size, file_sha256, visual_timeline_fingerprint,
)

_JSON_CAP = 16 * 1024 * 1024
_STATES = {"pending", "review_required", "rendered_unreviewed", "render_failed", "error"}


@dataclass(frozen=True, slots=True)
class ExportRunArtifacts:
    manifest: Path
    summary: Path
    clips: tuple[Path, ...]


class _InputChanged(ValueError):
    pass


def _read(path: Path) -> Any:
    with path.open("rb") as stream:
        content = stream.read(_JSON_CAP + 1)
    if len(content) > _JSON_CAP:
        raise ValueError("Export JSON exceeds 16 MiB")
    value = json.loads(content, object_pairs_hook=_unique_fields)
    _hash(value)  # Reject NaN and overflowing/nonfinite JSON numbers.
    return value


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("Export input/output must be a regular file")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _reference(record: dict[str, Any], key: str) -> Path:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or malformed export reference: {key}")
    return Path(value).resolve(strict=True)


def _contained(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Generated export path escapes its root")
    return resolved


def _terminal(root: Path, *, ready: bool) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    # The shared validator has a larger legacy quota. Prebound its local JSON
    # dependencies without replacing any of its receipt/authority checks.
    paths = root.rglob("*.json") if ready else (
        root / "receipt.json", root / "provenance.json", root / "authority.json")
    for path in paths:
        if path.is_file():
            if path.stat().st_size > _JSON_CAP:
                raise ValueError("Director JSON exceeds export's 16 MiB limit")
    terminal = _load_terminal(root)
    if terminal is None:
        raise ValueError("Missing committed visual director terminal")
    return terminal


def _admit(source: Path, manifest_path: Path, transcript: Path | None,
           assets: RenderAssetRegistry, profile: RenderProfile,
           ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], StyleLibrary]:
    manifest = _read(manifest_path)
    if (not isinstance(manifest, dict)
            or any(type(manifest.get(key)) is not int or manifest[key] != version for key, version in (
                ("manifest_version", 1), ("editorial_contract_version", 1), ("visual_contract_version", 2)))):
        raise ValueError("Export requires the versioned visual planner manifest")
    editorial_path = _reference(manifest, "editorial_plan")
    timeline_path = _reference(manifest, "source_timeline")
    composition = _persisted_composition(editorial_path)
    timeline = SourceVisualTimeline.model_validate(_versioned_json(timeline_path, 1))
    source_fp = source_fingerprint(source)
    timeline_fp = visual_timeline_fingerprint(timeline)
    timeline_record = _record(timeline_path)
    if (manifest.get("source_fingerprint") != source_fp or timeline.source_fingerprint != source_fp
            or manifest.get("composition_fingerprint") != _hash(composition.model_dump(mode="json"))
            or manifest.get("source_timeline_fingerprint") != timeline_fp
            or manifest.get("source_timeline_sha256") != timeline_record["sha256"]):
        raise ValueError("Manifest source/editorial/timeline association mismatch")
    entries = manifest.get("clips")
    clip_ids = [clip.clip_id for clip in composition.clips]
    if (not isinstance(entries, list) or len(set(clip_ids)) != len(clip_ids)
            or any(not isinstance(entry, dict) for entry in entries)
            or [entry.get("clip_id") for entry in entries] != clip_ids):
        raise ValueError("Manifest must associate each editorial clip exactly once in order")
    library = StyleLibrary(_reference(manifest, "style_library_root"))
    style_fp = manifest.get("style_library_fingerprint")
    if (not isinstance(style_fp, str) or not re.fullmatch(r"[0-9a-f]{64}", style_fp)
            or any(e.get("status") == "ready" for e in entries) and library.fingerprint != style_fp):
        raise ValueError("Manifest style library association mismatch")
    files = [_record(source), _record(manifest_path), _record(editorial_path), timeline_record]
    text_record = None
    if transcript is not None:
        _read(transcript)
        text_record = _record(transcript)
        files.append(text_record)
    fixed_text = "transcript_path" in manifest or "transcript_sha256" in manifest
    if fixed_text and (text_record is None
            or manifest.get("transcript_path") != text_record["path"]
            or manifest.get("transcript_sha256") != text_record["sha256"]):
        raise ValueError("Manifest selected transcript association mismatch")
    asset_records = {name: {key: _record(path) for key, path in sorted(table.items())}
                     for name, table in (("brand_images", assets.brand_images),
                                         ("fonts", assets.fonts), ("sfx", assets.sfx))}
    files.extend(record for table in asset_records.values() for record in table.values())
    admitted = []
    for index, (clip, entry) in enumerate(zip(composition.clips, entries)):
        status = entry.get("status")
        if status not in {"ready", "review_required", "pending", "error"}:
            raise ValueError("Unknown visual manifest clip status")
        row: dict[str, Any] = {"clip_id": clip.clip_id,
            "host_id": f"clip-{index:04d}-{_hash(clip.clip_id)[:12]}",
            "status": status if status in {"review_required", "error"} else "pending",
            "reason": entry.get("reason", "visual planning incomplete"), "input": dict(entry)}
        if status in {"ready", "review_required"}:
            receipt_path = _reference(entry, "receipt")
            root = receipt_path.parent
            if receipt_path != root / "receipt.json":
                raise ValueError("Manifest receipt must name the committed terminal")
            output, receipt, companion = _terminal(root, ready=status == "ready")
            host = companion.get("host_context")
            upstream = host.get("upstream", {}) if isinstance(host, dict) else {}
            if (receipt.get("status") != status or receipt.get("clip_id") != clip.clip_id
                    or receipt.get("source") != str(source)
                    or receipt.get("source_fingerprint") != source_fp
                    or entry.get("provenance") != str(root / "provenance.json")
                    or entry.get("provenance_sha256") != receipt.get("companion_sha256")
                    or entry.get("host_context") != host
                    or upstream.get("path") != str(timeline_path)
                    or upstream.get("sha256") != timeline_record["sha256"]
                    or upstream.get("context", {}).get("timeline_fingerprint") != timeline_fp
                    or upstream.get("context", {}).get("source_fingerprint") != source_fp):
                raise ValueError("Manifest terminal/provenance/upstream association mismatch")
            terminal_files = [_record(receipt_path), _record(root / "provenance.json")]
            if status == "ready":
                plan = _reference(entry, "visual_plan")
                model = VisualCompositionPlan.model_validate(_versioned_json(plan, 2))
                if (output != plan or plan != root / "visual_plan.json"
                        or model.clip_id != clip.clip_id or any(getattr(model, key) != value
                            for key, value in _fingerprints(clip, timeline, style_fp).items())):
                    raise ValueError("Manifest visual plan association/fingerprint mismatch")
                policy = host.get("creator_overlays")
                needs_text = (fixed_text or any(shot.subtitles is not None for shot in model.shots)
                              or isinstance(policy, dict) and policy.get("required") is True)
                if needs_text:
                    captured = host.get("text")
                    if (text_record is None or not isinstance(captured, dict)
                            or any(captured.get(key) != text_record[key]
                                   for key in ("path", "bytes", "sha256"))):
                        raise ValueError("READY transcript association mismatch")
                cues = {cue.asset_id for shot in model.shots for cue in shot.audio.cues}
                authority_sfx = host.get("sfx")
                approved = ({item["asset_id"]: item for item in authority_sfx["assets"]}
                            if isinstance(authority_sfx, dict) else {})
                for asset_id in cues:
                    selected = asset_records["sfx"].get(asset_id)
                    expected = approved.get(asset_id)
                    if (selected is None or expected is None
                            or any(selected[key] != expected[key] for key in ("bytes", "sha256"))):
                        raise ValueError("READY SFX association mismatch")
                terminal_files.append(_record(plan))
            elif entry.get("visual_plan") is not None:
                raise ValueError("Review entries cannot contain an exportable visual plan")
            else:
                row["reason"] = receipt["reason"]
            row["input"]["terminal_files"] = terminal_files
            files.extend(terminal_files)
        admitted.append(row)
    identity = {"source_fingerprint": source_fp, "files": files, "assets": asset_records,
                "profile": profile.model_dump(mode="json"), "entries": admitted,
                "style_library_fingerprint": style_fp}
    return identity, admitted, files, library


def _unchanged(files: list[dict[str, Any]], source: Path, identity: dict[str, Any],
               library: StyleLibrary, *, ready: bool) -> None:
    if (source_fingerprint(source) != identity["source_fingerprint"]
            or any(_record(Path(record["path"])) != record for record in files)
            or ready and library.fingerprint != identity["style_library_fingerprint"]):
        raise _InputChanged("Export inputs changed; a fresh admitted run is required")


def _status(entries: list[dict[str, Any]]) -> str:
    statuses = [entry["status"] for entry in entries]
    if not statuses:
        return "empty"
    if all(status == "rendered_unreviewed" for status in statuses):
        return "rendered_unreviewed"
    if "rendered_unreviewed" in statuses:
        return "partial"
    return "error" if any(status in {"render_failed", "error"} for status in statuses) else "review_required"


def _reusable(entry: dict[str, Any], root: Path) -> bool:
    try:
        attempt = entry["attempts"][-1]
        attempt_id = attempt["attempt_id"]
        expected = _contained(root / entry["host_id"] / attempt_id, root)
        return bool(entry["status"] == "rendered_unreviewed" and entry["outputs"]
            and re.fullmatch(r"[0-9a-f]{32}", attempt_id)
            and attempt["path"] == str(expected)
            and attempt["status"] == "rendered_unreviewed"
            and attempt["outputs"] == entry["outputs"]
            and all(_contained(Path(record["path"]), expected) and _record(Path(record["path"])) == record
                    for record in entry["outputs"]))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def _diagnostic(error: BaseException, attempt: Path) -> str | None:
    candidates = [getattr(error, "diagnostic_path", None)]
    candidates.extend(getattr(getattr(error, "host", None), "diagnostic_paths", ()))
    for candidate in reversed(candidates):
        try:
            if candidate is not None:
                path = _contained(Path(candidate), attempt)
                if path.is_file():
                    return str(path)
        except (OSError, ValueError, TypeError):
            continue
    return None


def export_visual_manifest(
    source_video: Path, manifest_path: Path, output_dir: Path, *,
    transcript_path: Path | None = None, assets: RenderAssetRegistry | None = None,
    profile: RenderProfile | None = None, render: Callable[..., list[Path]] | None = None,
) -> ExportRunArtifacts:
    """Export each admitted READY clip, retaining and retrying independent attempts.

    Malformed input associations raise before rendering. Ordinary render failures
    are recorded and other clips continue; cancellation persists and propagates.
    """
    source = Path(source_video).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    transcript = None if transcript_path is None else Path(transcript_path).resolve(strict=True)
    assets = assets if assets is not None else default_render_assets()
    profile = RenderProfile.model_validate((profile or RenderProfile()).model_dump(mode="json"))
    identity, admitted, files, library = _admit(source, manifest_path, transcript, assets, profile)
    run_id = _hash(identity)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = _contained(output / "export_runs" / run_id, output)
    root.mkdir(parents=True, exist_ok=True)
    job = JobStore(source, root, identity["source_fingerprint"])
    renderer = render if render is not None else render_visual_composition_plan
    state_path = _contained(root / "export_manifest.json", root)
    summary_path = _contained(root / "export_summary.json", root)
    ready = any(e["input"]["status"] == "ready" for e in admitted)

    with job.stage_lock("export"):
        _unchanged(files, source, identity, library, ready=ready)
        if state_path.exists():
            state = _read(state_path)
            if (state.get("version") != 1 or state.get("inputs") != identity
                    or not isinstance(state.get("clips"), list) or len(state["clips"]) != len(admitted)):
                raise ValueError("Saved export state does not match its run identity")
            for entry, expected in zip(state["clips"], admitted):
                if (any(entry.get(key) != expected[key] for key in ("clip_id", "host_id", "input"))
                        or entry.get("status") not in _STATES or not isinstance(entry.get("attempts"), list)):
                    raise ValueError("Saved export clip association is malformed")
        else:
            state = {"version": 1, "kind": "visual_export_manifest", "run_id": run_id,
                     "source_manifest": str(manifest_path), "inputs": identity,
                     "clips": [dict(entry, attempts=[], outputs=[]) for entry in admitted]}

        def persist() -> None:
            state["status"] = _status(state["clips"])
            summary = {"status": state["status"], "manifest": str(state_path),
                       "total_clips": len(admitted), **{f"{status}_clips": sum(
                           e["status"] == status for e in state["clips"]) for status in sorted(_STATES)}}
            for path, payload in ((state_path, state), (summary_path, summary)):
                encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                if emitted_text_size(encoded) > _JSON_CAP:
                    raise ValueError("Export state exceeds 16 MiB")
                atomic_write_text(_contained(path, root), encoded)

        for entry in state["clips"]:
            if entry["input"]["status"] == "ready" and not _reusable(entry, root):
                entry.update(status="pending", outputs=[])
        persist()
        for entry in state["clips"]:
            if entry["input"]["status"] != "ready" or _reusable(entry, root):
                continue
            _unchanged(files, source, identity, library, ready=ready)
            plan = Path(entry["input"]["visual_plan"])
            _terminal(plan.parent, ready=True)
            attempt_id = uuid4().hex
            attempt_dir = _contained(root / entry["host_id"] / attempt_id, root)
            attempt_dir.mkdir(parents=True, exist_ok=False)
            attempt: dict[str, Any] = {"attempt_id": attempt_id, "path": str(attempt_dir),
                                       "status": "pending", "outputs": []}
            entry["attempts"].append(attempt)
            persist()
            try:
                paths = renderer(source, plan, attempt_dir, transcript_path=transcript,
                                 assets=assets, profile=profile)
                if not isinstance(paths, list) or not paths:
                    raise ValueError("Renderer must retain at least one output")
                outputs = [_record(_contained(Path(path), attempt_dir)) for path in paths]
                if (len({record["path"] for record in outputs}) != len(outputs)
                        or any(record["bytes"] == 0 for record in outputs)):
                    raise ValueError("Renderer retained empty or duplicate outputs")
                _unchanged(files, source, identity, library, ready=ready)
                _terminal(plan.parent, ready=True)
                attempt.update(status="rendered_unreviewed", outputs=outputs)
                entry.update(status="rendered_unreviewed", outputs=outputs)
            except BaseException as error:
                name = type(error).__name__
                attempt.update(status="render_failed", error_type=name if re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]{0,79}", name) else "RenderError",
                    diagnostic_path=_diagnostic(error, attempt_dir))
                entry.update(status="render_failed", outputs=[])
                persist()
                if not isinstance(error, Exception) or isinstance(error, _InputChanged):
                    raise
            persist()
        _unchanged(files, source, identity, library, ready=ready)
        for entry in state["clips"]:
            if entry["status"] == "rendered_unreviewed" and not _reusable(entry, root):
                entry.update(status="pending", outputs=[], reason="Retained output missing or changed")
        persist()
        return ExportRunArtifacts(state_path, summary_path, tuple(Path(record["path"])
            for entry in state["clips"] if entry["status"] == "rendered_unreviewed"
            for record in entry["outputs"]))
