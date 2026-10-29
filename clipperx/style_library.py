"""Provider-neutral visual analysis with immutable, integrity-checked publication.

One library lock serializes cache lookup/build/publication and readers. Complete
content-addressed artifacts precede the single atomic manifest replacement, so a
failed replacement leaves the previous generation readable. Source MP4s are never
copied; only host-owned provenance sidecars reference the historical edited clip.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal, Protocol

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from clipperx.jobs import _InterProcessFileLock
from clipperx.local_paths import atomic_write_text, source_fingerprint
from clipperx.structured_client import StructuredResult
from clipperx.style_analysis import StyleEvidence
from clipperx.style_contracts import (
    StyleCard,
    StyleCardAnalysisResponse,
    StyleKeyframe,
    _validate_library_ref,
)
from clipperx.style_prompts import STYLE_CARD_SYSTEM_PROMPT, build_style_card_prompt

Approval = Literal["approved", "unreviewed", "rejected"]
_HASH = re.compile(r"^[0-9a-f]{64}$")


class StyleAnalysisClient(Protocol):
    def generate(
        self,
        output_model: type[StyleCardAnalysisResponse],
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[Path],
    ) -> StructuredResult[StyleCardAnalysisResponse]: ...


@dataclass(frozen=True)
class StyleCardBuildResult:
    card: StyleCard
    usage: dict[str, int]
    cache_hit: bool
    evidence_hash: str
    analysis_key: str


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _CardRecord(_Record):
    source: _Record
    frames: list[_Record] = Field(min_length=3, max_length=5)
    analysis_key: str


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    cards: dict[str, _CardRecord] = Field(default_factory=dict)
    analyses: dict[str, _Record] = Field(default_factory=dict)


class _AnalysisCache(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    evidence_hash: str
    analysis: StyleCardAnalysisResponse
    usage: dict[str, int]


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _source_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, ref: str, *, exists: bool = True) -> Path:
    _validate_library_ref(ref)
    path = root / ref
    resolved = path.resolve(strict=exists)
    if not resolved.is_relative_to(root):
        raise ValueError("Library ref escapes resolved containment")
    if exists and not resolved.is_file():
        raise ValueError("Library ref must name an existing file")
    return path


def _read_record(root: Path, record: _Record) -> bytes:
    blob = _resolve(root, record.path).read_bytes()
    if _sha(blob) != record.sha256:
        raise ValueError("Library artifact integrity mismatch")
    return blob


def _decode(blob: bytes, suffix: str) -> None:
    if suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Keyframes must be JPEG or PNG images")
    try:
        with Image.open(BytesIO(blob)) as image:
            expected = "PNG" if suffix.lower() == ".png" else "JPEG"
            if image.format != expected or image.width * image.height > 16_000_000:
                raise ValueError("Invalid or oversized keyframe image")
            image.verify()
        with Image.open(BytesIO(blob)) as image:
            image.load()
    except (OSError, SyntaxError) as exc:
        raise ValueError("Keyframe is not a complete decoded image") from exc


def _immutable_text(root: Path, ref: str, text: str) -> _Record:
    blob = text.encode("utf-8")
    target = _resolve(root, ref, exists=False)
    if target.exists():
        if target.read_bytes() != blob:
            raise ValueError("Immutable artifact integrity mismatch")
    else:
        atomic_write_text(target, text)
    return _Record(path=ref, sha256=_sha(blob))


def _snapshot_frame(root: Path, blob: bytes, suffix: str) -> _Record:
    ref = f"frames/{_sha(blob)}{suffix.lower()}"
    target = _resolve(root, ref, exists=False)
    if target.exists():
        if target.read_bytes() != blob:
            raise ValueError("Immutable frame integrity mismatch")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return _Record(path=ref, sha256=_sha(blob))


def _select_frames(root: Path, evidence: StyleEvidence) -> tuple[list[StyleKeyframe], list[bytes]]:
    validated: list[tuple[StyleKeyframe, bytes, str]] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    # Validate every supplied saved frame, even those outside the selected five.
    for frame in evidence.keyframes:
        path = _resolve(root, frame.path)
        blob = path.read_bytes()
        digest = _sha(blob)
        expected = evidence.keyframe_sha256.get(frame.path, "")
        if not _HASH.fullmatch(expected) or digest != expected:
            raise ValueError("Keyframe SHA-256 integrity mismatch")
        _decode(blob, path.suffix)
        resolved = str(path.resolve())
        if resolved not in seen_paths and digest not in seen_hashes:
            validated.append((frame, blob, digest))
            seen_paths.add(resolved)
            seen_hashes.add(digest)
    if len(validated) < 3:
        raise ValueError("At least three distinct saved keyframes are required")
    count = min(5, len(validated))
    indices = [round(i * (len(validated) - 1) / (count - 1)) for i in range(count)]
    return [validated[i][0] for i in indices], [validated[i][1] for i in indices]


def _validate_analysis(value: StyleCardAnalysisResponse, count: int) -> StyleCardAnalysisResponse:
    result = StyleCardAnalysisResponse.model_validate(value.model_dump())
    if len(result.model_dump_json()) > 64_000:
        raise ValueError("Visual analysis exceeds bounded response size")
    allowed = {f"frame_{i:03d}" for i in range(count)}
    if len(set(result.evidence_refs)) != len(result.evidence_refs):
        raise ValueError("Duplicate model evidence_ref")
    if not set(result.evidence_refs) <= allowed:
        raise ValueError("Unknown model evidence_ref")
    if any(title == brand for title in result.title_regions for brand in result.brand_regions):
        raise ValueError("Identical component rectangle in title_regions and brand_regions")
    return result


def _usage(values: Mapping[str, int]) -> dict[str, int]:
    allowed = {"prompt_tokens", "completion_tokens", "total_tokens", "estimated_input_tokens"}
    return {k: v for k, v in values.items() if k in allowed and type(v) is int and v >= 0}


class StyleLibrary:
    """Explicit-root durable collection; methods fail closed on corrupt publication."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def _lock(self) -> _InterProcessFileLock:
        return _InterProcessFileLock(_resolve(self.root, ".library.lock", exists=False))

    def _manifest(self) -> _Manifest:
        path = _resolve(self.root, "library_manifest.json", exists=False)
        if not path.exists():
            return _Manifest()
        return _Manifest.model_validate_json(path.read_bytes())

    def _load(self, card_id: str, manifest: _Manifest) -> StyleCard:
        entry = manifest.cards[card_id]
        card = StyleCard.model_validate_json(_read_record(self.root, entry))
        if card.card_id != card_id or card.source_clip_ref != entry.source.path:
            raise ValueError("Published card identity integrity mismatch")
        source = json.loads(_read_record(self.root, entry.source))
        if source.get("clip_fingerprint") != card.clip_fingerprint:
            raise ValueError("Published source identity integrity mismatch")
        if [f.path for f in card.keyframes] != [f.path for f in entry.frames]:
            raise ValueError("Published frame manifest integrity mismatch")
        for frame in entry.frames:
            _decode(_read_record(self.root, frame), Path(frame.path).suffix)
        return card

    def load_card(self, card_id: str) -> StyleCard:
        with self._lock():
            return self._load(card_id, self._manifest())

    def list_cards(self) -> list[StyleCard]:
        with self._lock():
            manifest = self._manifest()
            return [self._load(card_id, manifest) for card_id in sorted(manifest.cards)]

    @property
    def fingerprint(self) -> str:
        """Generation identity of all published cards, after validating their integrity."""
        with self._lock():
            manifest = self._manifest()
            for card_id in manifest.cards:
                self._load(card_id, manifest)
            return _sha(_json({k: v.model_dump() for k, v in manifest.cards.items()}).encode())


def build_style_card(
    evidence: StyleEvidence,
    *,
    library_root: Path,
    source_clip: Path,
    client: StyleAnalysisClient,
    model_id: str,
    card_id: str | None = None,
    approval: Approval = "unreviewed",
    narrative_quality: float | None = None,
    created_at: datetime | None = None,
) -> StyleCardBuildResult:
    """Enrich objective evidence with 3..5 views and atomically publish one card.

    model_id must identify the actual selected model. Human labels are explicit
    arguments and never influence the visual cache key. Usage is the original
    request's usage on cache hits; cache_hit distinguishes zero new provider use.
    """
    evidence = StyleEvidence.model_validate(evidence.model_dump())
    if not model_id.strip() or len(model_id) > 256:
        raise ValueError("A bounded model_id is required")
    config = getattr(client, "config", None)
    if config is not None and getattr(config, "model", model_id) != model_id:
        raise ValueError("model_id does not match the configured client model")
    source = Path(source_clip).resolve(strict=True)
    if not source.is_file() or source_fingerprint(source) != evidence.clip_fingerprint:
        raise ValueError("Source fingerprint does not match objective evidence")
    # Validate labels before any provider request, using the reviewed card contract.
    host: dict[str, Any] = {
        "card_id": card_id or f"style-{evidence.clip_fingerprint}",
        "clip_fingerprint": evidence.clip_fingerprint,
        "source_clip_ref": f"source_refs/{evidence.clip_fingerprint}.json",
        "duration": evidence.duration_seconds,
        "created_at": created_at,
        "cut_rate_per_minute": evidence.cut_rate_per_minute,
        "zoom_count": evidence.zoom_count,
        "visual_quality": evidence.visual_quality,
        "approval": approval,
        "narrative_quality": narrative_quality,
    }
    # The source fingerprint must be safe for the fixed host-owned sidecar filename.
    if not re.fullmatch(r"[0-9a-f]{16}", evidence.clip_fingerprint):
        raise ValueError("Invalid host source fingerprint")
    library = StyleLibrary(library_root)
    root = library.root
    with library._lock():
        if source_fingerprint(source) != evidence.clip_fingerprint:
            raise ValueError("Source fingerprint changed while waiting for library lock")
        selected, blobs = _select_frames(root, evidence)
        objective = evidence.model_dump(
            mode="json",
            exclude={"narrative_quality", "keyframes", "keyframe_sha256", "evidence_path"},
        )
        objective["selected_frames"] = [
            {"timestamp": f.timestamp, "sha256": _sha(blob)} for f, blob in zip(selected, blobs)
        ]
        evidence_hash = _sha(_json(objective).encode())
        system_prompt = STYLE_CARD_SYSTEM_PROMPT
        user_prompt = build_style_card_prompt(evidence, selected)
        # Hash the actual request and response contract, not a manually bumped
        # version. Legacy keys/artifacts remain in the manifest and on disk.
        analysis_key = _sha(
            _json(
                {
                    "model_id": model_id,
                    "evidence_hash": evidence_hash,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response_contract": StyleCardAnalysisResponse.__name__,
                    "response_schema": StyleCardAnalysisResponse.model_json_schema(),
                }
            ).encode()
        )
        manifest = library._manifest()
        # Verify currently referenced generations before using them as the next base.
        for existing_id in manifest.cards:
            library._load(existing_id, manifest)
        records = [
            _snapshot_frame(root, blob, Path(f.path).suffix) for f, blob in zip(selected, blobs)
        ]
        keyframes = [
            StyleKeyframe(timestamp=f.timestamp, path=r.path) for f, r in zip(selected, records)
        ]
        placeholder = {
            "scene_family": "unknown",
            "camera_regions": [],
            "content_regions": [],
            "background_treatment": "cover",
            "title_regions": [],
            "brand_regions": [],
            "subtitle_regions": [],
            "subtitle_palette": [],
            "special_techniques": [],
            "description": "pending",
            "evidence": [],
        }
        StyleCard.model_validate({**placeholder, **host, "keyframes": keyframes})
        before_stat = source.stat()
        source_hash = _source_sha(source)
        source_data = {
            "version": 1,
            "clip_fingerprint": evidence.clip_fingerprint,
            "source_clip_path": str(source),
            "size_bytes": before_stat.st_size,
            "mtime_ns": before_stat.st_mtime_ns,
            "sha256": source_hash,
            "provenance": "historical_edited_clip",
            "source_vod_association": None,
        }
        source_text = _json(source_data)
        # Validate existing registration before cache lookup or upload.
        sidecar = _resolve(root, host["source_clip_ref"], exists=False)
        if sidecar.exists() and sidecar.read_text(encoding="utf-8") != source_text:
            raise ValueError("Source registration integrity mismatch or stale source evidence")
        cached_record = manifest.analyses.get(analysis_key)
        cache_hit = cached_record is not None
        if cached_record is not None:
            cache = _AnalysisCache.model_validate_json(_read_record(root, cached_record))
            if cache.model_id != model_id or cache.evidence_hash != evidence_hash:
                raise ValueError("Visual cache identity integrity mismatch")
            analysis = _validate_analysis(cache.analysis, len(selected))
            usage = _usage(cache.usage)
        else:
            response = client.generate(
                StyleCardAnalysisResponse,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=[_resolve(root, r.path) for r in records],
            )
            analysis = _validate_analysis(response.value, len(selected))
            usage = _usage(response.usage)
            cache = _AnalysisCache(
                model_id=model_id, evidence_hash=evidence_hash, analysis=analysis, usage=usage
            )
        # Reject mutation during inference, even when the source stat identity was restored.
        if (
            source_fingerprint(source) != evidence.clip_fingerprint
            or _source_sha(source) != source_hash
        ):
            raise ValueError("Source fingerprint/content changed during style enrichment")
        for record in records:
            _read_record(root, record)
        visual = analysis.model_dump(exclude={"evidence_refs"})
        card = StyleCard.model_validate(
            {**visual, **host, "keyframes": keyframes, "evidence": analysis.evidence_refs}
        )
        if cached_record is None:
            cache_text = _json(cache.model_dump(mode="json"))
            manifest.analyses[analysis_key] = _immutable_text(
                root, f"analysis/{_sha(cache_text.encode())}.json", cache_text
            )
        source_record = _immutable_text(root, host["source_clip_ref"], source_text)
        card_text = _json(card.model_dump(mode="json"))
        card_record = _immutable_text(root, f"cards/{_sha(card_text.encode())}.json", card_text)
        manifest.cards[card.card_id] = _CardRecord(
            **card_record.model_dump(),
            source=source_record,
            frames=records,
            analysis_key=analysis_key,
        )
        text = _json(manifest.model_dump(mode="json"))
        manifest_path = _resolve(root, "library_manifest.json", exists=False)
        if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != text:
            atomic_write_text(manifest_path, text)
        return StyleCardBuildResult(card, usage, cache_hit, evidence_hash, analysis_key)
