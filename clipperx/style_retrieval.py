"""Conservative compatibility filtering, then local semantic style ranking.

Cards are edited reference geometry, not proof of a source crop or exact people
count. Known scene families require an exact family match. UNKNOWN/MIXED queries
use only their explicit structural requirements; they never imply gameplay.
Exact visible_people requests fail closed because StyleCard stores no counts.
people_requirement='multiple' uses MULTI_PERSON only as categorical evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal, Protocol, TypeAlias, overload
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from clipperx.jobs import _InterProcessFileLock
from clipperx.local_paths import paths
from clipperx.style_contracts import BackgroundTreatment, SceneFamily, StyleCard
from clipperx.style_library import StyleLibrary
from clipperx.visual_contracts import NormalizedRect, _StrictVisualContract

VectorArray: TypeAlias = NDArray[np.float64]
_CAMERA_FAMILIES = {
    SceneFamily.GAMEPLAY_WITH_CAMERA,
    SceneFamily.BROWSER_WITH_CAMERA,
    SceneFamily.REACTED_VIDEO_WITH_CAMERA,
    SceneFamily.FULL_CAMERA,
}
_UNCERTAIN = {SceneFamily.UNKNOWN, SceneFamily.MIXED}
_MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
_MINILM_SHA = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"


class StyleQuery(_StrictVisualContract):
    scene_family: SceneFamily = SceneFamily.UNKNOWN
    has_camera: bool | None = None
    preserve_content: bool = False
    visible_people: int | None = Field(default=None, ge=0)
    people_requirement: Literal["any", "multiple"] = "any"
    camera_regions: list[NormalizedRect] = Field(default_factory=list)
    content_regions: list[NormalizedRect] = Field(default_factory=list)
    background_treatment: BackgroundTreatment = BackgroundTreatment.COVER
    title_regions: list[NormalizedRect] = Field(default_factory=list)
    brand_regions: list[NormalizedRect] = Field(default_factory=list)
    subtitle_regions: list[NormalizedRect] = Field(default_factory=list)
    subtitle_palette: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_camera_requirement(self) -> "StyleQuery":
        if self.has_camera is False and self.camera_regions:
            raise ValueError("has_camera=False contradicts nonempty camera_regions")
        return self


class StyleEncoder(Protocol):
    """Identity must change whenever model weights/tokenization/encoding change."""

    @property
    def model_identity(self) -> str: ...

    def encode(self, texts: list[str]) -> Any: ...


@dataclass(frozen=True)
class StyleMatch:
    card: StyleCard
    compatibility_score: float
    style_score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class StyleRejection:
    card_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalResult(Sequence[StyleMatch]):
    matches: tuple[StyleMatch, ...]
    coverage: Literal["sparse", "sufficient"]
    rejections: tuple[StyleRejection, ...] = ()

    def __len__(self) -> int:
        return len(self.matches)

    def __iter__(self) -> Iterator[StyleMatch]:
        return iter(self.matches)

    @overload
    def __getitem__(self, index: int) -> StyleMatch: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[StyleMatch, ...]: ...

    def __getitem__(self, index: int | slice) -> StyleMatch | tuple[StyleMatch, ...]:
        return self.matches[index]


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class LocalStyleEncoder:
    """Load only an explicitly installed project model; never acquire models.

    Default MiniLM weights are pinned to revision
    1110a243fdf4706b3f48f1d95db1a4f5529b4d41 via their safetensors hash.
    Cache identity additionally hashes every local model/config/tokenizer file.
    """

    def __init__(self, *, model_path: Path | None = None) -> None:
        # Only actual encoder construction may load environment-backed settings.
        from clipperx.config import settings

        configured = model_path or (
            Path(settings.style_embedding_model_path)
            if settings.style_embedding_model_path
            else Path("data/models/style_embeddings") / settings.embedding_model.rsplit("/", 1)[-1]
        )
        local = (configured if configured.is_absolute() else paths.base_dir / configured).resolve()
        required = ["modules.json", "config.json", "model.safetensors", "tokenizer.json"]
        if not local.is_dir() or any(not (local / name).is_file() for name in required):
            raise FileNotFoundError(
                "Local style embedding model missing/incomplete. Set style_embedding_model_path "
                "to an installed SentenceTransformer directory; authorize its download separately."
            )
        records = []
        weights_hash = ""
        for file in sorted(local.rglob("*")):
            if not file.is_file() or ".cache" in file.relative_to(local).parts:
                continue
            if file.suffix.lower() not in {".json", ".safetensors", ".txt", ".model"}:
                continue
            if not file.resolve().is_relative_to(local):
                raise ValueError("Local embedding model file escapes model directory")
            digest = hashlib.sha256()
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            value = digest.hexdigest()
            records.append((file.relative_to(local).as_posix(), value))
            if file.name == "model.safetensors" and file.parent == local:
                weights_hash = value
        if settings.embedding_model == _MINILM_ID and weights_hash != _MINILM_SHA:
            raise ValueError("Local MiniLM embedding weights do not match the pinned SHA-256")
        self._identity = _sha(_json({"model": settings.embedding_model, "files": records}).encode())
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            str(local),
            cache_folder=str(paths.base_dir / "data/models/style_embeddings"),
            local_files_only=True,
            trust_remote_code=False,
            device="cpu",
        )

    @property
    def model_identity(self) -> str:
        return self._identity

    def encode(self, texts: list[str]) -> Any:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def _compatibility(query: StyleQuery, card: StyleCard) -> tuple[bool, tuple[str, ...]]:
    failures = []
    if card.approval == "rejected":
        failures.append("approval:rejected")
    if query.scene_family not in _UNCERTAIN and card.scene_family != query.scene_family:
        failures.append("scene_family:mismatch_or_unknown")
    # Family and geometry must both support a requested camera, not just a label.
    camera = bool(card.camera_regions)
    camera_required = query.has_camera
    if camera_required is None and (query.camera_regions or query.scene_family in _CAMERA_FAMILIES):
        camera_required = True
    elif camera_required is None and query.scene_family == SceneFamily.CONTENT_ONLY:
        camera_required = False
    if camera_required is True and (not camera or card.scene_family == SceneFamily.CONTENT_ONLY):
        failures.append("camera:presence_not_supported")
    if camera_required is False and (camera or card.scene_family in _CAMERA_FAMILIES):
        failures.append("camera:absence_not_supported")
    needs_content = (
        query.preserve_content
        or bool(query.content_regions)
        or query.scene_family
        in {
            SceneFamily.GAMEPLAY_WITH_CAMERA,
            SceneFamily.BROWSER_WITH_CAMERA,
            SceneFamily.REACTED_VIDEO_WITH_CAMERA,
            SceneFamily.CONTENT_ONLY,
        }
    )
    if needs_content and (not card.content_regions or card.scene_family == SceneFamily.FULL_CAMERA):
        failures.append("content:preservation_not_supported")
    if query.visible_people is not None:
        failures.append("exact_people_count_unavailable")
    if query.people_requirement == "multiple" and card.scene_family != SceneFamily.MULTI_PERSON:
        failures.append("people:multiplicity_not_supported")
    if failures:
        return False, tuple(failures)
    reasons = ["scene_family:compatible", "camera:compatible", "content:compatible"]
    if card.scene_family in _UNCERTAIN:
        reasons.append("scene_family:unknown_or_mixed;explicit_structural_requirements_only")
    if card.scene_family == SceneFamily.MULTI_PERSON:
        reasons.append("people:multiple_family_only;exact_count_unknown")
    else:
        reasons.append("people:count_unknown;no_exact_constraint")
    return True, tuple(reasons)


def _regions(left: list[NormalizedRect], right: list[NormalizedRect]) -> float:
    if not left or not right:
        return float(not left and not right)

    def nearest(a: list[NormalizedRect], b: list[NormalizedRect]) -> float:
        return sum(
            min(
                (
                    abs(x.x - y.x)
                    + abs(x.y - y.y)
                    + abs(x.width - y.width)
                    + abs(x.height - y.height)
                )
                / 4
                for y in b
            )
            for x in a
        ) / len(a)

    return (1 - (nearest(left, right) + nearest(right, left)) / 2) * (
        min(len(left), len(right)) / max(len(left), len(right))
    )


def _recency(created_at: datetime | None, now: datetime) -> float:
    # Missing dates are neutral; naive dates are UTC. Future dates receive 1, never >1.
    if created_at is None:
        return 0.5
    created_at = (
        created_at.replace(tzinfo=timezone.utc) if created_at.tzinfo is None else created_at
    )
    now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now
    days = max(0.0, (now - created_at).total_seconds() / 86400)
    return 2.0 ** (-days / 365)


def _vectors(value: Any, count: int, dimension: int | None = None) -> VectorArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError("Invalid embedding numeric type")
    vectors = np.asarray(raw, dtype=np.float64)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != count
        or vectors.shape[1] == 0
        or (dimension is not None and vectors.shape[1] != dimension)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Invalid embedding shape, dimension or nonfinite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Invalid embedding norm")
    return vectors / norms


def _checksum(identity: str, keys: list[str], vectors: VectorArray) -> str:
    return _sha(_json([identity, keys]).encode() + vectors.astype("<f8").tobytes())


def _read_index(index: Path, identity: str, dimension: int) -> dict[str, VectorArray]:
    if not index.is_file():
        return {}
    try:
        with np.load(index, allow_pickle=False) as data:
            if int(data["version"].item()) != 1 or str(data["identity"].item()) != identity:
                return {}
            keys = data["keys"].tolist()
            if (
                not isinstance(keys, list)
                or any(not isinstance(k, str) or not re.fullmatch(r"[0-9a-f]{64}", k) for k in keys)
                or len(keys) != len(set(keys))
            ):
                return {}
            raw = np.asarray(data["vectors"], dtype=np.float64)
            # Validate checksum before normalizing, so corrupt vectors cannot gain validity.
            if str(data["checksum"].item()) != _checksum(identity, keys, raw):
                return {}
            vectors = _vectors(raw, len(keys), dimension)
            return {key: row for key, row in zip(keys, vectors)}
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile, OverflowError):
        return {}


def _write_index(index: Path, identity: str, entries: dict[str, VectorArray]) -> None:
    keys = sorted(entries)
    vectors = np.stack([entries[key] for key in keys])
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=index.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(
                handle,
                version=np.asarray(1),
                identity=np.asarray(identity),
                keys=np.asarray(keys),
                vectors=vectors,
                checksum=np.asarray(_checksum(identity, keys, vectors)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(index)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _card_vectors(
    cards: list[StyleCard],
    encoder: StyleEncoder,
    dimension: int,
    index: Path | None,
) -> VectorArray:
    identity = encoder.model_identity
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("A nonempty embedding model_identity is required")
    keys = [
        _sha(_json({"model": identity, "card": c.model_dump(mode="json")}).encode()) for c in cards
    ]

    def populate(entries: dict[str, VectorArray]) -> VectorArray:
        missing = dict(
            (key, card.description) for key, card in zip(keys, cards) if key not in entries
        )
        if missing:
            vectors = _vectors(encoder.encode(list(missing.values())), len(missing), dimension)
            entries.update(zip(missing, vectors))
            if index is not None:
                _write_index(index, identity, entries)
        return np.stack([entries[key] for key in keys])

    if index is None:
        return populate({})
    index.parent.mkdir(parents=True, exist_ok=True)
    with _InterProcessFileLock(index.with_suffix(".lock")):
        return populate(_read_index(index, identity, dimension))


def retrieve_style_cards(
    query: StyleQuery,
    cards: Sequence[StyleCard] | StyleLibrary,
    *,
    limit: int = 3,
    encoder: StyleEncoder | None = None,
    clock: Callable[[], datetime] | None = None,
    index_path: Path | None = None,
) -> RetrievalResult:
    """Return up to three unique compatible clips, with sparse coverage if <2 exist.

    Coverage counts available compatible unique fingerprints before the requested
    limit. Library input validates every published card and caches at its root in
    style_index.npz. Sequence input is explicit in-memory cards; pass index_path to
    persist vectors. All caches use content/model keys, no pickle, a writer lock
    and atomic replacement. Corrupt/stale caches rebuild; corrupt libraries raise.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    query = StyleQuery.model_validate(query.model_dump())
    if isinstance(cards, StyleLibrary):
        if index_path is None:
            index_path = cards.root / "style_index.npz"
        for candidate in (index_path, index_path.with_suffix(".lock")):
            if not candidate.resolve().is_relative_to(cards.root):
                raise ValueError("Style index escapes library containment")
        source = cards.list_cards()
    else:
        source = list(cards)
    compatible: list[tuple[StyleCard, tuple[str, ...]]] = []
    rejected = []
    for original in source:
        card = StyleCard.model_validate(original.model_dump())
        accepted, reasons = _compatibility(query, card)
        if accepted:
            compatible.append((card, reasons))
        else:
            rejected.append(StyleRejection(card.card_id, reasons))
    coverage: Literal["sparse", "sufficient"] = (
        "sufficient" if len({c.clip_fingerprint for c, _ in compatible}) >= 2 else "sparse"
    )
    if not compatible:
        return RetrievalResult((), coverage, tuple(rejected))
    encoder = encoder if encoder is not None else LocalStyleEncoder()
    query_vector = _vectors(encoder.encode([query.description]), 1)
    vectors = _card_vectors([c for c, _ in compatible], encoder, query_vector.shape[1], index_path)
    now = clock() if clock else datetime.now(timezone.utc)
    ranked = []
    for (card, reasons), vector in zip(compatible, vectors):
        scene = 1.0 if query.scene_family == card.scene_family else 0.5
        geometry = (
            0.45 * _regions(query.camera_regions, card.camera_regions)
            + 0.45 * _regions(query.content_regions, card.content_regions)
            + 0.10 * float(query.background_treatment == card.background_treatment)
        )
        palette_left = {c.lower() for c in query.subtitle_palette}
        palette_right = {c.lower() for c in card.subtitle_palette}
        palette = (
            len(palette_left & palette_right) / len(palette_left | palette_right)
            if palette_left or palette_right
            else 1.0
        )
        overlay = 0.25 * (
            _regions(query.title_regions, card.title_regions)
            + _regions(query.brand_regions, card.brand_regions)
            + _regions(query.subtitle_regions, card.subtitle_regions)
            + palette
        )
        semantic = float(np.clip(np.dot(query_vector[0], vector), 0, 1))
        recency = _recency(card.created_at, now)
        score = (
            0.35 * scene
            + 0.20 * geometry
            + 0.15 * overlay
            + 0.15 * semantic
            + 0.10 * card.visual_quality
            + 0.05 * recency
        )
        components = tuple(
            f"{name}={value:.6f}"
            for name, value in (
                ("scene", scene),
                ("geometry", geometry),
                ("overlay", overlay),
                ("semantic", semantic),
                ("visual_quality", card.visual_quality),
                ("recency", recency),
            )
        )
        ranked.append(StyleMatch(card, 1.0, float(np.clip(score, 0, 1)), reasons + components))
    ranked.sort(key=lambda match: (-match.style_score, match.card.card_id))
    matches = []
    fingerprints: set[str] = set()
    for match in ranked:
        if match.card.clip_fingerprint not in fingerprints:
            matches.append(match)
            fingerprints.add(match.card.clip_fingerprint)
        if len(matches) == min(3, limit):
            break
    return RetrievalResult(tuple(matches), coverage, tuple(rejected))
