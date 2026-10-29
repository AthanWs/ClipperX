"""Transcript-driven semantic segmentation for clip selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, Sequence, cast

import numpy as np
from sentence_transformers import SentenceTransformer

cosine_similarity = cast(
    Callable[[np.ndarray, np.ndarray], np.ndarray],
    import_module("sklearn.metrics.pairwise").cosine_similarity,
)

from .config import settings  # noqa: E402 - sklearn remains eager before local imports
from .schema import ClipCandidate, TranscriptSegment, SilenceWindow  # noqa: E402
from .logger import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass(slots=True)
class SegmentationConfig:
    min_duration: float = 15.0
    max_duration: float = 179.0
    overlap: float = 1.5
    similarity_threshold: float = 0.68
    min_coherence: float = 0.35


def load_embedder(model_name: str | None = None) -> SentenceTransformer:
    name = model_name or settings.embedding_model
    return SentenceTransformer(name)


def compute_embeddings(
    segments: Sequence[TranscriptSegment],
    *,
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    embedder = model or load_embedder()
    sentences = [segment.text for segment in segments]
    return embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)


def detect_boundaries(
    segments: Sequence[TranscriptSegment],
    embeddings: np.ndarray,
    *,
    config: SegmentationConfig | None = None,
) -> list[int]:
    cfg = config or SegmentationConfig()
    boundaries = [0]
    for idx in range(1, len(segments)):
        sim = float(cosine_similarity(embeddings[idx - 1 : idx], embeddings[idx : idx + 1])[0][0])
        duration = segments[idx].end - segments[idx].start
        if sim < cfg.similarity_threshold or duration > cfg.max_duration:
            boundaries.append(idx)
    boundaries.append(len(segments))
    return boundaries


def window_segments(
    segments: Sequence[TranscriptSegment],
    boundaries: Sequence[int],
    *,
    config: SegmentationConfig | None = None,
    embedder: SentenceTransformer | None = None,
) -> list[ClipCandidate]:
    cfg = config or SegmentationConfig()
    clips: list[ClipCandidate] = []
    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        block = segments[start_idx:end_idx]
        if not block:
            continue
        start = block[0].start
        end = block[-1].end
        duration = end - start
        if duration < cfg.min_duration:
            continue

        text = " ".join(seg.text for seg in block)
        coherence = _compute_coherence(block, embedder=embedder)
        if coherence < cfg.min_coherence:
            continue

        clips.append(
            ClipCandidate(
                start=start,
                end=end,
                text=text,
                coherence=coherence,
                segment_range=(start_idx, end_idx),
            )
        )

    return _extend_short_clips(clips, cfg.overlap)


def _compute_coherence(
    block: Sequence[TranscriptSegment],
    *,
    embedder: SentenceTransformer | None = None,
) -> float:
    sentences = [segment.text for segment in block]
    if len(sentences) <= 1:
        return 1.0
    try:
        emb = embedder or load_embedder()
        embeddings = emb.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        sims: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            sims.append(sim)
        return float(np.mean(sims)) if sims else 1.0
    except Exception:
        # Fallback to original length-uniformity heuristic
        lengths = np.array([len(sentence) for sentence in sentences], dtype=float)
        std = float(lengths.std())
        return math.exp(-std / (lengths.mean() + 1e-6))


def _extend_short_clips(clips: list[ClipCandidate], overlap: float) -> list[ClipCandidate]:
    for idx in range(1, len(clips)):
        prev_clip, clip = clips[idx - 1], clips[idx]
        if clip.start - prev_clip.end < overlap:
            midpoint = (prev_clip.end + clip.start) / 2
            prev_clip.end = midpoint
            clip.start = midpoint
    return clips


def generate_candidates(
    segments: Sequence[TranscriptSegment],
    *,
    embedder: SentenceTransformer | None = None,
    config: SegmentationConfig | None = None,
) -> list[ClipCandidate]:
    emb = embedder or load_embedder()
    embeddings = compute_embeddings(segments, model=emb)
    boundaries = detect_boundaries(segments, embeddings, config=config)
    logger.info(
        "[segmentation] límites detectados",
        extra={"total_segments": len(segments), "boundaries": len(boundaries)},
    )
    return window_segments(segments, boundaries, config=config, embedder=emb)


def extract_silences(
    segments: Sequence[TranscriptSegment],
    *,
    min_gap: float = 1.5,
) -> list[SilenceWindow]:
    silences: list[SilenceWindow] = []
    previous_end = segments[0].start if segments else 0.0
    for segment in segments:
        gap = segment.start - previous_end
        if gap >= min_gap:
            silences.append(SilenceWindow(start=previous_end, end=segment.start))
        previous_end = segment.end
    return silences


def segments_from_json(path: Path) -> list[TranscriptSegment]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TranscriptSegment.from_dict(item) for item in data]


