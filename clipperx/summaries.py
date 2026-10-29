"""Headline, keyword generation, and virality scoring utilities."""

from __future__ import annotations

from typing import Sequence
import re

from .helpers import clip_text
from .llm import generate_headline, score_virality
from .schema import ClipCandidate, TranscriptSegment
from .logger import get_logger


def enrich_clips(clips: Sequence[ClipCandidate], segments: Sequence[TranscriptSegment]) -> None:
    logger = get_logger(__name__)
    for clip in clips:
        text = clip_text(clip, segments)

        # Generate headline if missing
        if not clip.headline:
            result = generate_headline(text)
            if not result.headline:
                logger.info("[summaries] títulos omitidos por elección del usuario")
            clip.headline = result.headline or ""
            clip.keywords = result.keywords or []

        # Score virality
        if clip.virality is None and clip.headline:
            clip.virality = score_virality(text, clip.headline)
            logger.info(
                "[summaries] virality score",
                extra={
                    "headline": clip.headline,
                    "total": clip.virality.total,
                    "hook": clip.virality.hook,
                    "engagement": clip.virality.engagement,
                    "value": clip.virality.value,
                    "shareability": clip.virality.shareability,
                    "hook_type": clip.virality.hook_type,
                },
            )


def build_filename(headline: str, clip: ClipCandidate) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", headline).strip("-").lower()
    # Include virality score in filename if available
    score_part = ""
    if clip.virality and clip.virality.total > 0:
        score_part = f"-viral{clip.virality.total}"
    return f"{slug or 'clip'}{score_part}-{int(clip.start)}-{int(clip.end)}.mp4"
