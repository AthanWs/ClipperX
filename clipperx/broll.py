"""B-Roll integration via Pexels API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import requests

from .config import settings
from .logger import get_logger
from .schema import ClipCandidate, TranscriptSegment

logger = get_logger(__name__)


@dataclass(slots=True)
class BRollResult:
    """A single B-roll video search result from Pexels."""

    video_id: int
    url: str
    duration: float
    width: int
    height: int
    download_url: str


def search_broll(
    query: str,
    *,
    orientation: str = "portrait",
    count: int = 5,
) -> list[BRollResult]:
    """Search Pexels for B-roll footage matching *query*."""
    api_key = settings.pexels_api_key
    if not api_key:
        logger.warning("[broll] PEXELS_API_KEY not configured")
        return []

    headers = {"Authorization": api_key}
    params: dict[str, Any] = {
        "query": query,
        "orientation": orientation,
        "per_page": count,
    }

    try:
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("[broll] Pexels search failed", extra={"error": str(exc)})
        return []

    results: list[BRollResult] = []
    for video in data.get("videos", []):
        files = video.get("video_files", [])
        # Pick the best quality file with matching orientation
        best = None
        for f in files:
            if f.get("quality") == "hd" or best is None:
                best = f
        if best:
            results.append(
                BRollResult(
                    video_id=video["id"],
                    url=video.get("url", ""),
                    duration=video.get("duration", 0),
                    width=best.get("width", 0),
                    height=best.get("height", 0),
                    download_url=best.get("link", ""),
                )
            )
    return results


def download_broll(video_url: str, output_path: Path) -> Path:
    """Download a B-roll video from its URL to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(video_url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("[broll] downloaded", extra={"path": str(output_path)})
        return output_path
    except Exception as exc:
        logger.warning("[broll] download failed", extra={"error": str(exc)})
        raise


def select_best_broll(
    results: list[BRollResult],
    target_duration: float,
) -> BRollResult | None:
    """Pick the best B-roll result based on duration proximity and quality."""
    if not results:
        return None

    def score(r: BRollResult) -> float:
        # Lower is better: penalize duration mismatch + prefer higher resolution
        dur_diff = abs(r.duration - target_duration)
        quality_bonus = min(r.width, r.height) / 1920.0  # normalize
        return dur_diff - quality_bonus * 2.0

    return min(results, key=score)


def suggest_broll_moments(
    clips: Sequence[ClipCandidate],
    segments: Sequence[TranscriptSegment],
) -> list[dict[str, Any]]:
    """Use LLM to identify moments where B-roll would improve each clip.

    Returns a list of dicts with keys: clip_index, timestamp, duration, search_term.
    """
    from .llm import suggest_broll
    from .helpers import clip_text

    opportunities: list[dict[str, Any]] = []

    for idx, clip in enumerate(clips):
        text = clip_text(clip, segments)
        suggestions = suggest_broll(text, clip.duration)
        for s in suggestions:
            s["clip_index"] = idx
            opportunities.append(s)

    return opportunities
