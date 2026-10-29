"""Generic LLM client helpers with multi-provider support."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import requests

from .config import settings
from .logger import get_logger
from .progress import reporter
from .prompts import (
    HEADLINE_PROMPT,
    VISION_PROMPT,
    EDITORIAL_BRIEF_PROMPT,
    CHUNK_CLIP_PROMPT,
    GAP_AUDIT_PROMPT,
    BOUNDARY_AUDIT_PROMPT,
    VIRALITY_SCORE_PROMPT,
    BROLL_SUGGEST_PROMPT,
)
from .schema import ViralityScore

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(slots=True)
class LLMResponse:
    headline: str
    keywords: list[str]


class LLMDisabledError(RuntimeError):
    """Raised when inference is attempted without an explicitly selected provider."""


def _encode_image(path: Path) -> str:
    """Encode image to base64 data URI for OpenAI API."""
    mime_type = "image/png"
    if path.suffix.lower() in (".jpg", ".jpeg"):
        mime_type = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        mime_type = "image/webp"

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

def infer_provider() -> str:
    """Auto-detect the best available LLM provider."""
    configured = settings.llm_provider.lower()
    if configured != "auto":
        return configured

    # Cascade: google -> openai -> anthropic -> ollama -> proxy
    if settings.google_api_key:
        return "google"
    if settings.openai_api_key:
        return "openai"
    if settings.anthropic_api_key:
        return "anthropic"
    # Check if ollama is reachable
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
        if resp.ok:
            return "ollama"
    except Exception:
        pass
    return "proxy"


def list_providers() -> list[dict[str, str]]:
    """Return available providers and which one is active."""
    active = infer_provider()
    providers = []
    for name, available in [
        ("disabled", True),
        ("proxy", True),
        ("openai", bool(settings.openai_api_key)),
        ("google", bool(settings.google_api_key)),
        ("anthropic", bool(settings.anthropic_api_key)),
        ("ollama", _check_ollama()),
    ]:
        providers.append({
            "name": name,
            "available": "yes" if available else "no",
            "active": "active" if name == active else "",
        })
    return providers


def _check_ollama() -> bool:
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return resp.ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

def _request_json(
    prompt: str,
    *,
    model: str,
    images: list[Path] | None = None,
    temperature: float = 0.3,
    timeout: int = 240,
) -> dict[str, Any] | None:
    """Dispatch JSON request to the configured provider."""
    provider = infer_provider()

    if provider == "disabled":
        raise LLMDisabledError(
            "El LLM está deshabilitado. Configura LLM_PROVIDER y los modelos explícitamente."
        )

    if provider == "google":
        return _request_google_json(prompt, model=model, images=images,
                                    temperature=temperature, timeout=timeout)
    elif provider == "anthropic":
        return _request_anthropic_json(prompt, model=model, images=images,
                                       temperature=temperature, timeout=timeout)
    elif provider == "ollama":
        return _request_openai_json(prompt, model=model, images=images,
                                    temperature=temperature, timeout=timeout,
                                    base_url=f"{settings.ollama_base_url}/v1",
                                    api_key="ollama")
    else:
        # "proxy", "openai", or fallback
        return _request_openai_json(prompt, model=model, images=images,
                                    temperature=temperature, timeout=timeout)


def _request_openai_json(
    prompt: str,
    *,
    model: str,
    images: list[Path] | None = None,
    temperature: float = 0.3,
    timeout: int = 240,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any] | None:

    messages: list[JsonValue] = []

    content: list[JsonValue] = [{"type": "text", "text": prompt}]

    if images:
        for img_path in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": _encode_image(img_path)
                }
            })

    messages.append({"role": "user", "content": content})

    target_url = base_url or settings.openai_base_url
    target_key = api_key or settings.openai_api_key

    headers = {
        "Authorization": f"Bearer {target_key}",
        "Content-Type": "application/json"
    }

    payload: dict[str, JsonValue] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(
            f"{target_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()

        result = response.json()
        content_str = result["choices"][0]["message"]["content"]

        # Try to clean markdown code blocks if present
        clean_content = content_str.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        if clean_content.startswith("```"):
            clean_content = clean_content[3:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]

        return json.loads(clean_content)

    except Exception as exc:
        logger = get_logger(__name__)
        logger.warning("[llm] Failed to get JSON response", extra={"error": str(exc), "model": model})
        return None


def _request_google_json(
    prompt: str,
    *,
    model: str,
    images: list[Path] | None = None,
    temperature: float = 0.3,
    timeout: int = 240,
) -> dict[str, Any] | None:
    """Send request to Google Gemini via REST API."""
    logger = get_logger(__name__)
    api_key = settings.google_api_key
    if not api_key:
        return None

    # Use the generativelanguage REST endpoint
    gemini_model = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{gemini_model}:generateContent?key={api_key}"

    parts: list[JsonValue] = [{"text": prompt}]
    if images:
        for img_path in images:
            mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
            parts.append({"inline_data": {"mime_type": mime, "data": data}})

    payload: dict[str, JsonValue] = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        clean = text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean)
    except Exception as exc:
        logger.warning("[llm/google] request failed", extra={"error": str(exc)})
        return None


def _request_anthropic_json(
    prompt: str,
    *,
    model: str,
    images: list[Path] | None = None,
    temperature: float = 0.3,
    timeout: int = 240,
) -> dict[str, Any] | None:
    """Send request to Anthropic API via REST."""
    logger = get_logger(__name__)
    api_key = settings.anthropic_api_key
    if not api_key:
        return None

    content: list[JsonValue] = []
    if images:
        for img_path in images:
            mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data},
            })
    content.append({"type": "text", "text": prompt + "\n\nResponde SOLO con JSON válido."})

    anthropic_model = model
    # Map generic model names to Anthropic model IDs
    if not model.startswith("claude-"):
        anthropic_model = "claude-sonnet-4-20250514"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: dict[str, JsonValue] = {
        "model": anthropic_model,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        clean = text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean)
    except Exception as exc:
        logger.warning("[llm/anthropic] request failed", extra={"error": str(exc)})
        return None


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def generate_headline(text: str, *, model: str | None = None) -> LLMResponse:
    # Trim context based on configured context window (approx 4 chars per token)
    max_chars = int(settings.llm_num_ctx * 3.5)
    context = text.strip()
    if len(context) > max_chars:
        context = context[-max_chars:]

    target_model = model or settings.openai_model

    data = _request_json(
        HEADLINE_PROMPT.format(text=context),
        model=target_model,
        temperature=0.6,
    )

    if data is not None:
        return LLMResponse(headline=data.get("headline", ""), keywords=data.get("keywords", []))

    # Interactive retry gate
    logger = get_logger(__name__)
    logger.warning(
        "[summaries] LLM unavailable; requesting user action",
        extra={"model": target_model},
    )
    choice = reporter.request_choice(
        "El servicio LLM no está respondiendo. ¿Qué deseas hacer?",
        ["Reintentar", "Omitir títulos"],
    )
    if choice == "Reintentar":
        return generate_headline(text, model=model)
    return LLMResponse(headline="", keywords=[])


def score_virality(text: str, headline: str) -> ViralityScore:
    """Score a clip's viral potential across 4 dimensions."""
    logger = get_logger(__name__)
    target_model = settings.openai_model

    data = _request_json(
        VIRALITY_SCORE_PROMPT.format(text=text.strip()[:3000], headline=headline),
        model=target_model,
        temperature=0.3,
    )

    if data is None:
        logger.warning("[llm] virality scoring failed, returning zeros")
        return ViralityScore()

    try:
        hook = max(0, min(25, int(data.get("hook", 0))))
        engagement = max(0, min(25, int(data.get("engagement", 0))))
        value = max(0, min(25, int(data.get("value", 0))))
        shareability = max(0, min(25, int(data.get("shareability", 0))))
        total = hook + engagement + value + shareability
        # Use LLM total if it matches, otherwise recalculate
        llm_total = int(data.get("total", 0))
        if llm_total != total:
            logger.info("[llm] virality total mismatch, recalculated",
                        extra={"llm_total": llm_total, "calculated": total})
        return ViralityScore(
            hook=hook,
            engagement=engagement,
            value=value,
            shareability=shareability,
            total=total,
            hook_type=str(data.get("hook_type", "none")),
            reasoning=str(data.get("reasoning", "")),
            emotion=str(data.get("emotion", "none")),
        )
    except Exception as exc:
        logger.warning("[llm] virality parse error", extra={"error": str(exc)})
        return ViralityScore()


def suggest_broll(text: str, clip_duration: float) -> list[dict[str, Any]]:
    """Ask LLM to suggest B-roll insertion moments for a clip."""
    target_model = settings.openai_model

    data = _request_json(
        BROLL_SUGGEST_PROMPT.format(text=text.strip()[:2000], duration=clip_duration),
        model=target_model,
        temperature=0.4,
    )

    if data is None:
        return []

    suggestions = data.get("suggestions", [])
    valid: list[dict[str, Any]] = []
    for s in suggestions:
        if isinstance(s, dict) and "timestamp" in s and "search_term" in s:
            valid.append({
                "timestamp": float(s.get("timestamp", 0)),
                "duration": float(s.get("duration", 2.0)),
                "search_term": str(s.get("search_term", "")),
            })
    return valid


def analyze_visual_relevance(image_path: Path, context: str, *, model: str | None = None) -> dict[str, Any]:
    target_model = model or settings.openai_vision_model

    data = _request_json(
        VISION_PROMPT.format(context=context.strip()[:1000]),
        model=target_model,
        images=[image_path],
        temperature=0.2,
    )

    if data is not None:
        return data

    logger = get_logger(__name__)
    logger.warning(
        "[analysis] Vision model unavailable",
        extra={"model": target_model},
    )
    choice = reporter.request_choice(
        "Modelo de visión no disponible. ¿Qué deseas hacer?",
        ["Reintentar", "Omitir análisis visual"],
    )
    if choice == "Reintentar":
        return analyze_visual_relevance(image_path, context, model=model)
    return {"relevante": False, "motivo": "omitido"}


def analyze_gap_decision(
    image_paths: list[Path],
    *,
    before_text: str,
    after_text: str,
    audio_features: dict[str, Any],
    gap_start: float,
    gap_end: float,
    clip_headline: str = "",
    model: str | None = None,
) -> dict[str, Any]:
    prompt = GAP_AUDIT_PROMPT.format(
        before_text=before_text.strip()[:1000],
        after_text=after_text.strip()[:1000],
        audio_features=json.dumps(audio_features, ensure_ascii=False),
        gap_start=gap_start,
        gap_end=gap_end,
        gap_duration=gap_end - gap_start,
        clip_headline=clip_headline,
    )
    target_model = model or settings.openai_vision_model

    data = _request_json(
        prompt,
        model=target_model,
        images=image_paths,
        temperature=0.2,
    )
    return data or {}


def analyze_boundary_decision(
    image_paths: list[Path],
    *,
    before_text: str,
    after_text: str,
    audio_features: dict[str, Any],
    boundary_side: str,
    clip_duration: float,
    clip_headline: str,
    model: str | None = None,
) -> dict[str, Any]:
    prompt = BOUNDARY_AUDIT_PROMPT.format(
        before_text=before_text.strip()[:1000],
        after_text=after_text.strip()[:1000],
        audio_features=json.dumps(audio_features, ensure_ascii=False),
        boundary_side=boundary_side,
        clip_duration=clip_duration,
        clip_headline=clip_headline,
    )
    target_model = model or settings.openai_vision_model

    data = _request_json(
        prompt,
        model=target_model,
        images=image_paths,
        temperature=0.2,
    )
    return data or {}


def generate_editorial_brief(sample_text: str) -> str:
    """Generate an editorial brief for the entire stream from a representative sample."""
    logger = get_logger(__name__)
    target_model = settings.openai_model

    prompt = EDITORIAL_BRIEF_PROMPT.format(sample=sample_text)
    data = _request_json(prompt, model=target_model, temperature=0.4)

    if data is not None:
        import json as _json
        brief = _json.dumps(data, ensure_ascii=False, indent=2)
        logger.info("[llm] brief editorial generado", extra={"brief": brief})
        return brief

    logger.warning("[llm] no se pudo generar brief editorial, continuando sin contexto")
    return ""


def suggest_chunk_clips(
    chunk_text: str,
    *,
    chunk_start: float,
    chunk_end: float,
    editorial_brief: str = "",
    images: list[Path] | None = None,
    audio_features: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Ask LLM to propose clip cuts from a transcript chunk."""
    logger = get_logger(__name__)

    brief_section = editorial_brief or "No disponible."
    prompt = CHUNK_CLIP_PROMPT.format(
        chunk=chunk_text,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        editorial_brief=brief_section,
    )

    # If images or audio features are provided, append audio context to prompt
    if audio_features:
        prompt += f"\n\nAUDIO FEATURES DEL FRAGMENTO:\n{json.dumps(audio_features, ensure_ascii=False)}"

    use_images = images and len(images) > 0
    target_model = settings.openai_vision_model if use_images else settings.openai_model

    if use_images and images is not None:
        logger.info(
            "[llm] suggest_chunk_clips con visión y audio",
            extra={"frames": len(images), "has_audio_features": audio_features is not None},
        )

    data = _request_json(
        prompt,
        model=target_model,
        images=images if use_images else None,
        temperature=0.3,
    )

    if data is not None:
        clips = data.get("clips", [])
        valid = [c for c in clips if isinstance(c, dict) and "start" in c and "end" in c and "headline" in c]
        return valid

    logger.warning("[llm] failed to suggest clips")
    choice = reporter.request_choice(
        "No se pudo generar sugerencias de clips para este fragmento. ¿Qué deseas hacer?",
        ["Reintentar", "Omitir este fragmento"],
    )
    if choice == "Reintentar":
        return suggest_chunk_clips(
            chunk_text,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            editorial_brief=editorial_brief,
            images=images,
            audio_features=audio_features,
        )
    return []
