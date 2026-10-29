"""High level orchestration pipeline for ClipperX."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from . import audio, editing, face, segmentation, summaries, transcription
from .analysis import refine_clips, analyze_clip_layout, _extract_frame_at
from .audio_features import AudioFeatureExtractor
from .broll import search_broll, download_broll, select_best_broll, suggest_broll_moments
from .cleaner import cleanup_after_success
from .config import paths, settings
from .llm import generate_editorial_brief, suggest_chunk_clips
from .schema import BRollInsert, ClipCandidate, TranscriptSegment
from .progress import reporter, TranscriptionEstimator
from .youtube import is_youtube_url, download_video
from .logger import configure_logging, get_logger
from .jobs import JobStore
from .media import probe_media
from .timeline import build_timeline

# Configure logging once at import time for pipeline operations
configure_logging()
logger = get_logger(__name__)


def preprocess(video_path: Path, *, output_path: Path | None = None) -> Path:
    logger.info("[pipeline] normalizando audio", extra={"video": str(video_path)})
    reporter.set_percent("audio", 10.0, "Normalizando audio")
    return audio.preprocess_audio(video_path, output_path=output_path)


def load_segments(transcript_path: Path) -> list[TranscriptSegment]:
    return segmentation.segments_from_json(transcript_path)


def select_clips(segments: Sequence[TranscriptSegment]) -> list[ClipCandidate]:
    logger.info("[pipeline] generando candidatos", extra={"segments": len(segments)})
    return segmentation.generate_candidates(segments)


def render_clips(
    video_path: Path,
    clips: Iterable[ClipCandidate],
    *,
    output_dir: Path | None = None,
    transcripts: Sequence[TranscriptSegment] | None = None,
    layout: editing.LayoutConfig | None = None,
    preset: str = "default",
) -> list[Path]:
    target_dir = output_dir or paths.clips_dir
    logger.info("[pipeline] renderizando clips", extra={"target_dir": str(target_dir)})
    return editing.render_batch(
        video_path,
        clips,
        target_dir,
        layout=layout,
        transcripts=transcripts,
        preset=preset,
    )


def _resolve_video_input(video_input: Path | str) -> Path:
    """If *video_input* is a YouTube URL, download it first; otherwise return as Path."""
    if isinstance(video_input, str) and is_youtube_url(video_input):
        logger.info("[pipeline] descargando video de YouTube", extra={"url": video_input})
        reporter.set_percent("download", 5.0, "Descargando video de YouTube...")
        return download_video(video_input, paths.downloads_dir)
    return Path(video_input) if isinstance(video_input, str) else video_input


def full_pipeline(
    video_path: Path | str,
    *,
    output_dir: Path | None = None,
    layout_override: editing.LayoutConfig | None = None,
    face_stride: int = 12,
    caption_template: str | None = None,
    min_virality: int | None = None,
    export_preset: str | None = None,
    enable_broll: bool = False,
) -> list[Path]:
    logger.info("[pipeline] inicio")

    # Resolve YouTube URLs
    resolved_path = _resolve_video_input(video_path)

    if settings.llm_provider.lower() == "disabled":
        raise RuntimeError(
            "El LLM está deshabilitado. Usa `python -m clipperx.cli index VIDEO` "
            "para construir el timeline sin inferencia."
        )

    job = JobStore.for_video(resolved_path)
    if job.stage_complete("audio", artifact=job.audio_path):
        audio_path = job.audio_path
        logger.info("[pipeline] reutilizando audio", extra={"audio": str(audio_path)})
    else:
        job.mark_started("audio")
        try:
            audio_path = preprocess(resolved_path, output_path=job.audio_path)
            job.mark_complete("audio", artifacts=[audio_path])
        except Exception as exc:
            job.mark_failed("audio", str(exc))
            raise

    # Determine effective settings
    effective_template = caption_template or settings.caption_template
    effective_min_virality = min_virality if min_virality is not None else settings.min_virality_score
    effective_preset = export_preset or settings.export_preset

    # Estimar progreso de transcripción
    media = probe_media(resolved_path)
    duration_seconds = float(media.get("duration_seconds") or 3600.0)

    from .stages import transcription_cache_is_current
    if transcription_cache_is_current(job, audio_path, language=settings.transcription_language):
        transcript_path = job.transcript_path
        logger.info("[pipeline] reutilizando transcript", extra={"transcript": str(transcript_path)})
    else:
        job.mark_started("transcription")
        est = TranscriptionEstimator(
            duration_seconds=duration_seconds,
            rtf=0.6,
            start_percent=20.0,
            end_percent=55.0,
        )
        est.start()
        try:
            transcript_path = transcription.transcribe_to_json(
                audio_path,
                output_dir=job.root,
                output_name="transcript",
                language=settings.transcription_language,
            )
            job.mark_complete("transcription", artifacts=[transcript_path])
        except Exception as exc:
            job.mark_failed("transcription", str(exc))
            raise
        finally:
            est.stop()
    reporter.set_percent("transcription", 55.0, "Transcripción finalizada")
    segments = load_segments(transcript_path)
    _mark_low_confidence_words(segments)

    reporter.set_percent("timeline", 57.0, "Construyendo timeline multimodal")
    timeline_path = build_timeline(resolved_path, include_audio=True)
    logger.info("[pipeline] timeline listo", extra={"timeline": str(timeline_path)})

    reporter.set_percent("brief", 58.0, "Generando brief editorial del stream")
    editorial_brief = _generate_stream_brief(segments)

    reporter.set_percent("summaries", 62.0, "Analizando transcript por partes (LLM)")
    llm_clips = _suggest_llm_clips(
        segments,
        editorial_brief=editorial_brief,
        video_path=resolved_path,
        audio_path=audio_path,
    )

    reporter.set_percent("segmentation", 65.0, "Segmentando")
    clips = select_clips(segments)
    if llm_clips:
        clips.extend(llm_clips)

    pre_dedup = len(clips)
    clips = _deduplicate_candidates(clips)
    logger.info(
        "[pipeline] deduplicación de candidatos",
        extra={"antes": pre_dedup, "después": len(clips), "eliminados": pre_dedup - len(clips)},
    )

    reporter.set_percent("analysis", 75.0, "Refinando clips")
    clips = refine_clips(
        resolved_path,
        audio_path,
        segments,
        clips,
        video_duration=duration_seconds,
    )

    reporter.set_percent("summaries", 82.0, "Resumiendo/títulos y scoring viral")
    summaries.enrich_clips(clips, segments)

    # Sort clips by virality score (descending)
    clips.sort(key=lambda c: c.virality.total if c.virality else 0, reverse=True)

    # Log virality ranking
    for rank, clip in enumerate(clips, 1):
        score = clip.virality.total if clip.virality else 0
        logger.info(
            f"[pipeline] ranking viral #{rank}",
            extra={"headline": clip.headline, "score": score, "duration": f"{clip.duration:.1f}s"},
        )

    # Filter by minimum virality score
    if effective_min_virality > 0:
        before_filter = len(clips)
        clips = [c for c in clips if (c.virality and c.virality.total >= effective_min_virality)]
        filtered = before_filter - len(clips)
        if filtered:
            logger.info(
                "[pipeline] clips filtrados por virality",
                extra={"filtrados": filtered, "umbral": effective_min_virality, "restantes": len(clips)},
            )

    # B-Roll stage (optional)
    if enable_broll and settings.pexels_api_key:
        reporter.set_percent("broll", 85.0, "Buscando B-roll")
        _apply_broll(clips, segments)

    analyze_clip_layout(resolved_path, clips)

    layout = layout_override
    if layout is None:
        classifier = face.load_face_model()
        try:
            for clip in clips:
                webcam = face.detect_webcam_region(
                    resolved_path,
                    start=clip.start,
                    end=clip.end,
                    max_samples=20,
                    classifier=classifier,
                )
                if webcam:
                    clip.webcam_region = webcam.as_tuple()
                    clip.face_crop = webcam.as_tuple()[:4]

                if clip.layout_segments:
                    segment_crops: dict[int, tuple[int, int, int, int] | None] = {}
                    for seg_idx, (seg_start, seg_end, _layout_mode) in enumerate(
                        clip.layout_segments
                    ):
                        segment_webcam = face.detect_webcam_region(
                            resolved_path,
                            start=seg_start,
                            end=seg_end,
                            max_samples=15,
                            classifier=classifier,
                        )
                        if segment_webcam:
                            segment_crops[seg_idx] = segment_webcam.as_tuple()[:4]
                            logger.info(
                                "[pipeline] webcam_crop por segmento",
                                extra={
                                    "clip_start": clip.start,
                                    "seg_idx": seg_idx,
                                    "webcam_crop": segment_webcam.as_tuple()[:4],
                                },
                            )
                        else:
                            # An uncertain/new layout must not reuse a crop from another scene.
                            segment_crops[seg_idx] = None
                    if segment_crops:
                        clip.segment_face_crops = segment_crops
        finally:
            classifier.close()

    layout = editing.prepare_render_layout(layout, caption_template=effective_template,
                                          brand_image_path=settings.brand_image_path)

    rendered = render_clips(
        resolved_path,
        clips,
        transcripts=segments,
        layout=layout,
        output_dir=output_dir,
        preset=effective_preset,
    )
    reporter.set_percent("render", 95.0, "Renderizando")
    cleanup_after_success(None)
    job.mark_complete("pipeline", artifacts=rendered)
    logger.info("[pipeline] completado", extra={"clips": len(rendered)})
    reporter.set_percent("done", 100.0, "Completado")
    return rendered


def _apply_broll(
    clips: list[ClipCandidate],
    segments: Sequence[TranscriptSegment],
) -> None:
    """Suggest and download B-roll for clips."""
    opportunities = suggest_broll_moments(clips, segments)

    for opp in opportunities:
        clip_idx = opp.get("clip_index", -1)
        if clip_idx < 0 or clip_idx >= len(clips):
            continue

        search_term = opp.get("search_term", "")
        target_dur = float(opp.get("duration", 2.0))
        if not search_term:
            continue

        results = search_broll(search_term, orientation="portrait", count=3)
        best = select_best_broll(results, target_dur)
        if not best:
            continue

        broll_path = paths.broll_dir / f"broll_{clip_idx}_{search_term.replace(' ', '_')}.mp4"
        try:
            download_broll(best.download_url, broll_path)
            clips[clip_idx].broll_inserts.append(
                BRollInsert(
                    timestamp=float(opp.get("timestamp", 0)),
                    duration=target_dur,
                    search_term=search_term,
                    video_path=broll_path,
                )
            )
        except Exception as exc:
            logger.warning("[pipeline] broll download failed", extra={"error": str(exc)})


def _generate_stream_brief(segments: Sequence[TranscriptSegment]) -> str:
    """Build a representative sample from start/middle/end and ask the LLM for an editorial brief."""
    if not segments:
        return ""
    n = len(segments)
    sample_size = max(1, min(n // 6, 20))
    start_sample = segments[:sample_size]
    mid_start = max(0, n // 2 - sample_size // 2)
    mid_sample = segments[mid_start : mid_start + sample_size]
    end_sample = segments[-sample_size:]

    lines: list[str] = []
    lines.append("--- INICIO DEL STREAM ---")
    for s in start_sample:
        lines.append(f"[{s.start:.1f}-{s.end:.1f}] {s.text.strip()}")
    lines.append("\n--- MITAD DEL STREAM ---")
    for s in mid_sample:
        lines.append(f"[{s.start:.1f}-{s.end:.1f}] {s.text.strip()}")
    lines.append("\n--- FINAL DEL STREAM ---")
    for s in end_sample:
        lines.append(f"[{s.start:.1f}-{s.end:.1f}] {s.text.strip()}")

    sample_text = "\n".join(lines)
    return generate_editorial_brief(sample_text)


def _suggest_llm_clips(
    segments: Sequence[TranscriptSegment],
    *,
    editorial_brief: str = "",
    video_path: Path | None = None,
    audio_path: Path | None = None,
) -> list[ClipCandidate]:
    chunks = _iter_transcript_chunks(segments)
    extractor = AudioFeatureExtractor(audio_path) if audio_path else None
    llm_clips: list[ClipCandidate] = []
    for chunk_start, chunk_end, chunk_text in chunks:
        # Extract representative frames from the chunk (start, middle, end)
        chunk_frames: list[Path] | None = None
        chunk_audio: dict | None = None
        if video_path is not None:
            try:
                t_start = chunk_start + (chunk_end - chunk_start) * 0.15
                t_mid = (chunk_start + chunk_end) / 2
                t_end = chunk_start + (chunk_end - chunk_start) * 0.85
                chunk_frames = [
                    _extract_frame_at(video_path, t_start, paths.frames_dir / f"chunk_{chunk_start:.0f}_s.jpg"),
                    _extract_frame_at(video_path, t_mid, paths.frames_dir / f"chunk_{chunk_start:.0f}_m.jpg"),
                    _extract_frame_at(video_path, t_end, paths.frames_dir / f"chunk_{chunk_start:.0f}_e.jpg"),
                ]
                logger.info(
                    "[pipeline] frames extraídos para chunk",
                    extra={"chunk_start": chunk_start, "frames": len(chunk_frames)},
                )
            except Exception as exc:
                logger.warning("[pipeline] no se pudieron extraer frames para chunk", extra={"error": str(exc)})
                chunk_frames = None
        if extractor is not None:
            try:
                chunk_audio = extractor.window(chunk_start, chunk_end).to_dict()
            except Exception:
                chunk_audio = None

        for c in suggest_chunk_clips(
            chunk_text,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            editorial_brief=editorial_brief,
            images=chunk_frames,
            audio_features=chunk_audio,
        ):
            try:
                start = float(c["start"])
                end = float(c["end"])
            except Exception:
                continue
            if start >= end:
                continue
            if start < chunk_start - 0.01 or end > chunk_end + 0.01:
                continue
            duration = end - start
            if duration < 12.0 or duration > 179.0:
                continue
            llm_clips.append(
                ClipCandidate(
                    start=start,
                    end=end,
                    text="",
                    coherence=0.5,
                    headline=str(c.get("headline", "")),
                )
            )
    return llm_clips


def _iter_transcript_chunks(
    segments: Sequence[TranscriptSegment],
    *,
    overlap_segments: int = 4,
) -> list[tuple[float, float, str]]:
    approx_chars = max(2000, int(settings.llm_num_ctx * 4 * 0.75))
    chunks: list[tuple[float, float, str]] = []
    buffer: list[TranscriptSegment] = []
    buffer_len = 0

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        line = f"[{seg.start:.2f}-{seg.end:.2f}] {text}"
        if buffer and buffer_len + len(line) + 1 > approx_chars:
            chunk_text = "\n".join(_segment_line(s) for s in buffer)
            chunks.append((buffer[0].start, buffer[-1].end, chunk_text))
            if overlap_segments > 0:
                buffer = buffer[-overlap_segments:]
                buffer_len = sum(len(_segment_line(s)) + 1 for s in buffer)
            else:
                buffer = []
                buffer_len = 0

        buffer.append(seg)
        buffer_len += len(line) + 1

    if buffer:
        chunk_text = "\n".join(_segment_line(s) for s in buffer)
        chunks.append((buffer[0].start, buffer[-1].end, chunk_text))

    return chunks


def _segment_line(segment: TranscriptSegment) -> str:
    return f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text.strip()}"


def _deduplicate_candidates(
    clips: list[ClipCandidate],
    *,
    overlap_threshold: float = 0.6,
) -> list[ClipCandidate]:
    """Remove near-duplicate clips based on temporal overlap.

    Two clips are considered duplicates if the ratio of their intersection
    to the shorter clip's duration exceeds ``overlap_threshold`` (default 60%).
    When duplicates are found, the longer clip is kept (more content).
    """
    if len(clips) <= 1:
        return clips

    sorted_clips = sorted(clips, key=lambda c: c.start)
    kept: list[ClipCandidate] = []

    for clip in sorted_clips:
        is_dup = False
        for idx, existing in enumerate(kept):
            overlap_start = max(clip.start, existing.start)
            overlap_end = min(clip.end, existing.end)
            overlap = max(0.0, overlap_end - overlap_start)
            shorter = min(clip.duration, existing.duration)
            if shorter > 0 and overlap / shorter >= overlap_threshold:
                # Keep the longer one
                if clip.duration > existing.duration:
                    kept[idx] = clip
                is_dup = True
                break
        if not is_dup:
            kept.append(clip)

    return kept


_LOW_CONFIDENCE_THRESHOLD = 0.5
_LOW_CONFIDENCE_MARKER = "[?]"


def _mark_low_confidence_words(
    segments: Sequence[TranscriptSegment],
) -> None:
    """Replace word text with a marker for words below the confidence threshold.

    This prevents the LLM from making editorial decisions based on
    incorrectly-transcribed words. Timestamps are preserved.
    """
    marked = 0
    total = 0
    for seg in segments:
        for word in seg.words:
            total += 1
            if word.probability is not None and word.probability < _LOW_CONFIDENCE_THRESHOLD:
                word.word = _LOW_CONFIDENCE_MARKER
                marked += 1
        # Rebuild segment text from (potentially modified) words
        if seg.words:
            seg.text = "".join(w.word for w in seg.words).strip()
    if marked:
        logger.info(
            "[pipeline] palabras de baja confianza marcadas",
            extra={"marcadas": marked, "total": total, "umbral": _LOW_CONFIDENCE_THRESHOLD},
        )
