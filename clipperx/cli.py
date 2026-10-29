"""Typer CLI entrypoint for ClipperX."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from .logger import configure_logging

if TYPE_CHECKING:
    from .structured_client import OpenAICompatibleStructuredClient


app = typer.Typer(help="Analiza videos largos y renderiza planes de clips verticales.")


def _confirmed_visual_client(
    model: str, context_window: int, confirm_inference: bool,
) -> OpenAICompatibleStructuredClient:
    """The confirmation gate precedes settings/credential and client construction."""
    if not confirm_inference:
        raise typer.BadParameter("Falta --confirm-inference; no se enviará nada al modelo")
    if not model.strip() or len(model) > 256:
        raise typer.BadParameter("--model must contain 1..256 characters")
    from .config import settings
    from .structured_client import OpenAICompatibleStructuredClient, StructuredModelConfig

    if settings.llm_provider.lower() == "disabled":
        raise typer.BadParameter("Configura LLM_PROVIDER antes de confirmar inferencia")
    ollama = settings.llm_provider.lower() == "ollama"
    return OpenAICompatibleStructuredClient(StructuredModelConfig(
        provider="ollama" if ollama else "cliproxy",
        base_url=(f"{settings.ollama_base_url.rstrip('/')}/v1"
                  if ollama else settings.openai_base_url),
        api_key="ollama" if ollama else settings.openai_api_key,
        model=model, context_window_tokens=context_window,
        max_output_tokens=min(4096, max(1024, context_window // 4)),
    ))


def _proxy_config_path() -> Path:
    """The existing optional local proxy lives beside this project."""
    return Path(__file__).resolve().parents[1].parent / "CLIProxyAPI" / "config.yaml"


def _automatic_client(model: str, context_window: int, confirm_inference: bool) -> OpenAICompatibleStructuredClient:
    """Use the selected provider or an already-configured local CLIProxyAPI.

    This is called only for an explicitly started automatic job. The existing
    key remains in memory and no provider configuration is created or changed.
    """
    if not confirm_inference:
        raise typer.BadParameter("Falta --confirm-inference; no se enviará nada al modelo")
    if not model.strip() or len(model) > 256 or context_window < 4096:
        raise typer.BadParameter("Modelo o contexto inválido")
    from .config import settings
    if settings.llm_provider.lower() != "disabled":
        return _confirmed_visual_client(model, context_window, True)
    try:
        import yaml
        with _proxy_config_path().open("rb") as handle:
            raw = handle.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("oversized proxy config")
        document = yaml.safe_load(raw)
        if not isinstance(document, dict):
            raise ValueError("invalid proxy config")
        port, keys = document.get("port"), document.get("api-keys")
        if (type(port) is not int or not 1 <= port <= 65535 or not isinstance(keys, list)
                or not keys or not isinstance(keys[0], str) or not keys[0]
                or len(keys[0]) > 4096 or any(ord(c) < 32 for c in keys[0])):
            raise ValueError("missing local proxy client configuration")
        key = keys[0]
    except Exception:
        raise typer.BadParameter(
            "No se pudo usar la configuración local de CLIProxyAPI; comprueba que esté configurado y abierto"
        ) from None
    from .structured_client import OpenAICompatibleStructuredClient, StructuredModelConfig
    return OpenAICompatibleStructuredClient(StructuredModelConfig(
        provider="cliproxy", base_url=f"http://127.0.0.1:{port}/v1", api_key=key,
        model=model, context_window_tokens=context_window,
        max_output_tokens=min(4096, max(1024, context_window // 4)),
    ))


@app.command("auto-edit")
def auto_edit_command(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    output_dir: Path = typer.Option(..., "--output-dir", resolve_path=True),
    model: str = typer.Option("gpt-6-luna(low)", "--model"),
    context_window: int = typer.Option(65536, "--context-window", min=4096),
    max_total_tokens: int = typer.Option(60000, "--max-total-tokens", min=1000),
    style_library: Path = typer.Option(Path("data/processed/style_library"), "--style-library"),
    transcript: Optional[Path] = typer.Option(None, "--transcript", exists=True, dir_okay=False),
    source_timeline: Optional[Path] = typer.Option(None, "--source-timeline", exists=True, dir_okay=False),
    encoder: str = typer.Option("h264_nvenc", "--encoder"),
    confirm_inference: bool = typer.Option(False, "--confirm-inference"),
) -> None:
    """Prepara, selecciona sin cuota global, dirige y exporta con reanudación."""
    # Admission before credentials, style loading or any source preparation.
    if not confirm_inference:
        raise typer.BadParameter("Falta --confirm-inference; no se enviará nada al modelo")
    if encoder not in {"h264_nvenc", "libx264"}:
        raise typer.BadParameter("--encoder debe ser h264_nvenc o libx264")
    client = _automatic_client(model, context_window, confirm_inference)
    import json
    from .autonomous_pipeline import run_autonomous_pipeline
    from .planner_engine import PlannerRunLimits
    from .render_contracts import RenderProfile
    from .style_library import StyleLibrary
    from .visual_director import VisualDirectorLimits

    result = run_autonomous_pipeline(video, output_dir, client=client, model_name=model,
        context_window_tokens=context_window, max_output_tokens=min(4096, max(1024, context_window // 4)),
        style_library=StyleLibrary(style_library), transcript_path=transcript,
        source_timeline_path=source_timeline,
        editorial_limits=PlannerRunLimits(target_clips=None, max_total_reported_tokens=max_total_tokens),
        visual_limits=VisualDirectorLimits(require_creator_overlays=True),
        profile=RenderProfile(video_encoder=encoder))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    typer.echo(f"Automatic workflow: {summary['status']}; {len(result.clips)} clips exportados")
    typer.echo(f"Run summary: {result.summary_path}")
    typer.echo(f"Resume state: {result.state_path}")
    for path in result.clips:
        typer.echo(f"Clip: {path}")
    if summary["status"] not in {"rendered_unreviewed", "empty"}:
        raise typer.Exit(2)


@app.command("build-style-library")
def build_style_library_command(
    directory: Path = typer.Argument(..., exists=True, file_okay=False,
                                     readable=True, resolve_path=True),
    output_dir: Path = typer.Option(..., "--output-dir", resolve_path=True),
    model: Optional[str] = typer.Option(None, "--model"),
    context_window: Optional[int] = typer.Option(None, "--context-window", min=4096),
    confirm_inference: bool = typer.Option(False, "--confirm-inference"),
) -> None:
    """Extrae evidencia local de MP4 editados; enriquece sólo con modelo explícito."""
    import json
    import re

    # An environment model never turns a local extraction into enrichment.
    client = None
    if model is not None:
        if not confirm_inference:
            raise typer.BadParameter("Falta --confirm-inference para enriquecer estilos")
        if context_window is None:
            raise typer.BadParameter("Enrichment requires explicit --context-window")
        client = _confirmed_visual_client(model, context_window, confirm_inference)
    elif context_window is not None:
        raise typer.BadParameter("--context-window requires explicit --model")

    from .local_paths import atomic_write_text, source_fingerprint
    from .style_analysis import analyze_edited_clip
    from .style_library import build_style_card

    sources = sorted((item for item in directory.iterdir()
                      if item.is_file() and item.suffix.lower() == ".mp4"), key=lambda p: p.name)
    if len(sources) > 1000:
        raise typer.BadParameter("At most 1000 immediate edited MP4 references per invocation")
    # Resolve all input references before extracting or sending any images.
    for source in sources:
        if not source.resolve().is_relative_to(directory.resolve()):
            raise typer.BadParameter("Edited references must remain within DIR")
    records = []
    new_calls = 0
    new_tokens = 0
    for source in sources:
        evidence = analyze_edited_clip(source, style_library_root=output_dir)
        if (not re.fullmatch(r"[0-9a-f]{16}", evidence.clip_fingerprint)
                or source_fingerprint(source) != evidence.clip_fingerprint):
            raise typer.BadParameter("Style evidence source fingerprint mismatch")
        evidence_path = (output_dir / evidence.evidence_path).resolve()
        if not evidence_path.is_relative_to(output_dir.resolve()):
            raise typer.BadParameter("Style evidence reference escapes output directory")
        record: dict[str, object] = {
            "source": str(source), "clip_fingerprint": evidence.clip_fingerprint,
            "evidence": str(evidence_path), "status": "local_only",
        }
        if client is not None and model is not None:
            result = build_style_card(evidence, library_root=output_dir, source_clip=source,
                                      client=client, model_id=model)
            record.update(status="unreviewed", card_id=result.card.card_id,
                          cache_hit=result.cache_hit)
            if result.cache_hit:
                record["historical_usage"] = result.usage
                record["new_usage"] = {}
            else:
                record["new_usage"] = result.usage
                new_calls += 1
                new_tokens += result.usage.get("total_tokens", 0)
        records.append(record)
    status = "unreviewed" if client is not None else "local_only"
    manifest_path = output_dir / "build_manifest.json"
    atomic_write_text(manifest_path, json.dumps({
        "manifest_version": 1, "status": status, "clips": records,
        "new_inference_calls": new_calls, "new_inference_tokens": new_tokens,
        "provenance": "historical_edited_clips_without_source_vod_association",
    }, ensure_ascii=False, indent=2) + "\n")
    typer.echo(f"Style build manifest: {manifest_path} ({status}; {len(records)} clips)")


@app.command("plan-visual")
def plan_visual_command(
    video: Path = typer.Argument(..., exists=True, dir_okay=False,
                                readable=True, resolve_path=True),
    editorial_plan: Path = typer.Argument(..., exists=True, dir_okay=False,
                                         readable=True, resolve_path=True),
    source_timeline: Path = typer.Argument(..., exists=True, dir_okay=False,
                                          readable=True, resolve_path=True),
    style_library: Path = typer.Option(..., "--style-library", exists=True, file_okay=False,
                                       readable=True, resolve_path=True),
    model: str = typer.Option(..., "--model"),
    context_window: int = typer.Option(..., "--context-window", min=4096),
    confirm_inference: bool = typer.Option(False, "--confirm-inference"),
) -> None:
    """Planifica visualmente clips v1 persistidos; no renderiza ni aprueba resultados."""
    client = _confirmed_visual_client(model, context_window, confirm_inference)
    import json

    from .planner_engine import run_visual_planner
    from .style_library import StyleLibrary
    from .visual_director import VisualDirectorLimits

    result = run_visual_planner(
        video, editorial_plan, source_timeline_path=source_timeline, client=client,
        style_library=StyleLibrary(style_library), limits=VisualDirectorLimits(),
    )
    summary = json.loads(result.run_summary.read_text(encoding="utf-8"))
    typer.echo(f"Visual plan manifest: {result.visual_plan_manifest} ({summary['status']})")
    typer.echo(f"Run summary: {result.run_summary}")


@app.command()
def run(
    video: str = typer.Argument(..., help="Ruta local o URL de YouTube del video"),
    output_dir: Optional[Path] = typer.Option(None, help="Directorio de salida para los clips"),
    face_crop: Optional[str] = typer.Option(None, help="Corte manual formateado como x,y,w,h"),
    face_stride: int = typer.Option(12, help="Frecuencia de muestreo en frames para la detección automática"),
    caption_template: Optional[str] = typer.Option(
        None, "-ct", "--caption-template", help="Template de subtítulos (default, hormozi, mrbeast, minimal, tiktok, neon, podcast)"
    ),
    min_virality: Optional[int] = typer.Option(
        None, "-mv", "--min-virality", help="Score mínimo de viralidad (0-100) para incluir un clip"
    ),
    preset: Optional[str] = typer.Option(
        None, "-pr", "--preset", help="Export preset (tiktok, reels, shorts, default)"
    ),
    provider: Optional[str] = typer.Option(
        None, "-p", "--provider", help="LLM provider override (auto, openai, google, anthropic, ollama, proxy)"
    ),
    broll: bool = typer.Option(False, "--broll/--no-broll", help="Habilitar inserción de B-roll via Pexels"),
) -> None:
    """Ejecuta el pipeline heredado sobre un video local o URL de YouTube."""

    from . import editing, pipeline
    from .youtube import is_youtube_url

    layout = None
    if face_crop:
        parts = [int(value) for value in face_crop.split(",")]
        if len(parts) != 4:
            raise typer.BadParameter("El formato debe ser x,y,w,h")
        layout = editing.LayoutConfig(face_crop=(parts[0], parts[1], parts[2], parts[3]))

    # Override provider if specified
    if provider:
        from .config import settings
        settings.llm_provider = provider

    # Configure console logging for CLI mode
    configure_logging(use_console=True)

    # Accept both local paths and YouTube URLs
    video_input: Path | str
    if is_youtube_url(video):
        video_input = video
    else:
        video_path = Path(video)
        if not video_path.exists():
            raise typer.BadParameter(f"El archivo no existe: {video}")
        video_input = video_path

    clips = pipeline.full_pipeline(
        video_path=video_input,
        output_dir=output_dir,
        layout_override=layout,
        face_stride=face_stride,
        caption_template=caption_template,
        min_virality=min_virality,
        export_preset=preset,
        enable_broll=broll,
    )
    typer.echo(f"Clips generados: {len(clips)}")
    for path in clips:
        typer.echo(f" - {path}")


@app.command()
def ui() -> None:
    """Abre la interfaz minimalista con controles de inicio/detención."""
    from .interface import run_ui

    # In UI mode, logging is configured inside the UI to avoid console interference
    run_ui()


@app.command()
def cancel() -> None:
    """Detiene el proceso en curso conservando checkpoints válidos."""
    from .runner import stop

    stop()


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json", help="Emite diagnóstico JSON")) -> None:
    """Comprueba el runtime local sin invocar ningún LLM."""
    from .diagnostics import collect_diagnostics

    report = collect_diagnostics()
    if json_output:
        import json

        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for check in report["checks"]:
            marker = "OK" if check["ok"] else "FAIL"
            typer.echo(f"  {marker:<4} {check['name']:<20} {check['detail']}")
    if not report["ok"]:
        raise typer.Exit(code=1)



@app.command("setup-face-models")
def setup_face_models_command() -> None:
    """Instala y valida explícitamente los modelos locales YuNet y SFace."""
    from .config import settings
    from .model_assets import ModelAssetError, ensure_face_models

    project_root = Path(__file__).resolve().parent.parent
    model_dir = Path(settings.face_model_dir)
    if not model_dir.is_absolute():
        model_dir = project_root / model_dir

    try:
        models = ensure_face_models(
            model_dir,
            allow_download=settings.allow_model_downloads,
        )
    except ModelAssetError as exc:
        typer.echo(f"No se pudieron preparar los modelos faciales: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"YuNet listo: {models['yunet']}")
    typer.echo(f"SFace listo: {models['sface']}")

@app.command("index")
def index_video(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    sample_interval: float = typer.Option(30.0, "--sample-interval", "-s", min=2.0),
    audio: bool = typer.Option(True, "--audio/--no-audio"),
    force: bool = typer.Option(False, "--force", help="Reconstruye artefactos existentes"),
) -> None:
    """Construye el timeline técnico sin invocar ningún LLM."""
    from .timeline import build_timeline

    configure_logging(use_console=True)
    result = build_timeline(video, sample_interval=sample_interval, include_audio=audio, force=force)
    typer.echo(f"Timeline: {result}")


@app.command("index-visual")
def index_visual_command(
    video: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    sparse_interval: float = typer.Option(
        10.0,
        "--sparse-interval",
        min=0.1,
        help="Intervalo base en segundos para el barrido completo",
    ),
    dense_interval: float = typer.Option(
        0.5,
        "--dense-interval",
        min=0.05,
        help="Intervalo en segundos dentro de ventanas visuales densas",
    ),
    allow_model_download: bool = typer.Option(
        False,
        "--allow-model-download",
        help="Permite descargar una sola vez los assets locales YuNet/SFace",
    ),
    force: bool = typer.Option(False, "--force", help="Reconstruye el artefacto"),
) -> None:
    """Construye el índice visual local sin proveedor, LLM ni VL."""
    from .visual_timeline import build_source_visual_timeline

    configure_logging(use_console=True)
    result = build_source_visual_timeline(
        video,
        sparse_interval=sparse_interval,
        dense_interval=dense_interval,
        allow_model_download=allow_model_download,
        force=force,
    )
    typer.echo(f"Source visual timeline: {result}")


_MAX_VISUAL_INSPECTION_SECONDS = 300.0


@app.command("inspect-visual")
def inspect_visual_command(
    video: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    start: float = typer.Option(..., "--start", min=0.0),
    end: float = typer.Option(..., "--end", min=0.0),
    sample_interval: float = typer.Option(
        0.5,
        "--sample-interval",
        min=0.05,
    ),
) -> None:
    """Decodifica y exporta una ventana visual aislada de hasta 300 segundos."""
    if end <= start:
        raise typer.BadParameter("--end must be greater than --start")
    if end - start > _MAX_VISUAL_INSPECTION_SECONDS:
        raise typer.BadParameter("Visual inspection is limited to 300 seconds")

    from .visual_timeline import build_source_visual_timeline_window

    try:
        output = build_source_visual_timeline_window(
            video,
            start=start,
            end=end,
            sample_interval=sample_interval,
            allow_model_download=False,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Source visual timeline subset: {output}")


@app.command("enroll-streamer-face")
def enroll_streamer_face_command(
    video: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    start: float = typer.Option(..., "--start", min=0.0),
    end: float = typer.Option(..., "--end", min=0.0),
    sample_interval: float = typer.Option(
        0.5,
        "--sample-interval",
        min=0.05,
    ),
    track_id: Optional[str] = typer.Option(
        None,
        "--track-id",
        help="Face track serializable que corresponde al streamer",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Nombre descriptivo almacenado en el perfil local",
    ),
) -> None:
    """Enrola la cara del streamer desde una ventana local aislada."""
    import json
    import math
    from datetime import datetime, timezone

    from .local_paths import (
        atomic_write_text_batch,
        global_streamer_identity_path,
        source_fingerprint,
        source_job_root,
    )
    from .visual_contracts import SourceVisualTimeline
    from .visual_timeline import build_source_visual_timeline_window

    if end <= start:
        raise typer.BadParameter("--end must be greater than --start")

    try:
        timeline_path = build_source_visual_timeline_window(
            video,
            start=start,
            end=end,
            sample_interval=sample_interval,
            allow_model_download=False,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    timeline = SourceVisualTimeline.model_validate_json(
        timeline_path.read_text(encoding="utf-8")
    )
    tracks = timeline.face_tracks
    selected = None
    if track_id is not None:
        selected = next((track for track in tracks if track.track_id == track_id), None)
        if selected is None:
            available = ", ".join(track.track_id for track in tracks) or "none"
            raise typer.BadParameter(
                f"Unknown --track-id {track_id!r}; available tracks: {available}"
            )
    elif len(tracks) == 1:
        selected = tracks[0]
    elif not tracks:
        raise typer.BadParameter("No serializable face tracks found in the range")
    else:
        available = ", ".join(track.track_id for track in tracks)
        raise typer.BadParameter(
            f"Multiple face tracks found ({available}); pass --track-id"
        )

    embedding = [float(value) for value in selected.embedding_centroid]
    norm = math.sqrt(sum(value * value for value in embedding))
    if not math.isfinite(norm) or norm <= 1e-6:
        raise typer.BadParameter("Selected face track has an invalid embedding")
    normalized = [value / norm for value in embedding]
    if not all(math.isfinite(value) for value in normalized):
        raise typer.BadParameter("Selected face track has a non-finite embedding")

    resolved_video = video.resolve()
    payload = {
        "contract_version": 1,
        "profile_type": "streamer_face",
        "name": name.strip() if name and name.strip() else resolved_video.stem,
        "embedding": normalized,
        "track_id": selected.track_id,
        "source_fingerprint": source_fingerprint(resolved_video),
        "source_reference": str(resolved_video),
        "source_range": {"start": float(start), "end": float(end)},
        "observations": selected.observation_count,
        "model_hashes": dict(timeline.model_hashes),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    global_profile = global_streamer_identity_path()
    job_profile = source_job_root(resolved_video) / "streamer_identity.json"
    atomic_write_text_batch((global_profile, job_profile), serialized)
    typer.echo(f"Global streamer face profile: {global_profile}")
    typer.echo(f"Source streamer face profile: {job_profile}")


@app.command("export-visual-schemas")
def export_visual_schemas_command(
    output_dir: Path = typer.Option(..., "--output-dir"),
) -> None:
    """Exporta contratos visuales JSON sin proveedor ni inferencia."""
    import json

    from .local_paths import atomic_write_text
    from .visual_contracts import SourceVisualTimeline, VisualCompositionPlan

    for model in (SourceVisualTimeline, VisualCompositionPlan):
        output = output_dir / f"{model.__name__}.schema.json"
        atomic_write_text(
            output,
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2)
            + "\n",
        )
        typer.echo(f"Schema: {output}")


@app.command("inspect-window")
def inspect_window_command(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    start: float = typer.Option(..., "--start", min=0.0),
    end: float = typer.Option(..., "--end", min=0.0),
    sample_interval: float = typer.Option(1.0, "--sample-interval", "-s", min=0.25),
    audio: bool = typer.Option(True, "--audio/--no-audio"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Inspecciona con detalle una ventana sin invocar ningún LLM."""
    from .timeline import inspect_window

    configure_logging(use_console=True)
    result = inspect_window(
        video,
        start=start,
        end=end,
        sample_interval=sample_interval,
        include_audio=audio,
        force=force,
    )
    typer.echo(f"Window probe: {result}")


@app.command("transcribe")
def transcribe_video(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    language: Optional[str] = typer.Option("es", "--language", "-l"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Transcribe y alinea el VOD sin utilizar un LLM."""
    from .stages import ensure_transcript

    configure_logging(use_console=True)
    _, result = ensure_transcript(video, language=language, force=force)
    typer.echo(f"Transcript: {result}")


@app.command("diarize")
def diarize_video_command(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    min_speakers: Optional[int] = typer.Option(None, min=1),
    max_speakers: Optional[int] = typer.Option(None, min=1),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Diariza speakers con pyannote Community-1; no invoca un LLM."""
    from .diarization import diarize_video

    configure_logging(use_console=True)
    diarization_path, transcript_path = diarize_video(
        video,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        force=force,
    )
    typer.echo(f"Diarization: {diarization_path}")
    typer.echo(f"Speaker transcript: {transcript_path}")


@app.command("set-streamer-speaker")
def set_streamer_speaker_command(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    speaker: str = typer.Argument(..., help="Etiqueta del VOD, por ejemplo SPEAKER_00"),
    name: Optional[str] = typer.Option(None, "--name", help="Nombre visible del streamer"),
) -> None:
    """Confirma qué speaker diarizado corresponde al streamer en este VOD."""
    from .speaker_identity import confirm_streamer_speaker

    result = confirm_streamer_speaker(video, speaker, display_name=name)
    typer.echo(f"Streamer profile: {result}")


@app.command("enroll-streamer-voice")
def enroll_streamer_voice_command(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    start: float = typer.Option(..., "--start", min=0.0),
    end: float = typer.Option(..., "--end", min=0.0),
    name: str = typer.Option(..., "--name"),
) -> None:
    """Crea un voiceprint global a partir de un rango limpio confirmado."""
    from .voiceprint import enroll_streamer_voice

    result = enroll_streamer_voice(video, start=start, end=end, name=name)
    typer.echo(f"Voiceprint: {result}")


@app.command("match-streamer-voice")
def match_streamer_voice_command(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    profile: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Compara speakers del VOD contra un voiceprint global."""
    from .voiceprint import match_streamer_voice

    result = match_streamer_voice(video, profile)
    typer.echo(f"Voice matches: {result}")


@app.command("prepare-batches")
def prepare_batches_command(
    transcript: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    model: str = typer.Option(..., "--model", help="Modelo objetivo para contar tokens"),
    context_window: int = typer.Option(..., "--context-window", min=4096),
    output_dir: Path = typer.Option(..., "--output-dir"),
    reserved_output: Optional[int] = typer.Option(None, "--reserved-output", min=512),
    utilization: float = typer.Option(0.85, min=0.5, max=0.95),
    overlap_tokens: int = typer.Option(1000, min=0),
) -> None:
    """Divide un transcript con cursor y presupuesto real; no invoca un LLM."""
    from .planner import prepare_transcript_batches

    result = prepare_transcript_batches(
        transcript,
        output_dir,
        model=model,
        context_window_tokens=context_window,
        reserved_output_tokens=reserved_output,
        utilization=utilization,
        overlap_tokens=overlap_tokens,
    )
    typer.echo(f"Batch manifest: {result}")


@app.command("planner-schemas")
def planner_schemas_command(
    output_dir: Path = typer.Option(..., "--output-dir"),
) -> None:
    """Exporta los contratos JSON del planner sin invocar un LLM."""
    from .planner_contracts import export_contract_schemas

    for result in export_contract_schemas(output_dir):
        typer.echo(f"Schema: {result}")


@app.command("collect-evidence")
def collect_evidence_command(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    request_json: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Ejecuta un EvidenceRequest JSON sin invocar ningún LLM."""
    import json

    from .evidence import collect_evidence
    from .planner_contracts import EvidenceRequest

    request = EvidenceRequest.model_validate_json(request_json.read_text(encoding="utf-8"))
    result = collect_evidence(video, request, force=force)
    typer.echo(json.dumps({"evidence_bundle": str(result)}, ensure_ascii=False))


@app.command("render-plan", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def render_plan_command() -> None:
    """Deprecated: use render-visual-plan for v2 or render-legacy-plan for v1."""
    raise typer.BadParameter("render-plan está obsoleto. Usa render-visual-plan para v2 o render-legacy-plan para v1")


@app.command("render-visual-plan")
def render_visual_plan_command(
    video: Path = typer.Argument(...),
    plan: Path = typer.Argument(...),
    output_dir: Path = typer.Option(..., "--output-dir"),
    transcript: Optional[Path] = typer.Option(None, "--transcript"),
    encoder: str = typer.Option("h264_nvenc", "--encoder"),
) -> None:
    """Renderiza un plan visual v2 explícito."""
    from .plan_render import render_visual_composition_plan
    from .render_assets import default_render_assets
    from .render_contracts import RenderProfile
    from .render_sfx_gate import validate_default_sfx_render_plan
    if encoder not in ("h264_nvenc", "libx264"):
        raise typer.BadParameter("--encoder debe ser h264_nvenc o libx264")
    try:
        validate_default_sfx_render_plan(video, plan)
    except (OSError, ValueError):
        raise typer.BadParameter(
            "El plan v2 o su autorización SFX no son válidos; usa el visual_plan.json READY "
            "original, su fuente y el catálogo aprobado vigente"
        ) from None
    outputs = render_visual_composition_plan(video, plan, output_dir, transcript_path=transcript,
        assets=default_render_assets(), profile=RenderProfile.model_validate({"video_encoder": encoder}))
    for output in outputs:
        typer.echo(f"Clip: {output}")


@app.command("render-legacy-plan")
def render_legacy_plan_command(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    plan: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    output_dir: Path = typer.Option(..., "--output-dir"),
    transcript: Optional[Path] = typer.Option(None, "--transcript"),
    preset: str = typer.Option("default", "--preset"),
    watermark: Optional[str] = typer.Option(None, "--watermark"),
) -> None:
    """Renderiza un GlobalCompositionResponse validado a archivos MP4."""
    from .plan_render import load_transcript, render_legacy_composition_plan

    outputs = render_legacy_composition_plan(
        video,
        plan,
        output_dir,
        transcripts=load_transcript(transcript),
        preset=preset,
        watermark_text=watermark,
    )
    for output in outputs:
        typer.echo(f"Clip: {output}")


@app.command("plan-with-model")
def plan_with_model_command(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    transcript: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    model: str = typer.Option(..., "--model"),
    context_window: int = typer.Option(..., "--context-window", min=4096),
    max_output: Optional[int] = typer.Option(None, "--max-output", min=512),
    target_clips: int = typer.Option(3, "--target-clips", min=1, max=20),
    max_candidates: int = typer.Option(6, "--max-candidates", min=1, max=40),
    max_initial_requests: int = typer.Option(6, "--max-initial-requests", min=1, max=40),
    max_boundary_calls: int = typer.Option(8, "--max-boundary-calls", min=1, max=40),
    max_images: int = typer.Option(4, "--max-images", min=1, max=8),
    max_total_tokens: int = typer.Option(60_000, "--max-total-tokens", min=1000),
    confirm_inference: bool = typer.Option(False, "--confirm-inference"),
    supports_json_schema: bool = typer.Option(True, "--json-schema/--json-object"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Ejecuta el planner agente solo con confirmación explícita de inferencia."""
    if not confirm_inference:
        raise typer.BadParameter("Falta --confirm-inference; no se enviará nada al modelo")
    from .config import settings
    from .planner_engine import PlannerRunLimits, run_editorial_planner
    from .structured_client import OpenAICompatibleStructuredClient, StructuredModelConfig

    if settings.llm_provider.lower() == "disabled":
        raise typer.BadParameter("Configura LLM_PROVIDER antes de confirmar inferencia")
    effective_max_output = max_output or min(4096, max(1024, context_window // 4))
    client = OpenAICompatibleStructuredClient(
        StructuredModelConfig(
            provider="ollama" if settings.llm_provider.lower() == "ollama" else "cliproxy",
            base_url=(
                f"{settings.ollama_base_url.rstrip('/')}/v1"
                if settings.llm_provider.lower() == "ollama"
                else settings.openai_base_url
            ),
            api_key="ollama" if settings.llm_provider.lower() == "ollama" else settings.openai_api_key,
            model=model,
            context_window_tokens=context_window,
            max_output_tokens=effective_max_output,
            supports_json_schema=supports_json_schema,
        )
    )
    result = run_editorial_planner(
        video,
        transcript,
        discovery_client=client,
        evidence_client=client,
        model_name=model,
        context_window_tokens=context_window,
        max_output_tokens=effective_max_output,
        limits=PlannerRunLimits(
            target_clips=target_clips,
            max_candidates=max_candidates,
            max_initial_evidence_requests=max_initial_requests,
            max_boundary_calls=max_boundary_calls,
            max_images_per_request=max_images,
            max_total_reported_tokens=max_total_tokens,
        ),
        force=force,
    )
    typer.echo(f"Composition plan: {result}")


@app.command()
def templates() -> None:
    """Lista los templates de subtítulos disponibles."""
    from .caption_templates import list_templates, get_template

    for name in list_templates():
        tpl = get_template(name)
        typer.echo(
            f"  {name:<12} font={tpl.font_name}, animation={tpl.animation}, "
            f"size={tpl.font_size}"
        )


@app.command()
def providers() -> None:
    """Lista los LLM providers disponibles y cuál está activo."""
    from .llm import list_providers

    for p in list_providers():
        status = f" [{p['active']}]" if p["active"] else ""
        avail = "OK" if p["available"] == "yes" else "--"
        typer.echo(f"  {p['name']:<12} {avail}{status}")


@app.command()
def download(
    url: str = typer.Argument(..., help="URL de YouTube a descargar"),
    output_dir: Optional[Path] = typer.Option(None, help="Directorio de salida"),
) -> None:
    """Descarga un video de YouTube sin procesarlo."""
    from .youtube import is_youtube_url, download_video
    from .config import paths

    configure_logging(use_console=True)

    if not is_youtube_url(url):
        raise typer.BadParameter("La URL proporcionada no es de YouTube")

    target = output_dir or paths.downloads_dir
    result = download_video(url, target)
    typer.echo(f"Descargado: {result}")


@app.command()
def trim(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    start: float = typer.Option(..., "--start", "-s", help="Segundo de inicio"),
    end: float = typer.Option(..., "--end", "-e", help="Segundo de fin"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Archivo de salida"),
) -> None:
    """Recorta un clip entre los segundos indicados."""
    from .clip_editor import trim_clip

    out = output or clip.with_stem(f"{clip.stem}_trimmed")
    result = trim_clip(clip, out, start, end)
    typer.echo(f"Trimmed: {result}")


@app.command()
def split(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    at: float = typer.Option(..., "--at", help="Segundo donde dividir"),
) -> None:
    """Divide un clip en dos partes en el segundo indicado."""
    from .clip_editor import split_clip

    part1, part2 = split_clip(clip, clip.parent, at)
    typer.echo(f"Part 1: {part1}")
    typer.echo(f"Part 2: {part2}")


@app.command()
def merge(
    clips: list[Path] = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    output: Path = typer.Option(..., "-o", "--output", help="Archivo de salida"),
) -> None:
    """Concatena múltiples clips en uno solo."""
    from .clip_editor import merge_clips

    result = merge_clips(clips, output)
    typer.echo(f"Merged: {result}")


@app.command()
def speed(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    factor: float = typer.Option(..., "--factor", "-f", help="Factor de velocidad (1.5 = 50%% más rápido)"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Archivo de salida"),
) -> None:
    """Ajusta la velocidad de reproducción de un clip."""
    from .clip_editor import adjust_speed

    out = output or clip.with_stem(f"{clip.stem}_speed{factor}")
    result = adjust_speed(clip, out, factor)
    typer.echo(f"Speed adjusted: {result}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
