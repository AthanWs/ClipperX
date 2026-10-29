"""Native Parakeet transcription with explicitly selected legacy Whisper routes."""

from __future__ import annotations

import json
import hashlib
import shutil
import uuid
from dataclasses import dataclass, asdict
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Protocol, cast

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    import numpy as np


class _NativeWord(Protocol):
    start: float
    end: float


class _NativeSegment(Protocol):
    start: float
    end: float
    text: str
    words: Iterable[_NativeWord] | None


class _WhisperModel(Protocol):
    def transcribe(self, audio: str, *, language: str | None, beam_size: int,
                   word_timestamps: bool) -> tuple[Iterable[_NativeSegment], object]: ...


class _WhisperModelFactory(Protocol):
    def __call__(self, size: str, *, device: str, compute_type: str) -> _WhisperModel: ...


class _WhisperWord(TypedDict):
    start: NotRequired[float | None]
    end: NotRequired[float | None]
    word: NotRequired[str]
    text: NotRequired[str]
    score: NotRequired[float | None]


class _WhisperSegment(TypedDict):
    start: float
    end: float
    text: NotRequired[str]
    words: NotRequired[list[_WhisperWord]]


class _WhisperResult(TypedDict):
    segments: list[_WhisperSegment]
    language: NotRequired[str]


class _WhisperXModel(Protocol):
    def transcribe(self, audio: np.ndarray, *, language: str | None) -> _WhisperResult: ...


class _WhisperXLoader(Protocol):
    def __call__(self, size: str, *, device: str, compute_type: str) -> _WhisperXModel: ...


class _WhisperXAlignLoader(Protocol):
    def __call__(self, language: str, *, device: str) -> tuple[object, object]: ...


class _WhisperXAligner(Protocol):
    def __call__(self, segments: list[_WhisperSegment], model: object,
                 metadata: object, audio: np.ndarray, *, device: str,
                 return_char_alignments: bool) -> _WhisperResult: ...


class _WhisperXModule(Protocol):
    def load_model(self, size: str, *, device: str, compute_type: str) -> _WhisperXModel: ...
    def load_audio(self, path: str) -> np.ndarray: ...
    def load_align_model(self, language: str, *, device: str) -> tuple[object, object]: ...
    def align(self, segments: list[_WhisperSegment], model: object, metadata: object,
              audio: np.ndarray, *, device: str,
              return_char_alignments: bool) -> _WhisperResult: ...


WhisperModel = cast(
    _WhisperModelFactory, import_module("faster_whisper").WhisperModel
)

from .config import paths, settings  # noqa: E402 - preserve faster_whisper import order
from .logger import get_logger  # noqa: E402

logger = get_logger(__name__)

whisperx_align: _WhisperXAligner | None
load_align_model: _WhisperXAlignLoader | None
whisperx_load_audio: Callable[[str], np.ndarray] | None
load_whisperx_model: _WhisperXLoader | None

try:
    _whisperx = cast(_WhisperXModule, import_module("whisperx"))
    try:
        whisperx_align = _whisperx.align
        load_align_model = _whisperx.load_align_model
        whisperx_load_audio = _whisperx.load_audio
        load_whisperx_model = _whisperx.load_model
    except AttributeError as error:
        raise ImportError(f"WhisperX export unavailable: {error}") from error
    WHISPERX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"WhisperX no disponible: {e}. Se usará faster-whisper nativo sin alineación.")
    WHISPERX_AVAILABLE = False
    # Mock symbols to avoid NameErrors if logic path is taken (though we guard it)
    whisperx_align = None
    load_align_model = None
    whisperx_load_audio = None
    load_whisperx_model = None


@dataclass(slots=True)
class Segment:
    """Lightweight transcription segment."""

    start: float
    end: float
    text: str
    words: list[dict[str, Any]]


def load_model(model_size: str | None = None) -> _WhisperModel:
    size = model_size or settings.whisper_model
    # Detect if using a CT2 path or HF ID that needs int8
    compute_type = "int8" if "int8" in size or "turbo" in size else "float16"
    return WhisperModel(size, device="cuda", compute_type=compute_type)


def load_whisperx(model_size: str | None = None) -> _WhisperXModel:
    size = model_size or settings.whisper_model
    return cast(_WhisperXLoader, load_whisperx_model)(size, device="cuda", compute_type="float16")


def transcribe(
    audio_path: Path,
    *,
    language: str | None = "es",
    beam_size: int = 5,
    use_whisperx: bool = True,
) -> list[Segment]:
    logger.info("[transcribe] inicio", extra={"audio": str(audio_path), "whisperx": use_whisperx})
    if settings.asr_backend == "parakeet":
        return _transcribe_parakeet(audio_path, language=language)
    if settings.asr_backend != "whisper":
        raise ValueError("asr_backend must be parakeet or whisper")
    if use_whisperx and WHISPERX_AVAILABLE:
        try:
            return _transcribe_whisperx(audio_path, language=language, beam_size=beam_size)
        except Exception as e:
            logger.error(f"Fallo WhisperX: {e}. Cayendo a faster-whisper.", exc_info=True)

    # Fallback or default to native faster-whisper
    return _transcribe_native(audio_path, language=language, beam_size=beam_size)


def _transcribe_whisperx(
    audio_path: Path,
    *,
    language: str | None,
    beam_size: int,
) -> list[Segment]:
    logger.info("[transcribe] whisperx", extra={"audio": str(audio_path), "language": language})
    model = load_whisperx()
    logger.info("[transcribe] modelo cargado")
    audio = cast(Callable[[str], np.ndarray], whisperx_load_audio)(str(audio_path))
    logger.info("[transcribe] audio cargado")
    result = model.transcribe(
        audio,
        language=language,
    )
    logger.info("[transcribe] transcripción inicial completada")
    detected_lang = result.get("language", language or "es")
    align_model, metadata = cast(_WhisperXAlignLoader, load_align_model)(
        detected_lang, device="cuda"
    )
    logger.info("[transcribe] modelo de alineación cargado")
    aligned = cast(_WhisperXAligner, whisperx_align)(
        result["segments"],
        align_model,
        metadata,
        audio,
        device="cuda",
        return_char_alignments=False,
    )
    logger.info("[transcribe] alineación completada")

    segments = []
    for seg in aligned["segments"]:
        words = [
            {
                "start": word.get("start"),
                "end": word.get("end"),
                "word": word.get("word") or word.get("text", ""),
                "probability": word.get("score"),
            }
            for word in seg.get("words", [])
            if word.get("start") is not None and word.get("end") is not None
        ]
        segments.append(
            Segment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg.get("text", "").strip(),
                words=words,
            )
        )
    return segments


def _transcribe_parakeet(audio_path: Path, *, language: str | None) -> list[Segment]:
    from .parakeet import run_parakeet
    return [Segment(**row) for row in run_parakeet(audio_path,language=language)["segments"]]


def save_segments(segments: Iterable[Segment], output_path: Path) -> Path:
    data = [asdict(segment) for segment in segments]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[transcribe] guardado", extra={"segments": len(data), "output": str(output_path)})
    return output_path


def transcribe_to_json(
    audio_path: Path,
    *,
    output_dir: Path | None = None,
    output_name: str | None = None,
    language: str | None = "es",
) -> Path:
    if output_dir is None:
        output_dir = paths.transcripts_dir
    name = output_name or audio_path.stem
    output_path = output_dir / f"{name}.json"
    if settings.asr_backend == "parakeet":
        from .parakeet import run_parakeet
        payload=run_parakeet(audio_path,language=language)
        if output_path.exists():
            archive=output_path.parent/"transcript-history"/uuid.uuid4().hex
            archive.mkdir(parents=True,exist_ok=False)
            shutil.copy2(output_path,archive/output_path.name)
            old_metadata=output_path.with_suffix(".provenance.json")
            if old_metadata.is_file():
                shutil.copy2(old_metadata,archive/old_metadata.name)
        result=save_segments([Segment(**row) for row in payload["segments"]],output_path)
        metadata={key:value for key,value in payload.items() if key != "segments"}
        metadata["transcript_sha256"]=hashlib.sha256(result.read_bytes()).hexdigest()
        result.with_suffix(".provenance.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
        return result
    segments = transcribe(audio_path, language=language)
    return save_segments(segments, output_path)


def _transcribe_native(
    audio_path: Path,
    *,
    language: str | None,
    beam_size: int,
) -> list[Segment]:
    model = load_model()
    # faster-whisper returns a generator
    segments_gen, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
    )

    return [
        Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            words=[
                {
                    "start": word.start,
                    "end": word.end,
                    "word": getattr(word, "word", getattr(word, "text", "")),
                    "probability": getattr(word, "probability", None),
                }
                for word in (seg.words or [])
            ],
        )
        for seg in segments_gen
    ]
