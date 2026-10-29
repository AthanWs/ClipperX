"""Isolated pyannote.audio 4.x worker for Community-1."""

from __future__ import annotations

import argparse
from importlib import import_module
import importlib.metadata
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from ._media_types import load_soundfile

if TYPE_CHECKING:
    import numpy as np


class _Segment(Protocol):
    start: float
    end: float


class _SegmentFactory(Protocol):
    def __call__(self, start: float, end: float) -> _Segment: ...


class _DiarizationOutput(Protocol):
    speaker_diarization: object
    exclusive_speaker_diarization: object | None


class _Pipeline(Protocol):
    def to(self, device: object) -> object: ...
    def __call__(self, audio: dict[str, Any], **options: int) -> _DiarizationOutput: ...


class _PipelineFactory(Protocol):
    def from_pretrained(self, name: str, *, token: str) -> _Pipeline: ...


class _EmbeddingModel(Protocol):
    def to(self, device: object) -> object: ...


class _EmbeddingModelFactory(Protocol):
    def from_pretrained(self, name: str, *, token: str | None) -> _EmbeddingModel | None: ...


class _EmbeddingInference(Protocol):
    def crop(self, audio: dict[str, Any], segment: _Segment) -> np.ndarray: ...


class _InferenceFactory(Protocol):
    def __call__(self, model: _EmbeddingModel, *, window: str,
                 device: object) -> _EmbeddingInference: ...


def runtime_report() -> dict[str, Any]:
    soundfile = load_soundfile()
    import torch

    version = importlib.metadata.version("pyannote.audio")
    major = int(version.split(".", 1)[0])
    return {
        "ok": major >= 4 and bool(torch.cuda.is_available()),
        "pyannote_audio": version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "soundfile": soundfile.__version__,
        "audio_io": "preloaded_soundfile",
    }


def run_diarization(
    audio_path: Path,
    output_path: Path,
    *,
    model_name: str,
    min_speakers: int | None,
    max_speakers: int | None,
) -> Path:
    import torch
    Pipeline = cast(_PipelineFactory, import_module("pyannote.audio").Pipeline)

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN no está configurado")
    pipeline = Pipeline.from_pretrained(model_name, token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    options: dict[str, int] = {}
    if min_speakers is not None:
        options["min_speakers"] = min_speakers
    if max_speakers is not None:
        options["max_speakers"] = max_speakers
    output = pipeline(_load_audio(audio_path), **options)
    regular = _annotation_turns(output.speaker_diarization)
    exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
    exclusive = _annotation_turns(exclusive_annotation) if exclusive_annotation is not None else []
    payload = {
        "version": 1,
        "model": model_name,
        "runtime": runtime_report(),
        "turns": regular,
        "exclusive_turns": exclusive,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def run_embedding(
    audio_path: Path,
    output_path: Path,
    *,
    model_name: str,
    start: float,
    end: float,
) -> Path:
    Segment = cast(_SegmentFactory, import_module("pyannote.core").Segment)

    if end <= start:
        raise ValueError("El rango de voiceprint debe tener duración positiva")
    inference = _embedding_inference(model_name)
    embedding = _normalized_embedding(inference.crop(_load_audio(audio_path), Segment(start, end)))
    _write_payload(
        output_path,
        {
            "version": 1,
            "model": model_name,
            "range": {"start": start, "end": end},
            "embedding": embedding.tolist(),
        },
    )
    return output_path


def match_voiceprint(
    audio_path: Path,
    diarization_path: Path,
    profile_path: Path,
    output_path: Path,
    *,
    model_name: str,
) -> Path:
    import numpy as np
    Segment = cast(_SegmentFactory, import_module("pyannote.core").Segment)

    diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
    turns = diarization.get("exclusive_turns") or diarization.get("turns") or []
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    reference = np.asarray(profile["embedding"], dtype=np.float32)
    reference /= max(float(np.linalg.norm(reference)), 1e-9)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for turn in turns:
        if float(turn["end"]) - float(turn["start"]) >= 1.5:
            grouped.setdefault(str(turn["speaker"]), []).append(turn)

    audio = _load_audio(audio_path)
    inference = _embedding_inference(model_name)
    scores: list[dict[str, Any]] = []
    for speaker, speaker_turns in grouped.items():
        selected = sorted(
            speaker_turns,
            key=lambda turn: float(turn["end"]) - float(turn["start"]),
            reverse=True,
        )[:8]
        embeddings = [
            _normalized_embedding(
                inference.crop(audio, Segment(float(turn["start"]), float(turn["end"])))
            )
            for turn in selected
        ]
        if not embeddings:
            continue
        centroid = _normalized_embedding(np.stack(embeddings).mean(axis=0))
        score = float(np.dot(reference, centroid))
        scores.append(
            {
                "speaker": speaker,
                "cosine_similarity": round(score, 6),
                "turns_used": len(embeddings),
                "seconds_used": round(
                    sum(float(turn["end"]) - float(turn["start"]) for turn in selected),
                    3,
                ),
            }
        )
    scores.sort(key=lambda item: item["cosine_similarity"], reverse=True)
    _write_payload(
        output_path,
        {
            "version": 1,
            "model": model_name,
            "profile": str(profile_path),
            "scores": scores,
            "best_candidate": scores[0]["speaker"] if scores else None,
            "score_margin": (
                round(scores[0]["cosine_similarity"] - scores[1]["cosine_similarity"], 6)
                if len(scores) > 1
                else None
            ),
        },
    )
    return output_path


def _embedding_inference(model_name: str) -> _EmbeddingInference:
    import torch
    audio_module = import_module("pyannote.audio")
    Inference = cast(_InferenceFactory, audio_module.Inference)
    Model = cast(_EmbeddingModelFactory, audio_module.Model)

    token = os.environ.get("HF_TOKEN", "").strip() or None
    model = Model.from_pretrained(model_name, token=token)
    if model is None:
        raise RuntimeError(f"No se pudo cargar el modelo de embedding: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return Inference(model, window="whole", device=device)


def _normalized_embedding(value: Any) -> np.ndarray:
    import numpy as np

    array = np.asarray(value, dtype=np.float32).squeeze()
    if array.ndim > 1:
        array = array.mean(axis=0)
    norm = max(float(np.linalg.norm(array)), 1e-9)
    return array / norm


def _load_audio(audio_path: Path) -> dict[str, Any]:
    sf = load_soundfile()
    import torch

    data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    return {"waveform": waveform, "sample_rate": sample_rate}


def _write_payload(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)


def _annotation_turns(annotation: Any) -> list[dict[str, Any]]:
    try:
        return [
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
            for turn, speaker in annotation
        ]
    except (TypeError, ValueError):
        return [
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--audio", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--min-speakers", type=int)
    run_parser.add_argument("--max-speakers", type=int)
    embed_parser = subparsers.add_parser("embed")
    embed_parser.add_argument("--audio", type=Path, required=True)
    embed_parser.add_argument("--output", type=Path, required=True)
    embed_parser.add_argument("--model", required=True)
    embed_parser.add_argument("--start", type=float, required=True)
    embed_parser.add_argument("--end", type=float, required=True)
    match_parser = subparsers.add_parser("match")
    match_parser.add_argument("--audio", type=Path, required=True)
    match_parser.add_argument("--diarization", type=Path, required=True)
    match_parser.add_argument("--profile", type=Path, required=True)
    match_parser.add_argument("--output", type=Path, required=True)
    match_parser.add_argument("--model", required=True)
    arguments = parser.parse_args()
    if arguments.command == "doctor":
        report = runtime_report()
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit(0 if report["ok"] else 1)
    if arguments.command == "embed":
        run_embedding(
            arguments.audio,
            arguments.output,
            model_name=arguments.model,
            start=arguments.start,
            end=arguments.end,
        )
        return
    if arguments.command == "match":
        match_voiceprint(
            arguments.audio,
            arguments.diarization,
            arguments.profile,
            arguments.output,
            model_name=arguments.model,
        )
        return
    run_diarization(
        arguments.audio,
        arguments.output,
        model_name=arguments.model,
        min_speakers=arguments.min_speakers,
        max_speakers=arguments.max_speakers,
    )


if __name__ == "__main__":
    main()
