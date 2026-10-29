"""Local-only Nemotron 3 diarization in the isolated NVIDIA speech runtime."""

from __future__ import annotations

import argparse
from importlib import import_module
import importlib.metadata
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator


def validate_speaker_bounds(minimum: int | None, maximum: int | None) -> None:
    for value in (minimum, maximum):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                  or not 1 <= value <= 8):
            raise ValueError("Los límites de hablantes (speakers) deben ser enteros entre 1 y 8")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("El mínimo de hablantes (speakers) supera el máximo")
    if minimum not in (None, 1) or maximum not in (None, 8):
        raise ValueError(
            "Nemotron no permite forzar el número de hablantes (unsupported speaker bounds); "
            "omite min/max para usar su detección nativa de hasta 8 voces"
        )


def normalize_turns(segments: list[dict[str, Any]], *, duration: float) -> list[dict[str, Any]]:
    """Keep overlapping activity; only round the model's final frame to EOF."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("La duración de los turns debe ser finita y positiva")
    turns: list[dict[str, Any]] = []
    for segment in segments:
        try:
            start, end, speaker = segment["Start"], segment["End"], segment["Speaker"]
            if isinstance(start, bool) or isinstance(end, bool):
                raise ValueError("boolean timestamp")
            start, end = float(start), float(end)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Nemotron produjo un turn/speaker inválido") from exc
        if not isinstance(speaker, Integral) or isinstance(speaker, bool):
            raise ValueError("Nemotron produjo un turn/speaker inválido")
        speaker_index = int(speaker)
        if not 0 <= speaker_index < 8:
            raise ValueError("Nemotron produjo un turn/speaker inválido")
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0
                or start >= end or start >= duration or end > duration + 0.010001):
            raise ValueError("Nemotron produjo un turn fuera del audio o no finito")
        turns.append({"start": start, "end": min(end, duration),
                      "speaker": f"SPEAKER_{speaker_index:02d}"})
    return sorted(turns, key=lambda turn: (turn["start"], turn["end"], turn["speaker"]))


def _streaming_inputs(audio: Any, processor: Any) -> Iterator[Any]:
    import numpy as np

    total = audio.frames
    first_count = int(processor.num_samples_first_audio_chunk)
    count = int(processor.num_samples_per_audio_chunk)
    step = int(processor.num_mel_frames_per_step)
    if min(first_count, count, step) <= 0:
        raise ValueError("El processor Nemotron no declara bloques de streaming válidos")
    start, mel_frame_idx, first = 0, 0, True
    previous_start = -1
    while start < total:
        if start <= previous_start:
            raise ValueError("El processor Nemotron no avanza por el audio")
        size = min(first_count if first else count, total - start)
        audio.seek(start)
        samples = audio.read(size, dtype="float32", always_2d=False)
        if len(samples) != size or not np.isfinite(samples).all():
            raise ValueError("Audio Nemotron incompleto o no finito")
        last = start + size >= total
        yield processor(samples, sampling_rate=16000, is_streaming=True,
                        is_first_audio_chunk=first, is_last_audio_chunk=last)
        if last:
            return
        previous_start, first = start, False
        mel_frame_idx += step
        start = int(processor.audio_chunk_start(mel_frame_idx))
    raise ValueError("El processor Nemotron omitió el último bloque de audio")


def run_diarization(
    audio_path: Path,
    output_path: Path,
    *,
    model_path: Path,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str = "cuda",
) -> Path:
    validate_speaker_bounds(min_speakers, max_speakers)
    model_path = model_path.resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise RuntimeError(f"Falta el modelo Nemotron local completo: {model_path}")
    sf = import_module("soundfile")
    import torch

    with sf.SoundFile(str(audio_path)) as audio:
        if audio.samplerate != 16000 or audio.channels != 1 or audio.frames <= 0:
            raise ValueError("Nemotron requiere audio 16 kHz mono con duración positiva")
        duration = audio.frames / audio.samplerate
        if device not in ("cuda", "cpu"):
            raise ValueError("Dispositivo Nemotron desconocido")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Nemotron requiere CUDA en el runtime NVIDIA speech")
        try:
            from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("El runtime necesita Transformers con soporte Nemotron 3") from exc
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
        if processor.feature_extractor.sampling_rate != 16000:
            raise ValueError("El processor Nemotron no corresponde al modelo de 16 kHz")
        processor.set_streaming_mode("low_latency")
        model = AutoModelForAudioFrameClassification.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False, device_map=device,
        )
        model.eval()
        speaker_cache, logits = None, []
        with torch.inference_mode():
            for inputs in _streaming_inputs(audio, processor):
                inputs = inputs.to(model.device, dtype=model.dtype)
                outputs = model(**inputs, speaker_cache=speaker_cache)
                # Keep only bounded input/state on GPU. Four hours of 10 ms x 8
                # float32 scores occupy about 46 MB on CPU, not the full waveform.
                logits.append(outputs.logits.detach().to(device="cpu", dtype=torch.float32))
                speaker_cache = outputs.speaker_cache
                del outputs, inputs
        combined = torch.cat(logits, dim=1)
        del logits, speaker_cache, model
        segments = processor.extract_speaker_dict(combined)[0]
        turns = normalize_turns(segments, duration=duration)
    payload = {
        "version": 1, "backend": "nemotron", "model": str(model_path),
        "duration": duration, "turns": turns, "exclusive_turns": [],
        "speaker_identity": "unresolved", "speaker_limit": 8,
        "runtime": {
            "device": device, "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "streaming_mode": "low_latency", "audio_io": "soundfile_bounded_windows",
            "logits_storage": "cpu_float32", "sample_rate": 16000,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    arguments = parser.parse_args()
    run_diarization(arguments.audio, arguments.output, model_path=arguments.model,
                    min_speakers=arguments.min_speakers, max_speakers=arguments.max_speakers)


if __name__ == "__main__":
    main()
