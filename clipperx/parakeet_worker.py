"""Isolated, local-only Parakeet TDT inference; no Whisper/forced-alignment fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from importlib import import_module
from typing import Any


def timestamped_words(text: str, tokens: list[dict[str, Any]], *, offset: float,
                      duration: float) -> list[dict[str, Any]]:
    """Map native duration tokens onto decoded words, ignoring punctuation timing."""
    spans = list(re.finditer(r"\S+", text))
    bounds: list[list[float]] = [[] for _ in spans]
    cursor = index = 0
    for token in tokens:
        start, end = float(token["start"])+offset, float(token["end"])+offset
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= duration+.1):
            raise ValueError("Invalid native token time")
        for character in token["token"]:
            if character.isspace():
                continue
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != character:
                raise ValueError("Native token text does not match decoded text")
            while index < len(spans) and cursor >= spans[index].end():
                index += 1
            if character.isalnum():
                bounds[index].extend([start, min(end, duration)])
            cursor += 1
    if text[cursor:].strip():
        raise ValueError("Native token text does not cover decoded text")
    words: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        times = bounds[index]
        if not times:
            # Punctuation is preserved as text, never as seconds of imagined speech.
            times = next((b[:1] for b in bounds[index+1:] if b), [words[-1]["end"] if words else offset])
        words.append(dict(word=span.group(), start=min(times), end=max(times), probability=None))
    return words


def owned_words(words: list[dict[str, Any]], *, core_start: float,
                core_end: float) -> list[dict[str, Any]]:
    return [w for w in words if core_start <= (w["start"]+w["end"])/2 < core_end]


def words_to_segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] = []
    for word in words:
        if bucket and (word["start"]-bucket[-1]["end"] > .6 or word["end"]-bucket[0]["start"] > 8):
            segments.append(_segment(bucket))
            bucket = []
        bucket.append(word)
        if word["word"].endswith((".", "?", "!")):
            segments.append(_segment(bucket))
            bucket = []
    if bucket:
        segments.append(_segment(bucket))
    return segments


def _segment(words: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(start=words[0]["start"], end=max(w["end"] for w in words),
                text=" ".join(w["word"] for w in words), words=words)


def run(audio_path: Path, model_path: Path, *, chunk_seconds: float = 30.,
        context_seconds: float = 3.) -> dict[str, Any]:
    sf=import_module("soundfile")
    import torch
    import transformers
    from transformers import AutoModelForTDT, AutoProcessor

    if not 5 <= chunk_seconds <= 60 or not 0 <= context_seconds <= 5:
        raise ValueError("Bounded chunks/context required")
    if not torch.cuda.is_available():
        raise RuntimeError("Parakeet requires the isolated CUDA runtime; CPU fallback is disabled")
    torch.set_num_threads(4)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForTDT.from_pretrained(str(model_path), local_files_only=True,
        trust_remote_code=False, dtype=torch.float16).to("cuda").eval()
    load_seconds = time.perf_counter()-started
    torch.cuda.reset_peak_memory_stats()
    words: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    warnings: list[str] = []
    with sf.SoundFile(str(audio_path)) as audio:
        if audio.samplerate != 16000 or audio.channels != 1:
            raise ValueError("Expected prepared 16kHz mono audio")
        duration = audio.frames/audio.samplerate
        core_frames = round(chunk_seconds*16000)
        context_frames = round(context_seconds*16000)
        with torch.inference_mode():
            for core in range(0, audio.frames, core_frames):
                begin = max(0, core-context_frames)
                finish = min(audio.frames, core+core_frames+context_frames)
                audio.seek(begin)
                samples = audio.read(finish-begin, dtype="float32")
                before = time.perf_counter()
                inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
                inputs = inputs.to(model.device, dtype=model.dtype)
                output = model.generate(**inputs, return_dict_in_generate=True)
                decoded, native_tokens = processor.decode(output.sequences,
                    durations=output.durations, skip_special_tokens=True)
                text = decoded[0] if isinstance(decoded, list) else decoded
                candidates = timestamped_words(text, native_tokens[0], offset=begin/16000, duration=duration)
                selected = owned_words(candidates, core_start=core/16000,
                    core_end=min(duration,(core+core_frames)/16000))
                for word in selected:
                    if words and word["start"] < words[-1]["end"]:
                        previous = words[-1]
                        overlap = min(word["end"],previous["end"])-max(word["start"],previous["start"])
                        shortest = min(word["end"]-word["start"],previous["end"]-previous["start"])
                        if (word["word"] == previous["word"] and shortest > 0 and overlap/shortest >= .7):
                            continue
                        if word["start"] < previous["start"]:
                            raise ValueError("Native timestamps reverse at a chunk boundary")
                        warnings.append(f"Overlapping native words at {word['start']:.3f}")
                    words.append(word)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-before
                timings.append(dict(start=core/16000, end=min(duration,(core+core_frames)/16000), seconds=elapsed))
                print(f"Parakeet {min(duration,(core+core_frames)/16000):.1f}/{duration:.1f}s", flush=True)
    with audio_path.open("rb") as stream:
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
        audio_hash=digest.hexdigest()
    receipt = model_path/"clipperx-model.json"
    model_identity = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
    return dict(schema_version=1, backend="parakeet", native_timestamps=True,
        model=model_identity, audio_sha256=audio_hash, duration=duration,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0)),
        measurements=dict(load_seconds=load_seconds,total_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),chunks=timings),
        warnings=warnings,segments=words_to_segments(words))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--audio",type=Path,required=True)
    parser.add_argument("--model-path",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    payload=run(args.audio,args.model_path)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
