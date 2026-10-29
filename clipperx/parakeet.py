"""Application boundary for the isolated, local-only native-timestamp worker."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

from .config import paths, settings
from .speech_process import run_worker


def model_path() -> Path:
    path=Path(settings.parakeet_model_path)
    return path if path.is_absolute() else paths.base_dir/path


def audio_hash(path: Path) -> str:
    with path.open("rb") as stream:
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
        return digest.hexdigest()


def request_signature(language: str | None) -> str:
    receipt=model_path()/"clipperx-model.json"
    identity=receipt.read_text(encoding="utf-8") if receipt.is_file() else "missing"
    weights=model_path()/"model.safetensors"
    weight_stat=(weights.stat().st_size,weights.stat().st_mtime_ns) if weights.is_file() else None
    worker_hash=hashlib.sha256(Path(__file__).with_name("parakeet_worker.py").read_bytes()).hexdigest()
    payload=dict(backend="parakeet",worker_version=1,worker_hash=worker_hash,weights=weight_stat,model_path=str(model_path().resolve()),
        identity=identity,language=language,native_timestamps=True,chunk_seconds=30,context_seconds=3)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def transcript_is_current(transcript: Path, audio: Path, *, language: str | None) -> bool:
    try:
        payload=json.loads(transcript.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        return (payload.get("backend")=="parakeet" and payload.get("native_timestamps") is True
            and payload.get("request_signature")==request_signature(language)
            and payload.get("audio_sha256")==audio_hash(audio)
            and payload.get("transcript_sha256")==audio_hash(transcript))
    except (OSError,ValueError,TypeError,AttributeError):
        return False


def run_parakeet(audio: Path, *, language: str | None = "es") -> dict[str, Any]:
    sf=import_module("soundfile")
    if language not in {None,"es"}:
        raise ValueError("This validated Parakeet route currently supports Spanish/auto only")
    python=Path(settings.nvidia_speech_python) if settings.nvidia_speech_python else paths.base_dir/".venv-nvidia-speech/Scripts/python.exe"
    if not python.is_file() or not (model_path()/"model.safetensors").is_file():
        raise RuntimeError("Falta runtime/modelo local Parakeet; ejecuta scripts/setup_nvidia_speech.ps1")
    duration=sf.info(str(audio)).duration
    environment=os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONPATH=str(paths.base_dir))
    with tempfile.TemporaryDirectory(prefix="clipperx-parakeet-") as directory:
        destination=Path(directory)/"result.json"
        result=run_worker([str(python),"-m","clipperx.parakeet_worker","--audio",str(audio.resolve()),
            "--model-path",str(model_path()),"--output",str(destination)],cwd=paths.base_dir,
            env=environment,timeout=min(86400.,max(300.,duration*2+120)))
        if result.returncode:
            raise RuntimeError("Parakeet worker failed (no Whisper fallback): "+result.stderr[-6000:])
        if not destination.is_file() or destination.stat().st_size > 64*1024*1024:
            raise ValueError("Missing or oversized Parakeet receipt")
        payload=json.loads(destination.read_text(encoding="utf-8"))
    validate_payload(payload,duration=duration,audio_sha256=audio_hash(audio))
    payload["request_signature"]=request_signature(language)
    return payload


def validate_payload(payload: dict[str, Any], *, duration: float, audio_sha256: str) -> None:
    if payload.get("backend") != "parakeet" or payload.get("native_timestamps") is not True:
        raise ValueError("Expected native Parakeet receipt")
    if payload.get("audio_sha256") != audio_sha256 or abs(payload.get("duration",-1)-duration) > .1:
        raise ValueError("Parakeet receipt does not match source audio")
    previous=-1.
    for segment in payload["segments"]:
        if not isinstance(segment.get("text"),str) or not isinstance(segment.get("words"),list):
            raise ValueError("Malformed Parakeet segment")
        for word in segment["words"]:
            a,b=word["start"],word["end"]
            if (isinstance(a,bool) or isinstance(b,bool) or not all(math.isfinite(v) for v in (a,b))
                    or not 0 <= a <= b <= duration or a < previous or not isinstance(word["word"],str)):
                raise ValueError("Invalid native Parakeet word timestamps")
            previous=a
