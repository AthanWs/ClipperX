"""Global configuration objects and environment loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys
from glob import glob
from site import getsitepackages
from pathlib import Path
from typing import Sequence

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Environment-backed settings for external services and models."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI / CLIProxyAPI Configuration
    openai_base_url: str = "http://localhost:8317/v1"
    openai_api_key: str = ""
    openai_model: str = ""
    openai_vision_model: str = ""
    llm_num_ctx: int = 8192
    # Legacy Whisper choice; used only with an explicitly selected whisper backend.
    whisper_model: str = "nekusu/faster-whisper-large-v3-turbo-latam-int8-ct2"
    asr_backend: str = "parakeet"
    nvidia_speech_python: str = ""
    parakeet_model_path: str = "data/models/nvidia_speech/parakeet"
    nemotron_model_path: str = "data/models/nvidia_speech/nemotron"
    diarization_backend: str = "nemotron"
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices("EMBED_MODEL", "EMBEDDING_MODEL"),
    )
    # Empty selects data/models/style_embeddings/<embedding_model basename> in this project.
    style_embedding_model_path: str = ""
    transcription_language: str = "es"
    hf_token: str = ""
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_python: str = ""
    speaker_embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    face_model_dir: str = "data/models/face_identity"
    allow_model_downloads: bool = False

    # Caption template
    caption_template: str = "default"
    creator_handle: str = "kick.com/athanw"
    brand_image_path: str = "assets/branding/KickAthanW.png"

    # Virality scoring
    min_virality_score: int = 0

    # Multi-LLM provider remains disabled until a model is explicitly selected.
    llm_provider: str = "disabled"
    google_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # B-Roll (Pexels)
    pexels_api_key: str = ""

    # Export preset
    export_preset: str = "default"

@dataclass(slots=True)
class Paths:
    """Commonly used project directories."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)
    transcripts_dir: Path = field(init=False)
    clips_dir: Path = field(init=False)
    subtitles_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    downloads_dir: Path = field(init=False)
    broll_dir: Path = field(init=False)
    jobs_dir: Path = field(init=False)
    timelines_dir: Path = field(init=False)
    profiles_dir: Path = field(init=False)
    style_library_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.base_dir / "data"
        self.audio_dir = self.data_dir / "processed" / "audio"
        self.transcripts_dir = self.data_dir / "processed" / "transcripts"
        self.clips_dir = self.data_dir / "processed" / "clips"
        self.subtitles_dir = self.clips_dir / "subtitles"
        self.frames_dir = self.data_dir / "processed" / "frames"
        self.logs_dir = self.base_dir / "logs"
        self.downloads_dir = self.data_dir / "downloads"
        self.broll_dir = self.data_dir / "processed" / "broll"
        self.jobs_dir = self.data_dir / "processed" / "jobs"
        self.timelines_dir = self.data_dir / "processed" / "timelines"
        self.profiles_dir = self.data_dir / "profiles"
        self.style_library_dir = self.data_dir / "processed" / "style_library"

    def ensure(self) -> "Paths":
        for path in self.directories():
            path.mkdir(parents=True, exist_ok=True)
        return self

    def directories(self) -> Sequence[Path]:
        return (
            self.data_dir,
            self.audio_dir,
            self.transcripts_dir,
            self.clips_dir,
            self.subtitles_dir,
            self.frames_dir,
            self.logs_dir,
            self.downloads_dir,
            self.broll_dir,
            self.jobs_dir,
            self.timelines_dir,
            self.profiles_dir,
            self.style_library_dir,
        )


settings = EnvSettings()
paths = Paths()


# On Windows, ensure CUDA DLL search path includes cuDNN 8 / cuBLAS bins from pip packages
def _add_windows_cuda_dll_paths() -> None:  # pragma: no cover - environment specific
    if os.name != "nt":
        return
    try:
        candidates: list[str] = []
        for base in list(getsitepackages()) + [sys.prefix]:
            # Common pip package locations
            for pattern in (
                "nvidia/cudnn/bin",
                "nvidia/cublas/bin",
                "nvidia/cudnn*/bin",
                "nvidia/cublas*/bin",
            ):
                paths_glob = glob(os.path.join(base, pattern).replace("/", os.sep))
                candidates.extend(paths_glob)

        # De-duplicate preserving order
        seen: set[str] = set()
        dll_dirs: list[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                dll_dirs.append(candidate)

        # Prepend to PATH to support loaders that rely solely on PATH
        existing_path = os.environ.get("PATH", "")
        path_parts = [p for p in existing_path.split(os.pathsep) if p]
        new_prefixes = [p for p in dll_dirs if p not in path_parts]
        if new_prefixes:
            os.environ["PATH"] = os.pathsep.join(new_prefixes + path_parts)

        for dll_dir in dll_dirs:
            if os.path.isdir(dll_dir):
                try:
                    os.add_dll_directory(dll_dir)
                except Exception:
                    # Best effort; do not fail app startup
                    pass
    except Exception:
        pass


_add_windows_cuda_dll_paths()

