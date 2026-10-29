#!/usr/bin/env python
"""Normaliza audio y genera WAV mono 16 kHz listo para ASR."""

import argparse
import subprocess
from pathlib import Path


def build_ffmpeg_cmd(input_path: Path, output_path: Path, loudnorm: bool = False) -> list[str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if loudnorm:
        cmd.extend(["-af", "loudnorm" ])
    cmd.append(str(output_path))
    return cmd


def preprocess(input_path: Path, output_dir: Path, loudnorm: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + "_16k.wav")
    cmd = build_ffmpeg_cmd(input_path, output_path, loudnorm)
    subprocess.run(cmd, check=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normaliza audio a mono 16 kHz")
    parser.add_argument("input", type=Path, help="Ruta al video/audio fuente")
    parser.add_argument("--outdir", type=Path, default=Path("data/processed/audio"))
    parser.add_argument("--loudnorm", action="store_true", help="Aplicar loudnorm EBU R128")
    args = parser.parse_args()

    try:
        output_path = preprocess(args.input, args.outdir, args.loudnorm)
        print(f"Audio normalizado en: {output_path}")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Error ejecutando ffmpeg: {exc}") from exc


if __name__ == "__main__":
    main()

