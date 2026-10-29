"""
tools/preprocess_clips.py
-------------------------
Recorta + reduce resolucion de los clips para que V-JEPA2-ViT-g quepa en 12GB
de VRAM (si no, Windows derrama a RAM del sistema y la velocidad se desploma).

- Recorta a MAXDUR segundos (el inicio = hook, lo mas relevante para engagement).
- Escala a 256 px de alto. V-JEPA2 redimensiona internamente a 256x256, asi que
  bajar a 256p NO le quita informacion al modelo (es practicamente lossless para
  el encoder) y reduce muchisimo la memoria de carga de frames.

Uso:
  python tools/preprocess_clips.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/validation/clips")
    ap.add_argument("--dst", default="data/validation/clips_proc")
    ap.add_argument("--maxdur", type=int, default=15)
    args = ap.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    clips = sorted(src.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"No hay .mp4 en {src}")
    ok = 0
    for mp4 in clips:
        out = dst / mp4.name
        cmd = [
            FFMPEG, "-y", "-i", str(mp4),
            "-t", str(args.maxdur),
            "-vf", "scale=-2:256",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "96k",
            str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if out.exists() and out.stat().st_size > 0:
            ok += 1
            print(f"  {mp4.name}: OK")
        else:
            print(f"  {mp4.name}: FALLO -> {r.stderr[-200:]}")
    print(f"\n[ok] {ok}/{len(clips)} procesados -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
