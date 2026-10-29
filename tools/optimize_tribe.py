"""
tools/optimize_tribe.py  (driver del benchmark de optimizacion)
---------------------------------------------------------------
Corre bench_tribe.py en varias precisiones, luego compara velocidad / VRAM pico /
fidelidad (similitud coseno de los preds vs el baseline fp32) y reporta la mejor.

  python tools/optimize_tribe.py

- fp32 reusa la cache existente (.tribe_cache_proc) -> baseline rapido.
- fp16 / bf16 usan cache propia y se prueban en clip01 (fidelidad) + clip02 (que
  era el que derramaba -> mide si ahora cabe en 12GB y cuanto tarda).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = sys.executable
BENCH = Path("data/validation/bench")

# (precision, clips, cache)
CONFIGS = [
    ("fp32", "clip01.mp4", "data/validation/.tribe_cache_proc"),
    ("acbf16", "clip01.mp4,clip02.mp4", "data/validation/.bench_cache_acbf16"),
    ("acfp16", "clip01.mp4,clip02.mp4", "data/validation/.bench_cache_acfp16"),
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    BENCH.mkdir(parents=True, exist_ok=True)
    mpath = BENCH / "metrics.jsonl"
    if mpath.exists():
        mpath.unlink()

    for prec, clips, cache in CONFIGS:
        print(f"\n===== BENCH {prec} : {clips} =====", flush=True)
        subprocess.run(
            [PY, "tools/bench_tribe.py", "--precision", prec, "--clips", clips, "--cache", cache]
        )

    metrics = [json.loads(l) for l in open(mpath, encoding="utf-8")] if mpath.exists() else []
    base_path = BENCH / "preds_fp32_clip01.mp4.npy"
    base = np.load(base_path) if base_path.exists() else None

    print("\n================ RESUMEN OPTIMIZACION ================")
    print(f"{'prec':6} {'clip':14} {'seg':>7} {'VRAM_GB':>8} {'fidelidad':>10}")
    for m in metrics:
        if "error" in m:
            print(f"{m['precision']:6} {m['clip']:14} {'ERROR':>7}   {m['error'][:45]}")
            continue
        fid = "-"
        if base is not None and m["clip"] == "clip01.mp4" and m["precision"] != "fp32":
            p = BENCH / f"preds_{m['precision']}_{m['clip']}.npy"
            if p.exists():
                fid = f"{cosine(base, np.load(p)):.5f}"
        print(f"{m['precision']:6} {m['clip']:14} {m['seconds']:>7} {m['peak_gb']:>8} {fid:>10}")

    print("\nMejor = precision mas rapida en clip02 que NO derrame (VRAM<~11.5) y fidelidad>0.999 vs fp32.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
