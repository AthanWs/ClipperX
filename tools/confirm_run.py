"""
tools/confirm_run.py
--------------------
Experimento de CONFIRMACION (batch 2, independiente del primero).
Pipeline completo y autonomo:
  1) descarga ~50 clips NUEVOS (excluye los ya usados + dedup por hash de contenido)
  2) stats publicas -> labels
  3) preprocesa (15s, 256p)
  4) TRIBE en acfp16 (rapido) + dump de preds crudos por clip
  5) analisis POR REGION, con la hipotesis PREFRONTAL pre-fijada

  python tools/confirm_run.py

Si la activacion prefrontal vuelve a salir ~+0.5 en estos clips NUEVOS, la senal
queda confirmada limpia (sin p-hacking, hipotesis fijada de antemano).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEFAULT_PROFILE = os.environ.get("CLIPPERX_VALIDATION_PROFILE", "")
DEFAULT_PY_TRIBE = os.environ.get(
    "CLIPPERX_TRIBE_PYTHON",
    r"O:\Escritorio\ClipperX\.venv-tribe\Scripts\python.exe",
)


def step(name: str, py: str, cmd: list[str], env=None) -> int:
    print(f"\n########## {name} ##########", flush=True)
    r = subprocess.run([py] + cmd, env=env)
    print(f"########## {name} exit={r.returncode} ##########", flush=True)
    return r.returncode


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/validation2")
    ap.add_argument("--exclude", default="data/validation/manifest.csv")
    ap.add_argument("--exclude-hash", default="data/validation/clips")
    ap.add_argument("--n", default="65")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, required=not bool(DEFAULT_PROFILE))
    ap.add_argument("--python-main", default=sys.executable)
    ap.add_argument("--python-tribe", default=DEFAULT_PY_TRIBE)
    a = ap.parse_args()
    BASE = Path(a.base)
    BASE.mkdir(parents=True, exist_ok=True)

    if step("1/5 FETCH (nuevos + dedup)", a.python_main, [
        "tools/fetch_tiktoks.py", "--profile", a.profile, "--n", a.n,
        "--out", str(BASE / "clips"), "--manifest", str(BASE / "manifest.csv"),
        "--exclude", a.exclude,
        "--exclude-hash-dir", a.exclude_hash,
    ]):
        return 1

    if step("2/5 STATS publicas", a.python_main, [
        "tools/fetch_tiktok_stats.py", "--manifest", str(BASE / "manifest.csv"),
        "--out", str(BASE / "labels.csv"),
    ]):
        return 1

    if step("3/5 PREPROCESS (15s, 256p)", a.python_main, [
        "tools/preprocess_clips.py", "--src", str(BASE / "clips"), "--dst", str(BASE / "clips_proc"),
    ]):
        return 1

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if step("4/5 TRIBE acfp16 + dump preds", a.python_tribe, [
        "tools/validate_neuro.py", "--text-model", "unsloth/Llama-3.2-3B", "--device", "cuda",
        "--dir", str(BASE / "clips_proc"), "--cache", str(BASE / ".cache_acfp16"),
        "--out", str(BASE / "neuro_results.csv"), "--dump-preds", str(BASE / "preds"),
    ], env=env):
        return 1

    step("5/5 ROI (prefrontal pre-fijado)", a.python_tribe, [
        "tools/roi_analysis.py", "--preds", str(BASE / "preds"), "--labels", str(BASE / "labels.csv"),
    ])

    print("\n########## CONFIRMACION COMPLETA ##########")
    return 0


if __name__ == "__main__":
    sys.exit(main())
