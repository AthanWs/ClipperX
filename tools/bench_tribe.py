"""
tools/bench_tribe.py  (worker del benchmark de optimizacion)
------------------------------------------------------------
Corre TRIBE sobre uno o varios clips a una PRECISION dada, mide tiempo de pared
y VRAM pico, y guarda los `preds` crudos (n_TRs x ~20k) para comparar fidelidad.

  python tools/bench_tribe.py --precision fp16 --clips clip01.mp4,clip02.mp4 --cache <dir>

Precision:
  fp32 = como viene TRIBE (baseline de fidelidad).
  fp16 / bf16 = encoders cargados en media precision (mitad de VRAM, ~2x speed).
                Se inyecta torch_dtype en transformers.from_pretrained (encoders);
                la fusion sigue en fp32. bf16 ademas castea a fp32 antes de .numpy().

Salida: data/validation/bench/preds_<prec>_<clip>.npy  +  append a metrics.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pathlib as _pathlib

if sys.platform == "win32":
    _pathlib.PosixPath = _pathlib.WindowsPath

BENCH = Path("data/validation/bench")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp32", "acbf16", "acfp16"], required=True)
    ap.add_argument("--clips", required=True, help="nombres separados por comas")
    ap.add_argument("--dir", default="data/validation/clips_proc")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--text-model", default="unsloth/Llama-3.2-3B")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    BENCH.mkdir(parents=True, exist_ok=True)

    from tribev2 import TribeModel
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # --- autocast (mixed precision): acelera matmuls/convs en bf16/fp16 SIN tocar
    # los pesos (siguen fp32) ni romper dtypes de entrada (autocast castea solo).
    autocast_dtype = None
    if args.precision == "acfp16":
        autocast_dtype = torch.float16
    elif args.precision == "acbf16":
        autocast_dtype = torch.bfloat16

    cfg = {"data.text_feature.model_name": args.text_model}
    if sys.platform == "win32":
        cfg["data.num_workers"] = 0

    print(f"[bench] precision={args.precision} cache={args.cache} cargando modelo...", flush=True)
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=args.cache, device="cuda", config_update=cfg
    )

    metrics_path = BENCH / "metrics.jsonl"
    for name in [c.strip() for c in args.clips.split(",") if c.strip()]:
        vid = Path(args.dir) / name
        if not vid.exists():
            print(f"  {name}: NO EXISTE")
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            events = model.get_events_dataframe(video_path=str(vid))
            if autocast_dtype is not None:
                with torch.autocast("cuda", dtype=autocast_dtype):
                    preds, _ = model.predict(events=events, verbose=False)
            else:
                preds, _ = model.predict(events=events, verbose=False)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  {name}: FALLO {msg}")
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"precision": args.precision, "clip": name, "error": msg}) + "\n")
            gc.collect()
            torch.cuda.empty_cache()
            continue
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_reserved() / 1e9
        preds = np.asarray(preds, dtype=np.float32)
        np.save(BENCH / f"preds_{args.precision}_{name}.npy", preds)
        rec = {
            "precision": args.precision,
            "clip": name,
            "seconds": round(dt, 1),
            "peak_gb": round(peak, 2),
            "shape": list(preds.shape),
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  {name}: {dt:.1f}s  peak={peak:.2f}GB  shape={tuple(preds.shape)}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
