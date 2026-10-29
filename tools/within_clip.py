"""
tools/within_clip.py
--------------------
EL test que de verdad importa: DENTRO de cada clip, ¿la curva neuronal de TRIBE
predice donde la gente abandona (caida de retencion)?

- neural(t) = activacion media |pred| por TR.
- retencion del usuario, segundo a segundo -> drop-rate (cuanta gente se va).
- Si neural BAJA donde la gente se va (drop alto) -> correlacion NEGATIVA
  neural~drop -> TRIBE detecta las partes aburridas. Eso seria la senal usable.

Aprox honesta: asumo que los TRs cubren el clip de forma uniforme (no tengo el
timestamp exacto de cada TR). Para una primera lectura vale; si sale prometedor
se refina con los tiempos reales de segmento.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PREDS = Path("data/validation_recent/preds_ret")
RET = Path("data/validation_recent/retention.csv")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ret = pd.read_csv(RET)
    pooled_neural, pooled_drop = [], []
    pooled_neural_noh, pooled_drop_noh = [], []  # sin los primeros 3s (el hook universal)
    print(f"{'clip':10} {'dur':>5} {'TRs':>4} {'neural~ret':>11} {'neural~drop':>12} {'neural~drop(>3s)':>16}")
    for f in sorted(ret.file.unique()):
        npy = PREDS / (f.replace(".mp4", "") + ".npy")
        if not npy.exists():
            print(f"{f:10} sin preds")
            continue
        preds = np.load(npy)
        if preds.ndim != 2:
            continue
        neural = np.abs(preds).mean(axis=1)
        r = ret[ret.file == f].sort_values("sec")
        secs = r.sec.values.astype(float)
        rv = r.ret.values.astype(float)
        dur = float(secs.max())
        t = len(neural)
        tr_t = np.linspace(0, dur, t)
        ret_at = np.interp(tr_t, secs, rv)
        drop = -np.gradient(ret_at)  # alto = la gente se va
        rho_lvl = spearmanr(neural, ret_at)[0]
        rho_drop = spearmanr(neural, drop)[0]
        keep = tr_t > 3.0
        rho_drop_noh = spearmanr(neural[keep], drop[keep])[0] if keep.sum() >= 4 else float("nan")
        print(f"{f:10} {dur:5.0f} {t:4d} {rho_lvl:+11.2f} {rho_drop:+12.2f} {rho_drop_noh:+16.2f}")
        pooled_neural.extend(neural)
        pooled_drop.extend(drop)
        pooled_neural_noh.extend(neural[keep])
        pooled_drop_noh.extend(drop[keep])

    if pooled_neural:
        rp = spearmanr(pooled_neural, pooled_drop)[0]
        rpn = spearmanr(pooled_neural_noh, pooled_drop_noh)[0] if len(pooled_neural_noh) > 4 else float("nan")
        print(f"\nPOOLED neural~drop_rate: rho={rp:+.3f}   |  sin primeros 3s: rho={rpn:+.3f}")
        print("(NEGATIVO = TRIBE baja justo donde la gente abandona = detecta lo aburrido)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
