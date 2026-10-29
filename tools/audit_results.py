"""
tools/audit_results.py
----------------------
Auditoria de la validacion TRIBE: integridad de datos + correlaciones honestas.
  python tools/audit_results.py

- Detecta filas con NaN y DUPLICADOS de features (videos repetidos / colision).
- Quita los duplicados ambiguos y recalcula Spearman vs cada metrica.
- Corrige por comparaciones multiples (Benjamini-Hochberg FDR).
- Leave-one-out sobre la mejor feature (ver si un solo clip arrastra el resultado).
- Chequea confound de duracion (n_trs vs engagement).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CSV = Path("data/validation/neuro_results.csv")
FEATURES = [
    "n_trs", "mean_abs", "global_std", "energy_mean", "energy_peak",
    "energy_peak_to_mean", "energy_temporal_var", "spatial_std_mean",
    "spatial_std_peak", "max_rise", "peak_frac",
]
LABELS = ["engagement_rate", "views", "likes", "shares", "comments"]


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q_sorted, 0, 1)
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    df = pd.read_csv(CSV)
    print("================ AUDITORIA ================")
    print(f"[integridad] filas en CSV: {len(df)}")

    nan_rows = df[df[FEATURES + ["engagement_rate"]].isna().any(axis=1)]
    print(f"[integridad] filas con NaN: {len(nan_rows)} {list(nan_rows['file']) if len(nan_rows) else ''}")

    dup_mask = df.duplicated(subset=FEATURES, keep=False)
    if dup_mask.any():
        print(f"[integridad] DUPLICADOS de features (mismo video): {list(df[dup_mask]['file'])}")
        print("             -> se eliminan ambos (label ambiguo) para no contaminar.")
    clean = df[~dup_mask].dropna(subset=FEATURES + ["engagement_rate"]).reset_index(drop=True)
    print(f"[integridad] filas limpias usadas: {len(clean)}")
    print()

    # correlaciones + FDR sobre TODOS los tests (features x labels)
    recs = []
    for tgt in LABELS:
        sub = clean.dropna(subset=[tgt])
        for f in FEATURES:
            rho, p = spearmanr(sub[f], sub[tgt])
            recs.append({"target": tgt, "feature": f, "rho": float(rho), "p": float(p), "n": len(sub)})
    res = pd.DataFrame(recs)
    res["q_fdr"] = bh_fdr(res["p"].values)

    eng = res[res.target == "engagement_rate"].copy()
    eng = eng.reindex(eng.rho.abs().sort_values(ascending=False).index)
    print(f"== Spearman vs engagement_rate (n={int(eng['n'].iloc[0])}) ==")
    for _, r in eng.iterrows():
        flag = "  <== SIGNIFICATIVO (FDR<0.1)" if r.q_fdr < 0.1 else ""
        print(f"  {r.feature:22} rho={r.rho:+.3f}  p={r.p:.3f}  q_fdr={r.q_fdr:.3f}{flag}")

    best = res.reindex(res.rho.abs().sort_values(ascending=False).index).iloc[0]
    print(f"\n[global] mejor correlacion de todas: {best.feature} vs {best.target} "
          f"rho={best.rho:+.3f} p={best.p:.3f} q_fdr={best.q_fdr:.3f}")

    topf = str(eng.iloc[0]["feature"])
    sub = clean.dropna(subset=["engagement_rate"])
    loo = [spearmanr(sub.drop(sub.index[i])[topf], sub.drop(sub.index[i])["engagement_rate"])[0]
           for i in range(len(sub))]
    print(f"[robustez] leave-one-out de '{topf}' vs engagement: "
          f"min={min(loo):+.3f} max={max(loo):+.3f} (con todos {eng.iloc[0]['rho']:+.3f})")

    rho_d, p_d = spearmanr(sub["n_trs"], sub["engagement_rate"])
    print(f"[confound] duracion (n_trs) vs engagement: rho={rho_d:+.3f} p={p_d:.3f} "
          f"({'sin' if abs(rho_d) < 0.3 else 'POSIBLE'} confound de duracion)")

    print("\n== VEREDICTO ==")
    strong = bool((eng.rho.abs() >= 0.30).any())
    coherent = int((eng.rho >= 0.25).sum())
    survives = bool((eng.q_fdr < 0.1).any())
    print(f"  features con |rho|>=0.30 vs engagement: {int((eng.rho.abs()>=0.30).sum())}")
    print(f"  features positivas >=+0.25 (patron coherente): {coherent}")
    print(f"  sobrevive correccion por comparaciones multiples (FDR<0.1): {survives}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
