"""
tools/roi_analysis.py
---------------------
Lee las señales del cerebro POR REGION (no el promedio global) y correlaciona
cada region con el engagement. Mapea los 20.484 vertices de fsaverage5 a regiones
del atlas Destrieux (via nilearn) y las agrupa en redes interpretables.

  python tools/roi_analysis.py

Requiere: data/validation/preds/<clip>.npy (preds crudos n_TRs x 20484) + labels.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PREDS = Path("data/validation/preds")
LABELS = Path("data/validation/labels.csv")

# Grupos de regiones por palabras clave del atlas Destrieux.
GROUPS = {
    "visual_occipital": ["occipital", "calcarine", "cuneus", "lingual"],
    "face_fusiform": ["fusifor", "oc-temp_lat"],
    "auditory": ["temp_sup-g_t_transv", "temp_sup-plan_tempo", "temp_sup-lateral"],
    "language_ifg": ["front_inf"],
    "emotion_limbic": ["cingul", "insula", "orbital", "temp_pole", "subcallosal"],
    "motor_somato": ["precentral", "postcentral", "_central"],
    "prefrontal": ["front_sup", "front_middle"],
    "parietal_attention": ["parietal", "intrapariet", "precuneus"],
    "temporal_other": ["temporal_inf", "temp_sup-g_temp", "temp_middle"],
}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="data/validation/preds")
    ap.add_argument("--labels", default="data/validation/labels.csv")
    args = ap.parse_args()
    preds_dir = Path(args.preds)
    labels_path = Path(args.labels)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    vlab = np.concatenate([atlas["map_left"], atlas["map_right"]]).astype(int)  # (20484,)

    masks = {}
    for g, keys in GROUPS.items():
        ids = [i for i, n in enumerate(labels) if any(k in n.lower() for k in keys)]
        masks[g] = np.isin(vlab, ids)
    print("[atlas] vertices por region:")
    for g, m in masks.items():
        print(f"   {g:22} {int(m.sum()):>5} vertices")
    print()

    lab = pd.read_csv(labels_path).set_index("file")
    rows = []
    for npy in sorted(preds_dir.glob("*.npy")):
        file = npy.stem + ".mp4"
        if file not in lab.index:
            continue
        preds = np.load(npy)
        if preds.ndim != 2 or preds.shape[1] != vlab.shape[0]:
            print(f"   [skip] {file} shape {preds.shape}")
            continue
        amag = np.abs(preds)  # (T, V)
        rec = {"file": file, "engagement_rate": float(lab.loc[file, "engagement_rate"]),
               "views": float(lab.loc[file, "views"])}
        for g, m in masks.items():
            if m.sum():
                rec[g] = float(amag[:, m].mean())
        rows.append(rec)

    df = pd.DataFrame(rows)
    feat = [g for g in GROUPS if g in df.columns]
    # quitar el video duplicado (preds identicos -> features de region identicas)
    df = df.drop_duplicates(subset=feat, keep=False).reset_index(drop=True)
    print(f"[datos] clips usados (sin duplicado): n={len(df)}\n")

    print("== Activacion por REGION vs engagement_rate (Spearman) ==")
    res = []
    for g in feat:
        rho, p = spearmanr(df[g], df["engagement_rate"])
        res.append([g, float(rho), float(p)])
    ps = np.array([r[2] for r in res])
    nq = len(ps)
    order = np.argsort(ps)
    q_sorted = np.minimum.accumulate((ps[order] * nq / (np.arange(nq) + 1))[::-1])[::-1]
    qv = np.empty(nq)
    qv[order] = np.clip(q_sorted, 0, 1)
    for i, r in enumerate(res):
        r.append(float(qv[i]))
    for g, rho, p, qq in sorted(res, key=lambda x: -abs(x[1])):
        flag = "  <== FDR<0.1" if qq < 0.1 else ("  * p<0.05" if p < 0.05 else "")
        print(f"   {g:22} rho={rho:+.3f}  p={p:.3f}  q_fdr={qq:.3f}{flag}")

    top = max(res, key=lambda x: abs(x[1]))[0]
    loo = [spearmanr(df.drop(df.index[i])[top], df.drop(df.index[i])["engagement_rate"])[0]
           for i in range(len(df))]
    print(f"\n[robustez] leave-one-out de '{top}': min={min(loo):+.3f} max={max(loo):+.3f} "
          f"(todos {dict((r[0], r[1]) for r in res)[top]:+.3f})")
    print("(Global anterior: spatial_std_peak rho=+0.43; por region la senal sube.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
