"""
tools/gap_checks.py
-------------------
Chequeos de huecos metodologicos sobre los preds ya guardados (batch1+batch2):
  1) Sanity de alineacion del atlas: las regiones mas activadas deberian ser
     SENSORIALES (visual/auditivo) porque el estimulo es video+audio. Si no, la
     correspondencia vertice<->region (y por tanto el analisis por region) es dudosa.
  2) Readout CON SIGNO (no |abs|): activar recompensa o DESACTIVAR la red por
     defecto (DMN) son senales de atencion con signo opuesto; abs() las borra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BATCHES = [
    ("batch1", "data/validation/preds", "data/validation/labels.csv"),
    ("batch2", "data/validation2/preds", "data/validation2/labels.csv"),
]
DMN_KEYS = ["precuneus", "cingul-post", "angular", "rectus", "subcallosal"]
VAL_KEYS = ["orbital", "rectus", "subcallosal", "suborbital"]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from nilearn import datasets

    a = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in a["labels"]]
    vlab = np.concatenate([a["map_left"], a["map_right"]]).astype(int)
    region_masks = {labels[i]: (vlab == i) for i in range(len(labels)) if (vlab == i).sum() > 0}
    dmn = np.isin(vlab, [i for i, n in enumerate(labels) if any(k in n.lower() for k in DMN_KEYS)])
    val = np.isin(vlab, [i for i, n in enumerate(labels) if any(k in n.lower() for k in VAL_KEYS)])

    print("== CHECK 1: sanity de alineacion atlas (top regiones por |activacion| media) ==")
    arrs = []
    for _, pdir, _ in BATCHES:
        for npy in sorted(Path(pdir).glob("*.npy"))[:8]:
            arrs.append(np.abs(np.load(npy)).mean(axis=0))
    M = np.stack(arrs).mean(axis=0)
    reg = sorted(((float(M[m].mean()), n) for n, m in region_masks.items()), reverse=True)[:10]
    for v, n in reg:
        print(f"   {v:.3f}  {n}")
    print("   -> si dominan occipital/calcarine/temp_sup (sensorial) la alineacion es plausible\n")

    print("== CHECK 2: readout CON SIGNO vs engagement (no abs) ==")
    for name, pdir, lp in BATCHES:
        lab = pd.read_csv(lp).set_index("file")
        rows = []
        for npy in sorted(Path(pdir).glob("*.npy")):
            f = npy.stem + ".mp4"
            if f not in lab.index:
                continue
            p = np.load(npy)
            if p.ndim != 2:
                continue
            rows.append({
                "eng": float(lab.loc[f, "engagement_rate"]),
                "dmn_signed": float(p[:, dmn].mean()),
                "val_signed": float(p[:, val].mean()),
            })
        df = pd.DataFrame(rows).drop_duplicates(subset=["dmn_signed", "val_signed"], keep=False)
        rd, pdp = spearmanr(df["dmn_signed"], df["eng"])
        rv, pv = spearmanr(df["val_signed"], df["eng"])
        print(f"   {name} (n={len(df)}): DMN_signed rho={rd:+.3f} p={pdp:.3f}  |  value_signed rho={rv:+.3f} p={pv:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
