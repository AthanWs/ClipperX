"""
tools/check_partial_date.py
---------------------------
Recalcula las correlaciones neuro<->engagement CONTROLANDO por la fecha (confound
D2). Si la correlacion PARCIAL (eng | fecha) se desploma respecto a la cruda, la
"senal" era el confound temporal, no el contenido.

Correlacion parcial de Spearman = Pearson parcial sobre los rangos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BATCHES = [
    ("batch1", "data/validation/preds", "data/validation/labels.csv", "data/validation/manifest.csv"),
    ("batch2", "data/validation2/preds", "data/validation2/labels.csv", "data/validation2/manifest.csv"),
    ("batch3", "data/validation3/preds", "data/validation3/labels.csv", "data/validation3/manifest.csv"),
]
VAL = ["orbital", "rectus", "subcallosal", "suborbital"]
PFC = ["front_sup", "front_middle"]


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den > 0 else np.nan


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from nilearn import datasets

    a = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in a["labels"]]
    vlab = np.concatenate([a["map_left"], a["map_right"]]).astype(int)
    valm = np.isin(vlab, [i for i, n in enumerate(labels) if any(k in n.lower() for k in VAL)])
    pfcm = np.isin(vlab, [i for i, n in enumerate(labels) if any(k in n.lower() for k in PFC)])

    for name, pdir, lp, mp in BATCHES:
        lab = pd.read_csv(lp).set_index("file")
        man = pd.read_csv(mp).set_index("file")
        rows = []
        for npy in sorted(Path(pdir).glob("*.npy")):
            f = npy.stem + ".mp4"
            if f not in lab.index or f not in man.index:
                continue
            p = np.load(npy)
            if p.ndim != 2:
                continue
            amag = np.abs(p)
            g = amag.mean(axis=1)
            v = amag[:, valm].mean(axis=1)
            t = len(g)
            try:
                ts = int(man.loc[f, "id"]) >> 32
            except Exception:
                continue
            rows.append({
                "eng": float(lab.loc[f, "engagement_rate"]), "date": ts,
                "peak_frac": float(np.argmax(g) / (t - 1)),
                "global_mean": float(g.mean()),
                "value_mean": float(v.mean()),
                "value_tstd": float(v.std()),
                "pfc_mean": float(amag[:, pfcm].mean()),
            })
        df = pd.DataFrame(rows).drop_duplicates(subset=["peak_frac", "global_mean", "value_mean"])
        edate = spearmanr(df.eng, df.date)[0]
        print(f"\n== {name} (n={len(df)})  |  engagement~fecha rho={edate:+.2f} ==")
        print(f"   {'feature':12} {'raw(eng)':>9} {'vs_fecha':>9} {'PARCIAL(eng|fecha)':>19}")
        for c in ["peak_frac", "global_mean", "value_mean", "value_tstd", "pfc_mean"]:
            raw = spearmanr(df[c], df.eng)[0]
            vd = spearmanr(df[c], df.date)[0]
            par = partial_spearman(df[c].values, df.eng.values, df.date.values)
            print(f"   {c:12} {raw:+9.3f} {vd:+9.3f} {par:+19.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
