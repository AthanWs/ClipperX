"""
tools/roi_temporal.py
---------------------
Usa la DINAMICA TEMPORAL de la senal (no el promedio): TRIBE da (n_TRs x 20484),
y antes promediabamos el tiempo. Aqui extraemos features temporales A-PRIORI
(hook = activacion temprana, dinamismo = varianza temporal, tendencia, momento del
pico) de la red de VALOR y de la senal global, y correlacionamos con engagement
en las dos tandas (batch1 explora, batch2 confirma held-out).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VALUE_KEYS = ["orbital", "rectus", "subcallosal", "suborbital"]
BATCHES = [
    ("batch1", "data/validation/preds", "data/validation/labels.csv"),
    ("batch2", "data/validation2/preds", "data/validation2/labels.csv"),
    ("batch3", "data/validation3/preds", "data/validation3/labels.csv"),
]


def temporal_feats(act: np.ndarray) -> dict | None:
    t = len(act)
    if t < 3:
        return None
    k = max(1, t // 3)
    early = float(act[:k].mean())
    late = float(act[-k:].mean())
    return {
        "mean": float(act.mean()),
        "hook_early": early,
        "trend_late_minus_early": late - early,
        "temporal_std": float(act.std()),
        "peak_frac": float(np.argmax(act) / (t - 1)),
    }


def build(preds_dir: str, labels_path: str, vmask: np.ndarray):
    lab = pd.read_csv(labels_path).set_index("file")
    rows = []
    for npy in sorted(Path(preds_dir).glob("*.npy")):
        f = npy.stem + ".mp4"
        if f not in lab.index:
            continue
        p = np.load(npy)
        if p.ndim != 2:
            continue
        amag = np.abs(p)
        tg = temporal_feats(amag.mean(axis=1))
        tv = temporal_feats(amag[:, vmask].mean(axis=1))
        if tg is None or tv is None:
            continue
        rec = {"file": f, "engagement_rate": float(lab.loc[f, "engagement_rate"])}
        rec.update({f"global_{k}": v for k, v in tg.items()})
        rec.update({f"value_{k}": v for k, v in tv.items()})
        rows.append(rec)
    df = pd.DataFrame(rows)
    fcols = [c for c in df.columns if c not in ("file", "engagement_rate")]
    return df.drop_duplicates(subset=fcols, keep=False).reset_index(drop=True), fcols


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    vlab = np.concatenate([atlas["map_left"], atlas["map_right"]]).astype(int)
    ids = [i for i, n in enumerate(labels) if any(k in n.lower() for k in VALUE_KEYS)]
    vmask = np.isin(vlab, ids)

    out = {}
    for name, pdir, lpath in BATCHES:
        df, fcols = build(pdir, lpath, vmask)
        res = {c: spearmanr(df[c], df["engagement_rate"]) for c in fcols}
        out[name] = res
        print(f"== {name} (n={len(df)}) — features TEMPORALES vs engagement ==")
        for c, (rho, p) in sorted(res.items(), key=lambda x: -abs(x[1][0])):
            print(f"   {c:28} rho={rho:+.3f}  p={p:.3f}")
        print()

    print("== consistencia batch1 -> batch2 (held-out) ==")
    b1, b2 = out["batch1"], out["batch2"]
    for c in b1:
        if c in b2:
            r1 = b1[c][0]
            r2, p2 = b2[c]
            same = "si" if (r1 > 0) == (r2 > 0) else "NO"
            flag = "  <== consistente+sig" if (same == "si" and abs(r1) >= 0.3 and p2 < 0.05) else ""
            print(f"   {c:28} b1 {r1:+.2f} | b2 {r2:+.2f} (p={p2:.3f})  signo:{same}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
