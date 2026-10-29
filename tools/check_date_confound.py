"""
tools/check_date_confound.py
----------------------------
Hueco D2: ¿el engagement esta confundido por CUANDO se publico el clip
(crecimiento de cuenta / era del algoritmo) en vez de por el contenido?

Truco: el ID de video de TikTok lleva el timestamp Unix en sus bits altos
(ts = id >> 32), asi que la fecha sale del propio ID sin re-scrapear.

Si fecha vs engagement tiene |rho| alto -> hay confound temporal y habria que
controlarlo (correlacion parcial) al evaluar la senal neuro.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

BATCHES = [
    ("batch1", "data/validation/manifest.csv", "data/validation/labels.csv"),
    ("batch2", "data/validation2/manifest.csv", "data/validation2/labels.csv"),
    ("batch3", "data/validation3/manifest.csv", "data/validation3/labels.csv"),
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for name, mp, lp in BATCHES:
        if not (Path(mp).exists() and Path(lp).exists()):
            continue
        man = pd.read_csv(mp)
        lab = pd.read_csv(lp).set_index("file")
        rows = []
        for _, r in man.iterrows():
            f = r.get("file")
            if not isinstance(f, str) or f not in lab.index:
                continue
            try:
                ts = int(r["id"]) >> 32
            except Exception:
                continue
            rows.append({
                "ts": ts,
                "eng": float(lab.loc[f, "engagement_rate"]),
                "views": float(lab.loc[f, "views"]),
            })
        df = pd.DataFrame(rows).drop_duplicates(subset=["ts", "eng"])
        if len(df) < 4:
            print(f"{name}: pocos datos ({len(df)})")
            continue
        re_, pe = spearmanr(df["ts"], df["eng"])
        rv, pv = spearmanr(df["ts"], df["views"])
        dmin = datetime.fromtimestamp(int(df.ts.min()), tz=timezone.utc).date()
        dmax = datetime.fromtimestamp(int(df.ts.max()), tz=timezone.utc).date()
        print(f"{name} (n={len(df)}, {dmin} .. {dmax}):")
        print(f"   recencia vs engagement_rate: rho={re_:+.3f}  p={pe:.3f}")
        print(f"   recencia vs views:           rho={rv:+.3f}  p={pv:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
