"""
tools/roi_networks.py
---------------------
Lectura por REDES CEREBRALES definidas A-PRIORI desde la teoria de neuroforecasting
(no fishing). Redes que la literatura asocia a compartir/atencion/valor.
Testea en las DOS tandas (batch1 = explora, batch2 = confirma held-out) para ver
si alguna red es CONSISTENTE en datos independientes.

  python tools/roi_networks.py

Nota: Destrieux es anatomico, asi que estas redes son aproximaciones corticales.
El estriado ventral / nucleo accumbens (mejor predictor de exito) es SUBCORTICAL
y NO esta en el output cortical de TRIBE -> no se puede testear aqui.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Redes corticales definidas A-PRIORI (substrings de etiquetas Destrieux, lower).
NETWORKS = {
    "value_vmPFC":         ["orbital", "rectus", "subcallosal", "suborbital"],
    "mentalizing_social":  ["angular", "precuneus", "cingul-post", "temp_sup-plan_tempo"],
    "emotion_salience":    ["cingul-ant", "cingul-mid-ant", "insula", "_ins_", "temp_pole", "temporal_pole"],
    "attention_control":   ["intrapariet", "front_inf-opercular", "front_inf-triangul"],
    "visual_control":      ["occipital", "calcarine", "cuneus", "lingual"],
}

BATCHES = [
    ("batch1", "data/validation/preds", "data/validation/labels.csv"),
    ("batch2", "data/validation2/preds", "data/validation2/labels.csv"),
]


def network_activations(preds_dir: Path, labels_path: Path, masks: dict) -> pd.DataFrame:
    lab = pd.read_csv(labels_path).set_index("file")
    rows = []
    for npy in sorted(preds_dir.glob("*.npy")):
        file = npy.stem + ".mp4"
        if file not in lab.index:
            continue
        preds = np.load(npy)
        if preds.ndim != 2:
            continue
        amag = np.abs(preds)
        rec = {"file": file, "engagement_rate": float(lab.loc[file, "engagement_rate"])}
        for g, m in masks.items():
            if m.sum():
                rec[g] = float(amag[:, m].mean())
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=list(masks.keys()), keep=False).reset_index(drop=True)


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    vlab = np.concatenate([atlas["map_left"], atlas["map_right"]]).astype(int)
    masks = {}
    for g, keys in NETWORKS.items():
        ids = [i for i, n in enumerate(labels) if any(k in n.lower() for k in keys)]
        masks[g] = np.isin(vlab, ids)
    print("[redes a-priori] vertices:", {g: int(m.sum()) for g, m in masks.items()}, "\n")

    results = {}
    for name, pdir, lpath in BATCHES:
        df = network_activations(Path(pdir), Path(lpath), masks)
        res = {}
        for g in NETWORKS:
            if g in df.columns:
                rho, p = spearmanr(df[g], df["engagement_rate"])
                res[g] = (float(rho), float(p))
        results[name] = (len(df), res)
        print(f"== {name}  (n={len(df)}) — red vs engagement ==")
        for g, (rho, p) in sorted(res.items(), key=lambda x: -abs(x[1][0])):
            print(f"   {g:20} rho={rho:+.3f}  p={p:.3f}")
        print()

    # confirmacion: ¿alguna red consistente (mismo signo) y razonable en AMBAS?
    print("== CONSISTENCIA batch1 -> batch2 (held-out) ==")
    b1 = results["batch1"][1]; b2 = results["batch2"][1]
    # FDR sobre batch2 (la tanda de confirmacion)
    gs = list(NETWORKS)
    q2 = bh_fdr([b2[g][1] for g in gs if g in b2])
    qmap = {g: q for g, q in zip([g for g in gs if g in b2], q2)}
    for g in gs:
        if g in b1 and g in b2:
            r1, _ = b1[g]; r2, p2 = b2[g]
            same = "si" if (r1 > 0) == (r2 > 0) else "NO"
            conf = "  <== CONSISTENTE+SIG" if (same == "si" and abs(r1) >= 0.25 and p2 < 0.05) else ""
            print(f"   {g:20} batch1 {r1:+.2f} | batch2 {r2:+.2f} (p={p2:.3f}, q={qmap.get(g,1):.2f})  mismo signo: {same}{conf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
