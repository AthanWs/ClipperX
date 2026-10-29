"""
tools/validate_neuro.py
-----------------------
Fase 0 - Valida si la "respuesta cerebral" que predice TRIBE v2 (Meta) sirve
para distinguir clips buenos de malos ANTES de integrarlo en el pipeline.

Idea: corre TRIBE sobre un set de clips historicos (con su rendimiento real
conocido) y mide si alguna metrica derivada de la prediccion cerebral
correlaciona con la retencion / views reales de TikTok.

Uso:
    python tools/validate_neuro.py
    python tools/validate_neuro.py --dir data/validation/clips --labels data/validation/labels.csv --device cuda

labels.csv (lo genero yo a partir de tus pantallazos):
    file,retention,views,likes,shares
    clip01.mp4,0.58,15200,1300,210
    ...
    - 'file' debe coincidir con el nombre del .mp4 en --dir
    - 'retention' = fraccion 0-1 (o %, se normaliza). Es la etiqueta principal.
    - el resto (views, likes, shares) son opcionales.

Salida:
    data/validation/neuro_results.csv   (features neuro por clip + labels)
    + imprime la correlacion (Spearman) de cada feature vs retention/views,
      rankeada, para ver QUE senal (si alguna) predice el rendimiento real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# TRIBE serializa rutas como PosixPath dentro de config.yaml (creado en Linux).
# En Windows instanciar un PosixPath revienta -> lo mapeamos a WindowsPath.
import pathlib as _pathlib

if sys.platform == "win32":
    _pathlib.PosixPath = _pathlib.WindowsPath


def reduce_predictions(preds: np.ndarray) -> dict[str, float]:
    """Reduce la matriz (n_TRs, n_vertices) de TRIBE a features escalares.

    No sabemos a priori QUE resumen de la actividad cerebral se relaciona con el
    enganche, asi que calculamos varios candidatos y dejamos que la correlacion
    con el rendimiento real decida cual sirve (busqueda de features honesta).
    """
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim == 1:
        preds = preds[:, None]
    n_trs = preds.shape[0]

    # "Energia" por TR = magnitud de la respuesta cortical en ese instante.
    energy = np.linalg.norm(preds, axis=1)            # (n_TRs,)
    # Variabilidad espacial por TR = cuanto se diferencia la activacion del cerebro.
    spatial_std = preds.std(axis=1)                    # (n_TRs,)

    eps = 1e-9
    return {
        "n_trs": float(n_trs),
        "mean_abs": float(np.abs(preds).mean()),
        "global_std": float(preds.std()),
        "energy_mean": float(energy.mean()),
        "energy_peak": float(energy.max()),
        "energy_peak_to_mean": float(energy.max() / (energy.mean() + eps)),
        "energy_temporal_var": float(energy.var()),
        "spatial_std_mean": float(spatial_std.mean()),
        "spatial_std_peak": float(spatial_std.max()),
        # rise = subida mas fuerte entre TRs consecutivos (proxy del "hook").
        "max_rise": float(np.diff(energy).max()) if n_trs > 1 else 0.0,
        # donde cae el pico (fraccion del clip; robusto al valor real del TR).
        "peak_frac": float(int(np.argmax(energy)) / max(1, n_trs - 1)),
    }


def load_labels(path: Path) -> dict[str, dict]:
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "file" not in df.columns:
        raise SystemExit(f"labels.csv debe tener columna 'file'. Tiene: {list(df.columns)}")
    if "retention" in df.columns:
        r = df["retention"].astype(float)
        if r.max() > 1.5:  # viene en %, normaliza a 0-1
            df["retention"] = r / 100.0
    return {str(row["file"]).strip(): row.to_dict() for _, row in df.iterrows()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Valida TRIBE v2 contra el rendimiento real de clips.")
    ap.add_argument("--dir", default="data/validation/clips", help="Carpeta con los .mp4")
    ap.add_argument("--labels", default="data/validation/labels.csv", help="CSV con rendimiento real")
    ap.add_argument("--out", default="data/validation/neuro_results.csv")
    ap.add_argument("--cache", default="data/validation/.tribe_cache")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument(
        "--text-model",
        default=None,
        help="Override del encoder de texto (p.ej. mirror abierto de LLaMA). None = el del config.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Procesar solo los primeros N clips (0 = todos).")
    ap.add_argument("--dump-preds", default=None, help="Carpeta donde guardar los preds crudos (n_TRs x vertices) por clip, para analisis por ROI.")
    args = ap.parse_args()

    clips_dir = Path(args.dir)
    if not clips_dir.exists():
        raise SystemExit(f"No existe {clips_dir}. Mete ahi los .mp4.")
    videos = sorted(clips_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No hay .mp4 en {clips_dir}.")
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]
        print(f"[limit] procesando solo {len(videos)} clip(s)")

    labels: dict[str, dict] = {}
    labels_path = Path(args.labels)
    if labels_path.exists():
        labels = load_labels(labels_path)
    else:
        print(f"[aviso] No encuentro {labels_path}: corro TRIBE igual, pero sin correlacion.")

    try:
        from tribev2 import TribeModel
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "No puedo importar 'tribev2'. Instalalo primero:\n"
            "  pip install -e <ruta-al-repo-tribev2>\n"
            f"Detalle: {e}"
        )
    import pandas as pd
    import gc

    import torch

    # Ampere (RTX 3060): TF32 acelera los matmul fp32 sin cambio practico en la
    # salida; cudnn.benchmark autotunea kernels para shapes repetidos.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    cfg_update: dict = {}
    if args.text_model:
        cfg_update["data.text_feature.model_name"] = args.text_model
        print(f"[tribe] override rama de texto -> {args.text_model}")
    if sys.platform == "win32":
        # El DataLoader con num_workers=20 (config) usa spawn en Windows y es
        # lento/inestable; 0 = en proceso (el dataset por clip es pequeno).
        cfg_update["data.num_workers"] = 0
    cfg_update = cfg_update or None
    print(f"[tribe] cargando modelo (device={args.device})... (la 1a vez baja varios GB)")
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=args.cache, device=args.device, config_update=cfg_update
    )

    rows: list[dict] = []
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name} ...", flush=True)
        try:
            events = model.get_events_dataframe(video_path=str(video))
            # autocast fp16: ~2.7x mas rapido y cabe en 12GB; fidelidad 0.99937 vs fp32.
            with torch.autocast("cuda", dtype=torch.float16):
                preds, _segments = model.predict(events=events, verbose=False)
            preds_np = np.asarray(preds, dtype=np.float32)
            if args.dump_preds:
                _dp = Path(args.dump_preds)
                _dp.mkdir(parents=True, exist_ok=True)
                np.save(_dp / f"{video.stem}.npy", preds_np)
            feats = reduce_predictions(preds_np)
        except Exception as e:  # noqa: BLE001
            print(f"   ! fallo en {video.name}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        row = {"file": video.name, **feats}
        lab = labels.get(video.name, {})
        for k in ("retention", "views", "likes", "comments", "shares", "engagement_rate"):
            if k in lab:
                row[k] = lab[k]
        rows.append(row)
        rt = row.get("retention")
        print(
            f"   peak_to_mean={feats['energy_peak_to_mean']:.2f}  pico@{feats['peak_frac'] * 100:.0f}%"
            + (f"  retention={rt}" if rt is not None else "")
        )
        # Liberar VRAM entre clips: neuralset recarga los encoders por clip y el
        # allocator de torch retiene lo liberado -> sin esto la GPU se va llenando.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No se pudo procesar ningun clip.")

    res = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\n[ok] resultados -> {args.out}")

    from scipy.stats import spearmanr

    label_cols = ("retention", "views", "likes", "comments", "shares", "engagement_rate")
    feature_cols = [c for c in res.columns if c not in ("file", *label_cols)]
    targets = [c for c in label_cols if c in res.columns]
    for target in targets:
        if target not in res.columns:
            continue
        sub = res.dropna(subset=[target])
        if len(sub) < 4:
            print(f"\n[{target}] muy pocos datos ({len(sub)}) para correlacionar.")
            continue
        print(f"\n=== Correlacion (Spearman) feature vs {target}  [n={len(sub)}] ===")
        scored = []
        for col in feature_cols:
            try:
                rho, p = spearmanr(sub[col], sub[target])
            except Exception:  # noqa: BLE001
                continue
            if not np.isnan(rho):
                scored.append((col, float(rho), float(p)))
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, rho, p in scored:
            flag = "  <-- fuerte" if abs(rho) >= 0.5 and p < 0.1 else ""
            print(f"  {col:22s} rho={rho:+.3f}  p={p:.3f}{flag}")

    print("\nLectura: rho cerca de +1 = esa senal sube cuando sube el rendimiento.")
    print("Si nada pasa de |rho|~0.3, TRIBE no aporta para TU contenido y mejor no integrarlo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
