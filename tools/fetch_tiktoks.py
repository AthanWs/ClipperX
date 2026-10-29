"""
tools/fetch_tiktoks.py
----------------------
Descarga un set de TikToks de un perfil (abanico por views) para validacion.
Soporta excluir IDs ya usados y DEDUP por hash de contenido (el mismo video con
IDs distintos = bug de TikTok que vimos en la 1a tanda).

  python tools/fetch_tiktoks.py --profile URL --n 60 --out data/validation2/clips \
         --manifest data/validation2/manifest.csv \
         --exclude data/validation/manifest.csv --exclude-hash-dir data/validation/clips

Salida: <out>/clipNN.mp4 ...  +  manifest.csv (clip,id,url,views,title,file)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_videos(profile: str) -> list[dict]:
    cmd = [sys.executable, "-m", "yt_dlp", "--flat-playlist", "--no-warnings", "--dump-json", profile]
    r = run(cmd)
    vids: list[dict] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = j.get("id")
        if not vid:
            continue
        vids.append({
            "id": str(vid),
            "views": int(j.get("view_count") or 0),
            "title": (j.get("title") or "").replace("\n", " ").strip()[:80],
            "url": j.get("url") or f"{profile}/video/{vid}",
        })
    if not vids and r.stderr:
        print("   [yt-dlp stderr]", r.stderr.strip()[:300])
    return vids


def select_spread(vids: list[dict], n: int) -> list[dict]:
    vids = sorted(vids, key=lambda v: v["views"])
    if len(vids) <= n:
        return vids
    idx = sorted({round(i * (len(vids) - 1) / (n - 1)) for i in range(n)})
    return [vids[i] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="data/validation/clips")
    ap.add_argument("--manifest", default="data/validation/manifest.csv")
    ap.add_argument("--exclude", default=None, help="manifest CSV cuyos IDs hay que saltar")
    ap.add_argument("--exclude-hash-dir", default=None, help="dir cuyos hashes de contenido hay que evitar")
    ap.add_argument("--recent", action="store_true", help="coger los N mas RECIENTES (por id) en vez de abanico por views")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    excl_ids: set[str] = set()
    for mpath in (args.exclude or "").split(","):
        mpath = mpath.strip()
        if mpath and Path(mpath).exists():
            with open(mpath, encoding="utf-8") as f:
                excl_ids |= {str(r.get("id", "")) for r in csv.DictReader(f)}
    if excl_ids:
        print(f"[exclude] {len(excl_ids)} IDs a saltar")

    print("[1/4] listando perfil...")
    vids = list_videos(args.profile)
    print(f"   total: {len(vids)}")
    vids = [v for v in vids if v["id"] not in excl_ids]
    print(f"   tras excluir ya-usados: {len(vids)}")
    if not vids:
        raise SystemExit("No quedan videos para descargar.")

    if args.recent:
        sel = sorted(vids, key=lambda v: int(v["id"]), reverse=True)[: args.n]
        print(f"[2/4] seleccionados {len(sel)} MAS RECIENTES")
    else:
        sel = select_spread(vids, args.n)
        print(f"[2/4] seleccionados {len(sel)} (views {min(v['views'] for v in sel)}..{max(v['views'] for v in sel)})")
    sel.sort(key=lambda v: v["views"], reverse=True)

    print("[3/4] descargando...")
    rows: list[dict] = []
    for i, v in enumerate(sel, 1):
        name = f"clip{i:02d}"
        url = v["url"] if str(v["url"]).startswith("http") else f"{args.profile}/video/{v['id']}"
        cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--remux-video", "mp4",
               "-o", str(out / f"{name}.%(ext)s"), url]
        run(cmd)
        matches = list(out.glob(f"{name}.*"))
        fname = matches[0].name if matches else ""
        rows.append({"clip": name, "id": v["id"], "url": url, "views": v["views"],
                     "title": v["title"], "file": fname})

    # dedup por hash de contenido
    print("[4/4] dedup por hash...")
    seen: set[str] = set()
    for hd in (args.exclude_hash_dir or "").split(","):
        hd = hd.strip()
        if hd and Path(hd).exists():
            for f in Path(hd).glob("*.mp4"):
                seen.add(md5(f))
    if seen:
        print(f"   hashes de batches previos a evitar: {len(seen)}")
    dropped = 0
    for r in rows:
        if not r["file"]:
            continue
        h = md5(out / r["file"])
        if h in seen:
            (out / r["file"]).unlink(missing_ok=True)
            r["file"] = ""
            dropped += 1
        else:
            seen.add(h)
    ok = sum(1 for r in rows if r["file"])
    print(f"   duplicados/idénticos eliminados: {dropped}  ->  usables: {ok}")

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "id", "url", "views", "title", "file"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {ok} clips usables -> {out}  | manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
