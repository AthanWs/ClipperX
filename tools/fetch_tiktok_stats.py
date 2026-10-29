"""
tools/fetch_tiktok_stats.py
---------------------------
Lee manifest.csv y scrapea las metricas PUBLICAS de cada clip (views, likes,
comentarios, shares) con yt-dlp, y escribe labels.csv con engagement_rate.
Sin trabajo manual: todo sale de datos publicos de TikTok.

engagement_rate = (likes + comentarios + shares) / views

Uso:
  python tools/fetch_tiktok_stats.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def fetch_one(url: str, attempts: int = 4) -> dict | None:
    cmd = [sys.executable, "-m", "yt_dlp", "--skip-download", "--dump-json", "--no-warnings", url]
    for i in range(attempts):
        r = run(cmd)
        try:
            j = json.loads(r.stdout)
        except json.JSONDecodeError:
            time.sleep(4 * (i + 1))  # backoff ante rate-limit de TikTok
            continue
        return {
            "views": int(j.get("view_count") or 0),
            "likes": int(j.get("like_count") or 0),
            "comments": int(j.get("comment_count") or 0),
            "shares": int(j.get("repost_count") or 0),
        }
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/validation/manifest.csv")
    ap.add_argument("--out", default="data/validation/labels.csv")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"No existe {manifest}")

    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    out_rows: list[dict] = []
    for row in rows:
        if not row.get("file"):  # saltar los que no se descargaron
            continue
        clip, url = row["clip"], row["url"]
        time.sleep(1.5)  # espaciar peticiones para no gatillar rate-limit
        m = fetch_one(url)
        if not m:
            print(f"  {clip}: FALLO al leer stats")
            continue
        v = m["views"] or 1
        eng = (m["likes"] + m["comments"] + m["shares"]) / v
        out_rows.append(
            {
                "file": row["file"],
                "views": m["views"],
                "likes": m["likes"],
                "comments": m["comments"],
                "shares": m["shares"],
                "engagement_rate": round(eng, 5),
            }
        )
        print(
            f"  {clip}: views={m['views']:>7} likes={m['likes']:>6} "
            f"com={m['comments']:>5} shares={m['shares']:>5} eng={eng:.4f}"
        )

    if not out_rows:
        raise SystemExit("No se pudo leer stats de ningun clip.")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["file", "views", "likes", "comments", "shares", "engagement_rate"]
        )
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[ok] {len(out_rows)} clips -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
