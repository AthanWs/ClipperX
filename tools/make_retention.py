"""Escribe data/validation_recent/retention.csv a partir de las curvas que paso el usuario."""
import csv
from pathlib import Path

DATA = {
    "clip16.mp4": {0: 100, 1: 86, 2: 70, 3: 52, 4: 46, 5: 40, 6: 37, 7: 35, 8: 32, 9: 30, 11: 0},
    "clip01.mp4": {i: v for i, v in enumerate([100, 85, 75, 71, 66, 63, 62, 60, 59, 57, 55, 54, 53, 0])},
    "clip13.mp4": {i: v for i, v in enumerate(
        [100, 81, 63, 53, 46, 42, 40, 37, 34, 32, 32, 31, 30, 29, 27, 25, 24, 22, 19, 18])},
    "clip06.mp4": {i: v for i, v in enumerate(
        [100, 82, 72, 65, 60, 58, 55, 51, 46, 44, 44, 43, 42, 41, 40, 39, 49, 38, 36, 35, 34, 34, 34,
         33, 32, 31, 30, 28, 25, 24, 23, 22, 22, 21, 21, 21, 21, 20, 18, 17, 15, 14, 13, 12, 12, 11,
         11, 11, 11, 11, 10, 10, 10, 9, 9, 9, 0])},
}

out = Path("data/validation_recent/retention.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "sec", "ret"])
    for fn, d in DATA.items():
        for s, v in sorted(d.items()):
            w.writerow([fn, s, v])
print(f"[ok] {sum(len(d) for d in DATA.values())} puntos de retencion -> {out}")
