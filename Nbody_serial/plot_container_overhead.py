#!/usr/bin/env python3
"""Write compact SVG plots for native-versus-container timing and overhead."""
import csv
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    raise SystemExit("usage: plot_container_overhead.py container_overhead_summary.csv [prefix]")

prefix = sys.argv[2] if len(sys.argv) > 2 else "container_overhead"
groups = defaultdict(list)
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        row["ranks"] = int(row["ranks"])
        for key in ("native_median", "container_median", "overhead_percent"):
            row[key] = float(row[key])
        groups[row.get("kind", "strong")].append(row)

def svg(kind, rows):
    rows.sort(key=lambda row: row["ranks"])
    width, height, left, bottom = 820, 450, 70, 55
    top, right = 35, 30
    pw, ph = width-left-right, height-top-bottom
    max_x = max(row["ranks"] for row in rows)
    max_y = max(max(row["native_median"], row["container_median"]) for row in rows) or 1
    xs = lambda x: left + pw * (x - 1) / max(1, max_x - 1)
    ys = lambda y: top + ph * (1 - y/max_y)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="18">{kind}: native vs container (median)</text>',
             f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="black"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="black"/>']
    for color, key, label in [("#1f77b4", "native_median", "native"), ("#d62728", "container_median", "container")]:
        points = " ".join(f'{xs(r["ranks"]):.1f},{ys(r[key]):.1f}' for r in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{left+10}" y="{top+18 + (0 if key == "native_median" else 20)}" fill="{color}" font-family="sans-serif">{label}</text>')
    for row in rows:
        x = xs(row["ranks"])
        parts.append(f'<text x="{x:.1f}" y="{top+ph+18}" text-anchor="middle" font-family="sans-serif">{row["ranks"]}</text>')
        parts.append(f'<text x="{x:.1f}" y="{ys(row["container_median"])-8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{row["overhead_percent"]:+.1f}%</text>')
    parts.extend([f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-family="sans-serif">MPI ranks</text>',
                  f'<text transform="translate(18,{height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">seconds</text>', '</svg>'])
    path = f"{prefix}_{kind}.svg"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    print(f"wrote {path}")

for kind, rows in sorted(groups.items()):
    svg(kind, rows)
