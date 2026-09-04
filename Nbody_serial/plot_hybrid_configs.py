#!/usr/bin/env python3
"""Plot fixed-resource hybrid MPI x OpenMP configurations.

The generic scaling plot uses total resources on the x-axis.  That is correct
for pure strong/weak scaling, but it is not useful for the hybrid experiment
where all points use the same total number of cores and only the P x T
decomposition changes.  This helper uses the configuration label as a
categorical x-axis.
"""
import csv
import sys

if len(sys.argv) < 2:
    raise SystemExit("usage: plot_hybrid_configs.py hybrid_summary.csv [prefix]")

src = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "hybrid_config"

rows = []
with open(src, newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("kind") != "strong":
            continue
        row["ranks"] = int(row["ranks"])
        row["threads"] = int(row["threads"])
        row["total_median"] = float(row["total_median"])
        row["total_stdev"] = float(row.get("total_stdev", 0.0) or 0.0)
        row["gpairs_median"] = float(row["gpairs_median"])
        row["label"] = f'{row["ranks"]}x{row["threads"]}'
        rows.append(row)

rows.sort(key=lambda row: (-row["ranks"], row["threads"]))

def write_plot(metric, ylabel, path, lower_is_better=False):
    if not rows:
        return
    width, height = 920, 520
    left, top, right, bottom = 85, 45, 35, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_y = max(row[metric] for row in rows) * 1.08
    min_y = min(row[metric] for row in rows)
    if lower_is_better:
        min_y = min_y * 0.98
    else:
        min_y = 0.0

    def xs(index):
        return left + plot_w * index / max(1, len(rows) - 1)

    def ys(value):
        if max_y == min_y:
            return top + plot_h / 2
        return top + plot_h * (1 - (value - min_y) / (max_y - min_y))

    points = " ".join(f"{xs(i):.1f},{ys(row[metric]):.1f}" for i, row in enumerate(rows))
    color = "#1f77b4" if lower_is_better else "#2ca02c"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{ylabel} by P x T configuration</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">MPI ranks x OpenMP threads</text>',
        f'<text transform="translate(24,{height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{ylabel}</text>',
    ]

    for i, row in enumerate(rows):
        x = xs(i)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="black"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{row["label"]}</text>')

    for i in range(6):
        value = min_y + (max_y - min_y) * i / 5
        parts.append(f'<line x1="{left-5}" y1="{ys(value):.1f}" x2="{left}" y2="{ys(value):.1f}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{ys(value)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.3g}</text>')

    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for i, row in enumerate(rows):
        x = xs(i)
        y = ys(row[metric])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{row[metric]:.3f}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    print(f"wrote {path}")

write_plot("total_median", "median time (s)", f"{prefix}_time_by_config.svg", lower_is_better=True)
write_plot("gpairs_median", "Gpairs/s", f"{prefix}_gpairs_by_config.svg")

