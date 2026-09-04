#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    raise SystemExit("usage: plot_scaling.py benchmark_summary.csv [prefix]")

src = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "scaling"

rows = []
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        row["resources"] = int(row["resources"])
        row["speedup"] = float(row["speedup"])
        row["efficiency"] = float(row["efficiency"])
        row["total_median"] = float(row.get("total_median", 0.0) or 0.0)
        row["comm_bandwidth_GBps"] = float(row.get("comm_bandwidth_GBps", 0.0) or 0.0)
        row["N"] = int(row["N"])
        rows.append(row)

groups = defaultdict(list)
for row in rows:
    key = (row["kind"], row["integrator"], row["comm"],
           row.get("kernel", "direct"), row.get("rsqrt", "exact"))
    groups[key].append(row)

def polyline(points, xkey, ykey, xscale, yscale):
    return " ".join(f"{xscale(p[xkey]):.1f},{yscale(p[ykey]):.1f}" for p in points)

def ideal_points(x_values, metric):
    points = []
    for x in x_values:
        if metric == "speedup":
            y = float(x)
        elif metric == "efficiency":
            y = 1.0
        else:
            continue
        points.append({"resources": x, metric: y})
    return points

def write_plot(kind, metric, ylabel, path):
    selected = {k: sorted(v, key=lambda r: r["resources"])
                for k, v in groups.items() if k[0] == kind}
    if not selected:
        return
    max_x = max(r["resources"] for rows_ in selected.values() for r in rows_)
    x_ticks = sorted({r["resources"] for rows_ in selected.values() for r in rows_})
    if len(x_ticks) < 2:
        print(f"skipping {path}: only one resource value; use a configuration plot instead")
        return
    max_y = max(r[metric] for rows_ in selected.values() for r in rows_)
    if kind == "strong" and metric == "speedup":
        max_y = max(max_y, float(max_x))
    if metric == "efficiency":
        max_y = max(max_y, 1.0)
    max_y = max(max_y, 1.0)
    width, height = 900, 520
    left, top, right, bottom = 80, 40, 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_positions = {x: left + plot_w * idx / max(1, len(x_ticks) - 1)
                   for idx, x in enumerate(x_ticks)}
    def xs(x):
        return x_positions[x]
    def ys(y):
        return top + plot_h * (1 - y / max_y)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{ylabel if ylabel.startswith(kind) else kind + " " + ylabel}</text>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">resources = ranks x threads</text>',
        f'<text transform="translate(22,{height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{ylabel}</text>',
    ]
    for x in x_ticks:
        parts.append(f'<line x1="{xs(x):.1f}" y1="{top + plot_h}" x2="{xs(x):.1f}" y2="{top + plot_h + 5}" stroke="black"/>')
        parts.append(f'<text x="{xs(x):.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{x}</text>')
    for i in range(6):
        y = max_y * i / 5
        parts.append(f'<line x1="{left-5}" y1="{ys(y):.1f}" x2="{left}" y2="{ys(y):.1f}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{ys(y)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{y:.2g}</text>')

    if kind == "strong" and metric in ("speedup", "efficiency"):
        ideal = ideal_points(x_ticks, metric)
        parts.append(f'<polyline points="{polyline(ideal, "resources", metric, xs, ys)}" fill="none" stroke="#555" stroke-width="2" stroke-dasharray="6,4"/>')
        parts.append(f'<text x="{left + 10}" y="{top + 20}" font-family="sans-serif" font-size="13" fill="#555">ideal</text>')
        legend_offset = 1
    else:
        legend_offset = 0

    for idx, (key, points) in enumerate(sorted(selected.items())):
        color = colors[idx % len(colors)]
        label = "/".join(key[1:])
        parts.append(f'<polyline points="{polyline(points, "resources", metric, xs, ys)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for p in points:
            parts.append(f'<circle cx="{xs(p["resources"]):.1f}" cy="{ys(p[metric]):.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{left + 10}" y="{top + 20 + (idx + legend_offset)*20}" font-family="sans-serif" font-size="13" fill="{color}">{label}</text>')
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {path}")

def write_weak_time_plot(path):
    selected = {k: sorted(v, key=lambda r: r["resources"])
                for k, v in groups.items() if k[0] == "weak"}
    if not selected:
        return
    max_x = max(r["resources"] for rows_ in selected.values() for r in rows_)
    x_ticks = sorted({r["resources"] for rows_ in selected.values() for r in rows_})
    if len(x_ticks) < 2:
        print(f"skipping {path}: only one resource value; use a configuration plot instead")
        return
    max_y = max(r["total_median"] for rows_ in selected.values() for r in rows_)
    width, height = 900, 520
    left, top, right, bottom = 80, 40, 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_positions = {x: left + plot_w * idx / max(1, len(x_ticks) - 1)
                   for idx, x in enumerate(x_ticks)}
    def xs(x):
        return x_positions[x]
    def ys(y):
        return top + plot_h * (1 - y / max_y)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">weak absolute time</text>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">resources = ranks x threads</text>',
        f'<text transform="translate(22,{height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">median time (s)</text>',
    ]
    for x in x_ticks:
        parts.append(f'<line x1="{xs(x):.1f}" y1="{top + plot_h}" x2="{xs(x):.1f}" y2="{top + plot_h + 5}" stroke="black"/>')
        parts.append(f'<text x="{xs(x):.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{x}</text>')
    for i in range(6):
        y = max_y * i / 5
        parts.append(f'<line x1="{left-5}" y1="{ys(y):.1f}" x2="{left}" y2="{ys(y):.1f}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{ys(y)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{y:.2g}</text>')
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for idx, (key, points) in enumerate(sorted(selected.items())):
        color = colors[idx % len(colors)]
        label = "/".join(key[1:])
        parts.append(f'<polyline points="{polyline(points, "resources", "total_median", xs, ys)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for p in points:
            parts.append(f'<circle cx="{xs(p["resources"]):.1f}" cy="{ys(p["total_median"]):.1f}" r="4" fill="{color}"/>')
        # For the direct all-pairs weak-scaling experiment, the relevant
        # reference is O(P), not constant time, because N/P is fixed while each
        # local particle interacts with the full global N.
        base = min(points, key=lambda r: r["resources"])
        ideal = [{"resources": x, "total_median": base["total_median"] * (x / base["resources"])}
                 for x in x_ticks]
        parts.append(f'<polyline points="{polyline(ideal, "resources", "total_median", xs, ys)}" fill="none" stroke="#555" stroke-width="2" stroke-dasharray="6,4"/>')
        parts.append(f'<text x="{left + 10}" y="{top + 20 + idx*20}" font-family="sans-serif" font-size="13" fill="{color}">{label}</text>')
    parts.append(f'<text x="{left + 10}" y="{top + 40 + len(selected)*20}" font-family="sans-serif" font-size="13" fill="#555">dashed: ideal O(P) for direct all-pairs with fixed N/P</text>')
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {path}")

write_plot("strong", "speedup", "speedup", f"{prefix}_strong_speedup.svg")
write_plot("strong", "efficiency", "efficiency", f"{prefix}_strong_efficiency.svg")
write_plot("strong", "comm_bandwidth_GBps", "estimated communication GB/s", f"{prefix}_strong_comm_bandwidth.svg")
write_plot("weak", "speedup", "relative throughput", f"{prefix}_weak_speedup.svg")
write_plot("weak", "efficiency", "weak efficiency", f"{prefix}_weak_efficiency.svg")
write_plot("weak", "comm_bandwidth_GBps", "estimated communication GB/s", f"{prefix}_weak_comm_bandwidth.svg")
write_weak_time_plot(f"{prefix}_weak_time.svg")
