#!/usr/bin/env python3
"""Generate explanatory SVG plots used by the final report.

These plots are not generic scaling plots.  They summarize the evidence that
supports the discussion: bottleneck breakdown, ablation experiments, layout,
energy diagnostic overhead, and OSU native-vs-container microbenchmarks.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, parts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {path}")


def palette(index: int) -> str:
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    return colors[index % len(colors)]


def axes(width: int, height: int, left: int, top: int, right: int, bottom: int,
         title: str, xlabel: str, ylabel: str) -> tuple[list[str], int, int]:
    plot_w = width - left - right
    plot_h = height - top - bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">{xlabel}</text>',
        f'<text transform="translate(24,{height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{ylabel}</text>',
    ]
    return parts, plot_w, plot_h


def finish(parts: list[str], path: Path) -> None:
    parts.append("</svg>")
    write(path, parts)


def plot_bottleneck(summary_csv: Path, out: Path) -> None:
    rows = [r for r in read_csv(summary_csv) if r["kind"] == "strong"]
    rows.sort(key=lambda r: int(r["resources"]))
    width, height, left, top, right, bottom = 920, 520, 80, 45, 35, 80
    parts, plot_w, plot_h = axes(width, height, left, top, right, bottom,
                                 "strong scaling timing breakdown", "MPI ranks", "median time (s)")
    max_y = max(float(r["total_median"]) for r in rows)

    def x_of(i: int) -> float:
        return left + plot_w * i / max(1, len(rows) - 1)

    def y_of(v: float) -> float:
        return top + plot_h * (1.0 - v / max_y)

    stack_colors = [("#1f77b4", "force"), ("#d62728", "comm_wait"), ("#7f7f7f", "other")]
    bar_w = min(58, plot_w / max(1, len(rows)) * 0.55)
    for i, row in enumerate(rows):
        total = float(row["total_median"])
        force = float(row["force_median"])
        comm = float(row["comm_wait_median"])
        other = max(0.0, total - force - comm)
        x = x_of(i)
        y_base = top + plot_h
        for color, _, value in [(stack_colors[0][0], "force", force),
                                (stack_colors[1][0], "comm", comm),
                                (stack_colors[2][0], "other", other)]:
            h = plot_h * value / max_y
            y_base -= h
            parts.append(f'<rect x="{x - bar_w/2:.1f}" y="{y_base:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{row["ranks"]}</text>')

    for i in range(6):
        value = max_y * i / 5
        y = y_of(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2g}</text>')

    for i, (color, label) in enumerate(stack_colors):
        parts.append(f'<rect x="{left+10}" y="{top+15+i*20}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{left+28}" y="{top+26+i*20}" font-family="sans-serif" font-size="13">{label}</text>')
    finish(parts, out)


def plot_ablation(csv_path: Path, out: Path) -> None:
    rows = read_csv(csv_path)
    order = [("Kernel", "direct"), ("Kernel", "newton"),
             ("Math", "exact"), ("Math", "approx"),
             ("Comm", "sendrecv"), ("Comm", "overlap")]
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        groups.setdefault((row["Test_Type"], row["Config"]), []).append(float(row["Time_Sec"]))
    values = [(kind, config, sorted(vals)[len(vals)//2]) for kind, config in order if (vals := groups.get((kind, config)))]
    width, height, left, top, right, bottom = 920, 520, 85, 45, 35, 105
    parts, plot_w, plot_h = axes(width, height, left, top, right, bottom,
                                 "ablation experiments", "variant", "median time (s)")
    max_y = max(v for _, _, v in values) * 1.1
    bar_w = plot_w / len(values) * 0.55

    def y_of(v: float) -> float:
        return top + plot_h * (1.0 - v / max_y)

    for i, (kind, config, value) in enumerate(values):
        x = left + plot_w * (i + 0.5) / len(values)
        y = y_of(value)
        h = top + plot_h - y
        parts.append(f'<rect x="{x-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{palette(i)}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.3g}</text>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{kind}</text>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+38}" text-anchor="middle" font-family="sans-serif" font-size="11">{config}</text>')
    for i in range(6):
        value = max_y * i / 5
        y = y_of(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2g}</text>')
    finish(parts, out)


def plot_layout(csv_path: Path, out: Path) -> None:
    rows = read_csv(csv_path)
    threads = sorted({int(r["threads"]) for r in rows})
    layouts = ["aos", "soa"]
    width, height, left, top, right, bottom = 900, 520, 80, 45, 35, 80
    parts, plot_w, plot_h = axes(width, height, left, top, right, bottom,
                                 "AoS vs SoA force benchmark", "OpenMP threads", "median force time (s)")
    max_y = max(float(r["force_median"]) for r in rows) * 1.05

    def x_of(thread: int) -> float:
        return left + plot_w * threads.index(thread) / max(1, len(threads) - 1)

    def y_of(v: float) -> float:
        return top + plot_h * (1.0 - v / max_y)

    for li, layout in enumerate(layouts):
        points = []
        color = palette(li)
        for t in threads:
            row = next(r for r in rows if r["layout"] == layout and int(r["threads"]) == t)
            points.append(f'{x_of(t):.1f},{y_of(float(row["force_median"])):.1f}')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for t in threads:
            row = next(r for r in rows if r["layout"] == layout and int(r["threads"]) == t)
            parts.append(f'<circle cx="{x_of(t):.1f}" cy="{y_of(float(row["force_median"])):.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{left+10}" y="{top+20+li*20}" font-family="sans-serif" font-size="13" fill="{color}">{layout}</text>')
    for t in threads:
        parts.append(f'<text x="{x_of(t):.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{t}</text>')
    for i in range(6):
        value = max_y * i / 5
        y = y_of(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2g}</text>')
    finish(parts, out)


def plot_energy(csv_path: Path, out: Path) -> None:
    rows = read_csv(csv_path)
    rows.sort(key=lambda r: int(r["energy_every"]))
    width, height, left, top, right, bottom = 900, 520, 85, 45, 35, 80
    parts, plot_w, plot_h = axes(width, height, left, top, right, bottom,
                                 "energy diagnostic overhead", "energy_every", "overhead vs sparse (%)")
    max_y = max(float(r["overhead_vs_sparse_percent"]) for r in rows) * 1.08

    def x_of(i: int) -> float:
        return left + plot_w * i / max(1, len(rows) - 1)

    def y_of(v: float) -> float:
        return top + plot_h * (1.0 - v / max_y)

    points = []
    for i, row in enumerate(rows):
        value = float(row["overhead_vs_sparse_percent"])
        x = x_of(i)
        y = y_of(value)
        points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#d62728"/>')
        parts.append(f'<text x="{x:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.1f}%</text>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{row["energy_every"]}</text>')
    parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#d62728" stroke-width="2"/>')
    for i in range(6):
        value = max_y * i / 5
        y = y_of(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2g}</text>')
    finish(parts, out)


def plot_osu(csv_path: Path, out_prefix: Path) -> None:
    rows = read_csv(csv_path)
    for bench, metric, ylabel, suffix in [
        ("latency", "latency_us", "median latency (us)", "latency"),
        ("bandwidth", "bandwidth_MBps", "median bandwidth (MB/s)", "bandwidth"),
    ]:
        selected = [r for r in rows if r["benchmark"] == bench and r["metric"] == metric]
        if not selected:
            continue
        sizes = sorted({int(r["bytes"]) for r in selected})
        modes = ["native", "container"]
        width, height, left, top, right, bottom = 920, 520, 85, 45, 35, 90
        parts, plot_w, plot_h = axes(width, height, left, top, right, bottom,
                                     f"OSU {bench}: native vs container", "message size (bytes)", ylabel)
        max_y = max(float(r["median"]) for r in selected) * 1.08

        def x_of(size: int) -> float:
            return left + plot_w * sizes.index(size) / max(1, len(sizes) - 1)

        def y_of(v: float) -> float:
            return top + plot_h * (1.0 - v / max_y)

        for mi, mode in enumerate(modes):
            color = palette(mi)
            points = []
            by_size = {int(r["bytes"]): r for r in selected if r["mode"] == mode}
            for size in sizes:
                row = by_size.get(size)
                if row is None:
                    continue
                points.append(f'{x_of(size):.1f},{y_of(float(row["median"])):.1f}')
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            for size in sizes:
                row = by_size.get(size)
                if row is not None:
                    parts.append(f'<circle cx="{x_of(size):.1f}" cy="{y_of(float(row["median"])):.1f}" r="3" fill="{color}"/>')
            parts.append(f'<text x="{left+10}" y="{top+20+mi*20}" font-family="sans-serif" font-size="13" fill="{color}">{mode}</text>')
        # Label powers of two sparsely to keep the axis readable.
        label_sizes = [sizes[0], 1024, 1048576, sizes[-1]]
        for size in [s for s in label_sizes if s in sizes]:
            parts.append(f'<text x="{x_of(size):.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{size}</text>')
        for i in range(6):
            value = max_y * i / 5
            y = y_of(value)
            parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2g}</text>')
        finish(parts, out_prefix.with_name(out_prefix.name + f"_{suffix}.svg"))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_report_evidence.py Nbody_serial_root")
    root = Path(sys.argv[1])
    run64 = root / "runs/orfeo_64_20260818_195347"
    main_run = root / "runs/orfeo_seq_20260817_120943"
    plot_bottleneck(
        run64 / "01_mpi_64_retry/results_1_mpi_orfeo_summary.csv",
        run64 / "01_mpi_64_retry/results_1_mpi_orfeo_strong_breakdown.svg",
    )
    plot_ablation(
        run64 / "03_ablation_64_retry/results_3_ablation_orfeo.csv",
        run64 / "03_ablation_64_retry/results_3_ablation_orfeo.svg",
    )
    plot_layout(
        main_run / "04_evidence/layout_summary.csv",
        main_run / "04_evidence/layout_force_time.svg",
    )
    plot_energy(
        main_run / "04_evidence/energy_overhead_summary.csv",
        main_run / "04_evidence/energy_overhead.svg",
    )
    plot_osu(
        main_run / "07_container_pdf_final/osu_microbench_summary.csv",
        main_run / "07_container_pdf_final/osu_microbench",
    )


if __name__ == "__main__":
    main()

