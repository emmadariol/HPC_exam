#!/usr/bin/env python3
"""Unified CSV analysis and SVG plotting CLI for the N-body project."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def median_absolute_deviation(values: list[float]) -> float:
    med = statistics.median(values)
    return statistics.median(abs(v - med) for v in values)


def keep_non_outliers(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int, float]:
    totals = [float(r["total"]) for r in rows]
    med = statistics.median(totals)
    mad = median_absolute_deviation(totals)
    if mad == 0.0:
        return rows, 0, mad
    sigma = 1.4826 * mad
    kept = [r for r in rows if abs(float(r["total"]) - med) <= 3.0 * sigma]
    return (kept or rows), len(rows) - len(kept or rows), mad


def dtype_size(dtype: str) -> int:
    return 4 if dtype == "float" else 8


def finite(row: dict[str, object], keys: tuple[str, ...]) -> bool:
    return all(math.isfinite(float(row[k])) for k in keys)


def summarize_scaling(src: str, dst: str) -> None:
    rows: list[dict[str, object]] = []
    for row in read_csv(src):
        for key in ("N", "nsteps", "ranks", "threads"):
            row[key] = int(row[key])
        row["resources"] = int(row["ranks"]) * int(row["threads"])
        row["dtype"] = row.get("dtype", "double") or "double"
        for key in ("total", "io", "drift", "force", "comm_wait", "kick", "energy", "gpairs"):
            row[key] = float(row.get(key, "nan") or "nan")
        rows.append(row)

    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["kind"], row["N"], row["ranks"], row["threads"],
            row.get("integrator", "kdk"), row.get("comm", "sendrecv"),
            row.get("kernel", "direct"), row.get("rsqrt", "exact"),
            row.get("accumulators", "4"),
        )
        groups[key].append(row)

    summary: list[dict[str, object]] = []
    for key, values in sorted(groups.items()):
        kind, n, ranks, threads, integrator, comm, kernel, rsqrt, accumulators = key
        valid = [
            v for v in values
            if v.get("status") in ("OK", "WARNING")
            and finite(v, ("total", "force", "comm_wait", "gpairs"))
        ]
        if not valid:
            continue
        kept, outliers, total_mad = keep_non_outliers(valid)
        totals = [float(v["total"]) for v in kept]
        forces = [float(v["force"]) for v in kept]
        comm_waits = [float(v["comm_wait"]) for v in kept]
        gpairs = [float(v["gpairs"]) for v in kept]
        nsteps = max(int(v["nsteps"]) for v in values)
        force_evals = nsteps + (1 if integrator == "kdk" else 0)
        bytes_per_rank = (
            force_evals * max(0, int(ranks) - 1) *
            ((int(n) + int(ranks) - 1) // int(ranks)) * 3 *
            dtype_size(str(values[0].get("dtype", "double")))
        )
        comm_wait_median = statistics.median(comm_waits)
        summary.append({
            "kind": kind,
            "N": n,
            "nsteps": nsteps,
            "ranks": ranks,
            "threads": threads,
            "resources": int(ranks) * int(threads),
            "integrator": integrator,
            "comm": comm,
            "kernel": kernel,
            "rsqrt": rsqrt,
            "accumulators": accumulators,
            "dtype": values[0].get("dtype", "double"),
            "runs": len(values),
            "failed_runs": len(values) - len(valid),
            "used_runs": len(kept),
            "outliers": outliers,
            "total_mad": total_mad,
            "total_median": statistics.median(totals),
            "total_stdev": statistics.stdev(totals) if len(totals) > 1 else 0.0,
            "force_median": statistics.median(forces),
            "comm_wait_median": comm_wait_median,
            "comm_bandwidth_GBps": bytes_per_rank / comm_wait_median / 1.0e9 if comm_wait_median > 0 else 0.0,
            "gpairs_median": statistics.median(gpairs),
            "all_ok": all(v["status"] == "OK" for v in valid) and len(valid) == len(values),
        })

    baselines: dict[tuple[object, ...], dict[str, object]] = {}
    for row in summary:
        if row["kind"] == "strong":
            base_key = (
                row["kind"], row["N"], row["integrator"], row["comm"],
                row["kernel"], row["rsqrt"], row["accumulators"],
            )
        else:
            base_key = (
                row["kind"], row["integrator"], row["comm"],
                row["kernel"], row["rsqrt"], row["accumulators"],
            )
        if base_key not in baselines or int(row["resources"]) < int(baselines[base_key]["resources"]):
            baselines[base_key] = row

    for row in summary:
        if row["kind"] == "strong":
            base_key = (
                row["kind"], row["N"], row["integrator"], row["comm"],
                row["kernel"], row["rsqrt"], row["accumulators"],
            )
        else:
            base_key = (
                row["kind"], row["integrator"], row["comm"],
                row["kernel"], row["rsqrt"], row["accumulators"],
            )
        base = baselines[base_key]
        resource_ratio = float(row["resources"]) / float(base["resources"])
        speedup = float(base["total_median"]) / float(row["total_median"])
        row["speedup"] = speedup
        row["efficiency"] = speedup / resource_ratio
        if row["kind"] == "weak":
            row["weak_normalized_time"] = float(row["total_median"]) / (
                resource_ratio * float(base["total_median"])
            )
        else:
            row["weak_normalized_time"] = ""

    fields = [
        "kind", "N", "nsteps", "ranks", "threads", "resources", "integrator",
        "comm", "kernel", "rsqrt", "accumulators", "dtype", "runs",
        "failed_runs", "used_runs", "outliers", "total_mad", "total_median",
        "total_stdev", "force_median", "comm_wait_median",
        "comm_bandwidth_GBps", "gpairs_median", "speedup", "efficiency",
        "weak_normalized_time", "all_ok",
    ]
    write_csv(dst, fields, summary)


def summarize_layout(src: str, dst: str) -> None:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in read_csv(src):
        row["N"] = int(row["N"])
        row["threads"] = int(row["threads"])
        row["force"] = float(row["force"])
        row["gpairs"] = float(row["gpairs"])
        row["checksum"] = float(row["checksum"])
        groups[(row["layout"], row["N"], row["threads"], row["rsqrt"])].append(row)
    rows: list[dict[str, object]] = []
    for (layout, n, threads, rsqrt), values in sorted(groups.items()):
        rows.append({
            "layout": layout,
            "N": n,
            "threads": threads,
            "rsqrt": rsqrt,
            "runs": len(values),
            "force_median": statistics.median(float(v["force"]) for v in values),
            "force_stdev": statistics.stdev(float(v["force"]) for v in values) if len(values) > 1 else 0.0,
            "gpairs_median": statistics.median(float(v["gpairs"]) for v in values),
            "checksum_median": statistics.median(float(v["checksum"]) for v in values),
        })
    by_case = {(r["N"], r["threads"], r["rsqrt"], r["layout"]): r for r in rows}
    for row in rows:
        aos = by_case.get((row["N"], row["threads"], row["rsqrt"], "aos"))
        if row["layout"] == "soa" and aos:
            diff = abs(float(row["checksum_median"]) - float(aos["checksum_median"]))
            denom = max(abs(float(row["checksum_median"])), abs(float(aos["checksum_median"])), 1.0)
            row["soa_vs_aos_speedup"] = float(aos["force_median"]) / float(row["force_median"])
            row["checksum_abs_diff_vs_aos"] = diff
            row["checksum_rel_diff_vs_aos"] = diff / denom
        else:
            row["soa_vs_aos_speedup"] = ""
            row["checksum_abs_diff_vs_aos"] = ""
            row["checksum_rel_diff_vs_aos"] = ""
    fields = [
        "layout", "N", "threads", "rsqrt", "runs", "force_median",
        "force_stdev", "gpairs_median", "checksum_median",
        "soa_vs_aos_speedup", "checksum_abs_diff_vs_aos",
        "checksum_rel_diff_vs_aos",
    ]
    write_csv(dst, fields, rows)


def summarize_energy(src: str, dst: str) -> None:
    groups: dict[tuple[int, int, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in read_csv(src):
        for key in ("N", "nsteps", "ranks", "threads", "energy_every"):
            row[key] = int(row[key])
        for key in ("total", "force", "energy", "max_rel_drift"):
            row[key] = float(row[key])
        groups[(row["N"], row["nsteps"], row["ranks"], row["threads"], row["energy_every"])].append(row)
    rows: list[dict[str, object]] = []
    for (n, nsteps, ranks, threads, ee), values in sorted(groups.items()):
        valid = [v for v in values if v["status"] in ("OK", "WARNING") and finite(v, ("total", "force", "energy"))]
        if not valid:
            continue
        total = statistics.median(float(v["total"]) for v in valid)
        energy = statistics.median(float(v["energy"]) for v in valid)
        rows.append({
            "N": n,
            "nsteps": nsteps,
            "ranks": ranks,
            "threads": threads,
            "energy_every": ee,
            "runs": len(values),
            "failed_runs": len(values) - len(valid),
            "total_median": total,
            "force_median": statistics.median(float(v["force"]) for v in valid),
            "energy_median": energy,
            "energy_fraction": energy / total,
            "max_rel_drift": max(float(v["max_rel_drift"]) for v in valid),
            "all_ok": all(v["status"] == "OK" for v in valid) and len(valid) == len(values),
        })
    base: dict[tuple[int, int, int, int], dict[str, object]] = {}
    for row in rows:
        key = (int(row["N"]), int(row["nsteps"]), int(row["ranks"]), int(row["threads"]))
        if key not in base or int(row["energy_every"]) > int(base[key]["energy_every"]):
            base[key] = row
    for row in rows:
        b = base[(int(row["N"]), int(row["nsteps"]), int(row["ranks"]), int(row["threads"]))]
        row["overhead_vs_sparse_percent"] = 100.0 * (float(row["total_median"]) - float(b["total_median"])) / float(b["total_median"])
    fields = [
        "N", "nsteps", "ranks", "threads", "energy_every", "runs",
        "failed_runs", "total_median", "force_median", "energy_median",
        "energy_fraction", "overhead_vs_sparse_percent", "max_rel_drift", "all_ok",
    ]
    write_csv(dst, fields, rows)


def summarize_container(src: str, dst: str) -> None:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    failures: dict[tuple[object, ...], int] = defaultdict(int)
    for row in read_csv(src):
        key = (row.get("kind", "strong"), row["mode"], int(row["N"]), int(row["ranks"]), int(row["threads"]))
        try:
            total = float(row["total"])
        except ValueError:
            total = math.nan
        if row.get("status") in ("OK", "WARNING") and math.isfinite(total):
            groups[key].append(total)
        else:
            failures[key] += 1
    rows: list[dict[str, object]] = []
    configs = sorted({(kind, n, ranks, threads) for kind, _, n, ranks, threads in groups})
    for kind, n, ranks, threads in configs:
        native = groups.get((kind, "native", n, ranks, threads), [])
        cont = groups.get((kind, "container", n, ranks, threads), [])
        if not native or not cont:
            continue
        native_median = statistics.median(native)
        cont_median = statistics.median(cont)
        rows.append({
            "kind": kind,
            "N": n,
            "ranks": ranks,
            "threads": threads,
            "native_runs": len(native),
            "container_runs": len(cont),
            "native_failed_runs": failures.get((kind, "native", n, ranks, threads), 0),
            "container_failed_runs": failures.get((kind, "container", n, ranks, threads), 0),
            "native_median": native_median,
            "container_median": cont_median,
            "native_stdev": statistics.stdev(native) if len(native) > 1 else 0.0,
            "container_stdev": statistics.stdev(cont) if len(cont) > 1 else 0.0,
            "overhead_percent": 100.0 * (cont_median - native_median) / native_median,
        })
    fields = [
        "kind", "N", "ranks", "threads", "native_runs", "container_runs",
        "native_failed_runs", "container_failed_runs", "native_median",
        "container_median", "native_stdev", "container_stdev", "overhead_percent",
    ]
    write_csv(dst, fields, rows)


def summarize_osu(src: str, dst: str) -> None:
    groups: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in read_csv(src):
        groups[(row["mode"], row["benchmark"], row["metric"], int(row["bytes"]))].append(float(row["value"]))
    rows = []
    for (mode, bench, metric, nbytes), values in sorted(groups.items()):
        rows.append({
            "mode": mode,
            "benchmark": bench,
            "metric": metric,
            "bytes": nbytes,
            "runs": len(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    write_csv(dst, ["mode", "benchmark", "metric", "bytes", "runs", "median", "stdev"], rows)


def add_gflops(src: str, dst: str, flops_per_pair: float) -> None:
    with Path(src).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    rate_cols = [c for c in ("gpairs_median", "gpairs") if c in fields]
    if not rate_cols:
        raise SystemExit("CSV must contain gpairs_median or gpairs")
    for col in rate_cols:
        out_col = f"estimated_gflops_from_{col}"
        if out_col not in fields:
            fields.append(out_col)
        for row in rows:
            try:
                row[out_col] = f"{float(row[col]) * flops_per_pair:.9g}"
            except ValueError:
                row[out_col] = "nan"
    if "flops_per_pair_model" not in fields:
        fields.append("flops_per_pair_model")
    for row in rows:
        row["flops_per_pair_model"] = f"{flops_per_pair:g}"
    write_csv(dst, fields, rows)


def palette(i: int) -> str:
    return ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"][i % 6]


def write_svg(path: str | Path, parts: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")
    print(f"wrote {path}")


def axes(title: str, xlabel: str, ylabel: str, width: int = 900, height: int = 520) -> tuple[list[str], int, int, int, int, int, int]:
    left, top, right, bottom = 82, 42, 35, 75
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">{xlabel}</text>',
        f'<text transform="translate(24,{height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{ylabel}</text>',
    ]
    return parts, width, height, left, top, right, bottom


def plot_xy(
    rows: list[dict[str, object]],
    x_field: str,
    y_field: str,
    title: str,
    ylabel: str,
    path: str,
    ideal: str | None = None,
    xlabel: str = "resources = ranks x threads",
) -> None:
    rows = sorted(rows, key=lambda r: int(r[x_field]))
    if len(rows) < 2:
        print(f"skipping {path}: only one x value")
        return
    parts, width, height, left, top, right, bottom = axes(title, xlabel, ylabel)
    pw, ph = width - left - right, height - top - bottom
    xs_vals = [int(r[x_field]) for r in rows]
    ys_vals = [float(r[y_field]) for r in rows]
    min_x, max_x = min(xs_vals), max(xs_vals)
    max_y = max(max(ys_vals), 1.0)
    if ideal == "speedup":
        max_y = max(max_y, float(max_x))
    if ideal == "efficiency":
        max_y = max(max_y, 1.0)

    def xs(x: float) -> float:
        return left + pw * (x - min_x) / max(1.0, max_x - min_x)

    def ys(y: float) -> float:
        return top + ph * (1.0 - y / max_y)

    for x in xs_vals:
        parts.append(f'<text x="{xs(x):.1f}" y="{height-bottom+28}" text-anchor="middle" font-family="sans-serif" font-size="12">{x}</text>')
    for i in range(6):
        yv = max_y * i / 5
        parts.append(f'<line x1="{left-5}" y1="{ys(yv):.1f}" x2="{left}" y2="{ys(yv):.1f}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{ys(yv)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{yv:.2g}</text>')
    if ideal == "speedup":
        ipoints = " ".join(f"{xs(x):.1f},{ys(x):.1f}" for x in xs_vals)
        parts.append(f'<polyline points="{ipoints}" fill="none" stroke="#555" stroke-width="2" stroke-dasharray="6,4"/>')
        parts.append(f'<text x="{left+10}" y="{top+20}" font-family="sans-serif" font-size="13" fill="#555">ideal</text>')
    if ideal == "efficiency":
        parts.append(f'<line x1="{left}" y1="{ys(1.0):.1f}" x2="{width-right}" y2="{ys(1.0):.1f}" stroke="#555" stroke-width="2" stroke-dasharray="6,4"/>')
        parts.append(f'<text x="{left+10}" y="{top+20}" font-family="sans-serif" font-size="13" fill="#555">ideal</text>')
    points = " ".join(f"{xs(int(r[x_field])):.1f},{ys(float(r[y_field])):.1f}" for r in rows)
    parts.append(f'<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="2"/>')
    for r in rows:
        parts.append(f'<circle cx="{xs(int(r[x_field])):.1f}" cy="{ys(float(r[y_field])):.1f}" r="4" fill="#1f77b4"/>')
    write_svg(path, parts)


def plot_scaling(src: str, prefix: str) -> None:
    rows = read_csv(src)
    for r in rows:
        r["resources"] = int(r["resources"])
        for k in ("speedup", "efficiency", "total_median", "comm_bandwidth_GBps"):
            r[k] = float(r[k])
        r["weak_normalized_time"] = float(r["weak_normalized_time"]) if r.get("weak_normalized_time") else math.nan
    strong = [r for r in rows if r["kind"] == "strong"]
    weak = [r for r in rows if r["kind"] == "weak"]
    plot_xy(strong, "resources", "speedup", "strong scaling speedup", "speedup", f"{prefix}_strong_speedup.svg", "speedup")
    plot_xy(strong, "resources", "efficiency", "strong scaling efficiency", "efficiency", f"{prefix}_strong_efficiency.svg", "efficiency")
    plot_xy(strong, "resources", "comm_bandwidth_GBps", "strong communication bandwidth", "GB/s", f"{prefix}_strong_comm_bandwidth.svg")
    plot_xy(weak, "resources", "total_median", "weak scaling absolute time", "median time (s)", f"{prefix}_weak_time.svg")
    plot_xy([r for r in weak if math.isfinite(float(r["weak_normalized_time"]))], "resources", "weak_normalized_time", "weak normalized time", "T(P)/(P*T1)", f"{prefix}_weak_normalized_time.svg")


def plot_hybrid(src: str, prefix: str) -> None:
    rows = read_csv(src)
    for row in rows:
        row["label"] = f'P{row["ranks"]}xT{row["threads"]}'
        row["total_median"] = float(row["total_median"])
        row["gpairs_median"] = float(row["gpairs_median"])
    for metric, ylabel, suffix, lower in [
        ("total_median", "median time (s)", "time_by_config", True),
        ("gpairs_median", "Gpairs/s", "gpairs_by_config", False),
    ]:
        parts, width, height, left, top, right, bottom = axes(f"hybrid {ylabel}", "configuration", ylabel)
        pw, ph = width - left - right, height - top - bottom
        vals = [float(r[metric]) for r in rows]
        min_y = min(vals) if lower else 0.0
        max_y = max(vals) if max(vals) > min_y else min_y + 1.0
        def xs(i: int) -> float: return left + pw * i / max(1, len(rows) - 1)
        def ys(v: float) -> float: return top + ph * (1.0 - (v - min_y) / (max_y - min_y))
        points = " ".join(f"{xs(i):.1f},{ys(float(r[metric])):.1f}" for i, r in enumerate(rows))
        parts.append(f'<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="2"/>')
        for i, r in enumerate(rows):
            parts.append(f'<circle cx="{xs(i):.1f}" cy="{ys(float(r[metric])):.1f}" r="4" fill="#1f77b4"/>')
            parts.append(f'<text transform="translate({xs(i):.1f},{height-bottom+22}) rotate(-35)" text-anchor="end" font-family="sans-serif" font-size="11">{r["label"]}</text>')
        write_svg(f"{prefix}_{suffix}.svg", parts)


def plot_container(src: str, prefix: str) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(src):
        groups[row.get("kind", "strong")].append(row)
    for kind, rows in groups.items():
        rows.sort(key=lambda r: int(r["ranks"]))
        parts, width, height, left, top, right, bottom = axes(f"{kind}: native vs container median time", "MPI ranks", "seconds", 820, 450)
        pw, ph = width - left - right, height - top - bottom
        max_x = max(int(r["ranks"]) for r in rows)
        max_y = max(max(float(r["native_median"]), float(r["container_median"])) for r in rows) or 1
        def xs(x: int) -> float: return left + pw * (x - 1) / max(1, max_x - 1)
        def ys(y: float) -> float: return top + ph * (1.0 - y / max_y)
        for color, key, label in [("#1f77b4", "native_median", "native"), ("#d62728", "container_median", "container")]:
            points = " ".join(f'{xs(int(r["ranks"])):.1f},{ys(float(r[key])):.1f}' for r in rows)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
            for r in rows:
                parts.append(f'<circle cx="{xs(int(r["ranks"])):.1f}" cy="{ys(float(r[key])):.1f}" r="4" fill="{color}"/>')
            parts.append(f'<text x="{left+10}" y="{top+18 + (0 if key == "native_median" else 20)}" fill="{color}" font-family="sans-serif">{label}</text>')
        for r in rows:
            x = xs(int(r["ranks"]))
            parts.append(f'<text x="{x:.1f}" y="{height-bottom+18}" text-anchor="middle" font-family="sans-serif">{r["ranks"]}</text>')
            parts.append(f'<text x="{x:.1f}" y="{ys(float(r["container_median"]))-8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{float(r["overhead_percent"]):+.1f}%</text>')
        write_svg(f"{prefix}_{kind}.svg", parts)


def plot_ablation(src: str, prefix: str) -> None:
    raw = read_csv(src)
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in raw:
        try:
            value = float(row["Time_Sec"])
        except ValueError:
            continue
        if math.isfinite(value):
            groups[(row["Test_Type"], row["Config"])].append(value)
    order = [
        ("Kernel", "direct"), ("Kernel", "newton"),
        ("Math", "exact"), ("Math", "approx"),
        ("Comm", "sendrecv"), ("Comm", "overlap"),
        ("Accumulators", "1"), ("Accumulators", "4"),
    ]
    rows = [(kind, config, statistics.median(groups[(kind, config)])) for kind, config in order if groups.get((kind, config))]
    if not rows:
        print(f"skipping {prefix}.svg: no finite ablation rows")
        return
    parts, width, height, left, top, right, bottom = axes("ablation experiments", "variant", "median time (s)")
    pw, ph = width - left - right, height - top - bottom
    max_y = max(v for _, _, v in rows) * 1.1
    bar_w = pw / len(rows) * 0.55
    def y_of(v: float) -> float: return top + ph * (1.0 - v / max_y)
    for i, (kind, config, value) in enumerate(rows):
        x = left + pw * (i + 0.5) / len(rows)
        y = y_of(value)
        parts.append(f'<rect x="{x-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height-bottom-y:.1f}" fill="{palette(i)}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.3g}</text>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="10">{kind}</text>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+38}" text-anchor="middle" font-family="sans-serif" font-size="10">{config}</text>')
    write_svg(f"{prefix}.svg", parts)


def plot_layout(src: str, prefix: str) -> None:
    rows = read_csv(src)
    if not rows:
        return
    for row in rows:
        row["threads"] = int(row["threads"])
        row["force_median"] = float(row["force_median"])
    threads = sorted({int(r["threads"]) for r in rows})
    parts, width, height, left, top, right, bottom = axes("AoS vs SoA force benchmark", "OpenMP threads", "median force time (s)")
    pw, ph = width - left - right, height - top - bottom
    max_y = max(float(r["force_median"]) for r in rows) * 1.05
    def x_of(t: int) -> float: return left + pw * threads.index(t) / max(1, len(threads) - 1)
    def y_of(v: float) -> float: return top + ph * (1.0 - v / max_y)
    for li, layout in enumerate(["aos", "soa"]):
        selected = [r for r in rows if r["layout"] == layout]
        if not selected:
            continue
        by_t = {int(r["threads"]): r for r in selected}
        points = " ".join(f'{x_of(t):.1f},{y_of(float(by_t[t]["force_median"])):.1f}' for t in threads if t in by_t)
        color = palette(li)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for t in threads:
            if t in by_t:
                parts.append(f'<circle cx="{x_of(t):.1f}" cy="{y_of(float(by_t[t]["force_median"])):.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{left+10}" y="{top+20+li*20}" font-family="sans-serif" font-size="13" fill="{color}">{layout}</text>')
    for t in threads:
        parts.append(f'<text x="{x_of(t):.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{t}</text>')
    write_svg(f"{prefix}.svg", parts)


def plot_energy(src: str, prefix: str) -> None:
    rows = read_csv(src)
    for row in rows:
        row["energy_every"] = int(row["energy_every"])
        row["overhead_vs_sparse_percent"] = float(row["overhead_vs_sparse_percent"])
    rows.sort(key=lambda r: int(r["energy_every"]))
    plot_xy(
        rows,
        "energy_every",
        "overhead_vs_sparse_percent",
        "energy diagnostic overhead",
        "overhead vs sparse (%)",
        f"{prefix}.svg",
        xlabel="energy_every",
    )


def plot_osu(src: str, prefix: str) -> None:
    rows = read_csv(src)
    for row in rows:
        row["bytes"] = int(row["bytes"])
        row["median"] = float(row["median"])
    for bench, metric, ylabel, suffix in [
        ("latency", "latency_us", "median latency (us)", "latency"),
        ("bandwidth", "bandwidth_MBps", "median bandwidth (MB/s)", "bandwidth"),
    ]:
        selected = [r for r in rows if r["benchmark"] == bench and r["metric"] == metric]
        if not selected:
            continue
        sizes = sorted({int(r["bytes"]) for r in selected})
        parts, width, height, left, top, right, bottom = axes(f"OSU {bench}: native vs container", "message size (bytes, log2 scale)", ylabel)
        pw, ph = width - left - right, height - top - bottom
        max_y = max(float(r["median"]) for r in selected) * 1.08
        def x_of(size: int) -> float: return left + pw * sizes.index(size) / max(1, len(sizes) - 1)
        def y_of(v: float) -> float: return top + ph * (1.0 - v / max_y)
        for mi, mode in enumerate(["native", "container"]):
            by_size = {int(r["bytes"]): r for r in selected if r["mode"] == mode}
            if not by_size:
                continue
            color = palette(mi)
            points = " ".join(f'{x_of(s):.1f},{y_of(float(by_size[s]["median"])):.1f}' for s in sizes if s in by_size)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
            for s in sizes:
                if s in by_size:
                    parts.append(f'<circle cx="{x_of(s):.1f}" cy="{y_of(float(by_size[s]["median"])):.1f}" r="3" fill="{color}"/>')
            parts.append(f'<text x="{left+10}" y="{top+20+mi*20}" font-family="sans-serif" font-size="13" fill="{color}">{mode}</text>')
        for s in [sizes[0], 1024, 1048576, sizes[-1]]:
            if s in sizes:
                parts.append(f'<text x="{x_of(s):.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{s}</text>')
        write_svg(f"{prefix}_{suffix}.svg", parts)


def plot_evidence(root: str) -> None:
    base = Path(root)
    plot_scaling(str(base / "results_final/scaling_64_summary.csv"), str(base / "results_final/scaling_64"))
    plot_hybrid(str(base / "results_final/hybrid_64_summary.csv"), str(base / "results_final/hybrid_64"))
    plot_container(str(base / "results_final/container_overhead_summary.csv"), str(base / "results_final/container_overhead"))
    plot_ablation(str(base / "results_final/ablation_64.csv"), str(base / "results_final/ablation_64"))
    plot_layout(str(base / "results_final/layout_summary.csv"), str(base / "results_final/layout_force_time"))
    plot_energy(str(base / "results_final/energy_overhead_summary.csv"), str(base / "results_final/energy_overhead"))
    plot_osu(str(base / "results_final/osu_microbench_summary.csv"), str(base / "results_final/osu_microbench"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="section", required=True)

    s = sub.add_parser("summarize")
    ssub = s.add_subparsers(dest="kind", required=True)
    for name in ("scaling", "layout", "energy", "container", "osu"):
        p = ssub.add_parser(name)
        p.add_argument("input")
        p.add_argument("output")

    g = ssub.add_parser("gflops")
    g.add_argument("input")
    g.add_argument("output")
    g.add_argument("--flops-per-pair", type=float, default=20.0)

    p = sub.add_parser("plot")
    psub = p.add_subparsers(dest="kind", required=True)
    for name in ("scaling", "hybrid", "container", "ablation", "layout", "energy", "osu"):
        q = psub.add_parser(name)
        q.add_argument("input")
        q.add_argument("prefix")
    q = psub.add_parser("evidence")
    q.add_argument("root")

    args = parser.parse_args()
    if args.section == "summarize":
        if args.kind == "gflops":
            add_gflops(args.input, args.output, args.flops_per_pair)
        else:
            {
                "scaling": summarize_scaling,
                "layout": summarize_layout,
                "energy": summarize_energy,
                "container": summarize_container,
                "osu": summarize_osu,
            }[args.kind](args.input, args.output)
    elif args.section == "plot":
        if args.kind == "evidence":
            plot_evidence(args.root)
        else:
            {
                "scaling": plot_scaling,
                "hybrid": plot_hybrid,
                "container": plot_container,
                "ablation": plot_ablation,
                "layout": plot_layout,
                "energy": plot_energy,
                "osu": plot_osu,
            }[args.kind](
                args.input,
                args.prefix,
            )


if __name__ == "__main__":
    main()
