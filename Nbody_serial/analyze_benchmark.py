#!/usr/bin/env python3
import csv
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    raise SystemExit("usage: analyze_benchmark.py benchmark_results.csv [summary.csv]")

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "benchmark_summary.csv"

rows = []
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        row["N"] = int(row["N"])
        row["ranks"] = int(row["ranks"])
        row["threads"] = int(row["threads"])
        row["resources"] = row["ranks"] * row["threads"]
        for key in ("total", "io", "drift", "force", "kick", "energy", "gpairs"):
            row[key] = float(row[key])
        rows.append(row)

groups = defaultdict(list)
for row in rows:
    key = (row["kind"], row["N"], row["ranks"], row["threads"],
           row["integrator"], row["comm"], row.get("kernel", "direct"),
           row.get("rsqrt", "exact"))
    groups[key].append(row)

summary = []
for key, values in sorted(groups.items()):
    kind, n, ranks, threads, integrator, comm, kernel, rsqrt = key
    totals = [v["total"] for v in values]
    forces = [v["force"] for v in values]
    gpairs = [v["gpairs"] for v in values]
    summary.append({
        "kind": kind,
        "N": n,
        "ranks": ranks,
        "threads": threads,
        "resources": ranks * threads,
        "integrator": integrator,
        "comm": comm,
        "kernel": kernel,
        "rsqrt": rsqrt,
        "runs": len(values),
        "total_median": statistics.median(totals),
        "total_stdev": statistics.stdev(totals) if len(totals) > 1 else 0.0,
        "force_median": statistics.median(forces),
        "gpairs_median": statistics.median(gpairs),
        "all_ok": all(v["status"] == "OK" for v in values),
    })

baselines = {}
for row in summary:
    if row["kind"] == "strong":
        base_key = (row["kind"], row["N"], row["integrator"], row["comm"],
                    row["kernel"], row["rsqrt"])
        old = baselines.get(base_key)
        if old is None or row["resources"] < old["resources"]:
            baselines[base_key] = row
    else:
        base_key = (row["kind"], row["integrator"], row["comm"],
                    row["kernel"], row["rsqrt"])
        old = baselines.get(base_key)
        if old is None or row["resources"] < old["resources"]:
            baselines[base_key] = row

for row in summary:
    if row["kind"] == "strong":
        base = baselines[(row["kind"], row["N"], row["integrator"], row["comm"],
                          row["kernel"], row["rsqrt"])]
    else:
        base = baselines[(row["kind"], row["integrator"], row["comm"],
                          row["kernel"], row["rsqrt"])]
    resource_ratio = row["resources"] / base["resources"]
    speedup = base["total_median"] / row["total_median"]
    row["speedup"] = speedup
    row["efficiency"] = speedup / resource_ratio

fields = [
    "kind", "N", "ranks", "threads", "resources", "integrator", "comm",
    "kernel", "rsqrt",
    "runs", "total_median", "total_stdev", "force_median", "gpairs_median",
    "speedup", "efficiency", "all_ok",
]
with open(dst, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

print(f"wrote {dst}")
