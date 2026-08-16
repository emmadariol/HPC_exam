#!/usr/bin/env python3
import csv
import math
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
        row["nsteps"] = int(row.get("nsteps", 0) or 0)
        row["ranks"] = int(row["ranks"])
        row["threads"] = int(row["threads"])
        row["resources"] = row["ranks"] * row["threads"]
        row["dtype"] = row.get("dtype", "double") or "double"
        for key in ("total", "io", "drift", "force", "comm_wait", "kick", "energy", "gpairs"):
            row[key] = float(row.get(key, 0.0) or 0.0)
        rows.append(row)

def median_absolute_deviation(values):
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])

def keep_non_outliers(values):
    totals = [v["total"] for v in values]
    med = statistics.median(totals)
    mad = median_absolute_deviation(totals)
    if mad == 0.0:
        return values, 0, mad
    sigma = 1.4826 * mad
    kept = [v for v in values if abs(v["total"] - med) <= 3.0 * sigma]
    if not kept:
        return values, 0, mad
    return kept, len(values) - len(kept), mad

def dtype_size(dtype_name):
    return 4 if dtype_name == "float" else 8

def is_valid_run(row):
    return (row.get("status") in ("OK", "WARNING") and
            all(math.isfinite(row[key]) for key in
                ("total", "force", "comm_wait", "gpairs")))

groups = defaultdict(list)
for row in rows:
    key = (row["kind"], row["N"], row["ranks"], row["threads"],
           row["integrator"], row["comm"], row.get("kernel", "direct"),
           row.get("rsqrt", "exact"))
    groups[key].append(row)

summary = []
for key, values in sorted(groups.items()):
    kind, n, ranks, threads, integrator, comm, kernel, rsqrt = key
    valid = [v for v in values if is_valid_run(v)]
    if not valid:
        continue
    kept, outliers, total_mad = keep_non_outliers(valid)
    totals = [v["total"] for v in kept]
    forces = [v["force"] for v in kept]
    comm_waits = [v["comm_wait"] for v in kept]
    gpairs = [v["gpairs"] for v in kept]
    nsteps = max(v["nsteps"] for v in values)
    force_evals = nsteps + (1 if integrator == "kdk" else 0)
    bytes_per_rank = (force_evals * max(0, ranks - 1) *
                      ((n + ranks - 1) // ranks) * 3 *
                      dtype_size(values[0].get("dtype", "double")))
    comm_wait_median = statistics.median(comm_waits)
    comm_bandwidth = (bytes_per_rank / comm_wait_median / 1.0e9
                      if comm_wait_median > 0.0 else 0.0)
    summary.append({
        "kind": kind,
        "N": n,
        "nsteps": nsteps,
        "ranks": ranks,
        "threads": threads,
        "resources": ranks * threads,
        "integrator": integrator,
        "comm": comm,
        "kernel": kernel,
        "rsqrt": rsqrt,
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
        "comm_bandwidth_GBps": comm_bandwidth,
        "gpairs_median": statistics.median(gpairs),
        "all_ok": all(v["status"] == "OK" for v in valid) and len(valid) == len(values),
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
    "kind", "N", "nsteps", "ranks", "threads", "resources", "integrator", "comm",
    "kernel", "rsqrt", "dtype",
    "runs", "failed_runs", "used_runs", "outliers", "total_mad", "total_median",
    "total_stdev", "force_median", "comm_wait_median",
    "comm_bandwidth_GBps", "gpairs_median",
    "speedup", "efficiency", "all_ok",
]
with open(dst, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

print(f"wrote {dst}")
