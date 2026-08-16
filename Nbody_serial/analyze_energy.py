#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    raise SystemExit("usage: analyze_energy.py energy_overhead.csv [summary.csv]")

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "energy_overhead_summary.csv"

rows = []
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        for key in ("N", "nsteps", "ranks", "threads", "energy_every"):
            row[key] = int(row[key])
        for key in ("total", "force", "energy", "max_rel_drift"):
            row[key] = float(row[key])
        rows.append(row)

groups = defaultdict(list)
for row in rows:
    key = (row["N"], row["nsteps"], row["ranks"], row["threads"], row["energy_every"])
    groups[key].append(row)

summary = []
for key, values in sorted(groups.items()):
    n, nsteps, ranks, threads, energy_every = key
    valid = [v for v in values if v["status"] in ("OK", "WARNING") and
             all(math.isfinite(v[k]) for k in ("total", "force", "energy"))]
    if not valid:
        continue
    totals = [v["total"] for v in valid]
    forces = [v["force"] for v in valid]
    energies = [v["energy"] for v in valid]
    summary.append({
        "N": n,
        "nsteps": nsteps,
        "ranks": ranks,
        "threads": threads,
        "energy_every": energy_every,
        "runs": len(values),
        "failed_runs": len(values) - len(valid),
        "total_median": statistics.median(totals),
        "force_median": statistics.median(forces),
        "energy_median": statistics.median(energies),
        "energy_fraction": statistics.median(energies) / statistics.median(totals),
        "max_rel_drift": max(v["max_rel_drift"] for v in valid),
        "all_ok": all(v["status"] == "OK" for v in valid) and len(valid) == len(values),
    })

base_by_case = {}
for row in summary:
    case = (row["N"], row["nsteps"], row["ranks"], row["threads"])
    old = base_by_case.get(case)
    if old is None or row["energy_every"] > old["energy_every"]:
        base_by_case[case] = row

for row in summary:
    base = base_by_case[(row["N"], row["nsteps"], row["ranks"], row["threads"])]
    row["overhead_vs_sparse_percent"] = (
        100.0 * (row["total_median"] - base["total_median"]) / base["total_median"]
        if base["total_median"] > 0.0 else 0.0
    )

fields = [
    "N", "nsteps", "ranks", "threads", "energy_every", "runs", "failed_runs",
    "total_median", "force_median", "energy_median", "energy_fraction",
    "overhead_vs_sparse_percent", "max_rel_drift", "all_ok",
]
with open(dst, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

print(f"wrote {dst}")
