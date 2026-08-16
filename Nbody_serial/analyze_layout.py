#!/usr/bin/env python3
import csv
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    raise SystemExit("usage: analyze_layout.py layout_results.csv [summary.csv]")

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "layout_summary.csv"

rows = []
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        row["N"] = int(row["N"])
        row["threads"] = int(row["threads"])
        row["force"] = float(row["force"])
        row["gpairs"] = float(row["gpairs"])
        rows.append(row)

groups = defaultdict(list)
for row in rows:
    key = (row["layout"], row["N"], row["threads"], row["rsqrt"])
    groups[key].append(row)

summary = []
for key, values in sorted(groups.items()):
    layout, n, threads, rsqrt = key
    forces = [v["force"] for v in values]
    gpairs = [v["gpairs"] for v in values]
    summary.append({
        "layout": layout,
        "N": n,
        "threads": threads,
        "rsqrt": rsqrt,
        "runs": len(values),
        "force_median": statistics.median(forces),
        "force_stdev": statistics.stdev(forces) if len(forces) > 1 else 0.0,
        "gpairs_median": statistics.median(gpairs),
    })

by_case = {(r["N"], r["threads"], r["rsqrt"], r["layout"]): r for r in summary}
for row in summary:
    other = by_case.get((row["N"], row["threads"], row["rsqrt"], "aos"))
    if row["layout"] == "soa" and other is not None:
        row["soa_vs_aos_speedup"] = other["force_median"] / row["force_median"]
    else:
        row["soa_vs_aos_speedup"] = ""

fields = [
    "layout", "N", "threads", "rsqrt", "runs", "force_median",
    "force_stdev", "gpairs_median", "soa_vs_aos_speedup",
]
with open(dst, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

print(f"wrote {dst}")
